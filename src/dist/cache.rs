#[cfg(feature = "dist-client")]
pub use self::client::ClientToolchains;
#[cfg(feature = "dist-server")]
pub use self::server::{ServerInputs, ServerToolchains};

#[cfg(feature = "dist-client")]
mod client {

    use anyhow::{Context, Error, Result, bail};
    use fs_err as fs;
    use futures::lock::Mutex;
    use std::collections::{HashMap, HashSet};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use crate::{
        config,
        dist::{
            PathTransformer, Toolchain,
            pkg::{ArchivePackager, ToolchainPackager},
        },
        lru_disk_cache::Error as LruError,
        lru_disk_cache::LruDiskCache,
        util::Digest,
    };

    async fn path_key<T: AsRef<Path>>(path: T) -> Result<String> {
        Digest::from_file(path).await.map(|digest| digest.finish())
    }

    #[derive(Clone, Debug)]
    pub struct CustomToolchain {
        archive: PathBuf,
        compiler_executable: String,
    }

    // TODO: possibly shouldn't be public
    pub struct ClientToolchains {
        cache_dir: PathBuf,
        cache: Mutex<LruDiskCache>,
        // Lookup from dist toolchain -> path to custom toolchain archive
        custom_toolchain_archives: Mutex<HashMap<Toolchain, PathBuf>>,
        // Lookup from local path -> toolchain details
        // The Option<Toolchain> could be populated on startup, but it's lazy for efficiency
        custom_toolchain_paths: Mutex<HashMap<PathBuf, (CustomToolchain, Option<Toolchain>)>>,
        // Toolchains configured to not be distributed
        disabled_toolchains: HashSet<PathBuf>,
        // Toolchains that have been hashed but not yet cached to disk
        pending_toolchains: Mutex<HashMap<String, Arc<dyn ArchivePackager>>>,
        // Local machine mapping from 'weak' hashes to strong toolchain hashes
        // - Weak hashes are what sccache uses to determine if a compiler has changed
        //   on the local machine - they're fast and 'good enough' (assuming we trust
        //   the local machine), but not safe if other users can update the cache.
        // - Strong hashes (or 'archive ids') are the hash of the complete compiler contents that
        //   will be sent over the wire for use in distributed compilation - it is assumed
        //   that if two of them match, the contents of a compiler archive cannot
        //   have been tampered with
        weak_map: Mutex<HashMap<String, String>>,
    }

    impl ClientToolchains {
        pub fn new(
            cache_dir: &Path,
            cache_size: u64,
            toolchain_configs: &[config::client::dist::Toolchain],
        ) -> Result<Self> {
            let cache_dir = cache_dir.to_owned();
            fs::create_dir_all(&cache_dir).context(format!(
                "failed to create top level toolchain cache dir: {}",
                cache_dir.display()
            ))?;

            let weak_map_path = cache_dir.join("weak_map.json");
            if !weak_map_path.exists() {
                fs::File::create(&weak_map_path)
                    .and_then(|mut f| f.write_all(b"{}"))
                    .context(format!(
                        "failed to create new toolchain weak map file: {}",
                        weak_map_path.display()
                    ))?;
            }
            let weak_map = fs::File::open(&weak_map_path)
                .map_err(Error::from)
                .and_then(|f| serde_json::from_reader(f).map_err(Error::from))
                .context(format!(
                    "failed to load toolchain weak map: {}",
                    weak_map_path.display()
                ))?;

            let tc_cache_dir = cache_dir.join("tc");
            let cache = LruDiskCache::new(&tc_cache_dir, cache_size)
                .map(Mutex::new)
                .context("failed to initialise a toolchain cache")?;

            // Load in toolchain configuration
            let mut custom_toolchain_paths = HashMap::new();
            let mut disabled_toolchains = HashSet::new();
            for ct in toolchain_configs.iter() {
                match ct {
                    config::client::dist::Toolchain::PathOverride {
                        compiler_executable,
                        archive,
                        archive_compiler_executable,
                    } => {
                        debug!(
                            "Registering custom toolchain for {}",
                            compiler_executable.display()
                        );
                        let custom_tc = CustomToolchain {
                            archive: archive.clone(),
                            compiler_executable: archive_compiler_executable.clone(),
                        };
                        if custom_toolchain_paths
                            .insert(compiler_executable.clone(), (custom_tc, None))
                            .is_some()
                        {
                            bail!("Multiple toolchains for {}", compiler_executable.display())
                        }
                        if disabled_toolchains.contains(compiler_executable) {
                            bail!(
                                "Override for toolchain {} conflicts with it being disabled",
                                compiler_executable.display()
                            )
                        }
                    }
                    config::client::dist::Toolchain::NoDist {
                        compiler_executable,
                    } => {
                        debug!("Disabling toolchain {}", compiler_executable.display());
                        if !disabled_toolchains.insert(compiler_executable.clone()) {
                            bail!(
                                "Disabled toolchain {} multiple times",
                                compiler_executable.display()
                            )
                        }
                        if custom_toolchain_paths.contains_key(compiler_executable) {
                            bail!(
                                "Override for toolchain {} conflicts with it being disabled",
                                compiler_executable.display()
                            )
                        }
                    }
                }
            }
            let custom_toolchain_paths = Mutex::new(custom_toolchain_paths);

            Ok(Self {
                cache_dir,
                cache,
                custom_toolchain_archives: Mutex::new(HashMap::new()),
                custom_toolchain_paths,
                disabled_toolchains,
                pending_toolchains: Default::default(),
                // TODO: shouldn't clear on restart, but also should have some
                // form of pruning
                weak_map: Mutex::new(weak_map),
            })
        }

        // Get the bytes of a toolchain tar
        // TODO: by this point the toolchain should be known to exist
        pub async fn get_toolchain(&self, tc: &Toolchain) -> Result<Option<fs::File>> {
            // TODO: be more relaxed about path casing and slashes on Windows
            let file = if let Some(custom_tc_archive) =
                self.custom_toolchain_archives.lock().await.get(tc)
            {
                fs::File::open(custom_tc_archive).with_context(|| {
                    format!(
                        "could not open file for toolchain {}",
                        custom_tc_archive.display()
                    )
                })?
            } else {
                match self.cache.lock().await.get_file(&tc.archive_id) {
                    Ok(file) => file,
                    Err(LruError::FileNotInCache) => return Ok(None),
                    Err(e) => return Err(e).context("error while retrieving toolchain from cache"),
                }
            };
            Ok(Some(file))
        }

        // Get the toolchain hash from the cache, or hash it and insert it into the cache
        pub async fn hash_toolchain(
            &self,
            compiler_path: &Path,
            weak_toolchain_key: &str,
            toolchain_packager: Box<dyn ToolchainPackager>,
            path_transformer: &mut PathTransformer,
        ) -> Result<(
            Toolchain,
            Option<(String, PathBuf)>,
            Option<Arc<dyn ArchivePackager>>,
        )> {
            if self.disabled_toolchains.contains(compiler_path) {
                bail!(
                    "Toolchain distribution for {:?} is disabled",
                    compiler_path.display()
                )
            }
            if let Some(tc_and_paths) = self.get_custom_toolchain(compiler_path).await {
                debug!("Using custom toolchain for {compiler_path:?}");
                let (tc, compiler_path, archive) = tc_and_paths?;
                return Ok((tc, Some((compiler_path, archive)), None));
            }

            // Only hash one toolchain at a time.
            // Not an issue if there are multiple attempts to
            // create the same toolchain, just a waste of time
            let mut cache = self.cache.lock().await;
            let mut weak_map = self.weak_map.lock().await;
            let mut pending_toolchains = self.pending_toolchains.lock().await;

            if let Some(archive_id) = weak_map.get(weak_toolchain_key) {
                let archive = cache.get_file(archive_id).ok();
                let package = pending_toolchains.get(archive_id);
                if archive.is_some() || package.is_some() {
                    if let Some(file) = archive {
                        trace!(
                            "Using cached toolchain {weak_toolchain_key:?} -> {archive_id:?} ({:?})",
                            file.path()
                        );
                    } else {
                        trace!("Using pending toolchain {weak_toolchain_key:?} -> {archive_id:?}");
                    }
                    return Ok((
                        Toolchain {
                            archive_id: archive_id.to_owned(),
                        },
                        None,
                        package.cloned(),
                    ));
                }
            }

            debug!("Weak key appears to be new: {weak_toolchain_key:?}");

            let package = toolchain_packager
                .package(path_transformer)
                .await
                .context("Could not package toolchain")?;

            let archive_id = package
                .compute_hash()
                .await
                .context("Could not hash toolchain")?;

            pending_toolchains.insert(archive_id.clone(), package.clone());
            weak_map.insert(weak_toolchain_key.to_owned(), archive_id.clone());

            fs::File::create(self.cache_dir.join("weak_map.json"))
                .map_err(Error::from)
                .and_then(|f| serde_json::to_writer(f, &*weak_map).map_err(Error::from))
                .context("failed to enter toolchain in weak map")?;

            Ok((Toolchain { archive_id }, None, Some(package)))
        }

        // If the toolchain doesn't already exist, compress it and insert into the local cache
        pub async fn put_toolchain(
            &self,
            toolchain: &Toolchain,
            package: &dyn ArchivePackager,
        ) -> Result<Option<fs::File>> {
            let archive_id = &toolchain.archive_id;
            // Only compress one toolchain at a time, because the default is to compress with all the cores.
            let mut cache = self.cache.lock().await;
            let weak_map = self.weak_map.lock().await;
            let mut pending_toolchains = self.pending_toolchains.lock().await;

            if let Ok(file) = cache.get_file(archive_id) {
                trace!("Using cached toolchain {archive_id:?} ({:?})", file.path());
                pending_toolchains.remove(archive_id);
                return Ok(Some(file));
            }

            if let Some((weak_toolchain_key, archive_id)) = weak_map
                .iter()
                .find(|&(_, strong_key)| strong_key == archive_id)
            {
                debug!("Compressing toolchain {weak_toolchain_key:?} -> {archive_id:?}");
            }

            let mut f = cache
                .prepare_add(archive_id, 0)
                .await
                .context("Failed to prepare toolchain cache dir")?;

            crate::util::block_in_place(&mut || package.write_tar_gz(f.as_file_mut()))
                .context("Could not compress toolchain")?;

            cache.commit(f).await.context("Failed to cache toolchain")?;

            pending_toolchains.remove(archive_id);

            match cache.get_file(archive_id) {
                Ok(file) => Ok(Some(file)),
                Err(LruError::FileNotInCache) => Ok(None),
                Err(e) => Err(e).context("error while retrieving toolchain from cache"),
            }
        }

        pub async fn get_custom_toolchain(
            &self,
            compiler_path: &Path,
        ) -> Option<Result<(Toolchain, String, PathBuf)>> {
            match self
                .custom_toolchain_paths
                .lock()
                .await
                .get_mut(compiler_path)
            {
                Some((custom_tc, Some(tc))) => Some(Ok((
                    tc.clone(),
                    custom_tc.compiler_executable.clone(),
                    custom_tc.archive.clone(),
                ))),
                Some((custom_tc, maybe_tc @ None)) => {
                    let archive_id = match path_key(&custom_tc.archive).await {
                        Ok(archive_id) => archive_id,
                        Err(e) => return Some(Err(e)),
                    };
                    let tc = Toolchain { archive_id };
                    *maybe_tc = Some(tc.clone());
                    // If this entry already exists, someone has two custom toolchains with the same strong hash
                    if let Some(old_path) = self
                        .custom_toolchain_archives
                        .lock()
                        .await
                        .insert(tc.clone(), custom_tc.archive.clone())
                    {
                        // Log a warning if the user has identical toolchains at two different locations - it's
                        // not strictly wrong, but it is a bit odd
                        if old_path != custom_tc.archive {
                            warn!(
                                "Detected interchangeable toolchain archives at {} and {}",
                                old_path.display(),
                                custom_tc.archive.display()
                            );
                        }
                    }
                    Some(Ok((
                        tc,
                        custom_tc.compiler_executable.clone(),
                        custom_tc.archive.clone(),
                    )))
                }
                None => None,
            }
        }
    }

    #[cfg(all(
        test,
        feature = "dist-client",
        any(
            all(
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ),
            target_os = "freebsd"
        )
    ))]
    mod test_dist {
        use crate::{config, dist::PathTransformer, errors::*, test::utils::create_file};
        use std::{io::Write, sync::Arc};

        use {
            crate::dist::pkg::{ArchivePackager, ToolchainPackager},
            async_trait::async_trait,
        };

        use super::ClientToolchains;

        struct PanicToolchainPackager;

        #[async_trait]
        impl ToolchainPackager for PanicToolchainPackager {
            async fn package(
                self: Box<Self>,
                _: &mut PathTransformer,
            ) -> Result<Arc<dyn ArchivePackager>> {
                panic!("should not have called packager")
            }
        }

        #[tokio::test]
        async fn test_client_toolchains_custom() {
            let td = crate::util::temp_dir().unwrap();

            let ct1 =
                create_file(td.path(), "ct1", |mut f| f.write_all(b"toolchain_contents")).unwrap();

            let client_toolchains = ClientToolchains::new(
                &td.path().join("cache"),
                1024,
                &[config::client::dist::Toolchain::PathOverride {
                    compiler_executable: "/my/compiler".into(),
                    archive: ct1.clone(),
                    archive_compiler_executable: "/my/compiler/in_archive".into(),
                }],
            )
            .unwrap();

            let (_, newpath, _) = client_toolchains
                .hash_toolchain(
                    "/my/compiler".as_ref(),
                    "weak_key",
                    Box::new(PanicToolchainPackager),
                    &mut PathTransformer::new(),
                )
                .await
                .unwrap();
            assert!(newpath.unwrap() == ("/my/compiler/in_archive".to_string(), ct1));
        }

        #[tokio::test]
        async fn test_client_toolchains_custom_multiuse_archive() {
            let td = crate::util::temp_dir().unwrap();

            let ct1 =
                create_file(td.path(), "ct1", |mut f| f.write_all(b"toolchain_contents")).unwrap();

            let client_toolchains = ClientToolchains::new(
                &td.path().join("cache"),
                1024,
                &[
                    config::client::dist::Toolchain::PathOverride {
                        compiler_executable: "/my/compiler".into(),
                        archive: ct1.clone(),
                        archive_compiler_executable: "/my/compiler/in_archive".into(),
                    },
                    // Uses the same archive, but a maps a different external compiler to a different archive compiler
                    config::client::dist::Toolchain::PathOverride {
                        compiler_executable: "/my/compiler2".into(),
                        archive: ct1.clone(),
                        archive_compiler_executable: "/my/compiler2/in_archive".into(),
                    },
                    // Uses the same archive, but a maps a different external compiler to the same archive compiler as the first
                    config::client::dist::Toolchain::PathOverride {
                        compiler_executable: "/my/compiler3".into(),
                        archive: ct1.clone(),
                        archive_compiler_executable: "/my/compiler/in_archive".into(),
                    },
                ],
            )
            .unwrap();

            let (_, newpath, _) = client_toolchains
                .hash_toolchain(
                    "/my/compiler".as_ref(),
                    "weak_key",
                    Box::new(PanicToolchainPackager),
                    &mut PathTransformer::new(),
                )
                .await
                .unwrap();
            assert!(newpath.unwrap() == ("/my/compiler/in_archive".to_string(), ct1.clone()));
            let (_, newpath, _) = client_toolchains
                .hash_toolchain(
                    "/my/compiler2".as_ref(),
                    "weak_key",
                    Box::new(PanicToolchainPackager),
                    &mut PathTransformer::new(),
                )
                .await
                .unwrap();
            assert!(newpath.unwrap() == ("/my/compiler2/in_archive".to_string(), ct1.clone()));
            let (_, newpath, _) = client_toolchains
                .hash_toolchain(
                    "/my/compiler3".as_ref(),
                    "weak_key",
                    Box::new(PanicToolchainPackager),
                    &mut PathTransformer::new(),
                )
                .await
                .unwrap();
            assert!(newpath.unwrap() == ("/my/compiler/in_archive".to_string(), ct1));
        }

        #[tokio::test]
        async fn test_client_toolchains_nodist() {
            let td = crate::util::temp_dir().unwrap();

            let client_toolchains = ClientToolchains::new(
                &td.path().join("cache"),
                1024,
                &[config::client::dist::Toolchain::NoDist {
                    compiler_executable: "/my/compiler".into(),
                }],
            )
            .unwrap();

            assert!(
                client_toolchains
                    .hash_toolchain(
                        "/my/compiler".as_ref(),
                        "weak_key",
                        Box::new(PanicToolchainPackager),
                        &mut PathTransformer::new(),
                    )
                    .await
                    .is_err()
            );
        }
    }

    #[cfg(test)]
    mod test_nodist {
        use crate::{config, test::utils::create_file};
        use std::io::Write;

        use super::ClientToolchains;

        #[test]
        fn test_client_toolchains_custom_nodist_conflict() {
            let td = crate::util::temp_dir().unwrap();

            let ct1 =
                create_file(td.path(), "ct1", |mut f| f.write_all(b"toolchain_contents")).unwrap();

            let client_toolchains = ClientToolchains::new(
                &td.path().join("cache"),
                1024,
                &[
                    config::client::dist::Toolchain::PathOverride {
                        compiler_executable: "/my/compiler".into(),
                        archive: ct1,
                        archive_compiler_executable: "/my/compiler".into(),
                    },
                    config::client::dist::Toolchain::NoDist {
                        compiler_executable: "/my/compiler".into(),
                    },
                ],
            );
            assert!(client_toolchains.is_err());
        }
    }
}

#[cfg(feature = "dist-server")]
mod server {
    use crate::{
        cache::{Cache, Storage, disk::DiskCache},
        dist::{
            InputsService, JobInputs, RunJobError, Toolchain, ToolchainService,
            io::{Jobs, Toolchains},
            metrics::ToolchainsMetrics,
        },
        errors::*,
        util::{AsyncMulticast, AsyncMulticastArgs, AsyncMulticastFunc, Digest},
    };
    use async_trait::async_trait;
    use flate2::bufread::{MultiGzDecoder as GzipDecoder, ZlibDecoder};
    use fs_err::{self as fs, os::unix::fs::OpenOptionsExt};
    use itertools::Itertools;
    use std::{
        collections::{BTreeMap, BTreeSet},
        io::{BufReader, Write},
        path::{Path, PathBuf},
        sync::Arc,
    };

    fn join_suffix<P: AsRef<Path>>(path: &Path, suffix: P) -> PathBuf {
        let suffixpath = suffix.as_ref();
        let mut components = suffixpath.components();
        if suffixpath.has_root() {
            assert_eq!(components.next(), Some(std::path::Component::RootDir));
        }
        path.join(components)
    }

    fn is_special_tokio_shutdown_io_error(err: &anyhow::Error) -> bool {
        if let Some(io_error) = err.downcast_ref::<std::io::Error>() {
            // tokio::fs returns an io::ErrorKind::Other error when a future
            // representing an IO operation is cancelled. This usually only
            // happens during server shutdown, so don't warn about this error
            // and bail early.
            if io_error.kind() == std::io::ErrorKind::Other {
                return true;
            }
        }
        false
    }

    fn gzip_inflate(deflated_path: PathBuf, inflated_path: PathBuf) -> std::io::Result<u64> {
        let deflated_file = fs::File::open(&deflated_path)?;
        unpack_tar(
            &mut GzipDecoder::new(BufReader::new(deflated_file)),
            &inflated_path,
        )
    }

    fn zlib_inflate(deflated_path: PathBuf, inflated_path: &Path) -> std::io::Result<u64> {
        let deflated_file = fs::File::open(&deflated_path)?;
        unpack_tar(
            &mut ZlibDecoder::new(BufReader::new(deflated_file)),
            inflated_path,
        )
    }

    // Takes `&mut dyn Read` so both decoders share one `tar::Archive` instantiation.
    fn unpack_tar(reader: &mut dyn std::io::Read, inflated_path: &Path) -> std::io::Result<u64> {
        // Unpack the tgz into the inflated dir
        // Manually implement `tar::Archive::unpack` in order to compute the inflated size
        let mut archive = tar::Archive::new(reader);

        let mut inflated_size = 0;
        let mut dirs = Vec::new();
        let dst = inflated_path
            .canonicalize()
            .unwrap_or_else(|_| inflated_path.to_owned());

        for entry in archive.entries()? {
            let mut entry = entry?;
            inflated_size += entry.size();
            if entry.header().entry_type() == tar::EntryType::Directory {
                dirs.push(entry);
            } else {
                entry.unpack_in(&dst)?;
            }
        }

        dirs.sort_by(|a, b| b.path_bytes().cmp(&a.path_bytes()));
        for mut dir in dirs {
            dir.unpack_in(&dst)?;
        }

        Ok(inflated_size)
    }

    #[derive(Clone)]
    struct InputFileHash(String);

    impl AsyncMulticastArgs for InputFileHash {
        type Key = String;
        fn hash(&self) -> Self::Key {
            self.0.clone()
        }
    }

    struct LoadInputFileFn {
        jobs_io: Jobs,
    }

    #[async_trait]
    impl AsyncMulticastFunc<InputFileHash, opendal::Buffer> for LoadInputFileFn {
        async fn call(&self, inputs: &InputFileHash) -> Result<opendal::Buffer> {
            self.jobs_io.get_job_inputs(&inputs.0).await
        }
    }

    #[derive(Clone)]
    pub struct ServerInputs {
        cache: Arc<DiskCache>,
        files: AsyncMulticast<InputFileHash, opendal::Buffer>,
    }

    #[async_trait]
    impl InputsService for ServerInputs {
        async fn get_inputs(&self, inputs: &JobInputs, outputs: &[PathBuf]) -> Result<PathBuf> {
            self.load(inputs.clone(), outputs.to_vec())
                .await
                .inspect_err(|err| {
                    tracing::error!("Error loading inputs: {err:?}");
                })
        }
    }

    impl ServerInputs {
        pub fn new(cache: Arc<DiskCache>, jobs_io: Jobs) -> Self {
            Self {
                cache,
                files: AsyncMulticast::new(LoadInputFileFn { jobs_io }),
            }
        }

        async fn load(&self, inputs: JobInputs, outputs: Vec<PathBuf>) -> Result<PathBuf> {
            let hash = match inputs {
                JobInputs::TarGZ { ref job_id } => job_id,
                JobInputs::Manifest { ref hash, .. } => hash,
            };

            if let Ok((inflated_path, _)) = self.cache.entry(hash).await {
                // Return early if the inputs are already loaded and unpacked
                Ok(inflated_path)
            } else {
                match inputs {
                    JobInputs::TarGZ { ref job_id } => self.load_tar_gz(job_id, outputs).await,
                    JobInputs::Manifest {
                        ref hash,
                        dirs,
                        files,
                        links,
                    } => self.load_manifest(hash, dirs, files, links).await,
                }
            }
        }

        async fn load_tar_gz(&self, job_id: &str, output_dirs: Vec<PathBuf>) -> Result<PathBuf> {
            let deflated_key = format!("{job_id}.tgz");
            if !self.cache.has(&deflated_key).await {
                let (_, entry) = self.files.call(InputFileHash(job_id.to_owned())).await?;
                self.cache.put(&deflated_key, entry).await?;
            }
            let (deflated_path, deflated_size) = self.cache.entry(&deflated_key).await?;
            self.cache
                .insert_with(job_id, deflated_size, |inflated_path: &Path| {
                    let inflated_path = inflated_path.to_owned();
                    async move {
                        crate::util::spawn_blocking(move || {
                            // Ensure the inflated dir exists first
                            fs::create_dir_all(&inflated_path)?;
                            let size = zlib_inflate(deflated_path, &inflated_path)?;
                            // Create output dirs
                            for output_dir in output_dirs.iter() {
                                // If it doesn't have a parent, nothing needs creating
                                if let Some(path) = output_dir.parent() {
                                    fs::create_dir_all(join_suffix(&inflated_path, path))?;
                                }
                            }
                            Ok(size)
                        })
                        .await?
                    }
                })
                .await
                .map(|(path, _)| path)
                .inspect_err(|err| {
                    tracing::error!("Error unpacking inputs: {err:?}");
                })
        }

        async fn load_manifest(
            &self,
            hash: &str,
            dirs: Vec<String>,
            files: BTreeMap<String, String>,
            links: BTreeMap<String, String>,
        ) -> Result<PathBuf> {
            if let Ok((inflated_path, _)) = self.cache.entry(hash).await {
                return Ok(inflated_path);
            }

            let (misses, inputs) =
                crate::util::join_all(files.values().unique().cloned().map(|hash| async move {
                    self.load_file(hash.clone()).await.map_err(|err| {
                        tracing::error!("Error loading input file: {err:?}");
                        hash
                    })
                }))
                .await
                .into_iter()
                .fold(
                    (BTreeSet::new(), BTreeMap::new()),
                    |(mut misses, mut inputs), result| {
                        match result {
                            Ok((hash, file)) => {
                                inputs.insert(hash, file);
                            }
                            Err(hash) => {
                                misses.insert(hash);
                            }
                        }
                        (misses, inputs)
                    },
                );

            if !misses.is_empty() {
                // tracing::error!("Misses: {misses:?}");
                return Err(RunJobError::MissingJobInputs {
                    inputs: misses.into_iter().collect(),
                }
                .into());
            }

            // tracing::trace!("Unpacking input files: {files:?}");

            let res = self
                .cache
                .insert_with(hash, 0, |inflated_path: &Path| {
                    let base = inflated_path.to_owned();
                    async move {
                        crate::util::spawn_blocking(move || {
                            // Ensure the inflated dir exists first
                            fs::create_dir_all(&base)?;

                            let mut inflated_size = 0;

                            for path in dirs {
                                let path = Path::new(&path);
                                inflated_size += path.components().count() as u64 * 4096;
                                fs::create_dir_all(join_suffix(&base, path))?;
                            }

                            for (name, target) in links {
                                let path = Path::new(&name);
                                if let Some(path) = path.parent() {
                                    inflated_size += path.components().count() as u64 * 4096;
                                    fs::create_dir_all(join_suffix(&base, path))?;
                                }
                                inflated_size += target.len() as u64;
                                std::os::unix::fs::symlink(target, join_suffix(&base, name))?;
                            }

                            let mut misses = vec![];

                            let files = files
                                .iter()
                                .filter_map(|(path, hash)| {
                                    inputs.get(hash).map(|data| (Path::new(path), hash, data))
                                });

                            for (path, hash, data) in files {
                                if let Some(path) = path.parent() {
                                    inflated_size += path.components().count() as u64 * 4096;
                                    fs::create_dir_all(join_suffix(&base, path))?;
                                }

                                let inflated_file = fs::OpenOptions::new()
                                    .create(true)
                                    .write(true)
                                    .mode(0o755)
                                    .truncate(true)
                                    .open(join_suffix(&base, path))?;

                                let mut inflated_file = Ok(inflated_file);

                                let inflated_hash = Digest::from_reader_with_each_sync(
                                    ZlibDecoder::new(data.clone()),
                                    |data| {
                                        if let Ok(file) = inflated_file.as_mut() {
                                            match file.write_all(data) {
                                                Ok(_) => {
                                                    inflated_size += data.len() as u64;
                                                }
                                                Err(err) => {
                                                    inflated_file = Err(err);
                                                }
                                            }
                                        }
                                    },
                                )
                                .map_err(std::io::Error::other)?
                                .finish();

                                if let Err(err) = inflated_file {
                                    tracing::error!(
                                        "Error inflating job input: path='{}', hash='{hash}', err={err:#}",
                                        path.display()
                                    );
                                    misses.push(hash.clone());
                                } else if inflated_hash.ne(hash) {
                                    tracing::error!(
                                        "Job input hash mismatch: path='{}' hash='{hash}'",
                                        path.display()
                                    );
                                    misses.push(hash.clone());
                                }
                            }

                            if misses.is_empty() {
                                Ok(inflated_size)
                            } else {
                                Err(std::io::Error::other(RunJobError::MissingJobInputs {
                                    inputs: misses,
                                }))
                            }
                        })
                        .await?
                    }
                })
                .await
                .map(|(inflated_path, _)| inflated_path)
                .inspect_err(|err| {
                    tracing::error!("Error unpacking files: {err:#}");
                });

            match res {
                Ok(res) => Ok(res),
                Err(err) => {
                    let io_err = match err.downcast::<std::io::Error>() {
                        Ok(io_err) => io_err,
                        Err(err) => return Err(err),
                    };
                    let job_err = match io_err.downcast::<RunJobError>() {
                        Ok(job_err) => job_err,
                        Err(io_err) => return Err(io_err.into()),
                    };
                    Err(job_err.into())
                }
            }
        }

        async fn load_file(&self, hash: String) -> Result<(String, opendal::Buffer)> {
            if !self.cache.has(&hash).await {
                let (InputFileHash(hash), data) = self.files.call(InputFileHash(hash)).await?;
                self.cache.put(&hash, data.clone()).await?;
                Ok((hash, data))
            } else {
                let data = match self.cache.get(&hash).await? {
                    Cache::Hit(data) => data,
                    Cache::Miss => {
                        bail!("Missing input: {hash}")
                    }
                };
                Ok((hash, data))
            }
        }
    }

    #[derive(Clone)]
    pub struct ServerToolchains {
        cache: Arc<DiskCache>,
        toolchains: Toolchains,
        metrics: ToolchainsMetrics,
        should_inflate_toolchains: bool,
    }

    #[async_trait]
    impl ToolchainService for ServerToolchains {
        async fn get_toolchain(&self, toolchain: &Toolchain) -> Result<PathBuf> {
            self.load(toolchain).await
        }
    }

    impl ServerToolchains {
        pub fn new(
            cache: Arc<DiskCache>,
            toolchains: Toolchains,
            metrics: ToolchainsMetrics,
            should_inflate_toolchains: bool,
        ) -> Self {
            Self {
                cache,
                toolchains,
                metrics,
                should_inflate_toolchains,
            }
        }

        async fn load(&self, tc: &Toolchain) -> Result<PathBuf> {
            // Record toolchain load time after retrying
            let _timer = self.metrics.load_toolchain_timer();
            // Load and cache the deflated toolchain.
            // Inflate, unpack, and cache it in a directory.
            // Return the path to the unpacked toolchain dir.
            self.load_toolchain(tc).await.map_err(|err| {
                if !is_special_tokio_shutdown_io_error(&err) {
                    tracing::error!("[ServerToolchains({tc})]: Error loading toolchain: {err:?}");
                }
                err
            })
        }

        async fn load_toolchain(&self, tc: &Toolchain) -> Result<PathBuf> {
            // Record toolchain load_inflated time
            let _timer = self.metrics.load_inflated_toolchain_timer();
            if let Ok((inflated_path, _)) = self.cache.entry(&tc.archive_id).await {
                // Return early if the toolchain is already loaded and unpacked
                Ok(inflated_path)
            } else {
                // Load the compressed toolchain
                let (deflated_path, deflated_size) = self.load_deflated_toolchain(tc).await?;
                if !self.should_inflate_toolchains {
                    Ok(deflated_path)
                } else {
                    // Inflate and unpack the toolchain archive
                    let inflated_path = self
                        .unpack_inflated_toolchain(&deflated_path, deflated_size, &tc.archive_id)
                        .await?;
                    Ok(inflated_path)
                }
            }
        }

        async fn load_deflated_toolchain(&self, toolchain: &Toolchain) -> Result<(PathBuf, u64)> {
            // Record toolchain load_deflated time
            let _timer = self.metrics.load_deflated_toolchain_timer();
            let deflated_key = format!("{toolchain}.tgz");
            if !self.cache.has(&deflated_key).await {
                let entry = self.toolchains.get_toolchain(toolchain).await?;
                self.cache.put(&deflated_key, entry).await?;
            }
            self.cache.entry(&deflated_key).await
        }

        async fn unpack_inflated_toolchain(
            &self,
            deflated_path: &Path,
            deflated_size: u64,
            inflated_key: &str,
        ) -> Result<PathBuf> {
            // Record toolchain unpack_inflated time
            let _timer = self.metrics.unpack_inflated_toolchain_timer();
            self.cache
                .insert_with(inflated_key, deflated_size, |inflated_path: &Path| {
                    let deflated_path = deflated_path.to_owned();
                    let inflated_path = inflated_path.to_owned();
                    async move {
                        crate::util::spawn_blocking(move || {
                            // Ensure the inflated dir exists first
                            fs::create_dir_all(&inflated_path)?;
                            gzip_inflate(deflated_path, inflated_path)
                        })
                        .await?
                    }
                })
                .await
                .map(|(path, _)| path)
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;
        use crate::test::{mock_storage::MockStorage, utils::Waiter};
        use std::{io::Write, path::Path, sync::Arc};

        fn tar_bytes() -> Vec<u8> {
            let mut builder = tar::Builder::new(Vec::new());
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::Directory);
            header.set_size(0);
            header.set_mode(0o755);
            header.set_cksum();
            builder
                .append_data(&mut header, "dir/", std::io::empty())
                .unwrap();
            let mut header = tar::Header::new_gnu();
            header.set_size(5);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, "dir/file.txt", &b"hello"[..])
                .unwrap();
            builder.into_inner().unwrap()
        }

        fn assert_unpacked(dir: &Path) {
            assert!(dir.join("dir").is_dir());
            assert_eq!(fs_err::read(dir.join("dir/file.txt")).unwrap(), b"hello");
        }

        #[test]
        fn gzip_and_zlib_inflate_unpack_entries_and_return_inflated_size() {
            let tmp = tempfile::tempdir().unwrap();
            let gzip_path = tmp.path().join("toolchain.tar.gz");
            let file = fs_err::File::create(&gzip_path).unwrap();
            let mut gzip = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            gzip.write_all(&tar_bytes()).unwrap();
            gzip.finish().unwrap();
            let zlib_path = tmp.path().join("inputs.zlib");
            let file = fs_err::File::create(&zlib_path).unwrap();
            let mut zlib = flate2::write::ZlibEncoder::new(file, flate2::Compression::default());
            zlib.write_all(&tar_bytes()).unwrap();
            zlib.finish().unwrap();

            let gzip_dst = tmp.path().join("gzip");
            fs_err::create_dir(&gzip_dst).unwrap();
            assert_eq!(super::gzip_inflate(gzip_path, gzip_dst.clone()).unwrap(), 5);
            assert_unpacked(&gzip_dst);

            let zlib_dst = tmp.path().join("zlib");
            fs_err::create_dir(&zlib_dst).unwrap();
            assert_eq!(super::zlib_inflate(zlib_path, &zlib_dst).unwrap(), 5);
            assert_unpacked(&zlib_dst);
        }

        #[test]
        fn load_manifest_hash_mismatch() -> Result<()> {
            let tempdir = tempfile::Builder::new()
                .prefix("load_manifest_hash_mismatch")
                .tempdir()
                .expect("Failed to create tempdir");
            let cache_dir = tempdir.path().join("cache");
            std::fs::create_dir(&cache_dir).unwrap();

            let jobs_cache = Arc::new(DiskCache::new(
                &cache_dir,
                1024 * 1024,
                crate::config::CacheMode::ReadWrite,
                vec![],
            ));
            let jobs_storage = Arc::new(MockStorage::new(None, true));

            let server_inputs = ServerInputs::new(
                jobs_cache.clone(),
                Jobs::new(jobs_storage.clone(), Default::default()),
            );

            let bad_key = {
                let mut m = Digest::new();
                m.update(b"bad key");
                m.finish()
            };

            let src_file = b"source file";

            let real_key = {
                let mut m = Digest::new();
                m.update(src_file);
                m.finish()
            };

            jobs_cache.put(&bad_key, src_file.to_vec().into()).wait()?;

            let res = server_inputs
                .load_manifest(
                    "",
                    vec![],
                    [("source_file.c".into(), real_key.clone())]
                        .into_iter()
                        .collect(),
                    Default::default(),
                )
                .wait();

            match res {
                Ok(res) => {
                    panic!("res: {res:?}");
                }
                Err(err) => match err.downcast::<RunJobError>() {
                    Ok(err) => match err {
                        RunJobError::MissingJobInputs { inputs } => {
                            assert_eq!(inputs, vec![real_key.clone()]);
                            Ok(())
                        }
                        err => {
                            panic!("err: {err:?}");
                        }
                    },
                    Err(err) => {
                        panic!("err: {err:?}");
                    }
                },
            }
        }
    }
}
