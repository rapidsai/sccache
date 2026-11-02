// Copyright 2016 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{errors::*, mock_command::ProcessOutput};
use async_trait::async_trait;
use bytes::Buf;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{
    collections::{BTreeMap, HashMap},
    ffi::OsString,
    fmt,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    time::Duration,
};

#[cfg(any(feature = "dist-client", feature = "dist-server"))]
mod cache;
#[cfg(feature = "dist-client")]
pub mod client_auth;
#[cfg(any(feature = "dist-client", feature = "dist-server"))]
pub mod http;
#[cfg(feature = "dist-server")]
pub mod io;
#[cfg(feature = "dist-server")]
pub mod metrics;
#[cfg(feature = "dist-server")]
pub mod scheduler;
#[cfg(feature = "dist-server")]
pub mod server;
#[cfg(feature = "dist-server")]
pub mod tasks;
#[cfg(feature = "dist-server")]
pub mod token_check;

#[cfg(feature = "dist-server")]
pub use crate::{
    cache::BufReadSeek,
    dist::cache::{ServerInputs, ServerToolchains},
};

#[cfg(feature = "dist-server")]
pub(crate) fn job_inputs_key(job_id: &str) -> String {
    format!("{job_id}-inputs")
}

#[cfg(feature = "dist-server")]
pub fn job_status_key(job_id: &str) -> String {
    format!("{job_id}-status")
}

#[cfg(feature = "dist-server")]
pub fn job_result_key(job_id: &str) -> String {
    format!("{job_id}-result")
}

#[cfg(feature = "dist-server")]
pub fn scheduler_to_servers_queue() -> String {
    queue_name_with_env_info("scheduler-to-servers")
}

#[cfg(feature = "dist-server")]
pub fn server_to_schedulers_queue() -> String {
    queue_name_with_env_info("server-to-schedulers")
}

#[cfg(feature = "dist-server")]
pub fn to_scheduler_queue(id: &str) -> String {
    queue_name_with_env_info(&format!("scheduler-{id}"))
}

#[cfg(feature = "dist-server")]
pub fn to_server_queue(id: &str) -> String {
    queue_name_with_env_info(&format!("server-{id}"))
}

#[cfg(feature = "dist-server")]
pub fn queue_name_with_env_info(prefix: &str) -> String {
    format!("{prefix}-{}", env_info())
}

#[cfg(feature = "dist-server")]
pub fn env_info() -> String {
    use itertools::Itertools;
    // "{os}-{arch}-{deployment}"
    // "linux-amd64-redis_zone_a" etc.
    [
        std::env::var("SCCACHE_DIST_OS").unwrap_or(std::env::consts::OS.to_owned()),
        std::env::var("SCCACHE_DIST_ARCH").unwrap_or(std::env::consts::ARCH.to_owned()),
        std::env::var("SCCACHE_DIST_DEPLOYMENT_NAME").unwrap_or_default(),
    ]
    .iter()
    .filter(|s| !s.is_empty())
    .join("-")
}

#[cfg(feature = "dist-server")]
pub fn init_logging() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    if std::env::var(crate::LOGGING_ENV).is_ok() {
        match tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_env(crate::LOGGING_ENV).unwrap_or_else(
                    |_| {
                        // axum logs rejections from built-in extractors with the `axum::rejection`
                        // target, at `TRACE` level. `axum::rejection=trace` enables showing those events
                        format!(
                            "{}=debug,tower_http=debug,axum::rejection=trace",
                            env!("CARGO_CRATE_NAME")
                        )
                        .into()
                    },
                ),
            )
            .with(tracing_subscriber::fmt::layer())
            .try_init()
        {
            Ok(_) => (),
            Err(e) => panic!("Failed to initialize logging: {e:?}"),
        }
    }
}

// TODO: paths (particularly outputs, which are accessed by an unsandboxed program)
// should be some pre-sanitised AbsPath type

pub use self::path_transform::PathTransformer;

#[cfg(feature = "dist-client")]
pub mod pkg;
#[cfg(not(feature = "dist-client"))]
mod pkg {
    pub trait ToolchainPackager {}
    pub trait ArchivePackager {}
    #[allow(dead_code)]
    pub trait InputsPackager {}
}

mod path_transform {
    use std::collections::HashMap;
    use std::ffi::{OsStr, OsString};
    use std::path::{Component, Components, Path, PathBuf, Prefix, PrefixComponent};
    use std::str;

    fn take_prefix<'a>(components: &'a mut Components<'_>) -> Option<PrefixComponent<'a>> {
        let prefix = components.next()?;
        let pc = match prefix {
            Component::Prefix(pc) => pc,
            _ => return None,
        };
        let root = components.next()?;
        if root != Component::RootDir {
            return None;
        }
        Some(pc)
    }

    fn transform_prefix_component(pc: PrefixComponent<'_>) -> Option<OsString> {
        match pc.kind() {
            // Transforming these to the same place means these may flip-flop
            // in the tracking map, but they're equivalent so not really an
            // issue
            Prefix::Disk(diskchar) | Prefix::VerbatimDisk(diskchar) => {
                assert!(diskchar.is_ascii_alphabetic());
                let diskchar = diskchar.to_ascii_lowercase();
                let mut path = OsString::new();
                path.push(OsStr::new("/prefix/drive_"));
                path.push(OsStr::new(
                    str::from_utf8(&[diskchar]).expect("invalid disk char"),
                ));
                Some(path)
            }
            Prefix::Verbatim(_)
            | Prefix::VerbatimUNC(_, _)
            | Prefix::DeviceNS(_)
            | Prefix::UNC(_, _) => None,
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct PathTransformer {
        dist_to_local_path: HashMap<OsString, PathBuf>,
    }

    impl PathTransformer {
        pub fn new() -> Self {
            PathTransformer {
                dist_to_local_path: HashMap::new(),
            }
        }
        ///
        /// Remove the extension so preprocessed file doesn't have the same name as
        /// the source file. This works around a GCC bug when compiling preprocessed
        /// input -- namely, if the preprocessed file has the same name as the source
        /// file, error source lines aren't mapped correctly:
        ///
        /// ```shell
        /// # Preprocess some file with an error:
        /// $ gcc -x c++ -E tests/test.c > tests/test.ii
        ///
        /// # Compile the preprocessed file (error source is correct):
        /// $ gcc -x c++-cpp-output -c tests/test.ii -o tests/test.c.o
        /// > tests/test.c: In function ‘void foo()’:
        /// > tests/test.c:5:19: error: invalid conversion from ‘const char*’ to ‘int’ [-fpermissive]
        /// >     5 |   const int foo = "5";
        /// >       |                   ^~~
        /// >       |                   |
        /// >       |                   const char*
        ///
        ///
        /// # Rename `test.ii` -> `test.c`
        /// $ mv tests/test.{ii,c}
        ///
        /// # Compile the renamed preprocessed file (error source is wrong):
        /// $ gcc -x c++-cpp-output -c tests/test.c -o tests/test.c.o
        /// > tests/test.c: In function ‘void foo()’:
        /// > tests/test.c:5:19: error: invalid conversion from ‘const char*’ to ‘int’ [-fpermissive]
        /// >     5 | # 0 "<command-line>" 2
        /// >       |                   ^~~
        /// >       |                   |
        /// >       |                   const char*
        /// ```
        ///
        pub fn with_dist_extension(&mut self, input_path: &Path) -> PathBuf {
            if let Some(ext) = input_path.extension() {
                input_path.with_extension([OsStr::new("dist"), ext].join(OsStr::new(".")))
            } else {
                input_path.with_extension("dist")
            }
        }
        pub fn as_dist_abs<P: AsRef<Path>>(&mut self, p: P) -> Option<PathBuf> {
            let p = p.as_ref();
            if !p.is_absolute() {
                return None;
            }
            self.as_dist(p)
        }
        pub fn as_dist<P: AsRef<Path>>(&mut self, p: P) -> Option<PathBuf> {
            let p = p.as_ref();

            let mut components = p.components();

            // Extract the prefix (e.g. "C:/") if present
            let maybe_dist_prefix = if p.is_absolute() {
                if let Some(pc) = take_prefix(&mut components) {
                    Some(transform_prefix_component(pc)?)
                } else {
                    Some(OsString::new())
                }
            } else {
                None
            };

            // Reconstruct the path as a Linux path
            let mut component = components.next();
            let mut dist_path = maybe_dist_prefix
                .map(|mut dist_prefix| {
                    if component.is_some() || dist_prefix.is_empty() {
                        dist_prefix.push(OsStr::new("/"));
                    }
                    dist_prefix
                })
                .unwrap_or_default();
            loop {
                if let Some(part) = component {
                    dist_path.push(part);
                    component = components.next();
                }
                if component.is_some() {
                    dist_path.push("/");
                } else {
                    break;
                }
            }

            self.dist_to_local_path
                .insert(dist_path.clone(), p.to_owned());

            Some(dist_path.into())
        }
        pub fn disk_mappings(&self) -> impl Iterator<Item = (PathBuf, PathBuf)> {
            let mut normal_mappings = HashMap::new();
            let mut verbatim_mappings = HashMap::new();

            for local_path in self.dist_to_local_path.values() {
                if !local_path.is_absolute() {
                    continue;
                }
                let mut components = local_path.components();
                let (local_prefix_component, mappings) =
                    if let Some(local_prefix) = take_prefix(&mut components) {
                        (
                            Component::Prefix(local_prefix),
                            if let Prefix::VerbatimDisk(_) = local_prefix.kind() {
                                &mut verbatim_mappings
                            } else {
                                &mut normal_mappings
                            },
                        )
                    } else {
                        (Component::RootDir, &mut normal_mappings)
                    };

                let local_prefix_path: &Path = local_prefix_component.as_ref();

                if mappings.contains_key(local_prefix_path) {
                    continue;
                }

                if let Component::Prefix(local_prefix) = local_prefix_component {
                    let dist_prefix = transform_prefix_component(local_prefix)
                        .expect("prefix already in tracking map could not be transformed");

                    mappings.insert(local_prefix_path.to_owned(), dist_prefix);
                } else {
                    mappings.insert(
                        local_prefix_path.to_owned(),
                        local_prefix_path.as_os_str().to_owned(),
                    );
                }
            }

            // Prioritise normal mappings for the same disk, as verbatim mappings can
            // look odd to users
            normal_mappings
                .into_iter()
                .chain(verbatim_mappings)
                .map(|(a, b)| (a, b.into()))
        }
        pub fn to_local<P: AsRef<Path>>(&self, p: P) -> Option<PathBuf> {
            self.dist_to_local_path.get(p.as_ref().as_os_str()).cloned()
        }
    }

    #[test]
    fn test_basic() {
        #[cfg(not(target_os = "windows"))]
        let (a, b, c, d) = {
            (
                (Path::new("/a"), Path::new("/a")),
                (Path::new("/a/b.c"), Path::new("/a/b.c")),
                (Path::new("/other.c"), Path::new("/other.c")),
                vec![(Path::new("/"), Path::new("/"))],
            )
        };

        #[cfg(target_os = "windows")]
        let (a, b, c, d) = {
            (
                (Path::new("C:/a"), Path::new("/prefix/drive_c/a")),
                (Path::new(r#"C:\a\b.c"#), Path::new("/prefix/drive_c/a/b.c")),
                (
                    Path::new("X:/other.c"),
                    Path::new("/prefix/drive_x/other.c"),
                ),
                vec![
                    (Path::new("C:"), Path::new("/prefix/drive_c")),
                    (Path::new("X:"), Path::new("/prefix/drive_x")),
                ],
            )
        };

        let mut pt = PathTransformer::new();

        // a
        assert_eq!(pt.as_dist(a.0).unwrap(), a.1);
        assert_eq!(pt.to_local(a.1).unwrap(), a.0);

        // b
        assert_eq!(pt.as_dist(b.0).unwrap(), b.1);
        assert_eq!(pt.to_local(b.1).unwrap(), b.0);

        // c
        assert_eq!(pt.as_dist(c.0).unwrap(), c.1);
        assert_eq!(pt.to_local(c.1).unwrap(), c.0);

        let mut disk_mappings: Vec<_> = pt.disk_mappings().collect();
        disk_mappings.sort();
        let disk_mappings = disk_mappings
            .iter()
            .map(|(a, b)| (a.as_path(), b.as_path()))
            .collect::<Vec<_>>();

        // d
        assert_eq!(disk_mappings, d);
    }

    #[test]
    fn test_relative_paths() {
        #[cfg(not(target_os = "windows"))]
        let (a, b) = {
            (
                (Path::new("a/b"), Path::new("a/b")),
                (Path::new("a/b"), Path::new("a/b")),
            )
        };

        #[cfg(target_os = "windows")]
        let (a, b) = {
            (
                (Path::new("a/b"), Path::new("a/b")),
                (Path::new(r#"a\b"#), Path::new("a/b")),
            )
        };

        let mut pt = PathTransformer::new();

        // a
        assert_eq!(pt.as_dist(a.0).unwrap(), a.1);
        assert_eq!(pt.to_local(a.1).unwrap(), a.0);
        // b
        assert_eq!(pt.as_dist(b.0).unwrap(), b.1);
        assert_eq!(pt.to_local(b.1).unwrap(), b.0);
    }

    #[test]
    fn test_verbatim_disks() {
        #[cfg(not(target_os = "windows"))]
        let (a, b, c) = {
            (
                (Path::new("/other.c"), Path::new("/other.c")),
                (Path::new("/out/other.o"), Path::new("/out/other.o")),
                vec![(Path::new("/"), Path::new("/"))],
            )
        };

        #[cfg(target_os = "windows")]
        let (a, b, c) = {
            (
                (
                    Path::new("X:/other.c"),
                    Path::new("/prefix/drive_x/other.c"),
                ),
                (
                    Path::new(r#"\\?\X:\out\other.o"#),
                    Path::new("/prefix/drive_x/out/other.o"),
                ),
                vec![
                    (Path::new("X:"), Path::new("/prefix/drive_x")),
                    (Path::new(r#"\\?\X:"#), Path::new("/prefix/drive_x")),
                ],
            )
        };

        let mut pt = PathTransformer::new();

        // a
        assert_eq!(pt.as_dist(a.0).unwrap(), a.1);
        assert_eq!(pt.to_local(a.1).unwrap(), a.0);

        // b
        assert_eq!(pt.as_dist(b.0).unwrap(), b.1);
        assert_eq!(pt.to_local(b.1).unwrap(), b.0);

        // Verbatim disks should come last
        let disk_mappings: Vec<_> = pt.disk_mappings().collect();
        let disk_mappings = disk_mappings
            .iter()
            .map(|(a, b)| (a.as_path(), b.as_path()))
            .collect::<Vec<_>>();

        // c
        assert_eq!(disk_mappings, c);
    }

    #[test]
    fn test_slash_directions() {
        #[cfg(not(target_os = "windows"))]
        let (a, b) = {
            (
                (Path::new("/a"), Path::new("/a")),
                (Path::new("/a"), Path::new("/a")),
            )
        };

        #[cfg(target_os = "windows")]
        let (a, b) = {
            (
                (Path::new("C:/a"), Path::new("/prefix/drive_c/a")),
                (Path::new("C:\\a"), Path::new("/prefix/drive_c/a")),
            )
        };

        let mut pt = PathTransformer::new();

        // a
        assert_eq!(pt.as_dist(a.0).unwrap(), a.1);
        assert_eq!(pt.to_local(a.1).unwrap(), a.0);

        // b
        assert_eq!(pt.as_dist(b.0).unwrap(), b.1);
        assert_eq!(pt.to_local(b.1).unwrap(), a.0);

        assert_eq!(pt.disk_mappings().count(), 1);
    }
}

pub fn osstrings_to_strings(osstrings: &[OsString]) -> Option<Vec<String>> {
    osstrings
        .iter()
        .map(|arg| arg.clone().into_string().ok())
        .collect::<Option<_>>()
}

pub fn osstring_tuples_to_strings(
    osstring_tuples: &[(OsString, OsString)],
) -> Option<Vec<(String, String)>> {
    osstring_tuples
        .iter()
        .map(|(k, v)| Some((k.clone().into_string().ok()?, v.clone().into_string().ok()?)))
        .collect::<Option<_>>()
}

pub fn strings_to_osstrings(strings: &[String]) -> Vec<OsString> {
    strings
        .iter()
        .map(|arg| std::ffi::OsStr::new(arg).to_os_string())
        .collect::<Vec<_>>()
}

#[derive(Clone)]
pub enum JobInput {
    Data(opendal::Buffer),
    File(PathBuf),
}

impl JobInput {
    pub fn into_async_read(self) -> Result<Pin<Box<dyn futures::io::AsyncBufRead + Send>>> {
        Ok(match self {
            Self::Data(data) => Box::pin(futures::io::BufReader::new(
                futures::io::AllowStdIo::new(data.reader()),
            )),
            Self::File(ref path) => {
                let file = fs_err::OpenOptions::new()
                    .read(true)
                    .open(path)
                    .with_context(|| format!("Failed to open file for uploading: {path:?}"))?;
                Box::pin(futures::io::BufReader::new(futures::io::AllowStdIo::new(
                    file,
                )))
            }
        })
    }
}

#[serde_as]
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputData(#[serde_as(as = "serde_with::Bytes")] Vec<u8>, u64);
impl OutputData {
    #[cfg(any(feature = "dist-server", all(feature = "dist-client", test)))]
    pub fn try_from_reader<R: std::io::Read>(reader: R) -> std::io::Result<Self> {
        let mut compressor = flate2::read::ZlibEncoder::new(
            reader,
            // Optimize for size since bandwidth costs more than CPU cycles
            flate2::Compression::best(),
        );
        let mut res = vec![];
        std::io::copy(&mut compressor, &mut res)?;
        Ok(OutputData(res, compressor.total_in()))
    }
    pub fn lens(&self) -> OutputDataLens {
        OutputDataLens {
            actual: self.1,
            compressed: self.0.len() as u64,
        }
    }
    pub fn compressed(&self) -> Vec<u8> {
        self.0.clone()
    }
    #[cfg(feature = "dist-client")]
    pub fn into_reader(self) -> impl std::io::Read {
        flate2::read::ZlibDecoder::new(std::io::Cursor::new(self.0))
    }
}

impl fmt::Debug for OutputData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Size: {}->{}", self.1, self.0.len())
    }
}

pub struct OutputDataLens {
    pub actual: u64,
    pub compressed: u64,
}

impl fmt::Display for OutputDataLens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Size: {}->{}", self.actual, self.compressed)
    }
}

// BuildResult

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildResult {
    pub output: ProcessOutput,
    pub outputs: Vec<(String, OutputData)>,
}

pub struct BuildOutput {
    pub output: ProcessOutput,
    pub hashed: Vec<(String, String, OutputData)>,
}

impl From<BuildOutput> for BuildResult {
    fn from(build: BuildOutput) -> Self {
        let mut outputs = vec![];
        for (path, _, data) in build.hashed {
            outputs.push((path, data));
        }
        Self {
            output: build.output,
            outputs,
        }
    }
}

#[cfg(feature = "dist-server")]
pub enum BuildError {
    PrepareOverlay(Error),
    EnterNewNS(nix::errno::Errno),
    MountNsRoot(nix::errno::Errno),
    MakeOverlayDir(PathBuf, std::io::Error),
    MountOverlayFS(Error),
    UnpackInputs(Error),
    CopyOutputDir(PathBuf, Error),
    MakeOutputDir(PathBuf, std::io::Error),
    SpawnChildProcess(std::io::Error),
    ReadChildOutput(std::io::Error),
    WaitChildOutput(std::io::Error),
    KillChildProcess(std::io::Error),
    ReadBuildResult(PathBuf, std::io::Error),
    Cancelled,
    Unknown(Error),
}

#[cfg(feature = "dist-server")]
impl fmt::Debug for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrepareOverlay(e) => write!(f, "Failed to prepare overlay dirs: {e:?}"),
            Self::EnterNewNS(e) => write!(f, "Failed to enter a new Linux namespace: {e:?}"),
            Self::MountNsRoot(e) => write!(f, "Failed to mount Linux namespace root: {e:?}"),
            Self::MakeOverlayDir(p, e) => write!(f, "Failed to create overlay dir {p:?}: {e:?}"),
            Self::MountOverlayFS(e) => write!(f, "Failed to mount overlay FS: {e:?}"),
            Self::UnpackInputs(e) => write!(f, "Failed to unpack inputs to overlay: {e:?}"),
            Self::CopyOutputDir(p, e) => write!(f, "Failed to copy output dir {p:?}: {e:?}"),
            Self::MakeOutputDir(p, e) => write!(f, "Failed to create output dir {p:?}: {e:?}"),
            Self::SpawnChildProcess(e) => write!(f, "Failed to spawn build process: {e:?}"),
            Self::ReadChildOutput(e) => write!(f, "Failed to read process output: {e:?}"),
            Self::WaitChildOutput(e) => write!(f, "Failed to wait for build process: {e:?}"),
            Self::KillChildProcess(e) => write!(f, "Failed to kill build process: {e:?}"),
            Self::ReadBuildResult(p, e) => write!(f, "Failed to read build result {p:?}: {e:?}"),
            Self::Cancelled => write!(f, "Build cancelled"),
            Self::Unknown(e) => write!(f, "Unknown build error: {e:?}"),
        }
    }
}

#[cfg(feature = "dist-server")]
impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrepareOverlay(e) => write!(f, "Failed to prepare overlay dirs: {e}"),
            Self::EnterNewNS(e) => write!(f, "Failed to enter a new Linux namespace: {e}"),
            Self::MountNsRoot(e) => write!(f, "Failed to mount Linux namespace root: {e}"),
            Self::MakeOverlayDir(p, e) => write!(f, "Failed to create overlay dir {p:?}: {e}"),
            Self::MountOverlayFS(e) => write!(f, "Failed to mount overlay FS: {e}"),
            Self::UnpackInputs(e) => write!(f, "Failed to unpack inputs to overlay: {e}"),
            Self::CopyOutputDir(p, e) => write!(f, "Failed to copy output dir {p:?}: {e}"),
            Self::MakeOutputDir(p, e) => write!(f, "Failed to create output dir {p:?}: {e}"),
            Self::SpawnChildProcess(e) => write!(f, "Failed to spawn build process: {e}"),
            Self::ReadChildOutput(e) => write!(f, "Failed to read process output: {e}"),
            Self::WaitChildOutput(e) => write!(f, "Failed to wait for build process: {e}"),
            Self::KillChildProcess(e) => write!(f, "Failed to kill build process: {e}"),
            Self::ReadBuildResult(p, e) => write!(f, "Failed to read build result {p:?}: {e}"),
            Self::Cancelled => write!(f, "Build cancelled"),
            Self::Unknown(e) => write!(f, "Unknown build error: {e:?}"),
        }
    }
}

#[cfg(feature = "dist-server")]
impl std::error::Error for BuildError {}

// CompileCommand

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompileCommand {
    pub executable: String,
    pub arguments: Vec<String>,
    pub env_vars: Vec<(String, String)>,
    pub cwd: String,
}

impl CompileCommand {
    pub fn as_dist(self, path_transformer: &mut PathTransformer) -> Result<Self> {
        let cwd = path_transformer
            .as_dist_abs(&self.cwd)
            .map(|p| crate::util::path_to_str(p).map_err(Into::into))
            .unwrap_or_else(|| bail!("Failed to translate cwd {:?}", self.cwd))?
            .into_owned();
        Ok(Self { cwd, ..self })
    }
}

impl fmt::Display for CompileCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, r#"cd "{}"; "#, self.cwd)?;
        if cfg!(target_os = "windows") {
            write!(f, "& ")?;
        }
        write!(f, r#""{}""#, self.executable)?;
        for arg in self.arguments.iter() {
            write!(f, r#" "{arg}""#)?;
        }
        fmt::Result::Ok(())
    }
}

// NewJob

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewJobResponse {
    pub has_inputs: bool,
    pub has_toolchain: bool,
    pub job_id: String,
    pub timeout: u32,
}

// RunJob

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunJobRequest {
    pub command: CompileCommand,
    pub outputs: Vec<String>,
    pub toolchain: Toolchain,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunJobRequestV2 {
    pub command: CompileCommand,
    pub outputs: Vec<String>,
    pub toolchain: Toolchain,
    pub labels: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RunJobError {
    Fatal(String),
    MissingJobInputs { inputs: Vec<String> },
    MissingJobResult,
    MissingToolchain,
    Retryable(String),
}

impl fmt::Display for RunJobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fatal(e) => write!(f, "Fatal error: {e:?}"),
            Self::MissingJobInputs { inputs } => write!(f, "Missing job inputs: {inputs:?}"),
            Self::MissingJobResult => write!(f, "Missing job result"),
            Self::MissingToolchain => write!(f, "Missing toolchain"),
            Self::Retryable(e) => write!(f, "Retryable error: {e}"),
        }
    }
}

impl RunJobError {
    pub fn job_cancelled() -> Self {
        Self::Retryable("Job cancelled".into())
    }
    pub fn server_terminated() -> Self {
        Self::Retryable("Server terminated".into())
    }
}

impl std::error::Error for RunJobError {}

#[cfg(feature = "dist-server")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RunJobStatus {
    Active { schedulers: Vec<String> },
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum RunJobResponse {
    Complete {
        result: BuildResult,
        server_id: String,
    },
    FatalError {
        message: String,
        server_id: String,
    },
    MissingJobInputs {
        server_id: String,
    },
    MissingJobResult {
        server_id: String,
    },
    MissingToolchain {
        server_id: String,
    },
    RetryableError {
        message: String,
        server_id: String,
    },
}

impl RunJobResponse {
    pub fn from_run_job_error(server_id: &str, err: &RunJobError) -> Self {
        match err {
            // Unrecoverable errors
            RunJobError::Fatal(e) => RunJobResponse::FatalError {
                message: format!("{e:#}"),
                server_id: server_id.into(),
            },
            // Retryable errors
            RunJobError::Retryable(e) => RunJobResponse::RetryableError {
                message: format!("{e:#}"),
                server_id: server_id.into(),
            },
            RunJobError::MissingJobInputs { .. } => RunJobResponse::MissingJobInputs {
                server_id: server_id.into(),
            },
            RunJobError::MissingJobResult => RunJobResponse::MissingJobResult {
                server_id: server_id.into(),
            },
            RunJobError::MissingToolchain => RunJobResponse::MissingToolchain {
                server_id: server_id.into(),
            },
        }
    }

    pub fn build_process_killed(server_id: &str) -> Self {
        RunJobResponse::RetryableError {
            message: "Build process killed".into(),
            server_id: server_id.to_owned(),
        }
    }

    pub fn job_cancelled(server_id: &str) -> Self {
        Self::from_run_job_error(server_id, &RunJobError::job_cancelled())
    }

    pub fn server_terminated(server_id: &str) -> Self {
        Self::from_run_job_error(server_id, &RunJobError::server_terminated())
    }
}

pub mod v3 {
    use super::*;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct NewJobRequest {
        pub toolchain: Toolchain,
        // List of hashes to input files (headers, modules, etc.)
        pub inputs: Vec<String>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct NewJobResponse {
        pub has_toolchain: bool,
        // List of hashes to input files (headers, modules, etc.)
        pub missing_inputs: Vec<String>,
        pub job_id: String,
        pub timeout: u32,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct RunJobRequest {
        pub command: CompileCommand,
        // Map of input file hashes to paths
        pub inputs: JobInputs,
        pub outputs: Vec<String>,
        pub toolchain: Toolchain,
        pub save_outputs_as_job_input: bool,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub enum RunJobResponse {
        Complete {
            result: BuildResult,
            server_id: String,
        },
        FatalError {
            message: String,
            server_id: String,
        },
        MissingJobInputs {
            inputs: Vec<String>,
            server_id: String,
        },
        MissingJobResult {
            server_id: String,
        },
        MissingToolchain {
            server_id: String,
        },
        RetryableError {
            message: String,
            server_id: String,
        },
    }

    impl From<RunJobResponse> for super::RunJobResponse {
        fn from(res: RunJobResponse) -> Self {
            match res {
                v3::RunJobResponse::Complete { result, server_id } => {
                    Self::Complete { result, server_id }
                }
                v3::RunJobResponse::FatalError { message, server_id } => {
                    Self::FatalError { message, server_id }
                }
                v3::RunJobResponse::MissingJobInputs { server_id, .. } => {
                    Self::MissingJobInputs { server_id }
                }
                v3::RunJobResponse::MissingJobResult { server_id } => {
                    Self::MissingJobResult { server_id }
                }
                v3::RunJobResponse::MissingToolchain { server_id } => {
                    Self::MissingToolchain { server_id }
                }
                v3::RunJobResponse::RetryableError { message, server_id } => {
                    Self::RetryableError { message, server_id }
                }
            }
        }
    }

    impl RunJobResponse {
        pub fn from_run_job_error(server_id: &str, err: &RunJobError) -> Self {
            match err {
                // Unrecoverable errors
                RunJobError::Fatal(e) => RunJobResponse::FatalError {
                    message: format!("{e:#}"),
                    server_id: server_id.into(),
                },
                // Retryable errors
                RunJobError::Retryable(e) => RunJobResponse::RetryableError {
                    message: format!("{e:#}"),
                    server_id: server_id.into(),
                },
                RunJobError::MissingJobInputs { inputs } => RunJobResponse::MissingJobInputs {
                    inputs: inputs.clone(),
                    server_id: server_id.into(),
                },
                RunJobError::MissingJobResult => RunJobResponse::MissingJobResult {
                    server_id: server_id.into(),
                },
                RunJobError::MissingToolchain => RunJobResponse::MissingToolchain {
                    server_id: server_id.into(),
                },
            }
        }

        pub fn build_process_killed(server_id: &str) -> Self {
            RunJobResponse::RetryableError {
                message: "Build process killed".into(),
                server_id: server_id.to_owned(),
            }
        }

        pub fn job_cancelled(server_id: &str) -> Self {
            Self::from_run_job_error(server_id, &RunJobError::job_cancelled())
        }

        pub fn server_terminated(server_id: &str) -> Self {
            Self::from_run_job_error(server_id, &RunJobError::server_terminated())
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum JobInputs {
    TarGZ {
        job_id: String,
    },
    Manifest {
        hash: String,
        dirs: Vec<String>,
        files: BTreeMap<String, String>,
        links: BTreeMap<String, String>,
    },
}

pub struct ArchiveContents {
    pub dirs: BTreeMap<PathBuf, PathBuf>,
    pub files: BTreeMap<PathBuf, PathBuf>,
    pub links: BTreeMap<PathBuf, (PathBuf, PathBuf)>,
}

// Toolchain

// TODO: Clone by assuming immutable/no GC for now
// TODO: make fields non-public?
// TODO: make archive_id validate that it's just a bunch of hex chars
#[derive(Debug, Hash, Eq, PartialEq, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Toolchain {
    pub archive_id: String,
}

impl fmt::Display for Toolchain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.archive_id)
    }
}

impl AsRef<str> for Toolchain {
    fn as_ref(&self) -> &str {
        self.archive_id.as_str()
    }
}

impl std::ops::Deref for Toolchain {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.archive_id.as_str()
    }
}

// Status

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerStatus {
    pub info: SysStats,
    pub jobs: JobStats,
    pub servers: Vec<ServerStatus>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerStatus {
    // Server id
    pub id: String,
    // Server performance stats
    pub info: SysStats,
    // Server job stats
    pub jobs: JobStats,
    // Duration since last update
    pub u_time: u64,
    // The age of the oldest active job (in seconds)
    pub max_job_age: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SysStats {
    pub cpu_usage: f32,
    pub mem_avail: u64,
    pub mem_total: u64,
    pub num_cpus: usize,
    pub occupancy: usize,
    pub pre_fetch: usize,
}

impl Default for SysStats {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            mem_avail: 0,
            mem_total: 0,
            num_cpus: 0,
            occupancy: 0,
            pre_fetch: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobStats {
    pub accepted: u64,
    pub finished: u64,
    pub loading: u64,
    pub pending: u64,
    pub running: u64,
}

// A struct describing the status of a scheduler or server
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusUpdate {
    // The id of the scheduler or server
    pub id: String,
    // The name of the queue for this scheduler or server
    pub queue: String,
    // The sysinfo stats of the scheduler or server
    pub info: SysStats,
    pub jobs: JobStats,
    // When this event was created as microseconds since the Unix epoch as (secs, nanos)
    pub timestamp: (u64, u32),
    // The age of the oldest active job in microseconds as (secs, nanos)
    pub max_job_age: (u64, u32),
}

// SubmitToolchain

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum SubmitToolchainResult {
    Success,
    Error { message: String },
}

///////////////////

///////////////////

#[cfg(feature = "dist-server")]
#[async_trait]
pub trait SchedulerService: Send + Sync {
    async fn get_status(&self) -> Result<SchedulerStatus>;

    async fn has_toolchain(&self, toolchain: &Toolchain) -> bool;

    async fn put_toolchain(
        &self,
        toolchain: &Toolchain,
        toolchain_archive: opendal::Buffer,
    ) -> Result<()>;

    async fn del_toolchain(&self, toolchain: &Toolchain) -> Result<()>;

    async fn new_job(
        &self,
        toolchain: Toolchain,
        inputs: opendal::Buffer,
    ) -> Result<NewJobResponse>;

    async fn new_job_with_inputs(
        &self,
        toolchain: Toolchain,
        inputs: Vec<String>,
    ) -> Result<v3::NewJobResponse>;
    async fn run_job(&self, job_id: &str, request: v3::RunJobRequest)
    -> Result<v3::RunJobResponse>;
    async fn put_job(&self, job_id: &str, inputs: opendal::Buffer) -> Result<()>;
    async fn del_job(&self, job_id: &str) -> Result<()>;

    async fn job_finished(&self, job_id: &str, server: Option<StatusUpdate>) -> Result<()>;

    async fn update_server_status(
        &self,
        status: StatusUpdate,
        job_status: Option<bool>,
    ) -> Result<()>;
}

#[cfg(feature = "dist-server")]
#[async_trait]
pub trait ServerService: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    async fn run_job(
        &self,
        job_id: &str,
        reply_to: &str,
        toolchain: Toolchain,
        command: CompileCommand,
        inputs: JobInputs,
        outputs: Vec<String>,
        save_outputs_as_job_input: bool,
    ) -> Result<v3::RunJobResponse>;

    async fn notify_run_job_err(&self, job_id: &str, err: RunJobError) -> Result<()>;
    async fn notify_run_job_res(&self, job_id: &str, res: &v3::RunJobResponse) -> Result<()>;

    async fn update_scheduler_status(&self, status: StatusUpdate) -> Result<()>;
}

#[cfg(feature = "dist-server")]
#[async_trait]
pub trait InputsService: Send + Sync {
    async fn get_inputs(&self, inputs: &JobInputs, outputs: &[PathBuf]) -> Result<PathBuf>;
}

#[cfg(feature = "dist-server")]
#[async_trait]
pub trait ToolchainService: Send + Sync {
    async fn get_toolchain(&self, toolchain: &Toolchain) -> Result<PathBuf>;
}

#[cfg(feature = "dist-server")]
#[async_trait]
pub trait BuilderIncoming: Send + Sync {
    // From Server
    async fn run_build(
        &self,
        job_id: &str,
        toolchain_dir: &Path,
        inputs_dir: &Path,
        command: CompileCommand,
        outputs: Vec<String>,
        save_outputs_as_job_input: bool,
    ) -> std::result::Result<BuildOutput, BuildError>;
    // From Server
    async fn finish_build(&self, job_id: &str);
    // From Server
    async fn shutdown(&self);
}

/////////
#[async_trait]
pub trait Client: Send + Sync {
    async fn get_job_inputs(
        &self,
        contents: ArchiveContents,
        inputs: BTreeMap<String, JobInput>,
    ) -> Result<(JobInputs, BTreeMap<String, JobInput>)>;
    // To Scheduler
    async fn new_job(
        &self,
        toolchain: Toolchain,
        inputs: Vec<String>,
    ) -> Result<v3::NewJobResponse>;
    // To Scheduler
    async fn put_job(&self, inputs: HashMap<String, JobInput>) -> Result<()>;
    // To Scheduler
    async fn del_job(&self, job_id: &str) -> Result<()>;
    // To Scheduler
    #[allow(clippy::too_many_arguments)]
    async fn run_job(
        &self,
        job_id: &str,
        timeout: Duration,
        toolchain: Toolchain,
        command: CompileCommand,
        inputs: JobInputs,
        outputs: Vec<String>,
        save_outputs_as_job_input: bool,
    ) -> Result<v3::RunJobResponse>;
    // To Scheduler
    async fn get_status(&self) -> Result<SchedulerStatus>;
    // Write to local toolchain cache if not already written, then upload to Scheduler
    async fn put_toolchain(
        &self,
        toolchain: Toolchain,
        packaged: Option<Arc<dyn pkg::ArchivePackager>>,
    ) -> Result<SubmitToolchainResult>;
    async fn hash_toolchain(
        &self,
        compiler_path: &Path,
        weak_toolchain_key: &str,
        toolchain_packager: Box<dyn pkg::ToolchainPackager>,
        path_transformer: &mut PathTransformer,
    ) -> Result<(
        Toolchain,
        Option<(String, PathBuf)>,
        Option<Arc<dyn pkg::ArchivePackager>>,
    )>;
    fn fallback_to_local_compile(&self) -> bool;
    fn max_retries(&self) -> f64;
    fn request_timeout(&self) -> u32;
    async fn get_custom_toolchain(&self, exe: &Path) -> Option<PathBuf>;
}
