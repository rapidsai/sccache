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

use crate::dist;
use async_trait::async_trait;
use fs_err as fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::str;
use std::sync::Arc;

use crate::errors::*;

#[cfg(all(
    feature = "dist-client",
    any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "freebsd"
    )
))]
pub use self::toolchain_imp::*;

#[async_trait]
pub trait ToolchainPackager: Send + Sync {
    async fn package(&self) -> Result<Arc<dyn PackagedToolchain>>;
}

#[async_trait]
pub trait PackagedToolchain: Send + Sync {
    async fn compute_hash(&self) -> Result<String>;
    async fn write_tar_gz(
        &self,
        toolchain: &dist::Toolchain,
        writer: &mut (dyn Write + Send),
    ) -> Result<()>;
}

pub trait InputsWriter: Write + Send {
    fn finish(self: Box<Self>) -> Result<(u64, u64)>;
}

pub struct InputsCompressor<W: Write + Send> {
    inner: flate2::write::ZlibEncoder<W>,
}

impl<W: Write + Send> InputsCompressor<W> {
    pub fn new(compressor: flate2::write::ZlibEncoder<W>) -> Box<Self> {
        Box::new(Self { inner: compressor })
    }
}

impl<W: Write + Send> Write for InputsCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<W: Write + Send> InputsWriter for InputsCompressor<W> {
    fn finish(self: Box<Self>) -> Result<(u64, u64)> {
        let mut inner = self.inner;
        inner.flush().context("failed to flush compressor")?;
        let total_in = inner.total_in();
        let total_out = inner.total_out();
        trace!("Compressed inputs from {total_in} -> {total_out}");
        inner.finish().context("failed to finish compressor")?;
        Ok((total_in, total_out))
    }
}

#[async_trait]
pub trait InputsPackager: Send {
    async fn write_inputs(
        self: Box<Self>,
        path_transformer: &mut dist::PathTransformer,
        inputs_writer: Box<dyn InputsWriter>,
    ) -> Result<()>;
}

#[cfg(not(all(
    feature = "dist-client",
    any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "freebsd"
    )
)))]
mod toolchain_imp {
    use std::sync::Arc;

    use super::{PackagedToolchain, ToolchainPackager};
    use async_trait::async_trait;

    use crate::errors::*;

    // Distributed client, but an unsupported platform for toolchain packaging so
    // create a failing implementation that will conflict with any others.
    #[async_trait]
    impl<T: Send + Sync> ToolchainPackager for T {
        async fn package(&self) -> Result<Arc<dyn PackagedToolchain>> {
            bail!("Automatic packaging not supported on this platform")
        }
    }
}

#[cfg(all(
    feature = "dist-client",
    any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "freebsd"
    )
))]
mod toolchain_imp {
    use async_trait::async_trait;
    use fs_err as fs;
    use is_executable::IsExecutable;
    use std::collections::BTreeMap;
    use std::ffi::OsString;
    use std::io::{Read, Write};
    use std::path::{Component, Path, PathBuf};
    use std::process;
    use std::str;
    use walkdir::WalkDir;

    use super::{PackagedToolchain, dist::Toolchain, tar_safe_path};
    use crate::errors::*;

    pub struct ToolchainPackaged {
        executable: PathBuf,
        // Put dirs and file in a deterministic order (map from tar_path -> real_path)
        dir_set: BTreeMap<PathBuf, PathBuf>,
        file_set: BTreeMap<PathBuf, PathBuf>,
        // Symlinks to add to the tar
        // These are _not_ tar safe, and must be made so before being added to the tar (see
        // `tar_safe_path`).
        symlinks: BTreeMap<PathBuf, PathBuf>,
    }

    impl ToolchainPackaged {
        pub fn new(executable: PathBuf) -> Self {
            Self {
                executable,
                dir_set: BTreeMap::new(),
                file_set: BTreeMap::new(),
                symlinks: BTreeMap::new(),
            }
        }

        pub fn add_common(&mut self) -> Result<()> {
            self.add_dir(PathBuf::from("/tmp"))
        }

        pub fn add_executable_and_deps(
            &mut self,
            env_vars: &[(OsString, OsString)],
            executable: &Path,
        ) -> Result<()> {
            let mut remaining = vec![executable.to_owned()];
            while let Some(obj_path) = remaining.pop() {
                assert!(obj_path.is_absolute());
                // If any parent directories are a symlink, resolve it first and record the link.
                // This is important because ld-linux may not be configured to look in the resolved
                // or non-resolved directory (i.e., both directories must work at runtime).
                //
                let tar_path = self.tarify_path(&obj_path)?;
                // If file already in the set, assume we've analysed all deps
                if self.file_set.contains_key(&tar_path) {
                    continue;
                }
                let ldd_libraries = find_ldd_libraries(env_vars, &obj_path).with_context(|| {
                    format!("Failed to analyse {} with ldd", obj_path.display())
                })?;
                remaining.extend(ldd_libraries);
                trace!(
                    "add_executable_and_deps {} -> {}",
                    obj_path.display(),
                    tar_path.display()
                );
                self.file_set.insert(tar_path, obj_path);
            }
            Ok(())
        }

        pub fn add_dir(&mut self, dir_path: PathBuf) -> Result<()> {
            assert!(dir_path.is_absolute());
            if !dir_path.is_dir() {
                bail!(format!(
                    "{} was not a dir when readying for tar",
                    dir_path.to_string_lossy()
                ))
            }
            if dir_path
                .components()
                .next_back()
                .expect("asserted absolute")
                == Component::RootDir
            {
                return Ok(());
            }
            let tar_path = self.tarify_path(&dir_path)?;
            trace!("add_dir {} -> {}", dir_path.display(), tar_path.display());
            self.dir_set.insert(tar_path, dir_path);
            Ok(())
        }

        pub fn add_file(
            &mut self,
            env_vars: &[(OsString, OsString)],
            file_path: PathBuf,
        ) -> Result<()> {
            assert!(file_path.is_absolute());
            if !file_path.is_file() {
                bail!(format!(
                    "{} was not a file when readying for tar",
                    file_path.to_string_lossy()
                ))
            }
            if file_path.is_executable()
                && self.add_executable_and_deps(env_vars, &file_path).is_ok()
            {
                return Ok(());
            }
            let tar_path = self.tarify_path(&file_path)?;
            trace!("add_file {} -> {}", file_path.display(), tar_path.display());
            self.file_set.insert(tar_path, file_path);
            Ok(())
        }

        pub fn add_link(&mut self, file_path: &Path, link_name: &Path) -> Result<()> {
            assert!(file_path.is_absolute());
            assert!(link_name.is_absolute());
            let mut tarify = |path: &Path| -> Result<PathBuf> {
                Ok(if let Some(parent) = path.parent() {
                    let root = path.components().next().unwrap();
                    let root = Path::new(root.as_os_str());
                    root.join(if let Some(name) = path.file_name() {
                        self.tarify_path(parent)?.join(name)
                    } else {
                        self.tarify_path(path)?
                    })
                } else {
                    self.tarify_path(path)?
                })
            };

            let tar_link_name = tarify(link_name)?;
            let tar_file_path = tarify(file_path)?;
            trace!(
                "add_link {} -> {}",
                tar_link_name.display(),
                tar_file_path.display()
            );
            self.symlinks.insert(tar_link_name, tar_file_path);
            Ok(())
        }

        pub fn add_dir_contents(
            &mut self,
            env_vars: &[(OsString, OsString)],
            dir_path: &Path,
        ) -> Result<()> {
            // Although by not following symlinks we could break a custom
            // constructed toolchain with links everywhere, this is just a
            // best-effort auto packaging
            for entry in WalkDir::new(dir_path).follow_links(false) {
                let entry = entry?;
                let path = entry.path();
                let file_type = entry.file_type();
                if file_type.is_dir() {
                    continue;
                } else if file_type.is_symlink() {
                    // Skip symlinks that point to nothing
                    if !path.exists() {
                        continue;
                    }
                    let metadata = fs::metadata(path)?;
                    if !metadata.file_type().is_file() {
                        continue;
                    }
                } else if !file_type.is_file() {
                    // Device or other oddity
                    continue;
                }
                trace!("walkdir add_file {}", path.display());
                // It's either a file, or a symlink pointing to a file
                self.add_file(env_vars, path.to_owned())?;
            }
            Ok(())
        }

        /// Simplify the path and strip the leading slash.
        ///
        /// Symlinks in the path are recorded for inclusion in the tarball.
        fn tarify_path(&mut self, path: &Path) -> Result<PathBuf> {
            super::tarify_path(&mut self.symlinks, path).map(tar_safe_path)
        }
    }

    #[async_trait]
    impl PackagedToolchain for ToolchainPackaged {
        async fn compute_hash(&self) -> Result<String> {
            let mut digest = crate::util::Digest::new();

            for (from_path, to_path) in self.symlinks.iter() {
                if to_path.is_file() {
                    digest = digest.with_file(to_path).await?;
                }
                digest.update(to_path.to_string_lossy().as_bytes());
                digest.update(from_path.to_string_lossy().as_bytes());
            }
            for (_, dir_path) in self.dir_set.iter() {
                digest.update(dir_path.to_string_lossy().as_bytes());
            }
            for (_, file_path) in self.file_set.iter() {
                digest = digest.with_file(file_path).await?;
                digest.update(file_path.to_string_lossy().as_bytes());
            }

            Ok(digest.finish())
        }

        async fn write_tar_gz(
            &self,
            toolchain: &Toolchain,
            writer: &mut (dyn Write + Send),
        ) -> Result<()> {
            use gzp::{
                deflate::Gzip,
                par::compress::{Compression, ParCompressBuilder},
            };

            let dir_set = self.dir_set.clone();
            let file_set = self.file_set.clone();
            let symlinks = self.symlinks.clone();

            debug!(
                "Compressing toolchain for {:?} -> {:?}",
                self.executable.display(),
                toolchain.archive_id,
            );

            tokio::task::block_in_place(move || {
                std::thread::scope(|scope| {
                    let compressor = ParCompressBuilder::<Gzip>::new()
                        .compression_level(Compression::default())
                        .num_threads(crate::util::num_cpus())?
                        .from_borrowed_writer(writer, scope);

                    let mut builder = tar::Builder::new(compressor);

                    // Add symlinks first to ensure the symlinks exist before
                    // files are unpacked to paths that may traverse them
                    for (from_path, to_path) in symlinks.iter() {
                        let mut header = tar::Header::new_gnu();
                        header.set_size(0);
                        header.set_mtime(0);
                        header.set_entry_type(tar::EntryType::Symlink);
                        // Leave `to_path` as absolute, assuming the tar will
                        // be used in a chroot-like environment.
                        builder.append_link(&mut header, tar_safe_path(from_path), to_path)?;
                    }
                    for (tar_path, dir_path) in dir_set.iter() {
                        builder.append_dir(tar_path, dir_path)?;
                    }
                    for (tar_path, file_path) in file_set.iter() {
                        builder.append_path_with_name(file_path, tar_path)?;
                    }

                    builder.finish().map_err(anyhow::Error::new)
                })
            })
        }
    }

    // The dynamic linker is the only thing that truly knows how dynamic libraries will be
    // searched for, so we need to ask it directly.
    //
    // This function will extract any absolute paths from output like the following:
    // $ ldd /bin/ls
    //         linux-vdso.so.1 =>  (0x00007ffeb41f6000)
    //         libselinux.so.1 => /lib/x86_64-linux-gnu/libselinux.so.1 (0x00007f6877f4f000)
    //         libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f6877b85000)
    //         libpcre.so.3 => /lib/x86_64-linux-gnu/libpcre.so.3 (0x00007f6877915000)
    //         libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f6877711000)
    //         /lib64/ld-linux-x86-64.so.2 (0x00007f6878171000)
    //         libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f68774f4000)
    //
    // Elf executables can be statically or dynamically linked, and position independent (PIE) or not:
    // - dynamic + PIE = ET_DYN, ldd stdouts something like the list above and exits with code 0
    // - dynamic + non-PIE = ET_EXEC, ldd stdouts something like the list above and exits with code 0
    // - static + PIE = ET_DYN, ldd stdouts something like "\tstatically linked" or
    //   "\tldd (0x7f79ef662000)" and exits with code 0
    // - static + non-PIE = ET_EXEC, ldd stderrs something like "\tnot a dynamic executable" or
    //   "ldd: a.out: Not a valid dynamic program" and exits with code 1
    //
    fn find_ldd_libraries(
        env_vars: &[(OsString, OsString)],
        executable: &Path,
    ) -> Result<Vec<PathBuf>> {
        let process::Output {
            status,
            stdout,
            stderr,
        } = process::Command::new("ldd")
            .envs(env_vars.to_vec())
            .arg(executable)
            .output()?;

        // Not a file ldd can handle. This can be a non-executable, or a static non-PIE
        if !status.success() {
            // Best-effort detection of static non-PIE
            let mut elf = fs::File::open(executable)?;
            let mut elf_bytes = [0; 0x12];
            elf.read_exact(&mut elf_bytes)?;
            if elf_bytes[..0x4] != [0x7f, 0x45, 0x4c, 0x46] {
                bail!("Elf magic not found")
            }
            let little_endian = match elf_bytes[0x5] {
                1 => true,
                2 => false,
                _ => bail!("Invalid endianness in elf header"),
            };
            let e_type = if little_endian {
                ((elf_bytes[0x11] as u16) << 8) | elf_bytes[0x10] as u16
            } else {
                ((elf_bytes[0x10] as u16) << 8) | elf_bytes[0x11] as u16
            };
            if e_type != 0x02 {
                bail!("ldd failed on a non-ET_EXEC elf")
            }
            // It appears to be an ET_EXEC, good enough for us
            return Ok(vec![]);
        }

        if !stderr.is_empty() {
            trace!(
                "ldd stderr non-empty: {:?}",
                String::from_utf8_lossy(&stderr)
            );
        }

        let stdout = str::from_utf8(&stdout).context("ldd output not utf8")?;
        Ok(parse_ldd_output(stdout))
    }

    // If it's a static PIE the output will be a line like "\tstatically linked", so be forgiving
    // in the parsing here and treat parsing oddities as an empty list.
    fn parse_ldd_output(stdout: &str) -> Vec<PathBuf> {
        let mut libs = vec![];
        for line in stdout.lines() {
            let line = line.trim();
            let mut parts: Vec<_> = line.split_whitespace().collect();

            // Remove a possible "(0xdeadbeef)" or assume this isn't a library line
            match parts.pop() {
                Some(s) if s.starts_with('(') && s.ends_with(')') => (),
                Some(_) | None => continue,
            }

            if parts.len() > 3 {
                continue;
            }

            let libpath = match (parts.first(), parts.get(1), parts.get(2)) {
                // "linux-vdso.so.1 =>  (0x00007ffeb41f6000)"
                (Some(_libname), Some(&"=>"), None) => continue,
                // "libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f6877b85000)"
                (Some(libname), Some(&"=>"), Some(libpath)) => {
                    // ldd (version 2.30) will output something like this:
                    //   ...
                    //   /lib64/ld-linux-x86-64.so.2 => /usr/lib64/ld-linux-x86-64.so.2
                    //   ...
                    // We need to add /lib64/ld-linux-x86-64.so.2 to deps, else we'll get error "No
                    // such file or directory".
                    //
                    // Workaround: add libname to deps if it's absolute and exists.
                    let libname_path = PathBuf::from(libname);
                    if libname_path.is_absolute() {
                        libs.push(libname_path);
                    }

                    PathBuf::from(libpath)
                }
                // "/lib64/ld-linux-x86-64.so.2 (0x00007f6878171000)"
                (Some(libpath), None, None) => PathBuf::from(libpath),
                _ => continue,
            };

            if !libpath.is_absolute() {
                continue;
            }

            libs.push(libpath);
        }

        libs
    }

    #[test]
    fn test_ldd_parse() {
        let ubuntu_ls_output = "\tlinux-vdso.so.1 =>  (0x00007fffcfffe000)
\tlibselinux.so.1 => /lib/x86_64-linux-gnu/libselinux.so.1 (0x00007f69caa6b000)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f69ca6a1000)
\tlibpcre.so.3 => /lib/x86_64-linux-gnu/libpcre.so.3 (0x00007f69ca431000)
\tlibdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f69ca22d000)
\t/lib64/ld-linux-x86-64.so.2 (0x00007f69cac8d000)
\tlibpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f69ca010000)
";
        assert_eq!(
            parse_ldd_output(ubuntu_ls_output)
                .iter()
                .map(|p| p.to_str().unwrap())
                .collect::<Vec<_>>(),
            &[
                "/lib/x86_64-linux-gnu/libselinux.so.1",
                "/lib/x86_64-linux-gnu/libc.so.6",
                "/lib/x86_64-linux-gnu/libpcre.so.3",
                "/lib/x86_64-linux-gnu/libdl.so.2",
                "/lib64/ld-linux-x86-64.so.2",
                "/lib/x86_64-linux-gnu/libpthread.so.0",
            ]
        );
    }

    #[test]
    fn test_ldd_parse_static() {
        let static_outputs = &[
            "\tstatically linked",    // glibc ldd output
            "\tldd (0x7f79ef662000)", // musl ldd output
        ];
        for static_output in static_outputs {
            assert_eq!(parse_ldd_output(static_output).len(), 0);
        }
    }

    #[test]
    fn test_ldd_parse_v2_30() {
        let archlinux_ls_output = "\tlinux-vdso.so.1 (0x00007ffddc1f6000)
\tlibcap.so.2 => /usr/lib/libcap.so.2 (0x00007f4980989000)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f69ca6a1000)
\tlibc.so.6 => /usr/lib/libc.so.6 (0x00007f49807c2000)
\t/lib64/ld-linux-x86-64.so.2 => /usr/lib64/ld-linux-x86-64.so.2 (0x00007f49809e9000)
";
        assert_eq!(
            parse_ldd_output(archlinux_ls_output)
                .iter()
                .map(|p| p.to_str().unwrap())
                .collect::<Vec<_>>(),
            &[
                "/usr/lib/libcap.so.2",
                "/lib/x86_64-linux-gnu/libc.so.6",
                "/usr/lib/libc.so.6",
                "/lib64/ld-linux-x86-64.so.2",
                "/usr/lib64/ld-linux-x86-64.so.2",
            ]
        );
    }
}

/// Strip a leading slash, if any.
pub fn tar_safe_path<P: AsRef<Path>>(path: P) -> PathBuf {
    path.as_ref()
        .strip_prefix(Component::RootDir)
        .map(ToOwned::to_owned)
        .unwrap_or(path.as_ref().to_path_buf())
}

pub fn make_tar_header(src: &Path, dest: &str) -> io::Result<(tar::Header, PathBuf)> {
    let metadata_res = fs::metadata(src);

    let mut file_header = tar::Header::new_ustar();
    // TODO: test this works
    if let Ok(metadata) = metadata_res {
        // TODO: if the source file is a symlink, I think this does bad things
        file_header.set_metadata(&metadata);
    } else {
        warn!("Couldn't get metadata of file {src:?}, falling back to some defaults");
        file_header.set_mode(0o644);
        file_header.set_uid(0);
        file_header.set_gid(0);
        file_header.set_mtime(0);
        file_header
            .set_device_major(0)
            .expect("expected a ustar header");
        file_header
            .set_device_minor(0)
            .expect("expected a ustar header");
        file_header.set_entry_type(tar::EntryType::file());
    }

    Ok((file_header, tar_safe_path(dest)))
}

pub fn tarify_path(
    symlinks: &mut std::collections::BTreeMap<PathBuf, PathBuf>,
    path: &Path,
) -> Result<PathBuf> {
    SimplifyPath {
        resolved_symlinks: Some(symlinks),
    }
    .simplify(path)
}

/// Simplify a path to one without any relative components, erroring if it looks
/// like there could be any symlink complexity that means a simplified path is not
/// equivalent to the original (see the documentation of `fs::canonicalize` for an
/// example).
///
/// So why avoid resolving symlinks? Any path that we are trying to simplify has
/// (usually) been added to an archive because something will try access it, but
/// resolving symlinks (be they for the actual file or directory components) can
/// make the accessed path 'disappear' in favour of the canonical path.
pub fn simplify_path(path: &Path) -> Result<PathBuf> {
    SimplifyPath {
        resolved_symlinks: None,
    }
    .simplify(path)
}

struct SimplifyPath<'a> {
    pub resolved_symlinks: Option<&'a mut std::collections::BTreeMap<PathBuf, PathBuf>>,
}

impl SimplifyPath<'_> {
    pub fn simplify(&mut self, path: &Path) -> Result<PathBuf> {
        let mut final_path = PathBuf::new();
        for component in path.components() {
            match component {
                c @ Component::RootDir | c @ Component::Prefix(_) | c @ Component::Normal(_) => {
                    final_path.push(c);
                    if self.resolved_symlinks.is_some() && final_path.is_symlink() {
                        let parent = final_path.parent().expect("symlinks have parents");
                        let link_target = final_path.read_link()?;
                        let new_final_path = self.simplify(&parent.join(&link_target))?;
                        let old_final_path =
                            std::mem::replace(&mut final_path, new_final_path.clone());
                        #[allow(clippy::unnecessary_unwrap)]
                        self.resolved_symlinks
                            .as_mut()
                            .unwrap()
                            .insert(old_final_path, new_final_path);
                    }
                }
                Component::ParentDir => {
                    // If the path is doing funny symlink traversals, just give up.
                    //
                    // This case should only occur if `resolved_symlinks` is `None`.
                    if final_path.is_symlink() {
                        bail!("Cannot handle symlinks in parent paths")
                    }
                    final_path.pop();
                }
                Component::CurDir => continue,
            }
        }
        Ok(final_path)
    }
}
