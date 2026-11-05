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

use crate::{
    dist::{self, JobInput},
    errors::*,
    util::{SimplifyPath, bytes_to_str, path_to_bytes},
};
use async_trait::async_trait;
use fs_err as fs;
use is_executable::IsExecutable;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    io::{self, Write},
    path::{Component, Path, PathBuf},
    process, str,
    sync::Arc,
};
use walkdir::WalkDir;

#[allow(unused)]
pub use self::toolchain_imp::*;

#[async_trait]
pub trait ToolchainPackager: Send + Sync {
    async fn package(
        self: Box<Self>,
        path_transformer: &mut dist::PathTransformer,
    ) -> Result<Arc<dyn ArchivePackager>>;
}

#[async_trait]
pub trait ArchivePackager: Send + Sync {
    async fn compute_hash(&self) -> Result<String>;
    fn compress(&self, writer: &mut (dyn Write + Send)) -> Result<()>;
    fn write_tar_gz(&self, writer: &mut (dyn Write + Send)) -> Result<()>;
}

#[async_trait]
pub trait InputsPackager: Send {
    async fn describe_inputs(
        self: Box<Self>,
        path_transformer: &mut dist::PathTransformer,
    ) -> Result<(dist::ArchiveContents, BTreeMap<String, JobInput>)>;
}

pub struct ArchiveBuilder<'a> {
    // Put dirs and file in a deterministic order (map from tar_path -> real_path)
    dirs_set: BTreeSet<PathBuf>,
    file_set: BTreeSet<PathBuf>,
    // Symlinks to add to the tar
    // These are _not_ tar safe, and must be made so before being added to the tar (see
    // `tar_safe_path`).
    symlinks: BTreeMap<PathBuf, PathBuf>,
    path_transformer: &'a mut dist::PathTransformer,
}

impl<'a> ArchiveBuilder<'a> {
    pub fn new(path_transformer: &'a mut dist::PathTransformer) -> Self {
        Self {
            dirs_set: BTreeSet::new(),
            file_set: BTreeSet::new(),
            symlinks: BTreeMap::new(),
            path_transformer,
        }
    }

    pub fn add_path<P: AsRef<Path>>(
        &mut self,
        env_vars: &[(OsString, OsString)],
        path: P,
    ) -> Result<()> {
        let path = path.as_ref();
        if path.is_absolute() && path.exists() {
            if path.is_dir() {
                self.add_dir(path)?;
            } else {
                self.add_file(env_vars, path)?;
            }
        }
        Ok(())
    }

    pub fn add_executable_and_deps<P: AsRef<Path>>(
        &mut self,
        env_vars: &[(OsString, OsString)],
        executable: P,
    ) -> Result<()> {
        let executable = executable.as_ref();
        let mut remaining = vec![executable.to_owned()];
        while let Some(obj_path) = remaining.pop() {
            assert!(obj_path.is_absolute());
            // If any parent directories are a symlink, resolve it first and record the link.
            // This is important because ld-linux may not be configured to look in the resolved
            // or non-resolved directory (i.e., both directories must work at runtime).
            //
            let obj_path = self.simplify_path(&obj_path)?;
            // If file already in the set, assume we've analysed all deps
            if self.file_set.contains(&obj_path) {
                continue;
            }
            let ldd_libraries = find_ldd_libraries(env_vars, &obj_path)
                .with_context(|| format!("Failed to analyse {} with ldd", obj_path.display()))?;
            remaining.extend(ldd_libraries);
            // trace!("add_executable_and_deps {}", obj_path.display());
            self.file_set.insert(obj_path);
        }
        Ok(())
    }

    pub fn add_dir<P: AsRef<Path>>(&mut self, dir_path: P) -> Result<()> {
        let dir_path = dir_path.as_ref();
        assert!(dir_path.is_absolute());
        if !dir_path.is_dir() {
            bail!("'{}' is not a directory", dir_path.display())
        }
        if dir_path
            .components()
            .next_back()
            .expect("asserted absolute")
            == Component::RootDir
        {
            return Ok(());
        }
        let dir_path = self.simplify_path(dir_path)?;
        // trace!("add_dir '{}'", dir_path.display());
        self.dirs_set.insert(dir_path);
        Ok(())
    }

    pub fn add_file<P: AsRef<Path>>(
        &mut self,
        env_vars: &[(OsString, OsString)],
        file_path: P,
    ) -> Result<()> {
        let file_path = file_path.as_ref();
        assert!(file_path.is_absolute());
        if !file_path.is_file() {
            bail!("'{}' is not a file", file_path.display())
        }
        if file_path.is_executable() && self.add_executable_and_deps(env_vars, file_path).is_ok() {
            return Ok(());
        }
        let file_path = self.simplify_path(file_path)?;
        // trace!("add_file '{}'", file_path.display());
        self.file_set.insert(file_path);
        Ok(())
    }

    pub fn add_link<T: AsRef<Path>, N: AsRef<Path>>(&mut self, name: N, target: T) -> Result<()> {
        let name = name.as_ref();
        assert!(name.is_absolute());
        let target = target.as_ref();

        let mut simplify = |path: &Path| -> Result<PathBuf> {
            if path.is_symlink()
                && let Some(name) = path.file_name()
                && let Some(path) = path.parent()
            {
                self.simplify_path(path).map(|path| path.join(name))
            } else {
                self.simplify_path(path)
            }
        };

        // Simplify the link name to record any symlinks it traverses.
        let name = simplify(name)?;

        // Simplify the link target
        let target = if let Some(base) = name.parent() {
            simplify(&base.join(target))?
        } else if target.is_absolute() {
            simplify(target)?
        } else {
            target.to_path_buf()
        };

        // trace!("add_link '{}' -> '{}'", name.display(), target.display());
        self.symlinks.insert(name, target);
        Ok(())
    }

    pub fn add_dir_contents<P: AsRef<Path>>(
        &mut self,
        env_vars: &[(OsString, OsString)],
        dir_path: P,
    ) -> Result<()> {
        let dir_path = dir_path.as_ref();
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
            // trace!("walkdir add_file '{}'", path.display());
            // It's either a file, or a symlink pointing to a file
            self.add_file(env_vars, path)?;
        }
        Ok(())
    }

    /// Simplify the path.
    /// Symlinks in the path are recorded for inclusion in the tarball.
    fn simplify_path<P: AsRef<Path>>(&mut self, path: P) -> Result<PathBuf> {
        SimplifyPath {
            dirs: Some(&mut self.dirs_set),
            resolved_symlinks: Some(&mut self.symlinks),
        }
        .simplify(path.as_ref())
    }

    pub fn describe(self) -> Result<dist::ArchiveContents> {
        use itertools::Itertools;

        // Adjust the archive paths with the PathTransformer

        let dirs_set = self
            .dirs_set
            .into_iter()
            .map(|dir_path| {
                self.path_transformer
                    .as_dist(&dir_path)
                    .with_context(|| format!("Unable to transform directory path {dir_path:?}"))
                    .map(|dist_path| (dist_path, dir_path))
            })
            .try_collect()
            .context("Failed transforming intermediate directory paths")?;

        let file_set = self
            .file_set
            .into_iter()
            .map(|src_path| {
                self.path_transformer
                    .as_dist(&src_path)
                    .with_context(|| format!("Unable to transform file path {src_path:?}"))
                    .map(|dist_path| (dist_path, src_path))
            })
            .try_collect()
            .context("Failed transforming file paths")?;

        let symlinks = self
            .symlinks
            .into_iter()
            .map(|(src_path, dst_path)| {
                self.path_transformer
                    .as_dist(&src_path)
                    .with_context(|| format!("Unable to transform symlink path {src_path:?}"))
                    .map(|dist_path| {
                        (
                            dist_path,
                            // Leave `dst_path` as absolute, assuming the tar will
                            // be used in a chroot-like environment.
                            (src_path, dst_path),
                        )
                    })
            })
            .try_collect()
            .context("Failed transforming symlink paths")?;

        Ok(dist::ArchiveContents {
            dirs: path_map(dirs_set),
            files: path_map(file_set),
            links: path_map(symlinks),
        })
    }

    pub fn build(self) -> Result<Arc<dyn ArchivePackager>> {
        let dist::ArchiveContents {
            dirs: dirs_set,
            files: file_set,
            links: symlinks,
        } = self.describe()?;
        Ok(Arc::new(ArchivePackagerImpl {
            dirs_set: path_map(
                dirs_set
                    .into_iter()
                    // Strip the leading slash
                    .map(|(k, v)| (tar_safe_path(k), v))
                    .collect(),
            ),
            file_set: path_map(
                file_set
                    .into_iter()
                    // Strip the leading slash
                    .map(|(k, v)| (tar_safe_path(k), v))
                    .collect(),
            ),
            symlinks: path_map(
                symlinks
                    .into_iter()
                    // Strip the leading slash
                    .map(|(k, v)| (tar_safe_path(k), v))
                    .collect(),
            ),
        }))
    }
}

// The maps above are collected through a `Vec` so they share one `BTreeMap`
// construction (and the stable sort inside it) per value type, instead of one
// per iterator type.
fn path_map<V>(entries: Vec<(PathBuf, V)>) -> BTreeMap<PathBuf, V> {
    entries.into_iter().collect()
}

pub struct ArchivePackagerImpl {
    // Put dirs and file in a deterministic order (map from tar_path -> real_path)
    dirs_set: BTreeMap<PathBuf, PathBuf>,
    file_set: BTreeMap<PathBuf, PathBuf>,
    // Symlinks to add to the tar
    // These are _not_ tar safe, and must be made so before being added to the tar (see
    // `tar_safe_path`).
    symlinks: BTreeMap<PathBuf, (PathBuf, PathBuf)>,
}

#[async_trait]
impl ArchivePackager for ArchivePackagerImpl {
    async fn compute_hash(&self) -> Result<String> {
        let mut digest = crate::util::Digest::new();

        for (tar_path, (src_path, dst_path)) in self.symlinks.iter() {
            if let Some(base) = src_path.parent() {
                let abs_path = base.join(dst_path);
                if abs_path.is_file() {
                    digest = digest.with_file(&abs_path).await?;
                }
            }
            digest.update(&path_to_bytes(tar_path.as_path())?);
            digest.update(&path_to_bytes(src_path.as_path())?);
            digest.update(&path_to_bytes(dst_path.as_path())?);
        }
        for (tar_path, dir_path) in self.dirs_set.iter() {
            digest.update(&path_to_bytes(tar_path.as_path())?);
            digest.update(&path_to_bytes(dir_path.as_path())?);
        }
        for (tar_path, src_path) in self.file_set.iter() {
            digest = digest.with_file(src_path).await?;
            digest.update(&path_to_bytes(tar_path.as_path())?);
            digest.update(&path_to_bytes(src_path.as_path())?);
        }

        Ok(digest.finish())
    }

    fn compress(&self, writer: &mut (dyn Write + Send)) -> Result<()> {
        let mut builder = tar::Builder::new(writer);

        builder.follow_symlinks(false);
        builder.mode(tar::HeaderMode::Deterministic);

        // Add directories before symlinks to ensure the dirs exist
        // before symlinks that may point at them are unpacked
        for (tar_path, dir_path) in self.dirs_set.iter() {
            // trace!("dir: '{}'", dir_path.display());
            builder.append_dir(tar_path, dir_path)?;
        }
        // Add symlinks before files to ensure the symlinks exist
        // before files are unpacked to paths that may traverse them
        for (tar_path, (_src_path, dst_path)) in self.symlinks.iter() {
            // trace!(
            //     "link: '{}' -> '{}'",
            //     _src_path.display(),
            //     dst_path.display()
            // );
            let mut header = tar::Header::new_gnu();
            header.set_size(0);
            header.set_mtime(0);
            header.set_entry_type(tar::EntryType::Symlink);
            // Leave `dst_path` as absolute, assuming the tar will
            // be used in a chroot-like environment.
            builder.append_link(&mut header, tar_path, dst_path)?;
        }
        for (tar_path, file_path) in self.file_set.iter() {
            // trace!("file: '{}'", file_path.display());
            builder.append_path_with_name(file_path, tar_path)?;
        }

        builder.finish().map_err(anyhow::Error::new)
    }

    fn write_tar_gz(&self, writer: &mut (dyn Write + Send)) -> Result<()> {
        use gzp::{
            deflate::Gzip,
            par::compress::{Compression as Comp, ParCompressBuilder},
        };

        std::thread::scope(|scope| {
            self.compress(
                &mut ParCompressBuilder::<Gzip>::new()
                    .compression_level(Comp::default())
                    .num_threads(crate::util::num_cpus())?
                    .from_borrowed_writer(writer, scope),
            )
        })
    }
}

/// Strip a leading slash, if any.
pub fn tar_safe_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut final_path = PathBuf::new();

    for component in path.as_ref().components() {
        match component {
            Component::RootDir | Component::Prefix(_) => continue,
            c @ Component::Normal(_) | c @ Component::CurDir | c @ Component::ParentDir => {
                final_path.push(c);
            }
        }
    }

    final_path
}

pub fn make_tar_header<P: AsRef<Path>>(src: P, dst: P) -> io::Result<(tar::Header, PathBuf)> {
    let src = src.as_ref();
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

    Ok((file_header, tar_safe_path(dst)))
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
#[cfg(not(windows))]
fn find_ldd_libraries(
    env_vars: &[(OsString, OsString)],
    executable: &Path,
) -> Result<Vec<PathBuf>> {
    use std::io::Read;

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
        trace!("ldd stderr non-empty: {:?}", bytes_to_str(stderr)?);
    }

    bytes_to_str(stdout)
        .context("ldd output not utf8")
        .map(|stdout| parse_ldd_output(&stdout))
}

// If it's a static PIE the output will be a line like "\tstatically linked", so be forgiving
// in the parsing here and treat parsing oddities as an empty list.
#[cfg(not(windows))]
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

#[cfg(windows)]
fn find_ldd_libraries(
    env_vars: &[(OsString, OsString)],
    executable: &Path,
) -> Result<Vec<PathBuf>> {
    use crate::util::OsStrExt;

    let mut paths = vec![];

    let env_path = env_vars
        .iter()
        .find(|(k, _)| k == "PATH")
        .map(|(_, v)| v.clone())
        .or_else(|| std::env::var_os("PATH"));

    let dumpbin = if let Some(dir) = executable.parent() {
        paths.push(dir.to_path_buf());
        which::which_in("dumpbin", env_path.as_ref(), dir)
    } else {
        which::which_in("dumpbin", env_path.as_ref(), ".")
    }
    .unwrap_or_else(|_| PathBuf::from("dumpbin"));

    let process::Output {
        status,
        stdout,
        stderr,
    } = process::Command::new(dumpbin)
        .envs(env_vars.to_vec())
        .arg("/dependents")
        .arg(executable)
        .output()?;

    if !stderr.is_empty() {
        trace!("dumpbin stderr: {:?}", bytes_to_str(stderr)?);
    }

    if !status.success() {
        return Ok(vec![]);
    }

    paths.extend(
        env_path
            .map(|ps| {
                std::env::split_paths(&ps)
                    .filter(|p| p.is_dir())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
    );

    // Skip OS system directories, do not package core Windows files
    paths.retain(|p| {
        let p = p.as_os_str();
        !(
            p.contains(r"\Windows")
                || p.contains(r"\windows")
                || p.contains(r"\WINDOWS")
                || p.contains(r"\System32")
                || p.contains(r"\system32")
                || p.contains(r"\SYSTEM32")
            //
        )
    });

    bytes_to_str(stdout)
        .context("invalid dumpbin output")
        .map(|stdout| parse_ldd_output(&stdout))
        .map(|libs| {
            libs.iter()
                .filter(|lib| {
                    // # Skip virtual API sets entirely
                    !(lib.starts_with("api-ms-win-") || lib.starts_with("ext-ms-win-"))
                })
                .filter_map(|lib| {
                    for dir in paths.iter() {
                        let lib = dir.join(lib);
                        if lib.exists() {
                            return Some(lib);
                        }
                    }
                    None
                })
                .collect::<Vec<_>>()
        })
}

#[cfg(windows)]
fn parse_ldd_output(stdout: &str) -> Vec<PathBuf> {
    stdout
        .lines()
        .map(|line| line.trim())
        .filter(|line| line.ends_with(std::env::consts::DLL_SUFFIX))
        .map(PathBuf::from)
        .collect::<Vec<_>>()
}

#[cfg(not(all(
    feature = "dist-client",
    any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(
            target_os = "windows",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "freebsd"
    )
)))]
mod toolchain_imp {
    use super::{ArchivePackager, ToolchainPackager, dist};
    use crate::errors::*;
    use async_trait::async_trait;
    use std::sync::Arc;

    // Distributed client, but an unsupported platform for toolchain packaging so
    // create a failing implementation that will conflict with any others.
    #[async_trait]
    impl<T: Send + Sync> ToolchainPackager for T {
        async fn package(
            self: Box<Self>,
            _: &mut dist::PathTransformer,
        ) -> Result<Arc<dyn ArchivePackager>> {
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
        all(
            target_os = "windows",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "freebsd"
    )
))]
mod toolchain_imp {
    #[allow(unused)]
    use super::*;

    #[test]
    #[cfg(not(windows))]
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
    #[cfg(not(windows))]
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
    #[cfg(not(windows))]
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
