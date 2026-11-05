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

//! Shared infrastructure for the integration tests: temporary directories,
//! sccache servers (local or distributed), compilers, and server stats.

// Not every helper is used on every platform or with every feature set.
#![allow(dead_code)]

macro_rules! vec_from {
    ( $t:ty, $( $x:expr ),* ) => {
        vec![$( Into::<$t>::into(&$x), )*]
    };
}

pub mod client;
pub mod compilers;
#[cfg(feature = "dist-tests")]
pub mod dist;
pub mod stats;

use fs_err as fs;
use sccache::{errors::*, mock_command::ProcessOutput};
use serde::Serialize;
use std::{
    env,
    ffi::OsString,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, SystemTime},
};

pub const TC_CACHE_SIZE: u64 = 2 * 1024 * 1024 * 1024; // 2GiB

pub fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

pub fn sccache_bin() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_sccache"))
}

/// `tests/`, which holds the test modules and the fixtures they compile.
pub fn tests_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

pub fn sccache_debug() -> bool {
    env::var_os("SCCACHE_DEBUG").is_some()
}

/// A temporary directory that is left on disk when `SCCACHE_DEBUG` is set.
pub struct TestDir {
    path: PathBuf,
    _dir: Option<tempfile::TempDir>,
}

impl TestDir {
    pub fn new(prefix: &str) -> Self {
        tempfile::Builder::new()
            .prefix(prefix)
            .tempdir()
            .unwrap()
            .into()
    }

    pub fn new_in(prefix: &str, parent: &Path) -> Self {
        tempfile::Builder::new()
            .prefix(prefix)
            .tempdir_in(parent)
            .unwrap()
            .into()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl From<tempfile::TempDir> for TestDir {
    fn from(dir: tempfile::TempDir) -> Self {
        if sccache_debug() {
            let path = dir.keep();
            println!("SCCACHE_DEBUG is set, keeping '{}'", path.display());
            Self { path, _dir: None }
        } else {
            Self {
                path: dir.path().to_owned(),
                _dir: Some(dir),
            }
        }
    }
}

pub fn check_output<O: Into<ProcessOutput>>(output: O) -> Result<()> {
    let output = output.into();
    if !output.success() {
        eprintln!("{}", output.desc());
        if !output.stdout.is_empty() {
            eprintln!(
                "\n\n[BEGIN STDOUT]\n===========\n{}\n===========\n[FIN STDOUT]",
                String::from_utf8_lossy(&output.stdout)
            );
        }
        if !output.stderr.is_empty() {
            eprintln!(
                "\n\n[BEGIN STDERR]\n===========\n{}\n===========\n[FIN STDERR]",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Err(anyhow!("{}", output.desc()))
    } else {
        Ok(())
    }
}

// Prune any environment variables that could adversely affect test execution.
pub fn prune_command(mut cmd: Command) -> Command {
    use sccache::util::OsStrExt;

    for (var, _) in env::vars_os() {
        if var.starts_with("SCCACHE_") {
            cmd.env_remove(var);
        }
    }
    cmd
}

/// `sccache` without any `SCCACHE_*` variables from the test's environment.
/// Tests that start a server should use a `client::SccacheClient`, so that
/// each server listens on its own port.
pub fn sccache_command() -> Command {
    prune_command(Command::new(sccache_bin()))
}

pub fn cargo_bin() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| "cargo".into())
}

pub fn write_json_cfg<T: Serialize>(path: &Path, filename: &str, contents: &T) {
    let p = path.join(filename);
    let mut f = fs::File::create(p).unwrap();
    f.write_all(&serde_json::to_vec(contents).unwrap()).unwrap();
}

pub fn write_source(path: &Path, filename: &str, contents: &str) {
    let p = path.join(filename);
    let mut f = fs::File::create(p).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
}

// Copy the source files into the tempdir so we can compile with relative paths, since the commandline winds up in the hash key.
pub fn copy_to_tempdir(inputs: &[&str], tempdir: &Path) {
    for f in inputs {
        let original_source_file = tests_dir().join(f);
        let source_file = tempdir.join(f);
        trace!("fs::copy({original_source_file:?}, {source_file:?})");
        fs::copy(&original_source_file, &source_file).unwrap();
        // Preprocessor cache will not cache files that are too recent.
        // Certain OS/FS combinations have a slow resolution (up to 2s for NFS),
        // leading to flaky tests.
        // We set the times for the new file to 10 seconds ago, to be safe.
        let new_time =
            filetime::FileTime::from_system_time(SystemTime::now() - Duration::from_secs(10));
        filetime::set_file_times(source_file, new_time, new_time).unwrap();
    }
}
