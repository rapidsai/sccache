//! Compiling Rust with cargo and rustc: cache hits, read-only caches, `env!`
//! dependencies, the hit rate in the stats, and toolchains behind symlinks.

use crate::harness::{TestDir, client::SccacheClient, init_logger, tests_dir};
use anyhow::Result;
use assert_cmd::{assert::AssertResult, prelude::*};
use fs_err as fs;
use predicates::prelude::*;
#[cfg(unix)]
use std::path::{Path, PathBuf};
use std::{
    ffi::{OsStr, OsString},
    process::Command,
};

/// A server for building tests/test-crate, with an empty config file and a
/// disk cache in a new temporary directory.
struct CargoTest {
    client: SccacheClient,
    tempdir: TestDir,
    rustflags: Option<OsString>,
}

#[allow(clippy::result_large_err)]
impl CargoTest {
    fn new() -> Self {
        init_logger();

        let tempdir = TestDir::new("sccache_test_rust_cargo");
        let cache_dir = tempdir.path().join("cache");
        fs::create_dir(&cache_dir).unwrap();
        fs::create_dir(tempdir.path().join("cargo")).unwrap();
        let config_file = tempdir.path().join("config");
        fs::File::create(&config_file).unwrap();

        let client = SccacheClient::new_no_cfg()
            .with_env("SCCACHE_CONF", &config_file)
            .with_env("SCCACHE_DIR", &cache_dir)
            .start();

        Self {
            client,
            tempdir,
            rustflags: None,
        }
    }

    #[cfg(feature = "unstable")]
    fn with_rustflags(rustflags: &str) -> Self {
        Self {
            rustflags: Some(rustflags.into()),
            ..Self::new()
        }
    }

    /// `cargo` in tests/test-crate, compiling through this server.
    fn cargo(&self) -> Command {
        let mut cmd = self.client.cargo();
        cmd.current_dir(tests_dir().join("test-crate"))
            .env("CARGO_TARGET_DIR", self.tempdir.path().join("cargo"))
            // Explicitly disable incremental compilation because sccache is unable to cache it at
            // the time of writing.
            .env("CARGO_INCREMENTAL", "0")
            .env("TEST_ENV_VAR", "1");
        if let Some(rustflags) = &self.rustflags {
            cmd.env("RUSTFLAGS", rustflags);
        }
        cmd
    }

    fn clean(&self) -> Result<()> {
        self.cargo().arg("clean").assert().try_success()?;
        Ok(())
    }

    /// The server's stats are reset when it restarts.
    fn restart_with_env(&mut self, vars: &[(&str, &OsStr)]) {
        self.client.restart_with_env(vars);
    }

    fn show_stats(&self) -> AssertResult {
        trace!("sccache --show-stats");

        self.client
            .cmd()
            .args(["--show-stats", "--stats-format=json"])
            .assert()
            .try_success()
    }

    fn show_text_stats(&self, advanced: bool) -> AssertResult {
        let cmd = if advanced {
            "--show-adv-stats"
        } else {
            "--show-stats"
        };

        trace!("sccache {cmd}");

        self.client
            .cmd()
            .args([cmd, "--stats-format=text"])
            .assert()
            .try_success()
    }
}

#[test]
fn test_rust_cargo_check() -> Result<()> {
    test_rust_cargo_cmd("check", CargoTest::new())
}

#[test]
fn test_rust_cargo_check_readonly() -> Result<()> {
    test_rust_cargo_cmd_readonly("check", CargoTest::new())
}

#[test]
fn test_rust_cargo_build() -> Result<()> {
    test_rust_cargo_cmd("build", CargoTest::new())
}

#[test]
fn test_rust_cargo_build_readonly() -> Result<()> {
    test_rust_cargo_cmd_readonly("build", CargoTest::new())
}

/// This test checks that changing an environment variable reference by env! is detected by
/// sccache, causes a rebuild and is correctly printed to stdout.
#[test]
fn test_rust_cargo_run_with_env_dep_parsing() -> Result<()> {
    test_rust_cargo_env_dep(CargoTest::new())
}

#[cfg(feature = "unstable")]
#[test]
fn test_rust_cargo_check_nightly() -> Result<()> {
    test_rust_cargo_cmd("check", CargoTest::with_rustflags("-Cprofile-generate=."))
}

#[cfg(feature = "unstable")]
#[test]
fn test_rust_cargo_check_nightly_readonly() -> Result<()> {
    test_rust_cargo_cmd_readonly("check", CargoTest::with_rustflags("-Cprofile-generate=."))
}

#[cfg(feature = "unstable")]
#[test]
fn test_rust_cargo_build_nightly() -> Result<()> {
    test_rust_cargo_cmd("build", CargoTest::with_rustflags("-Cprofile-generate=."))
}

#[cfg(feature = "unstable")]
#[test]
fn test_rust_cargo_build_nightly_readonly() -> Result<()> {
    test_rust_cargo_cmd_readonly("build", CargoTest::with_rustflags("-Cprofile-generate=."))
}

/// Test that building a simple Rust crate with cargo using sccache results in a cache hit
/// when built a second time and a cache miss, when the environment variable referenced via
/// env! is changed.
fn test_rust_cargo_cmd(cmd: &str, test_info: CargoTest) -> Result<()> {
    // `cargo clean` first, just to be sure there's no leftover build objects.
    test_info.clean()?;

    // Now build the crate with cargo.
    test_info
        .cargo()
        .args([cmd, "--color=never"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8().not())?
        .try_success()?;
    // Clean it so we can build it again.
    test_info.clean()?;
    test_info
        .cargo()
        .args([cmd, "--color=always"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8())?
        .try_success()?;

    test_info
        .show_stats()?
        .try_stdout(
            predicates::str::contains(
                r#""cache_hits":{"counts":{"Rust":2},"adv_counts":{"rust":2}}"#,
            )
            .from_utf8(),
        )?
        .try_success()?;

    Ok(())
}

/// Test that building a simple Rust crate with cargo using sccache results in the following behaviors (for three different runs):
/// - In read-only mode, a cache miss.
/// - In read-write mode, a cache miss.
/// - In read-only mode, a cache hit.
///
/// The environment variable for read/write mode is added by this function.
fn test_rust_cargo_cmd_readonly(cmd: &str, mut test_info: CargoTest) -> Result<()> {
    // `cargo clean` first, just to be sure there's no leftover build objects.
    test_info.clean()?;

    // The cache must be put into read-only mode, and that can only be configured
    // when the server starts up, so we need to restart it.
    test_info.restart_with_env(&[("SCCACHE_LOCAL_RW_MODE", "READ_ONLY".as_ref())]);

    // Now build the crate with cargo.
    test_info
        .cargo()
        .args([cmd, "--color=never"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8().not())?
        .try_success()?;

    // Stats reset on server restart, so this needs to be run for each build.
    test_info
        .show_stats()?
        .try_stdout(
            predicates::str::contains(r#""cache_hits":{"counts":{},"adv_counts":{}}"#).from_utf8(),
        )?
        .try_stdout(
            predicates::str::contains(
                r#""cache_misses":{"counts":{"Rust":2},"adv_counts":{"rust":2}}"#,
            )
            .from_utf8(),
        )?
        .try_success()?;

    test_info.clean()?;
    test_info.restart_with_env(&[("SCCACHE_LOCAL_RW_MODE", "READ_WRITE".as_ref())]);
    test_info
        .cargo()
        .args([cmd, "--color=always"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8())?
        .try_success()?;

    test_info
        .show_stats()?
        .try_stdout(
            predicates::str::contains(r#""cache_hits":{"counts":{},"adv_counts":{}}"#).from_utf8(),
        )?
        .try_stdout(
            predicates::str::contains(
                r#""cache_misses":{"counts":{"Rust":2},"adv_counts":{"rust":2}}"#,
            )
            .from_utf8(),
        )?
        .try_success()?;

    test_info.clean()?;
    test_info.restart_with_env(&[("SCCACHE_LOCAL_RW_MODE", "READ_ONLY".as_ref())]);
    test_info
        .cargo()
        .args([cmd, "--color=always"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8())?
        .try_success()?;

    test_info
        .show_stats()?
        .try_stdout(
            predicates::str::contains(
                r#""cache_hits":{"counts":{"Rust":2},"adv_counts":{"rust":2}}"#,
            )
            .from_utf8(),
        )?
        .try_stdout(
            predicates::str::contains(r#""cache_misses":{"counts":{},"adv_counts":{}}"#)
                .from_utf8(),
        )?
        .try_success()?;

    Ok(())
}

fn test_rust_cargo_env_dep(test_info: CargoTest) -> Result<()> {
    test_info.clean()?;
    // Now build the crate with cargo.
    test_info
        .cargo()
        .args(["run", "--color=never"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8().not())?
        .try_stdout(predicates::str::contains("Env var: 1"))?
        .try_success()?;
    // Clean it so we can build it again.
    test_info.clean()?;

    test_info
        .cargo()
        .args(["run", "--color=always"])
        .env("TEST_ENV_VAR", "OTHER_VALUE")
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8())?
        .try_stdout(predicates::str::contains("Env var: OTHER_VALUE"))?
        .try_success()?;

    // Now get the stats and ensure that we had one cache hit for the second build.
    // The test crate has one dependency (itoa) so there are two separate compilations, but only
    // itoa should be cached (due to the changed environment variable).
    test_info
        .show_stats()?
        .try_stdout(predicates::str::contains(r#""cache_hits":{"counts":{"Rust":1}"#).from_utf8())?
        .try_success()?;

    drop(test_info);
    Ok(())
}

/// Test that building a simple Rust crate with cargo using sccache in read-only mode with an empty cache results in
/// a cache miss that is produced by the readonly storage wrapper (and does not attempt to write to the underlying cache).
#[test]
fn test_rust_cargo_cmd_readonly_preemtive_block() -> Result<()> {
    let mut test_info = CargoTest::new();
    // `cargo clean` first, just to be sure there's no leftover build objects.
    test_info.clean()?;

    let sccache_log = test_info.tempdir.path().join("sccache.log");

    test_info.restart_with_env(&[
        ("SCCACHE_LOCAL_RW_MODE", "READ_ONLY".as_ref()),
        ("SCCACHE_SERVER_LOG", "sccache=debug".as_ref()),
        ("SCCACHE_ERROR_LOG", sccache_log.as_os_str()),
    ]);

    // Now build the crate with cargo.
    // Assert that our cache miss is due to the readonly storage wrapper, not due to the underlying disk cache.
    test_info
        .cargo()
        .args(["build", "--color=never"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8().not())?
        .try_success()?;

    let log_contents = fs::read_to_string(sccache_log)?;
    assert!(predicates::str::contains("Configured object storage").eval(log_contents.as_str()));
    assert!(
        predicates::str::contains("Error executing cache write: Cannot write to read-only storage")
            .eval(log_contents.as_str())
    );
    assert!(
        predicates::str::contains("DiskCache::finish_put")
            .not()
            .eval(log_contents.as_str())
    );

    // Stats reset on server restart, so this needs to be run for each build.
    test_info
        .show_stats()?
        .try_stdout(
            predicates::str::contains(r#""cache_hits":{"counts":{},"adv_counts":{}}"#).from_utf8(),
        )?
        .try_stdout(
            predicates::str::contains(
                r#""cache_misses":{"counts":{"Rust":2},"adv_counts":{"rust":2}}"#,
            )
            .from_utf8(),
        )?
        .try_success()?;
    Ok(())
}

#[test]
fn test_cache_hit_rate() -> Result<()> {
    let test_info = CargoTest::new();

    test_info
        .cargo()
        .args(["build", "--color=never"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8().not())?
        .try_success()?;

    test_info
        .show_text_stats(false)?
        .try_stdout(
            predicates::str::is_match(r"Cache hits rate\s+0\.00\s%")
                .unwrap()
                .from_utf8(),
        )?
        .try_stdout(
            predicates::str::is_match(r"Cache hits rate \(Rust\)\s+0\.00\s%")
                .unwrap()
                .from_utf8(),
        )?
        .try_success()?;

    // Clean it so we can build it again.
    test_info.clean()?;

    test_info
        .cargo()
        .args(["run", "--color=always"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8())?
        .try_success()?;

    test_info
        .show_text_stats(false)?
        .try_stdout(
            predicates::str::is_match(r"Cache hits rate\s+50\.00\s%")
                .unwrap()
                .from_utf8(),
        )?
        .try_stdout(
            predicates::str::is_match(r"Cache hits rate \(Rust\)\s+50\.00\s%")
                .unwrap()
                .from_utf8(),
        )?
        .try_success()?;

    Ok(())
}

#[test]
fn test_adv_cache_hit_rate() -> Result<()> {
    let test_info = CargoTest::new();

    test_info
        .cargo()
        .args(["build", "--color=never"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8().not())?
        .try_success()?;

    test_info
        .show_text_stats(true)?
        .try_stdout(
            predicates::str::is_match(r"Cache hits rate\s+0\.00\s%")
                .unwrap()
                .from_utf8(),
        )?
        .try_stdout(
            predicates::str::is_match(r"Cache hits rate \(rust\)\s+0\.00\s%")
                .unwrap()
                .from_utf8(),
        )?
        .try_success()?;

    test_info.clean()?;

    test_info
        .cargo()
        .args(["run", "--color=always"])
        .assert()
        .try_stderr(predicates::str::contains("\x1b[").from_utf8())?
        .try_success()?;

    test_info
        .show_text_stats(true)?
        .try_stdout(
            predicates::str::is_match(r"Cache hits rate\s+50\.00\s%")
                .unwrap()
                .from_utf8(),
        )?
        .try_stdout(
            predicates::str::is_match(r"Cache hits rate \(rust\)\s+50\.00\s%")
                .unwrap()
                .from_utf8(),
        )?
        .try_success()?;

    Ok(())
}

// (temp dir)
// ├── rust // symlinks to rust1 on the first run and rust2 on the second
// ├── rust1/
// │  ├── bin
// │  │  └── rustc
// │  ├── lib
// │  │  └── driver.so -> ../driver.so
// │  └── driver.so
// ├── rust2/
// │  ├── bin
// │  │  └── rustc
// │  ├── lib
// │  │  └── driver.so -> ../driver.so
// │  └── driver.so
// ├── sccache/
// ├── counter // increases by 1 for every compilation that is not cached
// ├── RUST_FILE // compile output copied from counter, same content means it was cached
// └── RUST_FILE.rs
#[test]
#[cfg(unix)]
fn test_symlinks() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    // The first compile starts the server, and dropping the client stops it.
    let client = SccacheClient::new_no_cfg().with_env("SCCACHE_DIR", root.join("sccache"));

    fs::write(root.join("counter"), b"0").unwrap();
    fs::write(root.join("RUST_FILE.rs"), []).unwrap();

    create_mock_rustc(root.join("rust1"));
    create_mock_rustc(root.join("rust2"));

    let rust = root.join("rust");
    let bin = rust.join("bin");
    let out_file = root.join("RUST_FILE");

    symlink(root.join("rust1"), &rust).unwrap();
    run_sccache(&client, root, &bin);
    let output1 = fs::read(&out_file).unwrap();

    fs::remove_file(&rust).unwrap();
    symlink(root.join("rust2"), &rust).unwrap();
    run_sccache(&client, root, &bin);
    let output2 = fs::read(out_file).unwrap();

    assert_ne!(output1, output2);
}

#[cfg(unix)]
fn create_mock_rustc(dir: PathBuf) {
    use std::{
        env::consts::DLL_SUFFIX,
        io::Write,
        os::unix::{
            fs::symlink,
            prelude::{OsStrExt, PermissionsExt},
        },
    };

    let bin = dir.join("bin");
    fs::create_dir_all(&bin).unwrap();

    let dll_name = format!("driver{DLL_SUFFIX}");
    let dll = dir.join(&dll_name);
    fs::write(&dll, dir.as_os_str().as_bytes()).unwrap();

    let lib = dir.join("lib");
    fs::create_dir(&lib).unwrap();
    symlink(dll, lib.join(&dll_name)).unwrap();

    let rustc = bin.join("rustc");
    write!(
        fs::File::create(&rustc).unwrap(),
        r#"#!/usr/bin/env sh

set -e
build=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        -vV)
            echo rustc 1.0.0
            exec echo "host: unknown"
            ;;
        +stable)
            exit 1
            ;;
        --print=sysroot)
            exec echo {}
            ;;
        --print)
            shift
            if [ "$1" = file-names ]; then
                exec echo RUST_FILE.rs
            fi
            ;;
        --emit)
            shift
            if [ "$1" = dep-info ]; then
                echo "deps.d: RUST_FILE.rs" > "$3"
                exec echo "RUST_FILE.rs:" "$3"
            fi
            ;;
        RUST_FILE.rs)
            build=1
            ;;
    esac
    shift
done

if [ "$build" -eq 1 ]; then
    echo $(($(cat counter) + 1)) > counter
    cp counter RUST_FILE
fi
"#,
        dir.display(),
    )
    .unwrap();

    let mut perm = rustc.metadata().unwrap().permissions();
    perm.set_mode(0o755);
    fs::set_permissions(&rustc, perm).unwrap();
}

#[cfg(unix)]
fn run_sccache(client: &SccacheClient, root: &Path, path: &Path) {
    let mut paths: OsString = path.into();
    paths.push(":");
    paths.push(std::env::var_os("PATH").unwrap());

    client
        .cmd()
        .current_dir(root)
        .env("PATH", paths)
        .arg("rustc")
        .arg("RUST_FILE.rs")
        .arg("--crate-name=sccache_rustc_tests")
        .arg("--crate-type=lib")
        .arg("--emit=link")
        .arg("--out-dir")
        .arg(root)
        .assert()
        .success();
}
