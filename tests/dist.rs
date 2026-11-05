//! Distributed compilation, with a scheduler and build servers in docker
//! containers, and rabbitmq or redis as the message broker. The deps
//! cache tests also run here, with a dist client.
//!
//! In case of panics, this command will destroy any dangling containers:
//!   docker rm -f $(docker ps -aq --filter "name=sccache_dist_*")

// TODO:
// * Test each builder (docker and overlay for Linux, pot for freebsd)

use crate::harness::{
    check_output,
    client::SccacheClient,
    compilers::{Compiler, find_compiler},
    dist::DistSystem,
    init_logger, write_source,
};
#[cfg(target_os = "linux")]
use crate::{deps, harness::compilers::find_cuda_compiler};
#[cfg(target_os = "linux")]
use pastey::paste;
use sccache::errors::*;
#[cfg(target_os = "linux")]
use serial_test::serial;
use std::{ffi::OsString, path::Path, process::Output, sync::Arc};
use test_case::test_case;

/// A client that fails rather than compiling locally if distributed
/// compilation fails.
fn dist_client(system: &DistSystem, cache_size: u64) -> Result<Arc<SccacheClient>> {
    let mut cfg = system.client_cfg(cache_size)?;
    cfg.dist.fallback_to_local_compile = false;
    Ok(system.new_client(&cfg))
}

/// Runs `test` with a dist client for a new scheduler and build server, if
/// `compiler` is installed.
#[cfg(target_os = "linux")]
async fn with_dist_client(
    test_name: &str,
    message_broker: &str,
    compiler: &str,
    test: impl FnOnce(&SccacheClient, &Compiler, &Path) -> Result<()>,
) -> Result<()> {
    init_logger();
    let Some(compiler) = find_compiler(compiler) else {
        return Ok(());
    };
    let system = DistSystem::builder()
        .with_name(&format!("{test_name}_{message_broker}"))
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        .build()
        .await?;
    let client = dist_client(
        &system,
        sccache::config::defaults::default_disk_cache_size(),
    )?;
    // The tests block, and may start their own runtime (Compiler::version).
    tokio::task::block_in_place(|| test(&client, &compiler, system.test_dir()))
}

/// The stats that show where a client's compilations ran.
#[derive(Debug, PartialEq)]
struct DistStats {
    dist_compiles: usize,
    dist_errors: u64,
    compile_requests: u64,
    requests_executed: u64,
    compilations: u64,
    cache_hits: u64,
}

impl DistStats {
    fn of(client: &SccacheClient) -> Result<Self> {
        let stats = client.stats()?;
        println!("{stats:?}");
        Ok(Self {
            dist_compiles: stats.dist_compiles.values().sum(),
            dist_errors: stats.dist_errors,
            compile_requests: stats.compile_requests,
            requests_executed: stats.requests_executed,
            compilations: stats.compilations,
            cache_hits: stats.cache_hits.all(),
        })
    }

    /// `n` compilations that all ran on a build server.
    fn distributed(n: u64) -> Self {
        Self {
            dist_compiles: n as usize,
            dist_errors: 0,
            compile_requests: n,
            requests_executed: n,
            compilations: n,
            cache_hits: 0,
        }
    }

    /// `n` compilations that failed on the build servers and ran locally.
    fn fell_back(n: u64) -> Self {
        Self {
            dist_compiles: 0,
            dist_errors: n,
            ..Self::distributed(n)
        }
    }
}

/// Compiles `source_file` in `tmpdir`, writing a depfile alongside the object.
async fn compile(
    client: &SccacheClient,
    tmpdir: &Path,
    compiler_args: impl IntoIterator<Item = OsString>,
    source_file: &str,
    obj_file: &str,
    dep_file: &str,
) -> Result<()> {
    tokio::process::Command::from(client.cmd())
        .args(compiler_args)
        .arg(tmpdir.join(source_file))
        .arg("-o")
        .arg(tmpdir.join(obj_file))
        .arg("-MD")
        .arg("-MT")
        .arg(tmpdir.join(obj_file))
        .arg("-MF")
        .arg(tmpdir.join(dep_file))
        .env("RUST_BACKTRACE", "1")
        .kill_on_drop(true)
        .output()
        .await
        .map_err(Into::into)
        .and_then(check_output)
}

async fn cc_compile(client: &SccacheClient, tmpdir: &Path) -> Result<()> {
    let source_file = "x.c";
    write_source(
        tmpdir,
        source_file,
        "#if !defined(SCCACHE_TEST_DEFINE)\n#error SCCACHE_TEST_DEFINE is not defined\n#endif\nint x() { return 5; }",
    );
    compile(
        client,
        tmpdir,
        [
            std::env::var_os("CC").unwrap_or_else(|| "gcc".into()),
            "-c".into(),
            "-DSCCACHE_TEST_DEFINE".into(),
        ],
        source_file,
        "x.o",
        "x.o.d",
    )
    .await
}

#[cfg(target_os = "linux")]
async fn nvcc_compile(
    client: &SccacheClient,
    cuda_compiler: &Compiler,
    host_compiler: &Compiler,
    tmpdir: &Path,
) -> Result<()> {
    let source_file = "x.cu";
    write_source(
        tmpdir,
        source_file,
        "#if !defined(SCCACHE_TEST_DEFINE)\n#error SCCACHE_TEST_DEFINE is not defined\n#endif\n__global__ void x(int* out) { out[0] = 5; }",
    );
    compile(
        client,
        tmpdir,
        [
            cuda_compiler.exe.clone(),
            format!("-ccbin={}", host_compiler.exe.to_string_lossy()).into(),
            "-allow-unsupported-compiler".into(),
            "-c".into(),
            "-DSCCACHE_TEST_DEFINE".into(),
        ],
        source_file,
        "x.cu.o",
        "x.cu.o.d",
    )
    .await
}

#[cfg(target_os = "linux")]
async fn stdpar_compile(client: &SccacheClient, compiler: &Compiler, tmpdir: &Path) -> Result<()> {
    let source_file = "x.cpp";
    write_source(
        tmpdir,
        source_file,
        r#"
        #if !defined(SCCACHE_TEST_DEFINE)
        #error SCCACHE_TEST_DEFINE is not defined
        #endif

        #include <algorithm>
        #include <cassert>
        #include <execution>
        #include <numeric>
        #include <vector>

        constexpr int N = 1000;

        int main()
        {
          std::vector<int> v(N);
          std::fill(std::execution::par_unseq, v.begin(), v.end(), 42);

          int sum = std::reduce(std::execution::par_unseq, v.begin(), v.end(), 100, [](int a, int b) {
            return a + b;
          });
          assert(sum == (42 * N) + 100);

          sum = std::reduce(std::execution::par_unseq, v.begin(), v.end(), 100);
          assert(sum == (42 * N) + 100);
        }"#,
    );
    compile(
        client,
        tmpdir,
        [
            compiler.exe.clone(),
            "-c".into(),
            "-x".into(),
            "cpp".into(),
            "-DSCCACHE_TEST_DEFINE".into(),
            "-stdpar=gpu".into(),
        ],
        source_file,
        "x.cpp.o",
        "x.cpp.o.d",
    )
    .await
}

async fn rust_compile(client: &SccacheClient, tmpdir: &Path) -> Result<Output> {
    let cargo_path = tmpdir.join("sccache-dist-test");
    std::fs::create_dir_all(cargo_path.join("src"))?;

    write_source(
        &cargo_path,
        "Cargo.toml",
        r#"[package]
        name = "sccache-dist-test"
        version = "0.1.0"
        edition = "2021"
        [dependencies]
        # Pinned: the point of this test is that distributed compilation works,
        # not that it keeps up with whatever libc published most recently.
        libc = "=0.2.169""#,
    );
    write_source(
        &cargo_path,
        "src/main.rs",
        r#"fn main() {
        println!("Hello, world!");
}"#,
    );

    tokio::process::Command::from(client.cargo())
        .current_dir(cargo_path)
        .args(["build", "--release", "--jobs", "1"])
        .env("CARGO_TARGET_DIR", "target")
        .env("RUST_BACKTRACE", "1")
        .kill_on_drop(true)
        .output()
        .await
        .map_err(Into::into)
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cargo_build(message_broker: &str) -> Result<()> {
    let test_name = format!("test_dist_cargo_build_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        .build()
        .await?;

    // This test is flaky on github's small runners,
    // so run a few times before reporting failure
    for i in 1..=5 {
        let client = dist_client(&system, 0)?;

        let output = rust_compile(&client, system.test_dir()).await?;

        // Ensure sccache ignores inherited jobservers in CARGO_MAKEFLAGS
        assert!(
            !String::from_utf8_lossy(&output.stderr)
                .contains("warning: failed to connect to jobserver from environment variable")
        );

        // Assert compilation succeeded
        check_output(output)?;

        let stats = client.stats()?;
        println!("{stats:?}");

        if i == 5 {
            // Assert on the last iteration
            assert_eq!(1, stats.dist_compiles.values().sum::<usize>());
            assert_eq!(0, stats.dist_errors);
            // check >= 5 because cargo >=1.82 does additional requests with -vV
            assert!(stats.compile_requests >= 5);
            assert_eq!(1, stats.requests_executed);
            assert_eq!(1, stats.compilations);
            assert_eq!(0, stats.cache_hits.all());
        } else {
            // check >= 5 because cargo >=1.82 does additional requests with -vV
            if stats.compile_requests >= 5
                && stats.dist_compiles.values().sum::<usize>() == 1
                && stats.dist_errors == 0
                && stats.requests_executed == 1
                && stats.compilations == 1
                && stats.cache_hits.all() == 0
            {
                break;
            }
        }
    }

    Ok(())
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cpp_disk_storage(message_broker: &str) -> Result<()> {
    let test_name = format!("test_dist_cpp_disk_storage_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        .build()
        .await?;

    let client = dist_client(&system, 0)?;

    cc_compile(&client, system.test_dir()).await?;

    assert_eq!(DistStats::distributed(1), DistStats::of(&client)?);

    Ok(())
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cpp_toolchain(message_broker: &str) -> Result<()> {
    let test_name = format!("test_dist_cpp_toolchain_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        .build()
        .await?;

    // Run twice to verify toolchain is only uploaded once
    for i in 0..2 {
        let client = dist_client(&system, 0)?;

        let tc_dir = client.clear_toolchains_cache()?;

        cc_compile(&client, system.test_dir()).await?;

        assert_eq!(DistStats::distributed(1), DistStats::of(&client)?);

        assert_eq!(system.count_server_toolchains(system.server(0)?)?, 1);

        // Ensure the client doesn't package the toolchain again the 2nd time
        if i == 0 {
            assert_eq!(std::fs::read_dir(tc_dir).map(|dir| dir.count())?, 1);
        } else {
            assert_eq!(std::fs::read_dir(tc_dir).map(|dir| dir.count())?, 0);
        }
    }

    Ok(())
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cpp_cloud_storage(message_broker: &str) -> Result<()> {
    let test_name = format!("test_dist_cpp_cloud_storage_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        .with_redis_storage()
        .build()
        .await?;

    let client = dist_client(&system, 0)?;

    cc_compile(&client, system.test_dir()).await?;

    assert_eq!(DistStats::distributed(1), DistStats::of(&client)?);

    Ok(())
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cpp_server_restart(message_broker: &str) -> Result<()> {
    let test_name = format!("test_dist_cpp_server_restart_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        .build()
        .await?;

    let client = system.new_client(&system.client_cfg(0)?);

    cc_compile(&client, system.test_dir()).await?;

    system.restart_server(system.server(0)?).await?;

    cc_compile(&client, system.test_dir()).await?;

    assert_eq!(DistStats::distributed(2), DistStats::of(&client)?);

    Ok(())
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cpp_no_server_times_out(message_broker: &str) -> Result<()> {
    let test_name = format!("test_dist_cpp_no_server_times_out_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_message_broker(message_broker)
        .with_job_time_limit(10)
        .build()
        .await?;

    let client = system.new_client(&system.client_cfg(0)?);

    cc_compile(&client, system.test_dir()).await?;

    assert_eq!(DistStats::fell_back(1), DistStats::of(&client)?);

    Ok(())
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_dist_cpp_two_servers(message_broker: &str) -> Result<()> {
    let test_name = format!("test_dist_cpp_two_servers_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_server()
        .with_message_broker(message_broker)
        .build()
        .await?;

    let client = dist_client(&system, 0)?;

    let _ = tokio::try_join!(
        cc_compile(&client, system.test_dir()),
        cc_compile(&client, system.test_dir()),
        cc_compile(&client, system.test_dir()),
        cc_compile(&client, system.test_dir()),
    );

    assert_eq!(DistStats::distributed(4), DistStats::of(&client)?);

    Ok(())
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cpp_errors_on_job_load_failures(message_broker: &str) -> Result<()> {
    let test_name = format!("test_dist_cpp_errors_on_job_load_failures_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        // Scheduler stores jobs in redis, but Server is loading from disk
        .with_scheduler_jobs_redis_storage()
        .build()
        .await?;

    let client = system.new_client(&system.client_cfg(0)?);

    cc_compile(&client, system.test_dir()).await?;

    assert_eq!(DistStats::fell_back(1), DistStats::of(&client)?);

    Ok(())
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cpp_errors_on_toolchain_load_failures(message_broker: &str) -> Result<()> {
    let test_name = format!("test_dist_cpp_errors_on_toolchain_load_failures_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        // Scheduler stores toolchains in redis, but Server is loading from disk
        .with_scheduler_toolchains_redis_storage()
        .build()
        .await?;

    let client = system.new_client(&system.client_cfg(0)?);

    cc_compile(&client, system.test_dir()).await?;

    assert_eq!(DistStats::fell_back(1), DistStats::of(&client)?);

    Ok(())
}

#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cpp_preprocesspr_cache_bug_2173(message_broker: &str) -> Result<()> {
    // Bug 2173: dependencies cache hit but main cache miss - because using the dependencies cache
    // means not doing regular preprocessing, there was no preprocessed translation unit to send
    // out for distributed compilation, so an empty u8 array was compiled - which "worked", but
    // the object file *was* the result of compiling an empty file.

    let test_name = format!("test_dist_cpp_preprocesspr_cache_bug_2173_{message_broker}");
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        .build()
        .await?;

    let client = dist_client(
        &system, 10_000_000, // enough for one tiny object file
    )?;

    let _ = client.clear_disk_cache()?;

    cc_compile(&client, system.test_dir()).await?;

    assert_eq!(DistStats::distributed(1), DistStats::of(&client)?);

    let obj_file = "x.o";
    let obj_path = system.test_dir().join(obj_file);
    let data_a = std::fs::read(&obj_path)?;

    // Delete the object cache to ensure a cache miss
    client.clear_compilations_cache()?;

    cc_compile(&client, system.test_dir()).await?;

    assert_eq!(DistStats::distributed(2), DistStats::of(&client)?);

    // Check that this gave the same result (i.e. that it didn't compile a completely empty file).
    // It would be nice to check directly that the object file contains the symbol for the x() function
    // from cc_compile(), but that seems pretty involved and this happens to work...
    let data_b = std::fs::read(&obj_path)?;

    assert!(data_a == data_b, "object files don't match");

    Ok(())
}

#[cfg(target_os = "linux")]
async fn test_dist_cuda_compiles(
    cuda_compiler: &Compiler,
    host_compiler: &Compiler,
    message_broker: &str,
) -> Result<()> {
    let test_name = format!(
        "test_dist_cuda_compiles_{}_{}_{message_broker}",
        cuda_compiler.name, host_compiler.name
    );
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        .build()
        .await?;

    let client = dist_client(
        &system, 10_000_000, // enough for one tiny object file
    )?;

    let _ = client.clear_disk_cache()?;

    // nvcc distributes each of the compilations it runs for its steps
    let expected = DistStats {
        dist_compiles: 6,
        dist_errors: 0,
        compile_requests: 1,
        requests_executed: 7,
        compilations: 7,
        cache_hits: 0,
    };

    nvcc_compile(&client, cuda_compiler, host_compiler, system.test_dir()).await?;

    assert_eq!(expected, DistStats::of(&client)?);

    // Test we don't regress bug 2173 for CUDA compilations

    let obj_path = system.test_dir().join("x.cu.o");
    let dep_path = system.test_dir().join("x.cu.o.d");
    let dep_a = std::fs::read(&dep_path)?;
    let obj_a = std::fs::read(&obj_path)?;

    // Delete the object cache to ensure a cache miss
    client.clear_compilations_cache()?;
    client.zero_stats();

    nvcc_compile(&client, cuda_compiler, host_compiler, system.test_dir()).await?;

    assert_eq!(expected, DistStats::of(&client)?);

    let dep_b = std::fs::read(&dep_path)?;
    let obj_b = std::fs::read(&obj_path)?;

    if host_compiler.name == "nvc++" {
        // nvc++ produces different binary contents for the same inputs.
        // TODO: Figure out which optimizations to disable to fix this.
        assert!(obj_a.len() == obj_b.len(), "object files don't match");
    } else {
        assert!(obj_a == obj_b, "object files don't match");
    }

    assert_eq!(
        String::from_utf8_lossy(&dep_a),
        String::from_utf8_lossy(&dep_b),
        "dependency files don't match"
    );

    Ok(())
}

#[cfg(target_os = "linux")]
async fn test_dist_stdpar_compiles(compiler: &Compiler, message_broker: &str) -> Result<()> {
    let test_name = format!(
        "test_dist_stdpar_compiles_{}_{message_broker}",
        compiler.name
    );
    let system = DistSystem::builder()
        .with_name(&test_name)
        .with_scheduler()
        .with_server()
        .with_message_broker(message_broker)
        .build()
        .await?;

    let client = dist_client(&system, 0)?;

    stdpar_compile(&client, compiler, system.test_dir()).await?;

    assert_eq!(DistStats::distributed(1), DistStats::of(&client)?);

    Ok(())
}

#[cfg(target_os = "linux")]
macro_rules! test_dist_if_compiler_available {
    ($name:ident, $compiler:ident, $compiler_name:expr) => {
        paste! {
            #[test_case("rabbitmq" ; "with rabbitmq")]
            #[test_case("redis" ; "with redis")]
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn [<test_dist_ $name _compiles_ $compiler>] (message_broker: &str) -> Result<()> {
                init_logger();
                if let Some(compiler) = find_compiler($compiler_name) {
                    return [<test_dist_ $name _compiles>](
                        &compiler,
                        message_broker,
                    ).await
                }
                Ok(())
            }
        }
    };
}

#[cfg(target_os = "linux")]
test_dist_if_compiler_available!(stdpar, nvcxx, "nvc++");
#[cfg(target_os = "linux")]
test_dist_if_compiler_available!(stdpar, mpicxx, "mpic++");

#[cfg(target_os = "linux")]
macro_rules! test_dist_if_cuda_compiler_available {
    ($cuda_compiler:ident, $host_compiler:ident, $cuda_compiler_name:expr, $host_compiler_name:expr) => {
        paste! {
            #[test_case("rabbitmq" ; "with rabbitmq")]
            #[test_case("redis" ; "with redis")]
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn [<test_dist_cuda_compiles_ $cuda_compiler _ $host_compiler>] (message_broker: &str) -> Result<()> {
                init_logger();
                if let Some(cuda_compiler) = find_cuda_compiler($cuda_compiler_name)
                    && let Some(host_compiler) = find_compiler($host_compiler_name)
                {
                    return test_dist_cuda_compiles(
                        &cuda_compiler,
                        &host_compiler,
                        message_broker,
                    ).await
                }
                Ok(())
            }
        }
    };
}

#[cfg(target_os = "linux")]
test_dist_if_cuda_compiler_available!(nvcc, gcc, "nvcc", "gcc");
#[cfg(target_os = "linux")]
test_dist_if_cuda_compiler_available!(nvcc, clang, "nvcc", "clang++");
#[cfg(target_os = "linux")]
test_dist_if_cuda_compiler_available!(nvcc, nvcxx, "nvcc", "nvc++");

// Clang-CUDA cannot dist-compile
// #[cfg(target_os = "linux")]
// test_dist_if_cuda_compiler_available!(clang, clang, "clang++", "clang++");

#[cfg(target_os = "linux")]
macro_rules! test_dist_deps {
    ($compiler:ident, $compiler_name:expr) => {
        paste! {
            #[test_case("rabbitmq" ; "with rabbitmq")]
            #[test_case("redis" ; "with redis")]
            #[serial($compiler)]
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn [<test_dist_deps_dependency_changes_ $compiler>] (message_broker: &str) -> Result<()> {
                with_dist_client(
                    stringify!([<test_dist_deps_dependency_changes_ $compiler>]),
                    message_broker,
                    $compiler_name,
                    deps::test_dependency_changes,
                ).await
            }

            #[test_case("rabbitmq" ; "with rabbitmq")]
            #[test_case("redis" ; "with redis")]
            #[serial($compiler)]
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn [<test_dist_deps_macros_time_ $compiler>] (message_broker: &str) -> Result<()> {
                with_dist_client(
                    stringify!([<test_dist_deps_macros_time_ $compiler>]),
                    message_broker,
                    $compiler_name,
                    deps::test_time_macros,
                ).await
            }

            #[test_case("rabbitmq" ; "with rabbitmq")]
            #[test_case("redis" ; "with redis")]
            #[serial($compiler)]
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn [<test_dist_deps_macros_date_ $compiler>] (message_broker: &str) -> Result<()> {
                // NVHPC doesn't respect SOURCE_DATE_EPOCH
                if matches!($compiler_name, "nvc" | "nvc++") {
                    return Ok(());
                }
                with_dist_client(
                    stringify!([<test_dist_deps_macros_date_ $compiler>]),
                    message_broker,
                    $compiler_name,
                    deps::test_date_macros,
                ).await
            }

            #[test_case("rabbitmq" ; "with rabbitmq")]
            #[test_case("redis" ; "with redis")]
            #[serial($compiler)]
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn [<test_dist_deps_macros_timestamp_ $compiler>] (message_broker: &str) -> Result<()> {
                with_dist_client(
                    stringify!([<test_dist_deps_macros_timestamp_ $compiler>]),
                    message_broker,
                    $compiler_name,
                    deps::test_timestamp_macros,
                ).await
            }
        }
    };
}

#[cfg(target_os = "linux")]
test_dist_deps!(gcc, "gcc");
#[cfg(target_os = "linux")]
test_dist_deps!(clang, "clang");
#[cfg(target_os = "linux")]
test_dist_deps!(clangxx, "clang++");
#[cfg(target_os = "linux")]
test_dist_deps!(nvc, "nvc");
#[cfg(target_os = "linux")]
test_dist_deps!(nvcxx, "nvc++");
