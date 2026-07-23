#![cfg(all(feature = "dist-client", feature = "dist-server"))]

#[macro_use]
extern crate log;
extern crate sccache;
extern crate serde_json;

mod harness;

use harness::{
    Compiler, check_output,
    client::SccacheClient,
    dist::{DistSystem, cargo_command},
    init_cargo, write_source,
};
#[cfg(not(any(target_os = "macos", target_os = "freebsd")))]
use harness::{find_compilers, find_cuda_compilers};
#[cfg(not(any(target_os = "macos", target_os = "freebsd")))]
use pastey::paste;
use sccache::{config::HTTPUrl, errors::*};
use std::path::Path;
use std::process::Output;
use test_case::test_case;

// In case of panics, this command will destroy any dangling containers:
//   docker rm -f $(docker ps -aq --filter "name=sccache_dist_*")

// TODO:
// * Test each builder (docker and overlay for Linux, pot for freebsd)

async fn cc_compile(client: &SccacheClient, tmpdir: &Path) -> Result<()> {
    let source_file = "x.c";
    let obj_file = "x.o";
    write_source(
        tmpdir,
        source_file,
        "#if !defined(SCCACHE_TEST_DEFINE)\n#error SCCACHE_TEST_DEFINE is not defined\n#endif\nint x() { return 5; }",
    );
    tokio::process::Command::from(client.cmd())
        .args([
            std::env::var("CC")
                .unwrap_or_else(|_| "gcc".to_string())
                .as_str(),
            "-c",
            "-DSCCACHE_TEST_DEFINE",
        ])
        .arg(tmpdir.join(source_file))
        .arg("-o")
        .arg(tmpdir.join(obj_file))
        .env("RUST_BACKTRACE", "1")
        .env("SCCACHE_THREADS", "2")
        .kill_on_drop(true)
        .output()
        .await
        .map_err(Into::into)
        .and_then(check_output)
}

async fn nvcc_compile(
    client: &SccacheClient,
    cuda_compiler: &Compiler,
    host_compiler: &Compiler,
    tmpdir: &Path,
) -> Result<()> {
    let source_file = "x.cu";
    let dep_file = "x.cu.o.d";
    let obj_file = "x.cu.o";
    write_source(
        tmpdir,
        source_file,
        "#if !defined(SCCACHE_TEST_DEFINE)\n#error SCCACHE_TEST_DEFINE is not defined\n#endif\n__global__ void x(int* out) { out[0] = 5; }",
    );
    tokio::process::Command::from(client.cmd())
        .arg(&cuda_compiler.exe)
        .arg(format!("-ccbin={}", host_compiler.exe.to_string_lossy()))
        .arg("-allow-unsupported-compiler")
        .args(["-c", "-DSCCACHE_TEST_DEFINE"])
        .arg(tmpdir.join(source_file))
        .arg("-o")
        .arg(tmpdir.join(obj_file))
        .arg("-MD")
        .arg("-MT")
        .arg(tmpdir.join(obj_file))
        .arg("-MF")
        .arg(tmpdir.join(dep_file))
        .env("RUST_BACKTRACE", "1")
        .env("SCCACHE_THREADS", "2")
        .kill_on_drop(true)
        .output()
        .await
        .map_err(Into::into)
        .and_then(check_output)
}

async fn stdpar_compile(client: &SccacheClient, compiler: &Compiler, tmpdir: &Path) -> Result<()> {
    let source_file = "x.cpp";
    let obj_file = "x.cpp.o";
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
    tokio::process::Command::from(client.cmd())
        .arg(&compiler.exe)
        .args(["-c", "-x", "cpp", "-DSCCACHE_TEST_DEFINE", "-stdpar=gpu"])
        .arg(tmpdir.join(source_file))
        .arg("-o")
        .arg(tmpdir.join(obj_file))
        .env("RUST_BACKTRACE", "1")
        .env("SCCACHE_THREADS", "2")
        .kill_on_drop(true)
        .output()
        .await
        .map_err(Into::into)
        .and_then(check_output)
}

async fn rust_compile(client: &SccacheClient, tmpdir: &Path) -> Result<Output> {
    let cargo_name = "sccache-dist-test";
    let cargo_path = init_cargo(tmpdir, cargo_name);

    let manifest_file = "Cargo.toml";
    let source_file = "src/main.rs";

    write_source(
        &cargo_path,
        manifest_file,
        r#"[package]
        name = "sccache-dist-test"
        version = "0.1.0"
        edition = "2021"
        [dependencies]
        libc = "=0.2.186""#,
    );
    write_source(
        &cargo_path,
        source_file,
        r#"fn main() {
        println!("Hello, world!");
}"#,
    );

    tokio::process::Command::from(cargo_command())
        .current_dir(cargo_path)
        .args(["build", "--release", "--jobs", "1"])
        .envs(
            client
                .cmd()
                .get_envs()
                .map(|(k, v)| (k, v.unwrap_or_default())),
        )
        .env("RUSTC_WRAPPER", &client.path)
        .env("CARGO_TARGET_DIR", "target")
        .env("RUST_BACKTRACE", "1")
        .env("SCCACHE_THREADS", "2")
        .kill_on_drop(true)
        .output()
        .await
        .map_err(Into::into)
}

pub fn dist_test_sccache_client_cfg(
    tmpdir: &Path,
    scheduler_url: HTTPUrl,
    preprocessor_cache_mode: bool,
) -> sccache::config::FileConfig {
    let mut sccache_cfg = harness::client::sccache_client_cfg(tmpdir, preprocessor_cache_mode);
    sccache_cfg.cache.disk.as_mut().unwrap().size = 0;
    sccache_cfg.dist.scheduler_url = Some(scheduler_url);
    sccache_cfg
}

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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
        .with_redis_storage()
        .build()
        .await?;

    // This test is flaky on github's small runners,
    // so run a few times before reporting failure
    for i in 1..=5 {
        let client = system.new_client(&dist_test_sccache_client_cfg(
            system.data_dir(),
            system.scheduler(0)?.url(),
            false,
        ));

        let output = rust_compile(&client, system.test_dir()).await?;

        // Ensure sccache ignores inherited jobservers in CARGO_MAKEFLAGS
        assert!(
            !String::from_utf8_lossy(&output.stderr)
                .contains("warning: failed to connect to jobserver from environment variable")
        );

        // Assert compilation succeeded
        check_output(output)?;

        let stats = client.stats()?;

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

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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

    let client = system.new_client(&dist_test_sccache_client_cfg(
        system.data_dir(),
        system.scheduler(0)?.url(),
        false,
    ));

    cc_compile(&client, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(1, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(0, stats.dist_errors);
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(1, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());

    Ok(())
}

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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
        let client = system.new_client(&dist_test_sccache_client_cfg(
            system.data_dir(),
            system.scheduler(0)?.url(),
            false,
        ));

        let tc_dir = client.clear_toolchains_cache()?;

        cc_compile(&client, system.test_dir()).await?;

        let stats = client.stats()?;
        assert_eq!(1, stats.dist_compiles.values().sum::<usize>());
        assert_eq!(0, stats.dist_errors);
        assert_eq!(1, stats.compile_requests);
        assert_eq!(1, stats.requests_executed);
        assert_eq!(1, stats.compilations);
        assert_eq!(0, stats.cache_hits.all());

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

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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

    let client = system.new_client(&dist_test_sccache_client_cfg(
        system.data_dir(),
        system.scheduler(0)?.url(),
        false,
    ));

    cc_compile(&client, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(1, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(0, stats.dist_errors);
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(1, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());

    Ok(())
}

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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
        .with_redis_storage()
        .build()
        .await?;

    let client = system.new_client(&dist_test_sccache_client_cfg(
        system.data_dir(),
        system.scheduler(0)?.url(),
        false,
    ));

    cc_compile(&client, system.test_dir()).await?;

    system.restart_server(system.server(0)?).await?;

    cc_compile(&client, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(2, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(0, stats.dist_errors);
    assert_eq!(2, stats.compile_requests);
    assert_eq!(2, stats.requests_executed);
    assert_eq!(2, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());

    Ok(())
}

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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
        .with_redis_storage()
        .build()
        .await?;

    let client = system.new_client(&dist_test_sccache_client_cfg(
        system.data_dir(),
        system.scheduler(0)?.url(),
        false,
    ));

    cc_compile(&client, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(0, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(1, stats.dist_errors);
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(1, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());

    Ok(())
}

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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
        .with_redis_storage()
        .build()
        .await?;

    let client = system.new_client(&dist_test_sccache_client_cfg(
        system.data_dir(),
        system.scheduler(0)?.url(),
        false,
    ));

    let _ = tokio::try_join!(
        cc_compile(&client, system.test_dir()),
        cc_compile(&client, system.test_dir()),
        cc_compile(&client, system.test_dir()),
        cc_compile(&client, system.test_dir()),
    );

    let stats = client.stats()?;
    assert_eq!(4, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(0, stats.dist_errors);
    assert_eq!(4, stats.compile_requests);
    assert_eq!(4, stats.requests_executed);
    assert_eq!(4, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());

    Ok(())
}

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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

    let client = system.new_client(&dist_test_sccache_client_cfg(
        system.data_dir(),
        system.scheduler(0)?.url(),
        false,
    ));

    cc_compile(&client, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(0, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(1, stats.dist_errors);
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(1, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());

    Ok(())
}

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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

    let client = system.new_client(&dist_test_sccache_client_cfg(
        system.data_dir(),
        system.scheduler(0)?.url(),
        false,
    ));

    cc_compile(&client, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(0, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(1, stats.dist_errors);
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(1, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());

    Ok(())
}

#[cfg_attr(not(feature = "dist-tests"), ignore)]
#[test_case("rabbitmq" ; "with rabbitmq")]
#[test_case("redis" ; "with redis")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_dist_cpp_preprocesspr_cache_bug_2173(message_broker: &str) -> Result<()> {
    // Bug 2173: preprocessor cache hit but main cache miss - because using the preprocessor cache
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

    let client = system.new_client(&{
        let mut config =
            dist_test_sccache_client_cfg(system.data_dir(), system.scheduler(0)?.url(), true);
        config.cache.disk.as_mut().unwrap().size = 10_000_000; // enough for one tiny object file
        config
    });

    let (_, preprocessor_cache_path) = client.clear_disk_cache()?;

    cc_compile(&client, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(1, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(0, stats.dist_errors);
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(1, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());
    assert_eq!(0, stats.preprocessor_cache_hits.all());

    let obj_file = "x.o";
    let obj_path = system.test_dir().join(obj_file);
    let data_a = std::fs::read(&obj_path)?;

    // Don't touch the preprocessor cache - and check that it exists
    assert!(
        preprocessor_cache_path.is_dir(),
        "The preprocessor cache should exist"
    );

    // Delete the object cache to ensure a cache miss
    client.clear_object_cache()?;

    cc_compile(&client, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(2, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(0, stats.dist_errors);
    assert_eq!(2, stats.compile_requests);
    assert_eq!(2, stats.requests_executed);
    assert_eq!(2, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());
    assert_eq!(1, stats.preprocessor_cache_hits.all());

    // Check that this gave the same result (i.e. that it didn't compile a completely empty file).
    // It would be nice to check directly that the object file contains the symbol for the x() function
    // from cc_compile(), but that seems pretty involved and this happens to work...
    let data_b = std::fs::read(&obj_path)?;

    assert!(data_a == data_b, "object files don't match");

    Ok(())
}

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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

    let client = system.new_client(&{
        let mut config =
            dist_test_sccache_client_cfg(system.data_dir(), system.scheduler(0)?.url(), true);
        config.cache.disk.as_mut().unwrap().size = 10_000_000; // enough for one tiny object file
        config
    });

    let (_, preprocessor_cache_path) = client.clear_disk_cache()?;

    nvcc_compile(&client, cuda_compiler, host_compiler, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(4, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(0, stats.dist_errors);
    assert_eq!(1, stats.compile_requests);
    assert_eq!(5, stats.requests_executed);
    assert_eq!(5, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());
    assert_eq!(0, stats.preprocessor_cache_hits.all());

    // Test we don't regress bug 2173 for CUDA compilations

    let obj_path = system.test_dir().join("x.cu.o");
    let dep_path = system.test_dir().join("x.cu.o.d");
    let dep_a = std::fs::read(&dep_path)?;
    let obj_a = std::fs::read(&obj_path)?;

    // Don't touch the preprocessor cache - and check that it exists
    assert!(
        preprocessor_cache_path.is_dir(),
        "The preprocessor cache should exist"
    );

    // Delete the object cache to ensure a cache miss
    client.clear_object_cache()?;
    client.zero_stats();

    nvcc_compile(&client, cuda_compiler, host_compiler, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(4, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(0, stats.dist_errors);
    assert_eq!(1, stats.compile_requests);
    assert_eq!(5, stats.requests_executed);
    assert_eq!(5, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());
    assert_eq!(1, stats.preprocessor_cache_hits.all());

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

#[cfg_attr(not(feature = "dist-tests"), ignore)]
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

    let client = system.new_client(&dist_test_sccache_client_cfg(
        system.data_dir(),
        system.scheduler(0)?.url(),
        false,
    ));

    stdpar_compile(&client, compiler, system.test_dir()).await?;

    let stats = client.stats()?;
    assert_eq!(1, stats.dist_compiles.values().sum::<usize>());
    assert_eq!(0, stats.dist_errors);
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(1, stats.compilations);
    assert_eq!(0, stats.cache_hits.all());

    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "freebsd")))]
macro_rules! test_dist_if_compiler_available {
    ($name:ident, $compiler:ident, $compiler_name:expr) => {
        paste! {
            #[cfg_attr(not(feature = "dist-tests"), ignore)]
            #[test_case("rabbitmq" ; "with rabbitmq")]
            #[test_case("redis" ; "with redis")]
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn [<test_dist_ $name _compiles_ $compiler>] (message_broker: &str) -> Result<()> {
                let _ = env_logger::try_init();
                let compilers = find_compilers();
                if let Some(compiler) = compilers.iter().find(|c| c.name == $compiler_name) {
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

// Linux
#[cfg(target_os = "linux")]
test_dist_if_compiler_available!(stdpar, nvcxx, "nvc++");
#[cfg(target_os = "linux")]
test_dist_if_compiler_available!(stdpar, mpicxx, "mpic++");

#[cfg(not(any(target_os = "macos", target_os = "freebsd")))]
macro_rules! test_dist_if_cuda_compiler_available {
    ($cuda_compiler:ident, $host_compiler:ident, $cuda_compiler_name:expr, $host_compiler_name:expr) => {
        paste! {
            #[cfg_attr(not(feature = "dist-tests"), ignore)]
            #[test_case("rabbitmq" ; "with rabbitmq")]
            #[test_case("redis" ; "with redis")]
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn [<test_dist_cuda_compiles_ $cuda_compiler _ $host_compiler>] (message_broker: &str) -> Result<()> {
                let _ = env_logger::try_init();
                let cuda_compilers = find_cuda_compilers();
                if let Some(cuda_compiler) = cuda_compilers.iter().find(|c| c.name == $cuda_compiler_name) {
                    let host_compilers = find_compilers();
                    if let Some(host_compiler) = host_compilers.iter().find(|c| c.name == $host_compiler_name) {
                        return test_dist_cuda_compiles(
                            &cuda_compiler,
                            &host_compiler,
                            message_broker,
                        ).await
                    }
                }
                Ok(())
            }
        }
    };
}

// Linux
#[cfg(target_os = "linux")]
test_dist_if_cuda_compiler_available!(nvcc, gcc, "nvcc", "gcc");
#[cfg(target_os = "linux")]
test_dist_if_cuda_compiler_available!(nvcc, clang, "nvcc", "clang++");
#[cfg(target_os = "linux")]
test_dist_if_cuda_compiler_available!(nvcc, nvcxx, "nvcc", "nvc++");

// Clang-CUDA cannot dist-compile
// #[cfg(target_os = "linux")]
// test_dist_if_cuda_compiler_available!(clang, clang, "clang++", "clang++");
