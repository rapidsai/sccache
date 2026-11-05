use std::{env, sync::Arc, time::Duration};

#[cfg_attr(target_os = "freebsd", path = "build_freebsd.rs")]
mod build;

mod cmdline;
use cmdline::Command;

#[cfg(target_os = "freebsd")]
use sccache::config::dist::server::PotBuilder;
#[cfg(not(target_os = "freebsd"))]
use sccache::config::dist::server::{DockerBuilder, OverlayBuilder};
use sccache::{
    cache::{StorageKind, disk::DiskCache},
    config::{self, CacheMode, dist::server::Builder},
    dist::{
        self, BuilderIncoming, env_info, metrics::Metrics, scheduler, server, tasks,
        token_check::new_client_auth_check,
    },
    errors::*,
};

// Only supported on x86_64/aarch64 Linux machines and on FreeBSD
#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "freebsd"
)))]
fn main() {
    compile_error!("Distributed compilation is only supported on Linux/x86_64 and FreeBSD!");
}

// Only supported on x86_64/aarch64 Linux machines and on FreeBSD
#[cfg(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    target_os = "freebsd"
))]
fn main() {
    dist::init_logging();

    rustls::crypto::ring::default_provider()
        .install_default()
        .unwrap();

    let command = match cmdline::try_parse_from(env::args()) {
        Ok(cmd) => cmd,
        Err(e) => match e.downcast::<clap::error::Error>() {
            Ok(clap_err) => clap_err.exit(),
            Err(some_other_err) => {
                println!("sccache-dist: {some_other_err}");
                for source_err in some_other_err.chain().skip(1) {
                    println!("sccache-dist: caused by: {source_err}");
                }
                std::process::exit(1);
            }
        },
    };

    std::process::exit(match run(command) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("sccache-dist: error: {e}");

            for e in e.chain().skip(1) {
                eprintln!("sccache-dist: caused by: {e}");
            }
            2
        }
    });
}

fn run(command: Command) -> Result<()> {
    let num_cpus = sccache::util::num_cpus();

    sccache::util::block_on(
        &tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?,
        async move {
            match command {
                Command::Scheduler(config::dist::scheduler::Config {
                    auth: client_auth,
                    heartbeat_interval_ms,
                    job_time_limit_secs,
                    jobs,
                    keepalive,
                    max_body_size,
                    max_concurrent_streams,
                    message_broker,
                    metrics,
                    public_addr,
                    id: scheduler_id,
                    shutdown_timeout_secs,
                    toolchains,
                }) => {
                    tracing::info!(
                        "Starting {sccache} v{version} scheduler",
                        sccache = env!("CARGO_BIN_NAME"),
                        version = env!("CARGO_PKG_VERSION")
                    );

                    let metrics = Metrics::new(
                        metrics,
                        [
                            ("env".into(), env_info()),
                            ("type".into(), "scheduler".into()),
                            ("scheduler_id".into(), scheduler_id.clone()),
                        ]
                        .into(),
                    )?;

                    let jobs = StorageKind::Compilations
                        .create(&jobs, &[])
                        .await
                        .context("Failed to initialize jobs storage")?;

                    // Verify read/write access to jobs storage
                    match jobs.check().await {
                        Ok(CacheMode::ReadWrite) => {}
                        _ => {
                            bail!("Scheduler jobs storage must be read/write")
                        }
                    }

                    let toolchains = StorageKind::Compilations
                        .create(&toolchains, &[])
                        .await
                        .context("Failed to initialize toolchain storage")?;

                    // Verify read/write access to toolchain storage
                    match toolchains.check().await {
                        Ok(CacheMode::ReadWrite) => {}
                        _ => {
                            bail!("Scheduler toolchain storage must be read/write")
                        }
                    }

                    // Create ClientAuthCheck and bail on Err before registering with Celery
                    if client_auth.is_empty() {
                        bail!(
                            "Scheduler must be configured to use at least one client authentication mechanism"
                        );
                    }

                    let client_auth_check = sccache::util::try_join_all(
                        client_auth.into_iter().map(new_client_auth_check),
                    )
                    .await?;

                    let tasks = tasks::Tasks::scheduler(
                        &scheduler_id,
                        u16::MAX,
                        job_time_limit_secs,
                        message_broker,
                    )
                    .await?;

                    let scheduler = scheduler::Scheduler::builder()
                        .with_scheduler_id(scheduler_id)
                        .with_jobs_storage(jobs)
                        .with_metrics(metrics.clone())
                        .with_tasks(tasks)
                        .with_toolchains_storage(toolchains)
                        .build()?;

                    let (handle, server) =
                        dist::http::Scheduler::new(scheduler.clone(), client_auth_check).serve(
                            metrics,
                            public_addr,
                            keepalive,
                            max_body_size,
                            max_concurrent_streams,
                        );

                    scheduler
                        .start(
                            handle,
                            server,
                            Duration::from_millis(heartbeat_interval_ms),
                            Duration::from_secs(shutdown_timeout_secs),
                        )
                        .await
                        .inspect_err(|e| tracing::error!("Server shutdown with error: {e:#}"))
                }

                Command::Server(config::dist::server::Config {
                    message_broker,
                    builder,
                    cache_dir,
                    health_check_bind_addr,
                    heartbeat_interval_ms,
                    jobs,
                    max_per_core_load,
                    max_per_core_prefetch,
                    metrics,
                    id: server_id,
                    shutdown_timeout_secs,
                    toolchain_cache_size,
                    toolchains,
                }) => {
                    tracing::info!(
                        "Starting {sccache} v{version} server",
                        sccache = env!("CARGO_BIN_NAME"),
                        version = env!("CARGO_PKG_VERSION")
                    );

                    let metrics = Metrics::new(
                        metrics,
                        [
                            ("env".into(), env_info()),
                            ("type".into(), "server".into()),
                            ("server_id".into(), server_id.clone()),
                        ]
                        .into(),
                    )?;

                    let jobs = StorageKind::Compilations
                        .create(&jobs, &[])
                        .await
                        .context("Failed to initialize jobs storage")?;

                    // Verify read/write access to jobs storage
                    match jobs.check().await {
                        Ok(CacheMode::ReadWrite) => {}
                        _ => {
                            bail!("Server jobs storage must be read/write")
                        }
                    }

                    let toolchains = StorageKind::Compilations
                        .create(&toolchains, &[])
                        .await
                        .context("Failed to initialize toolchain storage")?;

                    // Verify toolchain storage
                    toolchains
                        .check()
                        .await
                        .context("Failed to initialize toolchain storage")?;

                    let occupancy = (num_cpus as f64 * f64::from(max_per_core_load))
                        .floor()
                        .max(1.0) as usize;

                    let pre_fetch = (num_cpus as f64 * f64::from(max_per_core_prefetch))
                        .floor()
                        .max(0.0) as usize;

                    let job_queue = Arc::new(tokio::sync::Semaphore::new(occupancy));

                    let should_inflate_toolchains = !matches!(builder, Builder::Docker { .. });
                    let builder = init_builder(builder, job_queue.clone()).await?;

                    let tasks = tasks::Tasks::server(
                        &server_id,
                        (occupancy as u16).saturating_add(pre_fetch as u16),
                        message_broker,
                    )
                    .await?;

                    server::Server::builder()
                        .with_builder(builder)
                        .with_job_queue(job_queue)
                        .with_jobs_storage(jobs)
                        .with_metrics(metrics)
                        .with_num_cpus(num_cpus)
                        .with_occupancy(occupancy)
                        .with_pre_fetch(pre_fetch)
                        .with_server_id(server_id)
                        .with_tasks(tasks)
                        .with_toolchains_storage(toolchains)
                        .with_disk_cache(Arc::new(DiskCache::new(
                            cache_dir.join("tc"), // root
                            toolchain_cache_size, // max_size,
                            crate::config::CacheMode::ReadWrite,
                            vec![],
                        )))
                        .with_should_inflate_toolchains(should_inflate_toolchains)
                        .build()?
                        .start(
                            // Report status every `heartbeat_interval_ms` milliseconds
                            Duration::from_millis(heartbeat_interval_ms),
                            Duration::from_secs(shutdown_timeout_secs),
                            health_check_bind_addr,
                        )
                        .await
                        .inspect_err(|e| tracing::error!("Server shutdown with error: {e:#}"))
                }
            }
        },
    )
}

async fn init_builder(
    config: Builder,
    job_queue: Arc<tokio::sync::Semaphore>,
) -> Result<Arc<dyn BuilderIncoming>> {
    match config {
        #[cfg(not(target_os = "freebsd"))]
        Builder::Docker(DockerBuilder {
            image,
            run_cmd,
            exec_cmd,
        }) => Ok(Arc::new(
            build::DockerBuilder::new(image, run_cmd, exec_cmd, job_queue.clone())
                .await
                .context("Docker builder failed to start")?,
        ) as Arc<dyn BuilderIncoming>),
        #[cfg(not(target_os = "freebsd"))]
        Builder::Overlay(OverlayBuilder {
            bwrap_path,
            build_dir,
            exec_cmd,
            lower_dirs,
            env: overlay_env,
        }) => Ok(Arc::new(
            build::OverlayBuilder::new(
                bwrap_path,
                build_dir,
                exec_cmd,
                lower_dirs,
                overlay_env,
                job_queue.clone(),
            )
            .await
            .context("Overlay builder failed to start")?,
        ) as Arc<dyn BuilderIncoming>),
        #[cfg(target_os = "freebsd")]
        Builder::Pot(PotBuilder {
            pot_fs_root,
            clone_from,
            pot_cmd,
            pot_clone_args,
        }) => Ok(Arc::new(
            build::PotBuilder::new(
                pot_fs_root,
                clone_from,
                pot_cmd,
                pot_clone_args,
                job_queue.clone(),
            )
            .await
            .context("Pot builder failed to start")?,
        ) as Arc<dyn BuilderIncoming>),
        _ => bail!(
            "Builder type `{}` not supported on this platform",
            format!("{config:?}")
                .split_whitespace()
                .next()
                .unwrap_or("")
        ),
    }
}
