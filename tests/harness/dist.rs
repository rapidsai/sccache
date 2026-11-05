use bytes::Buf;
use fs_err as fs;
use futures::{FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use std::{
    collections::HashMap,
    env,
    io::{BufRead, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU16, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use nix::{
    sys::{
        signal::Signal,
        wait::{WaitPidFlag, WaitStatus},
    },
    unistd::Pid,
};

use sccache::{
    config::{
        self, CacheType, ClientConfig, INSECURE_DIST_CLIENT_TOKEN,
        dist::{
            MessageBroker,
            scheduler::Config as SchedulerConfig,
            server::{Builder, Config as ServerConfig, OverlayBuilder},
        },
        utils::HTTPUrl,
    },
    dist::{SchedulerStatus, http::urls::scheduler_status as scheduler_status_url},
    errors::*,
};

use uuid::Uuid;

use super::{TC_CACHE_SIZE, check_output, client::SccacheClient, prune_command, write_json_cfg};

const CONTAINER_NAME_PREFIX: &str = "sccache_dist";
const CONTAINER_EXTERNAL_PATH: &str = "/sccache/external";
const CONTAINER_INTERNAL_PATH: &str = "/sccache/internal";
const DIST_IMAGE: &str = "sccache_dist_test_image";
const DIST_DOCKERFILE: &str = include_str!("Dockerfile.sccache-dist");
const DIST_IMAGE_BWRAP_PATH: &str = "/usr/bin/bwrap";
const MAX_STARTUP_WAIT: Duration = Duration::from_secs(30);

static SCHEDULER_PORT: AtomicU16 = AtomicU16::new(10500);
static RABBITMQ_PORT: AtomicU16 = AtomicU16::new(5672);
static REDIS_PORT: AtomicU16 = AtomicU16::new(6379);

pub fn cargo_command() -> Command {
    prune_command(Command::new("cargo"))
}

pub fn sccache_dist_path() -> PathBuf {
    env!("CARGO_BIN_EXE_sccache-dist").into()
}

fn sccache_scheduler_cfg(message_broker: &MessageBroker) -> SchedulerConfig {
    let scheduler_port = SCHEDULER_PORT.fetch_add(1, Ordering::SeqCst);

    SchedulerConfig {
        message_broker: Some(message_broker.clone()),
        public_addr: SocketAddr::from(([0, 0, 0, 0], scheduler_port)),
        jobs: vec![
            config::cache::Disk {
                dir: Path::new(CONTAINER_EXTERNAL_PATH).join("jobs"),
                ..config::cache::Disk::default()
            }
            .into(),
        ]
        .into(),
        toolchains: vec![
            config::cache::Disk {
                dir: Path::new(CONTAINER_EXTERNAL_PATH).join("toolchains"),
                ..config::cache::Disk::default()
            }
            .into(),
        ]
        .into(),
        ..Default::default()
    }
}

fn sccache_server_cfg(message_broker: &MessageBroker) -> ServerConfig {
    ServerConfig {
        max_per_core_load: 0.0.into(),
        max_per_core_prefetch: 0.0.into(),
        message_broker: Some(message_broker.clone()),
        builder: Builder::Overlay(OverlayBuilder {
            build_dir: CONTAINER_EXTERNAL_PATH.into(),
            bwrap_path: DIST_IMAGE_BWRAP_PATH.into(),
            ..Default::default()
        }),
        // Unpack toolchains inside each container
        cache_dir: Path::new(CONTAINER_INTERNAL_PATH).into(),
        // Store job results outside the container
        jobs: vec![
            config::cache::Disk {
                dir: Path::new(CONTAINER_EXTERNAL_PATH).join("jobs"),
                ..config::cache::Disk::default()
            }
            .into(),
        ]
        .into(),
        // Store toolchain archives outside the container
        toolchains: vec![
            config::cache::Disk {
                dir: Path::new(CONTAINER_EXTERNAL_PATH).join("toolchains"),
                ..config::cache::Disk::default()
            }
            .into(),
        ]
        .into(),
        toolchain_cache_size: TC_CACHE_SIZE,
        ..ServerConfig::default()
    }
}

static DIST_SYSTEM_GLOBALS: OnceLock<Mutex<Option<Arc<DistSystemGlobals>>>> = OnceLock::new();

struct DistSystemGlobals {
    dropped_handles_count: AtomicU64,
    message_brokers: Mutex<Vec<DistHandle>>,
    teardown_failures: Mutex<HashMap<(u64, String), std::process::Output>>,
    teardown_successes: Mutex<HashMap<(u64, String), std::process::Output>>,
    started: futures::future::Shared<tokio::time::Sleep>,
}

impl DistSystemGlobals {
    fn inst<'a>() -> std::sync::MutexGuard<'a, Option<Arc<DistSystemGlobals>>> {
        DIST_SYSTEM_GLOBALS
            .get_or_init(|| {
                // Make sure the docker image is available, building it if necessary.
                // This is here (and not below) so that it only happens once.
                let mut last_output = None;
                for attempt in 1..=3 {
                    let mut cmd = Command::new("docker")
                        .arg("build")
                        .arg("-q")
                        .args(["-t", DIST_IMAGE])
                        .arg("-")
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                        .unwrap();

                    cmd.stdin
                        .as_mut()
                        .unwrap()
                        .write_all(DIST_DOCKERFILE.as_bytes())
                        .unwrap();

                    let output = cmd.wait_with_output().unwrap();

                    if output.status.success() {
                        last_output = Some(output);
                        break;
                    }

                    eprintln!(
                        "docker build attempt {attempt}/3 failed (exit {}), retrying...",
                        output.status
                    );

                    last_output = Some(output);

                    std::thread::sleep(Duration::from_secs(5 * attempt as u64));
                }

                check_output(last_output.unwrap()).expect("docker build success");

                Mutex::new(Some(DistSystemGlobals::new()))
            })
            .lock()
            .unwrap()
    }

    fn get() -> Arc<Self> {
        let mut maybe = DistSystemGlobals::inst();
        if let Some(inst) = maybe.as_ref() {
            Arc::clone(inst)
        } else {
            let inst = DistSystemGlobals::new();
            *maybe = Some(Arc::clone(&inst));
            inst
        }
    }

    fn new() -> Arc<Self> {
        Arc::new(Self {
            dropped_handles_count: AtomicU64::new(0),
            message_brokers: Mutex::new(vec![
                Self::add_message_broker("rabbitmq"),
                Self::add_message_broker("redis"),
            ]),
            teardown_failures: Mutex::new(HashMap::new()),
            teardown_successes: Mutex::new(HashMap::new()),
            started: tokio::time::sleep(Duration::from_secs(10)).shared(),
        })
    }

    fn add_message_broker(message_broker: &str) -> DistHandle {
        let container_name = name_with_uuid(message_broker);
        let message_broker = DistMessageBroker::new(message_broker);

        let DistMessageBroker {
            image,
            host_port,
            container_port,
            ..
        } = &message_broker;

        Command::new("docker")
            .arg("run")
            .args(["--name", &container_name])
            .args(["-p", &format!("{host_port}:{container_port}")])
            .arg("-d")
            .arg(image)
            .output()
            .map_err(Into::into)
            .and_then(check_output)
            .expect("docker run success");

        DistHandle::MessageBroker(MessageBrokerHandle {
            broker: message_broker,
            handle: ResourceHandle::Container {
                id: container_name.clone(),
                cid: container_name,
            },
        })
    }

    pub fn rabbitmq(&self) -> Option<DistMessageBroker> {
        self.message_broker("rabbitmq")
    }

    pub fn redis(&self) -> Option<DistMessageBroker> {
        self.message_broker("redis")
    }

    pub fn message_broker(&self, message_broker: &str) -> Option<DistMessageBroker> {
        self.message_brokers
            .lock()
            .unwrap()
            .iter()
            .filter_map(|component| {
                if let DistHandle::MessageBroker(handle) = component {
                    Some(handle)
                } else {
                    None
                }
            })
            .find(|b| match message_broker {
                "rabbitmq" => b.is_amqp(),
                "redis" => b.is_redis(),
                _ => false,
            })
            .map(|b| b.broker.clone())
    }

    pub fn resource_key(&self) -> u64 {
        self.dropped_handles_count.fetch_add(1, Ordering::SeqCst)
    }

    pub fn teardown_failure(&self, idx: u64, id: &str, output: std::process::Output) {
        self.teardown_failures
            .lock()
            .unwrap()
            .insert((idx, id.to_owned()), output.clone());
    }

    pub fn teardown_success(&self, idx: u64, id: &str, output: std::process::Output) {
        self.teardown_successes
            .lock()
            .unwrap()
            .insert((idx, id.to_owned()), output.clone());
    }

    fn destroy(self: &Arc<Self>) {
        let refcount = Arc::strong_count(self).saturating_sub(1);
        // If the only reference to DistSystemGlobals is the OnceLock,
        // destroy DistSystemGlobals and replace the `Some` with `None`
        if refcount == 1 {
            self.message_brokers
                .lock()
                .unwrap()
                .drain(..)
                .for_each(|component| {
                    if let DistHandle::MessageBroker(broker) = component {
                        broker.handle.destroy(self, false);
                    }
                });

            self.print_ouputs_and_errors();

            // Replace the current Some(DistSystemGlobals) with None
            DistSystemGlobals::inst().take();
        }
    }

    fn print_ouputs_and_errors(&self) {
        fn print_output(prefix: &str, output: &std::process::Output) {
            println!("\n{prefix}");
            if !output.stdout.is_empty() {
                println!("\n=== begin stdout {}\n", "=".repeat(43));
                output
                    .stdout
                    .reader()
                    .lines()
                    .for_each(|line| println!("    {}", line.unwrap_or_default()));
                println!("\n===== end stdout  {}\n", "=".repeat(43));
            }
            if !output.stderr.is_empty() {
                println!("\n=== begin stderr  {}\n", "=".repeat(43));
                output
                    .stderr
                    .reader()
                    .lines()
                    .for_each(|line| println!("    {}", line.unwrap_or_default()));
                println!("\n===== end stderr  {}\n", "=".repeat(43));
            }
        }

        let mut has_errors = false;

        for ((_, id), output) in self
            .teardown_successes
            .lock()
            .unwrap()
            .drain()
            .sorted_by(|((a, _), _), ((b, _), _)| a.cmp(b))
        {
            print_output(
                &format!(
                    "=== server output {}\n\nid: {id}\n{}",
                    "=".repeat(61),
                    output.status
                ),
                &output,
            );
        }

        for ((_, id), errors) in self
            .teardown_failures
            .lock()
            .unwrap()
            .drain()
            .sorted_by(|((a, _), _), ((b, _), _)| a.cmp(b))
        {
            has_errors = true;
            print_output(
                &format!(
                    "=== server errors {}\n\nid: {id}\n{}",
                    "=".repeat(61),
                    errors.status
                ),
                &errors,
            );
        }

        if has_errors && !thread::panicking() {
            panic!("Encountered failures during dist system teardown");
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct DistMessageBroker {
    kind: String,
    host_port: u16,
    container_port: u16,
    config: MessageBroker,
    image: String,
}

impl DistMessageBroker {
    pub fn new(message_broker: &str) -> Self {
        match message_broker {
            "rabbitmq" => {
                let host_port = RABBITMQ_PORT.fetch_add(1, Ordering::SeqCst);
                let addr = format!("amqp://127.0.0.1:{host_port}//");
                Self {
                    kind: "rabbitmq".into(),
                    config: addr.into(),
                    image: "rabbitmq:latest".into(),
                    container_port: 5672,
                    host_port,
                }
            }
            "redis" => {
                let host_port = REDIS_PORT.fetch_add(1, Ordering::SeqCst);
                let addr = format!("redis://127.0.0.1:{host_port}/");
                Self {
                    kind: "redis".into(),
                    config: addr.into(),
                    image: "redis:7".into(),
                    container_port: 6379,
                    host_port,
                }
            }
            _ => unreachable!(""),
        }
    }

    pub fn is_amqp(&self) -> bool {
        matches!(self.kind.as_str(), "rabbitmq")
    }

    pub fn is_redis(&self) -> bool {
        matches!(self.kind.as_str(), "redis")
    }

    pub fn addr(&self) -> HTTPUrl {
        HTTPUrl::from_url(reqwest::Url::parse(&self.config.addr).unwrap())
    }
}

#[derive(Clone)]
pub struct MessageBrokerHandle {
    broker: DistMessageBroker,
    handle: ResourceHandle,
}

impl MessageBrokerHandle {
    pub fn is_amqp(&self) -> bool {
        self.broker.is_amqp()
    }
    pub fn is_redis(&self) -> bool {
        self.broker.is_redis()
    }
}

#[derive(Clone)]
pub struct SchedulerHandle {
    port: u16,
    handle: ResourceHandle,
    globals: Arc<DistSystemGlobals>,
}

impl Drop for SchedulerHandle {
    fn drop(&mut self) {
        self.handle.destroy(&self.globals, true);
        self.globals.destroy();
    }
}

#[derive(Clone)]
pub struct ServerHandle {
    handle: ResourceHandle,
    globals: Arc<DistSystemGlobals>,
}

impl SchedulerHandle {
    pub fn id(&self) -> &str {
        self.handle.id()
    }
    pub fn name(&self) -> &str {
        self.handle.name()
    }
    pub fn config<P: AsRef<Path>>(&self, root: P) -> Result<SchedulerConfig> {
        SchedulerConfig::load(self.config_path(root))
    }
    pub fn config_file<P: AsRef<Path>>(&self, root: P, cfg: SchedulerConfig) -> Result<()> {
        let mut f = fs::File::create(self.config_path(root))?;
        let buf = serde_json::to_vec(&cfg)?;
        f.write_all(&buf).map_err(Into::into)
    }
    pub fn config_path<P: AsRef<Path>>(&self, root: P) -> PathBuf {
        root.as_ref().join(format!("{}-cfg.json", self.id()))
    }
    pub fn jobs_dir<P: AsRef<Path>>(&self, root: P) -> Result<PathBuf> {
        self.config(root.as_ref())
            .and_then(|cfg| {
                cfg.jobs
                    .configs
                    .iter()
                    .find_map(|(_, c)| match c {
                        CacheType::Disk(config::cache::Disk { dir, .. }) => Some(dir.clone()),
                        _ => None,
                    })
                    .context("Scheduler should have a job storage disk cache config")
            })
            .and_then(|jobs_path| {
                jobs_path
                    .strip_prefix(CONTAINER_EXTERNAL_PATH)
                    .map(|p| root.as_ref().join(p))
                    .map_err(Into::into)
            })
    }
    pub fn toolchains_dir<P: AsRef<Path>>(&self, root: P) -> Result<PathBuf> {
        self.config(root.as_ref())
            .and_then(|cfg| {
                cfg.toolchains
                    .configs
                    .iter()
                    .find_map(|(_, c)| match c {
                        CacheType::Disk(config::cache::Disk { dir, .. }) => Some(dir.clone()),
                        _ => None,
                    })
                    .context("Scheduler should have a toolchain storage disk cache config")
            })
            .and_then(|toolchains_path| {
                toolchains_path
                    .strip_prefix(CONTAINER_EXTERNAL_PATH)
                    .map(|p| root.as_ref().join(p))
                    .map_err(Into::into)
            })
    }
    pub fn url(&self) -> HTTPUrl {
        let url = format!("http://127.0.0.1:{}", self.port);
        HTTPUrl::from_url(reqwest::Url::parse(&url).unwrap())
    }
    pub async fn status(&self) -> Result<SchedulerStatus> {
        sccache::util::install_crypto_provider();
        let req = reqwest::Client::builder()
            .build()?
            .get(scheduler_status_url(&self.url().to_url()))
            .header(http::header::ACCEPT, "application/octet-stream")
            .bearer_auth(INSECURE_DIST_CLIENT_TOKEN);
        sccache::dist::http::bincode_req_fut(req).await
    }
}

impl ServerHandle {
    pub fn id(&self) -> &str {
        self.handle.id()
    }
    pub fn name(&self) -> &str {
        self.handle.name()
    }
    pub fn config<P: AsRef<Path>>(&self, root: P) -> Result<ServerConfig> {
        ServerConfig::load(self.config_path(root))
    }
    pub fn config_file<P: AsRef<Path>>(&self, root: P, cfg: ServerConfig) -> Result<()> {
        let mut f = fs::File::create(self.config_path(root))?;
        let buf = serde_json::to_vec(&cfg)?;
        f.write_all(&buf).map_err(Into::into)
    }
    pub fn config_path<P: AsRef<Path>>(&self, root: P) -> PathBuf {
        root.as_ref().join(format!("{}-cfg.json", self.id()))
    }
    pub fn jobs_dir<P: AsRef<Path>>(&self, root: P) -> Result<PathBuf> {
        self.config(root.as_ref())
            .and_then(|cfg| {
                cfg.jobs
                    .configs
                    .iter()
                    .find_map(|(_, c)| match c {
                        CacheType::Disk(config::cache::Disk { dir, .. }) => Some(dir.clone()),
                        _ => None,
                    })
                    .context("Server should have a job storage disk cache config")
            })
            .and_then(|jobs_path| {
                jobs_path
                    .strip_prefix(CONTAINER_EXTERNAL_PATH)
                    .map(|p| root.as_ref().join(p))
                    .map_err(Into::into)
            })
    }
    pub fn toolchains_dir<P: AsRef<Path>>(&self, root: P) -> Result<PathBuf> {
        self.config(root.as_ref())
            .and_then(|cfg| {
                cfg.toolchains
                    .configs
                    .iter()
                    .find_map(|(_, c)| match c {
                        CacheType::Disk(config::cache::Disk { dir, .. }) => Some(dir.clone()),
                        _ => None,
                    })
                    .context("Server should have a toolchain storage disk cache config")
            })
            .and_then(|toolchains_path| {
                toolchains_path
                    .strip_prefix(CONTAINER_EXTERNAL_PATH)
                    .map(|p| root.as_ref().join(p))
                    .map_err(Into::into)
            })
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.handle.destroy(&self.globals, true);
        self.globals.destroy();
    }
}

#[derive(Clone)]
pub enum ResourceHandle {
    Container { id: String, cid: String },
    Process { id: String, pid: Pid },
}

impl ResourceHandle {
    pub fn id(&self) -> &str {
        match self {
            ResourceHandle::Container { id, .. } => id,
            ResourceHandle::Process { id, .. } => id,
        }
    }
    pub fn name(&self) -> &str {
        match self {
            ResourceHandle::Container { cid, .. } => cid,
            ResourceHandle::Process { id, .. } => id,
        }
    }

    fn destroy(&self, globals: &Arc<DistSystemGlobals>, log_outputs: bool) {
        let resource_key = globals.resource_key();

        // Panicking halfway through drop would either abort (if it's a double panic) or leave us with
        // resources that aren't yet cleaned up. Instead, do as much as possible then decide what to do
        // at the end - panic (if not already doing so) or let the panic continue
        macro_rules! log_err {
            ($e:expr) => {
                match $e {
                    Ok(()) => (),
                    Err((id, err)) => {
                        globals.teardown_failure(
                            resource_key,
                            id,
                            std::process::Output {
                                status: sccache::mock_command::exit_status(1),
                                stdout: vec![],
                                stderr: err.into_bytes(),
                            },
                        );
                    }
                }
            };
        }

        match self {
            ResourceHandle::Container { id, cid } => {
                log_err!(
                    Command::new("docker")
                        .args(["logs", cid])
                        .output()
                        .map(|o| {
                            if log_outputs {
                                globals.teardown_success(resource_key, cid, o);
                            }
                        })
                        .map_err(|e| (id, format!("[{id}, {cid})]: {e:?}")))
                );
                log_err!(
                    Command::new("docker")
                        .args(["kill", cid])
                        .output()
                        .map(|_| ())
                        .map_err(|e| (id, format!("[{id}, {cid})]: {e:?}")))
                );
                // If you want containers to hang around (e.g. for debugging), comment out these lines
                log_err!(
                    Command::new("docker")
                        .args(["rm", "-f", cid])
                        .output()
                        .map(|_| ())
                        .map_err(|e| (id, format!("[{id}, {cid})]: {e:?}")))
                );
            }
            ResourceHandle::Process { id, pid } => {
                log_err!(
                    nix::sys::signal::kill(*pid, Signal::SIGINT)
                        .map_err(|e| (id, format!("[{id}, {pid})]: {e:?}")))
                );
                thread::sleep(Duration::from_secs(5));
                // Default to trying to kill again, e.g. if there was an error waiting on the pid
                let mut killagain = true;
                log_err!(
                    nix::sys::wait::waitpid(*pid, Some(WaitPidFlag::WNOHANG))
                        .map(|ws| {
                            if ws != WaitStatus::StillAlive {
                                killagain = false;
                            }
                        })
                        .map_err(|e| (id, format!("[{id}, {pid})]: {e:?}")))
                );
                if killagain {
                    eprintln!("[{id}, {pid})]: process ignored SIGINT, trying SIGKILL");
                    log_err!(
                        nix::sys::signal::kill(*pid, Signal::SIGKILL)
                            .map_err(|e| (id, format!("[{id}, {pid})]: {e:?}")))
                    );
                    log_err!(
                        nix::sys::wait::waitpid(*pid, Some(WaitPidFlag::WNOHANG))
                            .map_err(|e| e.to_string())
                            .and_then(|ws| if ws == WaitStatus::StillAlive {
                                eprintln!("[{id}, {pid})]: process still alive after SIGKILL");
                                Err("process still alive after SIGKILL".into())
                            } else {
                                Ok(())
                            })
                            .map_err(|e| (id, format!("[{id}, {pid})]: {e}")))
                    );
                }
            }
        }
    }
}

#[derive(Clone)]
pub enum DistHandle {
    MessageBroker(MessageBrokerHandle),
    Scheduler(SchedulerHandle),
    Server(ServerHandle),
}

pub struct DistSystemBuilder {
    dist_system_name: String,
    globals: Arc<DistSystemGlobals>,
    job_time_limit: Option<u32>,
    message_broker: Option<DistMessageBroker>,
    scheduler_count: u64,
    scheduler_jobs_redis_storage: Option<DistMessageBroker>,
    scheduler_toolchains_redis_storage: Option<DistMessageBroker>,
    server_count: u64,
    server_jobs_redis_storage: Option<DistMessageBroker>,
    server_toolchains_redis_storage: Option<DistMessageBroker>,
}

impl DistSystemBuilder {
    pub fn new() -> Self {
        Self {
            dist_system_name: name_with_uuid("sccache_dist_test"),
            globals: DistSystemGlobals::get(),
            job_time_limit: None,
            message_broker: None,
            scheduler_count: 0,
            scheduler_jobs_redis_storage: None,
            scheduler_toolchains_redis_storage: None,
            server_count: 0,
            server_jobs_redis_storage: None,
            server_toolchains_redis_storage: None,
        }
    }

    pub fn with_name(&mut self, name: &str) -> &mut Self {
        self.dist_system_name = name.replace("+", "x");
        self
    }

    pub fn with_scheduler(&mut self) -> &mut Self {
        self.scheduler_count += 1;
        self
    }

    pub fn with_server(&mut self) -> &mut Self {
        self.server_count += 1;
        self
    }

    pub fn with_job_time_limit(&mut self, job_time_limit: u32) -> &mut Self {
        self.job_time_limit = Some(job_time_limit);
        self
    }

    pub fn with_message_broker(&mut self, message_broker: &str) -> &mut Self {
        self.message_broker = self.globals.message_broker(message_broker);
        self
    }

    pub fn with_redis_storage(&mut self) -> &mut Self {
        self.with_scheduler_jobs_redis_storage()
            .with_scheduler_toolchains_redis_storage()
            .with_server_jobs_redis_storage()
            .with_server_toolchains_redis_storage()
    }

    pub fn with_scheduler_jobs_redis_storage(&mut self) -> &mut Self {
        self.scheduler_jobs_redis_storage = self.globals.redis();
        self
    }

    pub fn with_scheduler_toolchains_redis_storage(&mut self) -> &mut Self {
        self.scheduler_toolchains_redis_storage = self.globals.redis();
        self
    }

    pub fn with_server_jobs_redis_storage(&mut self) -> &mut Self {
        self.server_jobs_redis_storage = self.globals.redis();
        self
    }

    pub fn with_server_toolchains_redis_storage(&mut self) -> &mut Self {
        self.server_toolchains_redis_storage = self.globals.redis();
        self
    }

    pub async fn build(&mut self) -> Result<DistSystem> {
        self.globals.started.clone().await;
        let name = &self.dist_system_name;
        let mut system = DistSystem::new(name);
        let message_broker = self.message_broker.as_ref().expect("Message broker exists");

        fn storage_cfg(redis: &DistMessageBroker) -> sccache::config::Caches {
            vec![config::cache::Redis::from_url(redis.addr().to_url()).into()].into()
        }

        for i in 0..self.scheduler_count {
            let mut cfg = SchedulerConfig {
                id: format!("{name}_scheduler_{i}"),
                ..system.scheduler_cfg(&message_broker.config)
            };
            if let Some(job_time_limit) = self.job_time_limit {
                cfg.job_time_limit_secs = job_time_limit;
            }
            if let Some(redis) = self.scheduler_jobs_redis_storage.as_ref() {
                cfg.jobs = storage_cfg(redis);
            }
            if let Some(redis) = self.scheduler_toolchains_redis_storage.as_ref() {
                cfg.toolchains = storage_cfg(redis);
            }
            system.add_scheduler(cfg).await?;
        }

        for i in 0..self.server_count {
            let mut cfg = ServerConfig {
                id: format!("{name}_server_{i}"),
                ..system.server_cfg(&message_broker.config)
            };
            if let Some(redis) = self.server_jobs_redis_storage.as_ref() {
                cfg.jobs = storage_cfg(redis);
            }
            if let Some(redis) = self.server_toolchains_redis_storage.as_ref() {
                cfg.toolchains = storage_cfg(redis);
            }
            system.add_server(cfg).await?;
        }

        Ok(system)
    }
}

impl Drop for DistSystemBuilder {
    fn drop(&mut self) {
        self.globals.destroy();
    }
}

pub struct DistSystem {
    name: String,
    handles: Vec<DistHandle>,
    root_dir: PathBuf,
    data_dir: PathBuf,
    dist_dir: (PathBuf, Option<tempfile::TempDir>),
    test_dir: (PathBuf, Option<tempfile::TempDir>),
    sccache_dist: PathBuf,
}

impl DistSystem {
    pub fn builder() -> DistSystemBuilder {
        DistSystemBuilder::new()
    }

    fn new(name: &str) -> Self {
        let root_dir = std::env::temp_dir().join("sccache_dist").join(name);
        let data_dir = root_dir.join("data");
        fs::create_dir_all(&data_dir).unwrap();

        let dist_dir = tempfile::Builder::new()
            .prefix("dist_")
            .tempdir_in(&root_dir)
            .unwrap();

        // Persist the dist dir if SCCACHE_DEBUG is defined
        let (dist_dir_path, dist_dir) = if env::var("SCCACHE_DEBUG").is_ok() {
            (dist_dir.keep(), None)
        } else {
            (dist_dir.path().to_path_buf(), Some(dist_dir))
        };

        let test_dir = tempfile::Builder::new()
            .prefix("test_")
            .tempdir_in(&root_dir)
            .unwrap();

        // Persist the test dir if SCCACHE_DEBUG is defined
        let (test_dir_path, test_dir) = if env::var("SCCACHE_DEBUG").is_ok() {
            (test_dir.keep(), None)
        } else {
            (test_dir.path().to_path_buf(), Some(test_dir))
        };

        Self {
            name: name.to_owned(),
            handles: vec![],
            root_dir,
            data_dir,
            dist_dir: (dist_dir_path, dist_dir),
            test_dir: (test_dir_path, test_dir),
            sccache_dist: sccache_dist_path(),
        }
    }

    pub fn data_dir(&self) -> &Path {
        self.data_dir.as_path()
    }

    pub fn dist_dir(&self) -> &Path {
        self.dist_dir.0.as_path()
    }

    pub fn test_dir(&self) -> &Path {
        self.test_dir.0.as_path()
    }

    pub fn new_client(&self, client_config: &ClientConfig) -> Arc<SccacheClient> {
        let data_dir = self.data_dir();
        write_json_cfg(data_dir, "sccache-client.json", client_config);
        Arc::new(
            SccacheClient::new(
                &data_dir.join("sccache-client.json"),
                &data_dir.join("sccache-cached-cfg"),
            )
            .start(),
        )
    }

    pub fn scheduler_cfg(&self, message_broker: &MessageBroker) -> SchedulerConfig {
        sccache_scheduler_cfg(message_broker)
    }

    pub async fn add_scheduler(&mut self, scheduler_cfg: SchedulerConfig) -> Result<&mut Self> {
        let scheduler_id = scheduler_cfg.id.clone();
        let scheduler_port = scheduler_cfg.public_addr.port();
        let container_name = name_with_uuid(&scheduler_id);

        let scheduler = SchedulerHandle {
            port: scheduler_port,
            globals: DistSystemGlobals::get(),
            handle: ResourceHandle::Container {
                id: scheduler_id,
                cid: container_name,
            },
        };

        let dist_dir = self.dist_dir();

        scheduler.config_file(dist_dir, scheduler_cfg)?;

        [
            scheduler.jobs_dir(dist_dir),
            scheduler.toolchains_dir(dist_dir),
        ]
        .into_iter()
        .filter_map(|p| p.ok())
        .flat_map(fs::create_dir_all)
        .for_each(drop);

        // Create the scheduler
        tokio::process::Command::new("docker")
            .arg("run")
            .args(["--name", scheduler.name()])
            .args(["-e", "SCCACHE_NO_DAEMON=1"])
            .args([
                "-e",
                &format!(
                    "SCCACHE_LOG={}",
                    (
                        // Prefer sccache=trace if SCCACHE_DEBUG=1
                        env::var("SCCACHE_DEBUG").and(Ok(
                            "celery=info,sccache=trace,tower_http=debug,axum::rejection=trace"
                        ))
                    )
                    .or(env::var("SCCACHE_SERVER_LOG").as_deref())
                    .or(env::var("SCCACHE_LOG").as_deref())
                    .unwrap_or("celery=info,sccache=debug,tower_http=debug,axum::rejection=trace") // default to debug
                ),
            ])
            .args([
                "-e",
                &format!("SCCACHE_DIST_DEPLOYMENT_NAME={}", &self.name),
            ])
            .args(["-e", "RUST_BACKTRACE=1"])
            .args(["--network", "host"])
            .args(["--restart", "always"])
            .args([
                "-v",
                &format!(
                    "{exe}:/sccache-dist:z",
                    exe = self.sccache_dist.as_path().display()
                ),
            ])
            .args([
                "-v",
                &format!(
                    "{src}:{CONTAINER_EXTERNAL_PATH}:z",
                    src = dist_dir.display()
                ),
            ])
            .arg("-d")
            .arg(DIST_IMAGE)
            .args([
                "bash",
                "-c",
                &sccache_dist_docker_run_script(
                    "scheduler",
                    scheduler.config_path(CONTAINER_EXTERNAL_PATH),
                ),
            ])
            .output()
            .await
            .map_err(Into::into)
            .and_then(check_output)?;

        self.handles.push(DistHandle::Scheduler(scheduler));

        let scheduler = self.schedulers().into_iter().last().unwrap();

        tokio::time::timeout(MAX_STARTUP_WAIT, async {
            loop {
                let status = scheduler.status().await;
                if matches!(status, Ok(SchedulerStatus { .. })) {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        })
        .await
        .with_context(|| {
            format!(
                "Timeout waiting for scheduler {scheduler_id:?}",
                scheduler_id = scheduler.id()
            )
        })?;

        Ok(self)
    }

    pub fn server_cfg(&self, message_broker: &MessageBroker) -> ServerConfig {
        sccache_server_cfg(message_broker)
    }

    pub async fn add_server(&mut self, server_cfg: ServerConfig) -> Result<&mut Self> {
        let server_id = server_cfg.id.clone();
        let container_name = name_with_uuid(&server_id);

        let server = ServerHandle {
            globals: DistSystemGlobals::get(),
            handle: ResourceHandle::Container {
                id: server_id,
                cid: container_name,
            },
        };

        let dist_dir = self.dist_dir();

        server.config_file(dist_dir, server_cfg)?;

        [server.jobs_dir(dist_dir), server.toolchains_dir(dist_dir)]
            .into_iter()
            .filter_map(|p| p.ok())
            .flat_map(fs::create_dir_all)
            .for_each(drop);

        tokio::process::Command::new("docker")
            .arg("run")
            // Important for the bubblewrap builder
            .arg("--privileged")
            .args(["--name", server.name()])
            .args(["-e", "SCCACHE_NO_DAEMON=1"])
            .args([
                "-e",
                &format!(
                    "SCCACHE_LOG={}",
                    (
                        // Prefer sccache=trace if SCCACHE_DEBUG=1
                        env::var("SCCACHE_DEBUG").and(Ok("celery=info,sccache=trace"))
                    )
                    .or(env::var("SCCACHE_SERVER_LOG").as_deref())
                    .or(env::var("SCCACHE_LOG").as_deref())
                    .unwrap_or("celery=info,sccache=debug") // default to debug
                ),
            ])
            .args([
                "-e",
                &format!("SCCACHE_DIST_DEPLOYMENT_NAME={}", &self.name),
            ])
            .args(["-e", "RUST_BACKTRACE=1"])
            .args(["--network", "host"])
            .args(["--restart", "always"])
            .args([
                "-v",
                &format!(
                    "{exe}:/sccache-dist:z",
                    exe = self.sccache_dist.as_path().display()
                ),
            ])
            .args([
                "-v",
                &format!(
                    "{src}:{CONTAINER_EXTERNAL_PATH}:z",
                    src = dist_dir.display()
                ),
            ])
            .arg("-d")
            .arg(DIST_IMAGE)
            .args([
                "bash",
                "-c",
                &sccache_dist_docker_run_script(
                    "server",
                    server.config_path(CONTAINER_EXTERNAL_PATH),
                ),
            ])
            .output()
            .await
            .map_err(Into::into)
            .and_then(check_output)?;

        self.handles.push(DistHandle::Server(server));

        self.wait_server_ready(self.servers().into_iter().last().unwrap())
            .await?;

        Ok(self)
    }

    fn schedulers(&self) -> impl IntoIterator<Item = &SchedulerHandle> {
        self.handles.iter().filter_map(|component| {
            if let DistHandle::Scheduler(handle) = component {
                Some(handle)
            } else {
                None
            }
        })
    }

    fn servers(&self) -> impl IntoIterator<Item = &ServerHandle> {
        self.handles.iter().filter_map(|component| {
            if let DistHandle::Server(handle) = component {
                Some(handle)
            } else {
                None
            }
        })
    }

    pub fn scheduler(&self, scheduler_index: usize) -> Result<&SchedulerHandle> {
        self.schedulers()
            .into_iter()
            .nth(scheduler_index)
            .ok_or_else(|| anyhow!("Scheduler {scheduler_index} doesn't exist"))
    }

    pub fn server(&self, server_index: usize) -> Result<&ServerHandle> {
        self.servers()
            .into_iter()
            .nth(server_index)
            .ok_or_else(|| anyhow!("Server {server_index} doesn't exist"))
    }

    pub async fn restart_server(&self, server: &ServerHandle) -> Result<()> {
        match server.handle {
            ResourceHandle::Container { ref cid, .. } => tokio::process::Command::new("docker")
                .args(["restart", cid])
                .output()
                .await
                .map_err(Into::into)
                .and_then(check_output)?,
            ResourceHandle::Process { .. } => {
                // TODO: pretty easy, just no need yet
                bail!("restart not yet implemented for pids")
            }
        }
        self.wait_server_ready(server).await
    }

    pub async fn wait_server_ready(&self, server: &ServerHandle) -> Result<()> {
        tokio::time::timeout(MAX_STARTUP_WAIT, async {
            loop {
                use futures::{future::ready, stream::iter};

                let found = iter(self.schedulers())
                    .then(|scheduler| scheduler.status())
                    .map_ok(|sched| iter(sched.servers).map(Result::Ok))
                    .try_flatten()
                    .try_any(|serv| ready(serv.id == server.id()))
                    .await?;

                if found {
                    break Ok(());
                }

                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        })
        .await
        .with_context(|| {
            format!(
                "Timeout waiting for server {server_id:?}",
                server_id = server.id()
            )
        })?
    }

    pub fn count_server_toolchains(&self, server: &ServerHandle) -> Result<usize> {
        server
            .toolchains_dir(self.dist_dir())
            .and_then(|dir| fs::read_dir(dir).map_err(Into::into))
            .map(|dir| {
                dir.filter_map(|d| d.ok().filter(|d| d.file_name() != "_"))
                    .count()
            })
    }
}

fn name_with_uuid(name: &str) -> String {
    format!(
        "{}_{}_{}",
        CONTAINER_NAME_PREFIX,
        name,
        Uuid::new_v4().simple()
    )
}

fn sccache_dist_docker_run_script<P: AsRef<Path>>(dist_type: &str, config_path: P) -> String {
    format!(
        // umask 000 so files created under the CONTAINER_EXTERNAL_PATH
        // bind mount by the root user in the container are writable.
        //
        // This ensures the scheduler and servers can share these dirs,
        // but the tempfile parent dirs (created on the host) are still
        // cleaned up when the TempDir instances are dropped.
        r#"
            set -o errexit;
            umask 000;
            exec /sccache-dist {dist_type} --config {cfg:?}
        "#,
        cfg = config_path.as_ref()
    )
}
