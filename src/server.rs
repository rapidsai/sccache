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
// limitations under the License.SCCACHE_MAX_FRAME_LENGTH

use crate::{
    cache::{Cache, CacheMode, Storage, StorageKind, multilevel::MultiLevelStats},
    compiler::{
        CacheControl, CompileResult, Compiler, CompilerArguments, CompilerHasher, CompilerKind,
        CompilerProxy, DistType, Language, MissType, compiler_info_args, get_compiler_info,
    },
    config::Config,
    dist,
    errors::*,
    jobserver::Client,
    mock_command::{CommandCreatorSync, ProcessCommandCreator, ProcessOutput},
    protocol::{
        Compile, CompileFinished, CompileResponse, Request, Response, StorageHandshakeInfo,
    },
    util::{self, AsyncMulticast, AsyncMulticastFunc},
};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut, buf::BufMut};
use filetime::FileTime;
use fs::metadata;
use fs_err as fs;
use futures::{FutureExt, Sink, SinkExt, Stream, StreamExt, TryFutureExt, channel::mpsc, future};
use serde::{Deserialize, Serialize};

#[cfg(target_os = "android")]
use std::os::android::net::SocketAddrExt;

#[cfg(target_os = "linux")]
use std::os::linux::net::SocketAddrExt;
use std::sync::atomic::AtomicU64;

use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    env,
    ffi::OsString,
    future::Future,
    io::{self, Write},
    marker::Unpin,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
    time::Duration,
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    runtime::Runtime,
    sync::{Mutex, RwLock},
    time::{self, Sleep, sleep},
};
use tokio_serde::Framed;
use tokio_util::codec::{LengthDelimitedCodec, length_delimited};
use tower::Service;
use unit_prefix::NumberPrefix;

#[cfg(feature = "dist-client")]
use {
    crate::config,
    anyhow::Context as _,
    std::{mem, time::Instant},
};

/// If the dist client couldn't be created, retry creation at this number
/// of seconds from now (or later)
#[cfg(feature = "dist-client")]
const DIST_CLIENT_RECREATE_TIMEOUT: Duration = Duration::from_secs(30);

/// Result of background server startup.
#[derive(Debug, Serialize, Deserialize)]
pub enum ServerStartup {
    /// Server started successfully on `addr`.
    Ok { addr: String },
    /// Server Addr already in use.
    AddrInUse,
    /// Timed out waiting for server startup.
    TimedOut,
    /// Server encountered an error.
    Err { reason: String },
}

/// Get the time the server should idle for before shutting down, in seconds.
fn get_idle_timeout() -> u64 {
    // A value of 0 disables idle shutdown entirely.
    env::var("SCCACHE_IDLE_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn notify_server_startup_internal<W: Write>(mut w: W, status: ServerStartup) -> Result<()> {
    util::write_length_prefixed_bincode(&mut w, status)
}

#[cfg(unix)]
fn notify_server_startup(name: Option<&OsString>, status: ServerStartup) -> Result<()> {
    use std::os::unix::net::UnixStream;
    let name = match name {
        Some(s) => s,
        None => return Ok(()),
    };
    debug!("notify_server_startup({status:?})");
    let stream = UnixStream::connect(name)?;
    notify_server_startup_internal(stream, status)
}

#[cfg(windows)]
fn notify_server_startup(name: Option<&OsString>, status: ServerStartup) -> Result<()> {
    use fs::OpenOptions;

    let name = match name {
        Some(s) => s,
        None => return Ok(()),
    };
    let pipe = OpenOptions::new().write(true).read(true).open(name)?;
    notify_server_startup_internal(pipe, status)
}

pub struct DistClientContainer {
    // The actual dist client state
    #[cfg(feature = "dist-client")]
    state: futures::lock::Mutex<DistClientState>,
}

#[cfg(feature = "dist-client")]
pub struct DistClientConfig {
    // From the static dist configuration
    scheduler_url: Option<config::HTTPUrl>,
    auth: config::DistAuth,
    fallback_to_local_compile: bool,
    max_retries: f64,
    net: config::DistNetworking,
    cache_dir: PathBuf,
    toolchain_cache_size: u64,
    toolchains: Vec<config::DistToolchainConfig>,
    rewrite_includes_only: bool,
}

#[cfg(feature = "dist-client")]
pub enum DistClientState {
    #[cfg(feature = "dist-client")]
    Some(Box<DistClientConfig>, Arc<dyn dist::Client>),
    #[cfg(feature = "dist-client")]
    FailWithMessage(Box<DistClientConfig>, String),
    #[cfg(feature = "dist-client")]
    RetryCreateAt(Box<DistClientConfig>, Instant),
    Disabled,
}

#[cfg(not(feature = "dist-client"))]
impl DistClientContainer {
    #[cfg(not(feature = "dist-client"))]
    fn new(config: &Config, _: &tokio::runtime::Handle) -> Self {
        if config.dist.scheduler_url.is_some() {
            warn!(
                "Scheduler address configured but dist feature disabled, disabling distributed sccache"
            )
        }
        Self {}
    }

    pub fn new_disabled() -> Self {
        Self {}
    }

    #[cfg(feature = "dist-client")]
    pub fn new_with_state(_: DistClientState) -> Self {
        Self {}
    }

    pub async fn reset_state(&self) {}

    pub async fn get_status(&self) -> DistInfo {
        DistInfo::Disabled("dist-client feature not selected".to_string())
    }

    async fn get_client(&self) -> Result<Option<Arc<dyn dist::Client>>> {
        Ok(None)
    }
}

#[cfg(feature = "dist-client")]
impl DistClientContainer {
    fn new(config: &Config, pool: &tokio::runtime::Handle) -> Self {
        let config = DistClientConfig {
            scheduler_url: config.dist.scheduler_url.clone(),
            auth: config.dist.auth.clone(),
            fallback_to_local_compile: config.dist.fallback_to_local_compile,
            max_retries: config.dist.max_retries,
            net: config.dist.net.clone(),
            cache_dir: config.dist.cache_dir.clone(),
            toolchain_cache_size: config.dist.toolchain_cache_size,
            toolchains: config.dist.toolchains.clone(),
            rewrite_includes_only: config.dist.rewrite_includes_only,
        };
        let state = Self::create_state(config);
        let state = pool.block_on(state);
        Self {
            state: futures::lock::Mutex::new(state),
        }
    }

    #[cfg(feature = "dist-client")]
    pub fn new_with_state(state: DistClientState) -> Self {
        Self {
            state: futures::lock::Mutex::new(state),
        }
    }

    pub fn new_disabled() -> Self {
        Self {
            state: futures::lock::Mutex::new(DistClientState::Disabled),
        }
    }

    pub async fn reset_state(&self) {
        let mut guard = self.state.lock().await;
        let state = &mut *guard;
        match mem::replace(state, DistClientState::Disabled) {
            DistClientState::Some(cfg, _)
            | DistClientState::FailWithMessage(cfg, _)
            | DistClientState::RetryCreateAt(cfg, _) => {
                warn!("State reset. Will recreate");
                *state = DistClientState::RetryCreateAt(
                    cfg,
                    Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
                );
            }
            DistClientState::Disabled => (),
        }
    }

    pub async fn get_status(&self) -> DistInfo {
        let mut guard = self.state.lock().await;
        let state = &mut *guard;
        let (client, scheduler_url) = match state {
            DistClientState::Disabled => return DistInfo::Disabled("disabled".to_string()),
            DistClientState::FailWithMessage(cfg, _) => {
                return DistInfo::NotConnected(
                    cfg.scheduler_url.clone(),
                    "enabled, auth not configured".to_string(),
                );
            }
            DistClientState::RetryCreateAt(cfg, time) => {
                return DistInfo::NotConnected(
                    cfg.scheduler_url.clone(),
                    format!(
                        "enabled, not connected, will retry in {:.1}s",
                        time.duration_since(Instant::now()).as_secs_f32()
                    ),
                );
            }
            DistClientState::Some(cfg, client) => (Arc::clone(client), cfg.scheduler_url.clone()),
        };

        match client.get_status().await {
            Ok(res) => DistInfo::SchedulerStatus(scheduler_url.clone(), res),
            Err(_) => DistInfo::NotConnected(
                scheduler_url.clone(),
                "could not communicate with scheduler".to_string(),
            ),
        }
    }

    async fn get_client(&self) -> Result<Option<Arc<dyn dist::Client>>> {
        let mut guard = self.state.lock().await;
        let state = &mut *guard;
        Self::maybe_recreate_state(state).await;
        let res = match state {
            DistClientState::Some(_, dc) => Ok(Some(dc.clone())),
            DistClientState::Disabled | DistClientState::RetryCreateAt(_, _) => Ok(None),
            DistClientState::FailWithMessage(_, msg) => Err(anyhow!(msg.clone())),
        };
        if res.is_err() {
            let config = match mem::replace(state, DistClientState::Disabled) {
                DistClientState::FailWithMessage(config, _) => config,
                _ => unreachable!(),
            };
            // The client is most likely mis-configured, make sure we
            // re-create on our next attempt.
            *state = DistClientState::RetryCreateAt(
                config,
                Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
            );
        }
        res
    }

    async fn maybe_recreate_state(state: &mut DistClientState) {
        if let DistClientState::RetryCreateAt(_, instant) = *state {
            if instant > Instant::now() {
                return;
            }
            let config = match mem::replace(state, DistClientState::Disabled) {
                DistClientState::RetryCreateAt(config, _) => config,
                _ => unreachable!(),
            };
            info!("Attempting to recreate the dist client");
            *state = Self::create_state(*config).await;
        }
    }

    // Attempt to recreate the dist client
    async fn create_state(config: DistClientConfig) -> DistClientState {
        macro_rules! try_or_retry_later {
            ($v:expr) => {{
                match $v {
                    Ok(v) => v,
                    Err(e) => {
                        // `{:?}` prints the full cause chain and backtrace.
                        error!("{:?}", e);
                        return DistClientState::RetryCreateAt(
                            Box::new(config),
                            Instant::now() + DIST_CLIENT_RECREATE_TIMEOUT,
                        );
                    }
                }
            }};
        }

        macro_rules! try_or_fail_with_message {
            ($v:expr) => {{
                match $v {
                    Ok(v) => v,
                    Err(e) => {
                        // `{:?}` prints the full cause chain and backtrace.
                        let errmsg = format!("{:?}", e);
                        error!("{}", errmsg);
                        return DistClientState::FailWithMessage(
                            Box::new(config),
                            errmsg.to_string(),
                        );
                    }
                }
            }};
        }
        match config.scheduler_url {
            Some(ref addr) => {
                let url = addr.to_url();
                info!("Enabling distributed sccache to {url}");
                let auth_token = match &config.auth {
                    config::DistAuth::Token { token } => Ok(token.to_owned()),
                    config::DistAuth::Oauth2CodeGrantPKCE { auth_url, .. }
                    | config::DistAuth::Oauth2Implicit { auth_url, .. } => {
                        Self::get_cached_config_auth_token(auth_url)
                    }
                };
                let auth_token = try_or_fail_with_message!(
                    auth_token
                        .context("could not load client auth token, run |sccache --dist-auth|")
                );
                let dist_client = dist::http::Client::new(
                    url,
                    &config.cache_dir.join("client"),
                    config.toolchain_cache_size,
                    &config.toolchains,
                    auth_token,
                    config.fallback_to_local_compile,
                    config.max_retries,
                    config.rewrite_includes_only,
                    &config.net,
                )
                .await;
                let dist_client =
                    try_or_retry_later!(dist_client.context("failure during dist client creation"));
                use crate::dist::Client;
                match dist_client.get_status().await {
                    Ok(res) => {
                        info!(
                            "Successfully created dist client with {:?} cores across {:?} servers",
                            res.info.occupancy,
                            res.servers.len()
                        );
                        DistClientState::Some(Box::new(config), Arc::new(dist_client))
                    }
                    Err(_) => {
                        warn!(
                            "Scheduler address configured, but could not communicate with scheduler"
                        );
                        DistClientState::RetryCreateAt(
                            Box::new(config),
                            Instant::now() + DIST_CLIENT_RECREATE_TIMEOUT,
                        )
                    }
                }
            }
            None => {
                info!("No scheduler address configured, disabling distributed sccache");
                DistClientState::Disabled
            }
        }
    }

    fn get_cached_config_auth_token(auth_url: &str) -> Result<String> {
        let cached_config = config::CachedConfig::reload()?;
        cached_config
            .with(|c| c.dist.auth_tokens.get(auth_url).map(String::to_owned))
            .with_context(|| format!("token for url {auth_url} not present in cached config"))
    }
}

thread_local! {
    /// catch_unwind doesn't provide panic location, so we store that
    /// information via a panic hook to be used when catch_unwind
    /// catches a panic.
    static PANIC_LOCATION: Cell<Option<(String, u32, u32)>> = const { Cell::new(None) };
}

/// Start an sccache server, listening on `addr`.
///
/// Spins an event loop handling client connections until a client
/// requests a shutdown.
pub fn start_server(config: Config, addr: &crate::net::SocketAddr) -> Result<()> {
    info!("start_server: {addr}");
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        PANIC_LOCATION.with(|l| {
            l.set(
                info.location()
                    .map(|loc| (loc.file().to_string(), loc.line(), loc.column())),
            );
        });
        panic_hook(info);
    }));
    let client = Client::new();
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    if std::env::var("TOKIO_WORKER_THREADS").is_err() {
        builder.worker_threads(2 * util::num_cpus());
    }
    let runtime = builder.enable_all().build()?;
    let pool = runtime.handle().clone();
    let dist_client = DistClientContainer::new(&config, &pool);

    let notify = env::var_os("SCCACHE_STARTUP_NOTIFY");

    let init_storage = || -> Result<(Arc<dyn Storage>, Arc<dyn Storage>)> {
        runtime.block_on(async {
            Ok((
                StorageKind::Compilations
                    .create(&config.caches, &config.basedirs)
                    .await?,
                StorageKind::Preprocessor
                    .create(&config.caches, &config.basedirs)
                    .await?,
            ))
        })
    };

    let (storage, preprocessor_storage) = init_storage().or_else(|err| {
        notify_server_startup(
            notify.as_ref(),
            ServerStartup::Err {
                reason: err.to_string(),
            },
        )?;
        Err(err)
    })?;

    let res: io::Result<(crate::net::SocketAddr, Box<dyn FnOnce(_) -> io::Result<()>>)> = (|| {
        match addr {
            crate::net::SocketAddr::Net(addr) => {
                trace!("binding TCP {addr}");
                let l = runtime.block_on(tokio::net::TcpListener::bind(addr))?;
                let srv = SccacheServer::<_>::with_listener(
                    l,
                    runtime,
                    client,
                    dist_client,
                    storage,
                    preprocessor_storage,
                );
                Ok((
                    srv.local_addr().unwrap(),
                    Box::new(move |f| srv.run(f)) as Box<dyn FnOnce(_) -> _>,
                ))
            }
            #[cfg(unix)]
            crate::net::SocketAddr::Unix(path) => {
                trace!("binding unix socket {}", path.display());
                // Unix socket will report addr in use on any unlink file.
                let _ = std::fs::remove_file(path);
                let l = {
                    let _guard = runtime.enter();
                    tokio::net::UnixListener::bind(path)?
                };
                let srv = SccacheServer::<_>::with_listener(
                    l,
                    runtime,
                    client,
                    dist_client,
                    storage,
                    preprocessor_storage,
                );
                Ok((
                    srv.local_addr().unwrap(),
                    Box::new(move |f| srv.run(f)) as Box<dyn FnOnce(_) -> _>,
                ))
            }
            #[cfg(any(target_os = "linux", target_os = "android"))]
            crate::net::SocketAddr::UnixAbstract(p) => {
                trace!("binding abstract unix socket {}", p.escape_ascii());
                let abstract_addr = std::os::unix::net::SocketAddr::from_abstract_name(p)?;
                let l = std::os::unix::net::UnixListener::bind_addr(&abstract_addr)?;
                l.set_nonblocking(true)?;
                let l = {
                    let _guard = runtime.enter();
                    tokio::net::UnixListener::from_std(l)?
                };
                let srv = SccacheServer::<_>::with_listener(
                    l,
                    runtime,
                    client,
                    dist_client,
                    storage,
                    preprocessor_storage,
                );
                Ok((
                    srv.local_addr()
                        .unwrap_or_else(|| crate::net::SocketAddr::UnixAbstract(p.clone())),
                    Box::new(move |f| srv.run(f)) as Box<dyn FnOnce(_) -> _>,
                ))
            }
        }
    })();
    match res {
        Ok((addr, run)) => {
            info!("server started, listening on {addr}");
            notify_server_startup(
                notify.as_ref(),
                ServerStartup::Ok {
                    addr: addr.to_string(),
                },
            )?;
            run(future::pending::<()>()).map_err(anyhow::Error::from)?;
            Ok(())
        }
        Err(e) => {
            error!("failed to start server: {e}");
            if io::ErrorKind::AddrInUse == e.kind() {
                notify_server_startup(notify.as_ref(), ServerStartup::AddrInUse)?;
            } else if cfg!(windows) && Some(10013) == e.raw_os_error() {
                // 10013 is the "WSAEACCES" error, which can occur if the requested port
                // has been allocated for other purposes, such as winNAT or Hyper-V.
                let windows_help_message = "A Windows port exclusion is blocking use of the configured port.\nTry setting SCCACHE_SERVER_PORT to a new value.";
                let reason: String = format!("{windows_help_message}\n{e}");
                notify_server_startup(notify.as_ref(), ServerStartup::Err { reason })?;
            } else {
                let reason = e.to_string();
                notify_server_startup(notify.as_ref(), ServerStartup::Err { reason })?;
            }
            Err(e.into())
        }
    }
}

pub struct SccacheServer<A: crate::net::Acceptor, C: CommandCreatorSync = ProcessCommandCreator> {
    runtime: Runtime,
    listener: A,
    rx: mpsc::Receiver<ServerMessage>,
    timeout: Duration,
    service: SccacheService<C>,
    wait: WaitUntilZero,
}

impl<C: CommandCreatorSync> SccacheServer<tokio::net::TcpListener, C> {
    pub fn new(
        port: u16,
        runtime: Runtime,
        client: Client,
        dist_client: DistClientContainer,
        storage: Arc<dyn Storage>,
        preprocessor_storage: Arc<dyn Storage>,
    ) -> Result<Self> {
        let addr = crate::net::SocketAddr::with_port(port);
        let listener = runtime.block_on(tokio::net::TcpListener::bind(addr.as_net().unwrap()))?;

        Ok(Self::with_listener(
            listener,
            runtime,
            client,
            dist_client,
            storage,
            preprocessor_storage,
        ))
    }
}

impl<A: crate::net::Acceptor, C: CommandCreatorSync> SccacheServer<A, C> {
    pub fn with_listener(
        listener: A,
        runtime: Runtime,
        client: Client,
        dist_client: DistClientContainer,
        storage: Arc<dyn Storage>,
        preprocessor_storage: Arc<dyn Storage>,
    ) -> Self {
        // Prepare the service which we'll use to service all incoming TCP
        // connections.
        let (tx, rx) = mpsc::channel(1);
        let (wait, info) = WaitUntilZero::new();
        let pool = runtime.handle().clone();
        let service = SccacheService::new(
            dist_client,
            storage,
            preprocessor_storage,
            &client,
            pool,
            tx,
            info,
        );

        SccacheServer {
            runtime,
            listener,
            rx,
            service,
            timeout: Duration::from_secs(get_idle_timeout()),
            wait,
        }
    }

    /// Configures how long this server will be idle before shutting down.
    #[allow(dead_code)]
    pub fn set_idle_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Set the storage this server will use.
    #[allow(dead_code)]
    pub fn set_storage(&mut self, storage: Arc<dyn Storage>) {
        self.service.storage = storage;
    }

    /// Returns a reference to a thread pool to run work on
    #[allow(dead_code)]
    pub fn pool(&self) -> &tokio::runtime::Handle {
        &self.service.rt
    }

    /// Returns a reference to the command creator this server will use
    #[allow(dead_code)]
    pub fn command_creator(&self) -> &C {
        &self.service.creator
    }

    /// Returns the port that this server is bound to
    #[allow(dead_code)]
    pub fn local_addr(&self) -> Option<crate::net::SocketAddr> {
        self.listener.local_addr().unwrap()
    }

    /// Runs this server to completion.
    ///
    /// If the `shutdown` future resolves then the server will be shut down,
    /// otherwise the server may naturally shut down if it becomes idle for too
    /// long anyway.
    pub fn run<F>(self, shutdown: F) -> io::Result<()>
    where
        F: Future,
        C: Send,
        A::Socket: 'static,
    {
        let SccacheServer {
            runtime,
            listener,
            rx,
            service,
            timeout,
            wait,
        } = self;

        let sysinfo = Arc::clone(&service.sysinfo);

        // Create our "server future" which will simply handle all incoming
        // connections in separate tasks.
        let server = async move {
            loop {
                let socket = listener.accept().await?;
                let conn = service.clone().bind(socket).map_err(|res| {
                    error!("Failed to bind socket: {res}");
                });

                // We're not interested if the task panicked; immediately process
                // another connection
                #[allow(clippy::let_underscore_future)]
                let _ = tokio::spawn(conn);
            }
        };

        // Right now there's a whole bunch of ways to shut down this server for
        // various purposes. These include:
        //
        // 1. The `shutdown` future above.
        // 2. An RPC indicating the server should shut down
        // 3. A period of inactivity (no requests serviced)
        //
        // These are all encapsulated with the future that we're creating below.
        // The `ShutdownOrInactive` indicates the RPC or the period of
        // inactivity, and this is then select'd with the `shutdown` future
        // passed to this function.

        let shutdown = shutdown.map(|_| {
            info!("shutting down due to explicit signal");
        });

        let shutdown_idle = async {
            ShutdownOrInactive {
                rx,
                timeout: if timeout != Duration::new(0, 0) {
                    Some(Box::pin(sleep(timeout)))
                } else {
                    None
                },
                timeout_dur: timeout,
            }
            .await;
            info!("shutting down due to being idle or request");
        };

        // Optionally log sysinfo (helpful in CI)
        let sys_info = async move {
            use crate::config::{bool_from_env_var, number_from_env_var};
            if !log_enabled!(log::Level::Info)
                || !matches!(bool_from_env_var("SCCACHE_LOG_SYSINFO"), Ok(Some(true)))
            {
                return futures::future::pending().await;
            }
            let interval = number_from_env_var("SCCACHE_LOG_SYSINFO_INTERVAL")
                .unwrap_or(Ok(1000))
                .unwrap_or(1000);
            loop {
                if let Ok(mut sys) = sysinfo.lock() {
                    sys.refresh_cpu_specifics(sysinfo::CpuRefreshKind::nothing().with_cpu_usage());
                    sys.refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());
                    info!(
                        "sysinfo: {{ cpu: {}%, mem: {{ total: {}b, available: {}b, used: {}b }} }}",
                        sys.global_cpu_usage(),
                        sys.total_memory(),
                        sys.available_memory(),
                        sys.used_memory(),
                    );
                }
                tokio::time::sleep(Duration::from_millis(interval)).await;
            }
        };

        runtime.block_on(async move {
            tokio::select! {
                server = server => server,
                _res = sys_info => Ok(()),
                _res = shutdown => Ok(()),
                _res = shutdown_idle => Ok::<_, io::Error>(()),
            }
        })?;

        const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
        info!(
            "moving into the shutdown phase now, waiting at most {} seconds \
             for all client requests to complete",
            SHUTDOWN_TIMEOUT.as_secs()
        );

        // Once our server has shut down either due to inactivity or a manual
        // request we still need to give a bit of time for all active
        // connections to finish. This `wait` future will resolve once all
        // instances of `SccacheService` have been dropped.
        //
        // Note that we cap the amount of time this can take, however, as we
        // don't want to wait *too* long.
        runtime.block_on(async { time::timeout(SHUTDOWN_TIMEOUT, wait).await })?;

        info!("ok, fully shutting down now");

        Ok(())
    }
}

/// Maps a compiler proxy path to a compiler proxy and it's last modification time
type CompilerProxyMap<C> = HashMap<PathBuf, (Box<dyn CompilerProxy<C>>, FileTime)>;
type CompilerMap<C> = HashMap<PathBuf, Option<CompilerCacheEntry<C>>>;

/// entry of the compiler cache
struct CompilerCacheEntry<C> {
    /// compiler argument trait obj
    pub compiler: Box<dyn Compiler<C>>,
    /// modification time of the compilers executable file
    pub mtime: FileTime,
    /// distributed compilation extra info
    pub dist_info: Option<(PathBuf, FileTime)>,
}

impl<C> CompilerCacheEntry<C> {
    fn new(
        compiler: Box<dyn Compiler<C>>,
        mtime: FileTime,
        dist_info: Option<(PathBuf, FileTime)>,
    ) -> Self {
        Self {
            compiler,
            mtime,
            dist_info,
        }
    }
}

#[derive(Clone)]
struct CompilerInfoFn<C: Send + Sync> {
    /// A cache of known compiler info.
    compilers: Arc<RwLock<CompilerMap<C>>>,

    /// map the cwd with compiler proxy path to a proxy resolver, which
    /// will dynamically resolve the input compiler for the current context
    /// (usually file or current working directory)
    /// the associated `FileTime` is the modification time of
    /// the compiler proxy, in order to track updates of the proxy itself
    compiler_proxies: Arc<RwLock<CompilerProxyMap<C>>>,

    // Cloned from SccacheService
    creator: C,
    dist_client: Arc<DistClientContainer>,
    rt: tokio::runtime::Handle,
}

#[async_trait]
impl<C> AsyncMulticastFunc<Compile, Box<dyn Compiler<C>>> for CompilerInfoFn<C>
where
    C: CommandCreatorSync + Clone + Send + Sync,
{
    async fn call(&self, compile: &Compile) -> Result<Box<dyn Compiler<C>>> {
        compiler_info(
            &self.compiler_proxies,
            &self.compilers,
            &self.creator,
            &self.dist_client,
            self.rt.clone(),
            compile,
        )
        .await
    }
}

impl util::AsyncMulticastArgs for Compile {
    type Key = String;
    fn hash(&self) -> String {
        [self.exe.as_encoded_bytes(), self.cwd.as_encoded_bytes()]
            .into_iter()
            .chain(self.args.iter().map(|arg| arg.as_encoded_bytes()))
            .chain(
                self.env_vars
                    .iter()
                    .flat_map(|(key, val)| [key.as_encoded_bytes(), val.as_encoded_bytes()]),
            )
            .fold(util::Digest::new(), |mut m, b| {
                m.update(b);
                m
            })
            .finish()
    }
}

#[derive(Clone, Default)]
pub struct SccacheGauge {
    value: Arc<AtomicU64>,
}

impl SccacheGauge {
    pub fn increment(&self) -> SccacheGaugeIncrement {
        self.increment_if(|_| true)
            .unwrap_or_else(|_| SccacheGaugeIncrement {
                value: self.value.clone(),
            })
    }

    pub fn increment_if<F>(&self, mut f: F) -> std::result::Result<SccacheGaugeIncrement, u64>
    where
        F: FnMut(u64) -> bool,
    {
        let value = self.value.clone();
        value
            .fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |value| {
                    if f(value) {
                        Some(value.saturating_add(1))
                    } else {
                        None
                    }
                },
            )
            .map(|_| SccacheGaugeIncrement { value })
    }

    pub fn value(&self) -> u64 {
        self.value.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[derive(Clone, Default)]
pub struct SccacheGaugeIncrement {
    value: Arc<AtomicU64>,
}

impl Drop for SccacheGaugeIncrement {
    fn drop(&mut self) {
        let _ = self.value.fetch_update(
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
            |value| Some(value.saturating_sub(1)),
        );
    }
}

/// Service implementation for sccache
#[derive(Clone)]
pub struct SccacheService<C>
where
    C: Send,
{
    /// Server statistics.
    pub stats: Arc<Mutex<ServerStats>>,

    /// Distributed sccache client
    dist_client: Arc<DistClientContainer>,

    /// Cache storage.
    storage: Arc<dyn Storage>,

    /// Preprocessor cache storage.
    preprocessor_storage: Arc<dyn Storage>,

    /// Limits the number of concurrent pending compilations
    pending_compilations_limit: u64,
    /// The count of currently queued compile requests.
    queued_compilations: Arc<SccacheGauge>,
    /// The count of currently pending compile requests.
    pending_compilations: Arc<SccacheGauge>,
    /// The count of currently active cache lookups.
    pending_cache_lookups: Arc<SccacheGauge>,
    /// The count of currently active compile requests.
    active_compilations: Arc<SccacheGauge>,

    /// Ensures asynchronous compiler_info lookups for the same compiler
    /// execute once and are cached
    compiler_info_queue: Arc<AsyncMulticast<Compile, Box<dyn Compiler<C>>>>,

    /// Task pool for blocking (used mostly for disk I/O-bound tasks) and
    // non-blocking tasks
    rt: tokio::runtime::Handle,

    /// An object for creating commands.
    ///
    /// This is mostly useful for unit testing, where we
    /// can mock this out.
    creator: C,

    /// Message channel used to learn about requests received by this server.
    ///
    /// Note that messages sent along this channel will keep the server alive
    /// (reset the idle timer) and this channel can also be used to shut down
    /// the entire server immediately via a message.
    tx: mpsc::Sender<ServerMessage>,

    /// Information tracking how many services (connected clients) are active.
    /// This field causes [WaitUntilZero] to wait until this struct drops.
    #[allow(dead_code)]
    info: ActiveInfo,

    // System info (CPU/mem usage)
    sysinfo: Arc<std::sync::Mutex<sysinfo::System>>,
}

impl<C> SccacheService<C>
where
    C: Send,
{
    pub fn active_compilations(&self) -> u64 {
        self.active_compilations.value()
    }

    pub fn increment_active_compilations(&self) -> SccacheGaugeIncrement {
        self.active_compilations.increment()
    }

    pub fn queued_compilations(&self) -> u64 {
        self.queued_compilations.value()
    }

    pub fn increment_queued_compilations(&self) -> SccacheGaugeIncrement {
        self.queued_compilations.increment()
    }

    pub fn pending_cache_lookups(&self) -> u64 {
        self.pending_cache_lookups.value()
    }

    pub fn increment_pending_cache_lookups(&self) -> SccacheGaugeIncrement {
        self.pending_cache_lookups.increment()
    }

    pub fn pending_compilations(&self) -> u64 {
        self.pending_compilations.value()
    }

    pub fn increment_pending_compilations(&self) -> SccacheGaugeIncrement {
        self.pending_compilations.increment()
    }
}

impl<C> std::fmt::Debug for SccacheService<C>
where
    C: Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.stats.fmt(f)
    }
}

type SccacheRequest = Message<Request, Body<()>>;
type SccacheResponse = Message<Response, Pin<Box<dyn Future<Output = Result<Response>> + Send>>>;

/// Messages sent from all services to the main event loop indicating activity.
///
/// Whenever a request is receive a `Request` message is sent which will reset
/// the idle shutdown timer, and otherwise a `Shutdown` message indicates that
/// a server shutdown was requested via an RPC.
pub enum ServerMessage {
    /// A message sent whenever a request is received.
    Request,
    /// Message sent whenever a shutdown request is received.
    Shutdown,
}

impl<C> Service<SccacheRequest> for Arc<SccacheService<C>>
where
    C: CommandCreatorSync + Send + Sync + 'static,
{
    type Response = SccacheResponse;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response>> + Send + 'static>>;

    fn call(&mut self, req: SccacheRequest) -> Self::Future {
        // Opportunistically let channel know that we've received a request. We
        // ignore failures here as well as backpressure as it's not imperative
        // that every message is received.
        drop(self.tx.clone().start_send(ServerMessage::Request));

        let me = self.clone();
        Box::pin(async move {
            match req.into_inner() {
                Request::Compile(compile) => {
                    me.stats.lock().await.compile_requests += 1;
                    Ok(me.handle_compile(compile).await)
                }
                Request::GetStats => me
                    .get_info()
                    .await
                    .map(|i| Response::Stats(Box::new(i)))
                    .map(Message::WithoutBody),
                Request::DistStatus => me
                    .get_dist_status()
                    .await
                    .map(Response::DistStatus)
                    .map(Message::WithoutBody),
                Request::ZeroStats => {
                    me.zero_stats().await;
                    Ok(Message::WithoutBody(Response::ZeroStats))
                }
                Request::Shutdown => {
                    let mut tx = me.tx.clone();
                    future::try_join(
                        async {
                            let _ = tx.send(ServerMessage::Shutdown).await;
                            Ok(())
                        },
                        me.get_info(),
                    )
                    .await
                    .map(move |(_, info)| {
                        Message::WithoutBody(Response::ShuttingDown(Box::new(info)))
                    })
                }
                Request::StorageHandshake { kind } => {
                    let storage = if kind == StorageKind::Compilations {
                        &me.storage
                    } else {
                        &me.preprocessor_storage
                    };

                    trace!("handle_client: storage_handshake kind={kind}");
                    let info = StorageHandshakeInfo {
                        location: storage.location().await,
                        cache_type_name: storage.cache_type_name().to_owned(),
                        basedirs: storage.basedirs().to_vec(),
                        preprocessor_cache_mode_config: storage.preprocessor_cache_mode_config(),
                        cache_mode: storage.check().await.unwrap_or(CacheMode::ReadWrite),
                        max_size: storage.max_size().await.unwrap_or(None),
                    };
                    Ok(Message::WithoutBody(Response::StorageHandshake(info)))
                }
                Request::StorageGetPath { key, kind } => {
                    trace!("handle_client: storage_get_path kind={kind}, key={key}");
                    let storage = if kind == StorageKind::Compilations {
                        &me.storage
                    } else {
                        &me.preprocessor_storage
                    };
                    Ok(Message::WithoutBody(Response::StorageGetPath(
                        storage.get_path(&key).await,
                    )))
                }
                Request::StorageDelPath { key, kind } => {
                    trace!("handle_client: storage_del_path kind={kind}, key={key}");
                    let storage = if kind == StorageKind::Compilations {
                        &me.storage
                    } else {
                        &me.preprocessor_storage
                    };
                    Ok(Message::WithoutBody(Response::StorageDelPath(
                        storage.del(&key).await.map_err(|e| format!("{e:#}")),
                    )))
                }
                Request::StorageGetRaw { key, kind } => {
                    trace!("handle_client: storage_get kind={kind}, key={key}");
                    let storage = if kind == StorageKind::Compilations {
                        &me.storage
                    } else {
                        &me.preprocessor_storage
                    };
                    let resp = match storage.get(&key).await {
                        Ok(Cache::Hit(data)) => Response::StorageGetRaw(Some(data.to_vec())),
                        Ok(Cache::Miss) => Response::StorageGetRaw(None),
                        Err(e) => {
                            warn!("storage_get error: {e:#}");
                            Response::StorageGetRaw(None)
                        }
                    };
                    Ok(Message::WithoutBody(resp))
                }
                Request::StoragePutRaw { key, data, kind } => {
                    trace!("handle_client: storage_put kind={kind}, key={key}");
                    let storage = if kind == StorageKind::Compilations {
                        &me.storage
                    } else {
                        &me.preprocessor_storage
                    };
                    let result = storage
                        .put(&key, data.into())
                        .await
                        .map(|_| ())
                        .map_err(|e| format!("{e:#}"));
                    Ok(Message::WithoutBody(Response::StoragePutRaw(result)))
                }
                Request::RecordStats(delta) => {
                    trace!("handle_client: record_stats");
                    me.merge_stats(*delta).await;
                    Ok(Message::WithoutBody(Response::RecordStats))
                }
            }
        })
    }

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }
}

fn init_sysinfo() -> sysinfo::System {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

    System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::nothing().with_cpu_usage())
            .with_memory(MemoryRefreshKind::nothing().with_ram()),
    )
}

use futures::TryStreamExt;

impl<C> SccacheService<C>
where
    C: CommandCreatorSync + Clone + Send + Sync + 'static,
{
    pub fn new(
        dist_client: DistClientContainer,
        storage: Arc<dyn Storage>,
        preprocessor_storage: Arc<dyn Storage>,
        client: &Client,
        rt: tokio::runtime::Handle,
        tx: mpsc::Sender<ServerMessage>,
        info: ActiveInfo,
    ) -> SccacheService<C> {
        let creator = C::new(client);
        let dist_client = Arc::new(dist_client);
        let compiler_info_queue = Arc::new(AsyncMulticast::new(CompilerInfoFn {
            compiler_proxies: Default::default(),
            compilers: Default::default(),
            creator: creator.clone(),
            dist_client: Arc::clone(&dist_client),
            rt: rt.clone(),
        }));
        SccacheService {
            stats: Arc::default(),
            queued_compilations: Arc::default(),
            pending_compilations: Arc::default(),
            pending_cache_lookups: Arc::default(),
            active_compilations: Arc::default(),
            dist_client,
            storage,
            preprocessor_storage,
            pending_compilations_limit: util::num_cpus() as u64,
            compiler_info_queue,
            rt,
            creator,
            tx,
            info,
            sysinfo: Arc::new(std::sync::Mutex::new(init_sysinfo())),
        }
    }

    pub fn mock_with_storage(
        storage: Arc<dyn Storage>,
        preprocessor_storage: Arc<dyn Storage>,
        rt: tokio::runtime::Handle,
    ) -> SccacheService<C> {
        let (tx, _) = mpsc::channel(1);
        let (_, info) = WaitUntilZero::new();
        let client = Client::new_num(1);
        let creator = C::new(&client);
        let dist_client = Arc::new(DistClientContainer::new_disabled());
        let compiler_info_queue = Arc::new(AsyncMulticast::new(CompilerInfoFn {
            compiler_proxies: Default::default(),
            compilers: Default::default(),
            creator: creator.clone(),
            dist_client: Arc::clone(&dist_client),
            rt: rt.clone(),
        }));
        SccacheService {
            stats: Arc::default(),
            queued_compilations: Arc::default(),
            pending_compilations: Arc::default(),
            pending_cache_lookups: Arc::default(),
            active_compilations: Arc::default(),
            dist_client,
            storage,
            preprocessor_storage,
            pending_compilations_limit: util::num_cpus() as u64,
            compiler_info_queue,
            rt,
            creator,
            tx,
            info,
            sysinfo: Arc::new(std::sync::Mutex::new(init_sysinfo())),
        }
    }

    #[cfg(feature = "dist-client")]
    pub fn mock_with_dist_client(
        dist_client: Arc<dyn dist::Client>,
        storage: Arc<dyn Storage>,
        preprocessor_storage: Arc<dyn Storage>,
        rt: tokio::runtime::Handle,
    ) -> SccacheService<C> {
        let (tx, _) = mpsc::channel(1);
        let (_, info) = WaitUntilZero::new();
        let client = Client::new_num(1);
        let creator = C::new(&client);
        let dist_client = Arc::new(DistClientContainer::new_with_state(DistClientState::Some(
            Box::new(DistClientConfig {
                scheduler_url: None,
                auth: config::DistAuth::Token {
                    token: String::new(),
                },
                fallback_to_local_compile: true,
                max_retries: 0f64,
                net: Default::default(),
                cache_dir: "".into(),
                toolchain_cache_size: 0,
                toolchains: vec![],
                rewrite_includes_only: false,
            }),
            dist_client,
        )));
        let compiler_info_queue = Arc::new(AsyncMulticast::new(CompilerInfoFn {
            compiler_proxies: Default::default(),
            compilers: Default::default(),
            creator: creator.clone(),
            dist_client: Arc::clone(&dist_client),
            rt: rt.clone(),
        }));
        SccacheService {
            stats: Arc::default(),
            queued_compilations: Arc::default(),
            pending_compilations: Arc::default(),
            pending_cache_lookups: Arc::default(),
            active_compilations: Arc::default(),
            dist_client,
            storage,
            preprocessor_storage,
            pending_compilations_limit: util::num_cpus() as u64,
            compiler_info_queue,
            rt: rt.clone(),
            creator,
            tx,
            info,
            sysinfo: Arc::new(std::sync::Mutex::new(init_sysinfo())),
        }
    }

    async fn bind<T>(self, socket: T) -> Result<()>
    where
        T: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let mut builder = length_delimited::Builder::new();
        if let Ok(max_frame_length_str) = env::var("SCCACHE_MAX_FRAME_LENGTH") {
            if let Ok(max_frame_length) = max_frame_length_str.parse::<usize>() {
                builder.max_frame_length(max_frame_length);
            } else {
                warn!("Content of SCCACHE_MAX_FRAME_LENGTH is not a valid number, using default");
            }
        }
        let io = builder.new_framed(socket);

        let (sink, mut stream) = SccacheTransport {
            inner: Framed::new(io.sink_err_into().err_into(), BincodeCodec),
        }
        .split();
        let mut sink = sink.sink_err_into::<Error>();

        let (incoming, mut requests) = tokio::sync::mpsc::unbounded_channel();

        let me = Arc::new(self);

        let _handle = util::spawn(async move {
            while let Some(req) = requests.recv().await {
                match req {
                    Ok(req) => {
                        let res = match util::spawn(me.clone().call(req)).await? {
                            Ok(res) => res,
                            Err(err) => {
                                return Err(err);
                            }
                        };
                        match res {
                            Message::WithoutBody(message) => {
                                sink.send(Frame::Message { message }).await?;
                            }
                            Message::WithBody(message, body) => {
                                sink.send(Frame::Message { message }).await?;
                                sink.send(Frame::Body {
                                    chunk: Some(util::spawn(body).await??),
                                })
                                .await?;
                                sink.send(Frame::Body { chunk: None }).await?;
                            }
                        }
                    }
                    Err(err) => {
                        return Err(err);
                    }
                }
            }

            Ok(())
        });

        while let Some(req) = stream.next().await {
            match req {
                Ok(req) => {
                    incoming.send(Ok(req))?;
                }
                Err(err) => {
                    return Err(err);
                }
            }
        }

        Ok(())
    }

    /// Get dist status.
    async fn get_dist_status(&self) -> Result<DistInfo> {
        Ok(self.dist_client.get_status().await)
    }

    /// Get info and stats about the cache.
    async fn get_info(&self) -> Result<ServerInfo> {
        let stats = ServerStats {
            active_compilations: self.active_compilations(),
            queued_compilations: self.queued_compilations(),
            pending_cache_lookups: self.pending_cache_lookups(),
            pending_compilations: self.pending_compilations(),
            ..self.stats.lock().await.clone()
        };
        ServerInfo::new(
            stats,
            Some(&*self.storage),
            Some(&*self.preprocessor_storage),
        )
        .await
    }

    /// Zero stats about the cache.
    async fn zero_stats(&self) {
        *self.stats.lock().await = ServerStats::default();
    }

    /// Snapshot and reset the current stats (used by client-side processes before exit).
    pub async fn take_stats(&self) -> ServerStats {
        let mut s = self.stats.lock().await;
        std::mem::take(&mut *s)
    }

    async fn merge_stats(&self, delta: ServerStats) {
        *self.stats.lock().await += delta;
    }

    /// Handle a compile request from a client.
    ///
    /// This will handle a compile request entirely, generating a response with
    /// the initial information and an optional body which will eventually
    /// contain the results of the compilation.
    async fn handle_compile(&self, mut compile: Compile) -> SccacheResponse {
        let pending = if cfg!(not(feature = "dist-client")) {
            self.increment_pending_compilations()
        } else {
            // Limit the number of concurrent compile requests that are in the
            // pre-compilation stage to `self.num_cpus`.
            //
            // `stats.pending_compilations` is the number of compile requests that have
            // been submitted to the server but have not started compiling yet. In other
            // words, compile requests that are in the preprocess or `generate_hash_key`
            // stages (collectively, the pre-compilation stages).
            //
            // When the client is configured for local compilation, this has no impact
            // on job throughput, since subprocess invocation is already guarded by the
            // jobserver's semaphore.
            //
            // If the client is configured for distributed compilation, this gives the
            // client time to send the `new_job`/`run_job` requests to the build cluster
            // as soon as they're done with the pre-compilation stages.
            //
            // Without this, all the incoming pre-compilation tasks are interleaved,
            // because tokio's scheduler is fair/breadth-first. The implication is all
            // the compile requests are preprocessed at roughly the same time, then all
            // their hash keys are computed, then all cache lookups occur, and finally
            // all the cache misses are compiled.
            //
            // This is a problem in massively parallel (`make -j$(ulimit -n)`)
            // distributed compilations -- especially with CUDA in the mix, because that
            // fans out into potentially many parallel subcompilations.
            //
            // We don't want to peg the client's CPU at ~100% preprocessing every file,
            // generating all hash keys, whipping up a cache lookup request storm, etc.
            // before any jobs are sent to the build cluster.
            //
            // Jobs should be sent to the build cluster as soon as we know they're not
            // in the cache. Then we can handle more compile requests while the build
            // cluster is busy.
            //
            // So this loop is effectively a semaphore, but not implemented with an
            // actual Semaphore because we need to introduce a short a delay before
            // "unlocking".
            //
            // Why is that? Good question.
            //
            // The "lock" for a compile request is "released" just before the first
            // async call of a compilation phase starts. But with a CUDA compilation,
            // a re-entrant subcompilation is scheduled, which should take precedence
            // over (and continue to block) processing another outer compile request.
            //
            // Delaying a short amount of time here is an approximation of this logic.
            // It's unlikely the CUDA subcompilation will land exactly on a timeout
            // boundary and be scheduled after the next outer compile request. But
            // even if a few sometimes do, most won't, which is good enough.
            let _queued = self.increment_queued_compilations();
            loop {
                match self
                    .pending_compilations
                    .increment_if(|pending_compilations| {
                        pending_compilations < self.pending_compilations_limit
                    }) {
                    Ok(pending) => {
                        break pending;
                    }
                    Err(_) => {
                        // Jitter so not every task retries at the same time
                        tokio::time::sleep(tokio_retry2::strategy::jitter_range(1.0, 2.0)(
                            // Make this configurable?
                            Duration::from_millis(250),
                        ))
                        .await;
                    }
                }
            }
        };

        let args = compile.args;
        compile.args = compiler_info_args(&args);

        match self.compiler_info_queue.call(compile).await {
            Ok((compile, compiler)) => {
                self.check_compile(
                    compiler,
                    args,
                    compile.cwd.into(),
                    compile.env_vars,
                    pending,
                )
                .await
            }
            Err(err) => {
                warn!("handle_compile: Unsupported compiler: {err}");
                drop(pending);
                let mut stats = self.stats.lock().await;
                stats.requests_unsupported_compiler += 1;
                Message::WithoutBody(Response::Compile(CompileResponse::UnsupportedCompiler(
                    err.to_string().into(),
                )))
            }
        }
    }

    /// Run a compile entirely in the current process (used in client-side mode).
    ///
    /// Returns the `CompileResponse` variant and, when compilation started, the
    /// accompanying `CompileFinished` result.
    pub async fn compile_direct(
        &self,
        compile: Compile,
    ) -> Result<(CompileResponse, Option<CompileFinished>)> {
        match self.handle_compile(compile).await {
            Message::WithBody(Response::Compile(resp), body) => {
                let finished = match body.await? {
                    Response::CompileFinished(f) => f,
                    _ => bail!("unexpected body response from compile_direct"),
                };
                Ok((resp, Some(finished)))
            }
            Message::WithoutBody(Response::Compile(resp)) => Ok((resp, None)),
            _ => bail!("unexpected response from handle_compile in compile_direct"),
        }
    }

    /// Look up compiler info from the cache for the compiler `path`.
    /// If not cached, determine the compiler type and cache the result.
    pub async fn compiler_info(
        &self,
        exe: &Path,
        cwd: &Path,
        args: &[OsString],
        env_vars: &[(OsString, OsString)],
    ) -> Result<Box<dyn Compiler<C>>> {
        // lookup if compiler proxy exists for the current compiler path
        self.compiler_info_queue
            .call(Compile {
                exe: exe.into(),
                cwd: cwd.into(),
                args: compiler_info_args(args),
                env_vars: env_vars.to_vec(),
            })
            .await
            .map(|(_, compiler)| compiler)
    }

    /// Check that we can handle and cache `cmd` when run with `compiler`.
    /// If so, run `start_compile_task` to execute it.
    async fn check_compile(
        &self,
        compiler: Box<dyn Compiler<C>>,
        cmd: Vec<OsString>,
        cwd: PathBuf,
        env_vars: Vec<(OsString, OsString)>,
        pending: SccacheGaugeIncrement,
    ) -> SccacheResponse {
        // Check that we can handle this compiler with the provided commandline.
        match compiler.parse_arguments(&cmd, &cwd, &env_vars) {
            CompilerArguments::Ok(hasher) => {
                let body = self
                    .clone()
                    .start_compile_task(compiler, hasher, cmd, cwd, env_vars, pending)
                    .and_then(|res| async move { Ok(Response::CompileFinished(res)) })
                    .boxed();

                return Message::WithBody(Response::Compile(CompileResponse::CompileStarted), body);
            }
            CompilerArguments::CannotCache(why, extra_info) => {
                if let Some(extra_info) = extra_info {
                    debug!("parse_arguments: CannotCache({why}, {extra_info}): {cmd:?}");
                } else {
                    debug!("parse_arguments: CannotCache({why}): {cmd:?}");
                }
                let mut stats = self.stats.lock().await;
                stats.requests_not_cacheable += 1;
                *stats.not_cached.entry(why.to_string()).or_insert(0) += 1;
            }
            CompilerArguments::NotCompilation => {
                debug!("parse_arguments: NotCompilation: {cmd:?}");
                self.stats.lock().await.requests_not_compile += 1;
            }
        }

        let res = CompileResponse::UnhandledCompile;
        Message::WithoutBody(Response::Compile(res))
    }

    /// Given compiler arguments `arguments`, look up
    /// a compile result in the cache or execute the compilation and store
    /// the result in the cache.
    pub async fn start_compile_task(
        self,
        compiler: Box<dyn Compiler<C>>,
        hasher: Box<dyn CompilerHasher<C>>,
        arguments: Vec<OsString>,
        cwd: PathBuf,
        env_vars: Vec<(OsString, OsString)>,
        pending: SccacheGaugeIncrement,
    ) -> Result<CompileFinished> {
        self.stats.lock().await.requests_executed += 1;

        let force_recache = env_vars.iter().any(|(k, v)| {
            k == "SCCACHE_RECACHE" && matches!(v.to_str().unwrap_or_default(), "true" | "on" | "1")
        });
        let force_no_cache = env_vars.iter().any(|(k, v)| {
            k == "SCCACHE_NO_CACHE" && matches!(v.to_str().unwrap_or_default(), "true" | "on" | "1")
        });

        let cache_control = if force_no_cache {
            CacheControl::ForceNoCache
        } else if force_recache {
            CacheControl::ForceRecache
        } else {
            CacheControl::Default
        };

        let out_pretty = hasher.output_pretty().into_owned();
        let color_mode = hasher.color_mode();

        let kind = compiler.kind();
        let lang = hasher.language();

        let me = self.clone();

        util::spawn_on(&self.rt, async move {
            let result = match me.dist_client.get_client().await {
                Ok(client) => {
                    std::panic::AssertUnwindSafe(hasher
                        .get_cached_or_compile(
                            &me,
                            client,
                            me.creator.clone(),
                            me.storage.clone(),
                            me.preprocessor_storage.clone(),
                            arguments,
                            cwd,
                            env_vars,
                            cache_control,
                            me.rt.clone(),
                            pending,
                    ))
                    .catch_unwind()
                    .await
                    .map_err(|e| {
                        let panic = e
                            .downcast_ref::<&str>()
                            .map(|s| &**s)
                            .or_else(|| e.downcast_ref::<String>().map(|s| &**s))
                            .unwrap_or("An unknown panic was caught.");
                        let thread = std::thread::current();
                        let thread_name = thread.name().unwrap_or("unnamed");
                        if let Some((file, line, column)) = PANIC_LOCATION.with(|l| l.take()) {
                            anyhow!(
                                "thread '{thread_name}' panicked at {file}:{line}:{column}: {panic}"
                            )
                        } else {
                            anyhow!("thread '{thread_name}' panicked: {panic}")
                        }
                    })
                    .and_then(std::convert::identity)
                },
                Err(e) => Err(e),
            };

            let mut cache_write = None;
            let mut res = CompileFinished {
                color_mode,
                ..Default::default()
            };

            let mut stats = me.stats.lock().await;

            match result {
                Ok((compiled, output)) => {
                    let mut dist_type = DistType::NoDist;

                    match compiled {
                        CompileResult::Error(duration) => {
                            trace!("[{out_pretty}]: compile result: request error");
                            stats.compiler_write_duration += duration;
                            stats.compile_request_errors.increment(&kind, &lang);
                        }
                        CompileResult::CacheHit(duration) => {
                            trace!("[{out_pretty}]: compile result: cache hit");

                            stats.cache_hits.increment(&kind, &lang);
                            stats.cache_read_hit_duration += duration;
                        }
                        CompileResult::CacheMiss(miss_type, dt, duration, future) => {
                            trace!("[{out_pretty}]: compile result: cache miss");
                            dist_type = dt;

                            match miss_type {
                                MissType::Normal => {}
                                MissType::ForcedNoCache => {}
                                MissType::ForcedRecache => {
                                    stats.forced_recaches += 1;
                                }
                                MissType::TimedOut => {
                                    stats.cache_timeouts += 1;
                                }
                                MissType::CacheReadError => {
                                    stats.cache_read_errors += 1;
                                }
                            }
                            stats.compilations += 1;
                            stats.cache_misses.increment(&kind, &lang);
                            stats.compiler_write_duration += duration;
                            trace!("[{out_pretty}]: stats after compile result: {stats:?}");
                            cache_write = Some(future);
                        }
                        CompileResult::NotCached(dt, duration) => {
                            trace!("[{out_pretty}]: compile result: not cached");
                            dist_type = dt;
                            stats.compilations += 1;
                            stats.compiler_write_duration += duration;
                        }
                        CompileResult::NotCacheable(dt, duration) => {
                            trace!("[{out_pretty}]: compile result: not cacheable");
                            dist_type = dt;
                            stats.non_cacheable_compilations += 1;
                            stats.compiler_write_duration += duration;
                        }
                        CompileResult::CompileFailed(dt, duration) => {
                            trace!("[{out_pretty}]: compile result: compile failed");
                            debug!("[{out_pretty}]: Compilation failed: {output:?}");
                            dist_type = dt;
                            stats.compile_fails += 1;
                            stats.compiler_write_duration += duration;
                        }
                    }

                    match dist_type {
                        DistType::NoDist => {}
                        DistType::Ok(server_id) => {
                            stats.dist_compiles.entry(server_id).and_modify(|c| *c += 1).or_insert(1);
                        }
                        DistType::Error => stats.dist_errors += 1,
                    }

                    // Make sure the write guard has been dropped ASAP.
                    drop(stats);

                    res.output = output;
                }
                Err(err) => {

                    let mut reset_dist_client = false;

                    res.output = Err(err)
                        .or_else(|err| {
                            err.downcast::<HttpClientError>().map(|err| {
                                reset_dist_client = true;
                                error!("[{out_pretty}]: HTTP {err}");
                                let stderr = format!("sccache: HTTP {err}");
                                ProcessOutput::new(1, vec![], stderr.into_bytes())
                            })
                        })
                        .or_else(|err| {
                            #[cfg(feature = "dist-client")]
                            return err.downcast::<DistClientError>().map(|err| {
                                let stderr = "sccache: distributed compilation failed and local compile disabled\n";
                                error!("[{out_pretty}]: {err}");
                                stats.dist_errors += 1;
                                ProcessOutput::new(
                                    //TODO: figure out a better way to communicate this?
                                    -1,
                                    vec![],
                                    stderr.to_string().into_bytes()
                                )
                            });
                            #[cfg(not(feature = "dist-client"))]
                            return Err(err);
                        })
                        .or_else(|err| {
                            err.downcast::<ProcessError>().map(|ProcessError(output)| {
                                debug!("[{out_pretty}]: Compilation failed: {output:?}");
                                stats.compile_fails += 1;
                                output
                            })
                        })
                        .or_else(|err| {
                            stats.fatal_errors.increment(&kind, &lang);
                            use std::fmt::Write;
                            error!("[{out_pretty}] fatal error: {err:?}");
                            let mut stderr = "sccache: encountered fatal error\n".to_string();
                            let _ = writeln!(stderr, "sccache: error: {err}");
                            for err in err.chain() {
                                error!("[{out_pretty}] {err:?}");
                                let _ = writeln!(stderr, "sccache: caused by: {err}");
                            }
                            Ok::<ProcessOutput, Error>(
                                //TODO: figure out a better way to communicate this?
                                ProcessOutput::new(-1, vec![], stderr.into_bytes())
                            )
                        })?;

                    // Make sure the write guard has been dropped ASAP.
                    drop(stats);

                    if reset_dist_client {
                        me.dist_client.reset_state().await;
                    }
                }
            }

            if let Some(cache_write) = cache_write {
                match cache_write.await {
                    Err(e) => {
                        warn!("[{out_pretty}]: Error executing cache write: {e}");
                        me.stats.lock().await.cache_write_errors += 1;
                    }
                    //TODO: save cache stats!
                    Ok(info) => {
                        trace!(
                            "[{}]: Cache write finished in {}",
                            info.object_file_pretty,
                            util::fmt_duration_as_secs(&info.duration)
                        );
                        let mut stats = me.stats.lock().await;
                        stats.cache_writes += 1;
                        stats.cache_write_duration += info.duration;
                    }
                }
            }

            Ok(res)
        })
        .map_err(anyhow::Error::new)
        .await?
    }
}

/// Look up compiler info from the cache for the compiler `path`.
/// If not cached, determine the compiler type and cache the result.
#[allow(clippy::too_many_arguments)]
async fn compiler_info<C>(
    compiler_proxies: &RwLock<CompilerProxyMap<C>>,
    compilers: &RwLock<CompilerMap<C>>,
    creator: &C,
    dist_client: &DistClientContainer,
    rt: tokio::runtime::Handle,
    compile: &Compile,
) -> Result<Box<dyn Compiler<C>>>
where
    C: CommandCreatorSync + Send + Sync,
{
    // lookup if compiler proxy exists for the current compiler path
    let path = Path::new(&compile.exe);
    let cwd = Path::new(&compile.cwd);
    let args = &compile.args;
    let env = compile.env_vars.clone();

    let resolved_with_proxy = {
        let compiler_proxies_borrow = compiler_proxies.read().await;
        // Create an owned future - compiler proxy is not Send so we can't
        // really await while borrowing the proxy since rustc is too conservative
        let resolve_proxied_executable =
            compiler_proxies_borrow
                .get(path)
                .map(|(compiler_proxy, _filetime)| {
                    compiler_proxy.resolve_proxied_executable(
                        creator.clone(),
                        cwd.to_owned(),
                        env.as_slice(),
                    )
                });

        match resolve_proxied_executable {
            Some(fut) => fut.await.ok(),
            None => None,
        }
    };

    // use the supplied compiler path as fallback, lookup its modification time too
    let (resolved_compiler_path, mtime) = match resolved_with_proxy {
        Some(x) => x, // TODO resolve the path right away
        _ => {
            // fallback to using the path directly
            metadata(path)
                .map(|attr| FileTime::from_last_modification_time(&attr))
                .ok()
                .map(move |filetime| (path.to_owned(), filetime))
                .expect("Must contain sane data, otherwise mtime is not avail")
        }
    };

    // canonicalize the path to follow symlinks
    // don't canonicalize if the file name differs so it works with clang's multicall
    let resolved_compiler_path = match dunce::canonicalize(&resolved_compiler_path) {
        Ok(path) if matches!(path.file_name(), Some(name) if resolved_compiler_path.file_name() == Some(name)) => {
            path
        }
        _ => resolved_compiler_path,
    };

    let dist_info = match dist_client.get_client().await {
        Ok(Some(ref client)) => {
            if let Some(archive) = client.get_custom_toolchain(&resolved_compiler_path).await {
                match metadata(&archive).map(|attr| FileTime::from_last_modification_time(&attr)) {
                    Ok(mtime) => Some((archive, mtime)),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    };

    let opt = match compilers.read().await.get(&resolved_compiler_path) {
        // It's a hit only if the mtime and dist archive data matches.
        Some(Some(entry)) => {
            if entry.mtime == mtime && entry.dist_info == dist_info {
                Some(entry.compiler.box_clone())
            } else {
                None
            }
        }
        _ => None,
    };

    match opt {
        Some(info) => Ok(info),
        None => {
            // Check the compiler type and return the result when
            // finished. This generally involves invoking the compiler,
            // so do it asynchronously.

            // the compiler path might be compiler proxy, so it is important to use
            // `path` (or its clone `path1`) to resolve using that one, not using `resolved_compiler_path`
            let info = get_compiler_info::<C>(
                creator.clone(),
                path,
                cwd,
                args,
                env.as_slice(),
                &rt,
                dist_info.clone().map(|(p, _)| p),
            )
            .await;

            let (c, proxy) = match info {
                Ok((c, proxy)) => (c.clone(), proxy.clone()),
                Err(err) => {
                    trace!("Inserting PLAIN cache map info for {:?}", &path);
                    compilers.write().await.insert(path.to_owned(), None);
                    return Err(err);
                }
            };

            // register the proxy for this compiler, so it will be used directly from now on
            // and the true/resolved compiler will create table hits in the hash map
            // based on the resolved path
            if let Some(proxy) = proxy {
                trace!(
                    "Inserting new path proxy {:?} @ {:?} -> {:?}",
                    &path, &cwd, resolved_compiler_path
                );
                compiler_proxies
                    .write()
                    .await
                    .insert(path.to_owned(), (proxy, mtime));
            }
            // TODO add some safety checks in case a proxy exists, that the initial `path` is not
            // TODO the same as the resolved compiler binary

            // cache
            let map_info = CompilerCacheEntry::new(c.clone(), mtime, dist_info);
            trace!(
                "Inserting POSSIBLY PROXIED cache map info for {:?}",
                &resolved_compiler_path
            );
            compilers
                .write()
                .await
                .insert(resolved_compiler_path, Some(map_info));

            // drop the proxy information, response is compiler only
            Ok(c)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct PerLanguageCount {
    counts: HashMap<String, u64>,
    adv_counts: HashMap<String, u64>,
}

impl PerLanguageCount {
    pub fn increment(&mut self, kind: &CompilerKind, lang: &Language) {
        let lang_comp_key = kind.lang_comp_kind(lang);
        let adv_count = self.adv_counts.entry(lang_comp_key).or_insert(0);
        *adv_count += 1;

        let lang_key = kind.lang_kind(lang);
        let count = self.counts.entry(lang_key).or_insert(0);
        *count += 1;
    }

    pub fn all(&self) -> u64 {
        self.counts.values().sum()
    }

    pub fn get(&self, key: &str) -> Option<&u64> {
        self.counts.get(key)
    }

    pub fn get_adv(&self, key: &str) -> Option<&u64> {
        self.adv_counts.get(key)
    }

    pub fn new() -> PerLanguageCount {
        Self::default()
    }
}

impl std::ops::AddAssign for PerLanguageCount {
    fn add_assign(&mut self, rhs: Self) {
        for (k, v) in rhs.counts {
            *self.counts.entry(k).or_default() += v;
        }
        for (k, v) in rhs.adv_counts {
            *self.adv_counts.entry(k).or_default() += v;
        }
    }
}

/// Statistics about the server.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ServerStats {
    /// The count of currently active compile requests.
    pub active_compilations: u64,
    /// The count of currently queued compile requests.
    pub queued_compilations: u64,
    /// The count of currently pending compile requests.
    pub pending_compilations: u64,
    /// The count of currently active cache lookups.
    pub pending_cache_lookups: u64,
    /// The count of client compile requests.
    pub compile_requests: u64,
    /// The count of errors handling compile requests (per language).
    pub compile_request_errors: PerLanguageCount,
    /// The count of client requests that used an unsupported compiler.
    pub requests_unsupported_compiler: u64,
    /// The count of client requests that were not compilation.
    pub requests_not_compile: u64,
    /// The count of client requests that were not cacheable.
    pub requests_not_cacheable: u64,
    /// The count of client requests that were executed.
    pub requests_executed: u64,
    /// The count of cache hits for handled compile requests (per language).
    pub cache_hits: PerLanguageCount,
    /// The count of preprocessor cache hits for handled compile requests (per language).
    pub preprocessor_cache_hits: PerLanguageCount,
    /// The count of cache misses for handled compile requests (per language).
    pub cache_misses: PerLanguageCount,
    /// The count of preprocessor cache misses for handled compile requests (per language).
    pub preprocessor_cache_misses: PerLanguageCount,
    /// The count of cache misses because the cache took too long to respond.
    pub cache_timeouts: u64,
    /// The count of errors reading cache entries.
    pub cache_read_errors: u64,
    /// The count of compilations which were successful but couldn't be cached.
    pub non_cacheable_compilations: u64,
    /// The count of compilations which forcibly ignored the cache.
    pub forced_recaches: u64,
    /// The count of errors writing to cache.
    pub cache_write_errors: u64,
    /// The number of successful cache writes.
    pub cache_writes: u64,
    /// The total time spent writing cache entries.
    pub cache_write_duration: Duration,
    /// The total time spent reading cache hits.
    pub cache_read_hit_duration: Duration,
    /// The number of files successful preprocessor calls.
    pub preprocessed: u64,
    /// The number of successful compilations performed.
    pub compilations: u64,
    /// The total time spent verifying preprocessor cache hits.
    pub preprocessor_cache_hit_duration: Duration,
    /// The total time spent verifying preprocessor cache misses.
    pub preprocessor_cache_miss_duration: Duration,
    /// The total time spent preprocessing.
    pub preprocessor_duration: Duration,
    /// The total time spent compiling.
    pub compiler_write_duration: Duration,
    /// The count of compilation failures.
    pub compile_fails: u64,
    /// Counts of reasons why compiles were not cached.
    pub not_cached: HashMap<String, usize>,
    /// The count of compilations that were successfully distributed indexed
    /// by the server that ran those compilations.
    pub dist_compiles: HashMap<String, usize>,
    /// The count of compilations that were distributed but failed and had to be re-run locally
    pub dist_errors: u64,
    /// The count of sccache fatal errors (per language).
    pub fatal_errors: PerLanguageCount,
    /// Multi-level cache statistics (if multi-level caching is enabled)
    pub cache_multi_level: Option<MultiLevelStats>,
    /// Multi-level preprocessor cache statistics (if multi-level caching is enabled)
    pub preprocessor_multi_level: Option<MultiLevelStats>,
}

impl std::ops::AddAssign for ServerStats {
    fn add_assign(&mut self, rhs: Self) {
        self.active_compilations += rhs.active_compilations;
        self.queued_compilations += rhs.queued_compilations;
        self.pending_compilations += rhs.pending_compilations;
        self.pending_cache_lookups += rhs.pending_cache_lookups;
        self.compile_requests += rhs.compile_requests;
        self.compile_request_errors += rhs.compile_request_errors;
        self.requests_unsupported_compiler += rhs.requests_unsupported_compiler;
        self.requests_not_compile += rhs.requests_not_compile;
        self.requests_not_cacheable += rhs.requests_not_cacheable;
        self.requests_executed += rhs.requests_executed;
        self.cache_hits += rhs.cache_hits;
        self.preprocessor_cache_hits += rhs.preprocessor_cache_hits;
        self.cache_misses += rhs.cache_misses;
        self.preprocessor_cache_misses += rhs.preprocessor_cache_misses;
        self.cache_timeouts += rhs.cache_timeouts;
        self.cache_read_errors += rhs.cache_read_errors;
        self.non_cacheable_compilations += rhs.non_cacheable_compilations;
        self.forced_recaches += rhs.forced_recaches;
        self.cache_write_errors += rhs.cache_write_errors;
        self.cache_writes += rhs.cache_writes;
        self.cache_write_duration += rhs.cache_write_duration;
        self.cache_read_hit_duration += rhs.cache_read_hit_duration;
        self.preprocessed += rhs.preprocessed;
        self.compilations += rhs.compilations;
        self.preprocessor_cache_hit_duration += rhs.preprocessor_cache_hit_duration;
        self.preprocessor_cache_miss_duration += rhs.preprocessor_cache_miss_duration;
        self.preprocessor_duration += rhs.preprocessor_duration;
        self.compiler_write_duration += rhs.compiler_write_duration;
        self.compile_fails += rhs.compile_fails;
        for (k, v) in rhs.not_cached {
            *self.not_cached.entry(k).or_default() += v;
        }
        for (k, v) in rhs.dist_compiles {
            *self.dist_compiles.entry(k).or_default() += v;
        }
        self.dist_errors += rhs.dist_errors;
        self.fatal_errors += rhs.fatal_errors;
        self.cache_multi_level = match (self.cache_multi_level.take(), rhs.cache_multi_level) {
            (Some(mut a), Some(b)) => {
                a += b;
                Some(a)
            }
            (a, None) => a,
            (None, b) => b,
        };
        self.preprocessor_multi_level = match (
            self.preprocessor_multi_level.take(),
            rhs.preprocessor_multi_level,
        ) {
            (Some(mut a), Some(b)) => {
                a += b;
                Some(a)
            }
            (a, None) => a,
            (None, b) => b,
        };
    }
}

/// Info and stats about the server.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ServerInfo {
    pub stats: ServerStats,
    pub cache_location: String,
    pub cache_size: Option<u64>,
    pub max_cache_size: Option<u64>,
    pub preprocessor_cache_location: String,
    pub preprocessor_cache_size: Option<u64>,
    pub preprocessor_max_cache_size: Option<u64>,
    pub use_preprocessor_cache_mode: bool,
    pub version: String,
    pub basedirs: Vec<String>,
}

/// Status of the dist client.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DistInfo {
    Disabled(String),
    #[cfg(feature = "dist-client")]
    NotConnected(Option<config::HTTPUrl>, String),
    #[cfg(feature = "dist-client")]
    SchedulerStatus(Option<config::HTTPUrl>, dist::SchedulerStatus),
}

impl Default for ServerStats {
    fn default() -> ServerStats {
        ServerStats {
            active_compilations: u64::default(),
            queued_compilations: u64::default(),
            pending_compilations: u64::default(),
            pending_cache_lookups: u64::default(),
            compile_requests: u64::default(),
            compile_request_errors: PerLanguageCount::new(),
            requests_unsupported_compiler: u64::default(),
            requests_not_compile: u64::default(),
            requests_not_cacheable: u64::default(),
            requests_executed: u64::default(),
            cache_hits: PerLanguageCount::new(),
            preprocessor_cache_hits: PerLanguageCount::new(),
            cache_misses: PerLanguageCount::new(),
            preprocessor_cache_misses: PerLanguageCount::new(),
            cache_timeouts: u64::default(),
            cache_read_errors: u64::default(),
            non_cacheable_compilations: u64::default(),
            forced_recaches: u64::default(),
            cache_write_errors: u64::default(),
            cache_writes: u64::default(),
            cache_write_duration: Duration::new(0, 0),
            cache_read_hit_duration: Duration::new(0, 0),
            preprocessed: u64::default(),
            compilations: u64::default(),
            preprocessor_cache_hit_duration: Duration::new(0, 0),
            preprocessor_cache_miss_duration: Duration::new(0, 0),
            preprocessor_duration: Duration::new(0, 0),
            compiler_write_duration: Duration::new(0, 0),
            compile_fails: u64::default(),
            not_cached: HashMap::new(),
            dist_compiles: HashMap::new(),
            dist_errors: u64::default(),
            fatal_errors: PerLanguageCount::new(),
            cache_multi_level: None,
            preprocessor_multi_level: None,
        }
    }
}

pub trait ServerStatsWriter {
    fn write(&mut self, text: &str);
}

pub struct StdoutServerStatsWriter;

impl ServerStatsWriter for StdoutServerStatsWriter {
    fn write(&mut self, text: &str) {
        println!("{text}");
    }
}

impl ServerStats {
    /// Print stats in a human-readable format.
    ///
    /// Return the formatted width of each of the (name, value) columns.
    fn print<T: ServerStatsWriter>(
        &self,
        writer: &mut T,
        advanced: bool,
        use_preprocessor_cache_mode: bool,
    ) -> (usize, usize) {
        macro_rules! set_stat {
            ($vec:ident, $var:expr, $name:expr) => {{
                // name, value, suffix length
                $vec.push(($name.to_string(), $var.to_string(), 0));
            }};
        }

        macro_rules! set_lang_stat {
            ($vec:ident, $var:expr, $name:expr) => {{
                $vec.push(($name.to_string(), $var.all().to_string(), 0));
                let mut sorted_stats: Vec<_> = $var.counts.iter().collect();
                sorted_stats.sort_by_key(|v| v.0);
                for (lang, count) in sorted_stats.iter() {
                    $vec.push((format!("{} ({})", $name, lang), count.to_string(), 0));
                }
            }};
        }
        macro_rules! set_compiler_stat {
            ($vec:ident, $var:expr, $name:expr) => {{
                $vec.push(($name.to_string(), $var.all().to_string(), 0));
                let mut sorted_stats: Vec<_> = $var.adv_counts.iter().collect();
                sorted_stats.sort_by_key(|v| v.0);
                for (lang, count) in sorted_stats.iter() {
                    $vec.push((format!("{} ({})", $name, lang), count.to_string(), 0));
                }
            }};
        }
        macro_rules! set_counted_stat {
            ($vec:ident, $var:expr, $name:expr) => {{
                if advanced {
                    set_compiler_stat!($vec, $var, $name);
                } else {
                    set_lang_stat!($vec, $var, $name);
                }
            }};
        }

        macro_rules! set_duration_stat {
            ($vec:ident, $dur:expr, $num:expr, $name:expr) => {{
                let s = if $num > 0 {
                    $dur / $num as u32
                } else {
                    Default::default()
                };
                // name, value, suffix length
                $vec.push(($name.to_string(), util::fmt_duration_as_secs(&s), 2));
            }};
        }

        let mut stats_vec = vec![];
        //TODO: this would be nice to replace with a custom derive implementation.
        set_stat!(stats_vec, self.queued_compilations, "Queued compilations");
        set_stat!(stats_vec, self.pending_compilations, "Pending compilations");
        set_stat!(
            stats_vec,
            self.pending_cache_lookups,
            "Pending cache lookups"
        );
        set_stat!(stats_vec, self.active_compilations, "Active compilations");
        set_stat!(stats_vec, self.compile_requests, "Compile requests");
        set_stat!(
            stats_vec,
            self.requests_executed,
            "Compile requests executed"
        );
        set_counted_stat!(
            stats_vec,
            self.compile_request_errors,
            "Compile request errors"
        );
        set_counted_stat!(stats_vec, self.cache_hits, "Cache hits");
        if use_preprocessor_cache_mode {
            set_counted_stat!(
                stats_vec,
                self.preprocessor_cache_hits,
                "Preprocessor cache hits"
            );
        }
        set_counted_stat!(stats_vec, self.cache_misses, "Cache misses");
        if use_preprocessor_cache_mode {
            set_counted_stat!(
                stats_vec,
                self.preprocessor_cache_misses,
                "Preprocessor cache misses"
            );
        }

        self.set_percentage_stats(&mut stats_vec, advanced);

        set_stat!(stats_vec, self.cache_timeouts, "Cache timeouts");
        set_stat!(stats_vec, self.cache_read_errors, "Cache read errors");
        set_stat!(stats_vec, self.forced_recaches, "Forced recaches");
        set_stat!(stats_vec, self.cache_write_errors, "Cache write errors");

        set_stat!(
            stats_vec,
            self.compilations + self.compile_fails + self.non_cacheable_compilations,
            "Compilations"
        );
        set_stat!(stats_vec, self.compile_fails, "Compilation failures");

        set_stat!(
            stats_vec,
            self.non_cacheable_compilations,
            "Non-cacheable compilations"
        );
        set_stat!(
            stats_vec,
            self.requests_not_cacheable,
            "Non-cacheable calls"
        );
        set_stat!(
            stats_vec,
            self.requests_not_compile,
            "Non-compilation calls"
        );
        set_stat!(
            stats_vec,
            self.requests_unsupported_compiler,
            "Unsupported compiler calls"
        );
        set_duration_stat!(
            stats_vec,
            self.cache_read_hit_duration,
            self.cache_hits.all(),
            "Average cache read hit"
        );
        set_duration_stat!(
            stats_vec,
            self.cache_write_duration,
            self.cache_writes,
            "Average cache write"
        );
        if use_preprocessor_cache_mode && self.preprocessor_cache_hits.all() > 0 {
            set_duration_stat!(
                stats_vec,
                self.preprocessor_cache_hit_duration,
                self.preprocessor_cache_hits.all(),
                "Average preprocessor cache hit"
            );
        }
        if use_preprocessor_cache_mode && self.preprocessor_cache_misses.all() > 0 {
            set_duration_stat!(
                stats_vec,
                self.preprocessor_cache_miss_duration,
                self.preprocessor_cache_misses.all(),
                "Average preprocessor cache miss"
            );
        }
        if self.preprocessed > 0 {
            set_duration_stat!(
                stats_vec,
                self.preprocessor_duration,
                self.preprocessed,
                "Average preprocessor"
            );
        }
        set_duration_stat!(
            stats_vec,
            self.compiler_write_duration,
            self.compilations
                + self.compile_fails
                + self.non_cacheable_compilations
                + self.compile_request_errors.all(),
            "Average compiler"
        );
        set_counted_stat!(stats_vec, self.fatal_errors, "Fatal errors");
        set_stat!(
            stats_vec,
            self.dist_compiles.values().sum::<usize>(),
            "Successful distributed compiles"
        );
        set_stat!(
            stats_vec,
            self.dist_errors,
            "Failed distributed compilations"
        );

        let mut name_width: usize = 0;
        let mut stat_width: usize = 0;

        macro_rules! write_stats_vec {
            ($vec:ident) => {{
                name_width = name_width.max($vec.iter().map(|(n, _, _)| n.len()).max().unwrap());
                stat_width = stat_width.max($vec.iter().map(|(_, s, _)| s.len()).max().unwrap());
                for (name, stat, suffix_len) in $vec.drain(..) {
                    writer.write(&format!(
                        "{:<name_width$} {:>stat_width$}",
                        name,
                        stat,
                        name_width = name_width,
                        stat_width = stat_width + suffix_len
                    ));
                }
            }};
        }

        write_stats_vec!(stats_vec);

        // Add multi-level cache statistics if available
        if let Some(ref ml_stats) = self.cache_multi_level {
            let mut stats_vec = vec![];
            for (name, value, suffix_type) in ml_stats.format_stats() {
                stats_vec.push((name, value, suffix_type));
            }
            write_stats_vec!(stats_vec);
        }

        if let Some(ref ml_stats) = self.preprocessor_multi_level {
            let mut stats_vec = vec![];
            let ml_stats = ml_stats.format_stats();
            for (idx, (name, value, suffix_type)) in ml_stats.iter().enumerate() {
                let name = if idx == 0 {
                    "Multi-level preprocessor cache levels"
                } else {
                    name.as_str()
                };
                stats_vec.push((name, value, suffix_type));
            }
            write_stats_vec!(stats_vec);
        }

        // Compare values. If equal, compare keys to have a fully deterministic order.
        let sort_func =
            |(k1, v1): &(&String, &usize), (k2, v2): &(&String, &usize)| match v1.cmp(v2).reverse()
            {
                std::cmp::Ordering::Equal => k1.cmp(k2),
                other => other,
            };

        if !self.not_cached.is_empty() {
            writer.write("\nNon-cacheable reasons:");
            let mut counts: Vec<_> = self.not_cached.iter().collect();
            counts.sort_by(sort_func);
            for (reason, count) in counts {
                writer.write(&format!("{reason:<name_width$} {count:>stat_width$}",));
            }
            writer.write("");
        }
        (name_width, stat_width)
    }

    fn print_dist<T: ServerStatsWriter>(
        &self,
        writer: &mut T,
        name_width: usize,
        stat_width: usize,
    ) {
        // Compare values. If equal, compare keys to have a fully deterministic order.
        let sort_func =
            |(k1, v1): &(&String, &usize), (k2, v2): &(&String, &usize)| match v1.cmp(v2).reverse()
            {
                std::cmp::Ordering::Equal => k1.cmp(k2),
                other => other,
            };

        if !self.dist_compiles.is_empty() {
            writer.write("\nSuccessful distributed compiles");
            let mut counts: Vec<_> = self.dist_compiles.iter().collect();
            counts.sort_by(sort_func);
            for (reason, count) in counts {
                writer.write(&format!(
                    "  {:<name_width$} {:>stat_width$}",
                    reason,
                    count,
                    name_width = name_width - 2,
                    stat_width = stat_width,
                ));
            }
        }
    }

    fn set_percentage_stats(&self, stats_vec: &mut Vec<(String, String, usize)>, advanced: bool) {
        set_percentage_stat(
            stats_vec,
            self.cache_hits.all(),
            self.cache_misses.all() + self.cache_hits.all(),
            "Cache hits rate",
        );

        let (stats_hits, stats_misses): (Vec<_>, Vec<_>) = if advanced {
            (
                self.cache_hits.adv_counts.iter().collect(),
                self.cache_misses.adv_counts.iter().collect(),
            )
        } else {
            (
                self.cache_hits.counts.iter().collect(),
                self.cache_misses.counts.iter().collect(),
            )
        };

        let mut all_languages: HashSet<&String> = HashSet::new();
        for (lang, _) in &stats_hits {
            all_languages.insert(lang);
        }
        for (lang, _) in &stats_misses {
            all_languages.insert(lang);
        }

        let mut all_languages: Vec<&String> = all_languages.into_iter().collect();
        all_languages.sort();

        for lang in all_languages {
            let count_hits = stats_hits
                .iter()
                .find(|&&(l, _)| l == lang)
                .map_or(0, |&(_, &count)| count);

            let count_misses = stats_misses
                .iter()
                .find(|&&(l, _)| l == lang)
                .map_or(0, |&(_, &count)| count);

            set_percentage_stat(
                stats_vec,
                count_hits,
                count_hits + count_misses,
                &format!("Cache hits rate ({lang})"),
            );
        }
    }
}

fn set_percentage_stat(
    vec: &mut Vec<(String, String, usize)>,
    count_hits: u64,
    total: u64,
    name: &str,
) {
    if total == 0 {
        vec.push((name.to_string(), "-".to_string(), 0));
    } else {
        let ratio = count_hits as f64 / total as f64;
        vec.push((name.to_string(), format!("{:.2} %", ratio * 100.0), 2));
    }
}

impl ServerInfo {
    pub async fn new(
        stats: ServerStats,
        storage: Option<&dyn Storage>,
        preprocessor_storage: Option<&dyn Storage>,
    ) -> Result<Self> {
        use crate::cache::multilevel::MultiLevelStats;
        async fn storage_info(
            storage: Option<&dyn Storage>,
        ) -> (
            Option<u64>,
            Option<u64>,
            String,
            Option<MultiLevelStats>,
            Vec<String>,
        ) {
            storage
                .map(|s| {
                    futures::future::try_join5(
                        s.current_size(),
                        s.max_size(),
                        async { Ok(s.location().await) },
                        async { Ok(s.multilevel_stats()) },
                        async {
                            Ok(s.basedirs()
                                .iter()
                                .map(|p| String::from_utf8_lossy(p).to_string())
                                .collect())
                        },
                    )
                    .boxed()
                })
                .unwrap_or_else(|| async { Ok((None, None, String::new(), None, vec![])) }.boxed())
                .await
                .unwrap_or_else(|_| (None, None, String::new(), None, vec![]))
        }

        let (cache_size, max_cache_size, cache_location, cache_multi_level, basedirs) =
            storage_info(storage).await;
        let (
            preprocessor_cache_size,
            preprocessor_max_cache_size,
            preprocessor_cache_location,
            preprocessor_multi_level,
            _,
        ) = storage_info(preprocessor_storage).await;

        let use_preprocessor_cache_mode = preprocessor_storage
            .map(|s| {
                s.preprocessor_cache_mode_config()
                    .use_preprocessor_cache_mode
            })
            .unwrap_or_default();

        let version = env!("CARGO_PKG_VERSION").to_string();
        Ok(ServerInfo {
            stats: ServerStats {
                cache_multi_level,
                preprocessor_multi_level,
                ..stats
            },
            cache_location,
            cache_size,
            max_cache_size,
            preprocessor_cache_location,
            preprocessor_cache_size,
            preprocessor_max_cache_size,
            use_preprocessor_cache_mode,
            version,
            basedirs,
        })
    }

    /// Print info to stdout in a human-readable format.
    pub fn print(&self, advanced: bool) {
        let (name_width, stat_width) = self.stats.print(
            &mut StdoutServerStatsWriter,
            advanced,
            self.use_preprocessor_cache_mode,
        );
        println!(
            "{:<name_width$} {}",
            "Cache location:",
            self.cache_location,
            name_width = name_width
        );
        if let Some(ref ml_stats) = self.stats.cache_multi_level {
            for level in &ml_stats.0 {
                println!(
                    "{:<name_width$} {}",
                    format!("  {}", level.name),
                    level.location,
                    name_width = name_width
                );
            }
        }

        println!(
            "{:<name_width$} {}",
            "Use direct/preprocessor mode?",
            if self.use_preprocessor_cache_mode {
                "yes"
            } else {
                "no"
            },
            name_width = name_width
        );

        if self.use_preprocessor_cache_mode {
            println!(
                "{:<name_width$} {}",
                "Preprocessor cache location:",
                self.preprocessor_cache_location,
                name_width = name_width
            );
            if let Some(ref ml_stats) = self.stats.preprocessor_multi_level {
                for level in &ml_stats.0 {
                    println!(
                        "{:<name_width$} {}",
                        format!("  {}", level.name),
                        level.location,
                        name_width = name_width
                    );
                }
            }
        }
        println!(
            "{:<name_width$} {}",
            "Base directories",
            if self.basedirs.is_empty() {
                "(none)".to_string()
            } else {
                self.basedirs.join(", ")
            },
            name_width = name_width
        );
        println!(
            "{:<name_width$} {}",
            "Version (client)",
            self.version,
            name_width = name_width
        );
        for &(name, val) in &[
            ("Cache size", &self.cache_size),
            ("Max cache size", &self.max_cache_size),
        ] {
            if let Some(val) = *val {
                let (val, suffix) = match NumberPrefix::binary(val as f64) {
                    NumberPrefix::Standalone(bytes) => (bytes.to_string(), "bytes".to_string()),
                    NumberPrefix::Prefixed(prefix, n) => (format!("{n:.0}"), format!("{prefix}B")),
                };
                println!("{name:<name_width$} {val:>stat_width$} {suffix}");
            }
        }
        if advanced {
            self.stats
                .print_dist(&mut StdoutServerStatsWriter, name_width, stat_width);
        }
    }
}

enum Frame<R, R1> {
    Body { chunk: Option<R1> },
    Message { message: R },
}

struct Body<R> {
    receiver: mpsc::Receiver<Result<R>>,
}

impl<R> futures::Stream for Body<R> {
    type Item = Result<R>;
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver).poll_next(cx)
    }
}

enum Message<R, B> {
    WithBody(R, B),
    WithoutBody(R),
}

impl<R, B> Message<R, B> {
    fn into_inner(self) -> R {
        match self {
            Message::WithBody(r, _) => r,
            Message::WithoutBody(r) => r,
        }
    }
}

struct BincodeCodec;
impl<T> tokio_serde::Serializer<T> for BincodeCodec
where
    T: serde::Serialize,
{
    type Error = Error;

    fn serialize(self: Pin<&mut Self>, item: &T) -> std::result::Result<Bytes, Self::Error> {
        let mut bytes = BytesMut::new();
        bincode::serialize_into((&mut bytes).writer(), item)?;
        Ok(bytes.freeze())
    }
}

impl<T> tokio_serde::Deserializer<T> for BincodeCodec
where
    T: serde::de::DeserializeOwned,
{
    type Error = Error;

    fn deserialize(self: Pin<&mut Self>, buf: &BytesMut) -> std::result::Result<T, Self::Error> {
        let ret = bincode::deserialize(buf)?;
        Ok(ret)
    }
}

/// Implementation of `Stream + Sink` that tokio-proto is expecting
///
/// This type is composed of a few layers:
///
/// * First there's `I`, the I/O object implementing `AsyncRead` and
///   `AsyncWrite`
/// * Next that's framed using the `length_delimited` module in tokio-io giving
///   us a `Sink` and `Stream` of `BytesMut`.
/// * Next that sink/stream is wrapped in `ReadBincode` which will cause the
///   `Stream` implementation to switch from `BytesMut` to `Request` by parsing
///   the bytes  bincode.
/// * Finally that sink/stream is wrapped in `WriteBincode` which will cause the
///   `Sink` implementation to switch from `BytesMut` to `Response` meaning that
///   all `Response` types pushed in will be converted to `BytesMut` and pushed
///   below.
struct SccacheTransport<I: AsyncRead + AsyncWrite + Unpin> {
    inner: Framed<
        futures::stream::ErrInto<
            futures::sink::SinkErrInto<
                tokio_util::codec::Framed<I, LengthDelimitedCodec>,
                Bytes,
                Error,
            >,
            Error,
        >,
        Request,
        Response,
        BincodeCodec,
    >,
}

impl<I: AsyncRead + AsyncWrite + Unpin> Stream for SccacheTransport<I> {
    type Item = Result<Message<Request, Body<()>>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner)
            .poll_next(cx)
            .map(|r| r.map(|s| s.map(Message::WithoutBody)))
    }
}

impl<I: AsyncRead + AsyncWrite + Unpin> Sink<Frame<Response, Response>> for SccacheTransport<I> {
    type Error = Error;

    fn poll_ready(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<()>> {
        Pin::new(&mut self.inner).poll_ready(cx)
    }

    fn start_send(mut self: Pin<&mut Self>, item: Frame<Response, Response>) -> Result<()> {
        match item {
            Frame::Message { message } => Pin::new(&mut self.inner).start_send(message),
            Frame::Body { chunk: Some(chunk) } => Pin::new(&mut self.inner).start_send(chunk),
            Frame::Body { chunk: None } => Ok(()),
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<()>> {
        Pin::new(&mut self.inner).poll_close(cx)
    }
}

struct ShutdownOrInactive {
    rx: mpsc::Receiver<ServerMessage>,
    timeout: Option<Pin<Box<Sleep>>>,
    timeout_dur: Duration,
}

impl Future for ShutdownOrInactive {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        loop {
            match Pin::new(&mut self.rx).poll_next(cx) {
                Poll::Pending => break,
                // Shutdown received!
                Poll::Ready(Some(ServerMessage::Shutdown)) => return Poll::Ready(()),
                Poll::Ready(Some(ServerMessage::Request)) => {
                    if self.timeout_dur != Duration::new(0, 0) {
                        self.timeout = Some(Box::pin(sleep(self.timeout_dur)));
                    }
                }
                // All services have shut down, in theory this isn't possible...
                Poll::Ready(None) => return Poll::Ready(()),
            }
        }
        match self.timeout {
            None => Poll::Pending,
            Some(ref mut timeout) => timeout.as_mut().poll(cx),
        }
    }
}

/// Helper future which tracks the `ActiveInfo` below. This future will resolve
/// once all instances of `ActiveInfo` have been dropped.
pub(crate) struct WaitUntilZero {
    info: std::sync::Weak<std::sync::Mutex<Info>>,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ActiveInfo {
    info: Arc<std::sync::Mutex<Info>>,
}

struct Info {
    waker: Option<Waker>,
}

impl Drop for Info {
    fn drop(&mut self) {
        if let Some(waker) = self.waker.as_ref() {
            waker.wake_by_ref();
        }
    }
}

impl WaitUntilZero {
    #[rustfmt::skip]
    pub(crate) fn new() -> (WaitUntilZero, ActiveInfo) {
        let info = Arc::new(std::sync::Mutex::new(Info { waker: None }));

        (WaitUntilZero { info: Arc::downgrade(&info) }, ActiveInfo { info })
    }
}

impl std::future::Future for WaitUntilZero {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> std::task::Poll<Self::Output> {
        match self.info.upgrade() {
            None => std::task::Poll::Ready(()),
            Some(arc) => {
                let mut info = arc.lock().expect("we can't panic when holding lock");
                info.waker = Some(cx.waker().clone());
                std::task::Poll::Pending
            }
        }
    }
}

#[test]
fn waits_until_zero() {
    let (wait, _active) = WaitUntilZero::new();
    assert_eq!(wait.now_or_never(), None);

    let (wait, active) = WaitUntilZero::new();
    let _active2 = active.clone();
    drop(active);
    assert_eq!(wait.now_or_never(), None);

    let (wait, _) = WaitUntilZero::new();
    assert_eq!(wait.now_or_never(), Some(()));

    let (wait, active) = WaitUntilZero::new();
    let active2 = active.clone();
    drop(active);
    drop(active2);
    assert_eq!(wait.now_or_never(), Some(()));
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StringWriter {
        buffer: String,
    }

    impl StringWriter {
        fn new() -> StringWriter {
            StringWriter {
                buffer: String::new(),
            }
        }

        fn get_output(self) -> String {
            self.buffer
        }
    }

    impl ServerStatsWriter for StringWriter {
        fn write(&mut self, text: &str) {
            self.buffer.push_str(&format!("{text}\n"));
        }
    }

    #[test]
    fn test_print_cache_hits_rate_default_server_stats() {
        let stats = ServerStats::default();

        let mut writer = StringWriter::new();
        stats.print(&mut writer, false, true);

        let output = writer.get_output();

        assert!(output.contains("Cache hits rate                       -"));
    }

    #[test]
    fn test_print_cache_hits_rate_server_stats() {
        let mut cache_hits_counts = HashMap::new();
        cache_hits_counts.insert("Rust".to_string(), 100);
        cache_hits_counts.insert("C/C++".to_string(), 200);

        let mut cache_misses_counts = HashMap::new();
        cache_misses_counts.insert("Rust".to_string(), 50);
        cache_misses_counts.insert("Cuda".to_string(), 300);

        let stats = ServerStats {
            cache_hits: PerLanguageCount {
                counts: cache_hits_counts,
                ..Default::default()
            },
            cache_misses: PerLanguageCount {
                counts: cache_misses_counts,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut writer = StringWriter::new();
        stats.print(&mut writer, false, true);

        let output = writer.get_output();

        assert!(output.contains("Cache hits rate                    46.15 %"));
        assert!(output.contains("Cache hits rate (C/C++)           100.00 %"));
        assert!(output.contains("Cache hits rate (Cuda)              0.00 %"));
        assert!(output.contains("Cache hits rate (Rust)             66.67 %"));
    }

    #[test]
    fn test_print_cache_hits_rate_advanced_server_stats() {
        let mut cache_hits_counts = HashMap::new();
        cache_hits_counts.insert("rust".to_string(), 50);
        cache_hits_counts.insert("c/c++ [clang]".to_string(), 30);

        let mut cache_misses_counts = HashMap::new();
        cache_misses_counts.insert("rust".to_string(), 100);
        cache_misses_counts.insert("cuda".to_string(), 70);

        let stats = ServerStats {
            cache_hits: PerLanguageCount {
                adv_counts: cache_hits_counts,
                ..Default::default()
            },
            cache_misses: PerLanguageCount {
                adv_counts: cache_misses_counts,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut writer = StringWriter::new();
        stats.print(&mut writer, true, true);

        let output = writer.get_output();

        assert!(output.contains("Cache hits rate                        -"));
        assert!(output.contains("Cache hits rate (c/c++ [clang])   100.00 %"));
        assert!(output.contains("Cache hits rate (cuda)              0.00 %"));
        assert!(output.contains("Cache hits rate (rust)             33.33 %"));
    }

    // Test that 2 servers with the same hits will always be printed in the same order.
    // This test will **randomly** (not consistently) fail in case of a regression, due to HashMap iteration behaviour.
    #[test]
    fn test_print_deterministic_hits() {
        // Test a bunch of time to make the test fail often enough
        for _ in 0..5 {
            let mut stats = ServerStats::default();
            stats.dist_compiles.insert("server1".to_string(), 10);
            stats.dist_compiles.insert("server2".to_string(), 10);

            let mut writer = StringWriter::new();
            let (name_width, stat_width) = stats.print(&mut writer, true, false);
            stats.print_dist(&mut writer, name_width, stat_width);

            // Check that the order is deterministic
            let output = writer.get_output();
            let lines: Vec<&str> = output.lines().collect();
            let find_s1 = lines
                .iter()
                .position(|line| line.contains("server1"))
                .unwrap();
            let find_s2 = lines
                .iter()
                .position(|line| line.contains("server2"))
                .unwrap();
            assert!(find_s1 < find_s2);
        }
    }
}
