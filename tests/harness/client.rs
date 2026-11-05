use super::{
    TC_CACHE_SIZE, TestDir, cargo_bin, prune_command, sccache_bin, sccache_debug, write_json_cfg,
};
#[cfg(any(unix, target_env = "msvc"))]
use super::{
    compilers::{Compiler, find_compiler},
    init_logger,
};
use fs_err as fs;
use sccache::{
    compiler::SourceFileDeps,
    config::{self, CacheType, ClientConfig},
    server::{ServerInfo, ServerStats},
};
use std::{
    env,
    ffi::{OsStr, OsString},
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str,
    sync::atomic::{AtomicU16, Ordering},
};

/// A client for a server with a local disk cache, configured by a file in a
/// new temporary directory. Compile the test sources in that directory too.
/// The server isn't started yet.
pub fn make_sccache_client() -> (TestDir, SccacheClient) {
    let tempdir = TestDir::new("sccache_system_test");
    let tempdir_path = tempdir.path();

    let sccache_cfg = sccache_client_cfg(
        tempdir_path,
        sccache::config::defaults::default_disk_cache_size(),
    );
    write_json_cfg(tempdir_path, "sccache-cfg.json", &sccache_cfg);
    let client = SccacheClient::new(
        &tempdir_path.join("sccache-cfg.json"),
        &tempdir_path.join("sccache-cached-cfg"),
    );

    (tempdir, client)
}

/// Runs `test` with a new local server, if `compiler` is installed.
#[cfg(any(unix, target_env = "msvc"))]
pub fn with_local_server<T>(
    compiler: &str,
    test: impl FnOnce(&SccacheClient, &Compiler, &Path) -> T,
) -> Option<T> {
    init_logger();
    let compiler = find_compiler(compiler)?;
    let (tempdir, client) = make_sccache_client();
    Some(test(&client.start(), &compiler, tempdir.path()))
}

pub fn sccache_client_cfg(data_dir: &Path, size: u64) -> ClientConfig {
    let disk_cache_dir = data_dir.join("cache");
    let toolchains_dir = data_dir.join("toolchains");
    fs::create_dir_all(&disk_cache_dir).unwrap();
    fs::create_dir_all(&toolchains_dir).unwrap();

    let disk = config::cache::Disk {
        dir: disk_cache_dir.clone(),
        size,
        ..Default::default()
    };

    ClientConfig {
        cache: vec![disk.clone().into()].into(),
        dist: config::client::dist::Config {
            auth: Default::default(), // dangerously_insecure
            url: None,
            cache_dir: toolchains_dir,
            toolchains: vec![],
            toolchain_cache_size: TC_CACHE_SIZE,
            net: config::client::dist::Networking {
                max_connections: 1,
                connect_timeout: 10,
                request_timeout: 300,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

// Every client talks to its own server, so tests can run concurrently.
static CLIENT_PORT: AtomicU16 = AtomicU16::new(4227);

pub struct SccacheClient {
    envvars: Vec<(OsString, OsString)>,
    pub path: PathBuf,
}

impl SccacheClient {
    pub fn new_no_cfg() -> Self {
        let path = sccache_bin().into();
        let port = CLIENT_PORT.fetch_add(1, Ordering::SeqCst);

        let mut envvars = vec![
            ("SCCACHE_SERVER_PORT".into(), port.to_string().into()),
            ("SCCACHE_THREADS".into(), "4".into()),
        ];

        // Send daemon logs to a file if SCCACHE_DEBUG is defined
        if sccache_debug() {
            envvars.extend_from_slice(&[
                // Allow overriding log level
                (
                    "SCCACHE_SERVER_LOG".into(),
                    env::var_os("SCCACHE_SERVER_LOG")
                        .or(env::var_os("SCCACHE_LOG"))
                        .unwrap_or("sccache=trace".into()),
                ),
                // Allow overriding log output path
                (
                    "SCCACHE_ERROR_LOG".into(),
                    env::var_os("SCCACHE_ERROR_LOG").unwrap_or(
                        env::temp_dir()
                            .join(format!("sccache_local_daemon.{port}.txt"))
                            .into_os_string(),
                    ),
                ),
            ]);
        }

        Self { envvars, path }
    }

    pub fn new(cfg_path: &Path, cached_cfg_path: &Path) -> Self {
        Self::new_no_cfg()
            .with_env("SCCACHE_CONF", cfg_path)
            .with_env("SCCACHE_CACHED_CONF", cached_cfg_path)
    }

    pub fn with_env<K: AsRef<OsStr>, V: AsRef<OsStr>>(mut self, key: K, val: V) -> Self {
        self.envvars
            .push((key.as_ref().into(), val.as_ref().into()));
        self
    }

    pub fn start(self) -> Self {
        self.start_server();
        self
    }

    /// Restarts the server on the same port, with `vars` replacing any
    /// earlier values.
    pub fn restart_with_env(&mut self, vars: &[(&str, &OsStr)]) {
        self.envvars
            .extend(vars.iter().map(|(k, v)| (k.into(), v.into())));
        self.start_server();
    }

    fn start_server(&self) {
        // Servers don't time out, so one left on this port by an interrupted
        // run would keep this one from starting.
        self.stop();
        trace!("sccache --start-server");
        // Don't run this with run() because on Windows `wait_with_output`
        // will hang because the internal server process is not detached.
        if !self
            .cmd()
            .arg("--start-server")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
        {
            panic!("Failed to start local daemon");
        }
    }

    pub fn stop(&self) -> bool {
        trace!("sccache --stop-server");
        self.cmd()
            .arg("--stop-server")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    /// `program` with this client's environment instead of the test's
    /// `SCCACHE_*` variables, so any sccache it runs uses this server.
    pub fn command<S: AsRef<OsStr>>(&self, program: S) -> Command {
        let mut cmd = prune_command(Command::new(program));
        cmd.envs(
            self.envvars
                .iter()
                .map(|(k, v)| (k.as_os_str(), v.as_os_str())),
        );
        cmd
    }

    pub fn cmd(&self) -> Command {
        self.command(&self.path)
    }

    /// `cargo`, with this client as the rustc wrapper.
    pub fn cargo(&self) -> Command {
        let mut cmd = self.command(cargo_bin());
        cmd.env("RUSTC_WRAPPER", &self.path);
        cmd
    }

    pub fn info(&self) -> sccache::errors::Result<ServerInfo> {
        self.cmd()
            .args(["--show-stats", "--stats-format=json"])
            .output()
            .map_err(anyhow::Error::new)
            .map_err(|e| e.context("`sccache --show-stats --stats-format=json` failed"))
            .map(|output| {
                let s = str::from_utf8(&output.stdout).expect("Output not UTF-8");
                serde_json::from_str(s).expect("Failed to parse JSON stats")
            })
    }

    pub fn stats(&self) -> sccache::errors::Result<ServerStats> {
        self.info().map(|info| info.stats)
    }

    pub fn zero_stats(&self) {
        trace!("sccache --zero-stats");
        drop(
            self.cmd()
                .arg("--zero-stats")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status(),
        );
    }

    pub fn config(&self) -> ClientConfig {
        ClientConfig::load_from_envs(
            self.envvars
                .iter()
                .map(|(k, v)| (k.as_os_str(), v.as_os_str())),
        )
        .unwrap_or_default()
    }

    /// Whether the server sends compilations to a dist scheduler.
    pub fn is_dist(&self) -> bool {
        self.config().dist.url.is_some()
    }

    pub fn compilations_cache(&self) -> config::cache::Disk {
        self.config()
            .cache
            .configs
            .iter()
            .find_map(|(_, c)| match c {
                CacheType::Disk(c) => Some(c.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    pub fn clear_disk_cache(&self) -> sccache::errors::Result<PathBuf> {
        let disk_cache = self.compilations_cache();
        let disk_cache_dir = disk_cache.dir;
        println!("clear_disk_cache '{}'", disk_cache_dir.display());
        for path in "0123456789abcdef"
            .chars()
            .map(|c| disk_cache_dir.join(String::from(c)))
        {
            if path.is_dir() {
                println!("rmdir: {path:?}");
                fs::remove_dir_all(path)?;
            }
        }

        Ok(disk_cache_dir)
    }

    pub fn clear_compilations_cache(&self) -> sccache::errors::Result<PathBuf> {
        let disk_cache = self.compilations_cache();
        let disk_cache_dir = disk_cache.dir;
        println!("clear_compilations_cache: '{}'", disk_cache_dir.display());

        let paths = walkdir::WalkDir::new(&disk_cache_dir)
            .contents_first(true)
            .into_iter()
            .filter_map(|e| {
                e.ok()
                    .filter(|e| e.file_type().is_file())
                    .map(|e| e.into_path())
            })
            .collect::<Vec<_>>();

        println!("paths: {paths:?}");

        for p in paths {
            if deps_cache_entry(&p).is_none() {
                println!("removing objs cache entry '{}'", p.display());
                fs::remove_file(p)?;
            }
        }

        Ok(disk_cache_dir)
    }

    pub fn clear_toolchains_cache(&self) -> sccache::errors::Result<PathBuf> {
        let dist_config = self.config().dist;
        let tc_dir = dist_config.cache_dir.join("client").join("tc");
        println!("clear_toolchains_cache: {tc_dir:?}");
        if tc_dir.exists() {
            for entry in std::fs::read_dir(&tc_dir)? {
                let path = entry?.path();
                if fs::symlink_metadata(&path)?.is_dir() {
                    fs::remove_dir_all(&path)?;
                } else {
                    fs::remove_file(&path)?;
                }
            }
        }
        Ok(tc_dir.clone())
    }

    pub fn dependencies_cache_entries(
        &self,
    ) -> impl Clone + IntoIterator<Item = (OsString, SourceFileDeps)> {
        let disk_cache = self.compilations_cache();

        walkdir::WalkDir::new(disk_cache.dir.as_path())
            .into_iter()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_type().is_file().then_some(e.into_path()))
            .filter_map(|p| deps_cache_entry(&p).map(|e| (p, e)))
            .map(|(path, deps)| {
                let dep_key = path
                    .components()
                    .next_back()
                    .unwrap()
                    .as_os_str()
                    .to_owned();
                (dep_key, deps)
            })
            .collect::<Vec<_>>()
    }
}

impl Drop for SccacheClient {
    fn drop(&mut self) {
        self.stop();
    }
}

fn deps_cache_entry<P: AsRef<Path>>(p: P) -> Option<SourceFileDeps> {
    let mut b = vec![];

    let Ok(mut f) = fs::File::open(p.as_ref()) else {
        return None;
    };

    if f.read_to_end(&mut b).is_err() {
        return None;
    }

    SourceFileDeps::from_buffer(b.into()).ok()
}
