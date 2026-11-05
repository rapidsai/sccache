// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

pub mod dist {
    pub use crate::config::dist::Keepalive;
    pub use crate::config::dist::client::*;
}

use crate::{
    config::{
        Loadable, Valid,
        cache::{Azure, AzureAuth, Cache, Caches},
        utils::{deserialize_basedirs, deserialize_bool, serialize_basedirs},
    },
    errors::*,
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeSet, HashMap},
    ffi::{OsStr, OsString},
    fmt,
    path::PathBuf,
    sync::{LazyLock, Mutex},
};
use typed_path::Utf8TypedPathBuf;

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Config {
    /// Object cache configs
    #[serde(default)]
    pub cache: Caches,

    /// Base directory (or directories) to strip from paths for cache key computation.
    /// Similar to ccache's CCACHE_BASEDIR.
    #[serde(
        default,
        deserialize_with = "deserialize_basedirs",
        serialize_with = "serialize_basedirs"
    )]
    pub basedirs: Basedirs,

    /// Distributed compilation configuration
    #[serde(default)]
    pub dist: dist::Config,

    /// The number of milliseconds to wait for server startup.
    #[serde(default = "defaults::default_server_startup_timeout")]
    pub server_startup_timeout_ms: u64,

    #[serde(default, deserialize_with = "deserialize_bool")]
    pub client_side_mode: bool,
}

impl Config {
    pub fn load() -> crate::errors::Result<Self> {
        Self::from_envs_and_path::<_, _, _, PathBuf>(std::env::vars_os(), None)
    }
    pub fn load_from_envs<I, S>(vars: I) -> crate::errors::Result<Self>
    where
        I: IntoIterator<Item = (S, S)>,
        S: AsRef<OsStr> + Clone,
    {
        Self::from_envs_and_path::<I, S, _, PathBuf>(vars, None)
    }

    fn ensure_default_disk_cache(mut self) -> Result<Self> {
        if !self.cache.configs.iter().any(|(_, c)| c.enabled()) {
            if let Some((disk_cache_name, disk_cache)) =
                self.cache
                    .configs
                    .iter_mut()
                    .find_map(|(name, cache)| match cache {
                        Cache::Disk(cache) => Some((name, cache)),
                        _ => None,
                    })
            {
                disk_cache.enabled = true;
                self.cache.multilevel.chain = Some(vec![disk_cache_name.clone()]);
            } else {
                self.cache
                    .configs
                    .insert("disk".into(), crate::config::cache::Disk::default().into());
                self.cache.multilevel.chain = Some(vec!["disk".into()]);
            }
        }

        Ok(self)
    }
}

impl_add_for_config! { Config }

// Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
impl Default for Config {
    fn default() -> Self {
        Self {
            cache: Caches::default(),
            basedirs: Basedirs::default(),
            dist: dist::Config::default(),
            server_startup_timeout_ms: defaults::default_server_startup_timeout(),
            client_side_mode: false,
        }
    }
}

impl AsRef<Config> for Config {
    fn as_ref(&self) -> &Config {
        self
    }
}

impl Valid for Config {
    fn validate_merged(mut self, orig: &Self) -> Result<Self> {
        for (name, orig_cache) in orig.cache.configs.iter() {
            if !self.cache.configs.contains_key(name) {
                self.cache.configs.insert(name.clone(), orig_cache.clone());
            }
        }

        self.cache.validate_storage_levels(
            if self.cache.multilevel.chain.as_deref() == orig.cache.multilevel.chain.as_deref() {
                "cache.multilevel.chain"
            } else {
                "SCCACHE_CACHE_MULTILEVEL_CHAIN"
            },
        )?;

        self.ensure_default_disk_cache()
    }

    fn validate_file(self) -> Result<Self> {
        self.cache
            .validate_storage_levels("cache.multilevel.chain")?;

        self.ensure_default_disk_cache()
    }

    fn validate_vars(self) -> Result<Self> {
        self.cache
            .validate_storage_levels("SCCACHE_CACHE_MULTILEVEL_CHAIN")?;

        self.ensure_default_disk_cache()
    }
}

impl Loadable<Self> for Config {
    fn env_prefix<'a>() -> Option<&'a str> {
        Some("SCCACHE")
    }

    fn from_var_list(vars: &[(OsString, OsString)]) -> Result<Self> {
        let mut conf = Self::from_vars_impl(vars)?;

        // Remove Azure configs that don't have an auth mechanism
        conf.cache.configs.retain(|name, cache| match cache {
            Cache::Azure(Azure { auth, .. }) => {
                if matches!(auth, AzureAuth::None) {
                    if let Some(chain) = conf.cache.multilevel.chain.as_mut()
                        && let Ok(idx) = chain.binary_search(name)
                    {
                        chain.remove(idx);
                    }
                    false
                } else {
                    true
                }
            }
            _ => true,
        });

        Ok(conf)
    }

    fn select_vars(vars: &[(OsString, OsString)]) -> Result<Vec<(String, String)>> {
        use crate::util::os_str_to_str;

        // Rewrite env vars to match this struct:
        // * SCCACHE_BUCKET/SCCACHE_REGION map to an S3 cache:
        //   SCCACHE_BUCKET -> SCCACHE_CACHE_S3_BUCKET
        //   SCCACHE_REGION -> SCCACHE_CACHE_S3_REGION
        // * Other S3 fields are prefixed with `SCCACHE_S3_`
        //   SCCACHE_S3_KEY_PREFIX -> SCCACHE_CACHE_S3_KEY_PREFIX
        // * Other cache types are available as top-level envvars:
        //   SCCACHE_GCS_BUCKET -> SCCACHE_CACHE_GCS_BUCKET

        let prefix_ = Self::env_prefix().map_or_else(
            || "SCCACHE_".into(),
            |prefix| {
                if prefix.ends_with("_") {
                    prefix.into()
                } else {
                    format!("{prefix}_")
                }
            },
        );

        let caches_ = [
            "AZURE_",
            "DISK_",
            "GCS_",
            "GHA_",
            "MEMCACHED_",
            "REDIS_",
            "S3_",
            "WEBDAV_",
            "OSS_",
            "COS_",
            "MULTILEVEL_",
        ];

        let vars = vars
            .iter()
            .map(|(key, val)| -> Result<Vec<(String, String)>> {
                let key = os_str_to_str(key.as_os_str())?;
                let val = os_str_to_str(val.as_os_str())?;

                // Only take vars that start with `SCCACHE_`
                let key = if let Some(key) = key.strip_prefix(&prefix_) {
                    key
                } else {
                    return Ok(vec![]);
                };

                let key = match key {
                    // SCCACHE_DIR -> SCCACHE_CACHE_DISK_DIR
                    "DIR" => "DISK_DIR",
                    // SCCACHE_LOCAL_RW_MODE -> SCCACHE_CACHE_DISK_RW_MODE
                    "LOCAL_RW_MODE" => "DISK_RW_MODE",
                    // SCCACHE_CACHE_SIZE -> SCCACHE_CACHE_DISK_SIZE
                    "CACHE_SIZE" => "DISK_SIZE",
                    // SCCACHE_BUCKET -> SCCACHE_CACHE_S3_BUCKET
                    "BUCKET" => "S3_BUCKET",
                    // SCCACHE_ENDPOINT -> SCCACHE_CACHE_S3_ENDPOINT
                    "ENDPOINT" => "S3_ENDPOINT",
                    // SCCACHE_REGION -> SCCACHE_CACHE_S3_REGION
                    "REGION" => "S3_REGION",
                    // SCCACHE_REDIS -> SCCACHE_CACHE_REDIS_URL
                    "REDIS" => "REDIS_URL",
                    // SCCACHE_MEMCACHED -> SCCACHE_CACHE_MEMCACHED_URL
                    "MEMCACHED" => "MEMCACHED_URL",
                    _ => key,
                };

                if key == "SKIP_CACHE_CHECK" {
                    // SCCACHE_SKIP_CACHE_CHECK -> SCCACHE_CACHE_SKIP_CHECK
                    let key = "CACHE_SKIP_CHECK".to_owned();
                    return Ok(vec![(key, val.into_owned())]);
                }

                if key == "SERVER_STARTUP_TIMEOUT" {
                    // SCCACHE_SERVER_STARTUP_TIMEOUT -> SCCACHE_SERVER_STARTUP_TIMEOUT_SECS
                    let key = "SERVER_STARTUP_TIMEOUT_MS".to_owned();
                    return Ok(vec![(key, val.into_owned())]);
                }

                if let Some(suf) = key.strip_prefix("DIST_") {
                    if matches!(
                        suf,
                        "CONNECT_TIMEOUT"
                            | "REQUEST_TIMEOUT"
                            | "CONNECTION_POOL"
                            | "MAX_CONNECTIONS"
                    ) {
                        // SCCACHE_DIST_CONNECT_TIMEOUT -> SCCACHE_DIST_NET_CONNECT_TIMEOUT
                        // SCCACHE_DIST_REQUEST_TIMEOUT -> SCCACHE_DIST_NET_REQUEST_TIMEOUT
                        // SCCACHE_DIST_CONNECTION_POOL -> SCCACHE_DIST_NET_CONNECTION_POOL
                        // SCCACHE_DIST_MAX_CONNECTIONS -> SCCACHE_DIST_NET_MAX_CONNECTIONS
                        let key = format!("DIST_NET_{suf}");
                        return Ok(vec![(key, val.into_owned())]);
                    }

                    if let Some(key) = suf.strip_prefix("KEEPALIVE_") {
                        // SCCACHE_DIST_KEEPALIVE_ENABLED -> SCCACHE_DIST_NET_KEEPALIVE_ENABLED
                        let key = format!("DIST_NET_KEEPALIVE_{key}");
                        return Ok(vec![(key, val.into_owned())]);
                    }

                    if matches!(suf, "SCHEDULER_URL") {
                        // SCCACHE_DIST_SCHEDULER_URL -> SCCACHE_DIST_URL
                        let key = "DIST_URL".to_owned();
                        return Ok(vec![(key, val.into_owned())]);
                    }

                    return Ok(vec![(key.into(), val.into_owned())]);
                }

                if matches!(key, "CACHE_READ_TIMEOUT" | "CACHE_WRITE_TIMEOUT") {
                    // SCCACHE_CACHE_READ_TIMEOUT -> SCCACHE_CACHE_READ_TIMEOUT_SECS
                    // SCCACHE_CACHE_WRITE_TIMEOUT -> SCCACHE_CACHE_WRITE_TIMEOUT_SECS
                    let key = format!("{key}_SECS");
                    return Ok(vec![(key, val.into_owned())]);
                }

                if let Some((kind, mut key)) = caches_.iter().find_map(|kind_| {
                    if let Some(key) = key.strip_prefix(kind_) {
                        kind_.strip_suffix("_").map(|kind| (kind, key))
                    } else {
                        None
                    }
                }) {
                    if kind == "AZURE" && key == "BLOB_CONTAINER" {
                        // SCCACHE_AZURE_BLOB_CONTAINER -> SCCACHE_AZURE_CONTAINER
                        key = "CONTAINER";
                    }

                    // SCCACHE_AZURE_CONNECTION_STRING -> SCCACHE_CACHE_AZURE_CONNECTION_STRING
                    // SCCACHE_S3_KEY_PREFIX -> SCCACHE_CACHE_S3_KEY_PREFIX
                    // SCCACHE_GCS_BUCKET -> SCCACHE_CACHE_GCS_BUCKET
                    // SCCACHE_MULTILEVEL_CHAIN -> SCCACHE_CACHE_MULTILEVEL_CHAIN
                    // etc.
                    let key = format!("CACHE_{kind}_{key}");
                    return Ok(vec![(key, val.into_owned())]);
                }

                Ok(vec![(key.into(), val.into_owned())])
            })
            .flatten_ok()
            .try_collect::<_, Vec<_>, _>()?;

        if !vars.is_empty() {
            // trace! because envvars might include secrets.
            // SCCACHE_LOG=debug is the lowest recommended log level in CI.
            trace!(
                "Building client config from env:\n{}",
                vars.iter()
                    .map(|(k, v)| format!("{prefix_}{k} = \"{v}\""))
                    .join("\n")
            );
        }

        Ok(vars)
    }
}

#[derive(Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct Basedirs(Option<Vec<PathBuf>>, #[serde(skip)] Option<Vec<Vec<u8>>>);

impl Basedirs {
    pub fn empty() -> Self {
        Self(Some(vec![]), Some(vec![]))
    }
    pub fn is_none(&self) -> bool {
        self.0.is_none()
    }
    pub fn is_some(&self) -> bool {
        self.0.is_some()
    }
    pub fn is_empty(&self) -> bool {
        self.1.as_deref().unwrap_or_default().is_empty()
    }
}

impl From<Basedirs> for Vec<PathBuf> {
    fn from(basedirs: Basedirs) -> Self {
        basedirs.0.unwrap_or_default()
    }
}

impl From<Basedirs> for Vec<Vec<u8>> {
    fn from(basedirs: Basedirs) -> Self {
        basedirs.1.unwrap_or_default()
    }
}

impl AsRef<[PathBuf]> for Basedirs {
    #[inline]
    fn as_ref(&self) -> &[PathBuf] {
        self.0.as_deref().unwrap_or_default()
    }
}

impl AsRef<[Vec<u8>]> for Basedirs {
    #[inline]
    fn as_ref(&self) -> &[Vec<u8>] {
        self.1.as_deref().unwrap_or_default()
    }
}

impl std::ops::Deref for Basedirs {
    type Target = [Vec<u8>];

    #[inline]
    fn deref(&self) -> &[Vec<u8>] {
        self.1.as_deref().unwrap_or_default()
    }
}

impl fmt::Debug for Basedirs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(|p| format!("{p:?}"))
                .join(", ")
        )
    }
}

impl fmt::Display for Basedirs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(|p| format!("{}", p.display()))
                .join(", ")
        )
    }
}

impl PartialEq<Vec<Vec<u8>>> for Basedirs {
    fn eq(&self, rhs: &Vec<Vec<u8>>) -> bool {
        self.1.as_deref().unwrap_or_default() == rhs
    }
}

impl From<Vec<Vec<u8>>> for Basedirs {
    fn from(basedirs: Vec<Vec<u8>>) -> Self {
        Basedirs::try_from(
            basedirs
                .into_iter()
                .map(|b| String::from_utf8(b).map(PathBuf::from).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
}

impl From<Vec<String>> for Basedirs {
    fn from(basedirs: Vec<String>) -> Self {
        Basedirs::try_from(basedirs.into_iter().map(PathBuf::from).collect::<Vec<_>>()).unwrap()
    }
}

impl From<Vec<OsString>> for Basedirs {
    fn from(basedirs: Vec<OsString>) -> Self {
        Basedirs::try_from(basedirs.into_iter().map(PathBuf::from).collect::<Vec<_>>()).unwrap()
    }
}

impl<'a> From<Vec<&'a str>> for Basedirs {
    fn from(basedirs: Vec<&'a str>) -> Self {
        Basedirs::try_from(basedirs.into_iter().map(PathBuf::from).collect::<Vec<_>>()).unwrap()
    }
}

impl<'a> From<Vec<&'a OsStr>> for Basedirs {
    fn from(basedirs: Vec<&'a OsStr>) -> Self {
        Basedirs::try_from(basedirs.into_iter().map(PathBuf::from).collect::<Vec<_>>()).unwrap()
    }
}

impl<const N: usize> From<[&str; N]> for Basedirs {
    fn from(basedirs: [&str; N]) -> Self {
        Basedirs::try_from(basedirs.into_iter().map(PathBuf::from).collect::<Vec<_>>()).unwrap()
    }
}

impl<'a> From<&'a [&'a str]> for Basedirs {
    fn from(basedirs: &'a [&'a str]) -> Self {
        Basedirs::try_from(basedirs.iter().map(PathBuf::from).collect::<Vec<_>>()).unwrap()
    }
}

impl<'a> From<&'a [&'a OsStr]> for Basedirs {
    fn from(basedirs: &'a [&'a OsStr]) -> Self {
        Basedirs::try_from(basedirs.iter().map(PathBuf::from).collect::<Vec<_>>()).unwrap()
    }
}

impl TryFrom<Vec<PathBuf>> for Basedirs {
    type Error = anyhow::Error;

    fn try_from(basedirs: Vec<PathBuf>) -> Result<Self> {
        basedirs
            .into_iter()
            .filter(|path| !path.as_os_str().is_empty())
            .map(|orig| {
                if !orig.is_absolute() {
                    bail!("Basedir path must be absolute: {orig:?}");
                }

                let path = crate::util::path_to_str(&orig)?;
                // Normalize basedir:
                // remove double separators, cur_dirs, parent_dirs, trailing slashes
                let path = Utf8TypedPathBuf::from(path.as_ref()).normalize();

                let path = path.as_str().as_bytes();

                #[cfg(not(target_os = "windows"))]
                let mut path = path.to_vec();

                #[cfg(target_os = "windows")]
                // normalize windows paths: use slashes and lowercase
                let mut path = crate::util::normalize_win_path(path);

                // Always add a trailing `/` to basedirs to ensure we only match complete path
                // components
                if !matches!(path.last(), Some(b'/')) {
                    path.push(b'/');
                }

                Ok((crate::util::bytes_to_path(&path)?.to_path_buf(), path))
            })
            // Only take unique paths
            .try_collect::<_, (BTreeSet<_>, BTreeSet<_>), _>()
            .map(|(orig, bufs)| {
                Self(
                    Some(orig.into_iter().collect()),
                    Some(bufs.into_iter().collect()),
                )
            })
    }
}

static CACHED_CONFIG_PATH: LazyLock<PathBuf> = LazyLock::new(CachedConfig::file_config_path);
static CACHED_CONFIG: Mutex<Option<CachedFileConfig>> = Mutex::new(None);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CachedDistConfig {
    pub auth_tokens: HashMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CachedFileConfig {
    pub dist: CachedDistConfig,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct CachedConfig(());

impl CachedConfig {
    pub fn load() -> Result<Self> {
        let mut cached_file_config = CACHED_CONFIG.lock().unwrap();

        if cached_file_config.is_none() {
            let cfg = Self::load_file_config().context("Unable to initialise cached config")?;
            *cached_file_config = Some(cfg);
        }
        Ok(CachedConfig(()))
    }

    pub fn reload() -> Result<Self> {
        {
            let mut cached_file_config = CACHED_CONFIG.lock().unwrap();
            *cached_file_config = None;
        };
        Self::load()
    }

    pub fn with<F: FnOnce(&CachedFileConfig) -> T, T>(&self, f: F) -> T {
        let cached_file_config = CACHED_CONFIG.lock().unwrap();
        let cached_file_config = cached_file_config.as_ref().unwrap();

        f(cached_file_config)
    }

    pub fn with_mut<F: FnOnce(&mut CachedFileConfig)>(&self, f: F) -> Result<()> {
        let mut cached_file_config = CACHED_CONFIG.lock().unwrap();
        let cached_file_config = cached_file_config.as_mut().unwrap();

        let mut new_config = cached_file_config.clone();
        f(&mut new_config);
        Self::save_file_config(&new_config)?;
        *cached_file_config = new_config;
        Ok(())
    }

    pub fn file_config_path() -> PathBuf {
        crate::config::utils::config_file("SCCACHE_CACHED_CONF", "cached-config")
    }

    fn load_file_config() -> Result<CachedFileConfig> {
        use fs_err as fs;

        let file_conf_path = &*CACHED_CONFIG_PATH;

        if !file_conf_path.exists() {
            let file_conf_dir = file_conf_path
                .parent()
                .expect("Cached conf file has no parent directory");
            if !file_conf_dir.is_dir() {
                fs::create_dir_all(file_conf_dir)
                    .context("Failed to create dir to hold cached config")?;
            }
            Self::save_file_config(&Default::default()).with_context(|| {
                format!(
                    "Unable to create cached config file at {}",
                    file_conf_path.display()
                )
            })?;
        }

        crate::config::utils::try_read_config_file(file_conf_path)
            .context("Failed to load cached config file")?
            .with_context(|| format!("Failed to load from {}", file_conf_path.display()))
    }

    fn save_file_config(c: &CachedFileConfig) -> Result<()> {
        use fs_err::File;
        use std::io::Write;

        let file_conf_path = &*CACHED_CONFIG_PATH;
        let mut file = File::create(file_conf_path).context("Could not open config for writing")?;
        file.write_all(toml::to_string(c).unwrap().as_bytes())
            .map_err(Into::into)
    }
}

pub mod defaults {

    pub use crate::config::defaults::default_disk_cache_size;

    pub fn default_server_startup_timeout() -> u64 {
        10_000
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::config::{
        WriteErrorPolicy,
        cache::{
            Azure, AzureAuth, COS, Cache, CacheMode, Disk, GCS, GHA, Memcached, MultiLevel, OSS,
            Redis, S3, Webdav,
        },
    };
    use std::{collections::BTreeMap, ffi::OsString};
    #[cfg(any(feature = "dist-client", feature = "dist-server"))]
    use {crate::config::utils::HTTPUrl, std::str::FromStr};

    #[test]
    fn defaults_match_serde_defaults() -> Result<()> {
        assert_eq!(Config::default(), serde_json::from_str("{}").unwrap());

        assert_eq!(
            Config::from_json_impl(r#"{"cache": {}, "dist": {"net": {"keepalive": {}}}}"#)?,
            Config::default()
        );
        assert_eq!(
            Config::from_toml_impl("[cache.disk]")?.cache.configs["disk"],
            Cache::Disk(Disk::default())
        );

        Ok(())
    }

    #[test]
    fn config_from_vars() -> Result<()> {
        drop(env_logger::try_init());

        #[cfg(not(target_os = "windows"))]
        let (basedirs, dirs_delim) = { (["/home/user/project", "/home/user/workspace"], ":") };
        #[cfg(target_os = "windows")]
        let (basedirs, dirs_delim) = { (["C:/home/user/project", "C:/home/user/workspace"], ";") };

        let mut config = Config {
            basedirs: basedirs.into(),
            cache: vec![
                COS::from_bucket("cos").into(),
                OSS::from_bucket("oss").into(),
                Webdav::from_endpoint("http://webdav:1234").into(),
                Azure {
                    auth: AzureAuth::SharedKey {
                        connection_string: "connection-string".into(),
                    },
                    ..Azure::from_container("azure")
                }
                .into(),
                GHA::from_version("").into(),
                GCS::from_bucket("gcs").into(),
                Memcached::from_url("memcached://memcached:1234").into(),
                Redis::from_url("redis://redis:1234").into(),
                S3::from_bucket("s3").into(),
                Disk::default().into(),
            ]
            .into(),
            dist: dist::Config {
                auth: dist::Auth::Token {
                    token: "hello-world".into(),
                },
                cache_dir: "/home/user/.cache/sccache/dist".into(),
                fallback_to_local_compile: false,
                max_retries: f64::INFINITY.into(),
                net: dist::Networking {
                    connect_timeout: 60,
                    request_timeout: 1200,
                    connection_pool: false,
                    max_connections: 100,
                    keepalive: dist::Keepalive {
                        enabled: true,
                        interval: 10,
                        timeout: 1200,
                    },
                },
                #[cfg(any(feature = "dist-client", feature = "dist-server"))]
                url: HTTPUrl::from_str("http://sccache.my-company.com").ok(),
                #[cfg(not(any(feature = "dist-client", feature = "dist-server")))]
                url: Some("http://sccache.my-company.com".into()),
                toolchains: vec![],
                toolchain_cache_size: 10_000,
            },
            server_startup_timeout_ms: 10000,
            client_side_mode: true,
        };

        config.cache.read_timeout_secs = 120;
        config.cache.write_timeout_secs = Some(300);
        config.cache.skip_check = true;
        config.cache.multilevel.write_error_policy = WriteErrorPolicy::All;
        config.cache.rate_limit.on_error_count = Some(5);
        config.cache.rate_limit.on_error_window_size_secs = Some(5);

        let vars = std::iter::empty()
            .chain([
                ("SCCACHE_BASEDIRS", basedirs.join(dirs_delim).as_str()),
                ("SCCACHE_SKIP_CACHE_CHECK", "true"),
                ("SCCACHE_CACHE_READ_TIMEOUT_SECS", "120"),
                ("SCCACHE_CACHE_WRITE_TIMEOUT_SECS", "300"),
                ("SCCACHE_CACHE_RATE_LIMIT_ON_ERROR_COUNT", "5"),
                ("SCCACHE_CACHE_RATE_LIMIT_ON_ERROR_WINDOW_SIZE_SECS", "5"),
                (
                    "SCCACHE_MULTILEVEL_CHAIN",
                    " cos, oss , webdav,azure,gha,gcs,memcached,redis,s3 ,disk",
                ),
                ("SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY", "all"),
            ])
            .chain(config.cache.configs.values().flat_map(|cache| match cache {
                Cache::Disk(c) => vec![("SCCACHE_DIR", c.dir.to_str().unwrap())],
                Cache::S3(c) => vec![("SCCACHE_BUCKET", c.bucket.as_str())],
                Cache::Redis(c) => vec![("SCCACHE_REDIS", c.url.as_deref().unwrap())],
                Cache::Memcached(c) => vec![("SCCACHE_MEMCACHED", c.url.as_str())],
                Cache::GCS(c) => vec![("SCCACHE_GCS_BUCKET", c.bucket.as_str())],
                Cache::GHA(_) => vec![("SCCACHE_GHA_ENABLED", "true")],
                Cache::Azure(Azure {
                    container,
                    auth: AzureAuth::SharedKey { connection_string },
                    ..
                }) => {
                    vec![
                        ("SCCACHE_AZURE_CONTAINER", container.as_str()),
                        (
                            "SCCACHE_AZURE_CONNECTION_STRING",
                            connection_string.as_str(),
                        ),
                    ]
                }
                Cache::Webdav(c) => {
                    vec![("SCCACHE_WEBDAV_ENDPOINT", c.endpoint.as_str())]
                }
                Cache::OSS(c) => vec![("SCCACHE_OSS_BUCKET", c.bucket.as_str())],
                Cache::COS(c) => vec![("SCCACHE_COS_BUCKET", c.bucket.as_str())],
                _ => vec![],
            }))
            .chain([
                ("SCCACHE_DIST_AUTH_TYPE", "token"),
                ("SCCACHE_DIST_AUTH_TOKEN", "hello-world"),
                (
                    "SCCACHE_DIST_CACHE_DIR",
                    config.dist.cache_dir.as_path().to_string_lossy().as_ref(),
                ),
                (
                    "SCCACHE_CACHE_MULTILEVEL_CHAIN",
                    config
                        .cache
                        .multilevel
                        .chain
                        .as_deref()
                        .map(|chain| chain.join(","))
                        .unwrap_or_default()
                        .as_str(),
                ),
                ("SCCACHE_DIST_FALLBACK_TO_LOCAL_COMPILE", "false"),
                ("SCCACHE_DIST_MAX_RETRIES", "inf"),
                ("SCCACHE_DIST_CONNECT_TIMEOUT", "60"),
                ("SCCACHE_DIST_REQUEST_TIMEOUT", "1200"),
                ("SCCACHE_DIST_CONNECTION_POOL", "false"),
                ("SCCACHE_DIST_MAX_CONNECTIONS", "100"),
                ("SCCACHE_DIST_KEEPALIVE_ENABLED", "true"),
                ("SCCACHE_DIST_KEEPALIVE_INTERVAL", "10"),
                ("SCCACHE_DIST_KEEPALIVE_TIMEOUT", "1200"),
                ("SCCACHE_DIST_URL", "http://sccache.my-company.com"),
                ("SCCACHE_DIST_TOOLCHAINS", ""),
                ("SCCACHE_DIST_TOOLCHAIN_CACHE_SIZE", "10000"),
                ("SCCACHE_SERVER_STARTUP_TIMEOUT_MS", "10000"),
                ("SCCACHE_CLIENT_SIDE_MODE", "true"),
            ])
            .map(|(key, val)| (key.into(), val.into()))
            .collect::<Vec<(OsString, OsString)>>();

        let env_conf = Config::from_vars(vars).and_then(|conf| conf.validate_vars())?;

        assert_eq!(config.basedirs, env_conf.basedirs);
        assert_eq!(config.cache.configs, env_conf.cache.configs);
        assert_eq!(config.cache.multilevel, env_conf.cache.multilevel);
        assert_eq!(
            config.cache.read_timeout_secs,
            env_conf.cache.read_timeout_secs
        );
        assert_eq!(
            config.cache.write_timeout_secs,
            env_conf.cache.write_timeout_secs
        );
        assert_eq!(config.cache.skip_check, env_conf.cache.skip_check);
        assert_eq!(config.dist, env_conf.dist);
        assert_eq!(
            config.server_startup_timeout_ms,
            env_conf.server_startup_timeout_ms
        );
        assert_eq!(config.client_side_mode, env_conf.client_side_mode);

        Ok(())
    }

    #[test]
    fn config_from_toml() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_toml(
            r#"
            server_startup_timeout_ms = 10000

            [cache.azure]
            container = "azurecontainer"
            key_prefix = "azureprefix"
            storage_account = "azureaccount"

            [cache.disk]
            dir = "/tmp/.cache/sccache"
            size = 7516192768 # 7 GiBytes

            [cache.gcs]
            rw_mode = "READ_ONLY"
            # rw_mode = "READ_WRITE"
            cred_path = "/psst/secret/cred"
            bucket = "bucket"
            key_prefix = "prefix"
            service_account = "example_service_account"

            [cache.gha]
            enabled = true
            version = "sccache"

            [cache.memcached]
            # Deprecated alias for `endpoint`
            # url = "127.0.0.1:11211"
            endpoint = "tcp://127.0.0.1:11211"
            # Username and password for authentication
            username = "user"
            password = "passwd"
            expiration = 90000
            key_prefix = "/custom/prefix/if/need"

            [cache.redis]
            # Deprecated in favor of `endpoint`.
            url = "redis://user:passwd@1.2.3.4:6379/?db=1"
            endpoint = "redis://127.0.0.1:6379"
            cluster_endpoints = "tcp://10.0.0.1:6379,redis://10.0.0.2:6379"
            reader_endpoints = "tcp://10.0.0.1:6380,redis://10.0.0.2:6380"
            username = "another_user"
            password = "new_passwd"
            db = 12
            expiration = 86400
            key_prefix = "/my/redis/cache"

            [cache.s3]
            bucket = "name"
            region = "us-east-2"
            endpoint = "s3-us-east-1.amazonaws.com"
            use_ssl = true
            key_prefix = "s3prefix"
            no_credentials = false
            server_side_encryption = false
            rw_mode = "READ_WRITE"

            [cache.webdav]
            endpoint = "http://127.0.0.1:8080"
            key_prefix = "webdavprefix"
            username = "webdavusername"
            password = "webdavpassword"
            token = "webdavtoken"

            [cache.oss]
            bucket = "name"
            endpoint = "oss-us-east-1.aliyuncs.com"
            key_prefix = "ossprefix"
            no_credentials = true
            rw_mode = "READ_ONLY"

            [cache.cos]
            bucket = "name"
            endpoint = "cos.na-siliconvalley.myqcloud.com"
            key_prefix = "cosprefix"

            [cache.custom-1]
            type = "cos"
            bucket = "name"
            endpoint = "cos.na-siliconvalley.myqcloud.com"
            key_prefix = "cosprefix"

            [cache.configs.custom-2]
            type = "cos"
            bucket = "name"
            endpoint = "cos.na-siliconvalley.myqcloud.com"
            key_prefix = "cosprefix"

            [cache.multilevel]
            chain = ["disk", "s3", "redis", "memcached", "gcs", "gha", "azure", "webdav", "oss", "cos", "custom-1", "custom-2"]

            [dist]
            # where to find the scheduler
            scheduler_url = "http://1.2.3.4:10600"
            # the maximum size of the toolchain cache in bytes
            toolchain_cache_size = 5368709120
            cache_dir = "/home/user/.cache/sccache-dist-client"

            [dist.auth]
            type = "token"
            token = "secrettoken"

            # a set of prepackaged toolchains
            [[dist.toolchains]]
            type = "path_override"
            archive = "/toolchains/cc.tar.gz"
            compiler_executable = "/usr/bin/cc"
            archive_compiler_executable = "/usr/bin/cc"

            [[dist.toolchains]]
            type = "path_override"
            archive = "/toolchains/rust.tar.gz"
            compiler_executable = "/usr/bin/rustc"
            archive_compiler_executable = "/usr/bin/rustc"
            "#,
        )?;

        let mut cache_configs = [
            Azure {
                auth: AzureAuth::StorageAccount {
                    storage_account: "azureaccount".into(),
                },
                key_prefix: "azureprefix".into(),
                ..Azure::from_container("azurecontainer")
            }
            .into(),
            Disk {
                dir: PathBuf::from("/tmp/.cache/sccache"),
                size: 7 * 1024 * 1024 * 1024,
                rw_mode: CacheMode::ReadWrite,
                ..Default::default()
            }
            .into(),
            GCS {
                key_path: Some("/psst/secret/cred".into()),
                service_account: Some("example_service_account".into()),
                rw_mode: CacheMode::ReadOnly,
                key_prefix: "prefix".into(),
                credentials_url: None,
                ..GCS::from_bucket("bucket")
            }
            .into(),
            GHA::from_version("sccache").into(),
            Redis {
                endpoint: Some("redis://127.0.0.1:6379".into()),
                cluster_endpoints: Some("tcp://10.0.0.1:6379,redis://10.0.0.2:6379".into()),
                reader_endpoints: Some("tcp://10.0.0.1:6380,redis://10.0.0.2:6380".into()),
                username: Some("another_user".into()),
                password: Some("new_passwd".into()),
                db: 12,
                ttl: 24 * 3600,
                key_prefix: "/my/redis/cache".into(),
                ..Redis::from_url("redis://user:passwd@1.2.3.4:6379/?db=1")
            }
            .into(),
            Memcached {
                username: Some("user".into()),
                password: Some("passwd".into()),
                expiration: 25 * 3600,
                key_prefix: "/custom/prefix/if/need".into(),
                rw_mode: CacheMode::ReadWrite,
                ..Memcached::from_url("tcp://127.0.0.1:11211")
            }
            .into(),
            S3 {
                region: Some("us-east-2".into()),
                endpoint: Some("s3-us-east-1.amazonaws.com".into()),
                use_ssl: Some(true),
                key_prefix: "s3prefix".into(),
                no_credentials: false,
                server_side_encryption: Some(false),
                ..S3::from_bucket("name")
            }
            .into(),
            Webdav {
                key_prefix: "webdavprefix".into(),
                username: Some("webdavusername".into()),
                password: Some("webdavpassword".into()),
                token: Some("webdavtoken".into()),
                ..Webdav::from_endpoint("http://127.0.0.1:8080")
            }
            .into(),
            OSS {
                endpoint: Some("oss-us-east-1.aliyuncs.com".into()),
                key_prefix: "ossprefix".into(),
                no_credentials: true,
                rw_mode: CacheMode::ReadOnly,
                ..OSS::from_bucket("name")
            }
            .into(),
            COS {
                endpoint: Some("cos.na-siliconvalley.myqcloud.com".into()),
                key_prefix: "cosprefix".into(),
                rw_mode: CacheMode::ReadWrite,
                ..COS::from_bucket("name")
            }
            .into(),
        ]
        .into_iter()
        .map(|(name, cache)| (name.to_owned(), cache))
        .collect::<BTreeMap<String, Cache>>();

        cache_configs.insert("custom-1".into(), cache_configs.get("cos").unwrap().clone());
        cache_configs.insert("custom-2".into(), cache_configs.get("cos").unwrap().clone());

        assert_eq!(
            config,
            Config {
                cache: Caches {
                    multilevel: MultiLevel {
                        chain: Some(
                            [
                                "disk",
                                "s3",
                                "redis",
                                "memcached",
                                "gcs",
                                "gha",
                                "azure",
                                "webdav",
                                "oss",
                                "cos",
                                "custom-1",
                                "custom-2"
                            ]
                            .into_iter()
                            .map(Into::into)
                            .collect()
                        ),
                        ..Default::default()
                    },
                    configs: cache_configs,
                    ..Default::default()
                },
                dist: dist::Config {
                    auth: dist::Auth::Token {
                        token: "secrettoken".into()
                    },
                    cache_dir: PathBuf::from("/home/user/.cache/sccache-dist-client"),
                    toolchains: vec![
                        dist::Toolchain::PathOverride {
                            archive: "/toolchains/cc.tar.gz".into(),
                            compiler_executable: "/usr/bin/cc".into(),
                            archive_compiler_executable: "/usr/bin/cc".into(),
                        },
                        dist::Toolchain::PathOverride {
                            archive: "/toolchains/rust.tar.gz".into(),
                            compiler_executable: "/usr/bin/rustc".into(),
                            archive_compiler_executable: "/usr/bin/rustc".into(),
                        }
                    ],
                    toolchain_cache_size: 5368709120,
                    #[cfg(any(feature = "dist-client", feature = "dist-server"))]
                    url: Some(HTTPUrl::from_str("http://1.2.3.4:10600")?),
                    #[cfg(not(any(feature = "dist-client", feature = "dist-server")))]
                    url: Some("http://1.2.3.4:10600".into()),
                    ..Default::default()
                },
                server_startup_timeout_ms: 10000,
                ..Default::default()
            }
        );

        Ok(())
    }

    #[test]
    fn config_from_json() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_json(
            r#"{
                "server_startup_timeout_ms": 5000,
                "cache": {
                    "read_timeout": 100,
                    "disk": {"dir": "/tmp/.cache/sccache", "size": "7G"},
                    "custom": {"type": "redis", "endpoint": "redis://127.0.0.1:6379", "ttl": 3600},
                    "multilevel": {"chain": "disk, custom"}
                },
                "dist": {
                    "scheduler_url": "http://1.2.3.4:10600",
                    "auth": {"type": "token", "token": "secrettoken"}
                }
            }"#,
        )?;

        let mut cache: Caches = vec![
            Disk {
                dir: "/tmp/.cache/sccache".into(),
                size: 7 * 1024 * 1024 * 1024,
                ..Default::default()
            }
            .into(),
        ]
        .into();
        cache.configs.insert(
            "custom".into(),
            Redis {
                endpoint: Some("redis://127.0.0.1:6379".into()),
                url: None,
                ttl: 3600,
                ..Redis::from_url("")
            }
            .into(),
        );
        cache.multilevel.chain = Some(vec!["disk".into(), "custom".into()]);
        cache.read_timeout_secs = 100;

        assert_eq!(
            config,
            Config {
                cache,
                dist: dist::Config {
                    auth: dist::Auth::Token {
                        token: "secrettoken".into()
                    },
                    #[cfg(any(feature = "dist-client", feature = "dist-server"))]
                    url: Some(HTTPUrl::from_str("http://1.2.3.4:10600")?),
                    #[cfg(not(any(feature = "dist-client", feature = "dist-server")))]
                    url: Some("http://1.2.3.4:10600".into()),
                    ..Default::default()
                },
                server_startup_timeout_ms: 5000,
                ..Default::default()
            }
        );

        Ok(())
    }

    #[test]
    fn config_from_vars_aliases() -> Result<()> {
        drop(env_logger::try_init());

        let mut config = Config {
            cache: vec![
                GCS {
                    key_path: Some("gcs_cred_path".into()),
                    credentials_url: Some("gcs_credential_url".into()),
                    ..GCS::from_bucket("gcs_bucket")
                }
                .into(),
                Memcached::from_url("memcached_url").into(),
                Redis {
                    ttl: 100,
                    ..Redis::from_url("redis_url")
                }
                .into(),
            ]
            .into(),
            dist: dist::Config {
                #[cfg(any(feature = "dist-client", feature = "dist-server"))]
                url: Some(HTTPUrl::from_str("http://scheduler.url")?),
                #[cfg(not(any(feature = "dist-client", feature = "dist-server")))]
                url: Some("http://scheduler.url".into()),
                net: dist::Networking {
                    connect_timeout: 100,
                    request_timeout: 100,
                    connection_pool: false,
                    max_connections: 100,
                    keepalive: dist::Keepalive {
                        enabled: false,
                        timeout: 100,
                        interval: 100,
                    },
                },
                ..Default::default()
            },
            ..Default::default()
        };

        config.cache.skip_check = true;
        config.cache.read_timeout_secs = 100;
        config.cache.write_timeout_secs = Some(100);
        config.cache.multilevel.chain = Some(
            ["gcs", "memcached", "redis"]
                .into_iter()
                .map(Into::into)
                .collect(),
        );

        let env_conf = Config::from_vars([
            // SCCACHE_MULTILEVEL_CHAIN -> SCCACHE_CACHE_MULTILEVEL_CHAIN
            ("SCCACHE_MULTILEVEL_CHAIN", "gcs,memcached,redis"),
            // SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY -> SCCACHE_CACHE_MULTILEVEL_WRITE_ERROR_POLICY
            ("SCCACHE_MULTILEVEL_WRITE_POLICY", "ALL"),
            // SCCACHE_SKIP_CACHE_CHECK -> SCCACHE_CACHE_SKIP_CHECK
            ("SCCACHE_SKIP_CACHE_CHECK", "true"),
            // SCCACHE_CACHE_READ_TIMEOUT -> SCCACHE_CACHE_READ_TIMEOUT_SECS
            ("SCCACHE_CACHE_READ_TIMEOUT", "100"),
            // SCCACHE_CACHE_WRITE_TIMEOUT -> SCCACHE_CACHE_WRITE_TIMEOUT_SECS
            ("SCCACHE_CACHE_WRITE_TIMEOUT", "100"),
            ("SCCACHE_GCS_BUCKET", "gcs_bucket"),
            // SCCACHE_GCS_CRED_PATH -> SCCACHE_GCS_KEY_PATH
            ("SCCACHE_GCS_CRED_PATH", "gcs_cred_path"),
            // SCCACHE_GCS_CREDENTIAL_URL -> SCCACHE_GCS_CREDENTIALS_URL
            ("SCCACHE_GCS_CREDENTIAL_URL", "gcs_credential_url"),
            // SCCACHE_MEMCACHED -> SCCACHE_MEMCACHED_URL
            ("SCCACHE_MEMCACHED", "memcached_url"),
            // SCCACHE_REDIS -> SCCACHE_CACHE_REDIS_URL
            ("SCCACHE_REDIS", "redis_url"),
            // SCCACHE_REDIS_EXPIRATION -> SCCACHE_CACHE_REDIS_TTL
            ("SCCACHE_REDIS_EXPIRATION", "100"),
            // SCCACHE_DIST_SCHEDULER_URL -> SCCACHE_DIST_URL
            ("SCCACHE_DIST_SCHEDULER_URL", "http://scheduler.url"),
            // SCCACHE_DIST_CONNECT_TIMEOUT -> SCCACHE_DIST_NET_CONNECT_TIMEOUT
            ("SCCACHE_DIST_CONNECT_TIMEOUT", "100"),
            // SCCACHE_DIST_REQUEST_TIMEOUT -> SCCACHE_DIST_NET_REQUEST_TIMEOUT
            ("SCCACHE_DIST_REQUEST_TIMEOUT", "100"),
            // SCCACHE_DIST_CONNECTION_POOL -> SCCACHE_DIST_NET_CONNECTION_POOL
            ("SCCACHE_DIST_CONNECTION_POOL", "false"),
            // SCCACHE_DIST_MAX_CONNECTIONS -> SCCACHE_DIST_NET_MAX_CONNECTIONS
            ("SCCACHE_DIST_MAX_CONNECTIONS", "100"),
            // SCCACHE_DIST_KEEPALIVE_ENABLED -> SCCACHE_DIST_NET_KEEPALIVE_ENABLED
            ("SCCACHE_DIST_KEEPALIVE_ENABLED", "false"),
            // SCCACHE_DIST_KEEPALIVE_TIMEOUT -> SCCACHE_DIST_NET_KEEPALIVE_TIMEOUT
            ("SCCACHE_DIST_KEEPALIVE_TIMEOUT", "100"),
            // SCCACHE_DIST_KEEPALIVE_INTERVAL -> SCCACHE_DIST_NET_KEEPALIVE_INTERVAL
            ("SCCACHE_DIST_KEEPALIVE_INTERVAL", "100"),
        ])
        .and_then(|conf| conf.validate_vars())?;

        assert_eq!(env_conf, config);

        Ok(())
    }

    #[test]
    fn config_from_toml_aliases() -> Result<()> {
        drop(env_logger::try_init());

        let mut config = Config {
            cache: vec![
                GCS {
                    key_path: Some("gcs_cred_path".into()),
                    credentials_url: Some("gcs_credential_url".into()),
                    ..GCS::from_bucket("gcs_bucket")
                }
                .into(),
                Memcached::from_url("memcached_url").into(),
                Redis {
                    ttl: 100,
                    ..Redis::from_url("redis_url")
                }
                .into(),
            ]
            .into(),
            dist: dist::Config {
                #[cfg(any(feature = "dist-client", feature = "dist-server"))]
                url: Some(HTTPUrl::from_str("http://scheduler.url")?),
                #[cfg(not(any(feature = "dist-client", feature = "dist-server")))]
                url: Some("http://scheduler.url".into()),
                net: dist::Networking {
                    connect_timeout: 100,
                    request_timeout: 100,
                    connection_pool: false,
                    max_connections: 100,
                    keepalive: dist::Keepalive {
                        enabled: false,
                        timeout: 100,
                        interval: 100,
                    },
                },
                ..Default::default()
            },
            ..Default::default()
        };

        config.cache.skip_check = true;
        config.cache.read_timeout_secs = 100;
        config.cache.write_timeout_secs = Some(100);
        config.cache.multilevel.chain = Some(
            ["gcs", "memcached", "redis"]
                .into_iter()
                .map(Into::into)
                .collect(),
        );

        let file_conf = Config::from_toml(
            r#"
            server_startup_timeout_ms = 10000
            client_side_mode = false

            [cache]
            read_timeout = 100
            write_timeout = 100
            skip_check = true

            [cache.gcs]
            bucket = "gcs_bucket"
            key_path = "gcs_cred_path"
            credential_url = "gcs_credential_url"

            [cache.memcached]
            url = "memcached_url"

            [cache.redis]
            url = "redis_url"
            expiration = 100

            [cache.multilevel]
            chain = ["gcs", "memcached", "redis"]

            [preprocessor.cache]
            read_timeout = 100
            write_timeout = 100
            skip_check = false

            [preprocessor.cache.redis]
            url = "redis_url"
            expiration = 100
            key_prefix = "key_prefix"

            [preprocessor.cache.multilevel]
            chain = ["redis"]

            [dist]
            scheduler_url = "http://scheduler.url"

            [dist.net]
            connect_timeout = 100
            request_timeout = 100
            connection_pool = false
            max_connections = 100

            [dist.net.keepalive]
            enabled = false
            timeout = 100
            interval = 100
            "#,
        )?;

        assert_eq!(file_conf, config);

        Ok(())
    }

    #[test]
    fn configs_can_be_added() -> Result<()> {
        drop(env_logger::try_init());

        let mut a = Config {
            cache: Caches {
                skip_check: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let b = Config::from_vars([("SCCACHE_SKIP_CACHE_CHECK", "true")])
            .and_then(|config| config.validate_vars())?;

        let c = (&a + &b)?;

        assert_ne!(a, b);
        assert_eq!(c, b);

        a += &b;

        assert_eq!(a, b);
        assert_eq!(a, c);

        Ok(())
    }

    #[test]
    fn env_config_overrides_file_config() -> Result<()> {
        drop(env_logger::try_init());

        let mut file_conf = Config {
            cache: vec![
                Disk {
                    dir: "/file-cache".into(),
                    size: 15,
                    rw_mode: CacheMode::ReadWrite,
                    ..Default::default()
                }
                .into(),
                Memcached {
                    expiration: 24 * 3600,
                    key_prefix: String::new(),
                    ..Memcached::from_url("memurl")
                }
                .into(),
                Redis {
                    ttl: 25 * 3600,
                    key_prefix: String::new(),
                    ..Redis::from_url("myredisurl")
                }
                .into(),
            ]
            .into(),
            ..Default::default()
        };

        file_conf.cache.multilevel.chain =
            MultiLevel::chain_from_configs(&file_conf.cache.configs).into();

        let env_conf = Config::from_vars([
            ("SCCACHE_CACHE_DISK_DIR", "/env-cache"),
            ("SCCACHE_CACHE_REDIS_URL", "myotherredisurl"),
            ("SCCACHE_CACHE_REDIS_KEY_PREFIX", "/redis/prefix"),
            ("SCCACHE_CACHE_REDIS_DB", "10"),
            ("SCCACHE_CACHE_REDIS_USERNAME", "user"),
            ("SCCACHE_CACHE_REDIS_PASSWORD", "secret"),
        ])?;

        let mut merged_conf = Config {
            cache: vec![
                Disk {
                    dir: "/env-cache".into(),
                    rw_mode: CacheMode::ReadWrite,
                    ..Default::default()
                }
                .into(),
                Memcached {
                    expiration: 24 * 3600,
                    key_prefix: String::new(),
                    ..Memcached::from_url("memurl")
                }
                .into(),
                Redis {
                    key_prefix: "/redis/prefix".into(),
                    db: 10,
                    username: Some("user".to_owned()),
                    password: Some("secret".to_owned()),
                    ..Redis::from_url("myotherredisurl")
                }
                .into(),
            ]
            .into(),
            ..Default::default()
        };

        merged_conf.cache.multilevel = env_conf.cache.multilevel.clone();

        assert_eq!((file_conf + env_conf)?, merged_conf);

        Ok(())
    }

    #[test]
    fn config_file_from_path_or_sccache_conf() -> Result<()> {
        drop(env_logger::try_init());

        let dir = tempfile::tempdir()?;
        let conf_path = dir.path().join("conf.toml");
        let explicit_path = dir.path().join("explicit.toml");
        std::fs::write(&conf_path, "[cache.disk]\nsize = 1024\n")?;
        std::fs::write(&explicit_path, "[cache.disk]\nsize = 2048\n")?;

        let load = |vars: &[(&OsStr, &OsStr)], path: Option<&PathBuf>| {
            Config::from_envs_and_path::<_, _, _, &PathBuf>(vars.iter().copied(), path)
                .map(|config| config.cache.configs["disk"].clone())
        };
        let disk = |size| -> Cache {
            Disk {
                size,
                ..Default::default()
            }
            .into()
        };
        let sccache_conf = (OsStr::new("SCCACHE_CONF"), conf_path.as_os_str());
        let cache_size = (OsStr::new("SCCACHE_CACHE_SIZE"), OsStr::new("1G"));

        assert_eq!(load(&[sccache_conf], None)?, disk(1024));
        assert_eq!(load(&[sccache_conf, cache_size], None)?, disk(1 << 30));
        assert_eq!(load(&[sccache_conf], Some(&explicit_path))?, disk(2048));
        // A path that can't be opened falls back to SCCACHE_CONF
        let missing_path = dir.path().join("missing.toml");
        assert_eq!(load(&[sccache_conf], Some(&missing_path))?, disk(1024));

        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn non_utf8_env_var_fails() {
        use std::os::unix::ffi::OsStrExt;

        // Even when the variable isn't one of sccache's
        assert!(
            Config::from_vars([
                (OsStr::new("SCCACHE_CACHE_SIZE"), OsStr::new("1G")),
                (OsStr::new("UNRELATED"), OsStr::from_bytes(b"\xff")),
            ])
            .is_err()
        );
    }

    #[test]
    fn merge_switches_tagged_variant() -> Result<()> {
        drop(env_logger::try_init());

        let file_conf = Config::from_toml(
            r#"
            [dist.auth]
            type = "oauth2_code_grant_pkce"
            client_id = "file_client_id"
            auth_url = "file_auth_url"
            token_url = "file_token_url"
            "#,
        )?;

        let env_conf = Config::from_vars([
            ("SCCACHE_DIST_AUTH_TYPE", "oauth2_implicit"),
            ("SCCACHE_DIST_AUTH_CLIENT_ID", "env_client_id"),
            ("SCCACHE_DIST_AUTH_AUTH_URL", "env_auth_url"),
        ])?;

        assert_eq!(
            (file_conf + env_conf)?.dist.auth,
            dist::Auth::Oauth2Implicit {
                client_id: "env_client_id".into(),
                auth_url: "env_auth_url".into(),
            }
        );

        Ok(())
    }

    #[test]
    fn merge_env_equal_to_default_keeps_file_value() -> Result<()> {
        drop(env_logger::try_init());

        let file_conf = Config::from_toml("server_startup_timeout_ms = 5000")?;
        // The merge only applies what differs from Default, and 10000 is the default
        let env_conf = Config::from_vars([("SCCACHE_SERVER_STARTUP_TIMEOUT_MS", "10000")])?;

        assert_eq!((file_conf + env_conf)?.server_startup_timeout_ms, 5000);

        Ok(())
    }

    #[test]
    fn merge_chain_error_names_env_var() -> Result<()> {
        drop(env_logger::try_init());

        let file_conf = Config::from_toml(
            r#"
            [cache.custom]
            type = "redis"
            endpoint = "redis://127.0.0.1:6379"
            "#,
        )?;
        let env_conf = Config::from_vars([("SCCACHE_CACHE_MULTILEVEL_CHAIN", "nope")])?;

        let err = (file_conf + env_conf).unwrap_err().to_string();
        assert!(
            err.contains(
                r#"'nope' cache not configured, but specified in SCCACHE_CACHE_MULTILEVEL_CHAIN: ["nope"]"#
            ),
            "{err}"
        );

        Ok(())
    }

    #[test]
    fn merge_fails_on_conflicting_cache_types() -> Result<()> {
        drop(env_logger::try_init());

        let file_conf = Config::from_toml(
            r#"
            [cache.custom]
            type = "redis"
            endpoint = "redis://127.0.0.1:6379"
            "#,
        )?;
        let env_conf = Config::from_vars([
            ("SCCACHE_CACHE_CUSTOM_TYPE", "s3"),
            ("SCCACHE_CACHE_CUSTOM_BUCKET", "bucket"),
        ])?;

        // The merge patches the env's S3 fields over the file's Redis fields, and S3 rejects
        // the leftovers
        let err = (file_conf + env_conf).unwrap_err().to_string();
        assert!(err.contains("unknown field `cluster_endpoints`"), "{err}");

        Ok(())
    }

    #[test]
    fn skip_cache_check_from_env() -> Result<()> {
        drop(env_logger::try_init());

        const ENV_VAR: &str = "SCCACHE_SKIP_CACHE_CHECK";

        // skip_check defaults to false
        assert!(!Config::from_vars([("", ""); 0])?.cache.skip_check);

        for (value, expected) in [
            ("true", true),
            ("on", true),
            ("1", true),
            ("false", false),
            ("off", false),
            ("0", false),
        ] {
            assert_eq!(
                Config::from_vars([(ENV_VAR, value)])?.cache.skip_check,
                expected
            );
        }

        assert_eq!(
            Config::from_vars([(ENV_VAR, "invalid")])
                .unwrap_err()
                .to_string(),
            // "SCCACHE_SKIP_CACHE_CHECK must be 'true', 'on', '1', 'false', 'off' or '0'."
            "SCCACHE_CACHE_SKIP_CHECK: invalid value: 'invalid', expected 'true', 'on', '1', 'false', 'off' or '0'"
        );

        Ok(())
    }

    #[test]
    fn numeric_bools_from_env() -> Result<()> {
        drop(env_logger::try_init());

        // serde_env passes "1" and "0" on as integers, also inside the tagged `Cache` enum
        let config = Config::from_vars([
            ("SCCACHE_BUCKET", "bucket"),
            ("SCCACHE_S3_USE_SSL", "1"),
            ("SCCACHE_CACHE_CUSTOM_TYPE", "s3"),
            ("SCCACHE_CACHE_CUSTOM_BUCKET", "custom"),
            ("SCCACHE_CACHE_CUSTOM_USE_SSL", "0"),
        ])?;

        assert_eq!(
            config.cache.configs["s3"],
            S3 {
                use_ssl: Some(true),
                ..S3::from_bucket("bucket")
            }
            .into()
        );
        assert_eq!(
            config.cache.configs["custom"],
            S3 {
                use_ssl: Some(false),
                ..S3::from_bucket("custom")
            }
            .into()
        );

        Ok(())
    }

    #[test]
    fn dist_auth_token_from_vars() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_vars([
            ("SCCACHE_DIST_AUTH_TYPE", "token"),
            ("SCCACHE_DIST_AUTH_TOKEN", "secrettoken"),
        ])?;

        assert_eq!(
            config.dist.auth,
            dist::Auth::Token {
                token: "secrettoken".to_owned()
            }
        );

        Ok(())
    }

    #[test]
    fn dist_auth_token_from_toml() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_toml(
            r#"
            [dist.auth]
            type = "token"
            token = "secrettoken"
            "#,
        )?;

        assert_eq!(
            config.dist.auth,
            dist::Auth::Token {
                token: "secrettoken".into()
            }
        );

        Ok(())
    }

    #[test]
    fn dist_auth_oauth2_implicit_from_vars() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_vars([
            ("SCCACHE_DIST_AUTH_TYPE", "oauth2_implicit"),
            ("SCCACHE_DIST_AUTH_CLIENT_ID", "client_id"),
            ("SCCACHE_DIST_AUTH_AUTH_URL", "auth_url"),
        ])?;

        assert_eq!(
            config.dist.auth,
            dist::Auth::Oauth2Implicit {
                client_id: "client_id".into(),
                auth_url: "auth_url".into()
            }
        );

        Ok(())
    }

    #[test]
    fn dist_auth_oauth2_implicit_from_toml() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_toml(
            r#"
            [dist.auth]
            type = "oauth2_implicit"
            client_id = "client_id"
            auth_url = "auth_url"
            "#,
        )?;

        assert_eq!(
            config.dist.auth,
            dist::Auth::Oauth2Implicit {
                client_id: "client_id".into(),
                auth_url: "auth_url".into()
            }
        );

        Ok(())
    }

    #[test]
    fn dist_auth_oauth2_code_grant_pkce_from_vars() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_vars([
            ("SCCACHE_DIST_AUTH_TYPE", "oauth2_code_grant_pkce"),
            ("SCCACHE_DIST_AUTH_CLIENT_ID", "client_id"),
            ("SCCACHE_DIST_AUTH_AUTH_URL", "auth_url"),
            ("SCCACHE_DIST_AUTH_TOKEN_URL", "token_url"),
        ])?;

        assert_eq!(
            config.dist.auth,
            dist::Auth::Oauth2CodeGrantPKCE {
                client_id: "client_id".into(),
                auth_url: "auth_url".into(),
                token_url: "token_url".into(),
            }
        );

        Ok(())
    }

    #[test]
    fn dist_auth_oauth2_code_grant_pkce_from_toml() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_toml(
            r#"
            [dist.auth]
            type = "oauth2_code_grant_pkce"
            client_id = "client_id"
            auth_url = "auth_url"
            token_url = "token_url"
            "#,
        )?;

        assert_eq!(
            config.dist.auth,
            dist::Auth::Oauth2CodeGrantPKCE {
                client_id: "client_id".into(),
                auth_url: "auth_url".into(),
                token_url: "token_url".into(),
            }
        );

        Ok(())
    }

    #[test]
    fn human_units_parse() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_toml(
            r#"
            [cache.disk]
            size = "7g"

            [dist]
            toolchain_cache_size = "5g"
            "#,
        )?;

        assert_eq!(
            config,
            Config {
                cache: vec![
                    Disk {
                        size: 7 * 1024 * 1024 * 1024,
                        ..Default::default()
                    }
                    .into()
                ]
                .into(),
                dist: dist::Config {
                    toolchain_cache_size: 5 * 1024 * 1024 * 1024,
                    ..Default::default()
                },
                ..Default::default()
            }
        );

        Ok(())
    }

    #[test]
    fn cache_mode_config_from_env() {
        drop(env_logger::try_init());

        struct CacheEnvCheckInfo {
            env_var_value: Option<String>,
            expected_cache_mode: Option<CacheMode>,
        }

        let env_check_infos = vec![
            CacheEnvCheckInfo {
                env_var_value: None,
                expected_cache_mode: None,
                // expected_cache_mode: Some(CacheModeConfig::ReadWrite),
            },
            CacheEnvCheckInfo {
                env_var_value: Some(String::new()),
                expected_cache_mode: None,
            },
            CacheEnvCheckInfo {
                env_var_value: Some("READ_ONLY".to_string()),
                expected_cache_mode: Some(CacheMode::ReadOnly),
            },
            CacheEnvCheckInfo {
                env_var_value: Some("read_only".to_string()),
                expected_cache_mode: Some(CacheMode::ReadOnly),
            },
            CacheEnvCheckInfo {
                env_var_value: Some("ReAd_ONly".to_string()),
                expected_cache_mode: Some(CacheMode::ReadOnly),
            },
            CacheEnvCheckInfo {
                env_var_value: Some("READ_WRITE".to_string()),
                expected_cache_mode: Some(CacheMode::ReadWrite),
            },
            CacheEnvCheckInfo {
                env_var_value: Some("read_write".to_string()),
                expected_cache_mode: Some(CacheMode::ReadWrite),
            },
            CacheEnvCheckInfo {
                env_var_value: Some("read_WRITE".to_string()),
                expected_cache_mode: Some(CacheMode::ReadWrite),
            },
            CacheEnvCheckInfo {
                env_var_value: Some("read_only_typo".to_string()),
                expected_cache_mode: None,
            },
            CacheEnvCheckInfo {
                env_var_value: Some("foo".to_string()),
                expected_cache_mode: None,
            },
            CacheEnvCheckInfo {
                env_var_value: Some("any_unsupported_value".to_string()),
                expected_cache_mode: None,
            },
        ];

        #[derive(Deserialize)]
        struct EnvCfg {
            rw_mode: Option<CacheMode>,
        }

        let var_name = "RW_MODE";

        for info in env_check_infos {
            let cache_mode = if let Some(env_var_value) = info.env_var_value.as_deref() {
                serde_env::from_iter::<_, &str, EnvCfg>([(var_name, env_var_value)])
            } else {
                serde_env::from_iter::<_, &str, EnvCfg>([])
            };

            let cache_mode = cache_mode.ok().and_then(|env| env.rw_mode);

            assert_eq!(cache_mode, info.expected_cache_mode);
        }
    }

    mod basedirs {
        use super::*;

        #[test]
        #[cfg(target_os = "windows")]
        fn config_basedirs_overrides() -> Result<()> {
            drop(env_logger::try_init());

            // Test that env variable takes precedence over file config
            let env_conf = Config {
                basedirs: ["C:/env/basedir"].into(),
                ..Default::default()
            };

            let file_conf = Config {
                basedirs: ["C:/file/basedir"].into(),
                ..Default::default()
            };

            let config = (file_conf + env_conf)?;
            assert_eq!(config.basedirs, vec![b"c:/env/basedir/".to_vec()]);

            // Test that file config is used when env is None
            let env_conf = Config::default();

            let file_conf = Config {
                basedirs: ["C:/file/basedir"].into(),
                ..Default::default()
            };

            let config = (file_conf + env_conf)?;
            assert_eq!(config.basedirs, vec![b"c:/file/basedir/".to_vec()]);

            // Test that env config is used when env is set but empty
            let env_conf = Config {
                basedirs: Basedirs::empty(),
                ..Default::default()
            };

            let file_conf = Config {
                basedirs: ["C:/file/basedir"].into(),
                ..Default::default()
            };

            let config = (file_conf + env_conf)?;
            assert!(config.basedirs.is_empty());

            // Test that both empty results in empty
            let env_conf = Config::default();

            let file_conf = Config::default();

            let config = (file_conf + env_conf)?;
            assert!(config.basedirs.is_empty());

            Ok(())
        }

        #[test]
        #[cfg(not(target_os = "windows"))]
        fn config_basedirs_overrides() -> Result<()> {
            drop(env_logger::try_init());

            // Test that env variable takes precedence over file config
            let env_conf = Config {
                basedirs: ["/env/basedir"].into(),
                ..Default::default()
            };

            let file_conf = Config {
                basedirs: ["/file/basedir"].into(),
                ..Default::default()
            };

            let config = (file_conf + env_conf)?;
            assert_eq!(config.basedirs, vec![b"/env/basedir/".to_vec()]);

            // Test that file config is used when env is None
            let env_conf = Config::default();

            let file_conf = Config {
                basedirs: ["/file/basedir"].into(),
                ..Default::default()
            };

            let config = (file_conf + env_conf)?;
            assert_eq!(config.basedirs, vec![b"/file/basedir/".to_vec()]);

            // Test that env config is used when env is set but empty
            let env_conf = Config {
                basedirs: Basedirs::empty(),
                ..Default::default()
            };

            let file_conf = Config {
                basedirs: ["/file/basedir"].into(),
                ..Default::default()
            };

            let config = (file_conf + env_conf)?;
            assert!(config.basedirs.is_empty());

            // Test that both empty results in empty
            let env_conf = Config::default();

            let file_conf = Config::default();

            let config = (file_conf + env_conf)?;
            assert!(config.basedirs.is_empty());

            Ok(())
        }

        #[test]
        #[cfg(not(target_os = "windows"))]
        fn deserialize_basedirs() -> Result<()> {
            drop(env_logger::try_init());

            // Test array of paths
            let config = Config::from_toml(
                r#"
                basedirs = ["/home/user/project", "/home/user/workspace"]
                "#,
            )?;
            assert_eq!(
                config.basedirs,
                vec![
                    b"/home/user/project/".to_vec(),
                    b"/home/user/workspace/".to_vec()
                ]
            );

            Ok(())
        }

        #[test]
        fn deserialize_basedirs_missing() -> Result<()> {
            drop(env_logger::try_init());

            // Test no basedirs specified (should default to empty vec)
            let config = Config::from_toml(
                r#"
                [cache.disk]
                dir = "/tmp/cache"
                size = 1073741824
                "#,
            )?;
            assert!(config.basedirs.is_empty());
            Ok(())
        }

        #[test]
        #[cfg(target_os = "windows")]
        fn env_basedirs_single() -> Result<()> {
            drop(env_logger::try_init());
            let config = Config::from_vars([("SCCACHE_BASEDIRS", "C:/home/user/project")])?;
            assert_eq!(config.basedirs, vec![b"c:/home/user/project/".to_vec()]);
            Ok(())
        }

        #[test]
        #[cfg(not(target_os = "windows"))]
        fn env_basedirs_single() -> Result<()> {
            drop(env_logger::try_init());
            let config = Config::from_vars([("SCCACHE_BASEDIRS", "/home/user/project")])?;
            assert_eq!(config.basedirs, vec![b"/home/user/project/".to_vec()]);
            Ok(())
        }

        #[test]
        #[cfg(target_os = "windows")]
        fn env_basedirs_multiple() -> Result<()> {
            drop(env_logger::try_init());
            let config = Config::from_vars([(
                "SCCACHE_BASEDIRS",
                "C:/home/user/project;C:/home/user/workspace",
            )])?;

            assert_eq!(
                config.basedirs,
                vec![
                    b"c:/home/user/project/".to_vec(),
                    b"c:/home/user/workspace/".to_vec()
                ]
            );

            Ok(())
        }

        #[test]
        #[cfg(not(target_os = "windows"))]
        fn env_basedirs_multiple() -> Result<()> {
            drop(env_logger::try_init());
            let config = Config::from_vars([(
                "SCCACHE_BASEDIRS",
                "/home/user/project:/home/user/workspace",
            )])?;

            assert_eq!(
                config.basedirs,
                vec![
                    b"/home/user/project/".to_vec(),
                    b"/home/user/workspace/".to_vec()
                ]
            );

            Ok(())
        }

        #[test]
        #[cfg(target_os = "windows")]
        fn env_basedirs_with_spaces() -> Result<()> {
            drop(env_logger::try_init());
            // Test that spaces around paths are not trimmed
            // The lead to trailing spaces are preserved and server fails to start
            Config::from_vars([(
                "SCCACHE_BASEDIRS",
                " C:/home/user/project ; C:/home/user/workspace ",
            )])
            .expect_err("Should fail due to non-absolute path");

            Ok(())
        }

        #[test]
        #[cfg(not(target_os = "windows"))]
        fn env_basedirs_with_spaces() -> Result<()> {
            drop(env_logger::try_init());
            // Test that spaces around paths are not trimmed
            // The lead to trailing spaces are preserved and server fails to start
            Config::from_vars([(
                "SCCACHE_BASEDIRS",
                " /home/user/project : /home/user/workspace ",
            )])
            .expect_err("Should fail due to non-absolute path");

            Ok(())
        }

        #[test]
        #[cfg(target_os = "windows")]
        fn env_basedirs_empty_entries() -> Result<()> {
            drop(env_logger::try_init());
            // Test that empty entries are filtered out
            let config = Config::from_vars([(
                "SCCACHE_BASEDIRS",
                "c:/home/user/project;;c:/home/user/workspace",
            )])?;

            assert_eq!(
                config.basedirs,
                vec![
                    b"c:/home/user/project/".to_vec(),
                    b"c:/home/user/workspace/".to_vec()
                ]
            );

            Ok(())
        }

        #[test]
        #[cfg(not(target_os = "windows"))]
        fn env_basedirs_empty_entries() -> Result<()> {
            drop(env_logger::try_init());
            // Test that empty entries are filtered out
            let config = Config::from_vars([(
                "SCCACHE_BASEDIRS",
                "/home/user/project::/home/user/workspace",
            )])?;

            assert_eq!(
                config.basedirs,
                vec![
                    b"/home/user/project/".to_vec(),
                    b"/home/user/workspace/".to_vec()
                ]
            );

            Ok(())
        }

        #[test]
        fn env_basedirs_not_set() -> Result<()> {
            drop(env_logger::try_init());
            assert!(Config::from_vars([("", ""); 0])?.basedirs.is_empty());
            Ok(())
        }

        // Integration tests: Config normalization + strip_basedirs usage
        mod integration {
            use super::*;

            use crate::util::strip_basedirs;
            use std::borrow::Cow;

            #[test]
            #[cfg(not(target_os = "windows"))]
            fn config_normalizes_and_strips() -> Result<()> {
                // Test that Config normalizes basedirs and strip_basedirs uses them correctly
                drop(env_logger::try_init());

                let config = Config::from_vars([("SCCACHE_BASEDIRS", "/home/user/project")])?;

                // Verify config normalized the basedir with trailing slash
                assert_eq!(config.basedirs, vec![b"/home/user/project/".to_vec()]);

                // Test that strip_basedirs uses the normalized basedir
                let input = b"# 1 \"/home/user/project/src/main.c\"\nint main() { return 0; }";
                let output = strip_basedirs(input, &config.basedirs);

                // Should strip the basedir
                let expected = b"# 1 \"src/main.c\"\nint main() { return 0; }";
                assert_eq!(&*output, expected);
                assert!(matches!(output, Cow::Owned(_)));

                Ok(())
            }

            #[test]
            #[cfg(not(target_os = "windows"))]
            fn normalized_path_with_double_slashes() -> Result<()> {
                // Test that Config normalizes paths with double slashes
                drop(env_logger::try_init());

                let config = Config::from_vars([("SCCACHE_BASEDIRS", "/home//user///project/")])?;

                // Config should normalize to single slashes with one trailing slash
                assert_eq!(config.basedirs, vec![b"/home/user/project/".to_vec()]);

                // Verify it works with strip_basedirs
                let input = b"# 1 \"/home/user/project/src/main.c\"";
                let output = strip_basedirs(input, &config.basedirs);
                assert_eq!(&*output, b"# 1 \"src/main.c\"");

                Ok(())
            }

            #[test]
            #[cfg(target_os = "windows")]
            fn windows_path_normalization() -> Result<()> {
                // Test that Config normalizes Windows paths correctly
                drop(env_logger::try_init());

                let config = Config::from_vars([("SCCACHE_BASEDIRS", r"C:\Users\Test\Project")])?;

                // Should be normalized to lowercase with forward slashes
                assert_eq!(config.basedirs, vec![b"c:/users/test/project/".to_vec()]);

                // Test with mixed case preprocessor output
                let input = b"# 1 \"C:\\Users\\Test\\Project\\src\\main.c\"";
                let output = strip_basedirs(input, &config.basedirs);
                assert_eq!(&*output, b"# 1 \"src\\main.c\"");

                Ok(())
            }

            #[test]
            #[cfg(not(target_os = "windows"))]
            fn cow_borrowed_when_no_match() -> Result<()> {
                // Test that strip_basedirs returns Cow::Borrowed when no stripping occurs
                drop(env_logger::try_init());

                let config = Config::from_vars([("SCCACHE_BASEDIRS", "/home/user/project")])?;

                // Input doesn't contain the basedir
                let input = b"# 1 \"/other/path/main.c\"\nint main() { return 0; }";
                let output = strip_basedirs(input, &config.basedirs);

                // Should return borrowed reference (no allocation)
                assert!(matches!(output, Cow::Borrowed(_)));
                assert_eq!(&*output, input);

                Ok(())
            }

            #[test]
            #[cfg(not(target_os = "windows"))]
            fn cow_borrowed_when_empty_basedirs() -> Result<()> {
                // Test that strip_basedirs returns Cow::Borrowed when basedirs is empty
                drop(env_logger::try_init());

                let config = Config::from_vars([("", ""); 0])?;

                assert!(config.basedirs.is_empty());

                let input = b"# 1 \"/home/user/project/src/main.c\"";
                let output = strip_basedirs(input, &config.basedirs);

                // Should return borrowed reference when basedirs is empty
                assert!(matches!(output, Cow::Borrowed(_)));
                assert_eq!(&*output, input);

                Ok(())
            }

            #[test]
            #[cfg(not(target_os = "windows"))]
            fn multiple_basedirs_longest_match() -> Result<()> {
                // Test that strip_basedirs prefers longest match with normalized basedirs
                drop(env_logger::try_init());

                let config =
                    Config::from_vars([("SCCACHE_BASEDIRS", "/home/user:/home/user/project")])?;

                // Both should be normalized with trailing slashes
                assert_eq!(
                    config.basedirs,
                    vec![b"/home/user/".to_vec(), b"/home/user/project/".to_vec()]
                );

                // Input matches both, but longest should win
                let input = b"# 1 \"/home/user/project/src/main.c\"";
                let output = strip_basedirs(input, &config.basedirs);

                // Should match the longest basedir (/home/user/project/)
                let expected = b"# 1 \"src/main.c\"";
                assert_eq!(&*output, expected);

                Ok(())
            }

            #[test]
            #[cfg(not(target_os = "windows"))]
            fn paths_with_dots_normalized() -> Result<()> {
                // Test that paths with . and .. are normalized correctly
                drop(env_logger::try_init());

                let config =
                    Config::from_vars([("SCCACHE_BASEDIRS", "/home/user/./project/../project")])?;

                // Should be normalized to remove ./ and ../
                assert_eq!(config.basedirs, vec![b"/home/user/project/".to_vec()]);

                // Verify it works with strip_basedirs
                let input = b"# 1 \"/home/user/project/src/main.c\"";
                let output = strip_basedirs(input, &config.basedirs);
                let expected = b"# 1 \"src/main.c\"";
                assert_eq!(&*output, expected);

                Ok(())
            }

            #[test]
            #[cfg(target_os = "windows")]
            fn windows_mixed_slashes() -> Result<()> {
                // Test Windows path with mixed slashes in preprocessor output
                drop(env_logger::try_init());

                let config = Config::from_vars([("SCCACHE_BASEDIRS", r"C:\Users\test\project")])?;

                assert_eq!(config.basedirs, vec![b"c:/users/test/project/".to_vec()]);

                // Preprocessor output with mixed slashes
                let input = b"# 1 \"C:/Users\\test\\project\\src/main.c\"";
                let output = strip_basedirs(input, &config.basedirs);

                // Should strip despite mixed slashes
                let expected = b"# 1 \"src/main.c\"";
                assert_eq!(&*output, expected);
                assert!(matches!(output, Cow::Owned(_)));

                Ok(())
            }

            #[test]
            #[cfg(not(target_os = "windows"))]
            fn env_variable_to_strip() -> Result<()> {
                // Test full flow: SCCACHE_BASEDIRS env var -> Config -> strip_basedirs
                drop(env_logger::try_init());

                let config =
                    Config::from_vars([("SCCACHE_BASEDIRS", "/home/user/project:/tmp/build")])?;

                // Should have two normalized basedirs
                assert_eq!(
                    config.basedirs,
                    vec![b"/home/user/project/".to_vec(), b"/tmp/build/".to_vec()]
                );

                // Test stripping with both
                let input1 = b"# 1 \"/home/user/project/src/main.c\"";
                let output1 = strip_basedirs(input1, &config.basedirs);
                assert_eq!(&*output1, b"# 1 \"src/main.c\"");

                let input2 = b"# 1 \"/tmp/build/obj/file.o\"";
                let output2 = strip_basedirs(input2, &config.basedirs);
                assert_eq!(&*output2, b"# 1 \"obj/file.o\"");

                Ok(())
            }
        }
    }

    #[cfg(feature = "s3")]
    mod s3 {

        use super::*;
        use serial_test::serial;

        #[test]
        fn config_from_env() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_CLIENT_SIDE_MODE", "true"),
                ("SCCACHE_BUCKET", "my-bucket"),
                ("SCCACHE_REGION", "us-east-2"),
                ("SCCACHE_S3_KEY_PREFIX", "prefix"),
                ("SCCACHE_S3_NO_CREDENTIALS", "0"),
                ("SCCACHE_S3_RW_MODE", "ReAd_ONly"),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config,
                Config {
                    client_side_mode: true,
                    cache: vec![
                        S3 {
                            region: Some("us-east-2".into()),
                            key_prefix: "prefix".into(),
                            no_credentials: false,
                            rw_mode: CacheMode::ReadOnly,
                            ..S3::from_bucket("my-bucket")
                        }
                        .into()
                    ]
                    .into(),
                    ..Default::default()
                }
            );

            Ok(())
        }

        #[test]
        fn config_from_env_custom_name() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_CACHE_CUSTOM_S3_CACHE_TYPE", "s3"),
                ("SCCACHE_CACHE_CUSTOM_S3_CACHE_BUCKET", "my-bucket"),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config,
                Config {
                    cache: Caches {
                        configs: [(
                            "custom_s3_cache".into(),
                            Cache::S3(S3::from_bucket("my-bucket"))
                        )]
                        .into_iter()
                        .collect(),
                        multilevel: MultiLevel {
                            chain: vec!["custom_s3_cache".into()].into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    ..Default::default()
                }
            );

            Ok(())
        }

        #[test]
        #[serial(config_from_env)]
        fn no_credentials_conflict() -> Result<()> {
            drop(env_logger::try_init());

            unsafe {
                std::env::set_var("AWS_ACCESS_KEY_ID", "aws-access-key-id");
                std::env::set_var("AWS_SECRET_ACCESS_KEY", "aws-secret-access-key");
            }

            let err_str = Config::from_vars([
                ("SCCACHE_S3_NO_CREDENTIALS", "true"),
                ("SCCACHE_BUCKET", "my-bucket"),
            ])
            .unwrap_err()
            .to_string();

            unsafe {
                std::env::remove_var("AWS_ACCESS_KEY_ID");
                std::env::remove_var("AWS_SECRET_ACCESS_KEY");
            }

            assert_eq!(
                "If setting S3 credentials, SCCACHE_CACHE_S3_NO_CREDENTIALS must not be set.",
                err_str
            );

            Ok(())
        }

        #[test]
        #[serial(config_from_env)]
        fn no_credentials_invalid() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_S3_NO_CREDENTIALS", "yes"),
                ("SCCACHE_BUCKET", "my-bucket"),
            ])
            .and_then(|config| config.validate_vars());

            assert_eq!(
                "SCCACHE_CACHE_S3_NO_CREDENTIALS: invalid value: 'yes', expected 'true', 'on', '1', 'false', 'off' or '0'",
                config.unwrap_err().to_string()
            );

            Ok(())
        }

        #[test]
        #[serial(config_from_env)]
        fn no_credentials_valid_true() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_S3_NO_CREDENTIALS", "true"),
                ("SCCACHE_BUCKET", "my-bucket"),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config.cache,
                vec![
                    S3 {
                        no_credentials: true,
                        ..S3::from_bucket("my-bucket")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }

        #[test]
        fn no_credentials_valid_false() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_S3_NO_CREDENTIALS", "false"),
                ("SCCACHE_BUCKET", "my-bucket"),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config.cache,
                vec![
                    S3 {
                        no_credentials: false,
                        ..S3::from_bucket("my-bucket")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }

        #[test]
        fn sse_kms_from_env() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_BUCKET", "my-bucket"),
                ("SCCACHE_S3_SERVER_SIDE_ENCRYPTION_AWS_KMS", "true"),
                (
                    "SCCACHE_S3_SERVER_SIDE_ENCRYPTION_KMS_KEY_ID",
                    "arn:aws:kms:us-east-1:111:key/abc",
                ),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config.cache,
                vec![
                    S3 {
                        server_side_encryption_aws_kms: Some(true),
                        server_side_encryption_kms_key_id: Some(
                            "arn:aws:kms:us-east-1:111:key/abc".into()
                        ),
                        ..S3::from_bucket("my-bucket")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }
    }

    #[cfg(feature = "azure")]
    mod azure {
        use super::*;

        #[test]
        fn entra_storage_account_enables() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_AZURE_RW_MODE", "READ_WRITE"),
                ("SCCACHE_AZURE_BLOB_CONTAINER", "my-container"),
                ("SCCACHE_AZURE_STORAGE_ACCOUNT", "mystorageacct"),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config.cache,
                vec![
                    Azure {
                        auth: AzureAuth::StorageAccount {
                            storage_account: "mystorageacct".into()
                        },
                        ..Azure::from_container("my-container")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }

        #[test]
        fn entra_endpoint_enables() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_AZURE_BLOB_CONTAINER", "my-container"),
                (
                    "SCCACHE_AZURE_ENDPOINT",
                    "https://acct.blob.core.usgovcloudapi.net",
                ),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config.cache,
                vec![
                    Azure {
                        auth: AzureAuth::Endpoint {
                            endpoint: "https://acct.blob.core.usgovcloudapi.net".into()
                        },
                        ..Azure::from_container("my-container")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }

        #[test]
        fn connection_string_still_works() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_AZURE_BLOB_CONTAINER", "my-container"),
                ("SCCACHE_AZURE_CONNECTION_STRING", "some-connection-string"),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config.cache,
                vec![
                    Azure {
                        auth: AzureAuth::SharedKey {
                            connection_string: "some-connection-string".into()
                        },
                        ..Azure::from_container("my-container")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }

        #[test]
        fn conflicting_auth_sources() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_AZURE_BLOB_CONTAINER", "my-container"),
                ("SCCACHE_AZURE_CONNECTION_STRING", "some-connection-string"),
                ("SCCACHE_AZURE_STORAGE_ACCOUNT", "mystorageacct"),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(config, Config::default().validate_vars()?);

            Ok(())
        }

        #[test]
        fn conflicting_connection_string_and_endpoint() -> Result<()> {
            drop(env_logger::try_init());

            // The other operand of the mutual-exclusivity check: connection string paired
            // with an endpoint (rather than a storage account) must also be rejected.
            let config = Config::from_vars([
                ("SCCACHE_AZURE_BLOB_CONTAINER", "my-container"),
                ("SCCACHE_AZURE_CONNECTION_STRING", "some-connection-string"),
                (
                    "SCCACHE_AZURE_ENDPOINT",
                    "https://acct.blob.core.windows.net",
                ),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(config, Config::default().validate_vars()?);

            Ok(())
        }

        #[test]
        #[cfg(feature = "azure")]
        fn no_auth_source_disables_backend() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([("SCCACHE_AZURE_BLOB_CONTAINER", "my-container")])?;
            // A container with no auth source disables Azure (backwards compatible)
            // rather than failing the whole config load.
            assert!(config.cache.configs.is_empty());
            Ok(())
        }

        #[test]
        fn toml_endpoint_field() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_toml(
                r#"
                [cache.azure]
                container = "c"
                key_prefix = "p"
                endpoint = "https://acct.blob.core.usgovcloudapi.net"
                "#,
            )?;

            assert_eq!(
                config.cache,
                vec![
                    Azure {
                        auth: AzureAuth::Endpoint {
                            endpoint: "https://acct.blob.core.usgovcloudapi.net".into()
                        },
                        key_prefix: "p".into(),
                        ..Azure::from_container("c")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }

        #[test]
        fn toml_connection_string_field() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_toml(
                r#"
                [cache.azure]
                container = "c"
                key_prefix = "p"
                connection_string = "conn"
                "#,
            )?;

            assert_eq!(
                config.cache,
                vec![
                    Azure {
                        auth: AzureAuth::SharedKey {
                            connection_string: "conn".into()
                        },
                        key_prefix: "p".into(),
                        ..Azure::from_container("c")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }

        #[test]
        fn toml_container_only_deserializes_with_no_auth() -> Result<()> {
            // A container-only block is valid TOML and leaves every auth source None
            // (the file surface, unlike the env parser, does not warn-and-disable — the
            // missing-auth error is raised later at operator build, see
            // AzureBlobCache::build's test_build_requires_an_auth_source).

            drop(env_logger::try_init());

            let config = Config::from_toml(
                r#"
                [cache.azure]
                container = "c"
                key_prefix = "p"
                "#,
            )?;

            assert_eq!(
                config.cache,
                vec![
                    Azure {
                        key_prefix: "p".into(),
                        ..Azure::from_container("c")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }
    }

    #[cfg(feature = "gcs")]
    mod gcs {
        use super::*;

        #[test]
        fn service_account_from_env() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_GCS_BUCKET", "my-bucket"),
                ("SCCACHE_GCS_SERVICE_ACCOUNT", "my@example.com"),
                ("SCCACHE_GCS_RW_MODE", "READ_WRITE"),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config.cache,
                vec![
                    GCS {
                        service_account: Some("my@example.com".into()),
                        rw_mode: CacheMode::ReadWrite,
                        ..GCS::from_bucket("my-bucket")
                    }
                    .into()
                ]
                .into()
            );

            Ok(())
        }

        #[test]
        fn credentials_require_bucket_from_env() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_vars([
                ("SCCACHE_DIR", "/tmp/disk"),
                ("SCCACHE_GCS_KEY_PATH", "foo.json"),
            ])
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config,
                Config {
                    cache: vec![
                        Disk {
                            dir: "/tmp/disk".into(),
                            ..Default::default()
                        }
                        .into(),
                        GCS {
                            enabled: false,
                            key_path: Some("foo.json".into()),
                            ..GCS::from_bucket("")
                        }
                        .into()
                    ]
                    .into(),
                    ..Default::default()
                }
            );

            Ok(())
        }

        #[test]
        fn credentials_require_bucket_from_toml() -> Result<()> {
            drop(env_logger::try_init());

            let config = Config::from_toml(
                r#"
                [cache.disk]
                dir = "/tmp/disk"
                size = 1024

                [cache.gcs]
                key_path = "foo.json"
                "#,
            )
            .and_then(|config| config.validate_vars())?;

            assert_eq!(
                config,
                Config {
                    cache: vec![
                        Disk {
                            dir: "/tmp/disk".into(),
                            size: 1024,
                            ..Default::default()
                        }
                        .into(),
                        GCS {
                            enabled: false,
                            key_path: Some("foo.json".into()),
                            ..GCS::from_bucket("")
                        }
                        .into()
                    ]
                    .into(),
                    ..Default::default()
                }
            );
            Ok(())
        }
    }

    mod multilevel {
        use super::*;

        #[test]
        fn toml_parsing() -> Result<()> {
            // Test parsing cache levels from config
            drop(env_logger::try_init());

            let config = Config::from_toml(
                r#"
                [cache.disk]
                dir = "/tmp/disk"
                size = 1024

                [cache.s3]
                bucket = "my-bucket"
                region = "us-west-2"
                no_credentials = false

                [cache.redis]
                endpoint = "redis://localhost"

                [cache.multilevel]
                chain = ["disk", "redis", "s3"]
                "#,
            )?;

            assert_eq!(
                config.cache.multilevel.chain.unwrap_or_default(),
                vec!["disk".to_string(), "redis".to_string(), "s3".to_string()]
            );

            Ok(())
        }

        #[test]
        fn backward_compatibility() -> Result<()> {
            // Test that configs without levels still work (single cache selection)
            drop(env_logger::try_init());

            let config = Config::from_toml(
                r#"
                [cache.s3]
                bucket = "my-bucket"
                region = "us-west-2"
                no_credentials = false
                "#,
            )?;

            assert!(
                config
                    .cache
                    .configs
                    .iter()
                    .any(|(_, c)| matches!(c, Cache::S3(..)))
            );

            Ok(())
        }

        #[test]
        fn invalid_level_from_env() -> Result<()> {
            drop(env_logger::try_init());

            let err_str = Config::from_vars([("SCCACHE_CACHE_MULTILEVEL_CHAIN", "unknown_cache")])
                .and_then(|conf| conf.validate_vars())
                .unwrap_err()
                .to_string();

            eprintln!("{err_str}");

            assert!(err_str.contains(
                r#"'unknown_cache' cache not configured, but specified in SCCACHE_CACHE_MULTILEVEL_CHAIN: ["unknown_cache"]"#
            ));
            Ok(())
        }

        #[test]
        fn invalid_level_from_toml() -> Result<()> {
            drop(env_logger::try_init());

            let err_str = Config::from_toml(
                r#"
                [cache.multilevel]
                chain = ["unknown_cache"]
                "#,
            )
            .unwrap_err()
            .to_string();

            eprintln!("{err_str}");

            assert!(err_str.contains(
                r#"'unknown_cache' cache not configured, but specified in cache.multilevel.chain: ["unknown_cache"]"#
            ));
            Ok(())
        }

        #[test]
        fn missing_config_from_toml() {
            drop(env_logger::try_init());

            let err_str = Config::from_toml(
                r#"
                [cache.multilevel]
                chain = ["s3"]
                "#,
            )
            .unwrap_err()
            .to_string();

            eprintln!("{err_str}");

            assert!(err_str.contains(
                r#"'s3' cache not configured, but specified in cache.multilevel.chain: ["s3"]"#
            ));
        }
    }
}
