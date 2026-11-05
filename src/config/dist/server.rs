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

use crate::{
    config::{
        Loadable, Valid,
        cache::{Azure, AzureAuth, Cache, Caches},
        dist::{MessageBroker, Metrics},
        utils::{deserialize_command_or_list, deserialize_string_or_seq_to_map},
    },
    errors::*,
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_with::{FromInto, PickFirst, StringWithSeparator, formats::ColonSeparator, serde_as};
use std::{
    collections::HashMap,
    ffi::OsString,
    net::SocketAddr,
    path::{Path, PathBuf},
};

#[serde_as]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub jobs: Caches,

    #[serde(default)]
    pub toolchains: Caches,

    #[serde(default)]
    pub builder: Builder,

    #[serde(default = "defaults::default_disk_cache_dir")]
    pub cache_dir: PathBuf,

    #[serde(default)]
    pub health_check_bind_addr: Option<SocketAddr>,

    #[serde(default = "defaults::default_heartbeat_interval")]
    pub heartbeat_interval_ms: u64,

    #[serde_as(as = "FromInto<f64>")]
    #[serde(default)]
    pub max_per_core_load: Percent,

    #[serde_as(as = "FromInto<f64>")]
    #[serde(default)]
    pub max_per_core_prefetch: Percent,

    #[serde(default)]
    pub message_broker: Option<MessageBroker>,

    #[serde(default)]
    pub metrics: Metrics,

    #[serde(alias = "server_id", default)]
    pub id: String,

    #[serde(default = "defaults::default_shutdown_timeout")]
    pub shutdown_timeout_secs: u64,

    #[serde(default = "defaults::default_disk_cache_size")]
    pub toolchain_cache_size: u64,
}

impl Config {
    pub fn load<P: AsRef<Path>, O: Into<Option<P>>>(path: O) -> crate::errors::Result<Self> {
        Self::load_from_path(path.into().as_ref().map(AsRef::as_ref))
    }

    // Not generic, so sccache-dist calls this crate's copy of the config loader instead of
    // compiling its own (release builds don't share generic instances between crates).
    fn load_from_path(path: Option<&Path>) -> crate::errors::Result<Self> {
        Self::from_envs_and_path::<_, _, _, &Path>(std::env::vars_os(), path).map(|mut conf| {
            if conf.id.is_empty() {
                conf.id = defaults::default_server_id();
            }
            conf
        })
    }
}

// Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
impl Default for Config {
    fn default() -> Self {
        Self {
            jobs: Caches::default(),
            toolchains: Caches::default(),
            builder: Builder::default(),
            cache_dir: defaults::default_disk_cache_dir(),
            health_check_bind_addr: None,
            heartbeat_interval_ms: defaults::default_heartbeat_interval(),
            max_per_core_load: Percent::default(),
            max_per_core_prefetch: Percent::default(),
            message_broker: None,
            metrics: Metrics::default(),
            id: String::new(),
            shutdown_timeout_secs: defaults::default_shutdown_timeout(),
            toolchain_cache_size: defaults::default_disk_cache_size(),
        }
    }
}

impl Valid for Config {
    fn validate_merged(mut self, orig: &Self) -> Result<Self> {
        for (name, orig_cache) in orig.jobs.configs.iter() {
            if !self.jobs.configs.contains_key(name) {
                self.jobs.configs.insert(name.clone(), orig_cache.clone());
            }
        }

        for (name, orig_cache) in orig.toolchains.configs.iter() {
            if !self.toolchains.configs.contains_key(name) {
                self.toolchains
                    .configs
                    .insert(name.clone(), orig_cache.clone());
            }
        }

        self.jobs.validate_storage_levels(
            if self.jobs.multilevel.chain.as_deref() == orig.jobs.multilevel.chain.as_deref() {
                "jobs.multilevel.chain"
            } else {
                "SCCACHE_DIST_JOBS_MULTILEVEL_CHAIN"
            },
        )?;

        self.toolchains.validate_storage_levels(
            if self.toolchains.multilevel.chain.as_deref()
                == orig.toolchains.multilevel.chain.as_deref()
            {
                "toolchains.multilevel.chain"
            } else {
                "SCCACHE_DIST_TOOLCHAINS_MULTILEVEL_CHAIN"
            },
        )?;

        Ok(self)
    }

    fn validate_file(self) -> Result<Self> {
        self.jobs.validate_storage_levels("jobs.multilevel.chain")?;

        self.toolchains
            .validate_storage_levels("toolchains.multilevel.chain")?;

        Ok(self)
    }

    fn validate_vars(self) -> Result<Self> {
        self.jobs
            .validate_storage_levels("SCCACHE_DIST_JOBS_MULTILEVEL_CHAIN")?;

        self.toolchains
            .validate_storage_levels("SCCACHE_DIST_TOOLCHAINS_MULTILEVEL_CHAIN")?;

        Ok(self)
    }
}

impl Loadable<Self> for Config {
    fn env_prefix<'a>() -> Option<&'a str> {
        Some("SCCACHE_DIST")
    }

    fn from_var_list(vars: &[(OsString, OsString)]) -> Result<Self> {
        let mut conf = Self::from_vars_impl(vars)?;

        // Remove Azure configs that don't have an auth mechanism
        conf.jobs.configs.retain(|_, cache| match cache {
            Cache::Azure(Azure { auth, .. }) => !matches!(auth, AzureAuth::None),
            _ => true,
        });

        conf.toolchains.configs.retain(|_, cache| match cache {
            Cache::Azure(Azure { auth, .. }) => !matches!(auth, AzureAuth::None),
            _ => true,
        });

        Ok(conf)
    }

    fn select_vars(vars: &[(OsString, OsString)]) -> Result<Vec<(String, String)>> {
        use crate::util::os_str_to_str;

        let prefix_ = Self::env_prefix().map_or_else(
            || "SCCACHE_DIST_".into(),
            |prefix| {
                if prefix.ends_with("_") {
                    prefix.into()
                } else {
                    format!("{prefix}_")
                }
            },
        );

        let builders_ = ["DOCKER_", "OVERLAY_", "POT_"];
        let metrics_ = ["DOGSTATSD_", "PROMETHEUS_"];

        let vars = vars
            .iter()
            .map(|(key, val)| -> Result<Vec<(String, String)>> {
                let key = os_str_to_str(key.as_os_str())?;
                let val = os_str_to_str(val.as_os_str())?;

                // Only take vars that start with `SCCACHE_DIST_`
                let key = if let Some(key) = key.strip_prefix(&prefix_) {
                    key
                } else if matches!(key.as_ref(), "AMQP_ADDR" | "REDIS_ADDR") {
                    return Ok(vec![("MESSAGE_BROKER_ADDR".into(), val.into_owned())]);
                } else {
                    return Ok(vec![]);
                };

                let key = match key {
                    // SCCACHE_DIR -> SCCACHE_CACHE_DISK_DIR
                    "DIR" => "DISK_DIR",
                    // SCCACHE_CACHE_SIZE -> SCCACHE_CACHE_DISK_SIZE
                    "CACHE_SIZE" => "DISK_SIZE",
                    // SCCACHE_BUCKET -> SCCACHE_CACHE_S3_BUCKET
                    "BUCKET" => "S3_BUCKET",
                    // SCCACHE_REGION -> SCCACHE_CACHE_S3_REGION
                    "REGION" => "S3_REGION",
                    // SCCACHE_REDIS -> SCCACHE_CACHE_REDIS_URL
                    "REDIS" => "REDIS_URL",
                    // SCCACHE_MEMCACHED -> SCCACHE_CACHE_MEMCACHED_URL
                    "MEMCACHED" => "MEMCACHED_URL",
                    _ => key,
                };

                let res = if let Some(key) = key.strip_prefix("SERVER_") {
                    // SCCACHE_DIST_SERVER_ID -> SCCACHE_DIST_ID
                    // SCCACHE_DIST_SERVER_CACHE_DIR -> SCCACHE_DIST_CACHE_DIR
                    // SCCACHE_DIST_SERVER_HEARTBEAT_INTERVAL -> SCCACHE_DIST_HEARTBEAT_INTERVAL_SECS
                    if matches!(key, "HEARTBEAT_INTERVAL") {
                        let key = format!("{key}_SECS");
                        vec![(key, val.into_owned())]
                    } else {
                        let key = key.to_owned();
                        vec![(key, val.into_owned())]
                    }
                } else if matches!(key, "SHUTDOWN_TIMEOUT") {
                    let key = format!("{key}_SECS");
                    vec![(key, val.into_owned())]
                } else if let Some(key) = builders_.iter().find_map(|kind_| key.strip_prefix(kind_))
                {
                    // SCCACHE_DIST_OVERLAY_BUILD_DIR -> SCCACHE_DIST_BUILDER_BUILD_DIR
                    // SCCACHE_DIST_OVERLAY_EXEC_CMD -> SCCACHE_DIST_BUILDER_EXEC_CMD
                    // SCCACHE_DIST_POT_CLONE_DIR -> SCCACHE_DIST_BUILDER_CLONE_DIR
                    // etc.
                    let key = format!("BUILDER_{key}");
                    vec![(key, val.into_owned())]
                } else if matches!(key, "METRICS_TYPE") {
                    // SCCACHE_DIST_METRICS_TYPE -> SCCACHE_DIST_METRICS
                    let key = "METRICS".to_owned();
                    vec![(key, val.into_owned())]
                } else if let Some((kind, mut key)) = metrics_.iter().find_map(|kind_| {
                    if let Some(key) = key.strip_prefix(kind_) {
                        kind_.strip_suffix("_").map(|kind| (kind, key))
                    } else {
                        None
                    }
                }) {
                    if kind == "PROMETHEUS" {
                        // SCCACHE_DIST_PROMETHEUS_PUSH_ENDPOINT -> SCCACHE_DIST_METRICS_PROMETHEUS_ENDPOINT
                        // SCCACHE_DIST_PROMETHEUS_PUSH_INTERVAL -> SCCACHE_DIST_METRICS_PROMETHEUS_INTERVAL
                        // SCCACHE_DIST_PROMETHEUS_PUSH_USERNAME -> SCCACHE_DIST_METRICS_PROMETHEUS_USERNAME
                        // SCCACHE_DIST_PROMETHEUS_PUSH_PASSWORD -> SCCACHE_DIST_METRICS_PROMETHEUS_PASSWORD
                        // SCCACHE_DIST_PROMETHEUS_PUSH_HTTP_METHOD -> SCCACHE_DIST_METRICS_HTTP_METHOD
                        if let Some(k) = key.strip_prefix("PUSH_") {
                            key = k;
                        }
                        // SCCACHE_DIST_PROMETHEUS_BIND_ADDR -> SCCACHE_DIST_METRICS_PROMETHEUS_ADDR
                        else if let Some(k) = key.strip_prefix("BIND_") {
                            key = k;
                        }
                        // SCCACHE_DIST_PROMETHEUS_LISTEN_ADDR -> SCCACHE_DIST_METRICS_PROMETHEUS_ADDR
                        else if let Some(k) = key.strip_prefix("LISTEN_") {
                            key = k;
                        }
                    }

                    // SCCACHE_DIST_DOGSTATSD_ADDR -> SCCACHE_DIST_METRICS_DOGSTATSD_ADDR
                    // SCCACHE_DIST_PROMETHEUS_TYPE -> SCCACHE_DIST_METRICS_PROMETHEUS_TYPE
                    // SCCACHE_DIST_PROMETHEUS_ADDR -> SCCACHE_DIST_METRICS_PROMETHEUS_ADDR
                    // SCCACHE_DIST_PROMETHEUS_IDLE_TIMEOUT_SECS -> SCCACHE_DIST_METRICS_PROMETHEUS_IDLE_TIMEOUT_SECS
                    // etc.
                    let key = format!("METRICS_{kind}_{key}");
                    vec![(key, val.into_owned())]
                } else {
                    let key = key.to_owned();
                    vec![(key, val.into_owned())]
                };

                Ok(res)
            })
            .flatten_ok()
            .try_collect::<_, Vec<_>, _>()?;

        if !vars.is_empty() {
            // trace! because envvars might include secrets.
            // SCCACHE_LOG=debug is the lowest recommended log level in CI.
            trace!(
                "Building server config from env:\n{}",
                vars.iter()
                    .map(|(k, v)| format!("{prefix_}{k} = \"{v}\""))
                    .join("\n")
            );
        }

        Ok(vars)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Percent(u32);

impl Default for Percent {
    fn default() -> Self {
        Self(100)
    }
}

impl From<Percent> for f64 {
    fn from(Percent(load): Percent) -> Self {
        (load as f64 / 100.0).max(0.0)
    }
}

impl From<f64> for Percent {
    fn from(load: f64) -> Self {
        Self((load.max(0.0) * 100.0).floor() as u32)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Builder {
    Docker(DockerBuilder),
    Overlay(OverlayBuilder),
    Pot(PotBuilder),
}

impl Default for Builder {
    fn default() -> Self {
        #[cfg(target_os = "freebsd")]
        {
            Builder::Pot(PotBuilder::default())
        }
        #[cfg(not(target_os = "freebsd"))]
        {
            Builder::Docker(DockerBuilder::default())
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct DockerBuilder {
    // Name of the image to run
    #[serde(default = "defaults::default_docker_image")]
    pub image: String,
    // Command to pass to `docker run <image>`
    #[serde(
        default = "defaults::default_docker_run_cmd",
        deserialize_with = "deserialize_command_or_list"
    )]
    pub run_cmd: Vec<String>,
    // Command to pass to `docker exec <container>`
    #[serde(
        default = "defaults::default_docker_exec_cmd",
        deserialize_with = "deserialize_command_or_list"
    )]
    pub exec_cmd: Vec<String>,
}

// Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
impl Default for DockerBuilder {
    fn default() -> Self {
        Self {
            image: defaults::default_docker_image(),
            run_cmd: defaults::default_docker_run_cmd(),
            exec_cmd: defaults::default_docker_exec_cmd(),
        }
    }
}

#[serde_as]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct OverlayBuilder {
    #[serde(default = "defaults::default_disk_cache_dir")]
    pub build_dir: PathBuf,
    #[serde(default = "defaults::default_bwrap_path")]
    pub bwrap_path: PathBuf,
    #[serde(default, deserialize_with = "deserialize_command_or_list")]
    pub exec_cmd: Vec<String>,
    #[serde_as(as = "PickFirst<(Vec<_>, StringWithSeparator<ColonSeparator, PathBuf>)>")]
    #[serde(default)]
    pub lower_dirs: Vec<PathBuf>,
    #[serde(default, deserialize_with = "deserialize_string_or_seq_to_map")]
    pub env: HashMap<String, String>,
}

// Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
impl Default for OverlayBuilder {
    fn default() -> Self {
        Self {
            build_dir: defaults::default_disk_cache_dir(),
            bwrap_path: defaults::default_bwrap_path(),
            exec_cmd: vec![],
            lower_dirs: vec![],
            env: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct PotBuilder {
    #[serde(default = "defaults::default_pot_fs_root")]
    pub pot_fs_root: PathBuf,
    #[serde(default = "defaults::default_pot_clone_from")]
    pub clone_from: String,
    #[serde(default = "defaults::default_pot_cmd")]
    pub pot_cmd: PathBuf,
    #[serde(
        default = "defaults::default_pot_clone_args",
        deserialize_with = "deserialize_command_or_list"
    )]
    pub pot_clone_args: Vec<String>,
}

// Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
impl Default for PotBuilder {
    fn default() -> Self {
        Self {
            pot_fs_root: defaults::default_pot_fs_root(),
            clone_from: defaults::default_pot_clone_from(),
            pot_cmd: defaults::default_pot_cmd(),
            pot_clone_args: defaults::default_pot_clone_args(),
        }
    }
}

pub mod defaults {

    use super::*;

    pub use crate::config::{
        client::defaults::default_disk_cache_size,
        defaults::default_disk_cache_dir,
        dist::defaults::{default_heartbeat_interval, default_shutdown_timeout},
    };

    const DEFAULT_POT_CLONE_FROM: &str = "sccache-template";
    const DEFAULT_POT_FS_ROOT: &str = "/opt/pot";
    const DEFAULT_POT_CMD: &str = "pot";
    const DEFAULT_POT_CLONE_ARGS: &[&str] = &["-i", "lo0|127.0.0.2"];

    pub fn default_bwrap_path() -> PathBuf {
        "/usr/bin/bwrap".into()
    }

    pub fn default_pot_clone_from() -> String {
        DEFAULT_POT_CLONE_FROM.to_string()
    }

    pub fn default_pot_fs_root() -> PathBuf {
        DEFAULT_POT_FS_ROOT.into()
    }

    pub fn default_pot_cmd() -> PathBuf {
        DEFAULT_POT_CMD.into()
    }

    pub fn default_pot_clone_args() -> Vec<String> {
        DEFAULT_POT_CLONE_ARGS
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    pub fn default_docker_image() -> String {
        "ubuntu:latest".into()
    }

    pub fn default_docker_run_cmd() -> Vec<String> {
        ["sh", "-c", "while true; do sleep 365d && true; done"]
            .into_iter()
            .map(Into::into)
            .collect()
    }

    pub fn default_docker_exec_cmd() -> Vec<String> {
        vec![]
    }

    pub fn default_server_id() -> String {
        uuid::Uuid::new_v4().as_simple().to_string()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::config::{cache::*, dist::*};

    #[test]
    fn defaults_match_serde_defaults() -> Result<()> {
        assert_eq!(Config::default(), serde_json::from_str("{}").unwrap());
        assert_eq!(
            DockerBuilder::default(),
            serde_json::from_str("{}").unwrap()
        );
        assert_eq!(
            OverlayBuilder::default(),
            serde_json::from_str("{}").unwrap()
        );
        assert_eq!(PotBuilder::default(), serde_json::from_str("{}").unwrap());

        assert_eq!(
            Config::from_toml_impl("[jobs]\n[toolchains]\n")?,
            Config::default()
        );
        assert_eq!(
            Config::from_toml_impl("[builder]\ntype = \"docker\"\n")?.builder,
            Builder::Docker(DockerBuilder::default())
        );
        assert_eq!(
            Config::from_toml_impl("[builder]\ntype = \"overlay\"\n")?.builder,
            Builder::Overlay(OverlayBuilder::default())
        );
        assert_eq!(
            Config::from_toml_impl("[builder]\ntype = \"pot\"\n")?.builder,
            Builder::Pot(PotBuilder::default())
        );

        Ok(())
    }

    #[test]
    fn from_envs_and_path_merges_file_and_env() -> Result<()> {
        drop(env_logger::try_init());

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("server.toml");
        std::fs::write(
            &path,
            r#"
            id = "file-id"
            cache_dir = "/file/toolchains"

            [builder]
            type = "overlay"
            build_dir = "/file/build"
            "#,
        )?;

        let config = Config::from_envs_and_path(
            [
                ("SCCACHE_DIST_SERVER_ID", "env-id"),
                ("SCCACHE_DIST_BUILDER_TYPE", "overlay"),
                ("SCCACHE_DIST_OVERLAY_BUILD_DIR", "/env/build"),
            ],
            path,
        )?;

        assert_eq!(config.id, "env-id");
        assert_eq!(config.cache_dir, PathBuf::from("/file/toolchains"));
        assert_eq!(
            config.builder,
            Builder::Overlay(OverlayBuilder {
                build_dir: "/env/build".into(),
                ..Default::default()
            })
        );

        Ok(())
    }

    #[test]
    fn load_reads_given_path() -> Result<()> {
        drop(env_logger::try_init());

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("server.toml");
        std::fs::write(&path, "id = \"file-id\"\ntoolchain_cache_size = 1234\n")?;

        // `load` reads the process env too, so compare against a load from the same env
        let expected = Config::from_envs_and_path(std::env::vars_os(), path.clone())?;
        assert_eq!(Config::load::<&PathBuf, _>(Some(&path))?, expected);
        assert_eq!(Config::load(path.clone())?, expected);

        std::fs::write(&path, "")?;
        assert!(!Config::load(path)?.id.is_empty());

        Ok(())
    }

    #[test]
    fn config_from_vars_amqp_addr() -> Result<()> {
        drop(env_logger::try_init());

        assert_eq!(
            Config::from_vars([
                ("AMQP_ADDR", "amqp://127.0.0.1:5672//"),
                ("SCCACHE_DIST_MESSAGE_BROKER_MAX_RETRIES", "2")
            ])?
            .message_broker,
            Some(MessageBroker {
                addr: "amqp://127.0.0.1:5672//".into(),
                max_retries: 2,
                ..Default::default()
            })
        );

        Ok(())
    }

    #[test]
    fn config_from_vars_redis_addr() -> Result<()> {
        drop(env_logger::try_init());

        assert_eq!(
            Config::from_vars([
                ("REDIS_ADDR", "redis://127.0.0.1:6379"),
                ("SCCACHE_DIST_MESSAGE_BROKER_MAX_RETRIES", "2")
            ])?
            .message_broker,
            Some(MessageBroker {
                addr: "redis://127.0.0.1:6379".into(),
                max_retries: 2,
                ..Default::default()
            })
        );

        Ok(())
    }

    #[test]
    fn config_from_vars() -> Result<()> {
        drop(env_logger::try_init());

        assert_eq!(
            Config::from_vars([
                (
                    "SCCACHE_DIST_MESSAGE_BROKER_ADDR",
                    "amqp://127.0.0.1:5672//"
                ),
                ("SCCACHE_DIST_SERVER_ID", "server-1"),
                ("SCCACHE_DIST_BUILDER_TYPE", "overlay"),
                ("SCCACHE_DIST_BUILDER_BUILD_DIR", "/tmp/build"),
                ("SCCACHE_DIST_BUILDER_BWRAP_PATH", "/usr/bin/bwrap"),
                ("SCCACHE_DIST_BUILDER_EXEC_CMD", "/usr/bin/wine"),
                ("SCCACHE_DIST_BUILDER_LOWER_DIRS", "/foo:/bar"),
                (
                    "SCCACHE_DIST_BUILDER_ENV",
                    "WINEPREFIX=/prefix WINEARCH=win64 WINEDEBUG=-all XDG_RUNTIME_DIR=/tmp/run/user/0"
                ),
                ("SCCACHE_DIST_CACHE_DIR", "/tmp/toolchains"),
                ("SCCACHE_DIST_MAX_PER_CORE_LOAD", "1.25"),
                ("SCCACHE_DIST_MAX_PER_CORE_PREFETCH", "1.0"),
                ("SCCACHE_DIST_METRICS", "prometheus"),
                ("SCCACHE_DIST_METRICS_PROMETHEUS_TYPE", "push"),
                (
                    "SCCACHE_DIST_METRICS_PROMETHEUS_ENDPOINT",
                    "http://127.0.0.1:9091/metrics/job/server-1"
                ),
                ("SCCACHE_DIST_METRICS_PROMETHEUS_IDLE_TIMEOUT_SECS", "10"),
                ("SCCACHE_DIST_METRICS_PROMETHEUS_INTERVAL_MS", "1000"),
                ("SCCACHE_DIST_METRICS_PROMETHEUS_USERNAME", "sccache"),
                ("SCCACHE_DIST_METRICS_PROMETHEUS_PASSWORD", "sccache"),
                ("SCCACHE_DIST_METRICS_PROMETHEUS_HTTP_METHOD", "post"),
                ("SCCACHE_DIST_JOBS_REDIS_URL", "redis://127.0.0.1:6379"),
                ("SCCACHE_DIST_JOBS_REDIS_TTL", "3600"),
                ("SCCACHE_DIST_JOBS_REDIS_KEY_PREFIX", "/sccache-dist-jobs"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_BUCKET", "sccache"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_REGION", "auto"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_ENDPOINT", "192.168.1.69:9000"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_USE_SSL", "false"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_NO_CREDENTIALS", "false"),
                (
                    "SCCACHE_DIST_TOOLCHAINS_S3_KEY_PREFIX",
                    "sccache-dist-toolchains"
                ),
            ])?,
            Config {
                id: "server-1".into(),
                builder: Builder::Overlay(OverlayBuilder {
                    build_dir: PathBuf::from("/tmp/build"),
                    bwrap_path: PathBuf::from("/usr/bin/bwrap"),
                    exec_cmd: vec!["/usr/bin/wine".into()],
                    lower_dirs: vec!["/foo".into(), "/bar".into()],
                    env: HashMap::from_iter([
                        ("WINEPREFIX".into(), "/prefix".into()),
                        ("WINEARCH".into(), "win64".into()),
                        ("WINEDEBUG".into(), "-all".into()),
                        ("XDG_RUNTIME_DIR".into(), "/tmp/run/user/0".into()),
                    ]),
                }),
                cache_dir: PathBuf::from("/tmp/toolchains"),
                max_per_core_load: 1.25.into(),
                max_per_core_prefetch: 1.0.into(),
                message_broker: Some("amqp://127.0.0.1:5672//".into()),
                metrics: Metrics::Prometheus(Prometheus::PushGateway {
                    endpoint: "http://127.0.0.1:9091/metrics/job/server-1".into(),
                    interval_ms: 1000,
                    username: Some("sccache".into()),
                    password: Some("sccache".into()),
                    http_method: Some("post".into()),
                    idle_timeout_secs: Some(10)
                }),
                jobs: vec![
                    Redis {
                        ttl: 3600,
                        key_prefix: "/sccache-dist-jobs".into(),
                        ..Redis::from_url("redis://127.0.0.1:6379")
                    }
                    .into()
                ]
                .into(),
                toolchains: vec![
                    S3 {
                        region: Some("auto".into()),
                        endpoint: Some("192.168.1.69:9000".into()),
                        use_ssl: Some(false),
                        no_credentials: false,
                        key_prefix: "sccache-dist-toolchains".into(),
                        ..S3::from_bucket("sccache")
                    }
                    .into()
                ]
                .into(),
                toolchain_cache_size: 10737418240,
                ..Default::default()
            }
        );

        Ok(())
    }

    #[test]
    fn config_from_vars_aliases() -> Result<()> {
        drop(env_logger::try_init());

        assert_eq!(
            Config::from_vars([
                (
                    "SCCACHE_DIST_MESSAGE_BROKER_ADDR",
                    "amqp://127.0.0.1:5672//"
                ),
                ("SCCACHE_DIST_SERVER_ID", "server-1"),
                ("SCCACHE_DIST_BUILDER_TYPE", "overlay"),
                ("SCCACHE_DIST_OVERLAY_BUILD_DIR", "/tmp/build"),
                ("SCCACHE_DIST_OVERLAY_BWRAP_PATH", "/usr/bin/bwrap"),
                ("SCCACHE_DIST_OVERLAY_EXEC_CMD", "/usr/bin/wine"),
                ("SCCACHE_DIST_OVERLAY_LOWER_DIRS", "/foo:/bar"),
                (
                    "SCCACHE_DIST_OVERLAY_ENV",
                    "WINEPREFIX=/prefix WINEARCH=win64 WINEDEBUG=-all XDG_RUNTIME_DIR=/tmp/run/user/0"
                ),
                ("SCCACHE_DIST_CACHE_DIR", "/tmp/toolchains"),
                ("SCCACHE_DIST_MAX_PER_CORE_LOAD", "1.25"),
                ("SCCACHE_DIST_MAX_PER_CORE_PREFETCH", "1.0"),
                ("SCCACHE_DIST_METRICS_TYPE", "prometheus"),
                ("SCCACHE_DIST_PROMETHEUS_TYPE", "push"),
                (
                    "SCCACHE_DIST_PROMETHEUS_PUSH_ENDPOINT",
                    "http://127.0.0.1:9091/metrics/job/server-1"
                ),
                ("SCCACHE_DIST_PROMETHEUS_PUSH_IDLE_TIMEOUT_SECS", "10"),
                ("SCCACHE_DIST_PROMETHEUS_PUSH_INTERVAL_MS", "1000"),
                ("SCCACHE_DIST_PROMETHEUS_PUSH_USERNAME", "sccache"),
                ("SCCACHE_DIST_PROMETHEUS_PUSH_PASSWORD", "sccache"),
                ("SCCACHE_DIST_PROMETHEUS_PUSH_HTTP_METHOD", "post"),
                //
                ("SCCACHE_DIST_JOBS_REDIS_URL", "redis://127.0.0.1:6379"),
                ("SCCACHE_DIST_JOBS_REDIS_TTL", "3600"),
                ("SCCACHE_DIST_JOBS_REDIS_KEY_PREFIX", "/sccache-dist-jobs"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_BUCKET", "sccache"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_REGION", "auto"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_ENDPOINT", "192.168.1.69:9000"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_USE_SSL", "false"),
                ("SCCACHE_DIST_TOOLCHAINS_S3_NO_CREDENTIALS", "false"),
                (
                    "SCCACHE_DIST_TOOLCHAINS_S3_KEY_PREFIX",
                    "sccache-dist-toolchains"
                ),
            ])?,
            Config {
                id: "server-1".into(),
                builder: Builder::Overlay(OverlayBuilder {
                    build_dir: PathBuf::from("/tmp/build"),
                    bwrap_path: PathBuf::from("/usr/bin/bwrap"),
                    exec_cmd: vec!["/usr/bin/wine".into()],
                    lower_dirs: vec!["/foo".into(), "/bar".into()],
                    env: HashMap::from_iter([
                        ("WINEPREFIX".into(), "/prefix".into()),
                        ("WINEARCH".into(), "win64".into()),
                        ("WINEDEBUG".into(), "-all".into()),
                        ("XDG_RUNTIME_DIR".into(), "/tmp/run/user/0".into()),
                    ]),
                }),
                cache_dir: PathBuf::from("/tmp/toolchains"),
                max_per_core_load: 1.25.into(),
                max_per_core_prefetch: 1.0.into(),
                message_broker: Some("amqp://127.0.0.1:5672//".into()),
                metrics: Metrics::Prometheus(Prometheus::PushGateway {
                    endpoint: "http://127.0.0.1:9091/metrics/job/server-1".into(),
                    interval_ms: 1000,
                    username: Some("sccache".into()),
                    password: Some("sccache".into()),
                    http_method: Some("post".into()),
                    idle_timeout_secs: Some(10)
                }),
                jobs: vec![
                    Redis {
                        ttl: 3600,
                        key_prefix: "/sccache-dist-jobs".into(),
                        ..Redis::from_url("redis://127.0.0.1:6379")
                    }
                    .into()
                ]
                .into(),
                toolchains: vec![
                    S3 {
                        region: Some("auto".into()),
                        endpoint: Some("192.168.1.69:9000".into()),
                        use_ssl: Some(false),
                        no_credentials: false,
                        key_prefix: "sccache-dist-toolchains".into(),
                        ..S3::from_bucket("sccache")
                    }
                    .into()
                ]
                .into(),
                toolchain_cache_size: 10737418240,
                ..Default::default()
            }
        );

        Ok(())
    }

    #[test]
    fn config_from_toml() -> Result<()> {
        drop(env_logger::try_init());

        let config = Config::from_toml(
            r#"
            id = "server-1"

            # This is where client toolchains will be stored.
            cache_dir = "/tmp/toolchains"

            # Dedicate (nproc * 1.25) CPUs to building
            max_per_core_load = 1.25

            # Prefetch (nproc * 1) jobs
            max_per_core_prefetch = 1

            # The maximum size of the toolchain cache, in bytes.
            # If unspecified the default is 10GB.
            toolchain_cache_size = 10737418240

            [builder]
            type = "overlay"
            # The directory under which a sandboxed filesystem will be created for builds.
            build_dir = "/tmp/build"
            # The path to the bubblewrap version 0.3.0+ `bwrap` binary.
            bwrap_path = "/usr/bin/bwrap"
            exec_cmd = "/usr/bin/wine"
            lower_dirs = [
                "/foo",
                "/bar"
            ]

            [builder.env]
            WINEPREFIX = "/prefix"
            WINEARCH = "win64"
            WINEDEBUG = "-all"
            XDG_RUNTIME_DIR = "/tmp/run/user/0"

            [message_broker]
            addr = "amqp://127.0.0.1:5672//"

            [metrics.prometheus]
            type = "push"
            endpoint = "http://127.0.0.1:9091/metrics/job/server-1"
            interval_ms = 1000
            username = "sccache"
            password = "sccache"
            http_method = "post"
            idle_timeout_secs = 10

            [jobs.redis]
            url = "redis://127.0.0.1:6379"
            expiration = 3600
            key_prefix = "/sccache-dist-jobs"

            [toolchains.s3]
            bucket = "sccache"
            region = "auto"
            endpoint = "192.168.1.69:9000"
            use_ssl = false
            no_credentials = false
            key_prefix = "sccache-dist-toolchains"
            "#,
        )?;

        assert_eq!(
            config,
            Config {
                id: "server-1".into(),
                builder: Builder::Overlay(OverlayBuilder {
                    build_dir: PathBuf::from("/tmp/build"),
                    bwrap_path: PathBuf::from("/usr/bin/bwrap"),
                    exec_cmd: vec!["/usr/bin/wine".into()],
                    lower_dirs: vec!["/foo".into(), "/bar".into()],
                    env: HashMap::from_iter([
                        ("WINEPREFIX".into(), "/prefix".into()),
                        ("WINEARCH".into(), "win64".into()),
                        ("WINEDEBUG".into(), "-all".into()),
                        ("XDG_RUNTIME_DIR".into(), "/tmp/run/user/0".into()),
                    ]),
                }),
                cache_dir: PathBuf::from("/tmp/toolchains"),
                max_per_core_load: 1.25.into(),
                max_per_core_prefetch: 1.0.into(),
                message_broker: Some("amqp://127.0.0.1:5672//".into()),
                metrics: Metrics::Prometheus(Prometheus::PushGateway {
                    endpoint: "http://127.0.0.1:9091/metrics/job/server-1".into(),
                    interval_ms: 1000,
                    username: Some("sccache".into()),
                    password: Some("sccache".into()),
                    http_method: Some("post".into()),
                    idle_timeout_secs: Some(10)
                }),
                jobs: vec![
                    Redis {
                        ttl: 3600,
                        key_prefix: "/sccache-dist-jobs".into(),
                        ..Redis::from_url("redis://127.0.0.1:6379")
                    }
                    .into()
                ]
                .into(),
                toolchains: vec![
                    S3 {
                        region: Some("auto".into()),
                        endpoint: Some("192.168.1.69:9000".into()),
                        use_ssl: Some(false),
                        no_credentials: false,
                        key_prefix: "sccache-dist-toolchains".into(),
                        ..S3::from_bucket("sccache")
                    }
                    .into()
                ]
                .into(),
                toolchain_cache_size: 10737418240,
                ..Default::default()
            }
        );

        Ok(())
    }
}
