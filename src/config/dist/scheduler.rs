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
        dist::{Keepalive, MessageBroker, Metrics},
        utils::{deserialize_string_or_list, deserialize_string_or_seq_to_map},
    },
    errors::*,
};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_with::{OneOrMany, formats::PreferMany, serde_as};
use std::{collections::HashMap, ffi::OsString, net::SocketAddr, path::Path};

#[serde_as]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub jobs: Caches,

    #[serde(default)]
    pub toolchains: Caches,

    #[serde_as(as = "OneOrMany<_, PreferMany>")]
    #[serde(default = "defaults::default_auth", alias = "client_auth")]
    pub auth: Vec<Auth>,

    #[serde(default = "defaults::default_heartbeat_interval")]
    pub heartbeat_interval_ms: u64,

    #[serde(default = "defaults::default_job_time_limit")]
    pub job_time_limit_secs: u32,

    #[serde(default)]
    pub keepalive: Keepalive,

    #[serde(default = "defaults::default_max_body_size")]
    pub max_body_size: usize,

    #[serde(default)]
    pub max_concurrent_streams: Option<u32>,

    #[serde(default)]
    pub message_broker: Option<MessageBroker>,

    #[serde(default)]
    pub metrics: Metrics,

    #[serde(default = "defaults::default_public_addr")]
    pub public_addr: SocketAddr,

    #[serde(alias = "scheduler_id", default)]
    pub id: String,

    #[serde(default = "defaults::default_shutdown_timeout")]
    pub shutdown_timeout_secs: u64,
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
                conf.id = defaults::default_scheduler_id();
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
            auth: defaults::default_auth(),
            heartbeat_interval_ms: defaults::default_heartbeat_interval(),
            job_time_limit_secs: defaults::default_job_time_limit(),
            keepalive: Keepalive::default(),
            max_body_size: defaults::default_max_body_size(),
            max_concurrent_streams: None,
            message_broker: None,
            metrics: Metrics::default(),
            public_addr: defaults::default_public_addr(),
            id: String::new(),
            shutdown_timeout_secs: defaults::default_shutdown_timeout(),
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

                let res = if let Some(key) = key.strip_prefix("SCHEDULER_") {
                    // SCCACHE_DIST_SCHEDULER_ID -> SCCACHE_DIST_ID
                    // SCCACHE_DIST_SCHEDULER_HEARTBEAT_INTERVAL -> SCCACHE_DIST_HEARTBEAT_INTERVAL_SECS
                    if matches!(key, "HEARTBEAT_INTERVAL") {
                        let key = format!("{key}_SECS");
                        vec![(key, val.into_owned())]
                    } else {
                        let key = key.to_owned();
                        vec![(key, val.into_owned())]
                    }
                } else if matches!(key, "JOB_TIME_LIMIT" | "SHUTDOWN_TIMEOUT") {
                    let key = format!("{key}_SECS");
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
                "Building scheduler config from env:\n{}",
                vars.iter()
                    .map(|(k, v)| format!("{prefix_}{k} = \"{v}\""))
                    .join("\n")
            );
        }

        Ok(vars)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Auth {
    #[default]
    #[serde(rename = "DANGEROUSLY_INSECURE")]
    Insecure,
    Token {
        token: String,
    },
    #[serde(alias = "jwt_validate")]
    Jwt(JWTDecode),
    #[serde(rename = "proxy_token")]
    ProxyToken {
        url: String,
        cache_secs: Option<u64>,
        decode: Option<ProxyTokenDecode>,
        rate_limit_on_error_count: Option<usize>,
        rate_limit_on_error_window_size_secs: Option<u64>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct JWTDecode {
    #[serde(default, deserialize_with = "deserialize_string_or_list")]
    pub audience: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_string_or_list")]
    pub issuer: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_string_or_list")]
    pub jwks_url: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_string_or_seq_to_map")]
    pub claims: HashMap<String, String>,
    #[serde(default = "defaults::default_jwt_decode_leeway")]
    pub leeway: u64,
}

// Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
impl Default for JWTDecode {
    fn default() -> Self {
        Self {
            audience: vec![],
            issuer: vec![],
            jwks_url: vec![],
            claims: HashMap::new(),
            leeway: defaults::default_jwt_decode_leeway(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ProxyTokenDecode {
    #[default]
    None,
    Jwt(JWTDecode),
}

pub mod defaults {
    use super::*;

    use std::str::FromStr;

    pub use crate::config::dist::defaults::{default_heartbeat_interval, default_shutdown_timeout};

    pub fn default_auth() -> Vec<Auth> {
        vec![Auth::Insecure]
    }

    pub fn default_job_time_limit() -> u32 {
        // 10 minutes
        600
    }

    pub fn default_max_body_size() -> usize {
        // 1GiB should be enough for toolchains and compile inputs, right?
        1024 * 1024 * 1024
    }

    pub fn default_public_addr() -> SocketAddr {
        SocketAddr::from_str("0.0.0.0:10500").unwrap()
    }

    pub fn default_scheduler_id() -> String {
        uuid::Uuid::new_v4().as_simple().to_string()
    }

    pub fn default_jwt_decode_leeway() -> u64 {
        60
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::config::{cache::*, dist::*};

    use std::{path::PathBuf, str::FromStr};

    #[test]
    fn defaults_match_serde_defaults() -> Result<()> {
        assert_eq!(Config::default(), serde_json::from_str("{}").unwrap());
        assert_eq!(JWTDecode::default(), serde_json::from_str("{}").unwrap());

        assert_eq!(
            Config::from_toml_impl("[jobs]\n[toolchains]\n[keepalive]\n")?,
            Config::default()
        );

        let config = Config::from_toml_impl(
            r#"
            [message_broker]

            [[client_auth]]
            type = "jwt"
            "#,
        )?;
        assert_eq!(config.message_broker, Some(MessageBroker::default()));
        assert_eq!(config.auth, vec![Auth::Jwt(JWTDecode::default())]);

        Ok(())
    }

    #[test]
    fn from_envs_and_path_merges_file_and_env() -> Result<()> {
        drop(env_logger::try_init());

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("scheduler.toml");
        std::fs::write(
            &path,
            r#"
            id = "file-id"
            job_time_limit_secs = 1200

            [message_broker]
            addr = "amqp://file:5672//"
            "#,
        )?;

        let config = Config::from_envs_and_path(
            [
                ("SCCACHE_DIST_SCHEDULER_ID", "env-id"),
                ("AMQP_ADDR", "amqp://env:5672//"),
            ],
            path,
        )?;

        assert_eq!(config.id, "env-id");
        assert_eq!(config.job_time_limit_secs, 1200);
        assert_eq!(config.message_broker, Some("amqp://env:5672//".into()));

        Ok(())
    }

    #[test]
    fn load_reads_given_path() -> Result<()> {
        drop(env_logger::try_init());

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("scheduler.toml");
        std::fs::write(&path, "id = \"file-id\"\njob_time_limit_secs = 1234\n")?;

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
                ("SCCACHE_DIST_SCHEDULER_ID", "scheduler-1"),
                ("SCCACHE_DIST_PUBLIC_ADDR", "127.0.0.1:10500"),
                ("SCCACHE_DIST_JOB_TIME_LIMIT_SECS", "1200"),
                ("SCCACHE_DIST_METRICS", "prometheus"),
                ("SCCACHE_DIST_METRICS_PROMETHEUS_TYPE", "push"),
                (
                    "SCCACHE_DIST_METRICS_PROMETHEUS_ENDPOINT",
                    "http://127.0.0.1:9091/metrics/job/scheduler-1"
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
                id: "scheduler-1".into(),
                public_addr: SocketAddr::from_str("127.0.0.1:10500").unwrap(),
                job_time_limit_secs: 1200,
                message_broker: Some("amqp://127.0.0.1:5672//".into()),
                metrics: Metrics::Prometheus(Prometheus::PushGateway {
                    endpoint: "http://127.0.0.1:9091/metrics/job/scheduler-1".into(),
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
                ..Default::default()
            }
        );

        Ok(())
    }

    #[test]
    fn config_from_toml() -> Result<()> {
        drop(env_logger::try_init());

        use std::net::SocketAddr;

        let config = Config::from_toml(
            r#"
            # The socket address the scheduler will listen on. It's strongly recommended
            # to listen on localhost and put a HTTPS server in front of it.
            public_addr = "127.0.0.1:10500"

            id = "scheduler-1"

            # Don't allow jobs to run for longer than 20 minutes
            job_time_limit_secs = 1200

            # The address of the AMQP broker
            [message_broker]
            addr = "amqp://127.0.0.1:5672//"

            [[client_auth]]
            type = "token"
            token = "secrettoken"

            [[client_auth]]
            type = "jwt"
            audience = "token.mycompany.com"
            issuer = "https://token.mycompany.com"
            jwks_url = "https://token.mycompany.com/.well-known/jwks"

            [client_auth.claims]
            enterprise_id = "12345"
            actor = "*"
            repository = "*"
            workflow = "*"

            [[client_auth]]
            type = "proxy_token"
            url = "https://token.mycompany.com/12345?audience=sts.amazonaws.com"
            cache_secs = 300

            [client_auth.decode]
            type = "jwt"
            audience = "sts.amazonaws.com"
            issuer = "https://token.mycompany.com"
            jwks_url = "https://token.mycompany.com/.well-known/jwks"

            [client_auth.decode.claims]
            actor = "*"
            username = "*"

            [metrics.prometheus]
            type = "push"
            endpoint = "http://127.0.0.1:9091/metrics/job/scheduler-1"
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
                id: "scheduler-1".into(),
                auth: vec![
                    Auth::Token {
                        token: "secrettoken".into(),
                    },
                    Auth::Jwt(JWTDecode {
                        audience: vec!["token.mycompany.com".into()],
                        issuer: vec!["https://token.mycompany.com".into()],
                        jwks_url: vec!["https://token.mycompany.com/.well-known/jwks".into()],
                        claims: [
                            ("enterprise_id", "12345"),
                            ("actor", "*"),
                            ("repository", "*"),
                            ("workflow", "*"),
                        ]
                        .into_iter()
                        .map(|(k, v)| (k.into(), v.into()))
                        .collect(),
                        ..Default::default()
                    }),
                    Auth::ProxyToken {
                        url: "https://token.mycompany.com/12345?audience=sts.amazonaws.com".into(),
                        cache_secs: Some(300),
                        decode: Some(ProxyTokenDecode::Jwt(JWTDecode {
                            audience: vec!["sts.amazonaws.com".into()],
                            issuer: vec!["https://token.mycompany.com".into()],
                            jwks_url: vec!["https://token.mycompany.com/.well-known/jwks".into()],
                            claims: [("actor", "*"), ("username", "*")]
                                .into_iter()
                                .map(|(k, v)| (k.into(), v.into()))
                                .collect(),
                            ..Default::default()
                        })),
                        rate_limit_on_error_count: None,
                        rate_limit_on_error_window_size_secs: None,
                    },
                ],
                public_addr: SocketAddr::from_str("127.0.0.1:10500").unwrap(),
                job_time_limit_secs: 1200,
                message_broker: Some("amqp://127.0.0.1:5672//".into()),
                metrics: Metrics::Prometheus(Prometheus::PushGateway {
                    endpoint: "http://127.0.0.1:9091/metrics/job/scheduler-1".into(),
                    interval_ms: 1000,
                    username: Some("sccache".into()),
                    password: Some("sccache".into()),
                    http_method: Some("post".into()),
                    idle_timeout_secs: Some(10),
                }),
                jobs: vec![
                    Redis {
                        ttl: 3600,
                        key_prefix: "/sccache-dist-jobs".into(),
                        ..Redis::from_url("redis://127.0.0.1:6379")
                    }
                    .into(),
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
                    .into(),
                ]
                .into(),
                ..Default::default()
            }
        );

        Ok(())
    }
}
