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
    config::utils::{_Ignored, DeserializeBool, deserialize_bool, deserialize_size_from_str},
    errors::*,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize, de};
use serde_with::{PickFirst, StringWithSeparator, formats::CommaSeparator, serde_as};
use std::{collections::BTreeMap, fmt, path::PathBuf, str::FromStr};

#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct Caches {
    pub configs: BTreeMap<String, Cache>,
    pub multilevel: MultiLevel,
    /// Timeout for cache reads (default: 60s)
    pub read_timeout_secs: u64,
    /// Timeout for cache writes (default: None)
    pub write_timeout_secs: Option<u64>,
    pub skip_check: bool,
    /// Temporarily disable caching if errors occur too frequently within a time window
    pub rate_limit: RateLimits,
}

impl Caches {
    pub fn validate_storage_levels<S: AsRef<str>>(&self, err_key: S) -> Result<()> {
        if let Some(chain) = self.multilevel.chain.as_deref() {
            for level in chain {
                if !self.configs.contains_key(level) {
                    bail!(
                        "'{level}' cache not configured, but specified in {}: {chain:?}",
                        err_key.as_ref()
                    )
                }
            }
        }
        Ok(())
    }
}

impl From<Vec<(&'static str, Cache)>> for Caches {
    fn from(configs: Vec<(&'static str, Cache)>) -> Self {
        let configs = configs
            .into_iter()
            .map(|(name, cache)| (name.to_owned(), cache))
            .collect();
        Self {
            multilevel: MultiLevel {
                chain: Some(MultiLevel::chain_from_configs(&configs)),
                ..Default::default()
            },
            configs,
            ..Default::default()
        }
    }
}

// Same as deserializing an empty map (see `defaults_match_serde_defaults`). Not written
// as `serde_json::from_str("{}")` because that compiles one more deserializer for every
// config type reachable from here.
impl Default for Caches {
    fn default() -> Self {
        CachesFields::default().finish()
    }
}

// serde_env also yields the leading segments of multi-word keys (`key` for `key_prefix`) as
// keys of their own. The hand-written visitors skip keys that start with one of those.
fn is_env_subkey(name: &str, fields: &[&str]) -> bool {
    fields
        .iter()
        .any(|field| name.starts_with(field.split_once('_').map_or(*field, |(head, _)| head)))
}

impl<'de> Deserialize<'de> for Caches {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(CachesVisitor)
    }
}

struct CachesVisitor;

impl<'de> de::Visitor<'de> for CachesVisitor {
    type Value = Caches;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("map of named cache configurations")
    }

    fn visit_map<M>(self, mut map: M) -> std::result::Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        let mut f = CachesFields::default();

        while let Ok(Some(name)) = map.next_key::<String>() {
            match name.as_str() {
                // Expected fields
                "configs" => {
                    for (name, config) in map.next_value::<BTreeMap<String, Cache>>()? {
                        f.configs.insert(name, config);
                    }
                }
                "multilevel" => {
                    f.multilevel = map.next_value::<Option<MultiLevel>>()?.map(trim_chain);
                }
                "read_timeout_secs" => {
                    f.read_timeout = Some(map.next_value()?);
                }
                // Infallible alias for read_timeout_secs
                "read_timeout" => {
                    f.read_timeout = map.next_value().ok().or(f.read_timeout);
                }
                "write_timeout_secs" => {
                    f.write_timeout = map.next_value()?;
                }
                // Infallible alias for write_timeout_secs
                "write_timeout" => {
                    f.write_timeout = map.next_value().ok().or(f.write_timeout);
                }
                "skip_check" => {
                    f.skip_check = Some(map.next_value::<DeserializeBool>()?.into());
                }
                "rate_limit" => {
                    f.rate_limit = map.next_value()?;
                }
                // Legacy: Before supporting multiple configs for the same
                // storage backend, the type was denoted by field name, i.e.
                // `cache.s3` was the configuration for S3 storage.
                //
                // ```toml
                // # Can only be one S3 cache
                // [cache.s3]
                // bucket = "my-bucket"
                // region = "us-west-2"
                // ```
                //
                // Since now we can have multiple configurations for the same
                // storage type, the type must be declared as part of the
                // config so serde knows which CacheType enum variant to
                // deserialize:
                //
                // ```toml
                // # Can be multiple S3 caches, e.g. for geographic locality
                // [cache.primary.s3]
                // bucket = "my-bucket-us-west-2"
                // region = "us-west-2"
                //
                // [cache.secondary.s3]
                // bucket = "my-bucket-us-east-2"
                // region = "us-east-2"
                //
                // [cache.multilevel]
                // chain = ["primary", "secondary"]
                // ```
                //
                // But to support users with config files in the old format,
                // assume any caches whose key is the CacheType enum tag are
                // still that storage type.
                "disk" => {
                    f.configs.insert(name, map.next_value::<Disk>()?.into());
                }
                "s3" => {
                    f.configs.insert(name, map.next_value::<S3>()?.into());
                }
                "redis" => {
                    f.configs.insert(name, map.next_value::<Redis>()?.into());
                }
                "memcached" => {
                    f.configs
                        .insert(name, map.next_value::<Memcached>()?.into());
                }
                "gcs" => {
                    f.configs.insert(name, map.next_value::<GCS>()?.into());
                }
                "gha" => {
                    f.configs.insert(name, map.next_value::<GHA>()?.into());
                }
                "azure" => {
                    f.configs.insert(name, map.next_value::<Azure>()?.into());
                }
                "webdav" => {
                    f.configs.insert(name, map.next_value::<Webdav>()?.into());
                }
                "oss" => {
                    f.configs.insert(name, map.next_value::<OSS>()?.into());
                }
                "cos" => {
                    f.configs.insert(name, map.next_value::<COS>()?.into());
                }
                // New format: a named Cache enum
                _ => {
                    // Skip serde_env sub-selections for fields we've already deserialized
                    if is_caches_env_subkey(&name) {
                        let _ = map.next_value::<_Ignored>();
                    } else if let Ok(config) = map.next_value::<Cache>() {
                        f.configs.insert(name, config);
                    }
                }
            }
        }

        Ok(f.finish())
    }
}

// Collects the fields so that building the config happens in non-generic code, instead of
// being compiled again for every deserializer.
#[derive(Default)]
struct CachesFields {
    configs: BTreeMap<String, Cache>,
    multilevel: Option<MultiLevel>,
    read_timeout: Option<u64>,
    write_timeout: Option<u64>,
    skip_check: Option<bool>,
    rate_limit: Option<RateLimits>,
}

impl CachesFields {
    fn finish(self) -> Caches {
        let CachesFields {
            configs,
            multilevel,
            read_timeout,
            write_timeout,
            skip_check,
            rate_limit,
        } = self;

        let mut multilevel = multilevel.unwrap_or_default();
        multilevel.chain = multilevel
            .chain
            .or_else(|| Some(MultiLevel::chain_from_configs(&configs)));

        Caches {
            configs,
            multilevel,
            read_timeout_secs: read_timeout.unwrap_or_else(defaults::default_read_timeout),
            write_timeout_secs: write_timeout,
            skip_check: skip_check.unwrap_or_default(),
            rate_limit: rate_limit.unwrap_or_default(),
        }
    }
}

fn trim_chain(mut multilevel: MultiLevel) -> MultiLevel {
    if let Some(chain) = multilevel.chain.as_mut() {
        for level in chain.iter_mut() {
            *level = level.trim().into();
        }
    }
    multilevel
}

fn is_caches_env_subkey(name: &str) -> bool {
    is_env_subkey(
        name,
        &[
            "configs",
            "multilevel",
            "read_timeout_secs",
            "write_timeout_secs",
            "skip_check",
            "rate_limit",
        ],
    ) || [
        "disk_",
        "s3_",
        "redis_",
        "memcached_",
        "gcs_",
        "gha_",
        "azure_",
        "webdav_",
        "oss_",
        "cos_",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Cache {
    Azure(Azure),
    Disk(Disk),
    GCS(GCS),
    GHA(GHA),
    Memcached(Memcached),
    Redis(Redis),
    S3(S3),
    Webdav(Webdav),
    OSS(OSS),
    COS(COS),
}

impl From<&Cache> for &str {
    fn from(cache: &Cache) -> Self {
        match cache {
            Cache::Disk(_) => "disk",
            Cache::S3(_) => "s3",
            Cache::Redis(_) => "redis",
            Cache::Memcached(_) => "memcached",
            Cache::GCS(_) => "gcs",
            Cache::GHA(_) => "gha",
            Cache::Azure(_) => "azure",
            Cache::Webdav(_) => "webdav",
            Cache::OSS(_) => "oss",
            Cache::COS(_) => "cos",
        }
    }
}

impl From<Azure> for Cache {
    fn from(config: Azure) -> Self {
        Self::Azure(config)
    }
}

impl From<Disk> for Cache {
    fn from(config: Disk) -> Self {
        Self::Disk(config)
    }
}

impl From<GCS> for Cache {
    fn from(config: GCS) -> Self {
        Self::GCS(config)
    }
}

impl From<GHA> for Cache {
    fn from(config: GHA) -> Self {
        Self::GHA(config)
    }
}

impl From<Memcached> for Cache {
    fn from(config: Memcached) -> Self {
        Self::Memcached(config)
    }
}

impl From<Redis> for Cache {
    fn from(config: Redis) -> Self {
        Self::Redis(config)
    }
}

impl From<S3> for Cache {
    fn from(config: S3) -> Self {
        Self::S3(config)
    }
}

impl From<Webdav> for Cache {
    fn from(config: Webdav) -> Self {
        Self::Webdav(config)
    }
}

impl From<OSS> for Cache {
    fn from(config: OSS) -> Self {
        Self::OSS(config)
    }
}

impl From<COS> for Cache {
    fn from(config: COS) -> Self {
        Self::COS(config)
    }
}

impl From<Azure> for (&'static str, Cache) {
    fn from(config: Azure) -> Self {
        ("azure", Cache::Azure(config))
    }
}

impl From<Disk> for (&'static str, Cache) {
    fn from(config: Disk) -> Self {
        ("disk", Cache::Disk(config))
    }
}

impl From<GCS> for (&'static str, Cache) {
    fn from(config: GCS) -> Self {
        ("gcs", Cache::GCS(config))
    }
}

impl From<GHA> for (&'static str, Cache) {
    fn from(config: GHA) -> Self {
        ("gha", Cache::GHA(config))
    }
}

impl From<Memcached> for (&'static str, Cache) {
    fn from(config: Memcached) -> Self {
        ("memcached", Cache::Memcached(config))
    }
}

impl From<Redis> for (&'static str, Cache) {
    fn from(config: Redis) -> Self {
        ("redis", Cache::Redis(config))
    }
}

impl From<S3> for (&'static str, Cache) {
    fn from(config: S3) -> Self {
        ("s3", Cache::S3(config))
    }
}

impl From<Webdav> for (&'static str, Cache) {
    fn from(config: Webdav) -> Self {
        ("webdav", Cache::Webdav(config))
    }
}

impl From<OSS> for (&'static str, Cache) {
    fn from(config: OSS) -> Self {
        ("oss", Cache::OSS(config))
    }
}

impl From<COS> for (&'static str, Cache) {
    fn from(config: COS) -> Self {
        ("cos", Cache::COS(config))
    }
}

impl Cache {
    pub fn enabled(&self) -> bool {
        match self {
            Self::Azure(cfg) => cfg.enabled,
            Self::Disk(cfg) => cfg.enabled,
            Self::GCS(cfg) => cfg.enabled,
            Self::GHA(cfg) => cfg.enabled,
            Self::Memcached(cfg) => cfg.enabled,
            Self::Redis(cfg) => cfg.enabled,
            Self::S3(cfg) => cfg.enabled,
            Self::Webdav(cfg) => cfg.enabled,
            Self::OSS(cfg) => cfg.enabled,
            Self::COS(cfg) => cfg.enabled,
        }
    }
    pub fn get_order(&self) -> u64 {
        match self {
            Self::Azure(_) => DefaultCacheOrder::Azure as u64,
            Self::Disk(_) => DefaultCacheOrder::Disk as u64,
            Self::GCS(_) => DefaultCacheOrder::GCS as u64,
            Self::GHA(_) => DefaultCacheOrder::GHA as u64,
            Self::Memcached(_) => DefaultCacheOrder::Memcached as u64,
            Self::Redis(_) => DefaultCacheOrder::Redis as u64,
            Self::S3(_) => DefaultCacheOrder::S3 as u64,
            Self::Webdav(_) => DefaultCacheOrder::Webdav as u64,
            Self::OSS(_) => DefaultCacheOrder::OSS as u64,
            Self::COS(_) => DefaultCacheOrder::COS as u64,
        }
    }
}

impl PartialOrd for Cache {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cache {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get_order().cmp(&other.get_order())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Azure {
    pub enabled: bool,
    pub container: String,
    pub auth: AzureAuth,
    pub key_prefix: String,
    pub rw_mode: CacheMode,
}

impl Azure {
    pub fn from_container<S: Into<String>>(container: S) -> Self {
        Self {
            enabled: true,
            container: container.into(),
            auth: Default::default(),
            key_prefix: Default::default(),
            rw_mode: Default::default(),
        }
    }
}

impl<'de> Deserialize<'de> for Azure {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(AzureVisitor)
    }
}

struct AzureVisitor;

impl<'de> de::Visitor<'de> for AzureVisitor {
    type Value = Azure;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Azure cache configuration")
    }

    fn visit_map<M>(self, mut map: M) -> std::result::Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        let mut f = AzureFields::default();

        while let Ok(Some(name)) = map.next_key::<String>() {
            match name.as_str() {
                "auth" => {
                    f.auth = Some(map.next_value()?);
                }
                "enabled" => {
                    f.enabled = map.next_value::<DeserializeBool>()?.into();
                }
                "container" => {
                    f.container = Some(map.next_value::<String>()?);
                }
                "key_prefix" => {
                    f.key_prefix = Some(map.next_value::<String>()?);
                }
                "rw_mode" => {
                    f.rw_mode = Some(map.next_value()?);
                }
                "connection_string" => {
                    f.shared_key = Some(AzureAuth::SharedKey {
                        connection_string: map.next_value()?,
                    });
                }
                "endpoint" => {
                    f.endpoint = Some(AzureAuth::Endpoint {
                        endpoint: map.next_value()?,
                    });
                }
                "storage_account" => {
                    f.storage_account = Some(AzureAuth::StorageAccount {
                        storage_account: map.next_value()?,
                    });
                }
                name => {
                    if is_env_subkey(
                        name,
                        &[
                            "preprocessor",
                            "enabled",
                            "container",
                            "key_prefix",
                            "rw_mode",
                            "auth",
                            "connection_string",
                            "storage_account",
                            "endpoint",
                        ],
                    ) {
                        let _ = map.next_value::<_Ignored>();
                    } else {
                        return Err(de::Error::unknown_field(
                            name,
                            &[
                                "container",
                                "key_prefix",
                                "rw_mode",
                                "auth",
                                "connection_string",
                                "storage_account",
                                "endpoint",
                            ],
                        ));
                    }
                }
            }
        }

        Ok(f.finish())
    }
}

#[derive(Default)]
struct AzureFields {
    auth: Option<AzureAuth>,
    enabled: Option<bool>,
    endpoint: Option<AzureAuth>,
    shared_key: Option<AzureAuth>,
    storage_account: Option<AzureAuth>,
    container: Option<String>,
    key_prefix: Option<String>,
    rw_mode: Option<CacheMode>,
}

impl AzureFields {
    fn finish(self) -> Azure {
        let AzureFields {
            auth,
            enabled,
            endpoint,
            shared_key,
            storage_account,
            container,
            key_prefix,
            rw_mode,
        } = self;

        let mut enabled = enabled.unwrap_or(true);

        let container = if let Some(container) = container {
            container
        } else {
            warn!("Azure config missing required field `container`.");
            enabled = false;
            String::new()
        };

        let has_entra_source = storage_account.is_some() || endpoint.is_some();

        let auth = if let Some(auth) = auth {
            auth
        } else if let Some(shared_key) = shared_key {
            if has_entra_source {
                warn!(
                    "Set either `connection_string` (shared key) or `storage_account` / `endpoint` (Entra ID), not both.",
                );
                enabled = false;
                AzureAuth::None
            } else {
                shared_key
            }
        } else if let Some(endpoint) = endpoint {
            endpoint
        } else if let Some(storage_account) = storage_account {
            storage_account
        } else {
            // Backwards compatible with the historical behavior where a
            // container without any auth source left the Azure backend disabled
            // (previously the destructure required both the connection string
            // and the container). Warn rather than fail so a stray container
            // variable cannot take down an unrelated cache backend, while still
            // surfacing the misconfiguration. This is deliberately more lenient
            // than a file config: an explicit `[cache.azure]` block with no auth
            // source is a real misconfiguration and errors at operator build
            // (AzureBlobCache::build -> resolve_blob_endpoint), because a config
            // file — unlike an ambient env var — is an unambiguous opt-in.
            warn!(
                "`container` is set but no Azure auth source was provided (`connection_string`, `storage_account` or `endpoint`)."
            );
            AzureAuth::None
        };

        if !enabled {
            warn!("Azure storage disabled, see above messages.");
        }

        Azure {
            enabled,
            container,
            auth,
            key_prefix: key_prefix.unwrap_or_default(),
            rw_mode: rw_mode.unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AzureAuth {
    #[default]
    None,
    /// Shared-key connection string. When set, the shared-key path is used.
    /// When absent, the Microsoft Entra ID (passwordless) path is used and
    /// `storage_account`/`endpoint` supply the blob endpoint.
    #[serde(rename = "shared_key")]
    SharedKey { connection_string: String },
    /// Full blob endpoint override for the Entra ID path (sovereign clouds,
    /// custom DNS). Takes precedence over `storage_account`.
    #[serde(rename = "endpoint")]
    Endpoint { endpoint: String },
    /// Storage account name for the Entra ID path. The blob endpoint is
    /// synthesized as `https://{storage_account}.blob.core.windows.net`.
    #[serde(rename = "storage_account")]
    StorageAccount { storage_account: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Disk {
    #[serde(
        default = "defaults::default_true",
        deserialize_with = "deserialize_bool"
    )]
    pub enabled: bool,
    #[serde(default = "defaults::default_disk_cache_dir")]
    pub dir: PathBuf,
    #[serde(default)]
    pub key_prefix: String,
    #[serde(default = "defaults::default_disk_cache_size")]
    #[serde(deserialize_with = "deserialize_size_from_str")]
    pub size: u64,
    #[serde(default)]
    pub rw_mode: CacheMode,
}

// Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
impl Default for Disk {
    fn default() -> Self {
        Self {
            enabled: defaults::default_true(),
            dir: defaults::default_disk_cache_dir(),
            key_prefix: String::new(),
            size: defaults::default_disk_cache_size(),
            rw_mode: CacheMode::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GCS {
    #[serde(
        default = "defaults::default_true",
        deserialize_with = "deserialize_bool"
    )]
    pub enabled: bool,
    #[serde(default)]
    pub bucket: String,
    #[serde(default)]
    pub key_prefix: String,
    #[serde(alias = "cred_path")]
    pub key_path: Option<String>,
    pub service_account: Option<String>,
    #[serde(alias = "credential_url")]
    pub credentials_url: Option<String>,
    #[serde(default)]
    pub rw_mode: CacheMode,
}

impl GCS {
    pub fn from_bucket<S: Into<String>>(bucket: S) -> Self {
        Self {
            enabled: true,
            bucket: bucket.into(),
            key_prefix: Default::default(),
            key_path: Default::default(),
            service_account: Default::default(),
            credentials_url: Default::default(),
            rw_mode: Default::default(),
        }
    }
}

impl<'de> Deserialize<'de> for GCS {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(GCSVisitor)
    }
}

struct GCSVisitor;

impl<'de> de::Visitor<'de> for GCSVisitor {
    type Value = GCS;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GCS cache configuration")
    }

    fn visit_map<M>(self, mut map: M) -> std::result::Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        let mut f = GCSFields::default();

        while let Ok(Some(name)) = map.next_key::<String>() {
            match name.as_str() {
                "enabled" => {
                    f.enabled = map.next_value::<DeserializeBool>()?.into();
                }
                "bucket" => {
                    f.bucket = Some(map.next_value()?);
                }
                "key_prefix" => {
                    f.key_prefix = Some(map.next_value()?);
                }
                "cred_path" | "key_path" => {
                    f.key_path = map.next_value()?;
                }
                "oauth_url" => {
                    f.oauth_url = Some(map.next_value::<String>()?);
                }
                "service_account" => {
                    f.service_account = map.next_value()?;
                }
                "credentials_url" | "credential_url" => {
                    f.credentials_url = map.next_value()?;
                }
                "rw_mode" => {
                    f.rw_mode = Some(map.next_value()?);
                }
                name => {
                    if is_env_subkey(
                        name,
                        &[
                            "preprocessor",
                            "enabled",
                            "bucket",
                            "key_prefix",
                            "cred_path",
                            "key_path",
                            "oauth_url",
                            "service_account",
                            "credential_url",
                            "credentials_url",
                            "rw_mode",
                        ],
                    ) {
                        let _ = map.next_value::<_Ignored>();
                    } else {
                        return Err(de::Error::unknown_field(
                            name,
                            &[
                                "bucket",
                                "key_prefix",
                                "cred_path",
                                "service_account",
                                "credentials_url",
                                "rw_mode",
                            ],
                        ));
                    }
                }
            }
        }

        Ok(f.finish())
    }
}

#[derive(Default)]
struct GCSFields {
    enabled: Option<bool>,
    bucket: Option<String>,
    key_prefix: Option<String>,
    key_path: Option<String>,
    oauth_url: Option<String>,
    service_account: Option<String>,
    credentials_url: Option<String>,
    rw_mode: Option<CacheMode>,
}

impl GCSFields {
    fn finish(self) -> GCS {
        let GCSFields {
            enabled,
            bucket,
            key_prefix,
            key_path,
            oauth_url,
            service_account,
            credentials_url,
            rw_mode,
        } = self;

        let mut enabled = enabled.unwrap_or(true);

        if oauth_url.is_some() {
            warn!(
                "`oauth_url` has been deprecated. If you intend to use vm metadata for auth, please set `service_account` instead."
            );
        }

        if bucket.is_none() && (credentials_url.is_some() || key_path.is_some()) {
            warn!("If setting GCS credentials, `bucket` and an auth mechanism need to be set.");
            enabled = false;
        }

        if !enabled {
            warn!("GCS storage disabled, see above messages.");
        }

        GCS {
            enabled,
            bucket: bucket.unwrap_or_default(),
            key_prefix: key_prefix.unwrap_or_default(),
            key_path,
            service_account,
            credentials_url,
            rw_mode: rw_mode.unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GHA {
    #[serde(
        default, // default to false
        deserialize_with = "deserialize_bool"
    )]
    pub enabled: bool,
    #[serde(default)]
    pub key_prefix: String,
    /// Version for gha cache is a namespace. By setting different versions,
    /// we can avoid mixed caches.
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub rw_mode: CacheMode,
}

impl GHA {
    pub fn from_version<S: Into<String>>(version: S) -> Self {
        Self {
            enabled: true,
            key_prefix: Default::default(),
            version: version.into(),
            rw_mode: Default::default(),
        }
    }
}

/// Memcached's default value of expiration is 10800s (3 hours), which is too
/// short for use case of sccache.
///
/// We increase the default expiration to 86400s (1 day) to balance between
/// memory consumpation and cache hit rate.
///
/// Please change this value freely if we have a better choice.
const DEFAULT_MEMCACHED_CACHE_EXPIRATION: u32 = 86400;

fn default_memcached_cache_expiration() -> u32 {
    DEFAULT_MEMCACHED_CACHE_EXPIRATION
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Memcached {
    #[serde(
        default = "defaults::default_true",
        deserialize_with = "deserialize_bool"
    )]
    pub enabled: bool,
    #[serde(alias = "endpoint")]
    pub url: String,
    /// Username to authenticate with.
    pub username: Option<String>,
    /// Password to authenticate with.
    pub password: Option<String>,
    /// the expiration time in seconds.
    ///
    /// Default to 24 hours (86400)
    /// Up to 30 days (2592000)
    #[serde(default = "default_memcached_cache_expiration")]
    pub expiration: u32,
    #[serde(default)]
    pub key_prefix: String,
    /// The maximum number of connections allowed.
    ///
    /// Default to 10
    pub connection_pool_max_size: Option<usize>,
    #[serde(default)]
    pub rw_mode: CacheMode,
}

impl Memcached {
    pub fn from_url<S: Into<String>>(url: S) -> Self {
        Self {
            enabled: true,
            url: url.into(),
            username: Default::default(),
            password: Default::default(),
            expiration: default_memcached_cache_expiration(),
            key_prefix: Default::default(),
            connection_pool_max_size: Default::default(),
            rw_mode: Default::default(),
        }
    }
}

pub const DEFAULT_REDIS_DB: u32 = 0;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Redis {
    #[serde(
        default = "defaults::default_true",
        deserialize_with = "deserialize_bool"
    )]
    pub enabled: bool,
    /// The single-node redis endpoint.
    /// Mutually exclusive with `cluster_endpoints`.
    pub endpoint: Option<String>,
    /// A comma-delimited list of redis cluster endpoints.
    /// If reader_endpoints is not provided, these endpoints will be used to read and write.
    /// If reader_endpoints is provided, these endpoints will only be used to write.
    /// Mutually exclusive with `endpoint`.
    pub cluster_endpoints: Option<String>,
    /// A comma-delimited list of redis cluster read-only endpoints.
    pub reader_endpoints: Option<String>,
    /// Username to authenticate with.
    pub username: Option<String>,
    /// Password to authenticate with.
    pub password: Option<String>,
    /// The redis URL.
    /// Deprecated in favor of `endpoint`.
    pub url: Option<String>,
    /// the db number to use
    ///
    /// Default to 0
    #[serde(default)]
    pub db: u32,
    /// the ttl (expiration) time in seconds.
    ///
    /// Redis has no default TTL - all caches live forever.
    /// Default to infinity (0) as redis does
    #[serde(default, alias = "expiration")]
    pub ttl: u64,
    /// The maximum number of connections allowed.
    ///
    /// Default to 10
    pub connection_pool_max_size: Option<usize>,
    #[serde(default)]
    pub key_prefix: String,
    #[serde(default)]
    pub rw_mode: CacheMode,
}

impl Redis {
    pub fn from_url<S: Into<String>>(url: S) -> Self {
        Self {
            enabled: true,
            url: Some(url.into()),
            endpoint: Default::default(),
            cluster_endpoints: Default::default(),
            reader_endpoints: Default::default(),
            username: Default::default(),
            password: Default::default(),
            db: Default::default(),
            ttl: Default::default(),
            connection_pool_max_size: Default::default(),
            key_prefix: Default::default(),
            rw_mode: Default::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Webdav {
    #[serde(
        default = "defaults::default_true",
        deserialize_with = "deserialize_bool"
    )]
    pub enabled: bool,
    // Required
    pub endpoint: String,
    #[serde(default)]
    pub key_prefix: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub token: Option<String>,
    #[serde(default)]
    pub rw_mode: CacheMode,
}

impl Webdav {
    pub fn from_endpoint<S: Into<String>>(endpoint: S) -> Self {
        Self {
            enabled: true,
            endpoint: endpoint.into(),
            key_prefix: Default::default(),
            username: Default::default(),
            password: Default::default(),
            token: Default::default(),
            rw_mode: Default::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct S3 {
    pub enabled: bool,
    pub bucket: String, // <-- required
    pub region: Option<String>,
    pub key_prefix: String,
    pub no_credentials: bool,
    pub endpoint: Option<String>,
    pub use_ssl: Option<bool>,
    pub server_side_encryption: Option<bool>,
    pub server_side_encryption_aws_kms: Option<bool>,
    pub server_side_encryption_kms_key_id: Option<String>,
    pub enable_virtual_host_style: Option<bool>,
    pub rw_mode: CacheMode,
}

impl<'de> Deserialize<'de> for S3 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_map(S3Visitor)
    }
}

struct S3Visitor;

impl<'de> de::Visitor<'de> for S3Visitor {
    type Value = S3;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("S3 cache configuration")
    }

    fn visit_map<M>(self, mut map: M) -> std::result::Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        let mut f = S3Fields::default();

        while let Ok(Some(name)) = map.next_key::<String>() {
            match name.as_str() {
                "enabled" => {
                    f.enabled = map.next_value::<DeserializeBool>()?.into();
                }
                "bucket" => {
                    f.bucket = Some(map.next_value::<String>()?);
                }
                "region" => {
                    f.region = map.next_value::<Option<String>>()?;
                }
                "key_prefix" => {
                    f.key_prefix = Some(map.next_value::<String>()?);
                }
                "no_credentials" => {
                    f.no_credentials = map.next_value::<DeserializeBool>()?.into();
                }
                "enable_virtual_host_style" => {
                    f.enable_virtual_host_style = map.next_value::<DeserializeBool>()?.into();
                }
                "endpoint" => {
                    f.endpoint = map.next_value::<Option<String>>()?;
                }
                "use_ssl" => {
                    f.use_ssl = map.next_value::<DeserializeBool>()?.into();
                }
                "server_side_encryption" => {
                    f.server_side_encryption =
                        map.next_value::<DeserializeBool>().ok().map(Into::into);
                }
                "server_side_encryption_aws_kms" => {
                    f.server_side_encryption_aws_kms = map.next_value::<DeserializeBool>()?.into();
                }
                "server_side_encryption_kms_key_id" => {
                    f.server_side_encryption_kms_key_id = map.next_value::<Option<String>>()?;
                }
                "rw_mode" => {
                    f.rw_mode = Some(map.next_value()?);
                }
                name => {
                    if is_env_subkey(
                        name,
                        &[
                            "preprocessor",
                            "enabled",
                            "bucket",
                            "region",
                            "key_prefix",
                            "no_credentials",
                            "endpoint",
                            "use_ssl",
                            "server_side_encryption",
                            "server_side_encryption_aws_kms",
                            "server_side_encryption_kms_key_id",
                            "enable_virtual_host_style",
                            "rw_mode",
                        ],
                    ) {
                        let _ = map.next_value::<_Ignored>();
                    } else {
                        return Err(de::Error::unknown_field(
                            name,
                            &[
                                "enabled",
                                "bucket",
                                "region",
                                "key_prefix",
                                "no_credentials",
                                "endpoint",
                                "use_ssl",
                                "server_side_encryption",
                                "server_side_encryption_aws_kms",
                                "server_side_encryption_kms_key_id",
                                "enable_virtual_host_style",
                                "rw_mode",
                            ],
                        ));
                    }
                }
            }
        }

        f.finish().map_err(de::Error::custom)
    }
}

#[derive(Default)]
struct S3Fields {
    enabled: Option<bool>,
    bucket: Option<String>,
    region: Option<String>,
    key_prefix: Option<String>,
    no_credentials: Option<bool>,
    endpoint: Option<String>,
    use_ssl: Option<bool>,
    server_side_encryption: Option<bool>,
    server_side_encryption_aws_kms: Option<bool>,
    server_side_encryption_kms_key_id: Option<String>,
    enable_virtual_host_style: Option<bool>,
    rw_mode: Option<CacheMode>,
}

impl S3Fields {
    fn finish(self) -> std::result::Result<S3, &'static str> {
        let S3Fields {
            enabled,
            bucket,
            region,
            key_prefix,
            no_credentials,
            endpoint,
            use_ssl,
            server_side_encryption,
            server_side_encryption_aws_kms,
            server_side_encryption_kms_key_id,
            enable_virtual_host_style,
            rw_mode,
        } = self;

        let mut enabled = enabled.unwrap_or(true);

        let bucket = if let Some(bucket) = bucket {
            bucket
        } else {
            warn!("S3 config missing required field `bucket`.");
            enabled = false;
            String::new()
        };

        if enabled
            && no_credentials.unwrap_or_default()
            && (std::env::var_os("AWS_ACCESS_KEY_ID").is_some()
                || std::env::var_os("AWS_SECRET_ACCESS_KEY").is_some())
        {
            return Err("If setting S3 credentials, {{no_credentials}} must not be set.");
        }

        if !enabled {
            warn!("S3 storage disabled, see above messages.");
        }

        Ok(S3 {
            enabled,
            bucket,
            region,
            key_prefix: key_prefix.unwrap_or_default(),
            no_credentials: no_credentials.unwrap_or_default(),
            endpoint,
            use_ssl,
            server_side_encryption,
            server_side_encryption_aws_kms,
            server_side_encryption_kms_key_id,
            enable_virtual_host_style,
            rw_mode: rw_mode.unwrap_or_default(),
        })
    }
}

impl S3 {
    pub fn from_bucket<S: Into<String>>(bucket: S) -> Self {
        Self {
            enabled: true,
            bucket: bucket.into(),
            region: Default::default(),
            key_prefix: Default::default(),
            no_credentials: Default::default(),
            endpoint: Default::default(),
            use_ssl: Default::default(),
            server_side_encryption: Default::default(),
            server_side_encryption_aws_kms: Default::default(),
            server_side_encryption_kms_key_id: Default::default(),
            enable_virtual_host_style: Default::default(),
            rw_mode: Default::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OSS {
    #[serde(
        default = "defaults::default_true",
        deserialize_with = "deserialize_bool"
    )]
    pub enabled: bool,
    // Required
    pub bucket: String,
    #[serde(default)]
    pub key_prefix: String,
    pub endpoint: Option<String>,
    #[serde(default, deserialize_with = "deserialize_bool")]
    pub no_credentials: bool,
    #[serde(default)]
    pub rw_mode: CacheMode,
}

impl OSS {
    pub fn from_bucket<S: Into<String>>(bucket: S) -> Self {
        Self {
            enabled: true,
            bucket: bucket.into(),
            key_prefix: Default::default(),
            endpoint: Default::default(),
            no_credentials: Default::default(),
            rw_mode: Default::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct COS {
    #[serde(
        default = "defaults::default_true",
        deserialize_with = "deserialize_bool"
    )]
    pub enabled: bool,
    // Required
    pub bucket: String,
    #[serde(default)]
    pub key_prefix: String,
    pub endpoint: Option<String>,
    #[serde(default)]
    pub rw_mode: CacheMode,
}

impl COS {
    pub fn from_bucket<S: Into<String>>(bucket: S) -> Self {
        Self {
            enabled: true,
            bucket: bucket.into(),
            key_prefix: Default::default(),
            endpoint: Default::default(),
            rw_mode: Default::default(),
        }
    }
}

/// Configuration for multi-level cache.
#[serde_as]
#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
pub struct MultiLevel {
    /// Ordered list of cache backends (L0, L1, L2, ...)
    #[serde_as(
        as = "PickFirst<(Option<Vec<_>>, Option<StringWithSeparator<CommaSeparator, String>>)>"
    )]
    pub chain: Option<Vec<String>>,
    /// Write failure handling policy
    #[serde(default)]
    pub write_error_policy: WriteErrorPolicy,
}

impl MultiLevel {
    pub fn chain_from_configs(configs: &BTreeMap<String, Cache>) -> Vec<String> {
        configs
            .iter()
            .sorted_by(|(lhs_name, lhs_conf), (rhs_name, rhs_conf)| {
                lhs_conf.cmp(rhs_conf).then(lhs_name.cmp(rhs_name))
            })
            .map(|(name, _)| name.clone())
            .collect()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
pub struct RateLimits {
    pub on_error_count: Option<usize>,
    pub on_error_window_size_secs: Option<u64>,
}

/// CacheMode is used to represent which mode we are using.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub enum CacheMode {
    /// Only read cache from storage.
    #[serde(rename = "READ_ONLY")]
    ReadOnly,
    #[default]
    /// Full support of cache storage: read and write.
    #[serde(rename = "READ_WRITE")]
    ReadWrite,
}

impl fmt::Display for CacheMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadOnly => {
                write!(f, "READ_ONLY")
            }
            Self::ReadWrite => {
                write!(f, "READ_WRITE")
            }
        }
    }
}

impl<'de> Deserialize<'de> for CacheMode {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_any(CacheModeVisitor)
    }
}

struct CacheModeVisitor;

impl<'de> de::Visitor<'de> for CacheModeVisitor {
    type Value = CacheMode;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a cache rw_mode config")
    }

    fn visit_str<E>(self, rw_mode: &str) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        parse_cache_mode(rw_mode).map_err(E::custom)
    }
}

fn parse_cache_mode(rw_mode: &str) -> std::result::Result<CacheMode, String> {
    match rw_mode.to_uppercase().as_str() {
        "READ_ONLY" => Ok(CacheMode::ReadOnly),
        "READ_WRITE" => Ok(CacheMode::ReadWrite),
        _ => Err(format!("'{rw_mode}' must be 'READ_ONLY' or 'READ_WRITE'")),
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Eq, PartialEq, Ord, PartialOrd)]
enum DefaultCacheOrder {
    Disk = 0,
    S3 = 1,
    Redis = 2,
    Memcached = 3,
    GCS = 4,
    GHA = 5,
    Azure = 6,
    Webdav = 7,
    OSS = 8,
    COS = 9,
}

/// Defines how the multi-level cache handles write failures.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WriteErrorPolicy {
    /// Never fail on write errors - log warnings only (most permissive)
    Ignore,
    /// Fail only if L0 write fails (default - balances reliability and performance)
    #[default]
    L0,
    /// Fail if any read-write level fails (most strict)
    All,
}

impl FromStr for WriteErrorPolicy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "ignore" => Ok(WriteErrorPolicy::Ignore),
            "l0" => Ok(WriteErrorPolicy::L0),
            "all" => Ok(WriteErrorPolicy::All),
            _ => Err(anyhow!(
                "Invalid write policy '{s}'. Valid values: ignore, l0, all"
            )),
        }
    }
}

impl fmt::Display for WriteErrorPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteErrorPolicy::Ignore => write!(f, "ignore"),
            WriteErrorPolicy::L0 => write!(f, "l0"),
            WriteErrorPolicy::All => write!(f, "all"),
        }
    }
}

pub mod defaults {
    use std::time::Duration;

    pub use crate::config::defaults::{
        default_disk_cache_dir, default_disk_cache_size, default_true,
    };

    pub fn default_read_timeout() -> u64 {
        Duration::from_secs(60).as_secs()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::config::{ClientConfig, Loadable};

    #[test]
    fn test_default_cache_order() {
        let disk = Disk::default();
        let s3 = S3::from_bucket("s3");
        let redis = Redis::from_url("redis://redis:1234");
        let memcached = Memcached::from_url("memcached://memcached:1234");
        let gcs = GCS::from_bucket("gcs");
        let gha = GHA::from_version("1");
        let azure = Azure::from_container("azure");
        let webdav = Webdav::from_endpoint("http://webdav:1234");
        let oss = OSS::from_bucket("oss");
        let cos = COS::from_bucket("cos");

        let mut caches: Vec<Cache> = vec![
            oss.into(),
            redis.into(),
            gha.into(),
            cos.into(),
            memcached.into(),
            gcs.into(),
            s3.into(),
            webdav.into(),
            disk.into(),
            azure.into(),
        ];

        caches.sort();

        while let Some(cache) = caches.pop() {
            for other in caches.iter() {
                assert_eq!(cache.cmp(other), std::cmp::Ordering::Greater);
            }
        }
    }

    #[test]
    fn defaults_match_serde_defaults() {
        assert_eq!(Caches::default(), serde_json::from_str("{}").unwrap());
        assert_eq!(Disk::default(), serde_json::from_str("{}").unwrap());
    }

    #[test]
    fn unknown_cache_fields_rejected() {
        for (table, expected) in [
            (
                "[cache.s3]\nbucket = \"b\"",
                "unknown field `bogus`, expected one of `enabled`, `bucket`, `region`, `key_prefix`, \
                 `no_credentials`, `endpoint`, `use_ssl`, `server_side_encryption`, \
                 `server_side_encryption_aws_kms`, `server_side_encryption_kms_key_id`, \
                 `enable_virtual_host_style`, `rw_mode`",
            ),
            (
                "[cache.azure]\ncontainer = \"c\"",
                "unknown field `bogus`, expected one of `container`, `key_prefix`, `rw_mode`, \
                 `auth`, `connection_string`, `storage_account`, `endpoint`",
            ),
            (
                "[cache.gcs]\nbucket = \"b\"",
                "unknown field `bogus`, expected one of `bucket`, `key_prefix`, `cred_path`, \
                 `service_account`, `credentials_url`, `rw_mode`",
            ),
        ] {
            let err = ClientConfig::from_toml_impl(&format!("{table}\nbogus = 1\n"))
                .unwrap_err()
                .to_string();
            assert!(err.contains(expected), "{err}");
        }
    }

    #[test]
    fn env_subkeys_ignored() -> Result<()> {
        // serde_env also yields the leading segments of multi-word keys (`key` for
        // `key_prefix`), so keys starting with those are skipped in files too
        for (table, prefixes) in [
            (
                "[cache.s3]\nbucket = \"b\"",
                &[
                    "preprocessor",
                    "enabled",
                    "bucket",
                    "region",
                    "key",
                    "no",
                    "endpoint",
                    "use",
                    "server",
                    "enable",
                    "rw",
                ][..],
            ),
            (
                "[cache.azure]\ncontainer = \"c\"",
                &[
                    "preprocessor",
                    "enabled",
                    "container",
                    "key",
                    "rw",
                    "auth",
                    "connection",
                    "storage",
                    "endpoint",
                ][..],
            ),
            (
                "[cache.gcs]\nbucket = \"b\"",
                &[
                    "preprocessor",
                    "enabled",
                    "bucket",
                    "key",
                    "cred",
                    "oauth",
                    "service",
                    "credential",
                    "rw",
                ][..],
            ),
            (
                "[cache]",
                &[
                    "configs",
                    "multilevel",
                    "read",
                    "write",
                    "skip",
                    "rate",
                    "disk_",
                    "s3_",
                    "redis_",
                    "memcached_",
                    "gcs_",
                    "gha_",
                    "azure_",
                    "webdav_",
                    "oss_",
                    "cos_",
                ][..],
            ),
        ] {
            for prefix in prefixes {
                let toml = format!("{table}\n{prefix}x = 1\n");
                ClientConfig::from_toml_impl(&toml).with_context(|| toml.clone())?;
            }
        }
        Ok(())
    }

    #[test]
    fn invalid_named_cache_dropped() -> Result<()> {
        let config = ClientConfig::from_toml_impl(
            r#"
            [cache.custom]
            type = "redis"
            ttl = "not a number"

            [cache.unknown]
            type = "nope"
            "#,
        )?;
        assert!(config.cache.configs.is_empty());
        Ok(())
    }
}
