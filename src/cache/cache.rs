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
// limitations under the License.

#[cfg(any(
    feature = "azure",
    feature = "gcs",
    feature = "gha",
    feature = "memcached",
    feature = "redis",
    feature = "s3",
    feature = "webdav",
    feature = "oss",
    feature = "cos",
    feature = "watcher"
))]
use crate::cache;
#[cfg(any(
    feature = "azure",
    feature = "gcs",
    feature = "gha",
    feature = "memcached",
    feature = "redis",
    feature = "s3",
    feature = "webdav",
    feature = "oss",
    feature = "cos"
))]
use crate::cache::utils::normalize_key;
use crate::{
    cache::{
        disk::DiskCache,
        multilevel::{MultiLevelStats, MultiLevelStorage},
        rate_limit::RateLimitStorage,
        readonly::ReadOnlyStorage,
    },
    config::{self, CacheMode, CacheType, Caches, WriteErrorPolicy, cache::RateLimits},
    errors::*,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{fmt, path::PathBuf, sync::Arc, time::Duration};

use super::cache_io::*;

/// Result of [`Storage::get_path`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GetPathResult {
    /// Cache hit: the entry lives at this filesystem path.
    Found(PathBuf),
    /// Cache miss: the key is not in the cache.
    Miss,
    /// This backend does not support direct file access; use `get` instead.
    Unsupported,
}

/// An interface to cache storage.
#[async_trait]
pub trait Storage: Send + Sync {
    /// Get a cache entry by `key`.
    ///
    /// If an error occurs, this method should return a `Result::Err`.
    /// If nothing fails but the entry is not found in the cache,
    /// it should return a `Cache::Miss`.
    /// If the entry is successfully found in the cache, it should
    /// return a `Cache::Hit`.
    async fn get(&self, key: &str) -> Result<Cache<opendal::Buffer>>;

    /// Delete the cache entry for `key`.
    async fn del(&self, key: &str) -> Result<()>;

    /// Check if the cache has an entry for `key`.
    ///
    /// If the entry is successfully found in the cache, return true.
    /// If an error occurs, or the entry is not found in the cache, return false.
    async fn has(&self, key: &str) -> bool;

    /// Put `entry` in the cache under `key`.
    ///
    /// Returns a `Future` that will provide the result or error when the put is
    /// finished.
    async fn put(&self, key: &str, entry: opendal::Buffer) -> Result<Duration>;

    async fn size(&self, key: &str) -> Result<u64>;

    /// Check the cache capability.
    ///
    /// - `Ok(CacheMode::ReadOnly)` means cache can only be used to `get`
    ///   cache.
    /// - `Ok(CacheMode::ReadWrite)` means cache can do both `get` and `put`.
    /// - `Err(err)` means cache is not setup correctly or not match with
    ///   users input (for example, user try to use `ReadWrite` but cache
    ///   is `ReadOnly`).
    ///
    /// We will provide a default implementation which returns
    /// `Ok(CacheMode::ReadWrite)` for service that doesn't
    /// support check yet.
    async fn check(&self) -> Result<CacheMode> {
        Ok(CacheMode::ReadWrite)
    }

    /// Get the storage location.
    async fn location(&self) -> String;

    /// Get the cache backend type name (e.g., "disk", "redis", "s3").
    /// Used for statistics and display purposes.
    fn cache_type_name(&self) -> &'static str;

    /// Get the current storage usage, if applicable.
    async fn current_size(&self) -> Result<Option<u64>>;

    /// Get the maximum storage size, if applicable.
    async fn max_size(&self) -> Result<Option<u64>>;

    /// Get multi-level cache statistics, if this is a multi-level storage.
    fn multilevel_stats(&self) -> Option<MultiLevelStats> {
        None
    }

    /// Return whether the storage is enabled.
    /// Currently only NoStorage impl returns false.
    fn enabled(&self) -> bool;

    /// Return the base directories for path normalization if configured
    fn basedirs(&self) -> &[Vec<u8>];

    /// Return the filesystem path of the cached entry for `key`.
    /// Default impl returns [`GetPathResult::Unsupported`].
    async fn get_path(&self, _key: &str) -> GetPathResult {
        GetPathResult::Unsupported
    }
}

#[cfg(any(
    feature = "azure",
    feature = "gcs",
    feature = "gha",
    feature = "memcached",
    feature = "redis",
    feature = "s3",
    feature = "webdav",
    feature = "oss",
    feature = "cos"
))]
mod operator {
    use super::*;
    use crate::util::retry_with_jitter;
    use tokio_retry2::RetryError;

    fn to_retry_err<E: Into<opendal::Error>>(kind: &str, err: E) -> RetryError<opendal::Error> {
        let err = err.into();
        match err.kind() {
            opendal::ErrorKind::NotFound => {
                trace!("cache {kind} error (permanent): {err}");
                RetryError::permanent(err)
            }
            opendal::ErrorKind::PermissionDenied => {
                trace!("cache {kind} error (permanent): {err}");
                RetryError::permanent(err)
            }
            opendal::ErrorKind::RateLimited => {
                debug!("cache {kind} error (transient): {err}");
                RetryError::transient(err)
            }
            opendal::ErrorKind::Unsupported => {
                debug!("cache {kind} error (permanent): {err}");
                RetryError::permanent(err)
            }
            opendal::ErrorKind::Unexpected if err.is_temporary() => {
                warn!("cache {kind} error (transient): {err}");
                RetryError::transient(err)
            }
            _ => {
                warn!("cache {kind} error (permanent): {err:?}");
                RetryError::permanent(err)
            }
        }
    }

    pub async fn read_with_retry(
        storage: &opendal::Operator,
        key: &str,
    ) -> std::result::Result<opendal::Buffer, opendal::Error> {
        // TODO: Allow configuring the number of retries
        retry_with_jitter(usize::MAX, || async {
            storage
                .read(&normalize_key(key))
                .await
                .map_err(|e| to_retry_err("lookup", e))
        })
        .await
    }

    pub async fn write_with_retry(
        storage: &opendal::Operator,
        key: &str,
        buf: opendal::Buffer,
    ) -> std::result::Result<opendal::Metadata, opendal::Error> {
        // TODO: Allow configuring the number of retries
        retry_with_jitter(usize::MAX, || async {
            storage
                .write(&normalize_key(key), buf.clone())
                .await
                .map_err(|e| to_retry_err("write", e))
        })
        .await
    }
}

/// Wrapper for opendal::Operator that adds basedirs support
#[cfg(any(
    feature = "azure",
    feature = "gcs",
    feature = "gha",
    feature = "memcached",
    feature = "redis",
    feature = "s3",
    feature = "webdav",
    feature = "oss",
    feature = "cos"
))]
pub struct RemoteStorage {
    operator: opendal::Operator,
    basedirs: Vec<Vec<u8>>,
}

#[cfg(any(
    feature = "azure",
    feature = "gcs",
    feature = "gha",
    feature = "memcached",
    feature = "redis",
    feature = "s3",
    feature = "webdav",
    feature = "oss",
    feature = "cos"
))]
impl RemoteStorage {
    pub fn new(operator: opendal::Operator, basedirs: Vec<Vec<u8>>) -> Self {
        Self { operator, basedirs }
    }
}

/// Implement storage for operator.
#[cfg(any(
    feature = "azure",
    feature = "gcs",
    feature = "gha",
    feature = "memcached",
    feature = "redis",
    feature = "s3",
    feature = "webdav",
    feature = "oss",
    feature = "cos"
))]
#[async_trait]
impl Storage for RemoteStorage {
    async fn get(&self, key: &str) -> Result<Cache<opendal::Buffer>> {
        match operator::read_with_retry(&self.operator, key).await {
            Ok(data) => {
                trace!("opendal::Operator::get({key}): Found {} bytes", data.len());
                Ok(Cache::Hit(data))
            }
            Err(e) if e.kind() == opendal::ErrorKind::NotFound => {
                trace!("opendal::Operator::get({key}): NotFound");
                Ok(Cache::Miss)
            }
            Err(e) => {
                warn!("opendal::Operator::get({key}): Error: {e:?}");
                // Return error instead of silently returning None
                Err(anyhow!("Failed to read raw bytes: {e:?}"))
            }
        }
    }

    async fn del(&self, key: &str) -> Result<()> {
        self.operator
            .delete(&normalize_key(key))
            .await
            .map_err(anyhow::Error::new)
    }

    async fn has(&self, key: &str) -> bool {
        self.operator.stat(&normalize_key(key)).await.is_ok()
    }

    async fn put(&self, key: &str, data: opendal::Buffer) -> Result<Duration> {
        trace!("RemoteStorage::put({key})");
        let start = std::time::Instant::now();
        operator::write_with_retry(&self.operator, key, data).await?;
        Ok(start.elapsed())
    }

    async fn size(&self, key: &str) -> Result<u64> {
        Ok(self
            .operator
            .stat_with(&normalize_key(key))
            .await?
            .content_length())
    }

    async fn check(&self) -> Result<CacheMode> {
        use opendal::ErrorKind;

        let path = ".sccache_check";

        // Read is required, return error directly if we can't read .
        match self.operator.read(path).await {
            Ok(_) => (),
            // Read not exist file with not found is ok.
            Err(err) if err.kind() == ErrorKind::NotFound => (),
            // Tricky Part.
            //
            // We tolerate rate limited here to make sccache keep running.
            // For the worse case, we will miss all the cache.
            //
            // In some super rare cases, user could configure storage in wrong
            // and hitting other services rate limit. There are few things we
            // can do, so we will print our the error here to make users know
            // about it.
            Err(err) if err.kind() == ErrorKind::RateLimited => {
                eprintln!("cache storage read check: {err:?}, but we decide to keep running");
            }
            Err(err) => bail!("cache storage failed to read: {err:?}"),
        }

        let can_write = match self.operator.write(path, "Hello, World!").await {
            Ok(_) => true,
            Err(err) if err.kind() == ErrorKind::AlreadyExists => true,
            Err(err) if err.kind() == ErrorKind::RateLimited => {
                eprintln!("cache storage write check: {err:?}, but we decide to keep running");
                true
            }
            // Tolerate all other write errors because we can do read at least.
            Err(err) => {
                eprintln!("storage write check failed: {err:?}");
                false
            }
        };

        let mode = if can_write {
            CacheMode::ReadWrite
        } else {
            CacheMode::ReadOnly
        };

        debug!("storage check result: {mode:?}");

        Ok(mode)
    }

    async fn location(&self) -> String {
        let meta = self.operator.info();
        format!(
            "{}, name: {}, prefix: {}",
            meta.scheme(),
            meta.name(),
            meta.root()
        )
    }

    fn cache_type_name(&self) -> &'static str {
        // Use opendal's scheme as the cache type name
        // This returns "s3", "redis", "azure", "gcs", etc.
        self.operator.info().scheme()
    }

    async fn current_size(&self) -> Result<Option<u64>> {
        Ok(None)
    }

    async fn max_size(&self) -> Result<Option<u64>> {
        Ok(None)
    }

    /// Return whether the storage is enabled.
    /// Currently only NoStorage impl returns false.
    fn enabled(&self) -> bool {
        true
    }

    fn basedirs(&self) -> &[Vec<u8>] {
        &self.basedirs
    }
}

struct NoStorage;

impl NoStorage {
    pub fn create() -> Arc<dyn Storage> {
        Arc::new(Self {})
    }
}

#[async_trait]
impl Storage for NoStorage {
    async fn get(&self, _: &str) -> Result<Cache<opendal::Buffer>> {
        Ok(Cache::Miss)
    }

    async fn del(&self, _: &str) -> Result<()> {
        Ok(())
    }

    async fn has(&self, _: &str) -> bool {
        false
    }

    async fn put(&self, _: &str, _: opendal::Buffer) -> Result<Duration> {
        Ok(Duration::ZERO)
    }

    async fn size(&self, _: &str) -> Result<u64> {
        Ok(0)
    }

    /// Check the cache capability.
    async fn check(&self) -> Result<CacheMode> {
        Ok(CacheMode::ReadOnly)
    }

    /// Get the storage location.
    async fn location(&self) -> String {
        "nowhere".into()
    }

    /// Get the cache backend type name.
    fn cache_type_name(&self) -> &'static str {
        "none"
    }

    /// Get the current storage usage, if applicable.
    async fn current_size(&self) -> Result<Option<u64>> {
        Ok(Some(0))
    }

    /// Get the maximum storage size, if applicable.
    async fn max_size(&self) -> Result<Option<u64>> {
        Ok(Some(0))
    }

    fn enabled(&self) -> bool {
        false
    }

    fn basedirs(&self) -> &[Vec<u8>] {
        &[]
    }
}

#[derive(Default)]
pub struct StorageBuilder {
    basedirs: Vec<Vec<u8>>,
    enabled: bool,
    create_storage:
        Option<Box<dyn Fn(StorageKind, Vec<Vec<u8>>) -> Result<Arc<dyn Storage>> + Send>>,
    rw_mode: Option<CacheMode>,
    skip_check: bool,
    storage_kind: StorageKind,
    watch_paths: Vec<PathBuf>,
}

impl From<CacheType> for Result<StorageBuilder> {
    fn from(config: CacheType) -> Self {
        #[allow(unreachable_patterns, unused)]
        match config {
            CacheType::Azure(cfg) => {
                #[cfg(feature = "azure")]
                return Ok(cfg.into());
                #[cfg(not(feature = "azure"))]
                bail!("The 'azure' feature must be enabled to use the Azure storage backend")
            }
            CacheType::GCS(cfg) => {
                #[cfg(feature = "gcs")]
                return Ok(cfg.into());
                #[cfg(not(feature = "gcs"))]
                bail!("The 'gcs' feature must be enabled to use the GCS storage backend")
            }
            CacheType::GHA(cfg) => {
                #[cfg(feature = "gha")]
                return Ok(cfg.into());
                #[cfg(not(feature = "gha"))]
                bail!("The 'gha' feature must be enabled to use the GHA storage backend")
            }
            CacheType::Memcached(cfg) => {
                #[cfg(feature = "memcached")]
                return Ok(cfg.into());
                #[cfg(not(feature = "memcached"))]
                bail!(
                    "The 'memcached' feature must be enabled to use the Memcached storage backend"
                )
            }
            CacheType::Redis(cfg) => {
                #[cfg(feature = "redis")]
                return Ok(cfg.into());
                #[cfg(not(feature = "redis"))]
                bail!("The 'redis' feature must be enabled to use the Redis storage backend")
            }
            CacheType::S3(cfg) => {
                #[cfg(feature = "s3")]
                return Ok(cfg.into());
                #[cfg(not(feature = "s3"))]
                bail!("The 's3' feature must be enabled to use the S3 storage backend")
            }
            CacheType::Webdav(cfg) => {
                #[cfg(feature = "webdav")]
                return Ok(cfg.into());
                #[cfg(not(feature = "webdav"))]
                bail!("The 'webdav' feature must be enabled to use the Webdav storage backend")
            }
            CacheType::OSS(cfg) => {
                #[cfg(feature = "oss")]
                return Ok(cfg.into());
                #[cfg(not(feature = "oss"))]
                bail!("The 'oss' feature must be enabled to use the OSS storage backend")
            }
            CacheType::COS(cfg) => {
                #[cfg(feature = "cos")]
                return Ok(cfg.into());
                #[cfg(not(feature = "cos"))]
                bail!("The 'cos' feature must be enabled to use the COS storage backend")
            }
            CacheType::Disk(cfg) => Ok(cfg.into()),
            _ => Ok(StorageBuilder::default()),
        }
    }
}

impl StorageBuilder {
    pub fn basedirs(self, basedirs: Vec<Vec<u8>>) -> Self {
        Self { basedirs, ..self }
    }

    pub fn enabled(self, enabled: bool) -> Self {
        Self { enabled, ..self }
    }

    pub fn storage_kind(self, storage_kind: StorageKind) -> Self {
        Self {
            storage_kind,
            ..self
        }
    }

    pub fn create_storage<
        F: Fn(StorageKind, Vec<Vec<u8>>) -> Result<Arc<dyn Storage>> + Send + 'static,
    >(
        self,
        factory: F,
    ) -> Self {
        Self {
            create_storage: Some(Box::new(factory)),
            ..self
        }
    }

    pub fn rw_mode<P: Into<Option<CacheMode>>>(self, mode: P) -> Self {
        Self {
            rw_mode: mode.into(),
            ..self
        }
    }

    pub fn watch_paths<P: ToOwned<Owned = Vec<PathBuf>>>(self, paths: P) -> Self {
        Self {
            watch_paths: paths.to_owned(),
            ..self
        }
    }

    pub fn skip_check(self, skip_check: bool) -> Self {
        Self { skip_check, ..self }
    }

    pub async fn build(self) -> Result<Arc<dyn Storage>> {
        let Self {
            basedirs,
            enabled,
            create_storage,
            rw_mode,
            skip_check,
            storage_kind,
            #[allow(unused_variables)]
            watch_paths,
        } = self;

        if !enabled {
            // If a storage was configured but was disabled,
            // return a stub disabled Storage implementation
            return Ok(NoStorage::create());
        }

        let rw_mode = rw_mode.unwrap_or_default();
        let create_storage = create_storage.expect("create_storage should exist");

        let create_storage = move |storage_kind, basedirs| {
            use futures::{FutureExt, TryFutureExt, future};
            future::ready(create_storage(storage_kind, basedirs))
                .and_then(move |storage| async move {
                    let name = storage.cache_type_name();

                    let rw_mode = if skip_check {
                        debug!("Cache mode check skipped for {name} storage");
                        rw_mode
                    } else if matches!(rw_mode, CacheMode::ReadOnly) {
                        // No need to check write if we are in manually-set read-only mode
                        rw_mode
                    } else {
                        storage.check().await?
                    };

                    let storage = match rw_mode {
                        CacheMode::ReadWrite => storage,
                        CacheMode::ReadOnly => ReadOnlyStorage::create(storage),
                    };

                    info!("Created {rw_mode} {name} storage");

                    Ok(storage)
                })
                .boxed()
        };

        #[cfg(not(feature = "watcher"))]
        let storage = create_storage(storage_kind, basedirs).await?;
        #[cfg(feature = "watcher")]
        let storage = cache::watch::WatchStorage::from(
            move || create_storage(storage_kind, basedirs.clone()),
            &watch_paths,
        )
        .await?;

        Ok(storage)
    }
}

impl From<config::cache::Disk> for StorageBuilder {
    fn from(config: config::cache::Disk) -> Self {
        let config::cache::Disk {
            enabled,
            dir,
            key_prefix,
            size,
            rw_mode,
            ..
        } = config;

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(
                move |storage_kind, basedirs| {
                    let dir = dir.join(&key_prefix);

                    debug!("Init disk {storage_kind} cache with dir={dir:?}, size={size}, rw_mode={rw_mode:?}, basedirs={:?})", basedirs.iter().map(|b| String::from_utf8_lossy(b)).collect::<Vec<_>>());

                    Ok(Arc::new(DiskCache::new(&dir, size, rw_mode, basedirs)))
                }
            )
    }
}

#[cfg(feature = "azure")]
impl From<config::cache::Azure> for StorageBuilder {
    fn from(config: config::cache::Azure) -> Self {
        let config::cache::Azure {
            enabled,
            auth,
            container,
            key_prefix,
            rw_mode,
            ..
        } = config;

        let mut connection_string = None;
        let mut storage_account = None;
        let mut endpoint = None;

        match auth {
            config::cache::AzureAuth::Endpoint { endpoint: e } => {
                endpoint = Some(e);
            }
            config::cache::AzureAuth::SharedKey {
                connection_string: c,
            } => {
                connection_string = Some(c);
            }
            config::cache::AzureAuth::StorageAccount { storage_account: s } => {
                storage_account = Some(s);
            }
            _ => {}
        }

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(move |storage_kind, basedirs| {
                debug!("Init azure {storage_kind} cache with container={container:?}, key_prefix={key_prefix:?}, storage_account={storage_account:?}, endpoint={endpoint:?}");

                cache::azure::AzureBlobCache::build(connection_string.as_deref(), &container, &key_prefix, storage_account.as_deref(), endpoint.as_deref())
                    .map(|storage| Arc::new(RemoteStorage::new(storage, basedirs)) as Arc<dyn Storage>)
                    .map_err(|err| anyhow!("create azure cache failed: {err:?}"))
            })
    }
}

#[cfg(feature = "gcs")]
impl From<config::cache::GCS> for StorageBuilder {
    fn from(config: config::cache::GCS) -> Self {
        let config::cache::GCS {
            enabled,
            bucket,
            key_prefix,
            key_path: cred_path,
            service_account,
            credentials_url: credential_url,
            rw_mode,
            ..
        } = config;

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(move |storage_kind, basedirs| {
                debug!(
                    "Init gcs {storage_kind} cache with bucket={bucket:?}, key_prefix={key_prefix:?}, cred_path={cred_path:?}, service_account={service_account:?}, credential_url={credential_url:?}"
                );

                cache::gcs::GCSCache::build(
                    &bucket,
                    &key_prefix,
                    cred_path.as_deref(),
                    service_account.as_deref(),
                    rw_mode,
                    credential_url.as_deref(),
                )
                .map(|storage| {
                    Arc::new(RemoteStorage::new(storage, basedirs)) as Arc<dyn Storage>
                })
                .map_err(|err| anyhow!("create gcs cache failed: {err:?}"))
            })
    }
}

#[cfg(feature = "gha")]
impl From<config::cache::GHA> for StorageBuilder {
    fn from(config: config::cache::GHA) -> Self {
        let config::cache::GHA {
            enabled,
            version,
            key_prefix,
            rw_mode,
            ..
        } = config;

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(move |storage_kind, basedirs| {
                debug!(
                    "Init gha {storage_kind} cache with version={version:?}, key_prefix={key_prefix:?}"
                );

                cache::gha::GHACache::build(&version, key_prefix.as_str())
                    .map(|storage| {
                        Arc::new(RemoteStorage::new(storage, basedirs)) as Arc<dyn Storage>
                    })
                    .map_err(|err| anyhow!("create gha cache failed: {err:?}"))
            })
    }
}

#[cfg(feature = "memcached")]
impl From<config::cache::Memcached> for StorageBuilder {
    fn from(config: config::cache::Memcached) -> Self {
        let config::cache::Memcached {
            enabled,
            url,
            username,
            password,
            expiration,
            connection_pool_max_size,
            key_prefix,
            rw_mode,
            ..
        } = config;

        let connection_pool_max_size = connection_pool_max_size.unwrap_or(10);

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(move |storage_kind, basedirs| {
                debug!("Init memcached {storage_kind} cache with url={url:?}");

                cache::memcached::MemcachedCache::build(
                    &url,
                    username.as_deref(),
                    password.as_deref(),
                    &key_prefix,
                    expiration,
                    connection_pool_max_size,
                )
                .map(|storage| Arc::new(RemoteStorage::new(storage, basedirs)) as Arc<dyn Storage>)
                .map_err(|err| anyhow!("create memcached cache failed: {err:?}"))
            })
    }
}

#[cfg(feature = "redis")]
impl From<config::cache::Redis> for StorageBuilder {
    fn from(config: config::cache::Redis) -> Self {
        use crate::cache::simplex::SimplexCache;

        let config::cache::Redis {
            enabled,
            endpoint,
            cluster_endpoints,
            reader_endpoints,
            username,
            password,
            db,
            url,
            ttl,
            connection_pool_max_size,
            key_prefix,
            rw_mode,
            ..
        } = config;

        let connection_pool_max_size = connection_pool_max_size.unwrap_or(10);

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(move |storage_kind, basedirs| {
                match (&endpoint, &cluster_endpoints, &url) {
                    (Some(url), None, None) => {
                        debug!("Init redis single-node {storage_kind} cache with url={url:?}");
                        cache::redis::RedisCache::build_single(
                            url,
                            username.as_deref(),
                            password.as_deref(),
                            db,
                            &key_prefix,
                            ttl,
                            connection_pool_max_size,
                        )
                    }
                    (None, Some(urls), None) => {
                        debug!("Init redis cluster {storage_kind} cache with urls={urls:?}");
                        cache::redis::RedisCache::build_cluster(
                            urls,
                            username.as_deref(),
                            password.as_deref(),
                            db,
                            &key_prefix,
                            ttl,
                            connection_pool_max_size,
                        )
                    }
                    (None, None, Some(url)) => {
                        warn!("Init redis single-node {storage_kind} cache from deprecated API with url={url:?}");
                        if username.is_some() || password.is_some() || db != config::cache::DEFAULT_REDIS_DB {
                            bail!("`username`, `password` and `db` has no effect when `url` is set. Please use `endpoint` or `cluster_endpoints` for new API accessing");
                        }

                        cache::redis::RedisCache::build_from_url(url, &key_prefix, ttl, connection_pool_max_size)
                    }
                    _ => bail!("Only one of `endpoint`, `cluster_endpoints`, `url` must be set"),
                }
                .and_then(|storage| {
                    if let Some(reader_endpoints) = &reader_endpoints {
                        debug!("Init redis cluster {storage_kind} cache with reader_endpoints={reader_endpoints:?}");
                        let reader = if reader_endpoints.contains(",") {
                            cache::redis::RedisCache::build_cluster(
                                reader_endpoints,
                                username.as_deref(),
                                password.as_deref(),
                                db,
                                &key_prefix,
                                ttl,
                                connection_pool_max_size,
                            )?
                        } else {
                            cache::redis::RedisCache::build_single(
                                reader_endpoints,
                                username.as_deref(),
                                password.as_deref(),
                                db,
                                &key_prefix,
                                ttl,
                                connection_pool_max_size,
                            )?
                        };
                        let reader = Arc::new(RemoteStorage::new(reader, basedirs.clone()));
                        let storage = Arc::new(RemoteStorage::new(storage, basedirs));
                        Ok(SimplexCache::create(reader, storage))
                    } else {
                        Ok(Arc::new(RemoteStorage::new(storage, basedirs)))
                    }
                })
                .map_err(|err| anyhow!("create redis cache failed: {err:?}"))
        })
    }
}

#[cfg(feature = "s3")]
impl From<config::cache::S3> for StorageBuilder {
    fn from(config: config::cache::S3) -> Self {
        let config::cache::S3 {
            enabled,
            bucket,
            enable_virtual_host_style,
            endpoint,
            key_prefix,
            no_credentials,
            region,
            server_side_encryption,
            server_side_encryption_aws_kms,
            server_side_encryption_kms_key_id,
            use_ssl,
            rw_mode,
            ..
        } = config;

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(move |storage_kind, basedirs| {
                debug!("Init s3 {storage_kind} cache with endpoint={endpoint:?}, bucket={bucket:?}, key_prefix={key_prefix:?}");

                cache::s3::S3Cache::new(bucket.clone(), key_prefix.clone(), no_credentials)
                    .with_region(region.clone())
                    .with_endpoint(endpoint.clone())
                    .with_use_ssl(use_ssl)
                    .with_server_side_encryption(server_side_encryption)
                    .with_server_side_encryption_aws_kms(server_side_encryption_aws_kms)
                    .with_server_side_encryption_kms_key_id(server_side_encryption_kms_key_id.clone())
                    .with_enable_virtual_host_style(enable_virtual_host_style)
                    .build()
                    .map(|storage| Arc::new(RemoteStorage::new(storage, basedirs)) as Arc<dyn Storage>)
                    .map_err(|err| anyhow!("create s3 cache failed: {err:?}"))
            })
            .watch_paths({
                let ctx = reqsign::default_context();
                [
                    ctx.env_var("AWS_CONFIG_FILE").unwrap_or_else(|| "~/.aws/config".into()),
                    ctx.env_var("AWS_SHARED_CREDENTIALS_FILE").unwrap_or_else(|| "~/.aws/credentials".into()),
                ]
                .iter()
                .filter_map(|s| ctx.expand_home_dir(s))
                .map(PathBuf::from)
                .collect::<Vec<_>>()
            })
    }
}

#[cfg(feature = "webdav")]
impl From<config::cache::Webdav> for StorageBuilder {
    fn from(config: config::cache::Webdav) -> Self {
        let config::cache::Webdav {
            enabled,
            endpoint,
            key_prefix,
            password,
            token,
            username,
            rw_mode,
            ..
        } = config;

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(move |storage_kind, basedirs| {
                debug!("Init webdav {storage_kind} cache with endpoint {endpoint}");

                cache::webdav::WebdavCache::build(
                    &endpoint,
                    &key_prefix,
                    username.as_deref(),
                    password.as_deref(),
                    token.as_deref(),
                )
                .map(|storage| Arc::new(RemoteStorage::new(storage, basedirs)) as Arc<dyn Storage>)
                .map_err(|err| anyhow!("create webdav cache failed: {err:?}"))
            })
    }
}

#[cfg(feature = "oss")]
impl From<config::cache::OSS> for StorageBuilder {
    fn from(config: config::cache::OSS) -> Self {
        let config::cache::OSS {
            enabled,
            bucket,
            endpoint,
            key_prefix,
            no_credentials,
            rw_mode,
            ..
        } = config;

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(move |storage_kind, basedirs| {
                debug!("Init oss {storage_kind} cache with bucket {bucket}, endpoint {endpoint:?}");

                cache::oss::OSSCache::build(
                    &bucket,
                    &key_prefix,
                    endpoint.as_deref(),
                    no_credentials,
                )
                .map(|storage| Arc::new(RemoteStorage::new(storage, basedirs)) as Arc<dyn Storage>)
                .map_err(|err| anyhow!("create oss cache failed: {err:?}"))
            })
    }
}

#[cfg(feature = "cos")]
impl From<config::cache::COS> for StorageBuilder {
    fn from(config: config::cache::COS) -> Self {
        let config::cache::COS {
            enabled,
            bucket,
            endpoint,
            key_prefix,
            rw_mode,
            ..
        } = config;

        Self::default()
            .enabled(enabled)
            .rw_mode(Some(rw_mode))
            .create_storage(move |storage_kind, basedirs| {
                debug!("Init cos {storage_kind} cache with bucket {bucket}, endpoint {endpoint:?}");

                cache::cos::COSCache::build(&bucket, &key_prefix, endpoint.as_deref())
                    .map(|storage| {
                        Arc::new(RemoteStorage::new(storage, basedirs)) as Arc<dyn Storage>
                    })
                    .map_err(|err| anyhow!("create oss cache failed: {err:?}"))
            })
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum StorageKind {
    #[default]
    Compilations,
}

impl fmt::Display for StorageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compilations => write!(f, "object"),
        }
    }
}

pub struct StorageArgs<I>
where
    I: Iterator<Item = Result<StorageBuilder>>,
{
    pub builders: I,
    pub skip_check: bool,
    pub rate_limit: RateLimits,
    pub read_timeout: Duration,
    pub write_timeout: Option<Duration>,
    pub write_policy: WriteErrorPolicy,
}

pub struct StorageBuilderFromCacheTypeIter<B>
where
    B: Into<Result<StorageBuilder>>,
{
    configs: Vec<B>,
    pos: usize,
}

impl<B> StorageBuilderFromCacheTypeIter<B>
where
    B: Into<Result<StorageBuilder>>,
{
    fn new(configs: Vec<B>) -> Self {
        Self { configs, pos: 0 }
    }
}

impl<B> Iterator for StorageBuilderFromCacheTypeIter<B>
where
    B: Into<Result<StorageBuilder>> + Clone,
{
    type Item = Result<StorageBuilder>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(config) = self.configs.get(self.pos) {
            self.pos += 1;
            Some(config.clone().into())
        } else {
            None
        }
    }
}

impl<'a> From<&'a Caches> for StorageArgs<StorageBuilderFromCacheTypeIter<CacheType>> {
    fn from(caches: &'a Caches) -> StorageArgs<StorageBuilderFromCacheTypeIter<CacheType>> {
        StorageArgs {
            builders: StorageBuilderFromCacheTypeIter::new(
                caches
                    .multilevel
                    .chain
                    .as_deref()
                    .map(|chain| {
                        chain
                            .iter()
                            .filter_map(|name| caches.configs.get(name))
                            .cloned()
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
            ),
            skip_check: caches.skip_check,
            rate_limit: caches.rate_limit.clone(),
            read_timeout: Duration::from_secs(caches.read_timeout_secs),
            write_timeout: caches.write_timeout_secs.map(Duration::from_secs),
            write_policy: caches.multilevel.write_error_policy,
        }
    }
}

impl StorageKind {
    /// Create suitable `Storage` implementations from a list of storage configurations.
    pub async fn create<I, S: Into<StorageArgs<I>>>(
        &self,
        args: S,
        basedirs: &[Vec<u8>],
    ) -> Result<Arc<dyn Storage>>
    where
        I: Iterator<Item = Result<StorageBuilder>>,
    {
        let StorageArgs {
            builders,
            skip_check,
            rate_limit,
            read_timeout,
            write_timeout,
            write_policy,
        } = args.into();

        let kind = *self;

        let mut caches = vec![];

        for builder in builders {
            caches.push(
                builder?
                    .basedirs(basedirs.to_vec())
                    .skip_check(skip_check)
                    .storage_kind(kind)
                    .build()
                    .await
                    .inspect_err(|err| error!("storage init failed for {kind} cache: {err:?}"))?,
            );
        }

        // If any caches are enabled, remove the disabled ones
        caches.retain(|c| c.enabled());

        let storage = if caches.len() == 1 {
            caches[0].clone()
        } else if caches.len() > 1 {
            Arc::new(MultiLevelStorage::with_write_error_policy(caches, write_policy).await)
        } else {
            // If we build only with `cargo build --no-default-features`
            // only use sccache with a local cache (no remote storage)
            StorageBuilder::from(config::cache::Disk::default())
                .basedirs(basedirs.to_vec())
                .skip_check(skip_check)
                .storage_kind(kind)
                .build()
                .await?
        };

        info!("Configured {kind} storage");

        let storage = if let Some(on_error_count) = rate_limit.on_error_count {
            RateLimitStorage::create(
                storage,
                read_timeout,
                write_timeout,
                on_error_count,
                Duration::from_secs(rate_limit.on_error_window_size_secs.unwrap_or(60)),
            )
        } else {
            storage
        };

        Ok(storage)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::{
        config::{CacheMode, ClientConfig},
        test::utils::single_threaded_runtime,
    };
    use fs_err as fs;

    struct NoCheckStorage;

    #[async_trait]
    impl Storage for NoCheckStorage {
        async fn get(&self, _: &str) -> Result<Cache<opendal::Buffer>> {
            panic!("get() should not be called")
        }
        async fn del(&self, _: &str) -> Result<()> {
            panic!("del() should not be called")
        }
        async fn has(&self, _: &str) -> bool {
            panic!("has() should not be called")
        }
        async fn put(&self, _: &str, _: opendal::Buffer) -> Result<Duration> {
            panic!("put() should not be called")
        }
        async fn size(&self, _: &str) -> Result<u64> {
            panic!("size() should not be called")
        }
        async fn check(&self) -> Result<CacheMode> {
            panic!("check() should not be called")
        }
        async fn location(&self) -> String {
            "nowhere".into()
        }
        fn cache_type_name(&self) -> &'static str {
            "no-check"
        }
        async fn current_size(&self) -> Result<Option<u64>> {
            panic!("current_size() should not be called")
        }
        async fn max_size(&self) -> Result<Option<u64>> {
            panic!("max_size() should not be called")
        }
        fn enabled(&self) -> bool {
            true
        }
        fn basedirs(&self) -> &[Vec<u8>] {
            &[]
        }
    }

    impl From<NoCheckStorage> for StorageBuilder {
        fn from(storage: NoCheckStorage) -> Self {
            let storage = Arc::new(storage);
            Self::default()
                .enabled(true)
                .create_storage(move |_, _| Ok(storage.clone()))
        }
    }

    #[test]
    fn test_read_write_mode_local() {
        let runtime = single_threaded_runtime();

        // Use disk cache.
        let tempdir = crate::util::temp_dir()
            .context("Failed to create tempdir")
            .unwrap();

        let cache_dir = tempdir.path().join("cache");
        fs::create_dir(&cache_dir).unwrap();

        let make_config = |rw_mode| ClientConfig {
            cache: vec![
                config::cache::Disk {
                    dir: cache_dir.clone(),
                    rw_mode,
                    ..config::cache::Disk::default()
                }
                .into(),
            ]
            .into(),
            ..Default::default()
        };

        // Test Read Write
        {
            crate::util::block_on(&runtime, async {
                let config = make_config(CacheMode::ReadWrite);
                let storage = StorageKind::Compilations
                    .create(&config.cache, &[])
                    .await
                    .unwrap();
                storage.put("test1", "entry".into()).await.unwrap();
            });
        }

        // Test Read-only
        {
            crate::util::block_on(&runtime, async {
                let config = make_config(CacheMode::ReadOnly);
                let storage = StorageKind::Compilations
                    .create(&config.cache, &[])
                    .await
                    .unwrap();
                assert_eq!(
                    storage
                        .put("test1", "entry".into())
                        .await
                        .unwrap_err()
                        .to_string(),
                    "Cannot write to read-only storage"
                );
            });
        }
    }

    #[test]
    #[cfg(feature = "s3")]
    fn test_operator_storage_s3_with_basedirs() {
        // Create S3 operator (doesn't need real credentials for this test)
        let operator = crate::cache::s3::S3Cache::new(
            "test-bucket".to_string(),
            "test-prefix".to_string(),
            true, // no_credentials = true
        )
        .with_region(Some("us-east-1".to_string()))
        .build()
        .expect("Failed to create S3 cache operator");

        let basedirs = vec![b"/home/user/project".to_vec(), b"/opt/build".to_vec()];

        // Wrap with RemoteStorage
        let storage = RemoteStorage::new(operator, basedirs.clone());

        // Verify basedirs are stored and retrieved correctly
        let basedirs_actual = storage.basedirs();
        assert_eq!(basedirs_actual, basedirs.as_slice());
        assert_eq!(basedirs_actual.len(), 2);
        assert_eq!(basedirs_actual[0], b"/home/user/project".to_vec());
        assert_eq!(basedirs_actual[1], b"/opt/build".to_vec());
    }

    #[test]
    fn test_skip_cache_check() -> Result<()> {
        drop(env_logger::try_init());

        let runtime = single_threaded_runtime();

        crate::util::block_on(&runtime, async {
            StorageKind::Compilations
                .create(
                    StorageArgs {
                        builders: [Ok(NoCheckStorage.into())].into_iter(),
                        skip_check: true,
                        rate_limit: Default::default(),
                        read_timeout: Duration::from_secs(60),
                        write_timeout: None,
                        write_policy: Default::default(),
                    },
                    &[],
                )
                .await?;

            Ok(())
        })
    }

    #[test]
    fn test_skip_cache_check_multilevel() -> Result<()> {
        drop(env_logger::try_init());

        let runtime = single_threaded_runtime();

        crate::util::block_on(&runtime, async {
            StorageKind::Compilations
                .create(
                    StorageArgs {
                        builders: [Ok(NoCheckStorage.into()), Ok(NoCheckStorage.into())]
                            .into_iter(),
                        skip_check: true,
                        rate_limit: Default::default(),
                        read_timeout: Duration::from_secs(60),
                        write_timeout: None,
                        write_policy: Default::default(),
                    },
                    &[],
                )
                .await?;

            Ok(())
        })
    }

    #[test]
    #[cfg(feature = "redis")]
    fn test_operator_storage_redis_with_basedirs() {
        // Create Redis operator
        let operator = crate::cache::redis::RedisCache::build_single(
            "redis://localhost:6379",
            None,
            None,
            0,
            "test-prefix",
            0,
            10,
        )
        .expect("Failed to create Redis cache operator");

        let basedirs = vec![b"/workspace".to_vec()];

        // Wrap with RemoteStorage
        let storage = RemoteStorage::new(operator, basedirs.clone());

        // Verify basedirs work
        let basedirs_actual = storage.basedirs();
        assert_eq!(basedirs_actual, basedirs.as_slice());
        assert_eq!(basedirs_actual.len(), 1);
    }

    #[test]
    #[cfg(feature = "redis")]
    fn test_operator_storage_redis_with_read_only() {
        // Create Redis operator

        use crate::test::utils::Waiter;
        let operator = crate::cache::redis::RedisCache::build_single(
            "redis://localhost:6379",
            None,
            None,
            0,
            "test-prefix",
            0,
            10,
        )
        .expect("Failed to create Redis cache operator");

        // Wrap with ReadOnlyStorage(RemoteStorage)
        let storage = ReadOnlyStorage::create(Arc::new(RemoteStorage::new(operator, vec![])));

        // Verify put fails
        let result = storage.put("test", opendal::Buffer::new()).wait();
        match result {
            Ok(_) => panic!("expected error, got success {result:?}"),
            Err(err) => assert!(err.to_string().contains("read-only")),
        }
    }
}
