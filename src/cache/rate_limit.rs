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

use async_trait::async_trait;

use crate::{
    cache::{Cache, GetPathResult, Storage, multilevel::MultiLevelStats},
    config::CacheMode,
    errors::*,
};

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub struct RateLimitStorage {
    inner: Arc<dyn Storage>,
    errors: tokio::sync::Mutex<Vec<Instant>>,
    error_count: usize,
    read_timeout: Duration,
    write_timeout: Option<Duration>,
    window_size: Duration,
}

impl RateLimitStorage {
    pub fn create(
        inner: Arc<dyn Storage>,
        read_timeout: Duration,
        write_timeout: Option<Duration>,
        error_count: usize,
        window_size: Duration,
    ) -> Arc<dyn Storage> {
        Arc::new(Self {
            inner,
            read_timeout,
            write_timeout,
            error_count,
            window_size,
            errors: Default::default(),
        })
    }

    async fn is_rate_limited(&self) -> bool {
        fn truncate_window(count: usize, window_size: Duration, window: &mut Vec<Instant>) -> bool {
            let now = Instant::now();
            // Drop all error timestamps older than `window_size`
            window.retain(|then| now.duration_since(*then) < window_size);
            // Drop all but the last `count` number of error timestamps
            window.drain(..window.len().saturating_sub(count));
            window.len() >= count
        }
        let mut window = self.errors.lock().await;
        truncate_window(self.error_count, self.window_size, &mut window)
    }
}

macro_rules! maybe_rate_limited {
    ($self:ident, $timeout:expr, $inner:expr) => {
        if $self.is_rate_limited().await {
            bail!("Rate limited due to error count")
        } else if let Some(timeout) = $timeout {
            match tokio::time::timeout(timeout, $inner).await {
                Ok(res) => res,
                Err(err) => {
                    let now = Instant::now();
                    $self.errors.lock().await.push(now);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    bail!(err.to_string())
                }
            }
        } else {
            match $inner.await {
                Ok(res) => Ok(res),
                Err(err) => {
                    let now = Instant::now();
                    $self.errors.lock().await.push(now);
                    Err(err)
                }
            }
        }
    };
}

#[async_trait]
impl Storage for RateLimitStorage {
    async fn get(&self, key: &str) -> Result<Cache<opendal::Buffer>> {
        maybe_rate_limited!(self, Some(self.read_timeout), self.inner.get(key))
    }

    async fn del(&self, key: &str) -> Result<()> {
        maybe_rate_limited!(self, self.write_timeout, self.inner.del(key))
    }

    async fn has(&self, key: &str) -> bool {
        if self.is_rate_limited().await {
            false
        } else {
            self.inner.has(key).await
        }
    }

    async fn put(&self, key: &str, entry: opendal::Buffer) -> Result<Duration> {
        maybe_rate_limited!(self, self.write_timeout, self.inner.put(key, entry))
    }

    async fn size(&self, key: &str) -> Result<u64> {
        maybe_rate_limited!(self, Some(self.read_timeout), self.inner.size(key))
    }

    /// Check the cache capability.
    async fn check(&self) -> Result<CacheMode> {
        self.inner.check().await
    }

    /// Get the storage location.
    async fn location(&self) -> String {
        self.inner.location().await
    }

    /// Get the cache backend type name.
    fn cache_type_name(&self) -> &'static str {
        self.inner.cache_type_name()
    }

    /// Get the current storage usage, if applicable.
    async fn current_size(&self) -> Result<Option<u64>> {
        self.inner.current_size().await
    }

    /// Get the maximum storage size, if applicable.
    async fn max_size(&self) -> Result<Option<u64>> {
        self.inner.max_size().await
    }

    fn multilevel_stats(&self) -> Option<MultiLevelStats> {
        self.inner.multilevel_stats()
    }

    /// Return whether the storage is enabled.
    /// Currently only NoStorage impl returns false.
    fn enabled(&self) -> bool {
        self.inner.enabled()
    }

    /// Return the base directories for path normalization if configured
    fn basedirs(&self) -> &[Vec<u8>] {
        self.inner.basedirs()
    }

    async fn get_path(&self, key: &str) -> GetPathResult {
        self.inner.get_path(key).await
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        cache::readonly::ReadOnlyStorage,
        test::{mock_storage::MockStorage, *},
    };

    #[test]
    fn test_rate_limit_storage() -> Result<()> {
        let rt = utils::single_threaded_runtime();

        crate::util::block_on(&rt, async {
            let storage = Arc::new(MockStorage::new(None, true));

            storage.next_get(Ok(Cache::Hit(b"val".to_vec().into())));

            let storage = RateLimitStorage::create(
                ReadOnlyStorage::create(storage),
                Duration::from_secs(1), // read_timeout
                None,                   // write_timeout
                3,                      // error_count
                Duration::from_secs(1), // window_size
            );

            // Three errors should trigger the rate limiter
            let _ = tokio::join!(
                storage.put("key", opendal::Buffer::new()),
                storage.put("key", opendal::Buffer::new()),
                storage.put("key", opendal::Buffer::new()),
            );
            assert_eq!(
                storage.get("key").await.unwrap_err().to_string(),
                "Rate limited due to error count"
            );

            // Wait 2s for the rate limit window to pass
            tokio::time::sleep(Duration::from_secs(2)).await;

            // Ensure we can get the key seeded to MockStorage at the start
            assert!(matches!(storage.get("key").await.unwrap(), Cache::Hit(..)));

            Ok::<_, anyhow::Error>(())
        })?;

        Ok(())
    }

    #[test]
    fn test_rate_limit_storage_read_timeout() -> Result<()> {
        let rt = utils::single_threaded_runtime();

        crate::util::block_on(&rt, async {
            let storage = Arc::new(MockStorage::new(Some(Duration::from_secs(1)), true));

            storage.next_get(Ok(Cache::Hit(b"val".to_vec().into())));

            let storage = RateLimitStorage::create(
                storage,
                Duration::from_millis(1), // read_timeout
                None,                     // write_timeout
                3,                        // error_count
                Duration::from_secs(5),   // window_size
            );

            // Three timeout errors should trigger the rate limiter
            let _ = tokio::join!(
                storage.get("key"), //
                storage.get("key"), //
                storage.get("key"), //
            );
            assert_eq!(
                storage.get("key").await.unwrap_err().to_string(),
                "Rate limited due to error count"
            );

            Ok::<_, anyhow::Error>(())
        })?;

        Ok(())
    }

    #[test]
    fn test_rate_limit_storage_write_timeout() -> Result<()> {
        let rt = utils::single_threaded_runtime();

        crate::util::block_on(&rt, async {
            let storage = Arc::new(MockStorage::new(Some(Duration::from_secs(1)), true));

            storage.next_get(Ok(Cache::Hit(b"val".to_vec().into())));

            let storage = RateLimitStorage::create(
                storage,
                Duration::from_secs(5),         // read_timeout
                Some(Duration::from_millis(1)), // write_timeout
                3,                              // error_count
                Duration::from_secs(5),         // window_size
            );

            // Three timeout errors should trigger the rate limiter
            let _ = tokio::join!(
                storage.put("key", opendal::Buffer::new()),
                storage.put("key", opendal::Buffer::new()),
                storage.put("key", opendal::Buffer::new()),
            );
            assert_eq!(
                storage.get("key").await.unwrap_err().to_string(),
                "Rate limited due to error count"
            );

            // Wait 6s for the rate limit window to pass
            tokio::time::sleep(Duration::from_secs(6)).await;

            // Ensure we can get the key seeded to MockStorage at the start
            assert!(matches!(storage.get("key").await.unwrap(), Cache::Hit(..)));

            Ok::<_, anyhow::Error>(())
        })?;

        Ok(())
    }
}
