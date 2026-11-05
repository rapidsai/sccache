// Copyright 2026 Mozilla Foundation
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

use super::*;
use crate::{
    cache::{CacheRead, CacheWrite, StorageKind, disk::DiskCache, readonly::ReadOnlyStorage},
    config::ClientConfig,
};
use std::{collections::HashMap, env, fs, sync::Arc, time::Duration};
use tempfile::Builder as TempBuilder;
use tokio::{runtime::Builder as RuntimeBuilder, sync::Mutex, time::sleep};

#[test]
fn test_multi_level_storage_get() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let tempdir1 = TempBuilder::new()
        .prefix("sccache_test_l1_")
        .tempdir()
        .unwrap();
    let cache_dir1 = tempdir1.path().join("cache");
    fs::create_dir(&cache_dir1).unwrap();

    let tempdir2 = TempBuilder::new()
        .prefix("sccache_test_l2_")
        .tempdir()
        .unwrap();
    let cache_dir2 = tempdir2.path().join("cache");
    fs::create_dir(&cache_dir2).unwrap();

    let cache1 = DiskCache::new(&cache_dir1, 1024 * 1024 * 100, CacheMode::ReadWrite, vec![]);
    let cache2 = DiskCache::new(&cache_dir2, 1024 * 1024 * 100, CacheMode::ReadWrite, vec![]);

    let cache1_storage: Arc<dyn Storage> = Arc::new(cache1);
    let cache2_storage: Arc<dyn Storage> = Arc::new(cache2);

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            Arc::clone(&cache1_storage),
            Arc::clone(&cache2_storage),
        ])
        .await;

        // Write directly to level 2 (level 1 is empty)
        {
            let entry = opendal::Buffer::default();
            cache2_storage.put("test_key", entry).await.unwrap();
        }

        // Now try to read through multi-level storage
        match storage.get("test_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - found at level 2
            }
            _ => panic!("Expected cache hit at level 2"),
        }

        // Try non-existent key
        match storage.get("nonexistent").await.unwrap() {
            Cache::Miss => {
                // Expected
            }
            _ => panic!("Expected cache miss"),
        }
    });
}

#[test]
fn test_multi_level_storage_backfill_on_hit() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let tempdir1 = TempBuilder::new()
        .prefix("sccache_test_bf_l1_")
        .tempdir()
        .unwrap();
    let cache_dir1 = tempdir1.path().join("cache");
    fs::create_dir(&cache_dir1).unwrap();

    let tempdir2 = TempBuilder::new()
        .prefix("sccache_test_bf_l2_")
        .tempdir()
        .unwrap();
    let cache_dir2 = tempdir2.path().join("cache");
    fs::create_dir(&cache_dir2).unwrap();

    let cache1 = DiskCache::new(&cache_dir1, 1024 * 1024 * 100, CacheMode::ReadWrite, vec![]);
    let cache2 = DiskCache::new(&cache_dir2, 1024 * 1024 * 100, CacheMode::ReadWrite, vec![]);

    let cache1_storage: Arc<dyn Storage> = Arc::new(cache1);
    let cache2_storage: Arc<dyn Storage> = Arc::new(cache2);

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            Arc::clone(&cache1_storage),
            Arc::clone(&cache2_storage),
        ])
        .await;

        // Write directly to level 2 (level 1 is empty)
        {
            let entry = opendal::Buffer::default();
            cache2_storage.put("backfill_key", entry).await.unwrap();
        }

        // Verify level 1 doesn't have it yet
        match cache1_storage.get("backfill_key").await.unwrap() {
            Cache::Miss => {
                // Expected - level 1 is empty
            }
            _ => panic!("Level 1 should be empty"),
        }

        // Now read through multi-level storage - should hit level 2 and backfill to level 1
        match storage.get("backfill_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - found at level 2
            }
            _ => panic!("Expected cache hit at level 2"),
        }

        // Give background backfill task time to complete
        sleep(Duration::from_millis(200)).await;

        // Now level 1 should have the data (backfilled)
        match cache1_storage.get("backfill_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - backfilled from level 2
            }
            _ => panic!("Level 1 should now have the data (backfilled)"),
        }
    });
}

/// In-memory storage mock for testing multi-level backfill with remote-like backends.
///
/// This is used to test multi-level cache backfill logic without requiring:
/// - Network access to real remote services (S3, Redis, etc.)
/// - Complex mock infrastructure (channels, queues, etc.)
/// - Disk I/O operations
///
/// The mock implements both Storage trait and get() to simulate real backend
/// behavior where remote caches support raw byte retrieval for efficient backfilling.
struct InMemoryStorage {
    data: Arc<Mutex<HashMap<String, opendal::Buffer>>>,
    access_log: Arc<Mutex<Vec<String>>>,
}

impl InMemoryStorage {
    fn new() -> Self {
        Self {
            data: Arc::new(Mutex::new(HashMap::new())),
            access_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_access_log(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.access_log)
    }
}

#[async_trait]
impl Storage for InMemoryStorage {
    async fn get(&self, key: &str) -> Result<Cache<opendal::Buffer>> {
        self.access_log.lock().await.push(format!("get:{key}"));

        match self.data.lock().await.get(key).cloned() {
            Some(data) => Ok(Cache::Hit(data)),
            None => Ok(Cache::Miss),
        }
    }

    async fn del(&self, key: &str) -> Result<()> {
        self.data.lock().await.remove(key);
        Ok(())
    }

    async fn put(&self, key: &str, entry: opendal::Buffer) -> Result<Duration> {
        self.access_log.lock().await.push(format!("put:{key}"));

        self.data.lock().await.insert(key.to_string(), entry);
        Ok(Duration::ZERO)
    }

    async fn has(&self, key: &str) -> bool {
        self.size(key).await.is_ok()
    }

    async fn check(&self) -> Result<CacheMode> {
        Ok(CacheMode::ReadWrite)
    }

    async fn size(&self, key: &str) -> Result<u64> {
        match self.data.lock().await.get(key) {
            Some(data) => Ok(data.len() as u64),
            None => Err(anyhow!("Unknown key {key:?}")),
        }
    }

    async fn location(&self) -> String {
        "InMemory".to_string()
    }

    fn cache_type_name(&self) -> &'static str {
        "InMemory"
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
        &[]
    }
}

#[test]
fn test_multilevel_raw_hit_reads_backend_once() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let l0 = Arc::new(InMemoryStorage::new());
    let l1 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            l0.clone() as Arc<dyn Storage>,
            l1.clone() as Arc<dyn Storage>,
        ])
        .await;

        let entry = CacheWrite::default();
        l1.put(
            "single_read_key",
            opendal::Buffer::from(entry.finish().unwrap()),
        )
        .await
        .unwrap();

        // Ignore setup writes and assert the lookup path itself.  A raw-capable
        // level should be read once, then the same bytes should feed both
        // parsing and the asynchronous backfill.
        l0.get_access_log().lock().await.clear();
        l1.get_access_log().lock().await.clear();

        assert!(matches!(
            storage.get("single_read_key").await.unwrap(),
            Cache::Hit(_)
        ));

        assert_eq!(
            l1.get_access_log().lock().await.as_slice(),
            &["get:single_read_key"]
        );
    });
}

#[test]
fn test_disk_plus_remote_to_remote_backfill() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // Create multi-level cache: Disk (L0) + Memcached (L1) + Redis (L2) + S3 (L3)
    // This simulates a real-world setup with local disk cache and multiple remote caches
    let tempdir = TempBuilder::new()
        .prefix("sccache_test_multilevel_")
        .tempdir()
        .unwrap();
    let cache_dir = tempdir.path().join("cache");
    fs::create_dir(&cache_dir).unwrap();

    let disk_cache = Arc::new(DiskCache::new(
        &cache_dir,
        1024 * 1024 * 100,
        CacheMode::ReadWrite,
        vec![],
    ));

    let remote_l1 = Arc::new(InMemoryStorage::new()); // Memcached-like
    let remote_l2 = Arc::new(InMemoryStorage::new()); // Redis-like
    let remote_l3 = Arc::new(InMemoryStorage::new()); // S3-like

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            disk_cache.clone() as Arc<dyn Storage>,
            remote_l1.clone() as Arc<dyn Storage>,
            remote_l2.clone() as Arc<dyn Storage>,
            remote_l3.clone() as Arc<dyn Storage>,
        ])
        .await;
        // Scenario: Data only in S3 (L3), need to backfill all the way to local disk (L0)
        {
            let entry = opendal::Buffer::default();
            remote_l3.put("global_key", entry).await.unwrap();
        }

        // Verify only L3 has it
        assert!(matches!(
            disk_cache.get("global_key").await.unwrap(),
            Cache::Miss
        ));
        assert!(matches!(
            remote_l1.get("global_key").await.unwrap(),
            Cache::Miss
        ));
        assert!(matches!(
            remote_l2.get("global_key").await.unwrap(),
            Cache::Miss
        ));

        // Read through multi-level storage - should hit L3 and backfill everywhere
        match storage.get("global_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - found at L3
            }
            _ => panic!("Expected cache hit at L3"),
        }

        // Give all background backfill tasks time to complete
        // We have 3 backfill tasks (L3 -> L2, L3 -> L1, L3 -> L0)
        sleep(Duration::from_millis(400)).await;

        // Verify local disk was backfilled (closest to CPU)
        match disk_cache.get("global_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - backfilled from L3 to disk cache
            }
            _ => panic!("Disk cache should be backfilled from L3"),
        }

        // Verify remote L1 was backfilled
        match remote_l1.get("global_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected
            }
            _ => panic!("Remote L1 should be backfilled from L3"),
        }

        // Verify remote L2 was backfilled
        match remote_l2.get("global_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected
            }
            _ => panic!("Remote L2 should be backfilled from L3"),
        }

        // Now reading should hit at L0 (disk) - fastest
        match storage.get("global_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - immediate local disk hit
            }
            _ => panic!("Should hit at disk cache (L0)"),
        }
    });
}

#[test]
fn test_disk_plus_remotes_write_to_all() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // Test write path: ensure data is written to all levels
    let tempdir = TempBuilder::new()
        .prefix("sccache_test_write_all_")
        .tempdir()
        .unwrap();
    let cache_dir = tempdir.path().join("cache");
    fs::create_dir(&cache_dir).unwrap();

    let disk_cache = Arc::new(DiskCache::new(
        &cache_dir,
        1024 * 1024 * 100,
        CacheMode::ReadWrite,
        vec![],
    ));

    let remote_l1 = Arc::new(InMemoryStorage::new());
    let remote_l2 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            disk_cache.clone() as Arc<dyn Storage>,
            remote_l1.clone() as Arc<dyn Storage>,
            remote_l2.clone() as Arc<dyn Storage>,
        ])
        .await;

        // Write through multi-level should go to all levels
        {
            let entry = opendal::Buffer::default();
            storage.put("write_test_key", entry).await.unwrap();
        }

        // Give async writes time to complete
        sleep(Duration::from_millis(200)).await;

        // Verify disk cache has it
        match disk_cache.get("write_test_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - written to disk synchronously
            }
            _ => panic!("Disk cache should have data after put"),
        }

        // Verify both remote caches have it
        match remote_l1.get("write_test_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - written to L1 asynchronously
            }
            _ => panic!("Remote L1 should have data after put"),
        }

        match remote_l2.get("write_test_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - written to L2 asynchronously
            }
            _ => panic!("Remote L2 should have data after put"),
        }
    });
}

#[test]
fn test_remote_to_remote_backfill() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // Create three in-memory "remote" caches to simulate:
    // L0: Memcached (fast, small)
    // L1: Redis (medium, medium)
    // L2: S3 (slow, large)
    let cache_l0 = Arc::new(InMemoryStorage::new());
    let cache_l1 = Arc::new(InMemoryStorage::new());
    let cache_l2 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            cache_l0.clone() as Arc<dyn Storage>,
            cache_l1.clone() as Arc<dyn Storage>,
            cache_l2.clone() as Arc<dyn Storage>,
        ])
        .await;

        // Simulate cache miss at L0 and L1, hit at L2 (typical scenario)
        {
            let entry = opendal::Buffer::default();
            cache_l2.put("remote_key", entry).await.unwrap();
        }

        // Verify L0 and L1 are empty (cache misses at those levels)
        match cache_l0.get("remote_key").await.unwrap() {
            Cache::Miss => {}
            _ => panic!("L0 should be empty initially"),
        }
        match cache_l1.get("remote_key").await.unwrap() {
            Cache::Miss => {}
            _ => panic!("L1 should be empty initially"),
        }

        // Read through multi-level storage - should hit L2 and backfill to L0 and L1
        match storage.get("remote_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - found at L2
            }
            _ => panic!("Expected cache hit at L2"),
        }

        // Give background backfill tasks time to complete
        // Multiple levels means multiple concurrent spawn tasks
        sleep(Duration::from_millis(300)).await;

        // Verify L0 was backfilled from L2 (through L1)
        match cache_l0.get("remote_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - backfilled from L2 via L1
            }
            _ => panic!("L0 should be backfilled from L2"),
        }

        // Verify L1 was backfilled from L2
        match cache_l1.get("remote_key").await.unwrap() {
            Cache::Hit(_) => {
                // Expected - backfilled from L2
            }
            _ => panic!("L1 should be backfilled from L2"),
        }
    });
}

#[test]
#[serial_test::serial(config_from_env)]
fn test_config_validation_invalid_level_name() {
    // Test that invalid level names are rejected
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // Set invalid level name
    unsafe {
        env::set_var("SCCACHE_MULTILEVEL_CHAIN", "disk,invalid_backend,s3");
        env::set_var("SCCACHE_DIR", "/tmp/test-cache");
    }

    let config = match ClientConfig::load() {
        Ok(config) => config,
        Err(e) => {
            let err_msg = format!("{e}");
            assert!(err_msg.contains("Unknown cache level") || err_msg.contains("invalid_backend"));
            return;
        }
    };

    let result = crate::util::block_on(
        &runtime,
        StorageKind::Compilations.create(&config.cache, &[]),
    );

    // Should error with unknown cache level
    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = format!("{e}");
        assert!(err_msg.contains("Unknown cache level") || err_msg.contains("invalid_backend"));
    }

    unsafe {
        env::remove_var("SCCACHE_MULTILEVEL_CHAIN");
        env::remove_var("SCCACHE_DIR");
    }
}

#[test]
fn test_config_validation_empty_levels() {
    // Test that empty levels list is handled

    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![]).await;

        // Get should return miss (no levels to check)
        match storage.get("test_key").await.unwrap() {
            Cache::Miss => {} // Expected
            _ => panic!("Empty levels should always miss"),
        }
    });
}

#[test]
fn test_config_validation_single_level() {
    // Test that single level works (passthrough mode)
    let cache = Arc::new(InMemoryStorage::new());

    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![cache.clone() as Arc<dyn Storage>]).await;

        let entry = opendal::Buffer::default();
        storage.put("single_key", entry).await.unwrap();

        match storage.get("single_key").await.unwrap() {
            Cache::Hit(_) => {} // Expected
            _ => panic!("Single level should work as passthrough"),
        }

        // Should not backfill since only one level
        match cache.get("single_key").await.unwrap() {
            Cache::Hit(_) => {} // Expected - data is there
            _ => panic!("Data should be in the single level"),
        }
    });
}

#[test]
#[serial_test::serial(config_from_env)]
fn test_config_level_not_configured() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // Set level without configuration
    unsafe {
        env::set_var("SCCACHE_MULTILEVEL_CHAIN", "redis");
        // Don't set SCCACHE_REDIS_ENDPOINT
        env::remove_var("SCCACHE_REDIS");
        env::remove_var("SCCACHE_REDIS_ENDPOINT");
    }

    let config = match ClientConfig::load() {
        Ok(config) => config,
        Err(e) => {
            let err_msg = format!("{e}");
            assert!(
                err_msg.contains("not configured")
                    || err_msg.contains("missing")
                    || err_msg.contains("requires")
                    || err_msg.contains("none could be built"),
                "Expected error about missing config or feature, got: {err_msg}"
            );
            return;
        }
    };

    let result = crate::util::block_on(
        &runtime,
        StorageKind::Compilations.create(&config.cache, &[]),
    );

    // Should error with "not configured" or "requires" (when feature disabled)
    assert!(result.is_err());

    if let Err(e) = result {
        let err_msg = format!("{e}");
        assert!(
            err_msg.contains("not configured")
                || err_msg.contains("missing")
                || err_msg.contains("requires")
                || err_msg.contains("none could be built"),
            "Expected error about missing config or feature, got: {err_msg}"
        );
    }

    unsafe {
        env::remove_var("SCCACHE_MULTILEVEL_CHAIN");
    }
}

#[test]
fn test_concurrent_reads() {
    // Test multiple simultaneous reads to different levels
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .build()
        .unwrap();

    let cache_l0 = Arc::new(InMemoryStorage::new());
    let cache_l1 = Arc::new(InMemoryStorage::new());
    let cache_l2 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = Arc::new(
            MultiLevelStorage::new(vec![
                cache_l0.clone() as Arc<dyn Storage>,
                cache_l1.clone() as Arc<dyn Storage>,
                cache_l2.clone() as Arc<dyn Storage>,
            ])
            .await,
        );

        // Populate different keys at different levels
        cache_l0
            .put("key_l0", opendal::Buffer::default())
            .await
            .unwrap();
        cache_l1
            .put("key_l1", opendal::Buffer::default())
            .await
            .unwrap();
        cache_l2
            .put("key_l2", opendal::Buffer::default())
            .await
            .unwrap();

        // Concurrent reads
        let storage1 = Arc::clone(&storage);
        let storage2 = Arc::clone(&storage);
        let storage3 = Arc::clone(&storage);

        let (r1, r2, r3) = tokio::join!(
            async move { storage1.get("key_l0").await },
            async move { storage2.get("key_l1").await },
            async move { storage3.get("key_l2").await },
        );

        // All should hit
        assert!(matches!(r1.unwrap(), Cache::Hit(_)));
        assert!(matches!(r2.unwrap(), Cache::Hit(_)));
        assert!(matches!(r3.unwrap(), Cache::Hit(_)));
    });
}

#[test]
fn test_concurrent_write_and_read() {
    // Test concurrent writes and reads to same key
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(4)
        .build()
        .unwrap();

    let cache_l0 = Arc::new(InMemoryStorage::new());
    let cache_l1 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = Arc::new(
            MultiLevelStorage::new(vec![
                cache_l0.clone() as Arc<dyn Storage>,
                cache_l1.clone() as Arc<dyn Storage>,
            ])
            .await,
        );

        let storage_write = Arc::clone(&storage);
        let storage_read = Arc::clone(&storage);

        // Concurrent write and read
        let write_task = crate::util::spawn_detached(async move {
            storage_write
                .put("concurrent_key", opendal::Buffer::default())
                .await
        });

        let read_task = crate::util::spawn_detached(async move {
            sleep(Duration::from_millis(10)).await;
            storage_read.get("concurrent_key").await
        });

        let (write_result, read_result) = tokio::join!(write_task, read_task);

        // Write should succeed
        write_result.unwrap().unwrap();

        // Read might miss or hit depending on timing (both are valid)
        match read_result.unwrap().unwrap() {
            Cache::Hit(_) | Cache::Miss => {} // Both valid
        }
    });
}

#[test]
fn test_large_data_handling() {
    // Test with large cache entries
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let cache_l0 = Arc::new(InMemoryStorage::new());
    let cache_l1 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            cache_l0.clone() as Arc<dyn Storage>,
            cache_l1.clone() as Arc<dyn Storage>,
        ])
        .await;

        // Create large entry (1MB of data)
        let mut entry = CacheWrite::new();
        let large_data = vec![0xAB; 1024 * 1024]; // 1MB of data
        entry.put_stdout(&large_data).unwrap();
        let entry = opendal::Buffer::from(entry.finish().unwrap());
        cache_l1.put("large_key", entry).await.unwrap();

        // Read through multi-level - should hit at L1
        match storage.get("large_key").await.unwrap() {
            Cache::Hit(_) => {}
            _ => panic!("Should hit at L1"),
        }

        // Wait for backfill
        sleep(Duration::from_millis(200)).await;

        // Verify L0 was backfilled
        match cache_l0.get("large_key").await.unwrap() {
            Cache::Hit(_) => {} // Expected
            _ => panic!("L0 should have backfilled data from L1"),
        }
    });
}

#[test]
fn test_storage_trait_methods() {
    // Test Storage trait methods: check(), location(), current_size(), max_size()
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let cache_l0 = Arc::new(InMemoryStorage::new());
    let cache_l1 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            cache_l0 as Arc<dyn Storage>,
            cache_l1 as Arc<dyn Storage>,
        ])
        .await;

        // Test check() - should return ReadWrite
        match storage.check().await.unwrap() {
            CacheMode::ReadWrite => {} // Expected
            _ => panic!("Expected ReadWrite mode"),
        }

        // Test location() - should return multi-level description
        let location = storage.location().await;
        assert!(
            location.contains("Multi-level"),
            "Location should mention Multi-level: {location}"
        );

        // Test current_size() - should return None or Some
        let _ = storage.current_size().await.unwrap();

        // Test max_size() - should return None or Some
        let _ = storage.max_size().await.unwrap();
    });
}

#[test]
fn test_all_levels_fail_on_put() {
    // Test behavior when all storage levels fail on write
    // In multi-level design, put() succeeds if ANY level succeeds
    // Even if all fail, it should not panic
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // Create ReadOnly storages that will reject writes
    let cache_l0 = Arc::new(ReadOnlyStorage(Arc::new(InMemoryStorage::new())));
    let cache_l1 = Arc::new(ReadOnlyStorage(Arc::new(InMemoryStorage::new())));

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            cache_l0 as Arc<dyn Storage>,
            cache_l1 as Arc<dyn Storage>,
        ])
        .await;

        let entry = opendal::Buffer::new();

        // put() should complete without panic even when all levels fail
        // (writes to L0 are synchronous, L1+ are async background)
        let result = storage.put("fail_key", entry).await;

        assert!(result.is_ok(), "Put should succeed with read-only levels");
    });
}

#[tokio::test]
async fn test_empty_levels_new() {
    // Edge case: creating MultiLevelStorage with empty vec
    // This is allowed but from_config prevents it
    let storage = MultiLevelStorage::default();

    // Should have zero levels
    assert_eq!(storage.levels.len(), 0);

    // location() should still work
    let location = storage.location().await;
    assert!(location.contains("0"));
}

#[test]
fn test_readonly_level_in_check() {
    // Test that check() properly detects read-only levels
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let tempdir = TempBuilder::new()
        .prefix("sccache_test_ro_")
        .tempdir()
        .unwrap();
    let cache_dir = tempdir.path().join("cache");
    fs::create_dir(&cache_dir).unwrap();

    let disk_cache = DiskCache::new(&cache_dir, 1024 * 1024 * 100, CacheMode::ReadWrite, vec![]);

    // Wrap in ReadOnly
    let ro_cache = Arc::new(ReadOnlyStorage(Arc::new(disk_cache)));

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![ro_cache as Arc<dyn Storage>]).await;

        // check() should detect read-only mode
        match storage.check().await.unwrap() {
            CacheMode::ReadOnly => {} // Expected
            _ => panic!("Should detect read-only mode"),
        }
    });
}

#[test]
fn test_mixed_readonly_chain_is_readwrite_in_check() {
    // A chain that contains a writable level must report ReadWrite even
    // when other levels are read-only: put() skips read-only levels, so a
    // read-only remote behind a writable local disk (the documented
    // "read-only fallback" topology) must not demote the whole cache to
    // read-only. Regression test for #2773.
    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Writable L0, read-only L1
    let rw_l0 = Arc::new(InMemoryStorage::new());
    let ro_l1 = Arc::new(ReadOnlyStorage(Arc::new(InMemoryStorage::new())));
    crate::util::block_on(&runtime, async {
        let storage =
            MultiLevelStorage::new(vec![rw_l0 as Arc<dyn Storage>, ro_l1 as Arc<dyn Storage>])
                .await;
        match storage.check().await.unwrap() {
            CacheMode::ReadWrite => {} // Expected
            _ => panic!("Writable L0 + read-only L1 should be ReadWrite"),
        }
    });

    // Read-only L0, writable L1: still writable (put() skips L0)
    let ro_l0 = Arc::new(ReadOnlyStorage(Arc::new(InMemoryStorage::new())));
    let rw_l1 = Arc::new(InMemoryStorage::new());
    crate::util::block_on(&runtime, async {
        let storage =
            MultiLevelStorage::new(vec![ro_l0 as Arc<dyn Storage>, rw_l1 as Arc<dyn Storage>])
                .await;
        match storage.check().await.unwrap() {
            CacheMode::ReadWrite => {} // Expected
            _ => panic!("Read-only L0 + writable L1 should be ReadWrite"),
        }
    });
}

#[test]
fn test_all_readonly_chain_is_readonly_in_check() {
    // Only a chain in which EVERY level is read-only is itself read-only.
    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let ro_l0 = Arc::new(ReadOnlyStorage(Arc::new(InMemoryStorage::new())));
    let ro_l1 = Arc::new(ReadOnlyStorage(Arc::new(InMemoryStorage::new())));
    crate::util::block_on(&runtime, async {
        let storage =
            MultiLevelStorage::new(vec![ro_l0 as Arc<dyn Storage>, ro_l1 as Arc<dyn Storage>])
                .await;
        match storage.check().await.unwrap() {
            CacheMode::ReadOnly => {} // Expected
            _ => panic!("All read-only levels should be ReadOnly"),
        }
    });
}

#[test]
fn test_sequential_read_order() {
    // Test that reads happen sequentially (L0, L1, L2, ...), not in parallel
    // This verifies the documented behavior: "check multiple storage backends in sequence"
    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Create three storage levels with access tracking
    let l0 = Arc::new(InMemoryStorage::new());
    let l1 = Arc::new(InMemoryStorage::new());
    let l2 = Arc::new(InMemoryStorage::new());

    let l0_log = l0.get_access_log();
    let l1_log = l1.get_access_log();
    let l2_log = l2.get_access_log();

    // Put data only in L2 (slowest level)
    let key = "test_key_12345678901234567890";
    crate::util::block_on(&runtime, async {
        let mut entry = CacheWrite::default();
        entry.put_stdout(b"test data").unwrap();
        l2.put(key, opendal::Buffer::from(entry.finish().unwrap()))
            .await
            .unwrap();
    });

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            l0 as Arc<dyn Storage>,
            l1 as Arc<dyn Storage>,
            l2 as Arc<dyn Storage>,
        ])
        .await;

        let result = storage.get(key).await.unwrap();

        assert!(matches!(result, Cache::Hit(_)));

        // Check that all three levels were accessed in order
        let l0_accesses = l0_log.lock().await;
        let l1_accesses = l1_log.lock().await;
        let l2_accesses = l2_log.lock().await;

        // Each level should have been accessed exactly once for get
        assert_eq!(l0_accesses.len(), 1, "L0 should be checked first");
        assert_eq!(l1_accesses.len(), 1, "L1 should be checked second");
        assert_eq!(l2_accesses.len(), 2, "L2: put (setup) + get (check)");

        assert_eq!(l0_accesses[0], format!("get:{}", key));
        assert_eq!(l1_accesses[0], format!("get:{}", key));
        assert_eq!(l2_accesses[0], format!("put:{}", key)); // from setup
        assert_eq!(l2_accesses[1], format!("get:{}", key)); // from sequential check
    });
}

#[test]
fn test_read_stops_at_first_hit_not_parallel() {
    // Test that when L1 has data, L2 is NEVER accessed (proving sequential not parallel)
    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let l0 = Arc::new(InMemoryStorage::new());
    let l1 = Arc::new(InMemoryStorage::new());
    let l2 = Arc::new(InMemoryStorage::new());

    let l0_log = l0.get_access_log();
    let l1_log = l1.get_access_log();
    let l2_log = l2.get_access_log();

    let key = "test_key_early_hit_1234567890ab";

    // Put data in L1
    crate::util::block_on(&runtime, async {
        let mut entry = CacheWrite::default();
        entry.put_stdout(b"L1 data").unwrap();
        l1.put(key, opendal::Buffer::from(entry.finish().unwrap()))
            .await
            .unwrap();
    });

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            l0 as Arc<dyn Storage>,
            l1 as Arc<dyn Storage>,
            l2 as Arc<dyn Storage>,
        ])
        .await;

        let result = storage.get(key).await.unwrap();

        assert!(matches!(result, Cache::Hit(_)));

        // Verify L0 and L1 were accessed, but L2 was NOT
        let l0_accesses = l0_log.lock().await;
        let l1_accesses = l1_log.lock().await;
        let l2_accesses = l2_log.lock().await;

        assert_eq!(l0_accesses.len(), 1, "L0 should be checked first");
        assert_eq!(l1_accesses.len(), 2, "L1: put (setup) + get (check)");
        assert_eq!(
            l2_accesses.len(),
            0,
            "L2 should NOT be checked (sequential read stops at first hit)"
        );
    });
}

/// Storage mock that always fails on write (for testing error handling).
///
/// Unlike ReadOnlyStorage (which is a valid mode), this returns actual errors
/// to simulate real failure scenarios like disk full, network errors, etc.
struct FailingStorage;

#[async_trait]
impl Storage for FailingStorage {
    async fn get(&self, _key: &str) -> Result<Cache<opendal::Buffer>> {
        Ok(Cache::Miss)
    }

    async fn del(&self, _key: &str) -> Result<()> {
        Err(anyhow!("Intentional failure for testing"))
    }

    async fn has(&self, _key: &str) -> bool {
        false
    }

    async fn put(&self, _key: &str, _entry: opendal::Buffer) -> Result<Duration> {
        Err(anyhow!("Intentional failure for testing"))
    }

    async fn size(&self, _key: &str) -> Result<u64> {
        Err(anyhow!("Intentional failure for testing"))
    }

    async fn check(&self) -> Result<CacheMode> {
        Ok(CacheMode::ReadWrite) // It's RW but fails on put
    }

    async fn location(&self) -> String {
        "FailingStorage".to_string()
    }

    fn cache_type_name(&self) -> &'static str {
        "FailingStorage"
    }

    async fn current_size(&self) -> Result<Option<u64>> {
        Ok(None)
    }

    async fn max_size(&self) -> Result<Option<u64>> {
        Ok(None)
    }

    fn enabled(&self) -> bool {
        true
    }

    fn basedirs(&self) -> &[Vec<u8>] {
        &[]
    }
}

#[test]
fn test_put_mode_ignore() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // All levels fail with actual errors
    let cache_l0 = Arc::new(FailingStorage);
    let cache_l1 = Arc::new(FailingStorage);

    let storage = crate::util::block_on(
        &runtime,
        MultiLevelStorage::with_write_error_policy(
            vec![cache_l0 as Arc<dyn Storage>, cache_l1 as Arc<dyn Storage>],
            WriteErrorPolicy::Ignore,
        ),
    );

    crate::util::block_on(&runtime, async {
        let entry = opendal::Buffer::new();
        let result = storage.put("test_key", entry).await;

        assert!(
            result.is_ok(),
            "WriteErrorPolicy::Ignore should never fail, even when all levels error"
        );
    });
}

#[test]
fn test_put_mode_l0_fails_on_error() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // L0 fails with actual error, L1 succeeds
    let cache_l0 = Arc::new(FailingStorage);
    let cache_l1 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::with_write_error_policy(
            vec![cache_l0 as Arc<dyn Storage>, cache_l1 as Arc<dyn Storage>],
            WriteErrorPolicy::L0,
        )
        .await;

        let entry = opendal::Buffer::new();
        let result = storage.put("test_key", entry).await;

        assert!(
            result.is_err(),
            "WriteErrorPolicy::L0 should fail when L0 write fails"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Intentional") || err_msg.contains("put not implemented"),
            "Expected failure message, got: {err_msg}"
        );
    });
}

#[test]
fn test_put_mode_l0_succeeds_if_l0_ok() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // L0 succeeds, L1 fails (shouldn't matter in L0 mode)
    let cache_l0 = Arc::new(InMemoryStorage::new());
    let cache_l1 = Arc::new(FailingStorage);

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::with_write_error_policy(
            vec![cache_l0 as Arc<dyn Storage>, cache_l1 as Arc<dyn Storage>],
            WriteErrorPolicy::L0,
        )
        .await;

        let entry = opendal::Buffer::new();
        let result = storage.put("test_key", entry).await;

        assert!(
            result.is_ok(),
            "WriteErrorPolicy::L0 should succeed when L0 succeeds, even if L1+ fails"
        );
    });
}

#[test]
fn test_put_mode_all_fails_on_any_error() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // L0 succeeds, L1 fails
    let cache_l0 = Arc::new(InMemoryStorage::new());
    let cache_l1 = Arc::new(FailingStorage);

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::with_write_error_policy(
            vec![cache_l0 as Arc<dyn Storage>, cache_l1 as Arc<dyn Storage>],
            WriteErrorPolicy::All,
        )
        .await;

        let entry = opendal::Buffer::new();
        let result = storage.put("test_key", entry).await;

        // Give background L1 task time to complete and report failure
        sleep(Duration::from_millis(100)).await;

        assert!(
            result.is_err(),
            "WriteErrorPolicy::All should fail when any RW level fails"
        );
    });
}

#[test]
fn test_put_mode_all_succeeds_when_all_ok() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // Both levels succeed
    let cache_l0 = Arc::new(InMemoryStorage::new());
    let cache_l1 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::with_write_error_policy(
            vec![
                cache_l0.clone() as Arc<dyn Storage>,
                cache_l1.clone() as Arc<dyn Storage>,
            ],
            WriteErrorPolicy::All,
        )
        .await;

        let entry = opendal::Buffer::new();
        let result = storage.put("test_key", entry).await;

        // Give background tasks time to complete
        sleep(Duration::from_millis(100)).await;

        assert!(
            result.is_ok(),
            "WriteErrorPolicy::All should succeed when all levels succeed"
        );

        // Verify both levels have the data
        assert!(matches!(
            cache_l0.get("test_key").await.unwrap(),
            Cache::Hit(_)
        ));
        assert!(matches!(
            cache_l1.get("test_key").await.unwrap(),
            Cache::Hit(_)
        ));
    });
}

#[test]
fn test_put_mode_all_skips_readonly() {
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    // L0 writable, L1 read-only (should be skipped), L2 writable
    let cache_l0 = Arc::new(InMemoryStorage::new());
    let cache_l1 = Arc::new(ReadOnlyStorage(Arc::new(InMemoryStorage::new())));
    let cache_l2 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::with_write_error_policy(
            vec![
                cache_l0.clone() as Arc<dyn Storage>,
                cache_l1 as Arc<dyn Storage>,
                cache_l2.clone() as Arc<dyn Storage>,
            ],
            WriteErrorPolicy::All,
        )
        .await;

        let entry = opendal::Buffer::new();
        let result = storage.put("test_key", entry).await;

        // Give background tasks time to complete
        sleep(Duration::from_millis(100)).await;

        assert!(
            result.is_ok(),
            "WriteErrorPolicy::All should succeed when read-only levels are skipped"
        );

        // Verify writable levels have the data
        assert!(matches!(
            cache_l0.get("test_key").await.unwrap(),
            Cache::Hit(_)
        ));
        assert!(matches!(
            cache_l2.get("test_key").await.unwrap(),
            Cache::Hit(_)
        ));
    });
}

#[test]
fn test_multilevel_get_reads_from_all_levels() {
    // Verifies that MultiLevelStorage::get iterates levels in order
    // and returns the bytes from the first level that has them.
    //
    // Verifies that MultiLevelStorage::get backfills levels that
    // returned cache misses.

    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let l0 = Arc::new(InMemoryStorage::new()); // empty
    let l1 = Arc::new(InMemoryStorage::new()); // will hold the entry

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            l0.clone() as Arc<dyn Storage>,
            l1.clone() as Arc<dyn Storage>,
        ])
        .await;

        let src = CacheWrite::default().finish().unwrap().into();

        l1.put("key", src).await.unwrap();

        // L0 has nothing — get on L0 directly returns None.
        assert!(l0.get("key").await.unwrap().into_option().is_none());

        // MultiLevelStorage::get should skip L0 and find the entry at L1.
        let raw = storage.get("key").await.unwrap().into_option();
        assert!(raw.is_some(), "expected a hit via MultiLevelStorage::get");

        // The bytes should be parseable as a valid cache entry.
        let bytes = raw.unwrap();
        assert!(
            CacheRead::from(std::io::Cursor::new(bytes.to_vec())).is_ok(),
            "get bytes should be a valid zip archive"
        );

        // Wait a bit for the backfill task to complete
        tokio::time::sleep(Duration::from_secs(1)).await;

        // MultiLevelStorage::get should backfill from L1 to L0
        let raw = l0.get("key").await.unwrap().into_option();
        assert!(raw.is_some(), "expected hit at L0 after hitting at L1");

        // A key that exists in neither level should return None.
        assert!(
            storage
                .get("missing")
                .await
                .unwrap()
                .into_option()
                .is_none()
        );
    });
}

#[test]
fn test_multilevel_put_writes_to_all_levels() {
    // Verifies that MultiLevelStorage::put propagates the raw bytes
    // to every level, so a subsequent get on any individual level
    // returns the same bytes.
    let runtime = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let l0 = Arc::new(InMemoryStorage::new());
    let l1 = Arc::new(InMemoryStorage::new());

    crate::util::block_on(&runtime, async {
        let storage = MultiLevelStorage::new(vec![
            l0.clone() as Arc<dyn Storage>,
            l1.clone() as Arc<dyn Storage>,
        ])
        .await;

        // Build raw bytes for a valid (empty) cache entry.
        let src = CacheWrite::default().finish().unwrap();
        let src: opendal::Buffer = src.into();
        let raw_bytes = src.to_vec();

        storage.put("key", src).await.unwrap();

        // Give background writes time to complete.
        sleep(Duration::from_millis(50)).await;

        // Both levels should now hold identical bytes.
        let from_l0 = l0
            .get("key")
            .await
            .unwrap()
            .into_option()
            .map(|b| b.to_vec());
        let from_l1 = l1
            .get("key")
            .await
            .unwrap()
            .into_option()
            .map(|b| b.to_vec());
        assert_eq!(
            from_l0.as_deref(),
            Some(raw_bytes.as_ref()),
            "L0 should have the raw bytes"
        );
        assert_eq!(
            from_l1.as_deref(),
            Some(raw_bytes.as_ref()),
            "L1 should have the raw bytes"
        );
    });
}
