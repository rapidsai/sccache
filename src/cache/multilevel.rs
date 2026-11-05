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

use futures::{FutureExt, StreamExt};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    cache::{Cache, Storage},
    config::{CacheMode, WriteErrorPolicy},
    errors::*,
};

/// Increment an atomic stats counter, handling the Option check.
/// Usage: `inc_stat!(optional_stats, field_name, value)`
macro_rules! inc_stat {
    ($stats:expr, $field:ident, $value:expr) => {
        if let Some(s) = $stats {
            s.$field.fetch_add($value, Ordering::Relaxed);
        }
    };
}

/// Lock-free atomic counters for multi-level cache statistics.
/// Stored directly in MultiLevelStorage to avoid mutex contention.
struct AtomicLevelStats {
    name: String,
    location: String,
    hits: AtomicU64,
    misses: AtomicU64,
    writes: AtomicU64,
    write_failures: AtomicU64,
    backfills_from: AtomicU64,
    backfills_to: AtomicU64,
    hit_duration_nanos: AtomicU64,
    write_duration_nanos: AtomicU64,
}

impl AtomicLevelStats {
    fn new(name: String, location: String) -> Self {
        Self {
            name,
            location,
            hits: Default::default(),
            misses: Default::default(),
            writes: Default::default(),
            write_failures: Default::default(),
            backfills_from: Default::default(),
            backfills_to: Default::default(),
            hit_duration_nanos: Default::default(),
            write_duration_nanos: Default::default(),
        }
    }

    /// Create atomic stats for a specific cache level with formatted name
    async fn for_level(idx: usize, storage: &Arc<dyn Storage>) -> Self {
        Self::new(
            format!("L{} ({})", idx, storage.cache_type_name()),
            storage.location().await,
        )
    }

    /// Create a Vec of atomic stats from a slice of storage backends
    async fn from_levels(storages: &[Arc<dyn Storage>]) -> Vec<Arc<Self>> {
        futures::stream::iter(storages.iter())
            .enumerate()
            .then(|(idx, storage)| async move { Arc::new(Self::for_level(idx, storage).await) })
            .collect()
            .await
    }

    /// Take a consistent snapshot of all stats
    fn snapshot(&self) -> LevelStats {
        LevelStats {
            name: self.name.clone(),
            location: self.location.clone(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
            write_failures: self.write_failures.load(Ordering::Relaxed),
            backfills_from: self.backfills_from.load(Ordering::Relaxed),
            backfills_to: self.backfills_to.load(Ordering::Relaxed),
            hit_duration: Duration::from_nanos(self.hit_duration_nanos.load(Ordering::Relaxed)),
            write_duration: Duration::from_nanos(self.write_duration_nanos.load(Ordering::Relaxed)),
        }
    }
}

/// Statistics for a single cache level (snapshot for display/serialization).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LevelStats {
    /// Human-readable name of this level (e.g., "L0 (disk)")
    pub name: String,
    /// Detailed location string (e.g., "Local disk: \"/path\"" or "s3, name: bucket, prefix: /p/")
    pub location: String,
    /// Number of cache hits at this level
    pub hits: u64,
    /// Number of cache misses (checked but not found) at this level
    pub misses: u64,
    /// Number of successful writes to this level
    pub writes: u64,
    /// Number of failed writes to this level
    pub write_failures: u64,
    /// Number of times data from this level was backfilled to faster levels
    pub backfills_from: u64,
    /// Number of times data from slower levels was backfilled to this level
    pub backfills_to: u64,
    /// Total time spent reading hits from this level
    pub hit_duration: Duration,
    /// Total time spent writing to this level
    pub write_duration: Duration,
}

/// Per-level statistics for multi-level cache operation.
///
/// Serializes as a flat JSON array of level stats (no wrapper object).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MultiLevelStats(pub Vec<LevelStats>);

impl Serialize for MultiLevelStats {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MultiLevelStats {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        Vec::<LevelStats>::deserialize(deserializer).map(MultiLevelStats)
    }
}

impl std::ops::AddAssign for LevelStats {
    fn add_assign(&mut self, rhs: Self) {
        // name and location identify the level — keep lhs values
        self.hits += rhs.hits;
        self.misses += rhs.misses;
        self.writes += rhs.writes;
        self.write_failures += rhs.write_failures;
        self.backfills_from += rhs.backfills_from;
        self.backfills_to += rhs.backfills_to;
        self.hit_duration += rhs.hit_duration;
        self.write_duration += rhs.write_duration;
    }
}

impl std::ops::AddAssign for MultiLevelStats {
    fn add_assign(&mut self, rhs: Self) {
        let mut rhs_iter = rhs.0.into_iter();
        for lhs_level in &mut self.0 {
            if let Some(rhs_level) = rhs_iter.next() {
                *lhs_level += rhs_level;
            }
        }
        // Append any extra levels present only in rhs
        self.0.extend(rhs_iter);
    }
}

impl LevelStats {
    /// Calculate hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            (self.hits as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate average hit latency in milliseconds
    pub fn avg_hit_latency_ms(&self) -> f64 {
        if self.hits > 0 {
            self.hit_duration.as_secs_f64() * 1000.0 / self.hits as f64
        } else {
            0.0
        }
    }

    /// Calculate average write latency in milliseconds
    pub fn avg_write_latency_ms(&self) -> f64 {
        if self.writes > 0 {
            self.write_duration.as_secs_f64() * 1000.0 / self.writes as f64
        } else {
            0.0
        }
    }

    /// Format stats for human-readable display
    /// Returns a vector of (label, value_with_suffix, suffix_length) tuples
    /// suffix_length is used for width calculations in formatting
    /// Order: hits, misses, rate, writes, failures, backfills, write timing, read timing
    pub fn format_stats(&self) -> Vec<(String, String, usize)> {
        let mut stats = vec![];

        // 1. Hits/Misses/Rate
        stats.push((format!("  {} hits", self.name), self.hits.to_string(), 0));
        stats.push((
            format!("  {} misses", self.name),
            self.misses.to_string(),
            0,
        ));

        let total_checks = self.hits + self.misses;
        if total_checks > 0 {
            stats.push((
                format!("  {} hit rate", self.name),
                format!("{:.2} %", self.hit_rate()),
                2, // " %" is 2 chars
            ));
        } else {
            stats.push((format!("  {} hit rate", self.name), "-".to_string(), 0));
        }

        // 2. Writes and failures
        stats.push((
            format!("  {} writes", self.name),
            self.writes.to_string(),
            0,
        ));
        stats.push((
            format!("  {} write failures", self.name),
            self.write_failures.to_string(),
            0,
        ));

        // 3. Backfills
        stats.push((
            format!("  {} backfills from", self.name),
            self.backfills_from.to_string(),
            0,
        ));
        stats.push((
            format!("  {} backfills to", self.name),
            self.backfills_to.to_string(),
            0,
        ));

        // 4. Timing stats
        let avg_write_duration = if self.writes > 0 {
            self.write_duration / self.writes as u32
        } else {
            Duration::default()
        };
        stats.push((
            format!("  {} avg cache write", self.name),
            crate::util::fmt_duration_as_secs(&avg_write_duration),
            2, // " s" is 2 chars
        ));

        let avg_read_duration = if self.hits > 0 {
            self.hit_duration / self.hits as u32
        } else {
            Duration::default()
        };
        stats.push((
            format!("  {} avg cache read hit", self.name),
            crate::util::fmt_duration_as_secs(&avg_read_duration),
            2, // " s" is 2 chars
        ));

        stats
    }
}

impl MultiLevelStats {
    /// Format all stats for human-readable display.
    /// Returns a vector of (label, value, suffix_type) tuples.
    pub fn format_stats(&self) -> Vec<(String, String, usize)> {
        let mut result = vec![];

        if self.0.is_empty() {
            return result;
        }

        // Global stats
        result.push((
            "Multi-level cache levels".to_string(),
            self.0.len().to_string(),
            0,
        ));

        // Per-level stats
        for level_stats in &self.0 {
            result.extend(level_stats.format_stats());
        }

        result
    }
}

/// A multi-level cache storage that checks multiple storage backends in order.
///
/// This enables hierarchical caching similar to CPU L1/L2/L3 caches:
/// - Fast, small caches (e.g., disk) are checked first (L0)
/// - Slower, larger caches (e.g., S3) are checked on miss
/// - Cache hits trigger automatic async backfill to faster levels
/// - Writes go to all levels in parallel
///
/// Configure via SCCACHE_MULTILEVEL_CHAIN="disk,redis,s3" environment variable.
/// See docs/MultiLevel.md for details.
#[derive(Default)]
pub struct MultiLevelStorage {
    levels: Vec<Arc<dyn Storage>>,
    write_error_policy: WriteErrorPolicy,
    /// Lock-free atomic statistics per level
    atomic_stats: Vec<Arc<AtomicLevelStats>>,
    /// Base directories for path normalization, propagated to compiler pipeline
    basedirs: Vec<Vec<u8>>,
}

impl MultiLevelStorage {
    /// Collect and deduplicate basedirs from all cache levels.
    fn collect_basedirs(levels: &[Arc<dyn Storage>]) -> Vec<Vec<u8>> {
        let mut seen = Vec::new();
        for level in levels {
            for basedir in level.basedirs() {
                if !seen.contains(basedir) {
                    seen.push(basedir.clone());
                }
            }
        }
        seen
    }

    /// Create a new multi-level storage from a list of storage backends.
    ///
    /// Levels are checked in order (L0, L1, L2, ...) during reads.
    /// All levels receive writes in parallel.
    pub async fn new(levels: Vec<Arc<dyn Storage>>) -> Self {
        Self::with_write_error_policy(levels, WriteErrorPolicy::default()).await
    }

    /// Create a new multi-level storage with explicit write error policy.
    pub async fn with_write_error_policy(
        levels: Vec<Arc<dyn Storage>>,
        write_error_policy: WriteErrorPolicy,
    ) -> Self {
        let atomic_stats = AtomicLevelStats::from_levels(&levels).await;
        let basedirs = Self::collect_basedirs(&levels);

        MultiLevelStorage {
            levels,
            write_error_policy,
            atomic_stats,
            basedirs,
        }
    }

    /// Get a snapshot of current multi-level cache statistics.
    pub fn stats(&self) -> MultiLevelStats {
        MultiLevelStats(self.atomic_stats.iter().map(|s| s.snapshot()).collect())
    }

    /// Helper to write cache entry from raw bytes.
    ///
    /// Used during backfill operations to efficiently copy data between levels.
    async fn write_entry_from_bytes(
        level: &Arc<dyn Storage>,
        key: &str,
        data: opendal::Buffer,
    ) -> Result<()> {
        // Bytes::clone() is a cheap ref-count bump, no data copy
        level.put(key, data).await?;
        Ok(())
    }

    /// Write to levels starting from `start_idx` asynchronously
    async fn write_remaining_levels_async(
        &self,
        key: &str,
        data: opendal::Buffer,
        start_idx: usize,
    ) {
        for (idx, level) in self.levels.iter().enumerate().skip(start_idx) {
            // Check if level is read-only before spawning task
            if matches!(level.check().await, Ok(CacheMode::ReadOnly)) {
                debug!("Level {idx} is read-only, skipping write");
                continue;
            }

            let data = data.clone();
            let key = key.to_string();
            let level = Arc::clone(level);
            let stats_arc = self.atomic_stats.get(idx).map(Arc::clone);

            crate::util::spawn_detached(async move {
                let start = Instant::now();
                match Self::write_entry_from_bytes(&level, &key, data).await {
                    Ok(_) => {
                        let duration = start.elapsed();
                        trace!("Backfilled cache level {idx} on write in {duration:?}");
                        inc_stat!(stats_arc.as_deref(), writes, 1);
                        inc_stat!(
                            stats_arc.as_deref(),
                            write_duration_nanos,
                            duration.as_nanos() as u64
                        );
                    }
                    Err(e) => {
                        debug!("Background write to level {idx} failed: {e:?}");
                        inc_stat!(stats_arc.as_deref(), write_failures, 1);
                    }
                }
            });
        }
    }
}

#[async_trait]
impl Storage for MultiLevelStorage {
    async fn get(&self, key: &str) -> Result<Cache<opendal::Buffer>> {
        for (idx, level) in self.levels.iter().enumerate() {
            let start = Instant::now();
            match level.get(key).await {
                Ok(Cache::Hit(entry)) => {
                    let duration = start.elapsed();
                    debug!("Cache hit at level {idx} in {duration:?}");

                    // Update stats
                    inc_stat!(self.atomic_stats.get(idx), hits, 1);
                    inc_stat!(
                        self.atomic_stats.get(idx),
                        hit_duration_nanos,
                        duration.as_nanos() as u64
                    );
                    // Mark misses for all levels checked before this hit
                    for miss_idx in 0..idx {
                        inc_stat!(self.atomic_stats.get(miss_idx), misses, 1);
                    }

                    // If hit at level > 0, backfill to faster levels (L0 to L(idx-1))
                    if idx > 0 {
                        let key_str = key.to_string();
                        let hit_level = idx;

                        // Update backfill stats
                        inc_stat!(self.atomic_stats.get(hit_level), backfills_from, idx as u64);

                        // Spawn background backfill tasks for each faster level
                        // Iterate slice directly instead of creating Vec
                        for backfill_idx in 0..idx {
                            let key_bf = key_str.clone();
                            let bytes_bf = entry.clone();
                            let level_bf = Arc::clone(&self.levels[backfill_idx]);
                            let stats_arc = self.atomic_stats.get(backfill_idx).map(Arc::clone);

                            crate::util::spawn_detached(async move {
                                match Self::write_entry_from_bytes(&level_bf, &key_bf, bytes_bf)
                                    .await
                                {
                                    Ok(_) => {
                                        trace!(
                                            "Backfilled cache level {backfill_idx} from level {hit_level}"
                                        );
                                        // Update backfill_to stats
                                        inc_stat!(stats_arc.as_deref(), backfills_to, 1);
                                    }
                                    Err(e) => {
                                        debug!(
                                            "Background backfill from level {hit_level} to level {backfill_idx} failed: {e}"
                                        );
                                    }
                                }
                            });
                        }
                    }

                    return Ok(Cache::Hit(entry));
                }
                Ok(Cache::Miss) => {
                    trace!("Cache miss at level {idx}, trying next level");
                    continue;
                }
                Err(e) => {
                    warn!("Error checking cache level {idx}: {e}, trying next level");
                    continue;
                }
            }
        }
        debug!("Cache miss at all levels");

        // Mark final miss for all checked levels
        for idx in 0..self.levels.len() {
            inc_stat!(self.atomic_stats.get(idx), misses, 1);
        }

        Ok(Cache::Miss)
    }

    async fn del(&self, key: &str) -> Result<()> {
        let futures = self.levels.iter().map(|s| s.del(key).boxed());
        crate::util::try_join_all(futures).await?;
        Ok(())
    }

    async fn has(&self, key: &str) -> bool {
        futures::stream::iter(self.levels.iter())
            .any(|s| s.has(key))
            .await
    }

    async fn put(&self, key: &str, entry: opendal::Buffer) -> Result<Duration> {
        if self.levels.is_empty() {
            return Err(anyhow!("No cache levels configured"));
        }

        let key_str = key.to_string();

        match self.write_error_policy {
            WriteErrorPolicy::Ignore => {
                // Never fail, log warnings only
                self.write_remaining_levels_async(&key_str, entry, 0).await;
                Ok(Duration::ZERO)
            }

            WriteErrorPolicy::L0 => {
                // Fail only if L0 write fails (unless L0 is read-only)
                if let Some(l0) = self.levels.first() {
                    // Check if L0 is read-only before attempting write
                    if matches!(l0.check().await, Ok(CacheMode::ReadOnly)) {
                        debug!("Level 0 is read-only, skipping L0 write");
                    } else {
                        // Attempt write and propagate errors
                        let start = Instant::now();
                        match Self::write_entry_from_bytes(l0, &key_str, entry.clone()).await {
                            Ok(_) => {
                                let duration = start.elapsed();
                                trace!("Stored in cache level 0 in {duration:?}");
                                inc_stat!(self.atomic_stats.first(), writes, 1);
                                inc_stat!(
                                    self.atomic_stats.first(),
                                    write_duration_nanos,
                                    duration.as_nanos() as u64
                                );
                            }
                            Err(e) => {
                                inc_stat!(self.atomic_stats.first(), write_failures, 1);
                                return Err(e);
                            }
                        }
                    }

                    // Background writes for L1+ (best-effort)
                    self.write_remaining_levels_async(&key_str, entry, 1).await;
                }
                Ok(Duration::ZERO)
            }

            WriteErrorPolicy::All => {
                // Fail if any RW level fails
                use tokio::sync::mpsc;
                let (tx, mut rx) = mpsc::channel(self.levels.len());

                for (idx, level) in self.levels.iter().enumerate() {
                    let data = entry.clone();
                    let key_str = key_str.clone();
                    let level = Arc::clone(level);
                    let tx = tx.clone();
                    let stats_arc = self.atomic_stats.get(idx).map(Arc::clone);

                    let write_task = async move {
                        let start = Instant::now();
                        let result = Self::write_entry_from_bytes(&level, &key_str, data).await;
                        let duration = start.elapsed();
                        (idx, result, level, duration, stats_arc)
                    };

                    if idx == 0 {
                        // L0 synchronous
                        let (idx, result, level, duration, stats_arc) = write_task.await;
                        if let Err(e) = result {
                            // Check if read-only before failing
                            if !matches!(level.check().await, Ok(CacheMode::ReadOnly)) {
                                inc_stat!(stats_arc.as_deref(), write_failures, 1);
                                return Err(anyhow!("Failed to write to cache level {idx}: {e}"));
                            }
                        } else {
                            inc_stat!(stats_arc.as_deref(), writes, 1);
                            inc_stat!(
                                stats_arc.as_deref(),
                                write_duration_nanos,
                                duration.as_nanos() as u64
                            );
                        }
                    } else {
                        // L1+ async
                        crate::util::spawn_detached(async move {
                            let result = write_task.await;
                            let _ = tx.send(result).await;
                        });
                    }
                }
                drop(tx);

                // Check async results
                while let Some((idx, result, level, duration, stats_arc)) = rx.recv().await {
                    if let Err(e) = result {
                        // Check if read-only before failing
                        if !matches!(level.check().await, Ok(CacheMode::ReadOnly)) {
                            inc_stat!(stats_arc.as_deref(), write_failures, 1);
                            return Err(anyhow!("Failed to write to cache level {idx}: {e}"));
                        }
                    } else {
                        inc_stat!(stats_arc.as_deref(), writes, 1);
                        inc_stat!(
                            stats_arc.as_deref(),
                            write_duration_nanos,
                            duration.as_nanos() as u64
                        );
                    }
                }

                Ok(Duration::ZERO)
            }
        }
    }

    async fn size(&self, key: &str) -> Result<u64> {
        let mut err = None;
        for storage in self.levels.iter() {
            match storage.size(key).await {
                Ok(size) => return Ok(size),
                Err(e) => {
                    err.replace(e);
                }
            }
        }
        if let Some(err) = err {
            return Err(err);
        }
        Err(anyhow!("Unknown key {key:?}"))
    }

    async fn check(&self) -> Result<CacheMode> {
        // The composite is writable when any level is writable: put()
        // already skips read-only levels on writes, so a read-only level
        // must not demote the whole chain (e.g. a writable local disk in
        // front of a read-only shared remote). Only a chain in which
        // every level is read-only is itself read-only.
        let mut result = CacheMode::ReadOnly;
        if self.levels.is_empty() {
            return Ok(CacheMode::ReadWrite);
        }
        for (idx, level) in self.levels.iter().enumerate() {
            match level.check().await {
                Ok(CacheMode::ReadOnly) => {
                    debug!("Cache level {idx} is read-only");
                }
                Ok(CacheMode::ReadWrite) => {
                    result = CacheMode::ReadWrite;
                    trace!("Cache level {idx} is read-write");
                }
                Err(e) => {
                    warn!("Error checking cache level {idx}: {e}");
                    return Err(e);
                }
            }
        }
        debug!("Multi-level cache mode: {:?}", result);
        Ok(result)
    }

    async fn location(&self) -> String {
        format!("Multi-level ({} levels)", self.levels.len())
    }

    /// Get the cache backend type name (e.g., "disk", "redis", "s3").
    /// Used for statistics and display purposes.
    fn cache_type_name(&self) -> &'static str {
        "multi-level"
    }

    async fn current_size(&self) -> Result<Option<u64>> {
        let mut total = 0u64;
        for level in &self.levels {
            if let Some(size) = level.current_size().await? {
                total += size;
            }
        }
        if total > 0 { Ok(Some(total)) } else { Ok(None) }
    }

    async fn max_size(&self) -> Result<Option<u64>> {
        let mut total = 0u64;
        for level in &self.levels {
            if let Some(size) = level.max_size().await? {
                total += size;
            }
        }
        if total > 0 { Ok(Some(total)) } else { Ok(None) }
    }

    fn multilevel_stats(&self) -> Option<crate::cache::multilevel::MultiLevelStats> {
        Some(self.stats())
    }

    fn enabled(&self) -> bool {
        self.levels.iter().any(|level| level.enabled())
    }

    fn basedirs(&self) -> &[Vec<u8>] {
        &self.basedirs
    }
}

#[cfg(test)]
#[path = "multilevel_test.rs"]
mod test;
