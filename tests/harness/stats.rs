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

use super::client::SccacheClient;
use sccache::{
    compiler::{CCompilerKind, CompilerKind, Language},
    server::ServerStats,
};
use std::collections::HashMap;

/// What some compiles add to a server's stats. Unset fields add nothing.
#[derive(Clone, Debug, Default)]
pub struct AdditionalStats {
    pub cache_writes: Option<u64>,
    pub hashed: Option<u64>,
    pub preprocessed: Option<u64>,
    pub compilations: Option<u64>,
    pub compile_requests: Option<u64>,
    pub non_cacheable_compilations: Option<u64>,
    pub requests_executed: Option<u64>,
    pub cache_hits: Option<Vec<(CCompilerKind, Language, u64)>>,
    pub cache_misses: Option<Vec<(CCompilerKind, Language, u64)>>,
    pub dist_compiles: Option<u64>,
    /// How many of the distributed compilations could reuse the files listed
    /// in their dist dependencies cache entry, see `reusing_dist_deps`.
    pub dist_deps_hits: Option<u64>,
}

impl AdditionalStats {
    /// A compile whose dependencies cache entry leads to a cached object.
    pub fn hit(kind: CCompilerKind, lang: Language) -> Self {
        Self {
            hashed: Some(1),
            preprocessed: Some(0),
            compile_requests: Some(1),
            requests_executed: Some(1),
            cache_hits: Some(vec![(kind, lang, 1)]),
            ..Default::default()
        }
    }

    /// A compile that misses the dependencies cache, has to preprocess, and
    /// then misses the object cache too.
    pub fn miss(kind: CCompilerKind, lang: Language) -> Self {
        Self {
            hashed: Some(2),
            preprocessed: Some(1),
            compilations: Some(1),
            cache_writes: Some(1),
            compile_requests: Some(1),
            requests_executed: Some(1),
            cache_misses: Some(vec![(kind, lang, 1)]),
            ..Default::default()
        }
    }

    /// A compile whose output can't be cached, e.g. because it expands
    /// `__TIME__`.
    pub fn not_cacheable() -> Self {
        Self {
            hashed: Some(2),
            preprocessed: Some(1),
            compile_requests: Some(1),
            requests_executed: Some(1),
            non_cacheable_compilations: Some(1),
            ..Default::default()
        }
    }

    /// The dist dependencies cache entry only has to list files that still
    /// exist, unlike the local one, which needs them unchanged. So a compile
    /// that misses locally can still reuse it without preprocessing again.
    pub fn reusing_dist_deps(self) -> Self {
        Self {
            dist_deps_hits: Some(self.sent_to_build_server()),
            ..self
        }
    }

    fn sent_to_build_server(&self) -> u64 {
        self.compilations.unwrap_or(0) + self.non_cacheable_compilations.unwrap_or(0)
    }

    /// What these compiles add when the client sends them to a build server.
    /// Each one also hashes its dist dependencies cache key, and runs the
    /// preprocessor again unless that entry can be reused.
    ///
    /// This assumes each compilation is one build server job, as it is for
    /// gcc, clang and nvc. It isn't for nvcc: its own compilation runs
    /// locally, and only the compilations for each of its steps are sent.
    pub fn distributed(self) -> Self {
        let sent = self.sent_to_build_server();
        let preprocessed = sent - self.dist_deps_hits.unwrap_or(0);
        Self {
            hashed: Some(self.hashed.unwrap_or(0) + sent),
            preprocessed: Some(self.preprocessed.unwrap_or(0) + preprocessed),
            dist_compiles: Some(self.dist_compiles.unwrap_or(0) + sent),
            ..self
        }
    }
}

impl std::ops::AddAssign<AdditionalStats> for ServerStats {
    fn add_assign(&mut self, rhs: AdditionalStats) {
        self.cache_writes += rhs.cache_writes.unwrap_or(0);
        self.hashed += rhs.hashed.unwrap_or(0);
        self.preprocessed += rhs.preprocessed.unwrap_or(0);
        self.compilations += rhs.compilations.unwrap_or(0);
        self.compile_requests += rhs.compile_requests.unwrap_or(0);
        self.requests_executed += rhs.requests_executed.unwrap_or(0);
        self.non_cacheable_compilations += rhs.non_cacheable_compilations.unwrap_or(0);

        for (kind, lang, count) in rhs.cache_hits.unwrap_or_default() {
            let kind = CompilerKind::C(kind);
            for _ in 0..count {
                self.cache_hits.increment(&kind, &lang);
            }
        }

        for (kind, lang, count) in rhs.cache_misses.unwrap_or_default() {
            let kind = CompilerKind::C(kind);
            for _ in 0..count {
                self.cache_misses.increment(&kind, &lang);
            }
        }

        if let Some(count) = rhs.dist_compiles.filter(|count| *count > 0) {
            *self
                .dist_compiles
                .entry(ANY_BUILD_SERVER.into())
                .or_default() += count as usize;
        }
    }
}

const ANY_BUILD_SERVER: &str = "*";

/// The parts of `stats` that tests can predict: durations vary between runs,
/// and which build server ran a compilation doesn't matter.
fn normalize(mut stats: ServerStats) -> ServerStats {
    stats.cache_write_duration = Default::default();
    stats.cache_read_hit_duration = Default::default();
    stats.hasher_duration = Default::default();
    stats.preprocessor_duration = Default::default();
    stats.compiler_write_duration = Default::default();

    let dist_compiles = stats.dist_compiles.values().sum::<usize>();
    stats.dist_compiles = HashMap::new();
    if dist_compiles > 0 {
        stats
            .dist_compiles
            .insert(ANY_BUILD_SERVER.into(), dist_compiles);
    }

    stats
}

fn assert_stats_eq(expected_stats: &ServerStats, actual_stats: &ServerStats) {
    if actual_stats != expected_stats {
        panic!(
            "assertion `left == right` failed.\n  expected: {}\n    actual: {}",
            serde_json::to_string(&serde_patch::diff(actual_stats, expected_stats).unwrap())
                .unwrap(),
            serde_json::to_string(&serde_patch::diff(expected_stats, actual_stats).unwrap())
                .unwrap()
        )
    }
}

/// The stats a server should report, updated as a test compiles things.
///
/// For a dist client, every compilation is expected to be distributed (see
/// `AdditionalStats::distributed`), so the same test can check either kind of
/// client.
pub struct ExpectedStats<'a> {
    client: &'a SccacheClient,
    dist: bool,
    stats: ServerStats,
}

impl<'a> ExpectedStats<'a> {
    /// Starts from the server's current stats.
    pub fn new(client: &'a SccacheClient) -> Self {
        Self {
            client,
            dist: client.is_dist(),
            stats: normalize(client.stats().unwrap()),
        }
    }

    /// Adds `stats` to the expected stats, and checks the server agrees.
    pub fn assert_added(&mut self, stats: AdditionalStats) {
        self.stats += if self.dist {
            stats.distributed()
        } else {
            stats
        };
        assert_stats_eq(&self.stats, &normalize(self.client.stats().unwrap()));
    }
}
