// Copyright 2016 Mozilla Foundation
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    cache::{Cache, Storage},
    errors::*,
    lru_disk_cache::{LruCache, lru_cache},
    util::{Digest, MetadataCtimeExt, Timestamp},
};
use bytes::{Buf, BufMut, BytesMut};
use fs_err as fs;
use memchr::{memchr_iter, memchr2, memchr3, memmem};
use serde::{Deserialize, Serialize, de, ser::SerializeMap};
use serde_with::{
    DeserializeFromStr, DisplayFromStr, IfIsHumanReadable, SerializeDisplay, serde_as,
};
use std::{
    collections::{BTreeMap, HashMap},
    ffi::{OsStr, OsString},
    fmt,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct SourceFileEntry {
    pub compiled_at: chrono::DateTime<chrono::Utc>,
    pub meta: SourceFileMeta,
    pub found_time: bool,
    pub found_date: bool,
    pub found_timestamp: bool,
    pub hidden_includes: HiddenIncludes,
}

#[derive(Clone, Default, DeserializeFromStr, SerializeDisplay, PartialEq, Eq)]
pub struct OsStringDisplay(OsString);

impl std::ops::Deref for OsStringDisplay {
    type Target = OsStr;

    fn deref(&self) -> &Self::Target {
        self.0.as_os_str()
    }
}

impl AsRef<OsStr> for OsStringDisplay {
    fn as_ref(&self) -> &OsStr {
        self.0.as_os_str()
    }
}

impl fmt::Debug for OsStringDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl fmt::Display for OsStringDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display())
    }
}

impl std::str::FromStr for OsStringDisplay {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self(OsString::from(s)))
    }
}

impl<'a> From<&'a OsStr> for OsStringDisplay {
    fn from(value: &'a OsStr) -> Self {
        OsStringDisplay(value.to_owned())
    }
}

impl<'a> From<&'a Path> for OsStringDisplay {
    fn from(value: &'a Path) -> Self {
        OsStringDisplay(value.as_os_str().to_owned())
    }
}

impl From<OsString> for OsStringDisplay {
    fn from(value: OsString) -> Self {
        OsStringDisplay(value)
    }
}

impl From<PathBuf> for OsStringDisplay {
    fn from(value: PathBuf) -> Self {
        OsStringDisplay(value.into_os_string())
    }
}

impl From<OsStringDisplay> for OsString {
    fn from(value: OsStringDisplay) -> Self {
        value.0
    }
}

/// Corresponds to a cached include file used in the pre-processor stage
#[serde_as]
#[derive(Clone, Deserialize, Serialize, Debug, Default, PartialEq, Eq)]
pub struct SourceFileMeta {
    /// The file's absolute path.
    #[serde_as(as = "IfIsHumanReadable<DisplayFromStr>")]
    pub path: OsStringDisplay,
    /// The hash of the file's contents.
    pub digest: String,
    /// The file's size (in bytes).
    pub fsize: u64,
    /// The file's modification time, `None` if not recorded.
    pub mtime: Option<Timestamp>,
    /// The file's status change time, `None` if not recorded.
    pub ctime: Option<Timestamp>,
}

#[derive(Default)]
pub struct SourceFileAnalysis {
    pub digest: String,
    pub has_time: bool,
    pub has_date: bool,
    pub has_timestamp: bool,
    pub hidden_includes: HiddenIncludes,
}

pub trait SourceFileAnalyzer: fmt::Debug {
    fn analyze(
        &self,
        path: &Path,
        compile_timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<SourceFileAnalysis>;
}

#[derive(Debug)]
pub struct CSourceFileAnalyzer;

impl SourceFileAnalyzer for CSourceFileAnalyzer {
    fn analyze(
        &self,
        path: &Path,
        compile_timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<SourceFileAnalysis> {
        let digest_and_finder = Digest::from_file_with_macro_finders_sync(path, compile_timestamp)
            .with_context(|| format!("while reading '{}'", path.display()))
            .map(|(digest, time_finder, hinc_finder)| (digest.finish(), time_finder, hinc_finder));

        let (digest, time_finder, hinc_finder) = match digest_and_finder {
            Ok(val) => val,
            Err(err) => {
                bail!("Error computing digest: {err:#}");
            }
        };

        Ok(SourceFileAnalysis {
            digest,
            has_time: time_finder.found_time(),
            has_date: time_finder.found_date(),
            has_timestamp: time_finder.found_timestamp(),
            hidden_includes: hinc_finder.into(),
        })
    }
}

#[derive(Debug)]
pub struct SourceFileHasher;

impl SourceFileAnalyzer for SourceFileHasher {
    fn analyze(&self, path: &Path, _: chrono::DateTime<chrono::Utc>) -> Result<SourceFileAnalysis> {
        Ok(SourceFileAnalysis {
            digest: Digest::from_file_sync(path)?.finish(),
            ..Default::default()
        })
    }
}

#[derive(Clone, Debug)]
pub struct SourceFileCache {
    analyzer: Arc<dyn SourceFileAnalyzer + Send + Sync>,
    lru: Arc<Mutex<LruCache<PathBuf, Arc<SourceFileEntry>>>>,
}

impl Default for SourceFileCache {
    fn default() -> Self {
        Self::new(50_000, Arc::new(SourceFileHasher))
    }
}

impl SourceFileCache {
    pub fn new(capacity: u64, analyzer: Arc<dyn SourceFileAnalyzer + Send + Sync>) -> Self {
        SourceFileCache {
            lru: Arc::new(Mutex::new(LruCache::new(capacity))),
            analyzer,
        }
    }

    pub fn get(
        &self,
        path: &Path,
        compile_timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<Arc<SourceFileEntry>> {
        use chrono::Datelike;

        // Assumes path has been canonicalized
        let meta = match fs::symlink_metadata(path) {
            Ok(meta) => meta,
            Err(err) => {
                bail!("Error reading metadata for '{}': {err:#}", path.display());
            }
        };

        let mtime = meta.modified().map(Into::into).ok();
        let ctime = meta.ctime_or_creation().ok();
        let fsize = meta.len();

        let entry = self.lru.lock().unwrap().get(path).and_then(|entry| {
            // If the file hasn't changed, reuse the cached entry
            let file_good =
                entry.meta.fsize == fsize && entry.meta.ctime == ctime && entry.meta.mtime == mtime;

            // If __TIMESTAMP__ or __DATE__ were found, compare compile time to now (or SOURCE_DATE_EPOCH)
            let time_good = if entry.found_timestamp {
                // If __TIMESTAMP__ was found, compare timestamp seconds exactly
                entry.compiled_at == compile_timestamp
            } else if entry.found_date {
                // If __DATE__ was found, only compare the parts of the
                // timestamp the compiler embeds (day, month, and year)
                let date_0 = entry.compiled_at.date_naive();
                let date_1 = compile_timestamp.date_naive();
                date_0.day() == date_1.day()
                    && date_0.month() == date_1.month()
                    && date_0.year() == date_1.year()
            } else {
                // If neither were found, use this cached include entry
                true
            };

            if file_good && time_good {
                Some(entry.clone())
            } else {
                None
            }
        });

        let entry = if let Some(entry) = entry {
            entry
        } else {
            let analysis = self
                .analyzer
                .analyze(path, compile_timestamp)
                .inspect_err(|err| {
                    trace!(
                        "[SourceFileCache::get('{}', {compile_timestamp})]: {err:#}",
                        path.display()
                    );
                })?;

            let entry = Arc::new(SourceFileEntry {
                compiled_at: compile_timestamp,
                meta: SourceFileMeta {
                    ctime,
                    digest: analysis.digest,
                    fsize,
                    mtime,
                    path: path.into(),
                },
                found_time: analysis.has_time,
                found_date: analysis.has_date,
                found_timestamp: analysis.has_timestamp,
                hidden_includes: analysis.hidden_includes,
            });

            self.lru
                .lock()
                .unwrap()
                .insert(path.to_owned(), entry.clone());

            entry
        };

        Ok(entry)
    }
}

// The dependencies cache entry is a description of all information needed
// to cache preprocessor dependencies in C-family languages for a given
// input file.
// The current implementation is very much inspired from the "manifest"
// that `ccache` uses for its "direct mode", though the on-disk format is
// different.

/// An 8 byte magic string denoting the version of the [`SourceFileDeps`] struct encoding.
pub const DEPS_VERSION: &[u8] = b"rapids_1";
const MAX_REVISIONS: u64 = 1_000;

#[derive(Clone, Default, Deserialize, Serialize, Debug, PartialEq, Eq)]
pub struct Dependencies {
    /// The list of included dependencies
    pub sources: Vec<SourceFileMeta>,
    /// Dependency files produced the compiler (if any)
    pub outputs: BTreeMap<String, String>,
}

#[derive(Clone)]
pub struct SourceFileDeps {
    pub key: String,
    /// Information about the original source file
    pub meta: SourceFileMeta,
    /// An LRU reverse-map of object hash key -> dependencies + depfile(s).
    ///
    /// The dependencies are the included files reported by the compiler
    /// and the depfiles (if any) that were produced by the preprocessor's
    /// dependency generation routine.
    pub revs: LruCache<String, Dependencies>,
}

impl SourceFileDeps {
    pub fn new(key: String, meta: SourceFileMeta) -> Self {
        Self {
            key,
            meta,
            revs: LruCache::new(MAX_REVISIONS),
        }
    }

    /// Return the dependencies cache entry for a given dependencies key if it exists in storage.
    pub async fn get(storage: &dyn Storage, key: &str) -> Result<Cache<SourceFileDeps>> {
        match storage.get(key).await {
            Err(err) => Err(err),
            Ok(Cache::Miss) => Ok(Cache::Miss),
            Ok(Cache::Hit(deflated)) => crate::util::spawn_blocking(move || {
                Self::try_from(deflated).map_or_else(|_| Cache::Miss, Cache::Hit)
            })
            .await
            .or(Ok(Cache::Miss)),
        }
    }

    /// Write a dependencies cache entry for the given dependencies key to storage,
    /// overwriting if it exists.
    pub async fn put(self, storage: &dyn Storage) -> Result<Duration> {
        let key = self.key.clone();

        let buf = crate::util::spawn_blocking(move || self.into_buffer()).await??;

        storage.put(&key, buf).await
    }

    pub fn from_buffer(deflated: opendal::Buffer) -> Result<Self> {
        let mut inflated = vec![];
        flate2::read::ZlibDecoder::new(deflated.reader()).read_to_end(&mut inflated)?;

        let buffer: bytes::Bytes = inflated.into();
        let mut header = [0u8; DEPS_VERSION.len()];
        let mut reader = buffer.reader().take(DEPS_VERSION.len() as u64);
        if reader.read_exact(&mut header).is_err() || header != DEPS_VERSION {
            bail!("Format version mismatch");
        } else {
            Ok(bincode::deserialize_from(reader.into_inner())?)
        }
    }

    pub fn into_buffer(self) -> Result<opendal::Buffer> {
        let mut writer =
            flate2::write::ZlibEncoder::new(BytesMut::new().writer(), flate2::Compression::best());

        writer.write_all(DEPS_VERSION)?;
        crate::util::bincode_serialize_into(&mut writer, &self)?;

        writer
            .finish()
            .map_err(anyhow::Error::new)
            .map(|writer| writer.into_inner().freeze().into())
    }

    pub fn iter(&self) -> lru_cache::Iter<'_, String, Dependencies> {
        self.revs.iter()
    }

    pub fn iter_mut(&mut self) -> lru_cache::IterMut<'_, String, Dependencies> {
        self.revs.iter_mut()
    }

    /// Insert the full compilation key and included files for a given source file.
    ///
    /// There can be more than one result at once for a source file if one
    /// or more of the include files has changed but not the source file.
    pub fn add_revision(
        &mut self,
        obj_key: &str,
        dependencies: Dependencies,
        out_pretty: &str,
    ) -> &mut Self {
        let num_includes = dependencies.sources.len();

        self.revs.insert(obj_key.to_string(), dependencies);

        let num_results = self.revs.len();

        debug!(
            "[{out_pretty}, {src_key}, {obj_key}]: Added revision to dependencies cache entry: num_includes={num_includes}, num_results={num_results}",
            src_key = self.key
        );

        self
    }

    /// Returns the digest of the first result whose expected included files
    /// are already on disk and have not changed.
    pub fn find_dependencies(
        &mut self,
        source_file_cache: &SourceFileCache,
        compile_timestamp: chrono::DateTime<chrono::Utc>,
        transitive: bool,
        out_pretty: &str,
    ) -> (bool, Option<(String, Dependencies, HiddenIncludes)>) {
        let mut maybe_result = None;
        let mut needs_update = false;
        let mut log_messages = vec![];

        // Find the first result key whose include files on disk match this
        // dependencies cache entry. Check newest results first since they're
        // more likely to match.
        for (idx, (obj_key, deps)) in self.iter_mut().rev().enumerate() {
            let mut hidden_incs = HiddenIncludes::default();
            let (failure, updated) = Self::test_dependencies(
                source_file_cache,
                compile_timestamp,
                &mut deps.sources,
                &mut hidden_incs,
                transitive,
            );

            if let Some(reason) = updated {
                needs_update = true;
                if log_enabled!(log::Level::Trace) {
                    log_messages.push(format!("{obj_key}]: {reason}"));
                }
            }

            if let Some(reason) = failure {
                if log_enabled!(log::Level::Trace) {
                    log_messages.push(format!("{obj_key}]: Deps lookup failure: {reason}"));
                }
            } else {
                if idx != 0 {
                    // Need to write back to storage if the LRU order changes
                    needs_update = true;
                    if log_enabled!(log::Level::Trace) {
                        log_messages.push(format!("{obj_key}]: Deps LRU order changed"));
                    }
                }
                maybe_result = Some((obj_key.clone(), deps.clone(), hidden_incs.dedupe()));
                break;
            }
        }

        for msg in log_messages {
            trace!("[{out_pretty}, {src_key}, {msg}", src_key = self.key);
        }

        if let Some((obj_key, _, _)) = maybe_result.as_ref() {
            // Move the entry to the back of the LRU
            self.revs.get(obj_key);
        }

        (
            // Signal this dependencies cache entry was modified and should
            // be written back to storage.
            // Note: This can happen regardless whether a result was found.
            needs_update,
            // Return the object key and dependencies
            maybe_result,
        )
    }

    /// A result matches if all of its include files exist on disk and have not changed.
    fn test_dependencies(
        source_file_cache: &SourceFileCache,
        compile_timestamp: chrono::DateTime<chrono::Utc>,
        dependencies: &mut [SourceFileMeta],
        hidden_incs: &mut HiddenIncludes,
        transitive: bool,
    ) -> (Option<String>, Option<String>) {
        let mut failure = None;
        let mut updated = None;

        for prev in dependencies.iter_mut() {
            let path = Path::new(&prev.path);

            if transitive {
                let err = match fs::exists(path) {
                    Ok(true) => continue,
                    Ok(false) => std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        anyhow!("{} does not exist", path.display()),
                    ),
                    Err(err) => err,
                };
                failure = Some(format!("{err:#}"));
                break;
            }

            let curr = match source_file_cache.get(path, compile_timestamp) {
                Ok(entry) => entry,
                Err(err) => {
                    failure = Some(format!("{err:#}"));
                    break;
                }
            };

            let SourceFileEntry {
                meta: curr,
                found_time,
                found_date,
                found_timestamp,
                hidden_includes,
                ..
            } = curr.as_ref();

            hidden_incs.extend([hidden_includes]);

            if *found_time {
                // We don't know for sure that the program actually uses the __TIME__ macro,
                // but we have to assume it anyway and hash the time stamp. However, that's
                // not very useful since the chance that we get a cache hit later the same
                // second should be quite slim... So, just signal back to the caller that
                // __TIME__ has been found so that the dependencies cache can be disabled.
                failure = Some(format!("Found __TIME__ in {path:?}"));
                break;
            }

            // If the digests are different, disable dependencies cache
            if prev.digest != curr.digest {
                failure = Some(format!("Digest mismatch for {path:?}"));
                break;
            }

            // If __DATE__ or __TIMESTAMP__ found, update the includes for this result.
            if *found_date {
                // trace!("[{out_pretty}, {src_key}]: updating entry, __DATE__ found in {path:?}");
                updated = Some(format!("updating entry, __DATE__ found in {path:?}"));
                *prev = curr.clone();
            } else if *found_timestamp {
                // trace!("[{out_pretty}, {src_key}]: updating entry, __TIMESTAMP__ found in {path:?}");
                updated = Some(format!("updating entry, __TIMESTAMP__ found in {path:?}"));
                *prev = curr.clone();
            }
        }

        (failure, updated)
    }
}

impl TryFrom<SourceFileDeps> for opendal::Buffer {
    type Error = anyhow::Error;

    fn try_from(src: SourceFileDeps) -> std::result::Result<Self, Self::Error> {
        src.into_buffer()
    }
}

impl TryFrom<opendal::Buffer> for SourceFileDeps {
    type Error = anyhow::Error;

    fn try_from(buf: opendal::Buffer) -> std::result::Result<Self, Self::Error> {
        Self::from_buffer(buf)
    }
}

impl fmt::Debug for SourceFileDeps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}",
            serde_json::to_string_pretty(self).expect("json serialize to succeed")
        )
    }
}

impl fmt::Display for SourceFileDeps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}",
            serde_json::to_string(self).expect("json serialize to succeed")
        )
    }
}

impl<'de> Deserialize<'de> for SourceFileDeps {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct SourceFileDepsVisitor;
        impl<'de> de::Visitor<'de> for SourceFileDepsVisitor {
            type Value = SourceFileDeps;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("expected a depepdency cache entry")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut key = None;
                let mut meta = None;
                let mut revs = None;
                let mut size = None;

                while let Ok(Some(name)) = map.next_key::<String>() {
                    match name.as_str() {
                        "key" => {
                            key = Some(map.next_value::<String>()?);
                        }
                        "revs" => {
                            revs = Some(map.next_value::<Vec<(String, Dependencies)>>()?);
                        }
                        "meta" => {
                            meta = Some(map.next_value::<SourceFileMeta>()?);
                        }
                        "size" => {
                            size = Some(map.next_value::<u64>()?);
                        }
                        name => {
                            return Err(de::Error::unknown_field(
                                name,
                                &["key", "meta", "revs", "size"],
                            ));
                        }
                    }
                }

                let Some(key) = key else {
                    return Err(de::Error::missing_field("key"));
                };

                let Some(meta) = meta else {
                    return Err(de::Error::missing_field("meta"));
                };

                let mut deps = SourceFileDeps {
                    key,
                    meta,
                    revs: LruCache::new(size.unwrap_or(MAX_REVISIONS).max(MAX_REVISIONS)),
                };

                for (key, val) in revs.into_iter().flatten().rev() {
                    deps.revs.insert(key, val);
                }

                Ok(deps)
            }
        }

        deserializer.deserialize_map(SourceFileDepsVisitor)
    }
}

impl Serialize for SourceFileDeps {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut map = serializer.serialize_map(Some(4))?;
        map.serialize_entry("key", &self.key)?;
        map.serialize_entry("meta", &self.meta)?;
        map.serialize_entry("size", &self.revs.capacity())?;
        map.serialize_entry("revs", &self.iter().rev().collect::<Vec<_>>())?;
        map.end()
    }
}

#[cfg(unix)]
static GCC_PRAGMA_ONCE_FIX_LOCK: std::sync::LazyLock<tokio::sync::Semaphore> =
    std::sync::LazyLock::new(|| tokio::sync::Semaphore::new(1));

#[cfg(unix)]
pub async fn build_gcc_pragma_once_fix(
    exe: &Path,
    args: &[&OsStr],
    env_vars: &[(OsString, OsString)],
) -> Result<PathBuf> {
    use crate::{
        jobserver,
        mock_command::{AsyncCommand, RunCommand},
        util::{Digest, SCCACHE_TMPDIR, run_input_output},
    };
    use std::ffi::OsStr;

    let _permit = GCC_PRAGMA_ONCE_FIX_LOCK.acquire().await?;

    let temp = SCCACHE_TMPDIR.as_ref()?;
    let name = format!(
        "pragma_once_fix_{name}_{hash}.so",
        name = exe.file_stem().unwrap_or(OsStr::new("gcc")).display(),
        hash = {
            let mut digest = Digest::from_file(exe).await?;
            digest.update(GCC_PRAGMA_ONCE_FIX_SOURCE.as_bytes());
            digest.finish()
        }
    );
    let path = temp.join(&name);

    let file = tempfile::Builder::new()
        .rand_bytes(0)
        .prefix(&name)
        .suffix(".lock")
        .tempfile_in(temp)?;

    file.as_file().lock()?;

    if !path.exists() || !path.is_file() || path.metadata()?.len() == 0 {
        let mut cmd = AsyncCommand::new(exe, jobserver::Client::new_num(1));

        cmd.env_clear().envs(env_vars.iter().cloned());

        for arg in args {
            cmd.arg(arg);
        }

        cmd.args(&["-x", "c", "-O2", "-shared", "-fPIC"])
            .arg("-o")
            .arg(&path)
            .arg("-")
            .arg("-ldl");

        run_input_output(cmd, Some(GCC_PRAGMA_ONCE_FIX_SOURCE.as_bytes().to_vec())).await?;
    }

    file.as_file().unlock()?;

    Ok(path)
}

#[cfg(unix)]
static GCC_PRAGMA_ONCE_FIX_SOURCE: &str = r#"
/*
 * LD_PRELOAD shim that makes GCC's #pragma once identify a header by its
 * (st_dev, st_ino) instead of by its (st_mtime, st_size, contents).
 *
 * libcpp's has_unique_contents() treats a newly opened header as a file it has
 * already seen when an earlier #pragma once file has the same st_mtime (whole
 * seconds) and st_size and identical bytes.  Two distinct copies of a header
 * with identical bytes and mtimes are therefore merged, and the second copy is
 * silently skipped even if it would include different files.
 *
 * libcpp gets st_mtime from the fstat() in open_file().  This shim interposes
 * fstat() in GCC's compilers proper (cc1, cc1plus, ...) so that no two distinct
 * regular files report the same st_mtime and size: a file keeps its real mtime
 * unless a different file already reported that mtime and size, in which case
 * it gets the next free second.  A file gets the same value every time it is
 * opened, so libcpp only ever compares a file's contents with its own, which
 * still merges one file reached through different paths (symlinks, hard links,
 * "a/../b").
 *
 * Build: cc -O2 -shared -fPIC -o libpragma_once_fix.so pragma_once_fix.c -ldl
 * Use:   LD_PRELOAD=/abs/path/libpragma_once_fix.so gcc ...
 * Set PRAGMA_ONCE_FIX_DEBUG=1 to log each file whose st_mtime was changed: the
 * files stock GCC would have compared byte-by-byte against a different file.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* The pre-2.33 glibc ABI, still called by GCCs built against older glibc. */
int __fxstat(int ver, int fd, struct stat *st);
int __fxstat64(int ver, int fd, struct stat64 *st);

static bool active, debug;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

__attribute__((constructor)) static void init(void)
{
	static const char *const compilers[] = { "cc1", "cc1plus", "cc1obj", "cc1objplus" };

	for (size_t i = 0; i < sizeof(compilers) / sizeof(*compilers); i++)
		if (strcmp(program_invocation_short_name, compilers[i]) == 0)
			active = true;
	debug = getenv("PRAGMA_ONCE_FIX_DEBUG") != NULL;
	if (active && debug)
		fprintf(stderr, "pragma-once-fix: active in %s\n", program_invocation_short_name);
}

/* Open-addressing hash table from a (k1, k2) key to a value. */
struct slot {
	uint64_t k1, k2, val;
	bool used;
};
struct table {
	struct slot *slots;
	size_t cap, len;
};

/* Returns the slot holding (k1, k2), or the empty slot where it belongs. */
static struct slot *find(struct table *t, uint64_t k1, uint64_t k2)
{
	size_t i = ((k1 ^ k2 * 0x9e3779b97f4a7c15u) * 0xbf58476d1ce4e5b9u) >> 32;

	for (;; i++) {
		struct slot *s = &t->slots[i & (t->cap - 1)];
		if (!s->used || (s->k1 == k1 && s->k2 == k2))
			return s;
	}
}

/* Makes room for one more entry, keeping the table at most half full. */
static bool reserve(struct table *t)
{
	if (2 * (t->len + 1) <= t->cap)
		return true;

	struct table bigger = { NULL, t->cap ? 2 * t->cap : 256, t->len };
	if (!(bigger.slots = calloc(bigger.cap, sizeof(*bigger.slots))))
		return false;
	for (size_t i = 0; i < t->cap; i++)
		if (t->slots[i].used)
			*find(&bigger, t->slots[i].k1, t->slots[i].k2) = t->slots[i];
	free(t->slots);
	*t = bigger;
	return true;
}

/* (st_dev, st_ino) -> the st_mtime reported for that file. */
static struct table files;
/* (st_mtime, size) reported for some file -> a later st_mtime to try next. */
static struct table taken;

/* The size libcpp compares for UTF-8 input, which excludes a byte order mark. */
static uint64_t content_size(int fd, uint64_t size)
{
	unsigned char bom[3];

	if (size >= 3 && pread(fd, bom, 3, 0) == 3 && memcmp(bom, "\xef\xbb\xbf", 3) == 0)
		size -= 3;
	return size;
}

static void log_remap(int fd, uint64_t real, uint64_t fake)
{
	char link[32], path[PATH_MAX];

	snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
	ssize_t n = readlink(link, path, sizeof(path) - 1);
	path[n > 0 ? n : 0] = '\0';
	fprintf(stderr, "pragma-once-fix: %s : %lld -> %lld\n",
		path, (long long)real, (long long)fake);
}

/* Replaces *mtime with the second reserved for this file, reserving it if new. */
static void remap(int fd, uint64_t dev, uint64_t ino, mode_t mode, uint64_t size,
		  time_t *mtime)
{
	if (!active || !S_ISREG(mode))
		return;

	pthread_mutex_lock(&lock);
	if (reserve(&files) && reserve(&taken)) {
		struct slot *f = find(&files, dev, ino), *s;
		if (!f->used) {
			/* Skip past seconds other files of this size already have. */
			uint64_t real = (uint64_t)*mtime, t = real;
			size = content_size(fd, size);
			while ((s = find(&taken, t, size))->used)
				t = s->val;
			*s = (struct slot){ t, size, t + 1, true };
			taken.len++;
			/* Let later files with this mtime and size skip straight past t. */
			for (uint64_t u = real; u != t;) {
				s = find(&taken, u, size);
				u = s->val;
				s->val = t + 1;
			}
			*f = (struct slot){ dev, ino, t, true };
			files.len++;
			if (debug)
				log_remap(fd, real, t);
		}
		*mtime = (time_t)f->val;
	}
	pthread_mutex_unlock(&lock);
}

/* Declares `next`, the definition this library shadows, or fails the call. */
#define NEXT(name)                                                            \
	static __typeof__(name) *next;                                        \
	if (!next && !(next = (__typeof__(name) *)dlsym(RTLD_NEXT, #name))) { \
		errno = ENOSYS;                                               \
		return -1;                                                    \
	}

int fstat(int fd, struct stat *st)
{
	NEXT(fstat);
	int ret = next(fd, st);
	if (ret == 0)
		remap(fd, st->st_dev, st->st_ino, st->st_mode, st->st_size, &st->st_mtime);
	return ret;
}

int fstat64(int fd, struct stat64 *st)
{
	NEXT(fstat64);
	int ret = next(fd, st);
	if (ret == 0)
		remap(fd, st->st_dev, st->st_ino, st->st_mode, st->st_size, &st->st_mtime);
	return ret;
}

int __fxstat(int ver, int fd, struct stat *st)
{
	NEXT(__fxstat);
	int ret = next(ver, fd, st);
	if (ret == 0)
		remap(fd, st->st_dev, st->st_ino, st->st_mode, st->st_size, &st->st_mtime);
	return ret;
}

int __fxstat64(int ver, int fd, struct stat64 *st)
{
	NEXT(__fxstat64);
	int ret = next(ver, fd, st);
	if (ret == 0)
		remap(fd, st->st_dev, st->st_ino, st->st_mode, st->st_size, &st->st_mtime);
	return ret;
}
"#;

#[derive(Clone, Debug)]
pub struct HiddenIncludes {
    /// The operands of `__has_include`.
    pub includes: Vec<String>,
    /// The operands of `__has_include_next`.
    pub next_includes: Vec<String>,
    /// The operands of `__has_embed`, including any embed parameters.
    pub embeds: Vec<String>,
}

impl Default for HiddenIncludes {
    fn default() -> Self {
        Self {
            includes: Vec::with_capacity(0),
            next_includes: Vec::with_capacity(0),
            embeds: Vec::with_capacity(0),
        }
    }
}

impl HiddenIncludes {
    pub fn is_empty(&self) -> bool {
        self.includes.is_empty() && self.next_includes.is_empty() && self.embeds.is_empty()
    }

    pub fn dedupe(mut self) -> Self {
        use itertools::Itertools;
        self.includes = self.includes.into_iter().unique().collect();
        self.next_includes = self.next_includes.into_iter().unique().collect();
        self.embeds = self.embeds.into_iter().unique().collect();
        self
    }

    pub fn to_source_file(&self) -> String {
        let mut source = String::new();
        for include in self.includes.iter() {
            source.push_str(&format!(
                "#if __has_include({include})\n#include {include}\n#endif\n"
            ));
        }
        for include_next in self.next_includes.iter() {
            source.push_str(&format!(
                "#if __has_include_next({include_next})\n#include_next {include_next}\n#endif\n"
            ));
        }
        for embed in self.embeds.iter() {
            source.push_str(&format!(
                "#if __has_embed({embed})\n#embed {embed}\n#endif\n"
            ));
        }
        source
    }
}

impl<'a> Extend<&'a HiddenIncludes> for HiddenIncludes {
    fn extend<T: IntoIterator<Item = &'a HiddenIncludes>>(&mut self, iter: T) {
        for incs in iter {
            self.includes.extend_from_slice(&incs.includes[..]);
            self.next_includes
                .extend_from_slice(&incs.next_includes[..]);
            self.embeds.extend_from_slice(&incs.embeds[..]);
        }
    }
}

/// The `__has_include` operands. Those of the other operators are in `next_includes` and
/// `embeds`.
impl From<HiddenIncludesFinder> for HiddenIncludes {
    fn from(finder: HiddenIncludesFinder) -> Self {
        Self {
            includes: finder.includes,
            next_includes: finder.next_includes,
            embeds: finder.embeds,
        }
    }
}

/// Collects the operands of `__has_include(...)`, `__has_include_next(...)`, and
/// `__has_embed(...)` in C, C++, and CUDA sources.
///
/// Only `#if`, `#elif`, and `#define` lines are searched, and comments, literals, and line
/// splices are handled as the preprocessor handles them. Each operand is collected as written
/// (minus line splices) from its first token to its last, e.g. `<cuda_fp16.h>`,
/// `MAKE_HEADER(fp16)`, or `<data.bin> limit(4)`.
///
/// A function-like macro that passes its parameters to an operator, such as
/// `#define HAS_INCLUDE(x) __has_include(x)`, is followed through: each call like
/// `HAS_INCLUDE(<a.h>)` is collected as `<a.h>`, and the definition's own operand isn't.
/// Calls count only after the definition has been scanned, which may be in an earlier file.
#[derive(Debug, Default)]
pub struct HiddenIncludesFinder {
    /// The operands of `__has_include`, in the order they appear.
    pub includes: Vec<String>,
    /// The operands of `__has_include_next`, in the order they appear.
    pub next_includes: Vec<String>,
    /// The operands of `__has_embed`, including any embed parameters, in the order they appear.
    pub embeds: Vec<String>,
    /// A backslash, and any whitespace after it, that splices lines if a newline follows.
    backslash: bool,
    backslash_ws: Vec<u8>,
    lex: Lex,
    line: Line,
    expr: Expr,
    ident: Vec<u8>,
    /// The delimiter of the current raw string literal.
    raw_delim: [u8; 16],
    raw_len: usize,
    /// The operand or call arguments so far, the length through the last token, and how
    /// deeply that is nested in parentheses.
    operand: Vec<u8>,
    operand_end: usize,
    depth: u32,
    /// The wrapper being called, the bounds in `operand` of each argument so far, and the
    /// start of the current argument once it has a token.
    callee: Vec<u8>,
    args: Vec<(usize, usize)>,
    arg_start: Option<usize>,
    /// Whether a function-like macro is being defined on this line, and the definition, whose
    /// buffers are reused from line to line.
    defining: bool,
    definition: Definition,
    /// The function-like macros that pass their parameters to an operator, by name.
    wrappers: HashMap<Vec<u8>, Wrapper>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Lex {
    #[default]
    Code,
    Ident,
    Number,
    /// A `'` in a number: a digit separator, or the start of a character literal.
    NumberQuote,
    /// A `/` that may start a comment.
    Slash,
    /// A `%` at the start of a line that may be the `%:` spelling of `#`.
    Percent,
    LineComment,
    BlockComment,
    BlockCommentStar,
    Literal {
        quote: u8,
        escaped: bool,
    },
    RawDelim,
    /// Inside a raw string, having matched `matched` bytes of its `)delim"` terminator.
    RawBody {
        matched: usize,
    },
    /// A `<...>` or `"..."` operand, in which comments and escapes don't apply.
    HeaderName {
        close: u8,
    },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Line {
    /// Nothing but whitespace and comments so far.
    #[default]
    Start,
    /// After `#`, expecting the directive name.
    Hash,
    /// After `#define`, expecting the macro name.
    DefineName,
    /// In a function-like macro's parameter list, having seen `dots` dots of a `...`, and
    /// with `after_name` set right after a parameter's name.
    Params { dots: u8, after_name: bool },
    /// In an `#if`, `#elif`, or `#define`, where the operators may appear.
    Active,
    /// In any other line.
    Inactive,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Expr {
    #[default]
    Idle,
    /// After an operator's name, expecting `(`.
    Keyword(Operator),
    /// Inside the operator's parentheses.
    Operand(Operator),
    /// After a wrapper's name, expecting `(`.
    Callee,
    /// Inside the wrapper call's parentheses.
    Args,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    Include,
    IncludeNext,
    Embed,
}

#[derive(Debug, Default)]
struct Definition {
    name: Vec<u8>,
    /// The parameters' names, each followed by a comma.
    params: Vec<u8>,
    /// Whether the last parameter is variadic: `__VA_ARGS__`, or GNU C's `name...`.
    variadic: bool,
    /// The operands in the replacement list that use the parameters.
    templates: Vec<Template>,
}

/// An operand that a wrapper builds from its parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Template {
    operator: Operator,
    /// As in `Definition`.
    params: Vec<u8>,
    variadic: bool,
    operand: Vec<u8>,
}

#[derive(Debug)]
struct Wrapper {
    /// Finds the wrapper's name, to tell whether text might call it.
    name: memmem::Finder<'static>,
    templates: Vec<Template>,
}

impl HiddenIncludesFinder {
    pub fn is_empty(&self) -> bool {
        self.includes.is_empty() && self.next_includes.is_empty() && self.embeds.is_empty()
    }

    /// Scans the next part of the current file, which may be split at any byte. Given a whole
    /// file, `find_in_file` is faster.
    pub fn find_has_include(&mut self, haystack: &[u8]) {
        let mut rest = haystack;
        while let Some((&c, tail)) = rest.split_first() {
            self.splice(c);
            rest = &tail[self.skip(tail)..];
        }
    }

    /// Ends the current file, so the next `find_has_include` call starts a new one. Wrappers
    /// defined so far still apply.
    pub fn finish_file(&mut self) {
        // The last line may lack a newline.
        self.end_definition();
        *self = Self {
            includes: std::mem::take(&mut self.includes),
            next_includes: std::mem::take(&mut self.next_includes),
            embeds: std::mem::take(&mut self.embeds),
            wrappers: std::mem::take(&mut self.wrappers),
            ..Self::default()
        };
    }

    /// Scans `file` as a whole file, independently of earlier calls. Unlike
    /// `find_has_include`, this skips a file that can't contain an operator without lexing it.
    pub fn find_in_file(&mut self, file: &[u8]) {
        self.finish_file();
        if self.might_name_operator(file) {
            self.find_has_include(file);
            self.finish_file();
        }
    }

    /// Whether `src` might contain an operator or a call to a wrapper. It can't if their names
    /// don't appear and no line splice follows an identifier character, since such a splice
    /// could split a name.
    fn might_name_operator(&self, src: &[u8]) -> bool {
        self.first_name(src).is_some()
            || memchr_iter(b'\\', src).any(|i| {
                i > 0 && is_ident(src[i - 1]) && {
                    let after = &src[i + 1..];
                    let space = after
                        .iter()
                        .take_while(|&&b| matches!(b, b' ' | b'\t' | b'\r' | b'\x0b' | b'\x0c'))
                        .count();
                    after.get(space) == Some(&b'\n')
                }
            })
    }

    /// The position of the first operator's or wrapper's name in `text`, even if it's part of a
    /// longer identifier.
    fn first_name(&self, text: &[u8]) -> Option<usize> {
        let operator = memmem::find(text, b"__has_");
        if self.wrappers.is_empty() {
            return operator;
        }
        let wrapper = self
            .wrappers
            .values()
            .filter_map(|w| w.name.find(text))
            .min();
        operator.into_iter().chain(wrapper).min()
    }

    /// Returns how many of the next bytes `lex` would pass over without effect, updating the
    /// state as if it had. A backslash is never skipped, since it may splice lines.
    fn skip(&mut self, rest: &[u8]) -> usize {
        if self.backslash || matches!(self.expr, Expr::Operand(_) | Expr::Args) {
            return 0;
        }
        let end = match self.lex {
            Lex::LineComment => memchr2(b'\n', b'\\', rest),
            Lex::BlockComment => memchr2(b'*', b'\\', rest),
            Lex::Literal {
                quote,
                escaped: false,
            } => memchr3(quote, b'\n', b'\\', rest),
            Lex::RawBody { matched: 0 } => memchr2(b')', b'\\', rest),
            Lex::Code if self.line == Line::Start => {
                rest.iter().position(|&b| b != b' ' && b != b'\t')
            }
            // An ordinary line matters only where a comment or literal could start, or where
            // the line ends.
            Lex::Code | Lex::Ident | Lex::Number if self.line == Line::Inactive => {
                let n = before_last_word(&rest[..next_significant(rest)]);
                if n > 0 {
                    self.lex = Lex::Code;
                }
                Some(n)
            }
            // So does a line where operators may appear, apart from their and wrappers' names.
            Lex::Code if self.line == Line::Active && self.expr == Expr::Idle => {
                let run = &rest[..next_significant(rest)];
                let end = self.first_name(run).unwrap_or(run.len());
                Some(before_last_word(&run[..end]))
            }
            // The rest of an identifier or number.
            Lex::Ident => {
                let n = rest
                    .iter()
                    .position(|&b| !is_ident(b))
                    .unwrap_or(rest.len());
                self.ident.extend_from_slice(&rest[..n]);
                Some(n)
            }
            Lex::Number => rest.iter().position(|&b| !is_ident(b) && b != b'.'),
            _ => Some(0),
        };
        end.unwrap_or(rest.len())
    }

    /// Removes line splices (backslash-newline) before lexing.
    fn splice(&mut self, c: u8) {
        if self.backslash {
            match c {
                b'\n' => {
                    self.backslash = false;
                    self.backslash_ws.clear();
                    return;
                }
                // GCC and Clang also splice when whitespace separates the backslash and newline.
                b' ' | b'\t' | b'\r' | b'\x0b' | b'\x0c' => {
                    self.backslash_ws.push(c);
                    return;
                }
                _ => {
                    self.backslash = false;
                    self.lex(b'\\');
                    let ws = std::mem::take(&mut self.backslash_ws);
                    ws.iter().for_each(|&w| self.lex(w));
                    self.backslash_ws = ws;
                    self.backslash_ws.clear();
                }
            }
        }
        if c == b'\\' {
            self.backslash = true;
        } else {
            self.lex(c);
        }
    }

    fn lex(&mut self, c: u8) {
        match self.lex {
            Lex::Code => self.code(c),
            Lex::Ident if is_ident(c) => {
                self.ident.push(c);
                self.token(c);
            }
            Lex::Ident
                if c == b'"'
                    && matches!(self.ident.as_slice(), b"R" | b"u8R" | b"uR" | b"UR" | b"LR") =>
            {
                if matches!(self.line, Line::Hash | Line::DefineName) {
                    self.line = Line::Inactive;
                }
                self.token(c);
                self.raw_len = 0;
                self.lex = Lex::RawDelim;
            }
            Lex::Ident => {
                self.lex = Lex::Code;
                self.end_ident(c);
                self.code(c);
            }
            Lex::Number if is_ident(c) || c == b'.' => self.token(c),
            Lex::Number if c == b'\'' => {
                self.token(c);
                self.lex = Lex::NumberQuote;
            }
            Lex::Number => {
                self.lex = Lex::Code;
                self.code(c);
            }
            Lex::NumberQuote if is_ident(c) => {
                self.token(c);
                self.lex = Lex::Number;
            }
            Lex::NumberQuote => {
                self.lex = Lex::Literal {
                    quote: b'\'',
                    escaped: false,
                };
                self.lex(c);
            }
            Lex::Slash if c == b'*' || c == b'/' => {
                self.trivia(b'/');
                self.trivia(c);
                self.lex = if c == b'*' {
                    Lex::BlockComment
                } else {
                    Lex::LineComment
                };
            }
            Lex::Slash => {
                self.lex = Lex::Code;
                self.start_token(b'/');
                self.code(c);
            }
            Lex::Percent => {
                self.lex = Lex::Code;
                if c == b':' {
                    self.line = Line::Hash;
                } else {
                    self.line = Line::Inactive;
                    self.code(c);
                }
            }
            Lex::LineComment if c == b'\n' => {
                self.lex = Lex::Code;
                self.newline();
            }
            Lex::LineComment => self.trivia(c),
            Lex::BlockComment | Lex::BlockCommentStar => {
                self.trivia(c);
                self.lex = match (self.lex, c) {
                    (Lex::BlockCommentStar, b'/') => Lex::Code,
                    (_, b'*') => Lex::BlockCommentStar,
                    _ => Lex::BlockComment,
                };
            }
            // Like GCC and Clang, end an unterminated literal at the end of its line.
            Lex::Literal { .. } | Lex::HeaderName { .. } if c == b'\n' => {
                self.lex = Lex::Code;
                self.newline();
            }
            Lex::Literal { quote, escaped } => {
                self.token(c);
                if escaped || c == b'\\' {
                    self.lex = Lex::Literal {
                        quote,
                        escaped: !escaped,
                    };
                } else if c == quote {
                    self.lex = Lex::Code;
                }
            }
            Lex::HeaderName { close } => {
                self.operand.push(c);
                if c == close {
                    self.operand_end = self.operand.len();
                    self.lex = Lex::Code;
                }
            }
            Lex::RawDelim if c == b'(' => {
                self.token(c);
                self.lex = Lex::RawBody { matched: 0 };
            }
            Lex::RawDelim
                if c.is_ascii_graphic()
                    && c != b')'
                    && c != b'\\'
                    && self.raw_len < self.raw_delim.len() =>
            {
                self.token(c);
                self.raw_delim[self.raw_len] = c;
                self.raw_len += 1;
            }
            // Not a valid raw string delimiter, so lex it as an ordinary string.
            Lex::RawDelim => {
                self.lex = Lex::Literal {
                    quote: b'"',
                    escaped: false,
                };
                self.lex(c);
            }
            Lex::RawBody { matched } => {
                self.token(c);
                let delim = &self.raw_delim[..self.raw_len];
                self.lex = if matched == delim.len() + 1 && c == b'"' {
                    Lex::Code
                } else if matched > 0 && delim.get(matched - 1) == Some(&c) {
                    Lex::RawBody {
                        matched: matched + 1,
                    }
                } else {
                    Lex::RawBody {
                        matched: usize::from(c == b')'),
                    }
                };
            }
        }
    }

    fn code(&mut self, c: u8) {
        match c {
            b'\n' => self.newline(),
            b' ' | b'\t' | b'\r' | b'\x0b' | b'\x0c' => self.trivia(c),
            b'/' => self.lex = Lex::Slash,
            _ => self.start_token(c),
        }
    }

    fn start_token(&mut self, c: u8) {
        match self.line {
            Line::Start if c == b'#' => {
                self.line = Line::Hash;
                return;
            }
            Line::Start if c == b'%' => {
                self.lex = Lex::Percent;
                return;
            }
            Line::Start => self.line = Line::Inactive,
            Line::Hash | Line::DefineName if !is_ident_start(c) => self.line = Line::Inactive,
            Line::Params { .. } if c == b')' => {
                self.line = Line::Active;
                return;
            }
            // `...`, or GNU C's `name...`, which names the variadic parameter.
            Line::Params {
                dots: 2,
                after_name,
            } if c == b'.' => {
                if !after_name {
                    self.definition.params.extend_from_slice(b"__VA_ARGS__,");
                }
                self.definition.variadic = true;
                self.line = Line::Params {
                    dots: 0,
                    after_name: false,
                };
            }
            Line::Params { dots, after_name } if c == b'.' => {
                self.line = Line::Params {
                    dots: dots + 1,
                    after_name,
                }
            }
            Line::Params { .. } => {
                self.line = Line::Params {
                    dots: 0,
                    after_name: false,
                }
            }
            _ => {}
        }
        match self.expr {
            Expr::Idle => {}
            Expr::Keyword(operator) if c == b'(' => {
                self.expr = Expr::Operand(operator);
                self.operand_end = 0;
                self.depth = 0;
                return;
            }
            Expr::Keyword(_) => self.expr = Expr::Idle,
            Expr::Callee if c == b'(' => {
                self.expr = Expr::Args;
                self.operand_end = 0;
                self.depth = 0;
                self.args.clear();
                self.arg_start = None;
                return;
            }
            Expr::Callee => self.expr = Expr::Idle,
            Expr::Args => match c {
                b',' | b')' if self.depth == 0 => {
                    self.args.push(match self.arg_start.take() {
                        Some(start) => (start, self.operand_end),
                        None => (self.operand.len(), self.operand.len()),
                    });
                    if c == b')' {
                        self.finish_call();
                    } else {
                        // Kept, though not as part of an argument, for `__VA_ARGS__`.
                        self.operand.push(c);
                    }
                    return;
                }
                b'(' => self.depth += 1,
                b')' => self.depth -= 1,
                _ => {}
            },
            Expr::Operand(operator) => match c {
                b'<' | b'"' if self.operand.is_empty() => {
                    self.operand.push(c);
                    let close = if c == b'<' { b'>' } else { b'"' };
                    self.lex = Lex::HeaderName { close };
                    return;
                }
                b'(' => self.depth += 1,
                b')' if self.depth == 0 => {
                    self.finish_operand(operator);
                    return;
                }
                b')' => self.depth -= 1,
                _ => {}
            },
        }
        self.token(c);
        match c {
            b'"' | b'\'' => {
                self.lex = Lex::Literal {
                    quote: c,
                    escaped: false,
                }
            }
            b'0'..=b'9' => self.lex = Lex::Number,
            _ if is_ident_start(c) => {
                self.ident.clear();
                self.ident.push(c);
                self.lex = Lex::Ident;
            }
            _ => {}
        }
    }

    /// Handles the identifier just lexed, which `next` follows.
    fn end_ident(&mut self, next: u8) {
        match self.line {
            Line::Hash => {
                self.line = match self.ident.as_slice() {
                    b"if" | b"elif" => Line::Active,
                    b"define" => Line::DefineName,
                    _ => Line::Inactive,
                }
            }
            // The name being defined isn't a use, even when it is `__has_include` itself. A `(`
            // right after it starts the parameters of a function-like macro.
            Line::DefineName if next == b'(' => {
                self.defining = true;
                self.definition.name.clone_from(&self.ident);
                self.definition.params.clear();
                self.definition.variadic = false;
                self.line = Line::Params {
                    dots: 0,
                    after_name: false,
                };
            }
            Line::DefineName => self.line = Line::Active,
            Line::Params { .. } => {
                self.definition.params.extend_from_slice(&self.ident);
                self.definition.params.push(b',');
                self.line = Line::Params {
                    dots: 0,
                    after_name: true,
                };
            }
            Line::Active if self.expr == Expr::Idle => {
                self.expr = match self.ident.as_slice() {
                    b"__has_include" => Expr::Keyword(Operator::Include),
                    b"__has_include_next" => Expr::Keyword(Operator::IncludeNext),
                    b"__has_embed" => Expr::Keyword(Operator::Embed),
                    // A macro isn't expanded within its own definition.
                    name if !self.wrappers.is_empty()
                        && self.wrappers.contains_key(name)
                        && !(self.defining && self.definition.name == name) =>
                    {
                        self.callee.clone_from(&self.ident);
                        Expr::Callee
                    }
                    _ => Expr::Idle,
                }
            }
            _ => {}
        }
    }

    /// Records a byte of a token, which extends the operand or argument being collected.
    fn token(&mut self, c: u8) {
        match self.expr {
            Expr::Operand(_) => {}
            Expr::Args => {
                self.arg_start.get_or_insert(self.operand.len());
            }
            _ => return,
        }
        self.operand.push(c);
        self.operand_end = self.operand.len();
    }

    /// Records a byte of whitespace or comment, which is kept only between tokens.
    fn trivia(&mut self, c: u8) {
        if matches!(self.expr, Expr::Operand(_) | Expr::Args) && !self.operand.is_empty() {
            self.operand.push(c);
        }
    }

    /// Ends the line, and with it any directive and unfinished operand or call.
    fn newline(&mut self) {
        self.end_definition();
        self.line = Line::Start;
        self.expr = Expr::Idle;
        self.operand.clear();
    }

    fn finish_operand(&mut self, operator: Operator) {
        self.operand.truncate(self.operand_end);
        let operand = std::mem::take(&mut self.operand);
        self.found(operator, operand);
        self.expr = Expr::Idle;
    }

    /// Collects the operands that the call to `callee`, with arguments `args`, passes to
    /// operators.
    fn finish_call(&mut self) {
        let wrapper = &self.wrappers[&self.callee];
        let operands: Vec<_> = wrapper
            .templates
            .iter()
            .filter_map(|t| Some((t.operator, t.instantiate(&self.operand, &self.args)?)))
            .collect();
        self.operand.clear();
        self.expr = Expr::Idle;
        for (operator, operand) in operands {
            self.found(operator, operand);
        }
    }

    /// Collects an operand, unless it uses the parameters of the function-like macro being
    /// defined, which makes it a template for that macro's calls.
    fn found(&mut self, operator: Operator, operand: Vec<u8>) {
        if operand.is_empty() {
            return;
        }
        if self.defining && mentions(&operand, &self.definition.params) {
            let template = Template {
                operator,
                params: self.definition.params.clone(),
                variadic: self.definition.variadic,
                operand,
            };
            if !self.definition.templates.contains(&template) {
                self.definition.templates.push(template);
            }
            return;
        }
        let operand = String::from_utf8_lossy(&operand).into_owned();
        match operator {
            Operator::Include => self.includes.push(operand),
            Operator::IncludeNext => self.next_includes.push(operand),
            Operator::Embed => self.embeds.push(operand),
        }
    }

    /// Ends the definition on this line, recording the macro as a wrapper if it has templates.
    fn end_definition(&mut self) {
        if !std::mem::take(&mut self.defining) || self.definition.templates.is_empty() {
            return;
        }
        let templates = std::mem::take(&mut self.definition.templates);
        let wrapper = self
            .wrappers
            .entry(self.definition.name.clone())
            .or_insert_with_key(|name| Wrapper {
                name: memmem::Finder::new(name).into_owned(),
                templates: Vec::new(),
            });
        for template in templates {
            if !wrapper.templates.contains(&template) {
                wrapper.templates.push(template);
            }
        }
    }
}

impl Template {
    /// The operand for a call whose arguments are at `args` in `text`, or `None` if the call
    /// passes the wrong number of arguments.
    fn instantiate(&self, text: &[u8], mut args: &[(usize, usize)]) -> Option<Vec<u8>> {
        let params = self.params.iter().filter(|&&b| b == b',').count();
        let named = params - usize::from(self.variadic);
        // `F()` passes one empty argument, or none to a macro without parameters.
        if params == 0 && matches!(args, [(start, end)] if start == end) {
            args = &[];
        }
        if args.len() < named || (!self.variadic && args.len() > named) {
            return None;
        }
        let mut values: Vec<&[u8]> = args[..named].iter().map(|&(s, e)| &text[s..e]).collect();
        if self.variadic {
            // The variadic arguments, with the commas between them.
            values.push(match &args[named..] {
                [] => &[],
                [(start, _), .., (_, end)] | [(start, end)] => &text[*start..*end],
            });
        }
        Some(substitute(&self.operand, &self.params, &values))
    }
}

/// The position of `word` among `params`, which are each followed by a comma.
fn param_index(params: &[u8], word: &[u8]) -> Option<usize> {
    params.split(|&b| b == b',').position(|p| p == word)
}

/// Whether `text` uses any of `params`.
fn mentions(text: &[u8], params: &[u8]) -> bool {
    let mut i = 0;
    while i < text.len() {
        let (piece, end) = Piece::at(text, i);
        if piece == Piece::Word && param_index(params, &text[i..end]).is_some() {
            return true;
        }
        i = end;
    }
    false
}

/// Replaces `params` in `text` with `args`, applying `#` and `##` as macro expansion does,
/// though without expanding other macros.
fn substitute(text: &[u8], params: &[u8], args: &[&[u8]]) -> Vec<u8> {
    let arg = |word: &[u8]| param_index(params, word).map(|k| args[k]);
    let skip_space = |mut i: usize| {
        while text.get(i).is_some_and(u8::is_ascii_whitespace) {
            i += 1;
        }
        i
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let (piece, mut end) = Piece::at(text, i);
        match piece {
            Piece::Word => out.extend_from_slice(arg(&text[i..end]).unwrap_or(&text[i..end])),
            // `#` before a parameter makes a string literal of its argument.
            Piece::Hash => {
                let next = skip_space(end);
                let param = (next < text.len())
                    .then(|| Piece::at(text, next))
                    .filter(|&(piece, _)| piece == Piece::Word)
                    .and_then(|(_, word_end)| Some((arg(&text[next..word_end])?, word_end)));
                match param {
                    Some((value, word_end)) => {
                        stringize(value, &mut out);
                        end = word_end;
                    }
                    None => out.push(b'#'),
                }
            }
            // `##` joins what's on either side of it.
            Piece::Paste => {
                while out.last().is_some_and(u8::is_ascii_whitespace) {
                    out.pop();
                }
                end = skip_space(end);
            }
            Piece::Other => out.extend_from_slice(&text[i..end]),
        }
        i = end;
    }
    out
}

/// Appends `value` as a string literal, as `#` makes it.
fn stringize(value: &[u8], out: &mut Vec<u8>) {
    out.push(b'"');
    let mut space = false;
    for &b in value {
        if b.is_ascii_whitespace() {
            space = true;
            continue;
        }
        if std::mem::take(&mut space) {
            out.push(b' ');
        }
        if b == b'"' || b == b'\\' {
            out.push(b'\\');
        }
        out.push(b);
    }
    out.push(b'"');
}

/// A piece of a macro's replacement list, as far as substitution is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Piece {
    /// An identifier, which may be a parameter.
    Word,
    /// `#`
    Hash,
    /// `##`
    Paste,
    /// Anything else, including literals, numbers, comments, and whitespace.
    Other,
}

impl Piece {
    /// The piece of `text` starting at `i`, and where it ends.
    fn at(text: &[u8], i: usize) -> (Piece, usize) {
        // The end of the run of bytes after the first that satisfy `keep`.
        let run =
            |keep: fn(u8) -> bool| i + 1 + text[i + 1..].iter().take_while(|&&b| keep(b)).count();
        match &text[i..] {
            [quote @ (b'"' | b'\''), ..] => {
                let mut j = i + 1;
                while j < text.len() && text[j] != *quote {
                    j += if text[j] == b'\\' { 2 } else { 1 };
                }
                (Piece::Other, (j + 1).min(text.len()))
            }
            [b'/', b'*', tail @ ..] => (
                Piece::Other,
                memmem::find(tail, b"*/").map_or(text.len(), |e| i + 4 + e),
            ),
            [b'#', b'#', ..] => (Piece::Paste, i + 2),
            [b'#', ..] => (Piece::Hash, i + 1),
            [c, ..] if c.is_ascii_digit() => (Piece::Other, run(|b| is_ident(b) || b == b'.')),
            [c, ..] if is_ident_start(*c) => (Piece::Word, run(is_ident)),
            _ => (Piece::Other, i + 1),
        }
    }
}

/// The position of the first byte in `rest` that might start a comment or literal, splice
/// lines, or end the line.
fn next_significant(rest: &[u8]) -> usize {
    let end = memchr3(b'\n', b'/', b'"', rest).unwrap_or(rest.len());
    memchr2(b'\'', b'\\', &rest[..end]).unwrap_or(end)
}

/// The length of `run` before the identifier or number at its end, if any. That is left to
/// `lex`, since it may be a raw string prefix or a number with digit separators.
fn before_last_word(run: &[u8]) -> usize {
    run.iter()
        .rposition(|&b| !is_ident(b) && b != b'.')
        .map_or(0, |last| last + 1)
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c == b'$' || c >= 0x80
}

fn is_ident(c: u8) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::test::utils::*;
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_serialize_deserialize() -> Result<()> {
        let f = TestFixture::new();
        let p = f.touch("file.c")?;

        let mut entry_a = SourceFileDeps::new(
            "src_key".into(),
            SourceFileMeta {
                path: p.into(),
                digest: String::new(),
                fsize: 0,
                mtime: None,
                ctime: None,
            },
        );

        entry_a.add_revision("obj_key", Dependencies::default(), "file.c");

        assert_eq!(entry_a.iter().count(), 1);

        let buf = opendal::Buffer::try_from(entry_a)?;
        let entry_b = SourceFileDeps::try_from(buf.clone())?;

        assert_eq!(entry_b.iter().count(), 1);
        assert_eq!(buf.to_bytes(), entry_b.into_buffer()?.to_bytes());

        Ok(())
    }

    // Entries already in caches must stay readable.
    #[test]
    fn test_into_buffer_encoding() -> Result<()> {
        let meta = |path: &str| SourceFileMeta {
            path: PathBuf::from(path).into(),
            digest: "0123abcd".into(),
            fsize: 42,
            mtime: Some(Timestamp::new(1_700_000_000, 5)),
            ctime: None,
        };
        let mut entry = SourceFileDeps::new("src_key".into(), meta("/src/file.c"));
        entry.add_revision(
            "obj_key",
            Dependencies {
                sources: vec![meta("/src/file.h"), meta("/usr/include/stdio.h")],
                outputs: BTreeMap::from([("dep".into(), "file.o: file.c".into())]),
            },
            "file.c",
        );

        let mut inflated = vec![];
        flate2::read::ZlibDecoder::new(entry.clone().into_buffer()?.reader())
            .read_to_end(&mut inflated)?;
        assert_eq!(
            inflated,
            [DEPS_VERSION, &bincode::serialize(&entry)?].concat()
        );

        Ok(())
    }

    /// A finder's `includes`, `next_includes`, and `embeds`.
    type Found = (Vec<String>, Vec<String>, Vec<String>);

    /// Runs `scan` on a new finder in another thread, failing rather than hanging if it never
    /// returns.
    fn found(scan: impl FnOnce(&mut HiddenIncludesFinder) + Send + 'static) -> Found {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut finder = HiddenIncludesFinder::default();
            scan(&mut finder);
            tx.send((finder.includes, finder.next_includes, finder.embeds))
                .unwrap();
        });
        rx.recv_timeout(Duration::from_secs(10))
            .expect("the finder hung or panicked")
    }

    /// Scans `src` fed in chunks that end at each of `cuts`.
    fn find(src: &Arc<[u8]>, cuts: &[usize]) -> Found {
        let (src, cuts) = (Arc::clone(src), cuts.to_vec());
        found(move |finder| {
            let mut start = 0;
            for end in cuts.into_iter().chain([src.len()]) {
                finder.find_has_include(&src[start..end]);
                start = end;
            }
        })
    }

    /// Asserts that `src` yields exactly `includes`, and nothing for the other operators.
    fn assert_finds(src: &[u8], includes: &[&str]) {
        assert_finds_each(src, includes, &[], &[]);
    }

    /// Asserts that `src` yields exactly the given operands of each operator when scanned
    /// whole (by `find_has_include` and by `find_in_file`), one byte at a time, and split in
    /// two at every offset (about a thousand evenly spaced ones, if it's long).
    fn assert_finds_each(src: &[u8], includes: &[&str], next_includes: &[&str], embeds: &[&str]) {
        let set = |operands: &[&str]| operands.iter().map(|&s| s.into()).collect();
        let expected: Found = (set(includes), set(next_includes), set(embeds));
        let src: Arc<[u8]> = src.into();
        let text = String::from_utf8_lossy(&src);
        assert_eq!(find(&src, &[]), expected, "whole input: {text:?}");
        let file = Arc::clone(&src);
        assert_eq!(
            found(move |finder| finder.find_in_file(&file)),
            expected,
            "find_in_file: {text:?}"
        );
        let every_byte: Vec<usize> = (1..src.len()).collect();
        assert_eq!(
            find(&src, &every_byte),
            expected,
            "one byte at a time: {text:?}"
        );
        for cut in (1..src.len()).step_by(src.len().div_ceil(1000).max(1)) {
            assert_eq!(
                find(&src, &[cut]),
                expected,
                "split at byte {cut}: {text:?}"
            );
        }
    }

    #[test]
    fn test_find_has_includes() {
        assert_finds(
            TEST_SOURCE.as_bytes(),
            &[
                r#"<hello_kitty_1.h>"#,
                r#"<hello_kitty_2.h>"#,
                r#"<hello_kitty_3.h>"#,
                r#""hello_kitty_4.h""#,
                r#"_HELLO_KITTY(5, _HELLO_KITTY(10))"#,
                r#"<cuda_fp16.h>"#,
                r#"<cuda_bf16.h>"#,
                r#"<cuda_fp8.h>"#,
                r#"<cuda_fp6.h>"#,
                r#"<cuda_fp4.h>"#,
            ],
        );
    }

    #[test]
    fn test_macro_operands() {
        // The code after some operands has parentheses that must not be matched with theirs.
        assert_finds(
            b"#if __has_include(HDR)\n#endif\nint f(int x) { return (x); }\n",
            &["HDR"],
        );
        assert_finds(b"#if __has_include(HDR)\n#endif\n", &["HDR"]);
        assert_finds(b"#define HAS_A __has_include(HDR)", &["HDR"]);
        assert_finds(
            b"#if __has_include(MK(a))\n#endif\nint f(int x) { return (x); }\n",
            &["MK(a)"],
        );
        assert_finds(b"#if __has_include(MK2(a, b))\n#endif\n", &["MK2(a, b)"]);
        assert_finds(
            b"#if __has_include(M(N(a), O(b)))\n#endif\n",
            &["M(N(a), O(b))"],
        );
        assert_finds(
            b"#if __has_include(I(I(MK(a))))\n#endif\n",
            &["I(I(MK(a)))"],
        );
        assert_finds(
            b"#if __has_include(HDR) && __has_include(<b.h>)\n#endif\n",
            &["HDR", "<b.h>"],
        );
        // Parentheses inside literals and comments don't count.
        assert_finds(
            b"#if __has_include(ANY(\")\", ')', /* ) */ a))\n#endif\n",
            &["ANY(\")\", ')', /* ) */ a)"],
        );
    }

    #[test]
    fn test_wrapper_macros() {
        // Calls are collected, but not the wrapper's own operand.
        assert_finds(
            b"#ifdef __has_include\n#  define HAS_INC(x) __has_include(x)\n#else\n#  define HAS_INC(x) 0\n#endif\n#if HAS_INC(<a.h>) || HAS_INC /* c */ (\"b.h\")\n#endif\n",
            &["<a.h>", "\"b.h\""],
        );
        assert_finds_each(
            b"#define HAS_NEXT(x) __has_include_next(x)\n#define HAS_EMBED(x) __has_embed(x)\n#if HAS_NEXT(<a.h>) && HAS_EMBED(<data.bin> limit(1))\n#endif\n",
            &[],
            &["<a.h>"],
            &["<data.bin> limit(1)"],
        );
        // Parameters are replaced anywhere in the operand, with `#` and `##` applied.
        assert_finds(
            b"#define HAS_CUDA(name) __has_include(<cuda/name>)\n#define HAS_QUOTED(h) __has_include(#h)\n#define HAS_FP(bits) __has_include(<cuda_fp##bits.h>)\n#if HAS_CUDA(barrier) || HAS_QUOTED(a.h) || HAS_FP(16)\n#endif\n",
            &["<cuda/barrier>", "\"a.h\"", "<cuda_fp16.h>"],
        );
        // Several parameters, variadic parameters, and arguments containing commas.
        assert_finds(
            b"#define HAS_EITHER(a, b) (__has_include(a) || __has_include(b))\n#define HAS_ANY(...) __has_include(__VA_ARGS__)\n#define HAS_GNU(args...) __has_include(args)\n#if HAS_EITHER(<a.h>, MK2(b, c)) || HAS_ANY(<d.h>) || HAS_GNU(<e.h>)\n#endif\nint f(int x) { return (x); }\n",
            &["<a.h>", "MK2(b, c)", "<d.h>", "<e.h>"],
        );
        // Wrappers of wrappers, and operands that don't use the parameters.
        assert_finds(
            b"#define HAS_INC(x) __has_include(x)\n#define HAS_SUB(y) HAS_INC(<sub/y>)\n#define HAS_A(x) (__has_include(<a.h>) && x)\n#if HAS_SUB(b.h)\n#endif\n",
            &["<a.h>", "<sub/b.h>"],
        );
        // Neither the name alone nor an object-like macro is a call.
        assert_finds(
            b"#define HAS_INC(x) __has_include(x)\n#if defined(HAS_INC) && defined HAS_INC\n#endif\n#define NOT_A_CALL (x) __has_include(x)\n",
            &["x"],
        );
    }

    #[test]
    fn test_whitespace() {
        assert_finds(b"#if __has_include\t(<a.h>)\n#endif\n", &["<a.h>"]);
        assert_finds(
            b"#if __has_include(\t<a.h>\t)\n#endif\nint f(int x) { return (x); }\n",
            &["<a.h>"],
        );
        assert_finds(b"#if __has_include\x0b(\x0c<a.h>)\n#endif\n", &["<a.h>"]);
        assert_finds(b"#if __has_include \\\r\n(<a.h>)\r\n#endif\r\n", &["<a.h>"]);
        // GCC and Clang allow whitespace between a line continuation and its newline.
        assert_finds(b"#if __has_include \\ \t\n(<a.h>)\n#endif\n", &["<a.h>"]);
    }

    #[test]
    fn test_comments() {
        assert_finds(
            b"#if __has_include(/*/ c */ <a.h>)\n#endif\nint f(int x) { return (x); }\n",
            &["<a.h>"],
        );
        assert_finds(
            b"#if __has_include(/** c **/<a.h> /* c */)\n#endif\n",
            &["<a.h>"],
        );
        assert_finds(
            b"#if __has_include /* c\n c */ (<a.h>)\n#endif\n",
            &["<a.h>"],
        );
        assert_finds(
            b"/* c\n c */ # /* c */ if __has_include(<a.h>)\n#endif\n",
            &["<a.h>"],
        );
    }

    #[test]
    fn test_line_splices() {
        assert_finds(b"#if __has_\\\ninclude(<a.h>)\n#endif\n", &["<a.h>"]);
        assert_finds(b"#\\\nif __has_include(<a.h>)\n#endif\n", &["<a.h>"]);
        // Collected as the name the compiler searches for.
        assert_finds(b"#if __has_include(<a\\\n.h>)\n#endif\n", &["<a.h>"]);
        assert_finds(
            b"#if __has_include(MK2(a,\\\n b))\n#endif\n",
            &["MK2(a, b)"],
        );
        // A splice continues a line comment.
        assert_finds(b"// c \\\n#define HAS_A __has_include(<a.h>)\n", &[]);
    }

    #[test]
    fn test_header_names() {
        assert_finds(
            b"#if __has_include(<a.h>) && __has_include(\"b.h\")\n#elif __has_include(<c.h>)\n#endif\n",
            &["<a.h>", "\"b.h\"", "<c.h>"],
        );
        assert_finds(
            b"#if !(__has_include(<a.h>) || defined(X))\n#endif\n#if(__has_include(<b.h>))\n#endif\n",
            &["<a.h>", "<b.h>"],
        );
        assert_finds(b"#define HAS_A __has_include(<a.h>)", &["<a.h>"]);
        // Operands are kept in order, repeats included.
        assert_finds(
            b"#if __has_include(<b.h>)\n#elif __has_include(<a.h>) || __has_include(<b.h>)\n#endif\n",
            &["<b.h>", "<a.h>", "<b.h>"],
        );
        assert_finds(b"#if __has_include(<SUB/a.h>)\n#endif\n", &["<SUB/a.h>"]);
        assert_finds(b"%:if __has_include(<a.h>)\n%:endif\n", &["<a.h>"]);
        // Delimiter, comment, and escape characters are literal in header names.
        assert_finds(
            b"#if __has_include(\"a)b>c.h\") || __has_include(<a)b.h>)\n#endif\n",
            &["\"a)b>c.h\"", "<a)b.h>"],
        );
        assert_finds(
            b"#if __has_include(<a//b.h>) || __has_include(\"a/*b.h\") || __has_include(\"dir\\file.h\")\n#endif\n",
            &["<a//b.h>", "\"a/*b.h\"", "\"dir\\file.h\""],
        );
        // Bytes that aren't UTF-8 are replaced rather than causing a panic.
        assert_finds(
            b"#if __has_include(<caf\xe9.h>)\n#endif\n",
            &["<caf\u{FFFD}.h>"],
        );
    }

    #[test]
    fn test_literals() {
        // An unmatched quote ends with its line.
        assert_finds(
            b"#if 0\ndon't\n#endif\n#if __has_include(<a.h>)\n#endif\n",
            &["<a.h>"],
        );
        // Digit separators aren't character literals.
        assert_finds(b"#if 1'000 && __has_include(<a.h>)\n#endif\n", &["<a.h>"]);
        // A raw string may span lines and contain comment markers.
        assert_finds(
            b"auto s = R\"(\n/*\n)\";\n#if __has_include(<a.h>)\n#endif\n// */\n",
            &["<a.h>"],
        );
    }

    #[test]
    fn test_ignores_other_mentions() {
        assert_finds(
            b"#ifdef __has_include\n#endif\n#if defined(__has_include) || defined __has_include\n#endif\n",
            &[],
        );
        assert_finds(
            b"// __has_include(<a.h>)\n/* __has_include(<b.h>) */\n",
            &[],
        );
        assert_finds(
            b"auto s = \"__has_include(<a.h>)\";\nauto k = R\"(\n#if __has_include(<b.h>)\n#endif\n)\";\n",
            &[],
        );
        assert_finds(
            b"#if 0\nUse __has_include(<a.h>) here.\n#error __has_include(<b.h>) is required\n#endif\n",
            &[],
        );
        assert_finds(
            b"#define my__has_include(x) 0\n#if my__has_include(<a.h>)\n#endif\n",
            &[],
        );
        // The pre-C++17 fallback defines the name rather than using it.
        assert_finds(
            b"#ifndef __has_include\n#  define __has_include(x) 0\n#endif\nint f(int x) { return (x); }\n",
            &[],
        );
        // Directives end at a newline.
        assert_finds(
            b"void f() {\n#ifdef __has_include\n  (void)0;\n#endif\n}\n",
            &[],
        );
        assert_finds(
            b"void f() {\n#if __has_include(<a.h>)\n  (void)\"x\";\n#endif\n}\n",
            &["<a.h>"],
        );
    }

    #[test]
    fn test_has_include_next() {
        assert_finds_each(
            b"#if __has_include_next(<a.h>) || __has_include_next(\"b.h\")\n#endif\n",
            &[],
            &["<a.h>", "\"b.h\""],
            &[],
        );
        assert_finds_each(
            b"#if __has_include_next(MK(a))\n#endif\nint f(int x) { return (x); }\n",
            &[],
            &["MK(a)"],
            &[],
        );
        // Each operator collects into its own set.
        assert_finds_each(
            b"#if __has_include(<a.h>) && __has_include_next(<a.h>)\n#endif\n",
            &["<a.h>"],
            &["<a.h>"],
            &[],
        );
        assert_finds_each(
            b"#if __has_include_\\\nnext(<a.h>)\n#endif\n",
            &[],
            &["<a.h>"],
            &[],
        );
        assert_finds(
            b"#ifdef __has_include_next\n#endif\n#if defined(__has_include_next)\n#endif\n",
            &[],
        );
        assert_finds(
            b"#ifndef __has_include_next\n#  define __has_include_next(x) 0\n#endif\nint f(int x) { return (x); }\n",
            &[],
        );
        assert_finds(
            b"#define my__has_include_next(x) 0\n#if my__has_include_next(<a.h>)\n#endif\n",
            &[],
        );
    }

    #[test]
    fn test_has_embed() {
        assert_finds_each(
            b"#if __has_embed(<data.bin>) || __has_embed(\"data.txt\")\n#endif\n",
            &[],
            &[],
            &["<data.bin>", "\"data.txt\""],
        );
        // Embed parameters are part of the operand.
        assert_finds_each(
            b"#if __has_embed(<data.bin> limit(1) prefix(0xFF, ) if_empty(0)) == __STDC_EMBED_FOUND__\n#endif\n",
            &[],
            &[],
            &["<data.bin> limit(1) prefix(0xFF, ) if_empty(0)"],
        );
        assert_finds_each(
            b"#if __has_embed(\"data.txt\" clang::offset(2)) || __has_embed(<data.bin>limit(1))\n#endif\n",
            &[],
            &[],
            &["\"data.txt\" clang::offset(2)", "<data.bin>limit(1)"],
        );
        assert_finds_each(
            b"#if __has_embed( /* c */ <data.bin>\\\n limit(1) /* c */ )\n#endif\n",
            &[],
            &[],
            &["<data.bin> limit(1)"],
        );
        assert_finds_each(
            b"#if __has_embed(RES limit(1)) || __has_embed(MK(data) limit(1)) || __has_embed(MK (data))\n#endif\nint f(int x) { return (x); }\n",
            &[],
            &[],
            &["RES limit(1)", "MK(data) limit(1)", "MK (data)"],
        );
        assert_finds_each(
            b"#if __has_include(<a.h>) && __has_embed(<a.h>)\n#endif\n",
            &["<a.h>"],
            &[],
            &["<a.h>"],
        );
        assert_finds(
            b"#ifdef __has_embed\n#endif\n#if defined(__has_embed)\n#endif\n",
            &[],
        );
    }

    #[test]
    fn test_is_empty() {
        let mut finder = HiddenIncludesFinder::default();
        assert!(finder.is_empty());
        finder.find_has_include(b"#if __has_embed(<data.bin>)\n#endif\n");
        assert!(!finder.is_empty());
    }

    #[test]
    fn test_finish_file() {
        let mut finder = HiddenIncludesFinder::default();
        finder.find_has_include(
            b"#if __has_include_next(<a.h>) || __has_embed(<b.bin>)\n#endif\n#define A 1 // no newline at end of file",
        );
        finder.finish_file();
        finder.find_has_include(b"#if __has_include(<c.h>)\n#endif\n");
        assert_eq!(
            (&finder.next_includes, &finder.embeds),
            (&vec!["<a.h>".to_string()], &vec!["<b.bin>".to_string()])
        );
        assert_eq!(finder.includes, ["<c.h>"]);
    }

    #[test]
    fn test_find_in_file() {
        // `__has_` never appears whole in these, so `find_in_file` only lexes them because
        // a splice follows an identifier character.
        assert_finds(b"#if __ha\\\ns_include(<a.h>)\n#endif\n", &["<a.h>"]);
        assert_finds(b"#if _\\\n_has_include(<a.h>)\n#endif\n", &["<a.h>"]);
        assert_finds(b"#if __has\\ \t\n_include(<a.h>)\n#endif\n", &["<a.h>"]);
        assert_finds(b"#if __\\\n\\\nhas_include(<a.h>)\n#endif\n", &["<a.h>"]);
        assert_finds_each(
            b"#if __h\\\nas_include_next(<a.h>)\n#endif\n",
            &[],
            &["<a.h>"],
            &[],
        );
        assert_finds_each(
            b"#if __has\\\r\n_embed(<a.h> limit(1))\n#endif\n",
            &[],
            &[],
            &["<a.h> limit(1)"],
        );
        // Files with neither aren't lexed.
        assert!(
            !HiddenIncludesFinder::default()
                .might_name_operator(b"#define F(x) (x) \\\n  + 1\nconst char* s = \"a\\n\";\n")
        );
        // Each file starts fresh, whatever came before it.
        let mut finder = HiddenIncludesFinder::default();
        finder.find_has_include(b"/* unterminated");
        finder.find_in_file(b"#if __has_include(<a.h>)\n#endif\n");
        assert_eq!(finder.includes, ["<a.h>"]);
        // Except that wrappers carry over, so calls are found in files without `__has_`.
        let mut finder = HiddenIncludesFinder::default();
        finder.find_in_file(b"#define HAS_INC(x) __has_include(x)");
        finder.find_in_file(b"#if HAS_INC(<a.h>)\n#endif\n");
        finder.find_in_file(b"#if HAS_\\\nINC(<b.h>)\n#endif\n");
        assert_eq!(finder.includes, ["<a.h>", "<b.h>"]);
    }

    const TEST_SOURCE: &str = r#"
// Weird edge cases with comments, spaces, and newlines
#if __has_include(\
/*
  hello kitty
*/ \
<hello_kitty_1.h>)
hello kitty 1
#endif

#if __has_include (\
/*
  hello kitty
*/ \
<hello_kitty_2.h>)
hello kitty 2
#endif

#if __has_include \
(/* hello kitty */ \
\
<hello_kitty_3.h>)
hello kitty 3
#endif

#if _FOO && __has_include /* hello kitty */ \
( \
\
"hello_kitty_4.h")
hello kitty 4
#endif

#if _FOO && __has_include(_HELLO_KITTY(5, _HELLO_KITTY(10)))
hello kitty 5
#endif


// https://github.com/NVIDIA/cccl/blob/63707502c15f13a0196a9ad2a467e6c7b6d3c746/thrust/thrust/detail/config/host_system.h
// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// reserve 0 for undefined
#define THRUST_HOST_SYSTEM_CPP 1
#define THRUST_HOST_SYSTEM_OMP 2
#define THRUST_HOST_SYSTEM_TBB 3

#ifndef THRUST_HOST_SYSTEM
#  define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#endif // THRUST_HOST_SYSTEM

#if THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_CPP
#  define __THRUST_HOST_SYSTEM_NAMESPACE cpp
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_OMP
#  define __THRUST_HOST_SYSTEM_NAMESPACE omp
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_TBB
#  define __THRUST_HOST_SYSTEM_NAMESPACE tbb
#endif

// clang-format off
#define __THRUST_HOST_SYSTEM_ROOT thrust/system/__THRUST_HOST_SYSTEM_NAMESPACE
#define __THRUST_HOST_SYSTEM_ALGORITH_HEADER_INCLUDE(filename) <__THRUST_HOST_SYSTEM_ROOT/filename>
#define __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(filename) <__THRUST_HOST_SYSTEM_ROOT/detail/filename>
// clang-format on

// https://github.com/NVIDIA/cccl/blob/63707502c15f13a0196a9ad2a467e6c7b6d3c746/thrust/thrust/detail/config/device_system.h
// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// reserve 0 for undefined
#define THRUST_DEVICE_SYSTEM_CUDA 1
#define THRUST_DEVICE_SYSTEM_OMP  2
#define THRUST_DEVICE_SYSTEM_TBB  3
#define THRUST_DEVICE_SYSTEM_CPP  4

#ifndef THRUST_DEVICE_SYSTEM
#  define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA
#endif // THRUST_DEVICE_SYSTEM

#ifdef THRUST_DEVICE_BACKEND
#  error THRUST_DEVICE_BACKEND is no longer supported; use THRUST_DEVICE_SYSTEM instead.
#endif // THRUST_DEVICE_BACKEND

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  define __THRUST_DEVICE_SYSTEM_NAMESPACE cuda
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
#  define __THRUST_DEVICE_SYSTEM_NAMESPACE omp
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
#  define __THRUST_DEVICE_SYSTEM_NAMESPACE tbb
#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CPP
#  define __THRUST_DEVICE_SYSTEM_NAMESPACE cpp
#endif

// clang-format off
#define __THRUST_DEVICE_SYSTEM_ROOT thrust/system/__THRUST_DEVICE_SYSTEM_NAMESPACE
#define __THRUST_DEVICE_SYSTEM_ALGORITH_HEADER_INCLUDE(filename) <__THRUST_DEVICE_SYSTEM_ROOT/filename>
#define __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(filename) <__THRUST_DEVICE_SYSTEM_ROOT/detail/filename>
// clang-format on


// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/per_device_resource.h>
#include <thrust/system/detail/sequential/per_device_resource.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(per_device_resource.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(per_device_resource.h)

// https://github.com/NVIDIA/cccl/blob/6ad0be74a142127e0cf8a90f54210325b26538ee/libcudacxx/include/cuda/std/__cccl/extended_data_types.h
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_EXTENDED_DATA_TYPES_H
#define __CCCL_EXTENDED_DATA_TYPES_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/cuda_capabilities.h>
#include <cuda/std/__cccl/cuda_toolkit.h>
#include <cuda/std/__cccl/diagnostic.h>
#include <cuda/std/__cccl/host_arch.h>
#include <cuda/std/__cccl/os.h>
#include <cuda/std/__cccl/preprocessor.h>

#define _CCCL_HAS_INT128()   0
#define _CCCL_HAS_NVFP4()    0
#define _CCCL_HAS_NVFP6()    0
#define _CCCL_HAS_NVFP8()    0
#define _CCCL_HAS_NVFP16()   0
#define _CCCL_HAS_NVBF16()   0
#define _CCCL_HAS_FLOAT64()  0
#define _CCCL_HAS_FLOAT128() 0

#if _CCCL_TILE_COMPILATION() // TODO(miscco): Fix access to extended floating point types
#  define CCCL_DISABLE_NVFP4_SUPPORT
#  define CCCL_DISABLE_NVFP6_SUPPORT
#  define CCCL_DISABLE_NVFP8_SUPPORT
#  define CCCL_DISABLE_INT128_SUPPORT
#  define CCCL_DISABLE_FLOAT128_SUPPORT
#endif // _CCCL_TILE_COMPILATION()

#if !defined(CCCL_DISABLE_INT128_SUPPORT) && _CCCL_OS(LINUX) \
  && ((_CCCL_COMPILER(NVRTC) && defined(__CUDACC_RTC_INT128__)) || defined(__SIZEOF_INT128__))
#  undef _CCCL_HAS_INT128
#  define _CCCL_HAS_INT128() 1
#endif

#if __has_include(<cuda_fp16.h>) && (_CCCL_HAS_CTK() || defined(LIBCUDACXX_ENABLE_HOST_NVFP16)) \
                      && !defined(CCCL_DISABLE_FP16_SUPPORT)
#  undef _CCCL_HAS_NVFP16
#  define _CCCL_HAS_NVFP16() 1
struct __half;
struct __half2;
#endif

#if __has_include(<cuda_bf16.h>) && _CCCL_HAS_NVFP16() && !defined(CCCL_DISABLE_BF16_SUPPORT)
#  undef _CCCL_HAS_NVBF16
#  define _CCCL_HAS_NVBF16() 1
struct __nv_bfloat16;
struct __nv_bfloat162;
#endif

#if __has_include(<cuda_fp8.h>) && _CCCL_HAS_NVFP16() && _CCCL_HAS_NVBF16() && !defined(CCCL_DISABLE_NVFP8_SUPPORT)
#  undef _CCCL_HAS_NVFP8
#  define _CCCL_HAS_NVFP8() 1
struct __nv_fp8_e5m2;
struct __nv_fp8x2_e5m2;
struct __nv_fp8x4_e5m2;

struct __nv_fp8_e4m3;
struct __nv_fp8x2_e4m3;
struct __nv_fp8x4_e4m3;

#  if _CCCL_CTK_AT_LEAST(12, 8)
struct __nv_fp8_e8m0;
struct __nv_fp8x2_e8m0;
struct __nv_fp8x4_e8m0;
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
#endif

#if __has_include(<cuda_fp6.h>) && _CCCL_HAS_NVFP8() && !_CCCL_CUDA_COMPILER(NVHPC) \
                      && !defined(CCCL_DISABLE_NVFP6_SUPPORT)
#  undef _CCCL_HAS_NVFP6
#  define _CCCL_HAS_NVFP6() 1
struct __nv_fp6_e3m2;
struct __nv_fp6x2_e3m2;
struct __nv_fp6x4_e3m2;

struct __nv_fp6_e2m3;
struct __nv_fp6x2_e2m3;
struct __nv_fp6x4_e2m3;
#endif

#if __has_include(<cuda_fp4.h>) && _CCCL_HAS_NVFP6() && !defined(CCCL_DISABLE_NVFP4_SUPPORT)
#  undef _CCCL_HAS_NVFP4
#  define _CCCL_HAS_NVFP4() 1
struct __nv_fp4_e2m1;
struct __nv_fp4x2_e2m1;
struct __nv_fp4x4_e2m1;
#endif

#define _CCCL_HAS_NVFP4_E2M1() _CCCL_HAS_NVFP4()
#define _CCCL_HAS_NVFP6_E2M3() _CCCL_HAS_NVFP6()
#define _CCCL_HAS_NVFP6_E3M2() _CCCL_HAS_NVFP6()
#define _CCCL_HAS_NVFP8_E4M3() _CCCL_HAS_NVFP8()
#define _CCCL_HAS_NVFP8_E5M2() _CCCL_HAS_NVFP8()
#define _CCCL_HAS_NVFP8_E8M0() (_CCCL_HAS_NVFP8() && _CCCL_CTK_AT_LEAST(12, 8))

/***********************************************************************************************************************
 * _Float64
 **********************************************************************************************************************/

// _Float64, the type behind C++23's std::float64_t, is a distinct type from double even where the
// two are bit-identical, so code that accepts either has to name both. __STDCPP_FLOAT64_T__ guards
// the _Float64 token and guarantees the type is present, which is why no compiler version table is
// needed here; where _Float64 is merely an alias for double, the double spelling covers it anyway.
//
// nvcc does not support _Float64 in device code. A type that exists in one pass of a CUDA
// compilation and not the other would give the two passes different overload sets, so this reports
// false for nvcc in both passes rather than only in the device one.
#if defined(__STDCPP_FLOAT64_T__) && __STDCPP_FLOAT64_T__ == 1 && !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC)
#  undef _CCCL_HAS_FLOAT64
#  define _CCCL_HAS_FLOAT64() 1
#endif // __STDCPP_FLOAT64_T__ == 1 && !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC)

/***********************************************************************************************************************
 * __float128
 **********************************************************************************************************************/

#if !defined(CCCL_DISABLE_FLOAT128_SUPPORT) && _CCCL_HAS_INT128() && _CCCL_OS(LINUX) && !_CCCL_HOST_ARCH(ARM64)
// Detect host compiler support
#  if (defined(__CUDACC_RTC_FLOAT128__) || defined(__SIZEOF_FLOAT128__) || defined(__FLOAT128__))
#    if _CCCL_DEVICE_COMPILATION()
// Only NVCC and NVRTC 12.8+ on architectures at least SM100 supports __float128 on device
#      if (_CCCL_CUDA_COMPILER(NVCC, >=, 12, 8) || _CCCL_CUDA_COMPILER(NVRTC, >=, 12, 8)) && _CCCL_PTX_ARCH() >= 1000
#        undef _CCCL_HAS_FLOAT128
#        define _CCCL_HAS_FLOAT128() 1
#      endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_PTX_ARCH() >= 1000
#    else // ^^^ _CCCL_DEVICE_COMPILATION() ^^^ / vvv !_CCCL_DEVICE_COMPILATION() vvv
#      undef _CCCL_HAS_FLOAT128
#      define _CCCL_HAS_FLOAT128() 1
#    endif // ^^^ !_CCCL_DEVICE_COMPILATION() ^^^
#  endif // Host compiler support
#endif // !defined(CCCL_DISABLE_FLOAT128_SUPPORT) && _CCCL_HAS_INT128() && _CCCL_OS(LINUX) && !_CCCL_HOST_ARCH(ARM64)

// gcc does not allow to use q/Q floating point literals when __STRICT_ANSI__ is defined. They may be allowed by
// -fext-numeric-literals, but there is no way to detect it in the preprocessor. The user is required to define
// CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS in this case. Otherwise, we disable the __float128 support.
//
// Note: since GCC 13, we could use f128/F128 literals, but for values > DBL_MAX, the compilation with nvcc fails due to
//       "floating constant is out of range".
#if _CCCL_HAS_FLOAT128() && _CCCL_COMPILER(GCC) && defined(__STRICT_ANSI__) \
  && !defined(CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS)
#  undef _CCCL_HAS_FLOAT128
#  define _CCCL_HAS_FLOAT128() 0
#endif // _CCCL_HAS_FLOAT128()

#endif // __CCCL_EXTENDED_DATA_TYPES_H
"#;

    // #[test]
    // fn test_preprocessor_cache_entry_hash_key_basedirs() {
    //     #[cfg(target_os = "windows")]
    //     use crate::util::normalize_win_path;
    //     use std::fs;
    //     use tempfile::TempDir;

    //     // Create two different base directories
    //     let dir1 = TempDir::new().unwrap();
    //     let dir2 = TempDir::new().unwrap();
    //     let dirs = [&dir1, &dir2]
    //         .iter()
    //         .map(|dir| {
    //             let bytes = dir.path().to_string_lossy().into_owned().into_bytes();
    //             #[cfg(target_os = "windows")]
    //             return normalize_win_path(&bytes);
    //             #[cfg(not(target_os = "windows"))]
    //             bytes
    //         })
    //         .collect::<Vec<_>>();

    //     // Create identical files with the same relative path in each directory
    //     let file_path = Path::new("test.c");
    //     let content = b"int main() { return 0; }";
    //     fs::write(dir1.path().join(file_path), content).unwrap();
    //     fs::write(dir2.path().join(file_path), content).unwrap();

    //     let args = ParsedArguments {
    //         language: Language::C,
    //         input: file_path.into(),
    //         ..Default::default()
    //     };

    //     let compile_timestamp = chrono::Utc::now();

    //     // Test 1: With basedirs, hashes should be the same
    //     let hash1_with_basedirs = preprocessor_cache_entry_hash_key(
    //         "test_digest",
    //         None,
    //         &args,
    //         &[],
    //         &[],
    //         dir1.path(),
    //         false,
    //         &dirs,
    //         compile_timestamp,
    //     )
    //     .wait()
    //     .unwrap()
    //     .unwrap();

    //     let hash2_with_basedirs = preprocessor_cache_entry_hash_key(
    //         "test_digest",
    //         None,
    //         &args,
    //         &[],
    //         &[],
    //         dir2.path(),
    //         false,
    //         &dirs,
    //         compile_timestamp,
    //     )
    //     .wait()
    //     .unwrap()
    //     .unwrap();

    //     assert_ne!(
    //         hash1_with_basedirs, hash2_with_basedirs,
    //         "Hashes should be different when using basedirs with identical files in different directories"
    //     );

    //     // Test 2: With basedir1 for first, and basedir2 for second, hashes should be the same
    //     let hash1_with_basedirs = preprocessor_cache_entry_hash_key(
    //         "test_digest",
    //         None,
    //         &args,
    //         &[],
    //         &[],
    //         dir1.path(),
    //         false,
    //         &dirs[..1],
    //         compile_timestamp,
    //     )
    //     .wait()
    //     .unwrap()
    //     .unwrap();

    //     let hash2_with_basedirs = preprocessor_cache_entry_hash_key(
    //         "test_digest",
    //         None,
    //         &args,
    //         &[],
    //         &[],
    //         dir2.path(),
    //         false,
    //         &dirs[1..],
    //         compile_timestamp,
    //     )
    //     .wait()
    //     .unwrap()
    //     .unwrap();

    //     assert_ne!(
    //         hash1_with_basedirs, hash2_with_basedirs,
    //         "Hashes should be different when using basedirs with identical files in different directories"
    //     );

    //     // Test 3: Without basedirs, hashes should be different
    //     let hash1_no_basedirs = preprocessor_cache_entry_hash_key(
    //         "test_digest",
    //         None,
    //         &args,
    //         &[],
    //         &[],
    //         dir1.path(),
    //         false,
    //         &[],
    //         compile_timestamp,
    //     )
    //     .wait()
    //     .unwrap()
    //     .unwrap();

    //     let hash2_no_basedirs = preprocessor_cache_entry_hash_key(
    //         "test_digest",
    //         None,
    //         &args,
    //         &[],
    //         &[],
    //         dir2.path(),
    //         false,
    //         &[],
    //         compile_timestamp,
    //     )
    //     .wait()
    //     .unwrap()
    //     .unwrap();

    //     assert_ne!(
    //         hash1_no_basedirs, hash2_no_basedirs,
    //         "Hashes should be different without basedirs for files in different directories"
    //     );
    // }
}
