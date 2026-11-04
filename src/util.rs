// Copyright 2017 Mozilla Foundation
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
    compiler::HiddenIncludesFinder,
    errors::*,
    mock_command::{CommandChild, ProcessOutput, RunCommand},
};
use async_trait::async_trait;
use blake3::Hasher as blake3_Hasher;
use byteorder::{BigEndian, ByteOrder};
use bytes::BytesMut;
use fs_err as fs;
use futures::{
    AsyncRead, AsyncReadExt, FutureExt, StreamExt, TryFutureExt,
    io::{AllowStdIo, BufReader},
    lock::Mutex,
};
use object::read::{
    archive::ArchiveFile,
    macho::{FatArch, MachOFatFile32, MachOFatFile64},
};
use serde::{Deserialize, Serialize};
#[cfg(not(windows))]
use std::os::unix::io::RawFd;
use std::{
    borrow::Cow,
    cell::Cell,
    cmp::Eq,
    collections::HashMap,
    ffi::{OsStr, OsString},
    fmt::Debug,
    hash::{Hash, Hasher},
    io::prelude::*,
    path::{Path, PathBuf},
    process::{self, Stdio},
    str,
    sync::{Arc, LazyLock},
    time::{self, Duration, SystemTime},
};
use tokio_retry2::Retry;
use tokio_retry2::strategy::FibonacciBackoff;
use tokio_util::codec::{Decoder, FramedRead};

/// The url safe engine for base64.
pub const BASE64_URL_SAFE_ENGINE: base64::engine::GeneralPurpose =
    base64::engine::general_purpose::URL_SAFE_NO_PAD;

pub const HASH_BUFFER_SIZE: usize = 128 * 1024;

#[derive(Clone)]
pub struct Digest {
    inner: blake3_Hasher,
}

impl Digest {
    pub fn new() -> Digest {
        Digest {
            inner: blake3_Hasher::new(),
        }
    }

    fn open_file<T>(path: T) -> Result<BufReader<AllowStdIo<fs::File>>>
    where
        T: AsRef<Path>,
    {
        let path = path.as_ref();
        let file = fs::File::open(path)
            .with_context(|| format!("Failed to open file for hashing: {path:?}"))?;
        Ok(BufReader::new(AllowStdIo::new(file)))
    }

    fn open_file_sync<T>(path: T) -> Result<std::io::BufReader<fs::File>>
    where
        T: AsRef<Path>,
    {
        let path = path.as_ref();
        let file = fs::File::open(path)
            .with_context(|| format!("Failed to open file for hashing: {path:?}"))?;
        Ok(std::io::BufReader::new(file))
    }

    /// Calculate the BLAKE3 digest of the contents of `path`.
    pub async fn from_file<T>(path: T) -> Result<Self>
    where
        T: AsRef<Path>,
    {
        Self::new().with_file(path).await
    }

    /// Calculate the BLAKE3 digest of the contents of `path`.
    pub fn from_file_sync<T>(path: T) -> Result<Self>
    where
        T: AsRef<Path>,
    {
        Self::new().with_file_sync(path)
    }

    /// Calculate the BLAKE3 digest of the contents of `path`, while
    /// also checking for the presence of time macros.
    /// See [`TimeMacroFinder`] for more details.
    pub async fn from_file_with_macro_finders<T>(
        path: T,
        compile_timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<(Self, TimeMacroFinder, HiddenIncludesFinder)>
    where
        T: AsRef<Path>,
    {
        Self::new()
            .with_file_and_macro_finders(path, compile_timestamp)
            .await
    }

    /// Calculate the BLAKE3 digest of the contents of `path`, while
    /// also checking for the presence of time macros.
    /// See [`TimeMacroFinder`] for more details.
    pub fn from_file_with_macro_finders_sync<T>(
        path: T,
        compile_timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<(Self, TimeMacroFinder, HiddenIncludesFinder)>
    where
        T: AsRef<Path>,
    {
        Self::new().with_file_and_macro_finders_sync(path, compile_timestamp)
    }

    /// Calculate the BLAKE3 digest of the contents read from `reader`, calling
    /// `each` before each time the digest is updated.
    pub async fn from_reader_with_each<R: AsyncRead + Send, F: FnMut(&[u8])>(
        reader: R,
        each: F,
    ) -> Result<Self> {
        Digest::new()
            .with_reader_each(std::pin::pin!(reader), each)
            .await
    }

    /// Calculate the BLAKE3 digest of the contents read from `reader`, while
    /// also checking for the presence of time macros.
    /// See [`TimeMacroFinder`] for more details.
    pub async fn from_reader_with_macro_finders<R: AsyncRead + Send>(
        reader: R,
    ) -> Result<(Self, TimeMacroFinder, HiddenIncludesFinder)> {
        let mut time_finder = TimeMacroFinder::new();
        let mut hinc_finder = HiddenIncludesFinder::default();
        let digest = Self::from_reader_with_each(reader, |visit| {
            time_finder.find_time_macros(visit);
            hinc_finder.find_has_include(visit);
        })
        .await?;

        Ok((digest, time_finder, hinc_finder))
    }

    /// Update the BLAKE3 digest of the contents of `path` into this digest.
    pub async fn with_file<T>(self, path: T) -> Result<Self>
    where
        T: AsRef<Path>,
    {
        let path = path.as_ref();
        if path.is_dir() {
            // For directories (e.g., from
            // proc_macro::tracked_path::path()), recursively hash all file
            // names and contents in sorted order.
            let mut entries = Vec::new();
            Self::collect_dir_entries(path, &mut entries)?;
            entries.sort();
            let mut digest = self;
            for entry in entries {
                let entry = entry.as_path();
                let rel = entry.strip_prefix(path).unwrap_or(entry);
                digest.delimiter(&path_to_bytes(rel)?);
                digest = digest.with_reader(Self::open_file(entry)?).await?;
            }
            Ok(digest)
        } else {
            self.with_reader(Self::open_file(path)?).await
        }
    }

    /// Update the BLAKE3 digest of the contents of `path` into this digest.
    pub fn with_file_sync<T>(self, path: T) -> Result<Self>
    where
        T: AsRef<Path>,
    {
        let path = path.as_ref();
        if path.is_dir() {
            // For directories (e.g., from
            // proc_macro::tracked_path::path()), recursively hash all file
            // names and contents in sorted order.
            let mut entries = Vec::new();
            Self::collect_dir_entries(path, &mut entries)?;
            entries.sort();
            let mut digest = self;
            for entry in entries {
                let entry = entry.as_path();
                let rel = entry.strip_prefix(path).unwrap_or(entry);
                digest.delimiter(&path_to_bytes(rel)?);
                digest = digest.with_reader_sync(Self::open_file_sync(entry)?)?;
            }
            Ok(digest)
        } else {
            self.with_reader_sync(Self::open_file_sync(path)?)
        }
    }

    /// Update the BLAKE3 digest of the contents of `path` into this digest, while
    /// also checking for the presence of time macros.
    /// See [`TimeMacroFinder`] for more details.
    pub async fn with_file_and_macro_finders<T>(
        self,
        path: T,
        compile_timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<(Self, TimeMacroFinder, HiddenIncludesFinder)>
    where
        T: AsRef<Path>,
    {
        use chrono::Datelike;

        let path = path.as_ref();

        let (mut digest, time_finder, hinc_finder) = self
            .with_reader_and_macro_finders(Self::open_file(path)?)
            .await?;

        // If __DATE__ or __TIMESTAMP__ found, make sure that the digest changes
        // if the (potential) expansion of those macros changes by computing a new
        // digest comprising the file digest and time information that represents the
        // macro expansions.

        if time_finder.found_date() {
            debug!("found __DATE__ in {path:?}");
            digest.delimiter(b"date");
            let date = compile_timestamp.date_naive();
            digest.update(&date.year().to_le_bytes());
            digest.update(&date.month().to_le_bytes());
            digest.update(&date.day().to_le_bytes());
        }

        if time_finder.found_timestamp() {
            debug!("found __TIMESTAMP__ in {path:?}");
            let mtime: chrono::DateTime<chrono::Local> = tokio::fs::symlink_metadata(path)
                .await
                .with_context(|| format!("Failed to read file metadata for hashing: {path:?}"))?
                .modified()
                .with_context(|| format!("Failed to read file mtime for hashing: {path:?}"))?
                .into();

            digest.delimiter(b"timestamp");
            digest.update(&mtime.naive_local().and_utc().timestamp().to_le_bytes());
        }

        Ok((digest, time_finder, hinc_finder))
    }

    /// Update the BLAKE3 digest of the contents of `path` into this digest, while
    /// also checking for the presence of time macros.
    /// See [`TimeMacroFinder`] for more details.
    pub fn with_file_and_macro_finders_sync<T>(
        self,
        path: T,
        compile_timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Result<(Self, TimeMacroFinder, HiddenIncludesFinder)>
    where
        T: AsRef<Path>,
    {
        use chrono::Datelike;

        let path = path.as_ref();

        let (mut digest, time_finder, hinc_finder) =
            self.with_reader_and_macro_finders_sync(Self::open_file_sync(path)?)?;

        // If __DATE__ or __TIMESTAMP__ found, make sure that the digest changes
        // if the (potential) expansion of those macros changes by computing a new
        // digest comprising the file digest and time information that represents the
        // macro expansions.

        if time_finder.found_date() {
            debug!("found __DATE__ in {path:?}");
            digest.delimiter(b"date");
            let date = compile_timestamp.date_naive();
            digest.update(&date.year().to_le_bytes());
            digest.update(&date.month().to_le_bytes());
            digest.update(&date.day().to_le_bytes());
        }

        if time_finder.found_timestamp() {
            debug!("found __TIMESTAMP__ in {path:?}");
            let mtime: chrono::DateTime<chrono::Local> = fs::symlink_metadata(path)
                .with_context(|| format!("Failed to read file metadata for hashing: {path:?}"))?
                .modified()
                .with_context(|| format!("Failed to read file mtime for hashing: {path:?}"))?
                .into();

            digest.delimiter(b"timestamp");
            digest.update(&mtime.naive_local().and_utc().timestamp().to_le_bytes());
        }

        Ok((digest, time_finder, hinc_finder))
    }

    /// Update the BLAKE3 digest of the contents read from `reader` into this digest.
    pub async fn with_reader<R: AsyncRead + Send + Unpin>(self, reader: R) -> Result<Self> {
        self.with_reader_each(reader, |_| {}).await
    }

    /// Update the BLAKE3 digest of the contents of `reader` into this digest, while
    /// also checking for the presence of time macros.
    /// See [`TimeMacroFinder`] for more details.
    pub async fn with_reader_and_macro_finders<R: AsyncRead + Send + Unpin>(
        self,
        reader: R,
    ) -> Result<(Self, TimeMacroFinder, HiddenIncludesFinder)> {
        let mut time_finder = TimeMacroFinder::new();
        let mut hinc_finder = HiddenIncludesFinder::default();
        let digest = self
            .with_reader_each(reader, |lines| {
                time_finder.find_time_macros(lines);
                hinc_finder.find_has_include(lines);
            })
            .await?;
        Ok((digest, time_finder, hinc_finder))
    }

    /// Update the BLAKE3 digest of the contents read from `reader` into this digest.
    pub fn with_reader_sync<R: Read + Send>(self, reader: R) -> Result<Self> {
        self.with_reader_each_sync(reader, |_| {})
    }

    /// Update the BLAKE3 digest of the contents of `reader` into this digest, while
    /// also checking for the presence of time macros.
    /// See [`TimeMacroFinder`] for more details.
    pub fn with_reader_and_macro_finders_sync<R: Read + Send>(
        self,
        reader: R,
    ) -> Result<(Self, TimeMacroFinder, HiddenIncludesFinder)> {
        let mut time_finder = TimeMacroFinder::new();
        let mut hinc_finder = HiddenIncludesFinder::default();
        let digest = self.with_reader_each_sync(reader, |lines| {
            time_finder.find_time_macros(lines);
            hinc_finder.find_has_include(lines);
        })?;
        Ok((digest, time_finder, hinc_finder))
    }

    /// Update the BLAKE3 digest of the contents read from `reader` into this digest, calling
    /// `each` before each time the digest is updated.
    pub async fn with_reader_each<R: AsyncRead + Send + Unpin, F: FnMut(&[u8])>(
        mut self,
        mut reader: R,
        mut each: F,
    ) -> Result<Self> {
        // A buffer of 128KB should give us the best performance.
        // See https://eklitzke.org/efficient-file-copying-on-linux.
        let mut buffer = vec![0u8; HASH_BUFFER_SIZE];
        loop {
            let count = reader.read(&mut buffer[..]).await?;
            if count == 0 {
                break;
            }
            each(&buffer[..count]);
            self.update(&buffer[..count]);
        }
        Ok(self)
    }

    /// Update the BLAKE3 digest of the contents read from `reader` into this digest, calling
    /// `each` before each time the digest is updated.
    pub fn with_reader_each_sync<R: Read + Send, F: FnMut(&[u8])>(
        mut self,
        mut reader: R,
        mut each: F,
    ) -> Result<Self> {
        // A buffer of 128KB should give us the best performance.
        // See https://eklitzke.org/efficient-file-copying-on-linux.
        let mut buffer = [0u8; HASH_BUFFER_SIZE];
        loop {
            let count = reader.read(&mut buffer[..])?;
            if count == 0 {
                break;
            }
            each(&buffer[..count]);
            self.update(&buffer[..count]);
        }
        Ok(self)
    }

    /// Recursively collect all file paths within a directory.
    fn collect_dir_entries(dir: &Path, entries: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("Failed to read directory for hashing: {dir:?}"))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_dir_entries(&path, entries)?;
            } else {
                entries.push(path);
            }
        }
        Ok(())
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.inner.update(bytes);
    }

    pub fn delimiter(&mut self, name: &[u8]) {
        self.update(b"\0SCCACHE\0");
        self.update(name);
        self.update(b"\0");
    }

    pub fn finish(self) -> String {
        hex(self.inner.finalize().as_bytes())
    }
}

impl Default for Digest {
    fn default() -> Self {
        Self::new()
    }
}

/// The longest pattern we're looking for is `__TIMESTAMP__`
const MAX_HAYSTACK_LEN: usize = b"__TIMESTAMP__".len();

#[cfg(test)]
pub const MAX_TIME_MACRO_HAYSTACK_LEN: usize = MAX_HAYSTACK_LEN;

/// Used during the chunked hashing process to check for C preprocessor time
/// macros (namely `__TIMESTAMP__`, `__DATE__`, `__DATETIME__`) while reusing
/// the same buffer as the hashing function, for efficiency.
///
/// See `[Self::find_time_macros]` for details.
#[derive(Debug, Default)]
pub struct TimeMacroFinder {
    found_date: Cell<bool>,
    found_time: Cell<bool>,
    found_timestamp: Cell<bool>,
    overlap_buffer: [u8; MAX_HAYSTACK_LEN * 2],
    /// Counter of chunks of full size we've been through. Partial reads do
    /// not count and are handled separately.
    full_chunks_counter: usize,
    /// Contents of the previous read if it was smaller than `MAX_HAYSTACK_LEN`,
    /// plus MAX_HAYSTACK_LEN bytes of the previous chunk, to account for
    /// the possibility of partial reads splitting a time macro
    /// across two calls.
    previous_small_read: Vec<u8>,
}

impl TimeMacroFinder {
    /// Called for each chunk of a file during the hashing process
    /// when creating a dependencies cache entry.
    ///
    /// When buffer reading a file, we get something like this:
    ///
    /// `[xxxx....aaaa][bbbb....cccc][dddd....eeee][ffff...]`
    ///
    /// The brackets represent each buffer chunk. We use the fact that the largest
    /// pattern we're looking for is `__TIMESTAMP__` to avoid copying the entire
    /// file to memory and re-searching the entire buffer for each pattern.
    /// We can check inside each chunk for each pattern, and we use an overlap
    /// buffer to keep the last `b"__TIMESTAMP__".len()` bytes around from the
    /// last chunk, to also catch any pattern overlapping two chunks.
    ///
    /// In the above case, the overflow buffer would look like:
    ///
    /// ```text
    ///    Chunk 1
    ///    - aaaa0000
    ///    Chunk 2
    ///    - aaaabbbb
    ///    - cccc0000
    ///    Chunk 3
    ///    - ccccdddd
    ///    - eeee0000
    ///    Chunk 4
    ///    - eeeeffff
    ///    [...]
    /// ```
    ///
    /// We have to be careful to zero out the buffer right after each overlap check,
    /// otherwise we risk the (unlikely) case of a pattern being spread between the
    /// start of a chunk and its end.
    /// Finally, we need to account for partial reads: it's possible that a read
    /// smaller than the haystack hide a time macro because it spreads it across
    /// two calls. This makes the example more complicated and isn't necessary
    /// to get the point of the algorithm across.
    /// See unit tests for some concrete examples.
    pub fn find_time_macros(&mut self, visit: &[u8]) {
        if self.full_chunks_counter == 0 {
            if visit.len() <= MAX_HAYSTACK_LEN {
                // The read is smaller than the largest haystack.
                // We might get called again, if this was an incomplete read.
                if !self.previous_small_read.is_empty() {
                    // In a rare pathological case where all reads are small,
                    // this will grow up to the length of the file.
                    // It is *very* unlikely and of minor performance
                    // importance compared to just getting many small reads.
                    self.previous_small_read.extend(visit);
                } else {
                    visit.clone_into(&mut self.previous_small_read);
                }
                self.find_macros(&self.previous_small_read);
                return;
            }
            // Copy the right side of the visit to the left of the buffer
            let right_half = visit.len() - MAX_HAYSTACK_LEN;
            self.overlap_buffer[..MAX_HAYSTACK_LEN].copy_from_slice(&visit[right_half..]);
        } else {
            if visit.len() < MAX_HAYSTACK_LEN {
                // The read is smaller than the largest haystack.
                // We might get called again, if this was an incomplete read.
                if !self.previous_small_read.is_empty() {
                    self.previous_small_read.extend(visit);
                } else {
                    // Since this isn't the first non-small read (counter != 0)
                    // we need to start from MAX_HAYSTACK_LEN bytes of the previous
                    // read, otherwise we might miss a complete read followed
                    // by a small read.
                    let mut buf = self.overlap_buffer[..MAX_HAYSTACK_LEN].to_owned();
                    buf.extend(visit);
                    self.previous_small_read = buf;
                }

                // zero the right side of the buffer
                self.overlap_buffer[MAX_HAYSTACK_LEN..].copy_from_slice(&[0; MAX_HAYSTACK_LEN]);
                // Copy the visit to the right of the buffer, starting from the middle
                self.overlap_buffer[MAX_HAYSTACK_LEN..MAX_HAYSTACK_LEN + visit.len()]
                    .copy_from_slice(visit);

                // Check both the concatenation with the previous small read
                self.find_macros(&self.previous_small_read);
                // ...and the overlap buffer
                self.find_macros(&self.overlap_buffer);
                return;
            } else {
                // Copy the left side of the visit to the right of the buffer
                let left_half = MAX_HAYSTACK_LEN;
                self.overlap_buffer[left_half..].copy_from_slice(&visit[..left_half]);
                self.find_macros(&self.overlap_buffer);
                // zero the buffer
                self.overlap_buffer = Default::default();
                // Copy the right side of the visit to the left of the buffer
                let right_half = visit.len() - MAX_HAYSTACK_LEN;
                self.overlap_buffer[..MAX_HAYSTACK_LEN].copy_from_slice(&visit[right_half..]);
            }
            self.find_macros(&self.overlap_buffer);
        }
        // Also check the concatenation with the previous small read
        if !self.previous_small_read.is_empty() {
            let mut concatenated = self.previous_small_read.clone();
            concatenated.extend(visit);
            self.find_macros(&concatenated);
        }

        self.find_macros(visit);
        self.full_chunks_counter += 1;
        self.previous_small_read.clear();
    }

    fn find_macros(&self, buffer: &[u8]) {
        // TODO
        // This could be made more efficient, either by using a regex for all
        // three patterns, or by doing some SIMD trickery like `ccache` does.
        //
        // `ccache` reads the file twice, so we might actually already be
        // winning in most cases... though they have an inode cache.
        // In any case, let's only improve this if it ends up being slow.
        if memchr::memmem::find(buffer, b"__TIMESTAMP__").is_some() {
            self.found_timestamp.set(true);
        }
        if memchr::memmem::find(buffer, b"__TIME__").is_some() {
            self.found_time.set(true);
        }
        if memchr::memmem::find(buffer, b"__DATE__").is_some() {
            self.found_date.set(true);
        }
    }

    pub fn found_time_macros(&self) -> bool {
        self.found_date() || self.found_time() || self.found_timestamp()
    }

    pub fn found_time(&self) -> bool {
        self.found_time.get()
    }

    pub fn found_date(&self) -> bool {
        self.found_date.get()
    }

    pub fn found_timestamp(&self) -> bool {
        self.found_timestamp.get()
    }

    pub fn new() -> Self {
        Default::default()
    }
}

pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        s.push(hex(byte & 0xf));
        s.push(hex((byte >> 4) & 0xf));
    }
    return s;

    fn hex(byte: u8) -> char {
        match byte {
            0..=9 => (b'0' + byte) as char,
            _ => (b'a' + byte - 10) as char,
        }
    }
}

/// Calculate the digest of each file in `files` on background threads in
/// `pool`.
pub async fn hash_all<T>(files: &[T]) -> Result<Vec<String>>
where
    T: AsRef<Path>,
{
    let start = time::Instant::now();
    let hashes =
        futures::future::try_join_all(files.iter().map(|path| async move {
            Digest::from_file(path).await.map(|digest| digest.finish())
        }))
        .await?;
    if !hashes.is_empty() {
        trace!(
            "Hashed {} files in {}",
            files.len(),
            fmt_duration_as_secs(&start.elapsed())
        );
    }
    Ok(hashes)
}

/// Build a tokio runtime suitable for the (short-lived) sccache client.
///
/// A multi-threaded runtime would spawn one worker thread per CPU for every
/// client invocation; on a many-core host running many parallel clients that
/// adds up to a large number of short-lived threads. A current-thread runtime
/// is enough for the client's request/response work.
pub fn new_client_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
}

/// Calculate the digest of each static library archive in `files` on background threads in
/// `pool`.
///
/// The hash is calculated by adding the filename of each archive entry followed
/// by its contents, ignoring headers and other file metadata. This primarily
/// exists because Apple's `ar` tool inserts timestamps for each file with
/// no way to disable this behavior.
pub async fn hash_all_archives(
    files: &[PathBuf],
    pool: &tokio::runtime::Handle,
) -> Result<Vec<String>> {
    let start = time::Instant::now();
    let count = files.len();
    let iter = files.iter().map(|path| {
        let path = path.clone();
        pool.spawn_blocking(move || -> Result<String> {
            let mut m = Digest::new();
            let archive_file = fs::File::open(&path)
                .with_context(|| format!("Failed to open file for hashing: {path:?}"))?;
            let archive_mmap =
                unsafe { memmap2::MmapOptions::new().map_copy_read_only(&archive_file)? };

            if let Ok(fat) = MachOFatFile32::parse(&*archive_mmap) {
                for arch in fat.arches() {
                    hash_regular_archive(&mut m, arch.data(&*archive_mmap)?)?;
                }
            } else if let Ok(fat) = MachOFatFile64::parse(&*archive_mmap) {
                for arch in fat.arches() {
                    hash_regular_archive(&mut m, arch.data(&*archive_mmap)?)?;
                }
            } else {
                // Not a FatHeader at all, regular archive.
                hash_regular_archive(&mut m, &archive_mmap)?;
            }

            Ok(m.finish())
        })
    });

    let mut hashes = futures::future::try_join_all(iter).await?;
    if let Some(i) = hashes.iter().position(|res| res.is_err()) {
        return Err(hashes.swap_remove(i).unwrap_err());
    }

    trace!(
        "Hashed {} files in {}",
        count,
        fmt_duration_as_secs(&start.elapsed())
    );
    Ok(hashes.into_iter().map(|res| res.unwrap()).collect())
}

fn hash_regular_archive(m: &mut Digest, data: &[u8]) -> Result<()> {
    let archive = ArchiveFile::parse(data)?;
    for entry in archive.members() {
        let entry = entry?;
        m.update(entry.name());
        m.update(entry.data(data)?);
    }
    Ok(())
}

/// Format `duration` as seconds with a fractional component.
pub fn fmt_duration_as_secs(duration: &Duration) -> String {
    format!("{}.{:03} s", duration.as_secs(), duration.subsec_millis())
}

fn wait_with_input_buffer_stderr<T>(
    mut child: T,
    input: Option<Vec<u8>>,
) -> Result<(
    impl std::future::Future<Output = Result<process::ExitStatus>>,
    <T as CommandChild>::O,                             // stdout
    impl std::future::Future<Output = Result<Vec<u8>>>, // stderr
)>
where
    T: CommandChild + 'static,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let stdin = input.and_then(|i| {
        child.take_stdin().map(|mut stdin| async move {
            stdin.write_all(&i).await.context("failed to write stdin")
        })
    });

    let stdout = child.take_stdout().unwrap();
    let stderr = child.take_stderr();
    let stderr = async move {
        let mut buf = Vec::new();
        if let Some(mut stderr) = stderr {
            stderr
                .read_to_end(&mut buf)
                .await
                .context("failed to read stderr")?;
        }
        Ok(buf)
    };

    Ok((
        async move {
            // Finish writing stdin before waiting, because waiting drops stdin.
            if let Some(stdin) = stdin {
                let _ = stdin.await;
            }

            child
                .wait()
                .await
                .map(|(status, _)| status)
                .context("failed to wait for child")
        },
        stdout,
        stderr,
    ))
}

/// If `input`, write it to `child`'s stdin while also reading `child`'s stdout and stderr, then wait on `child` and return its status and output.
///
/// This was lifted from `std::process::Child::wait_with_output` and modified
/// to also write to stdin.
async fn wait_with_input_output<T>(
    mut child: T,
    input: Option<Vec<u8>>,
) -> Result<(ProcessOutput, Duration)>
where
    T: CommandChild + 'static,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let stdin = input.and_then(|i| {
        child.take_stdin().map(|mut stdin| async move {
            stdin.write_all(&i).await.context("failed to write stdin")
        })
    });
    let stdout = child.take_stdout();
    let stdout = async move {
        match stdout {
            Some(mut stdout) => {
                let mut buf = Vec::new();
                stdout
                    .read_to_end(&mut buf)
                    .await
                    .context("failed to read stdout")?;
                Result::Ok(Some(buf))
            }
            None => Ok(None),
        }
    };

    let stderr = child.take_stderr();
    let stderr = async move {
        match stderr {
            Some(mut stderr) => {
                let mut buf = Vec::new();
                stderr
                    .read_to_end(&mut buf)
                    .await
                    .context("failed to read stderr")?;
                Result::Ok(Some(buf))
            }
            None => Ok(None),
        }
    };

    // Finish writing stdin before waiting, because waiting drops stdin.
    let status = async move {
        if let Some(stdin) = stdin {
            let _ = stdin.await;
        }

        child.wait().await.context("failed to wait for child")
    };

    let ((status, duration), stdout, stderr) =
        futures::future::try_join3(status, stdout, stderr).await?;

    Ok((
        process::Output {
            status,
            stdout: stdout.unwrap_or_default(),
            stderr: stderr.unwrap_or_default(),
        }
        .into(),
        duration,
    ))
}

/// Run `command`, writing `input` to its stdin if it is `Some` and return the exit status and output.
///
/// If the command returns a non-successful exit status, an error of `SccacheError::ProcessError`
/// will be returned containing the process output.
pub async fn run_input_output<C>(mut command: C, input: Option<Vec<u8>>) -> Result<ProcessOutput>
where
    C: RunCommand,
{
    let child = command
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::inherit()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .await?;

    wait_with_input_output(child, input)
        .await
        .and_then(|(output, _)| output.into())
}

/// Run `command`, writing `input` to its stdin if it is `Some` and return the exit status and output.
///
/// If the command returns a non-successful exit status, an error of `SccacheError::ProcessError`
/// will be returned containing the process output.
pub async fn run_input_output_timed<C>(
    mut command: C,
    input: Option<Vec<u8>>,
) -> std::result::Result<(ProcessOutput, Duration), (anyhow::Error, Duration)>
where
    C: RunCommand,
{
    let child = command
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::inherit()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .await
        .map_err(|err| (err, Duration::default()))?;

    match wait_with_input_output(child, input).await {
        Ok((out, duration)) => {
            if out.success() {
                Ok((out, duration))
            } else {
                Err((ProcessError(out).into(), duration))
            }
        }
        Err(err) => Err((err, Duration::default())),
    }
}

/// Run a command and return a Stream of Result<BytesMut>.
///
/// * Each Ok(BytesMut) value is a chunk of the child process's stdout.
/// * An Err(anyhow::Error) may wrap an io::Error or a ProcessError.
/// * An Err(ProcessError) value is the exit status and buffered stderr.
///
/// If the process exits cleanly, the stream will complete without yielding
/// any Err results. If launching the process fails, the stream will yield
/// an Err(anyhow::Error) that wraps the io::Error, and then complete. If
/// the process launches successfully and exits uncleanly, the stream will
/// yield a Err(ProcessError) and then complete.
///
/// This method allows computing a rolling value (such as a hash) over stdout
/// without needing to buffer it all in memory, which runs the risk of OOM in
/// high-concurrency scenarios.
pub async fn run_input_stream_output<C>(
    command: C,
    batch_size: usize,
    input: Option<Vec<u8>>,
) -> Result<impl futures::Stream<Item = Result<BytesMut>>>
where
    C: RunCommand + 'static,
{
    let (status, stdout, stderr) = run_with_input_buffer_stderr(command, input).await?;

    let output = async move {
        match tokio::try_join!(status, stderr) {
            Err(err) => Err(err),
            Ok((status, stderr)) => ProcessOutput {
                status: status.into(),
                stderr,
                ..Default::default()
            }
            .into(),
        }
        .map_err(Arc::new)
    }
    .shared();

    Ok(make_process_output_stream(output, stdout, batch_size))
}

pub fn make_process_output_stream<O, R>(
    output: futures::future::Shared<O>,
    stdout: R,
    batch_size: usize,
) -> impl futures::Stream<Item = Result<BytesMut>>
where
    O: Future<Output = std::result::Result<ProcessOutput, Arc<Error>>>,
    R: tokio::io::AsyncRead + Send,
{
    let stdout =
        Box::pin(read_line_batches(stdout, batch_size)).take_until(output.clone().and_then(|_| {
            // Take stdout until output resolves to an error.
            // If output resolves to Ok(), don't interrupt stdout.
            futures::future::pending::<std::result::Result<(), Arc<anyhow::Error>>>()
        }));

    async_stream::try_stream! {
        // Yield each stdout bytes chunk
        for await bytes in stdout {
            yield bytes?;
        }

        // Unwrap the ProcessOutput result.
        // The stdout stream may finish before the output future resolves, so
        // wait for the output future result and unwrap the Err's Arc so we
        // return a result containing the original anyhow::Error.
        yield output.await
            .or_else(|err| Arc::into_inner(err).map(Err).unwrap_or_else(|| Ok(Default::default())))
            .map(|_| BytesMut::new())?;
    }
}

pub fn read_line_batches<R>(
    source: R,
    batch_size: usize,
) -> impl futures::Stream<Item = Result<BytesMut>>
where
    R: tokio::io::AsyncRead + Send,
{
    let stream = FramedRead::with_capacity(source, LineBatchesCodec::new(batch_size), batch_size);
    async_stream::try_stream! {
        for await batch in stream {
            yield batch?;
        }
    }
}

///
/// Similar to tokio_util::codec::LinesCodec, but yields batches of lines instead of a single line.
/// Line batches include their trailing carriage returns and/or newlines.
///
struct LineBatchesCodec {
    batch_size: usize,
    bytes_read: usize,
    finder: memchr::memmem::Finder<'static>,
}

impl LineBatchesCodec {
    fn new(batch_size: usize) -> LineBatchesCodec {
        LineBatchesCodec {
            batch_size,
            bytes_read: 0,
            finder: memchr::memmem::Finder::new("\n").into_owned(),
        }
    }
}

impl Decoder for LineBatchesCodec {
    type Item = BytesMut;
    type Error = anyhow::Error;

    fn decode(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>> {
        let start = self.bytes_read;
        let size = self.batch_size.min(buf.len());

        for offset in self.finder.find_iter(&buf[start..]) {
            // Found a line!
            let pos = start + offset + 1;
            if pos >= size {
                // Filled the batch, so split after the last trailing newline
                self.bytes_read = 0;
                return Ok(Some(buf.split_to(pos)));
            }
        }

        // Didn't fill the batch or didn't find a newline.
        // Resume searching at the current offset next time.
        self.bytes_read = buf.len();

        Ok(None)
    }

    fn decode_eof(&mut self, buf: &mut BytesMut) -> Result<Option<Self::Item>> {
        match self.decode(buf)? {
            Some(frame) => Ok(Some(frame)),
            None => {
                self.bytes_read = 0;
                // No terminating newline - return remaining data, if any
                if buf.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(buf.split_to(buf.len())))
                }
            }
        }
    }
}

pub async fn run_with_input_buffer_stderr<C>(
    mut command: C,
    input: Option<Vec<u8>>,
) -> Result<(
    impl std::future::Future<Output = Result<process::ExitStatus>>,
    <C::C as CommandChild>::O,                          // stdout
    impl std::future::Future<Output = Result<Vec<u8>>>, // stderr
)>
where
    C: RunCommand,
{
    let child = command
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::inherit()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .await?;

    wait_with_input_buffer_stderr(child, input)
}

/// Write `data` to `writer` with bincode serialization, prefixed by a `u32` length.
pub fn write_length_prefixed_bincode<W, S>(mut writer: W, data: S) -> Result<()>
where
    W: Write,
    S: Serialize,
{
    let bytes = bincode::serialize(&data)?;
    let mut len = [0; 4];
    BigEndian::write_u32(&mut len, bytes.len() as u32);
    writer.write_all(&len)?;
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}

pub trait OsStrExt {
    fn find<P: AsRef<OsStr>>(&self, pat: P) -> Option<usize>;
    fn contains<P: AsRef<OsStr>>(&self, pat: P) -> bool;
    fn ends_with<P: AsRef<OsStr>>(&self, pat: P) -> bool;
    fn starts_with<P: AsRef<OsStr>>(&self, pat: P) -> bool;
    fn split<P: AsRef<OsStr>>(&self, pat: P) -> impl Iterator<Item = &'_ OsStr>;
    fn split_once<P: AsRef<OsStr>>(&self, pat: P) -> Option<(&'_ OsStr, &'_ OsStr)>;
    fn strip_prefix<P: AsRef<OsStr>>(&self, pat: P) -> Option<&'_ OsStr>;
    fn strip_suffix<P: AsRef<OsStr>>(&self, pat: P) -> Option<&'_ OsStr>;
    fn trim(&self) -> &OsStr;
    fn trim_start(&self) -> &OsStr;
    fn trim_end(&self) -> &OsStr;
    fn trim_start_matches<P: AsRef<OsStr>>(&self, pat: P) -> &OsStr;
    fn trim_end_matches<P: AsRef<OsStr>>(&self, pat: P) -> &OsStr;
}

impl OsStrExt for OsStr {
    fn contains<P: AsRef<OsStr>>(&self, pat: P) -> bool {
        self.find(pat).is_some()
    }

    fn find<P: AsRef<OsStr>>(&self, pat: P) -> Option<usize> {
        let p = pat.as_ref().as_encoded_bytes();
        let s = self.as_encoded_bytes();
        let (m, n) = (s.len(), p.len());
        if m < n {
            None
        } else {
            s.iter()
                .enumerate()
                .map(|(i, _)| i)
                .find(|&i| p == &s[i..(i + n).min(m)])
        }
    }

    fn ends_with<P: AsRef<OsStr>>(&self, pat: P) -> bool {
        let p = pat.as_ref().as_encoded_bytes();
        let s = self.as_encoded_bytes();
        let (m, n) = (s.len(), p.len());
        if m < n { false } else { p == &s[m - n..] }
    }

    fn starts_with<P: AsRef<OsStr>>(&self, pat: P) -> bool {
        let p = pat.as_ref().as_encoded_bytes();
        let s = self.as_encoded_bytes();
        let (m, n) = (s.len(), p.len());
        if m < n { false } else { p == &s[..n] }
    }

    fn split<P: AsRef<OsStr>>(&self, pat: P) -> impl Iterator<Item = &'_ OsStr> {
        struct Split<'a> {
            rest: Option<&'a OsStr>,
            delim: OsString,
        }
        impl<'a> Iterator for Split<'a> {
            type Item = &'a OsStr;
            fn next(&mut self) -> Option<&'a OsStr> {
                if let Some(rest) = self.rest {
                    if let Some((lhs, rhs)) = rest.split_once(&self.delim) {
                        self.rest.replace(rhs);
                        Some(lhs)
                    } else {
                        self.rest.take()
                    }
                } else {
                    None
                }
            }
        }
        Split::<'_> {
            rest: Some(self),
            delim: pat.as_ref().to_owned(),
        }
    }

    fn split_once<P: AsRef<OsStr>>(&self, pat: P) -> Option<(&'_ OsStr, &'_ OsStr)> {
        self.find(pat.as_ref()).map(|idx| {
            let p = pat.as_ref().as_encoded_bytes();
            let s = self.as_encoded_bytes();
            let a = &s[..idx];
            let b = &s[idx + p.len()..];
            (
                unsafe { OsStr::from_encoded_bytes_unchecked(a) }, //
                unsafe { OsStr::from_encoded_bytes_unchecked(b) },
            )
        })
    }

    fn strip_prefix<P: AsRef<OsStr>>(&self, pat: P) -> Option<&'_ OsStr> {
        self.starts_with(pat.as_ref()).then(|| {
            let p = pat.as_ref().as_encoded_bytes();
            let s = self.as_encoded_bytes();
            let b = &s[p.len()..];
            unsafe { OsStr::from_encoded_bytes_unchecked(b) }
        })
    }

    fn strip_suffix<P: AsRef<OsStr>>(&self, pat: P) -> Option<&'_ OsStr> {
        self.ends_with(pat.as_ref()).then(|| {
            let p = pat.as_ref().as_encoded_bytes();
            let s = self.as_encoded_bytes();
            let b = &s[..s.len() - p.len()];
            unsafe { OsStr::from_encoded_bytes_unchecked(b) }
        })
    }

    fn trim(&self) -> &OsStr {
        self.trim_start().trim_end()
    }

    fn trim_start(&self) -> &OsStr {
        let mut buf = self.as_encoded_bytes();

        while !buf.is_empty()
            && char::from_u32(buf[1] as u32)
                .filter(|c| c.is_whitespace())
                .is_some()
        {
            buf = &buf[1..];
        }

        unsafe { OsStr::from_encoded_bytes_unchecked(buf) }
    }

    fn trim_end(&self) -> &OsStr {
        let mut buf = self.as_encoded_bytes();

        while !buf.is_empty()
            && let idx = buf.len() - 1
            && char::from_u32(buf[idx] as u32)
                .filter(|c| c.is_whitespace())
                .is_some()
        {
            buf = &buf[..idx];
        }

        unsafe { OsStr::from_encoded_bytes_unchecked(buf) }
    }

    fn trim_start_matches<P: AsRef<OsStr>>(&self, pat: P) -> &OsStr {
        let pat = pat.as_ref().as_encoded_bytes();
        let mut buf = self.as_encoded_bytes();
        loop {
            if pat.len() > buf.len() || &buf[..pat.len()] != pat {
                break;
            } else {
                buf = &buf[pat.len()..];
            }
        }
        unsafe { OsStr::from_encoded_bytes_unchecked(buf) }
    }

    fn trim_end_matches<P: AsRef<OsStr>>(&self, pat: P) -> &OsStr {
        let pat = pat.as_ref().as_encoded_bytes();
        let mut buf = self.as_encoded_bytes();
        loop {
            if pat.len() > buf.len() || &buf[buf.len() - pat.len()..] != pat {
                break;
            } else {
                buf = &buf[..buf.len() - pat.len()];
            }
        }
        unsafe { OsStr::from_encoded_bytes_unchecked(buf) }
    }
}

pub fn split_quoted_shell_str<S: AsRef<str>>(s: S) -> Option<Vec<String>> {
    #[cfg(unix)]
    let args = shlex::split(s.as_ref());
    #[cfg(windows)]
    let args = Some(winsplit::split(s.as_ref()));
    args
}

pub fn path_to_str<'a, B: Into<Cow<'a, Path>>>(p: B) -> std::io::Result<Cow<'a, str>> {
    path_to_os_str(p)
        .and_then(os_str_to_bytes)
        .and_then(bytes_to_str)
}

pub fn bytes_to_path<'a, B: Into<Cow<'a, [u8]>>>(b: B) -> std::io::Result<Cow<'a, Path>> {
    bytes_to_str(b).map(|s| match s {
        Cow::Borrowed(s) => Cow::Borrowed(Path::new(s)),
        Cow::Owned(s) => Cow::Owned(PathBuf::from(s)),
    })
}

pub fn path_to_bytes<'a, B: Into<Cow<'a, Path>>>(p: B) -> std::io::Result<Cow<'a, [u8]>> {
    path_to_str(p).map(|s| match s {
        Cow::Borrowed(s) => Cow::Borrowed(s.as_bytes()),
        Cow::Owned(s) => Cow::Owned(s.into_bytes()),
    })
}

pub fn path_to_os_str<'a, B: Into<Cow<'a, Path>>>(p: B) -> std::io::Result<Cow<'a, OsStr>> {
    match p.into() {
        Cow::Borrowed(p) => Ok(Cow::Borrowed(p.as_os_str())),
        Cow::Owned(p) => Ok(Cow::Owned(p.into_os_string())),
    }
}

pub fn os_str_to_str<'a, B: Into<Cow<'a, OsStr>>>(s: B) -> std::io::Result<Cow<'a, str>> {
    os_str_to_bytes(s).and_then(bytes_to_str)
}

#[cfg(unix)]
pub fn bytes_to_str<'a, B: Into<Cow<'a, [u8]>>>(b: B) -> std::io::Result<Cow<'a, str>> {
    match b.into() {
        Cow::Borrowed(b) => str::from_utf8(b)
            .map(Cow::Borrowed)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e)),
        Cow::Owned(b) => String::from_utf8(b)
            .map(Cow::Owned)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e)),
    }
}

#[cfg(windows)]
pub fn bytes_to_str<'a, B: Into<Cow<'a, [u8]>>>(b: B) -> std::io::Result<Cow<'a, str>> {
    use windows_sys::Win32::Globalization::{CP_OEMCP, MB_ERR_INVALID_CHARS};

    multi_byte_to_wide_char(CP_OEMCP, MB_ERR_INVALID_CHARS, b.into().as_ref())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
        .and_then(|buf| {
            String::from_utf16(&buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
                .map(Cow::Owned)
        })
}

#[cfg(unix)]
pub fn bytes_to_os_str<'a, B: Into<Cow<'a, [u8]>>>(b: B) -> std::io::Result<Cow<'a, OsStr>> {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    match b.into() {
        Cow::Borrowed(b) => Ok(Cow::Borrowed(OsStr::from_bytes(b))),
        Cow::Owned(b) => Ok(Cow::Owned(OsString::from_vec(b))),
    }
}

#[cfg(windows)]
pub fn bytes_to_os_str<'a, B: Into<Cow<'a, [u8]>>>(b: B) -> std::io::Result<Cow<'a, OsStr>> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Globalization::{CP_OEMCP, MB_ERR_INVALID_CHARS};
    multi_byte_to_wide_char(CP_OEMCP, MB_ERR_INVALID_CHARS, b.into().as_ref())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))
        .map(|buf| OsString::from_wide(&buf))
        .map(Cow::Owned)
}

#[cfg(unix)]
pub fn os_str_to_bytes<'a, B: Into<Cow<'a, OsStr>>>(s: B) -> std::io::Result<Cow<'a, [u8]>> {
    match s.into() {
        Cow::Borrowed(s) => Ok(Cow::Borrowed(s.as_encoded_bytes())),
        Cow::Owned(s) => Ok(Cow::Owned(s.as_encoded_bytes().to_vec())),
    }
}

#[cfg(windows)]
pub fn os_str_to_bytes<'a, B: Into<Cow<'a, OsStr>>>(s: B) -> std::io::Result<Cow<'a, [u8]>> {
    use std::os::windows::ffi::OsStrExt;
    wide_char_to_multi_byte(&s.into().encode_wide().collect::<Vec<_>>()).map(Cow::Owned)
}

#[cfg(windows)]
pub fn wide_char_to_multi_byte(wide_char_str: &[u16]) -> std::io::Result<Vec<u8>> {
    use windows_sys::Win32::Globalization::{CP_OEMCP, WideCharToMultiByte};

    let codepage = CP_OEMCP;
    let flags = 0;
    // Empty string
    if wide_char_str.is_empty() {
        return Ok(Vec::new());
    }
    unsafe {
        // Get length of multibyte string
        let len = WideCharToMultiByte(
            codepage,
            flags,
            wide_char_str.as_ptr(),
            wide_char_str.len() as i32,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
            std::ptr::null_mut(),
        );

        if len > 0 {
            // Convert from UTF-16 to multibyte
            let mut astr: Vec<u8> = Vec::with_capacity(len as usize);
            let len = WideCharToMultiByte(
                codepage,
                flags,
                wide_char_str.as_ptr(),
                wide_char_str.len() as i32,
                astr.as_mut_ptr().cast(),
                len,
                std::ptr::null(),
                std::ptr::null_mut(),
            );
            if len > 0 {
                astr.set_len(len as usize);
                if (len as usize) == astr.len() {
                    return Ok(astr);
                } else {
                    return Ok(astr[..(len as usize)].to_vec());
                }
            }
        }
        Err(std::io::Error::last_os_error())
    }
}

/// Wrapper for MultiByteToWideChar.
///
/// See https://msdn.microsoft.com/en-us/library/windows/desktop/dd319072(v=vs.85).aspx
/// for more details.
#[cfg(windows)]
fn multi_byte_to_wide_char<B: AsRef<[u8]>>(
    codepage: u32,
    flags: u32,
    multi_byte_str: B,
) -> std::io::Result<Vec<u16>> {
    use windows_sys::Win32::Globalization::MultiByteToWideChar;

    let multi_byte_str = multi_byte_str.as_ref();

    if multi_byte_str.is_empty() {
        return Ok(vec![]);
    }
    unsafe {
        // Get length of UTF-16 string
        let len = MultiByteToWideChar(
            codepage,
            flags,
            multi_byte_str.as_ptr().cast(),
            multi_byte_str.len() as i32,
            std::ptr::null_mut(),
            0,
        );
        if len > 0 {
            // Convert to UTF-16
            let mut wstr: Vec<u16> = Vec::with_capacity(len as usize);
            let len = MultiByteToWideChar(
                codepage,
                flags,
                multi_byte_str.as_ptr().cast(),
                multi_byte_str.len() as i32,
                wstr.as_mut_ptr().cast(),
                len,
            );
            if len > 0 {
                wstr.set_len(len as usize);
                return Ok(wstr);
            }
        }
        Err(std::io::Error::last_os_error())
    }
}

/// A Unix timestamp with nanoseconds precision
#[derive(Serialize, Deserialize, Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    seconds: i64,
    /// Always in the `0 .. 1_000_000_000` range.
    nanoseconds: u32,
}

const NSEC_PER_SEC: u32 = 1_000_000_000;

impl From<std::time::SystemTime> for Timestamp {
    fn from(system_time: std::time::SystemTime) -> Self {
        // On Unix, `SystemTime` is a wrapper for the `timespec` C struct:
        // https://www.gnu.org/software/libc/manual/html_node/Time-Types.html#index-struct-timespec
        // On Windows, `SystemTime` wraps a 100ns intervals-based struct.
        // We want to effectively access the inner fields, but the Rust standard
        // library does not expose them. The best we can do is:
        let seconds;
        let nanoseconds;
        match system_time.duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => {
                seconds = duration.as_secs() as i64;
                nanoseconds = duration.subsec_nanos();
            }
            Err(error) => {
                // `system_time` is before `UNIX_EPOCH`.
                // We need to undo this algorithm:
                // https://github.com/rust-lang/rust/blob/6bed1f0bc3cc50c10aab26d5f94b16a00776b8a5/library/std/src/sys/unix/time.rs#L40-L41
                let negative = error.duration();
                let negative_secs = negative.as_secs() as i64;
                let negative_nanos = negative.subsec_nanos();
                if negative_nanos == 0 {
                    seconds = -negative_secs;
                    nanoseconds = 0;
                } else {
                    // For example if `system_time` was 4.3 seconds before
                    // the Unix epoch we get a Duration that represents
                    // `(-4, -0.3)` but we want `(-5, +0.7)`:
                    seconds = -1 - negative_secs;
                    nanoseconds = NSEC_PER_SEC - negative_nanos;
                }
            }
        }
        Self {
            seconds,
            nanoseconds,
        }
    }
}

impl PartialEq<SystemTime> for Timestamp {
    fn eq(&self, other: &SystemTime) -> bool {
        self == &Self::from(*other)
    }
}

impl Timestamp {
    pub fn new(seconds: i64, nanoseconds: u32) -> Self {
        Self {
            seconds,
            nanoseconds,
        }
    }

    pub fn to_utc(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        Some(chrono::DateTime::from_timestamp(self.seconds, self.nanoseconds)?.to_utc())
    }
}

/// Adds a fallback for trying Unix's `ctime` semantics on Windows systems.
pub trait MetadataCtimeExt {
    fn ctime_or_creation(&self) -> std::io::Result<Timestamp>;
}

impl MetadataCtimeExt for std::fs::Metadata {
    #[cfg(unix)]
    fn ctime_or_creation(&self) -> std::io::Result<Timestamp> {
        use std::os::unix::prelude::MetadataExt;
        Ok(Timestamp {
            seconds: self.ctime(),
            nanoseconds: self.ctime_nsec().try_into().unwrap_or(0),
        })
    }
    #[cfg(windows)]
    fn ctime_or_creation(&self) -> std::io::Result<Timestamp> {
        // Windows does not have the actual notion of ctime in the Unix sense.
        // Best effort is creation time (also called ctime in windows libs...)
        self.created().map(Into::into)
    }
}

pub struct HashToDigest<'a> {
    pub digest: &'a mut Digest,
}

impl Hasher for HashToDigest<'_> {
    fn write(&mut self, bytes: &[u8]) {
        self.digest.update(bytes);
    }

    fn finish(&self) -> u64 {
        panic!("not supposed to be called");
    }
}

pub fn tempdir_in<P: AsRef<Path>>(root: P) -> Result<tempfile::TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.rand_bytes(16);
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    builder.permissions(std::fs::Permissions::from_mode(0o777));
    builder
        .tempdir_in(root.as_ref())
        .map_err(anyhow::Error::new)
}

pub fn tempdir_with_prefix_in<P: AsRef<Path>>(root: P, prefix: &str) -> Result<tempfile::TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.rand_bytes(16);
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    builder.permissions(std::fs::Permissions::from_mode(0o777));
    builder
        .prefix(prefix)
        .tempdir_in(root.as_ref())
        .map_err(anyhow::Error::new)
}

pub fn tempfile_in<P: AsRef<Path>>(root: P) -> Result<tempfile::NamedTempFile> {
    let mut builder = tempfile::Builder::new();
    builder.rand_bytes(16);
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    builder.permissions(std::fs::Permissions::from_mode(0o666));
    builder
        .tempfile_in(root.as_ref())
        .map_err(anyhow::Error::new)
}

pub fn tempfile_with_prefix_in<P: AsRef<Path>>(
    root: P,
    prefix: &str,
) -> Result<tempfile::NamedTempFile> {
    let mut builder = tempfile::Builder::new();
    builder.rand_bytes(16);
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    builder.permissions(std::fs::Permissions::from_mode(0o666));
    builder
        .prefix(prefix)
        .tempfile_in(root.as_ref())
        .map_err(anyhow::Error::new)
}

pub static SCCACHE_TMPDIR: LazyLock<std::result::Result<PathBuf, std::io::Error>> =
    LazyLock::new(|| {
        let tmpdir = std::env::temp_dir().join("sccache");
        std::fs::create_dir_all(&tmpdir).map(|_| tmpdir)
    });

pub fn temp_dir() -> Result<tempfile::TempDir> {
    SCCACHE_TMPDIR
        .as_ref()
        .map_err(anyhow::Error::new)
        .and_then(tempdir_in)
}

pub fn tempfile() -> Result<tempfile::NamedTempFile> {
    SCCACHE_TMPDIR
        .as_ref()
        .map_err(anyhow::Error::new)
        .and_then(tempfile_in)
}

pub fn temppath() -> Result<tempfile::TempPath> {
    tempfile().map(|p| p.into_temp_path())
}

#[cfg(not(windows))]
pub fn get_process_fds<F>(keep: F) -> Option<Vec<RawFd>>
where
    F: Fn(RawFd) -> bool,
{
    // macOS/BSD: /dev/fd
    std::fs::read_dir("/dev/fd")
        // Linux: /proc/self/fd
        .or_else(|_| std::fs::read_dir("/proc/self/fd"))
        .ok()
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| e.file_name().to_str()?.parse::<RawFd>().ok())
                .filter(|fd| !keep(*fd))
                .collect()
        })
}

/// Close every file descriptor we inherited from whoever spawned us, keeping
/// stdin/out/err and anything in `preserve`.
#[cfg(not(windows))]
fn close_inherited_fds(preserve: &[RawFd]) {
    let keep = |fd: RawFd| fd <= libc::STDERR_FILENO || preserve.contains(&fd);

    let victims = get_process_fds(keep);

    match victims {
        Some(fds) => {
            for fd in fds {
                unsafe { libc::close(fd) };
            }
        }
        // No fd directory to enumerate, so fall back to sweeping the whole
        // range. Bounded by the soft limit rather than the hard one to keep
        // this from turning into a million syscalls.
        None => {
            let max = unsafe { libc::sysconf(libc::_SC_OPEN_MAX) };
            let max = if max < 0 {
                4096
            } else {
                max.min(65536) as RawFd
            };
            for fd in (libc::STDERR_FILENO + 1)..max {
                if !keep(fd) {
                    unsafe { libc::close(fd) };
                }
            }
        }
    }
}

/// Pipe `cmd`'s stdio to `/dev/null`, unless a specific env var is set.
///
/// `preserve_fds` lists descriptors the caller opened before daemonizing and
/// still needs afterwards; everything else inherited is closed.
#[cfg(not(windows))]
pub fn daemonize(preserve_fds: &[std::os::unix::io::RawFd]) -> Result<()> {
    use daemonix::Daemonize;
    use std::env;
    use std::mem;

    match env::var("SCCACHE_NO_DAEMON") {
        Ok(ref val) if val == "1" => {}
        _ => {
            // umask() gets the current mask and sets a new one.
            let mask = rustix::process::umask(rustix::fs::Mode::from_bits_truncate(0));
            // set it back
            rustix::process::umask(mask);
            #[cfg(target_os = "linux")]
            let mut daemon = Daemonize::new().umask(mask.bits());
            #[cfg(not(target_os = "linux"))]
            let mut daemon = Daemonize::new().umask(mask.bits() as u32);

            if let Some(pidfile) = env::var_os("SCCACHE_PID_FILE") {
                daemon = daemon.pid_file(pidfile);
            }

            daemon.start().context("failed to daemonize")?;
        }
    }

    close_inherited_fds(preserve_fds);

    static mut PREV_SIGSEGV: *mut libc::sigaction = std::ptr::null_mut();
    static mut PREV_SIGBUS: *mut libc::sigaction = std::ptr::null_mut();
    static mut PREV_SIGILL: *mut libc::sigaction = std::ptr::null_mut();

    // We don't have a parent process any more once we've reached this point,
    // which means that no one's probably listening for our exit status.
    // In order to assist with debugging crashes of the server we configure our
    // rlimit to allow runtime dumps and we also install a signal handler for
    // segfaults which at least prints out what just happened.
    unsafe {
        match env::var("SCCACHE_ALLOW_CORE_DUMPS") {
            Ok(ref val) if val == "1" => {
                let rlim = libc::rlimit {
                    rlim_cur: libc::RLIM_INFINITY,
                    rlim_max: libc::RLIM_INFINITY,
                };
                libc::setrlimit(libc::RLIMIT_CORE, &rlim);
            }
            _ => {}
        }

        PREV_SIGSEGV = Box::into_raw(Box::new(mem::zeroed::<libc::sigaction>()));
        PREV_SIGBUS = Box::into_raw(Box::new(mem::zeroed::<libc::sigaction>()));
        PREV_SIGILL = Box::into_raw(Box::new(mem::zeroed::<libc::sigaction>()));
        let mut new: libc::sigaction = mem::zeroed();
        new.sa_sigaction = (handler as *const libc::c_void).expose_provenance();
        new.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
        libc::sigaction(libc::SIGSEGV, &new, &mut *PREV_SIGSEGV);
        libc::sigaction(libc::SIGBUS, &new, &mut *PREV_SIGBUS);
        libc::sigaction(libc::SIGILL, &new, &mut *PREV_SIGILL);
    }

    return Ok(());

    extern "C" fn handler(
        signum: libc::c_int,
        _info: *mut libc::siginfo_t,
        _ptr: *mut libc::c_void,
    ) {
        use std::fmt::{Result, Write};

        struct Stderr;

        impl Write for Stderr {
            fn write_str(&mut self, s: &str) -> Result {
                unsafe {
                    let bytes = s.as_bytes();
                    libc::write(libc::STDERR_FILENO, bytes.as_ptr().cast(), bytes.len());
                    Ok(())
                }
            }
        }

        unsafe {
            let _ = writeln!(Stderr, "signal {signum} received");

            // Configure the old handler and then resume the program. This'll
            // likely go on to create a runtime dump if one's configured to be
            // created.
            match signum {
                libc::SIGBUS => libc::sigaction(signum, &*PREV_SIGBUS, std::ptr::null_mut()),
                libc::SIGILL => libc::sigaction(signum, &*PREV_SIGILL, std::ptr::null_mut()),
                _ => libc::sigaction(signum, &*PREV_SIGSEGV, std::ptr::null_mut()),
            };
        }
    }
}

/// This is a no-op on Windows.
#[cfg(windows)]
pub fn daemonize(_preserve_fds: &[std::os::windows::io::RawHandle]) -> Result<()> {
    Ok(())
}

#[cfg(any(feature = "dist-server", feature = "dist-client"))]
pub fn new_reqwest_client<'a, C>(config: C) -> reqwest::Client
where
    C: Into<Option<&'a crate::config::client::dist::Networking>>,
{
    let builder = reqwest::Client::builder()
        .user_agent(format!(
            "{}/{}",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ))
        // HTTP/2
        .http2_adaptive_window(true)
        // Prefer HTTP/2
        .http2_prior_knowledge();

    // Query the native store; fallback if it's completely empty
    let builder = if rustls_native_certs::load_native_certs().certs.is_empty() {
        builder.tls_certs_only(Vec::new())
    } else {
        builder
    };

    let builder = if let Some(config) = config.into() {
        let request_timeout = Duration::from_secs(config.request_timeout as u64);
        let connect_timeout = Duration::from_secs(config.connect_timeout as u64);
        let keepalive_timeout = Duration::from_secs(config.keepalive.timeout);
        let keepalive_interval = Duration::from_secs(config.keepalive.interval);
        let keepalive = config.keepalive.enabled;

        let builder = builder
            // Timeouts
            .timeout(request_timeout)
            .connect_timeout(connect_timeout)
            // Keepalive
            .http2_keep_alive_while_idle(keepalive)
            .http2_keep_alive_timeout(keepalive_timeout)
            .tcp_keepalive_retries(keepalive.then_some(3))
            .tcp_keepalive(keepalive.then_some(keepalive_timeout))
            .tcp_keepalive_interval(keepalive.then_some(keepalive_interval))
            .http2_keep_alive_interval(keepalive.then_some(keepalive_interval));

        #[cfg(target_os = "linux")]
        let builder = builder.tcp_user_timeout(keepalive.then_some(keepalive_timeout));

        // Connection pool
        if config.connection_pool {
            // This has to be at least as long as `request_timeout`, otherwise
            // reqwest will close idle connections before build jobs are done.
            //
            // Users should set their load balancer's idle timeout to the same
            // value as `request_timeout` (AWS's ALB default is 60s).
            builder.pool_idle_timeout(request_timeout)
        } else {
            // Disable connection pool
            builder
                .pool_max_idle_per_host(0)
                .pool_idle_timeout(Duration::from_secs(0))
        }
    } else {
        builder
    };

    builder
        .build()
        .expect("http client must build with success")
}

pub fn spawn<F>(future: F) -> tokio_util::task::AbortOnDropHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    tokio_util::task::AbortOnDropHandle::new(tokio::spawn(future))
}

pub fn spawn_on<F>(
    handle: &tokio::runtime::Handle,
    future: F,
) -> tokio_util::task::AbortOnDropHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    tokio_util::task::AbortOnDropHandle::new(handle.spawn(future))
}

fn unhex(b: u8) -> std::io::Result<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid hex digit",
        )),
    }
}

/// A reverse version of std::ascii::escape_default
pub fn ascii_unescape_default(s: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() + 4);
    let mut offset = 0;
    while offset < s.len() {
        let c = s[offset];
        if c == b'\\' {
            offset += 1;
            if offset >= s.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "incomplete escape",
                ));
            }
            let c = s[offset];
            match c {
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'\'' => out.push(b'\''),
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'x' => {
                    offset += 1;
                    if offset + 1 >= s.len() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "incomplete hex escape",
                        ));
                    }
                    let v = (unhex(s[offset])? << 4) | unhex(s[offset + 1])?;
                    out.push(v);
                    offset += 1;
                }
                _ => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "invalid escape",
                    ));
                }
            }
        } else {
            out.push(c);
        }
        offset += 1;
    }
    Ok(out)
}

pub fn num_cpus() -> usize {
    let num_cpus = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    if let Some(req_cpus) = std::env::var("SCCACHE_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        req_cpus.max(1)
    } else if let Some(percent) = std::env::var("SCCACHE_THREADS_PERCENT")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|percent| percent / 100.0)
    {
        ((percent * (num_cpus as f64)).floor() as usize).max(1)
    } else {
        num_cpus
    }
}

pub async fn retry_with_jitter<F>(limit: usize, func: F) -> std::result::Result<F::Item, F::Error>
where
    F: tokio_retry2::Action,
{
    Retry::spawn(
        FibonacciBackoff::from_millis(1000) // wait 1s before retrying
            .max_delay_millis(60000) // set max interval to 60 seconds
            .map(tokio_retry2::strategy::jitter) // add jitter to the retry interval
            .take(limit), // limit retries
        func,
    )
    .await
}

#[async_trait]
pub trait AsyncMulticastFunc<K: AsyncMulticastArgs, V> {
    async fn call(&self, args: &K) -> Result<V>;
}

pub trait AsyncMulticastArgs {
    type Key: Clone + Eq + Hash;
    fn hash(&self) -> Self::Key;
}

#[derive(Clone)]
pub struct AsyncMulticast<K: AsyncMulticastArgs, V> {
    run_f: Arc<dyn AsyncMulticastFunc<K, V> + Send + Sync>,
    state: Arc<
        Mutex<
            HashMap<
                K::Key,
                tokio::sync::broadcast::Sender<std::result::Result<(K, V), Arc<anyhow::Error>>>,
            >,
        >,
    >,
}

impl<K, V> AsyncMulticast<K, V>
where
    K: AsyncMulticastArgs + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new<F>(run_f: F) -> Self
    where
        F: AsyncMulticastFunc<K, V> + Send + Sync + 'static,
    {
        Self {
            run_f: Arc::new(run_f),
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn call(&self, args: K) -> Result<(K, V)>
    where
        K::Key: Send,
    {
        // Lock state
        let mut state = self.state.lock().await;
        // Compute the key
        let hash = args.hash();

        let mut recv = if let Some(sndr) = state.get(&hash) {
            // Return shared broadcast receiver if pending
            sndr.subscribe()
        } else {
            // Create broadcast sender/receiver on first call
            let (sndr, recv) = tokio::sync::broadcast::channel(1);

            state.insert(hash.clone(), sndr);

            // Run the function on a worker thread. Boxed so every `AsyncMulticast<K, V>`
            // shares one tokio task type instead of instantiating tokio's task harness for each.
            let task: futures::future::BoxFuture<'static, ()> = Box::pin({
                let run_f = self.run_f.clone();
                let state = self.state.clone();
                async move {
                    // Call the function
                    let res = run_f.call(&args).await;
                    // Remove the sender
                    let sndr = state.lock().await.remove(&hash);
                    // Unwrap the result
                    let res = match res {
                        Ok(val) => Ok((args, val)),
                        Err(err) => Err(Arc::new(err)),
                    };
                    // Notify receivers
                    if let Some(sndr) = sndr {
                        let _ = sndr.send(res);
                    }
                }
            });
            tokio::spawn(task);

            recv
        };

        // Unlock state while we await the receiver
        drop(state);

        recv.recv()
            .await
            .map_err(anyhow::Error::new)
            .and_then(|res| res.map_err(|e| anyhow!(e)))
    }
}

/// Simplify a path to one without any relative components, erroring if it looks
/// like there could be any symlink complexity that means a simplified path is not
/// equivalent to the original (see the documentation of `fs::canonicalize` for an
/// example).
///
/// So why avoid resolving symlinks? Any path that we are trying to simplify has
/// (usually) been added to an archive because something will try access it, but
/// resolving symlinks (be they for the actual file or directory components) can
/// make the accessed path 'disappear' in favour of the canonical path.
pub fn simplify_path<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
    SimplifyPath {
        dirs: None,
        resolved_symlinks: None,
    }
    .simplify(path.as_ref())
}

pub struct SimplifyPath<'a> {
    pub dirs: Option<&'a mut std::collections::BTreeSet<PathBuf>>,
    pub resolved_symlinks: Option<&'a mut std::collections::BTreeMap<PathBuf, PathBuf>>,
}

impl SimplifyPath<'_> {
    pub fn simplify<P: AsRef<Path>>(&mut self, path: P) -> Result<PathBuf> {
        use std::path::Component;

        let mut final_path = PathBuf::new();
        for component in path.as_ref().components() {
            match component {
                c @ Component::RootDir | c @ Component::Prefix(_) | c @ Component::Normal(_) => {
                    final_path.push(c);
                    if self.resolved_symlinks.is_some() && final_path.is_symlink() {
                        let parent = final_path.parent().expect("symlinks have parents");
                        let link_target = final_path.read_link()?;
                        let new_final_path = self.simplify(parent.join(&link_target))?;
                        let old_final_path =
                            std::mem::replace(&mut final_path, new_final_path.clone());
                        #[allow(clippy::unnecessary_unwrap)]
                        self.resolved_symlinks
                            .as_mut()
                            .unwrap()
                            .insert(old_final_path, new_final_path);
                    }
                }
                Component::ParentDir => {
                    // If the path is doing funny symlink traversals, just give up.
                    //
                    // This case should only occur if `resolved_symlinks` is `None`.
                    if final_path.is_symlink() {
                        bail!("Cannot handle symlinks in parent paths")
                    }
                    if let Some(dirs) = self.dirs.as_mut() {
                        dirs.insert(final_path.clone());
                    }
                    final_path.pop();
                }
                Component::CurDir => continue,
            }
        }
        Ok(final_path)
    }
}

/// Strip base directories from absolute paths in preprocessor output.
///
/// This function searches for basedir paths in the preprocessor output and
/// replaces them with relative path markers. When multiple basedirs are provided,
/// the longest matching prefix is used. This is similar to ccache's CCACHE_BASEDIR.
///
/// Path matching is case-insensitive to handle various filesystem behaviors and build system
/// configurations uniformly across all operating systems. On Windows, this function also handles
/// paths with mixed forward and backward slashes, which can occur when different build tools
/// produce preprocessor output.
///
/// Only paths that start with one of the basedirs are modified. The paths are expected to be
/// in the format found in preprocessor output (e.g., `# 1 "/path/to/file"`).
pub fn strip_basedirs<'a>(preprocessor_output: &'a [u8], basedirs: &[Vec<u8>]) -> Cow<'a, [u8]> {
    if basedirs.is_empty() || preprocessor_output.is_empty() {
        return Cow::Borrowed(preprocessor_output);
    }

    trace!(
        "Stripping basedirs from preprocessor output with length {}",
        preprocessor_output.len(),
    );

    // Find all potential matches for each basedir using fast substring search
    // Store as (position, length, basedir_idx) sorted by position
    let mut matches: Vec<(usize, usize, usize)> = Vec::new();
    // We must return the original preprocessor output on all platforms,
    // so we only normalize a copy for searching.
    #[cfg(not(target_os = "windows"))]
    let normalized_output = preprocessor_output;
    #[cfg(target_os = "windows")]
    let normalized_output = &normalize_win_path(preprocessor_output);

    for (idx, basedir_bytes) in basedirs.iter().enumerate() {
        let basedir = basedir_bytes.as_slice();

        // On Windows, preprocessor output quotes paths as C string literals
        // (#line directives and GNU linemarkers), so backslash separators
        // appear escaped: `#line 1 "C:\\dir\\file.h"`. normalize_win_path
        // converts each backslash to a forward slash, which turns every
        // escaped separator into a DOUBLE forward slash ("c://dir//file.h").
        // Search for that doubled variant of the basedir as well, otherwise
        // such paths are never stripped.
        #[cfg(target_os = "windows")]
        let doubled_basedir = double_path_separators(basedir);
        #[cfg(target_os = "windows")]
        let needles: [&[u8]; 2] = [basedir, &doubled_basedir];
        #[cfg(not(target_os = "windows"))]
        let needles: [&[u8]; 1] = [basedir];

        for needle in needles {
            // Use memchr's fast substring search
            let finder = memchr::memmem::find_iter(normalized_output, needle);

            for pos in finder {
                // Check if this is a valid boundary (start, whitespace, quote, or '<')
                let is_boundary = pos == 0
                    || normalized_output[pos - 1].is_ascii_whitespace()
                    || normalized_output[pos - 1] == b'"'
                    || normalized_output[pos - 1] == b'<';

                if is_boundary {
                    matches.push((pos, needle.len(), idx));
                }
            }
        }
    }

    if matches.is_empty() {
        return Cow::Borrowed(preprocessor_output);
    }

    // Sort matches by position, then by length descending (longest first for overlaps)
    matches.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    // Remove overlapping matches, keeping the longest match at each position
    let mut filtered_matches: Vec<(usize, usize)> = Vec::with_capacity(matches.len());
    let mut last_end = 0;

    for (pos, len, idx) in matches {
        if pos >= last_end {
            filtered_matches.push((pos, len));
            last_end = pos + len;
            trace!(
                "Matched basedir {} at position {} with length {}",
                String::from_utf8_lossy(&basedirs[idx]),
                pos,
                len
            );
        }
    }

    // Build the result in a single pass
    let mut result = Vec::with_capacity(preprocessor_output.len());
    let mut current_pos = 0;

    for (match_pos, match_len) in filtered_matches {
        // Copy everything before the match
        result.extend_from_slice(&preprocessor_output[current_pos..match_pos]);
        // Replace the basedir is removed completely, including trailing slash (it is expected, see
        // Config::basedir)
        current_pos = match_pos + match_len;
    }

    // Copy remaining data
    result.extend_from_slice(&preprocessor_output[current_pos..]);

    Cow::Owned(result)
}

pub fn apply_basedirs_to_args<P: AsRef<Path>>(
    cwd: P,
    args: &mut [OsString],
    basedirs: &[Vec<u8>],
) -> Result<()> {
    if args.is_empty() || basedirs.is_empty() {
        return Ok(());
    }

    use itertools::Itertools;
    use std::collections::{BTreeMap, BTreeSet};

    let cwd = cwd.as_ref();
    let map = {
        let mut paths = SimplifyPath {
            dirs: Some(&mut BTreeSet::new()),
            resolved_symlinks: Some(&mut BTreeMap::new()),
        };
        let cwd = paths.simplify(cwd)?;
        basedirs
            .iter()
            .map(|basedir| {
                bytes_to_path(basedir)
                    .map_err(Into::into)
                    .and_then(|path| paths.simplify(path))
                    .map(|path| diff_paths(&path, &cwd).unwrap_or(path))
                    .and_then(|path| {
                        path_to_bytes(path)
                            .map_err(Into::into)
                            .map(|path| (basedir.clone(), path))
                    })
            })
            .try_collect::<_, BTreeMap<_, _>, _>()?
    };

    let mut matches = vec![];

    for arg in args {
        let mut lhs = 0;

        // We must return the original argument on all platforms, so we only normalize a copy for searching.
        #[cfg(not(target_os = "windows"))]
        let buf: &[u8] = arg.as_encoded_bytes();
        #[cfg(target_os = "windows")]
        let normalized = normalize_win_path(arg.as_encoded_bytes());
        #[cfg(target_os = "windows")]
        let buf: &[u8] = &normalized;

        for pos in pathname_positions(buf) {
            if pos < lhs {
                continue;
            }
            lhs = pos;

            let rest = &buf[pos..];
            for basedir in basedirs.iter().rev() {
                // Basedirs carry a trailing slash, see Config::basedirs. A path
                // component can also end at the end of the argument, or at a
                // separator such as the `=` of -ffile-prefix-map.
                let base = basedir.as_slice();
                let bare = &base[..base.len() - 1];
                if rest.starts_with(base) {
                    matches.push((pos..pos + bare.len(), base));
                    lhs = pos + base.len();
                    break;
                } else if rest.starts_with(bare)
                    && matches!(rest.get(bare.len()), None | Some(b'='))
                {
                    matches.push((pos..pos + bare.len(), base));
                    lhs = pos + bare.len();
                    break;
                }
            }
        }

        if !matches.is_empty() {
            let mut lhs = 0;
            let mut new = vec![];
            for (range, dir) in matches.drain(..) {
                new.extend_from_slice(&buf[lhs..range.start]);
                if let Some(rel) = map.get(dir) {
                    new.extend_from_slice(rel);
                }
                lhs = range.end;
            }
            new.extend_from_slice(&buf[lhs..]);
            *arg = unsafe { OsString::from_encoded_bytes_unchecked(new) };
        }
    }

    Ok(())
}

// This routine is adapted from the *old* Path's `path_relative_from`
// function, which works differently from the new `relative_from` function.
// In particular, this handles the case on unix where both paths are
// absolute but with only the root as the common directory.
fn diff_paths(path: &Path, base: &Path) -> Option<PathBuf> {
    use std::path::Component;

    if path.is_absolute() != base.is_absolute() {
        if path.is_absolute() {
            Some(PathBuf::from(path))
        } else {
            None
        }
    } else {
        let mut ita = path.components();
        let mut itb = base.components();
        let mut comps: Vec<Component<'_>> = vec![];
        loop {
            match (ita.next(), itb.next()) {
                (None, None) => break,
                (Some(a), None) => {
                    comps.push(a);
                    comps.extend(ita.by_ref());
                    break;
                }
                (None, _) => comps.push(Component::ParentDir),
                (Some(a), Some(b)) if comps.is_empty() && a == b => (),
                (Some(a), Some(Component::<'_>::CurDir)) => comps.push(a),
                (Some(_), Some(Component::<'_>::ParentDir)) => return None,
                (Some(a), Some(_)) => {
                    comps.push(Component::ParentDir);
                    for _ in itb {
                        comps.push(Component::ParentDir);
                    }
                    comps.push(a);
                    comps.extend(ita.by_ref());
                    break;
                }
            }
        }
        if comps.is_empty() {
            Some(Path::new(".").into())
        } else {
            Some(comps.iter().map(|c| c.as_os_str()).collect())
        }
    }
}

/// Strip the base directories from a single compiler argument, for hashing.
///
/// [`strip_basedirs`] looks for a basedir where preprocessor output would put
/// one: at the start of a line, after a quote, after whitespace.  A compiler
/// argument spells the same paths differently.  `-I/home/user/project/include`
/// has no boundary at all before the path, and `-ffile-prefix-map` names a
/// directory without a trailing slash, as in
/// `-ffile-prefix-map=/home/user/project=.`, which is what keeps two checkouts
/// of the same tree from agreeing on a hash.
///
/// Only the places where an argument is expected to spell a pathname are
/// considered, rather than any offset that happens to match:
///
/// * the whole argument, as in a source file or the path of a separated
///   option such as `-include-pch <path>`;
/// * the value of an option written with an `=`, as in `--sysroot=<path>` or
///   either half of `-ffile-prefix-map=<path>=<path>`;
/// * the value glued to a short option, as in `-I<path>` or `-L<path>`.
///
/// A basedir sitting anywhere else is left alone, so `-DROOT="/home/user/project"`
/// keeps a definition the compiler will bake into the output verbatim.  A match
/// also has to end where a path component ends, so `/home/user/project` does not
/// match the start of `/home/user/project-docs`.  The longest basedir wins, so a
/// nested one takes precedence over the tree that contains it.
pub fn strip_basedirs_from_arg<'a>(arg: &'a [u8], basedirs: &[Vec<u8>]) -> Cow<'a, [u8]> {
    if basedirs.is_empty() || arg.is_empty() {
        return Cow::Borrowed(arg);
    }

    // We must return the original argument on all platforms, so we only
    // normalize a copy for searching.
    #[cfg(not(target_os = "windows"))]
    let haystack: &[u8] = arg;
    #[cfg(target_os = "windows")]
    let normalized = normalize_win_path(arg);
    #[cfg(target_os = "windows")]
    let haystack: &[u8] = &normalized;

    let mut order: Vec<usize> = (0..basedirs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(basedirs[i].len()));

    let mut result: Option<Vec<u8>> = None;
    let mut copied = 0;
    for pos in pathname_positions(haystack) {
        if pos < copied {
            // Inside a basedir we already stripped.
            continue;
        }
        let rest = &haystack[pos..];
        let mut matched = 0;
        for &i in &order {
            // Basedirs carry a trailing slash, see Config::basedirs. A path
            // component can also end at the end of the argument, or at a
            // separator such as the `=` of -ffile-prefix-map.
            let with_slash = basedirs[i].as_slice();
            let bare = &with_slash[..with_slash.len() - 1];
            if rest.starts_with(with_slash) {
                matched = with_slash.len();
            } else if rest.starts_with(bare) && matches!(rest.get(bare.len()), None | Some(b'=')) {
                matched = bare.len();
            }
            if matched > 0 {
                trace!(
                    "Matched basedir {} at position {} of argument",
                    String::from_utf8_lossy(with_slash),
                    pos
                );
                break;
            }
        }
        if matched > 0 {
            let out = result.get_or_insert_with(|| Vec::with_capacity(arg.len()));
            out.extend_from_slice(&arg[copied..pos]);
            copied = pos + matched;
        }
    }

    match result {
        Some(mut out) => {
            out.extend_from_slice(&arg[copied..]);
            Cow::Owned(out)
        }
        None => Cow::Borrowed(arg),
    }
}

/// The offsets in a compiler argument where a pathname can begin.
///
/// See [`strip_basedirs_from_arg`] for what they are and why the rest of the
/// argument is off limits.
fn pathname_positions(arg: &[u8]) -> impl Iterator<Item = usize> + '_ {
    // The value glued to a short option: everything up to the first byte that
    // cannot be part of an option name. `-I/path` yields 2, `-isystem/path` 8.
    let glued = if arg.first() == Some(&b'-') {
        let name_len = arg[1..]
            .iter()
            .take_while(|b| b.is_ascii_alphanumeric() || **b == b'-' || **b == b'_')
            .count();
        Some(1 + name_len)
    } else {
        None
    };

    // The whole argument, the value of each `=`-separated option, and the one
    // glued position, in ascending order and without duplicates.
    let equals = arg
        .iter()
        .enumerate()
        .filter(|(_, b)| **b == b'=')
        .map(|(i, _)| i + 1);

    let mut positions: Vec<usize> = std::iter::once(0).chain(glued).chain(equals).collect();
    positions.sort_unstable();
    positions.dedup();
    positions.into_iter().filter(move |&p| p < arg.len())
}

/// Double every `/` in a normalized path.
///
/// Paths inside preprocessor output are C string literals, so on Windows
/// their backslash separators are escaped (`\\`). After normalize_win_path
/// each backslash becomes a `/`, i.e. every escaped separator ends up as
/// `//`. This produces the matching variant of a basedir for such paths.
#[cfg(target_os = "windows")]
fn double_path_separators(path: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(path.len() * 2);
    for &b in path {
        out.push(b);
        if b == b'/' {
            out.push(b'/');
        }
    }
    out
}

/// Normalize path for case-insensitive comparison.
/// On Windows: converts all backslashes to forward slashes;
///             lowercases characters for consistency.
/// This function is used for:
///     - basedir_path: already normalized by std::path::absolute
///     - preprocessor_output: plain text that may contain invalid UTF-8
/// Leave it for any platform for testing purposes.
pub fn normalize_win_path(path: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(path.len());
    let mut i = 0;

    while i < path.len() {
        let b = path[i];

        // Fast path: ASCII characters (most common case)
        if b < 128 {
            result.push(match b {
                b'A'..=b'Z' => b + (b'a' - b'A'),
                b'\\' => b'/',
                _ => b,
            });
            i += 1;
            continue;
        }

        // Non-ASCII: try to decode UTF-8 sequence
        // Determine expected length from the first byte
        let char_len = match b {
            0b1100_0000..=0b1101_1111 => 2, // 110xxxxx
            0b1110_0000..=0b1110_1111 => 3, // 1110xxxx
            0b1111_0000..=0b1111_0111 => 4, // 11110xxx
            _ => {
                // Invalid UTF-8 start byte, copy as-is
                result.push(b);
                i += 1;
                continue;
            }
        };

        // Check if we have enough bytes
        if i + char_len > path.len() {
            // Incomplete sequence, copy as-is
            result.push(b);
            i += 1;
            continue;
        }

        // Validate and decode the UTF-8 sequence
        match std::str::from_utf8(&path[i..i + char_len]) {
            Ok(s) => {
                // Valid UTF-8, lowercase it
                result.extend_from_slice(s.to_lowercase().as_bytes());
                i += char_len;
            }
            Err(_) => {
                // Invalid sequence, copy first byte as-is
                result.push(b);
                i += 1;
            }
        }
    }

    result
}

/// Resolve the compiler executable, avoiding ccache/sccache wrappers.
///
/// This function handles scenarios where ccache/sccache might interfere:
/// 1. PATH contains directories like /usr/lib/ccache or /usr/lib64/ccache
///    which contain wrapper scripts that would call ccache
/// 2. The compiler executable itself is a symlink to ccache or sccache
///
/// Searches PATH for the compiler name, skipping any candidate whose path
/// contains wrapper directories or that resolves (via symlink) to ccache/sccache.
/// Returns the absolute path to the first valid match, or the original
/// executable if no valid match is found.
///
/// If the executable is already an absolute path and resolves to a wrapper,
/// PATH is searched for a non-wrapper compiler matching the executable name.
pub fn resolve_compiler_avoiding_wrapper(
    executable: &Path,
    env_vars: &[(OsString, OsString)],
) -> PathBuf {
    const WRAPPER_DIR_COMPONENTS: [&str; 2] = ["ccache", "sccache"];

    // Helper to check if a path points to ccache/sccache (via symlink resolution)
    let resolves_to_wrapper = |path: &Path| -> bool {
        std::fs::canonicalize(path)
            .ok()
            .and_then(|canonical| canonical.file_name().map(|n| n.to_os_string()))
            .map(|name| {
                let name_lower = name.to_string_lossy().to_lowercase();
                name_lower == "ccache"
                    || name_lower.starts_with("ccache.")
                    || name_lower == "sccache"
                    || name_lower.starts_with("sccache.")
            })
            .unwrap_or(false)
    };

    // Helper to check if a path contains wrapper directory components.
    let path_contains_wrapper = |path: &Path| -> bool {
        path.components().any(|c| {
            WRAPPER_DIR_COMPONENTS
                .iter()
                .any(|component| c.as_os_str() == *component)
        })
    };

    // Get the compiler name to search for
    let compiler_name = match executable.file_name() {
        Some(name) => name,
        None => return executable.to_path_buf(),
    };

    // If the path is absolute and does not resolve to a wrapper, return it as-is.
    if executable.is_absolute() && !resolves_to_wrapper(executable) {
        return executable.to_path_buf();
    }

    // Get PATH from env_vars
    let path_value = match env_vars.iter().find(|(key, _)| key == "PATH") {
        Some((_, value)) => value,
        None => return executable.to_path_buf(),
    };

    // Search PATH for a valid compiler, skipping wrapper candidates.
    for dir in std::env::split_paths(path_value) {
        // Skip directories that contain wrapper components in the path.
        if path_contains_wrapper(&dir) {
            continue;
        }

        let candidate = dir.join(compiler_name);
        if candidate.exists() && !resolves_to_wrapper(&candidate) {
            return candidate;
        }
    }

    // No valid match found, return the original
    executable.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::{Digest, OsStrExt, TimeMacroFinder, resolve_compiler_avoiding_wrapper};
    use std::ffi::{OsStr, OsString};
    use std::path::Path;

    #[test]
    fn test_resolve_compiler_avoiding_ccache_filters_path() {
        use std::env;

        // Create a PATH with ccache directories
        let path_with_ccache = env::join_paths([
            "/usr/lib64/ccache",
            "/usr/lib/ccache",
            "/usr/bin",
            "/usr/local/bin",
            "/home/user/bin",
        ])
        .unwrap();

        let env_vars = vec![
            (OsString::from("HOME"), OsString::from("/home/user")),
            (OsString::from("PATH"), path_with_ccache),
            (OsString::from("LANG"), OsString::from("en_US.UTF-8")),
        ];

        // When searching for "gcc", it should skip ccache directories
        // and find gcc in /usr/bin (if it exists) or return original
        let resolved = resolve_compiler_avoiding_wrapper(Path::new("gcc"), &env_vars);

        // The resolved path should not contain "ccache" in its path
        let resolved_str = resolved.to_string_lossy();
        assert!(
            !resolved_str.contains("ccache"),
            "Resolved path should not contain ccache: {resolved_str}"
        );
    }

    #[test]
    fn test_resolve_compiler_avoiding_ccache_no_path() {
        // No PATH variable at all
        let env_vars = vec![
            (OsString::from("HOME"), OsString::from("/home/user")),
            (OsString::from("LANG"), OsString::from("en_US.UTF-8")),
        ];

        let resolved = resolve_compiler_avoiding_wrapper(Path::new("/usr/bin/gcc"), &env_vars);

        // Should return original when no PATH
        assert_eq!(resolved, Path::new("/usr/bin/gcc"));
    }

    #[test]
    #[cfg(unix)]
    fn test_resolve_compiler_avoiding_ccache_skips_symlink_to_ccache() {
        use std::env;
        use std::os::unix::fs::symlink;

        let temp_dir = tempfile::tempdir().unwrap();
        let ccache_dir = temp_dir.path().join("ccache_bin");
        let real_dir = temp_dir.path().join("real_bin");
        std::fs::create_dir_all(&ccache_dir).unwrap();
        std::fs::create_dir_all(&real_dir).unwrap();

        // Create a fake "ccache" binary
        let ccache_bin = ccache_dir.join("ccache");
        std::fs::write(&ccache_bin, "fake ccache").unwrap();

        // Create a symlink "gcc" -> "ccache" in ccache_dir
        let gcc_symlink = ccache_dir.join("gcc");
        symlink(&ccache_bin, &gcc_symlink).unwrap();

        // Create a real "gcc" binary in real_dir
        let real_gcc = real_dir.join("gcc");
        std::fs::write(&real_gcc, "real gcc").unwrap();

        // Set up PATH with ccache_dir first, then real_dir
        let path = env::join_paths([&ccache_dir, &real_dir]).unwrap();
        let env_vars = vec![(OsString::from("PATH"), path)];

        let resolved = resolve_compiler_avoiding_wrapper(Path::new("gcc"), &env_vars);

        // Should skip the symlink to ccache and find the real gcc
        assert_eq!(resolved, real_gcc);
    }

    #[test]
    fn test_resolve_compiler_avoiding_ccache_absolute_path_unchanged() {
        use std::env;

        // Create a PATH with ccache directories first
        let path_with_ccache =
            env::join_paths(["/usr/lib64/ccache", "/usr/lib/ccache", "/usr/bin"]).unwrap();

        let env_vars = vec![(OsString::from("PATH"), path_with_ccache)];

        // When given an absolute path, it should be returned as-is without searching PATH
        let absolute_path = Path::new("/some/specific/path/to/gcc");
        let resolved = resolve_compiler_avoiding_wrapper(absolute_path, &env_vars);

        assert_eq!(resolved, absolute_path);
    }

    #[test]
    #[cfg(unix)]
    fn test_resolve_compiler_avoiding_ccache_absolute_symlink_to_sccache() {
        use std::env;
        use std::os::unix::fs::symlink;

        let temp_dir = tempfile::tempdir().unwrap();
        let wrapper_dir = temp_dir.path().join("sccache");
        let real_dir = temp_dir.path().join("real_bin");
        std::fs::create_dir_all(&wrapper_dir).unwrap();
        std::fs::create_dir_all(&real_dir).unwrap();

        // Create a fake "sccache" binary and a symlink wrapper "gcc" -> "sccache".
        let sccache_bin = wrapper_dir.join("sccache");
        std::fs::write(&sccache_bin, "fake sccache").unwrap();
        let wrapped_gcc = wrapper_dir.join("gcc");
        symlink(&sccache_bin, &wrapped_gcc).unwrap();

        // Real compiler in PATH outside wrapper directories.
        let real_gcc = real_dir.join("gcc");
        std::fs::write(&real_gcc, "real gcc").unwrap();

        let path = env::join_paths([&wrapper_dir, &real_dir]).unwrap();
        let env_vars = vec![(OsString::from("PATH"), path)];

        // Absolute wrapper path should be replaced by real compiler via PATH lookup.
        let resolved = resolve_compiler_avoiding_wrapper(&wrapped_gcc, &env_vars);
        assert_eq!(resolved, real_gcc);
    }

    #[test]
    fn simple_starts_with() {
        let a: &OsStr = "foo".as_ref();
        assert!(a.starts_with(""));
        assert!(a.starts_with("f"));
        assert!(a.starts_with("fo"));
        assert!(a.starts_with("foo"));
        assert!(!a.starts_with("foo2"));
        assert!(!a.starts_with("b"));
        assert!(!a.starts_with("b"));

        let a: &OsStr = "".as_ref();
        assert!(!a.starts_with("a"));
    }

    #[test]
    fn simple_strip_prefix() {
        let a: &OsStr = "foo".as_ref();

        assert_eq!(a.strip_prefix(""), Some(OsStr::new("foo")));
        assert_eq!(a.strip_prefix("f"), Some(OsStr::new("oo")));
        assert_eq!(a.strip_prefix("fo"), Some(OsStr::new("o")));
        assert_eq!(a.strip_prefix("foo"), Some(OsStr::new("")));
        assert_eq!(a.strip_prefix("foo2"), None);
        assert_eq!(a.strip_prefix("b"), None);
    }

    #[test]
    fn test_time_macro_short_read() {
        // Normal "read" should succeed
        let mut finder = TimeMacroFinder::new();
        finder.find_time_macros(b"__TIME__");
        assert!(finder.found_time());

        // So should a partial "read"
        let mut finder = TimeMacroFinder::new();
        finder.find_time_macros(b"__");
        assert!(!finder.found_time());
        finder.find_time_macros(b"TIME__");
        assert!(finder.found_time());

        // So should a partial "read" later down the line
        let mut finder = TimeMacroFinder::new();
        finder.find_time_macros(b"Something or other larger than the haystack");
        finder.find_time_macros(b"__");
        assert!(!finder.found_time());
        finder.find_time_macros(b"TIME__");
        assert!(finder.found_time());

        // Even if the last "read" is large
        let mut finder = TimeMacroFinder::new();
        finder.find_time_macros(b"Something or other larger than the haystack");
        finder.find_time_macros(b"__");
        assert!(!finder.found_time());
        finder.find_time_macros(b"TIME__ something or other larger than the haystack");
        assert!(finder.found_time());

        // Pathological case
        let mut finder = TimeMacroFinder::new();
        finder.find_time_macros(b"__");
        assert!(!finder.found_time());
        finder.find_time_macros(b"TI");
        assert!(!finder.found_time());
        finder.find_time_macros(b"ME");
        assert!(!finder.found_time());
        finder.find_time_macros(b"__");
        assert!(finder.found_time());

        // Odd-numbered pathological case
        let mut finder = TimeMacroFinder::new();
        finder.find_time_macros(b"This is larger than the haystack __");
        assert!(!finder.found_time());
        finder.find_time_macros(b"TI");
        assert!(!finder.found_time());
        finder.find_time_macros(b"ME");
        assert!(!finder.found_time());
        finder.find_time_macros(b"__");
        assert!(finder.found_time());

        // Sawtooth length pathological case
        let mut finder = TimeMacroFinder::new();
        finder.find_time_macros(b"This is larger than the haystack __");
        assert!(!finder.found_time());
        finder.find_time_macros(b"TI");
        assert!(!finder.found_time());
        finder.find_time_macros(b"ME__ This is larger than the haystack");
        assert!(finder.found_time());
        assert!(!finder.found_timestamp());
        finder.find_time_macros(b"__");
        assert!(!finder.found_timestamp());
        finder.find_time_macros(b"TIMESTAMP__ This is larger than the haystack");
        assert!(finder.found_timestamp());

        // Odd-numbered sawtooth length pathological case
        let mut finder = TimeMacroFinder::new();
        finder.find_time_macros(b"__");
        assert!(!finder.found_time());
        finder.find_time_macros(b"TIME__ This is larger than the haystack");
        assert!(finder.found_time());
        assert!(!finder.found_timestamp());
        finder.find_time_macros(b"__");
        assert!(!finder.found_timestamp());
        finder.find_time_macros(b"TIMESTAMP__ This is larger than the haystack");
        assert!(finder.found_timestamp());
    }

    #[test]
    fn test_ascii_unescape_default() {
        let mut alphabet = r#"\\'"\t\n\r"#.as_bytes().to_vec();
        alphabet.push(b'a');
        alphabet.push(b'1');
        alphabet.push(0);
        alphabet.push(0xff);
        let mut input = vec![];
        let mut output = vec![];
        let mut alphabet_indexes = [0; 3];
        let mut tested_cases = 0;
        // Following loop may test duplicated inputs, but it's not a problem
        loop {
            input.clear();
            output.clear();
            for idx in alphabet_indexes {
                if idx < alphabet.len() {
                    input.push(alphabet[idx]);
                }
            }
            if input.is_empty() {
                break;
            }
            output.extend(input.as_slice().escape_ascii());
            let result = super::ascii_unescape_default(&output).unwrap();
            assert_eq!(input, result, "{output:?}");
            tested_cases += 1;
            for idx in &mut alphabet_indexes {
                *idx += 1;
                if *idx > alphabet.len() {
                    // Use `>` so we can test various input length.
                    *idx = 0;
                } else {
                    break;
                }
            }
        }
        assert_eq!(tested_cases, (alphabet.len() + 1).pow(3) - 1);
        let empty_result = super::ascii_unescape_default(&[]).unwrap();
        assert!(empty_result.is_empty(), "{empty_result:?}");
    }

    #[test]
    fn test_strip_basedirs_from_arg() {
        let basedir = b"/home/user/project/".to_vec();
        let strip = |arg: &[u8]| {
            super::strip_basedirs_from_arg(arg, std::slice::from_ref(&basedir)).into_owned()
        };

        // The flag that ties an object file to one checkout: the tree is named
        // without a trailing slash, so the path component ends at the `=`.
        assert_eq!(
            strip(b"-ffile-prefix-map=/home/user/project=."),
            b"-ffile-prefix-map==."
        );
        assert_eq!(
            strip(b"-ffile-prefix-map=/home/user/project/build=."),
            b"-ffile-prefix-map=build=."
        );
        assert_eq!(
            strip(b"-ffile-prefix-map=/home/user/project/build/=build/"),
            b"-ffile-prefix-map=build/=build/"
        );

        // Paths with no boundary before them.
        assert_eq!(strip(b"-I/home/user/project/include"), b"-Iinclude");
        assert_eq!(strip(b"/home/user/project/src/main.c"), b"src/main.c");

        // A sibling directory that merely starts with the same bytes is not a
        // match, in either spelling.
        assert_eq!(
            strip(b"-I/home/user/project-docs/include"),
            b"-I/home/user/project-docs/include"
        );
        assert_eq!(
            strip(b"-ffile-prefix-map=/home/user/project-docs=."),
            b"-ffile-prefix-map=/home/user/project-docs=."
        );

        // Untouched without basedirs, and when nothing matches.
        assert_eq!(
            &*super::strip_basedirs_from_arg(b"-ffile-prefix-map=/home/user/project=.", &[]),
            b"-ffile-prefix-map=/home/user/project=."
        );
        assert_eq!(strip(b"-O2"), b"-O2");
        assert_eq!(strip(b""), b"");
    }

    #[test]
    fn test_strip_basedirs_from_arg_only_at_pathname_positions() {
        let basedir = b"/home/user/project/".to_vec();
        let strip = |arg: &[u8]| {
            super::strip_basedirs_from_arg(arg, std::slice::from_ref(&basedir)).into_owned()
        };

        // A macro definition the compiler bakes into the output verbatim: the
        // path is inside a string literal, not where a pathname is expected.
        assert_eq!(
            strip(b"-DROOT=\"/home/user/project\""),
            b"-DROOT=\"/home/user/project\""
        );

        // Not the value of an option, just a path further along in one.
        assert_eq!(
            strip(b"-Wl,-rpath,/home/user/project/lib"),
            b"-Wl,-rpath,/home/user/project/lib"
        );
        assert_eq!(
            strip(b"-fplugin-arg-p=x/home/user/project/y"),
            b"-fplugin-arg-p=x/home/user/project/y"
        );

        // The value glued to a short option, however long its name.
        assert_eq!(
            strip(b"-isystem/home/user/project/abseil"),
            b"-isystemabseil"
        );
        assert_eq!(strip(b"-L/home/user/project/lib"), b"-Llib");

        // The value of an =-separated option, and both halves of a prefix map.
        assert_eq!(strip(b"--sysroot=/home/user/project/sys"), b"--sysroot=sys");
        assert_eq!(
            strip(b"-ffile-prefix-map=/home/user/project/build=/home/user/project"),
            b"-ffile-prefix-map=build="
        );
    }

    #[test]
    fn test_strip_basedirs_from_arg_skips_positions_inside_a_match() {
        // A directory whose name contains an `=` is legal, and it puts one of
        // the positions where an option value could start inside the basedir
        // itself. If a second basedir then matches at that position, the two
        // matches overlap, and the tail of the first directory name must not
        // be stripped a second time.
        let basedirs = vec![b"/home/user/a=/b/".to_vec(), b"/b/".to_vec()];
        let strip = |arg: &[u8]| super::strip_basedirs_from_arg(arg, &basedirs).into_owned();

        assert_eq!(strip(b"-I/home/user/a=/b/include"), b"-Iinclude");
        assert_eq!(strip(b"/home/user/a=/b/src/main.c"), b"src/main.c");
        assert_eq!(strip(b"--sysroot=/home/user/a=/b/sys"), b"--sysroot=sys");
    }

    #[test]
    fn test_strip_basedirs_from_arg_longest_match_wins() {
        // A build directory nested inside the tree, both listed.
        let basedirs = vec![
            b"/home/user/project/".to_vec(),
            b"/home/user/project/build/".to_vec(),
        ];
        assert_eq!(
            &*super::strip_basedirs_from_arg(
                b"-ffile-prefix-map=/home/user/project/build=.",
                &basedirs
            ),
            b"-ffile-prefix-map==."
        );
        assert_eq!(
            &*super::strip_basedirs_from_arg(b"-ffile-prefix-map=/home/user/project=.", &basedirs),
            b"-ffile-prefix-map==."
        );
    }

    #[test]
    fn test_strip_basedir_simple() {
        // Simple cases
        let basedir = b"/home/user/project/".to_vec();
        let input = b"# 1 \"/home/user/project/src/main.c\"\nint main() { return 0; }";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"# 1 \"src/main.c\"\nint main() { return 0; }";
        assert_eq!(&*output, expected);

        // Multiple occurrences
        let input =
            b"# 1 \"/home/user/project/src/main.c\"\n# 2 \"/home/user/project/include/header.h\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"# 1 \"src/main.c\"\n# 2 \"include/header.h\"";
        assert_eq!(&*output, expected);

        // No occurrences
        let input = b"# 1 \"/other/path/main.c\"\nint main() { return 0; }";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        assert_eq!(&*output, input);
    }

    #[test]
    fn test_strip_basedir_empty() {
        // Empty basedir slice
        let input = b"# 1 \"/home/user/project/src/main.c\"";
        let output = super::strip_basedirs(input, &[]);
        assert_eq!(&*output, input);

        // Empty input
        let basedir = b"/home/user/project/".to_vec();
        let input = b"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        assert_eq!(&*output, input);
    }

    #[test]
    fn test_strip_basedir_not_at_boundary() {
        // basedir should only match at word boundaries
        let basedir = b"/home/user/".to_vec();
        let input = b"text/home/user/file.c and \"/home/user/other.c\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        // Should only replace the second occurrence (after quote)
        let expected = b"text/home/user/file.c and \"other.c\"";
        assert_eq!(&*output, expected);
    }

    #[test]
    fn test_strip_basedir_trailing_slashes() {
        // Without trailing slash
        let basedir = b"/home/user/project".to_vec();
        let input = b"# 1 \"/home/user/project/src/main.c\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"# 1 \"/src/main.c\""; // Wrong, but expected
        assert_eq!(&*output, expected);

        // Trailing slashes aren't ignored, they must be cleaned in config reader
        let basedir = b"/home/user/project/".to_vec();
        let input = b"# 1 \"/home/user/project/src/main.c\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"# 1 \"src/main.c\"";
        assert_eq!(&*output, expected);
    }

    #[test]
    fn test_strip_basedirs_multiple() {
        // Multiple basedirs - should match longest first
        let basedirs = vec![
            b"/home/user1/project/".to_vec(),
            b"/home/user2/workspace/".to_vec(),
        ];
        let input =
            b"# 1 \"/home/user1/project/src/main.c\"\n# 2 \"/home/user2/workspace/lib/util.c\"";
        let output = super::strip_basedirs(input, &basedirs);
        let expected = b"# 1 \"src/main.c\"\n# 2 \"lib/util.c\"";
        assert_eq!(&*output, expected);

        // Longest prefix wins
        let basedirs = vec![b"/home/user/".to_vec(), b"/home/user/project/".to_vec()];
        let input = b"# 1 \"/home/user/project/src/main.c\"";
        let output = super::strip_basedirs(input, &basedirs);
        let expected = b"# 1 \"src/main.c\"";
        assert_eq!(&*output, expected);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_strip_basedir_windows_backslashes() {
        // Without trailing backslash
        let basedir = b"c:/users/test/project".to_vec();
        let input = b"# 1 \"C:\\Users\\test\\project\\Src\\Main.c\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        // normalized backslash to slash
        let expected = b"# 1 \"\\Src\\Main.c\""; // Wrong, but expected
        assert_eq!(&*output, expected);

        // Trailing slashes aren't ignored, they must be cleaned in config reader
        let basedir = b"c:/users/test/project/".to_vec();
        let input = b"# 1 \"C:\\Users\\test\\project\\src\\main.c\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"# 1 \"src\\main.c\"";
        assert_eq!(&*output, expected);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_strip_basedir_windows_mixed_slashes() {
        // The slashes may be mixed in preprocessor output, but the uncut output
        // should remain untouched.
        // Mixed forward and backslashes in input (common from certain build systems)
        let basedir = b"c:/users/test/project/".to_vec();
        let input = b"# 1 \"C:/Users\\test\\project\\src/main.c\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"# 1 \"src/main.c\"";
        assert_eq!(&*output, expected, "Failed to strip mixed slash path");

        // Also test the reverse case, it doesn't work, because basedir normalization must be done
        // in advance
        let input = b"# 1 \"C:\\Users/test/project/src\\main.c\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"# 1 \"src\\main.c\"";
        assert_eq!(
            &*output, expected,
            "Failed to strip reverse mixed slash path"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_strip_basedir_windows_escaped_backslashes() {
        // MSVC (cl -E) and clang on Windows quote paths in #line directives /
        // GNU linemarkers as C string literals, escaping every backslash:
        //     #line 1 "C:\\Users\\test\\project\\src\\main.c"
        // normalize_win_path converts each '\' to '/', so each escaped
        // separator becomes a DOUBLE forward slash ("c://users//...") and must
        // still match the basedir (normalized with single slashes).
        let basedir = b"c:/users/test/project/".to_vec();
        let input = b"#line 1 \"C:\\\\Users\\\\test\\\\project\\\\src\\\\main.c\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"#line 1 \"src\\\\main.c\"";
        assert_eq!(&*output, &expected[..]);

        // GNU-style linemarker as emitted by clang on Windows
        let input = b"# 1 \"C:\\\\Users\\\\test\\\\project\\\\include\\\\header.h\" 1\nint x;";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"# 1 \"include\\\\header.h\" 1\nint x;";
        assert_eq!(&*output, &expected[..]);

        // Multiple occurrences, plain and escaped forms mixed in one output
        let input =
            b"# 1 \"C:\\\\Users\\\\test\\\\project\\\\a.c\"\n# 2 \"C:/Users/test/project/b.h\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"# 1 \"a.c\"\n# 2 \"b.h\"";
        assert_eq!(&*output, &expected[..]);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_strip_basedir_windows_escaped_unc() {
        // UNC basedir: //server/share/dir/ appears as \\\\server\\share\\dir\\
        // in escaped form
        let basedir = b"//server/share/project/".to_vec();
        let input = b"#line 1 \"\\\\\\\\server\\\\share\\\\project\\\\src\\\\main.c\"";
        let output = super::strip_basedirs(input, std::slice::from_ref(&basedir));
        let expected = b"#line 1 \"src\\\\main.c\"";
        assert_eq!(&*output, &expected[..]);
    }

    #[test]
    fn test_normalize_win_path_ascii() {
        // Test basic ASCII normalization
        let input = b"C:\\Users\\Test\\Project";
        let normalized = super::normalize_win_path(input);
        assert_eq!(normalized, b"c:/users/test/project");

        // Test mixed case
        let input = b"C:\\USERS\\test\\PROJECT";
        let normalized = super::normalize_win_path(input);
        assert_eq!(normalized, b"c:/users/test/project");
    }

    #[test]
    fn test_normalize_win_path_utf8() {
        // Test with UTF-8 characters (e.g., German umlauts)
        let input = "C:\\Users\\Müller\\Projekt".as_bytes();
        let normalized = super::normalize_win_path(input);
        let expected = "c:/users/müller/projekt".as_bytes();
        assert_eq!(normalized, expected);

        // Test with Cyrillic characters
        let input = "C:\\Пользователь\\Проект".as_bytes();
        let normalized = super::normalize_win_path(input);
        let expected = "c:/пользователь/проект".as_bytes();
        assert_eq!(normalized, expected);

        // Test with Turkish İ (special case)
        let input = "C:\\İstanbul\\DİREKTÖRY".as_bytes();
        let normalized = super::normalize_win_path(input);
        // Turkish İ lowercases to i with dot
        let expected = "c:/i\u{307}stanbul/di\u{307}rektöry".as_bytes();
        assert_eq!(normalized, expected);
    }

    #[test]
    fn test_normalize_win_path_mixed_ascii_utf8() {
        // Test mixed ASCII and UTF-8
        let input = "C:\\Users\\Test\\Café\\Проект".as_bytes();
        let normalized = super::normalize_win_path(input);
        let expected = "c:/users/test/café/проект".as_bytes();
        assert_eq!(normalized, expected);
    }

    #[test]
    fn test_normalize_win_path_invalid_utf8() {
        // Test with invalid UTF-8 sequence (should preserve as-is)
        let mut input = b"C:\\Users\\".to_vec();
        input.push(0xFF); // Invalid UTF-8
        input.extend_from_slice(b"\\Test");

        let normalized = super::normalize_win_path(&input);

        // Should lowercase ASCII and convert backslashes, but preserve invalid byte
        let mut expected = b"c:/users/".to_vec();
        expected.push(0xFF);
        expected.extend_from_slice(b"/test");
        assert_eq!(normalized, expected);
    }

    #[test]
    fn test_normalize_win_path_incomplete_utf8() {
        // Test with incomplete UTF-8 sequence at the end
        let mut input = b"C:\\Users\\Test".to_vec();
        input.push(0xC3); // Start of 2-byte UTF-8 but incomplete

        let normalized = super::normalize_win_path(&input);

        // Should preserve incomplete byte as-is
        let mut expected = b"c:/users/test".to_vec();
        expected.push(0xC3);
        assert_eq!(normalized, expected);
    }

    #[test]
    fn test_normalize_win_path_empty() {
        let input = b"";
        let normalized = super::normalize_win_path(input);
        assert_eq!(normalized, b"");
    }

    #[tokio::test]
    async fn test_digest_reader_hashes_file() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("hello.txt");
        std::fs::write(&path, b"hello, world").unwrap();
        let hash = Digest::from_file(path).await.unwrap().finish();
        assert_eq!(hash.len(), 64, "blake3 hex digest is 64 chars");
    }

    // Regression for https://github.com/mozilla/sccache/issues/2653: rustc's
    // dep-info may list a directory path (e.g. from a proc macro that calls
    // `proc_macro::tracked_path::path()` on a directory). The four tests below
    // exercise Digest::from_file against directories.

    #[tokio::test]
    async fn test_digest_reader_hashes_directory() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        std::fs::create_dir_all(dir.join("nested/deeper")).unwrap();
        std::fs::write(dir.join("root.txt"), b"root").unwrap();
        std::fs::write(dir.join("nested/inner.txt"), b"inner").unwrap();
        std::fs::write(dir.join("nested/deeper/deep.txt"), b"deep").unwrap();
        let hash = Digest::from_file(dir).await.unwrap().finish();
        assert_eq!(hash.len(), 64);
    }

    #[tokio::test]
    async fn test_digest_reader_directory_is_deterministic() {
        fn populate(dir: &std::path::Path) {
            std::fs::create_dir_all(dir.join("a/b")).unwrap();
            std::fs::create_dir_all(dir.join("c")).unwrap();
            std::fs::write(dir.join("a/b/one.txt"), b"one").unwrap();
            std::fs::write(dir.join("a/two.txt"), b"two").unwrap();
            std::fs::write(dir.join("c/three.txt"), b"three").unwrap();
            std::fs::write(dir.join("four.txt"), b"four").unwrap();
        }
        let t1 = tempfile::tempdir().unwrap();
        let t2 = tempfile::tempdir().unwrap();
        populate(t1.path());
        populate(t2.path());
        let h1 = Digest::from_file(t1.path()).await.unwrap().finish();
        let h2 = Digest::from_file(t2.path()).await.unwrap().finish();
        assert_eq!(h1, h2);
    }

    #[tokio::test]
    async fn test_digest_reader_directory_detects_content_change() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub/data.txt"), b"before").unwrap();
        let before = Digest::from_file(dir).await.unwrap().finish();
        std::fs::write(dir.join("sub/data.txt"), b"after").unwrap();
        let after = Digest::from_file(dir).await.unwrap().finish();
        assert_ne!(before, after);
    }

    #[tokio::test]
    async fn test_digest_reader_directory_detects_path_change() {
        // Two dirs with identical file *contents* but different file *names*
        // must hash to different values, because collect_dir_entries mixes
        // the relative path into the digest as a delimiter.
        let t1 = tempfile::tempdir().unwrap();
        let t2 = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(t1.path().join("sub")).unwrap();
        std::fs::create_dir_all(t2.path().join("sub")).unwrap();
        std::fs::write(t1.path().join("sub/a.txt"), b"x").unwrap();
        std::fs::write(t2.path().join("sub/b.txt"), b"x").unwrap();
        let h1 = Digest::from_file(t1.path()).await.unwrap().finish();
        let h2 = Digest::from_file(t2.path()).await.unwrap().finish();
        assert_ne!(h1, h2);
    }

    #[test]
    #[cfg(windows)]
    fn local_oem_codepage_conversions() {
        use crate::util::{bytes_to_str, wide_char_to_multi_byte};
        use windows_sys::Win32::Globalization::GetOEMCP;

        let current_oemcp = unsafe { GetOEMCP() };
        // We don't control the local OEM codepage so test only if it is one of:
        // United States, Latin-1 and Latin-1 + euro symbol
        if current_oemcp == 437 || current_oemcp == 850 || current_oemcp == 858 {
            // Non-ASCII characters
            const INPUT_STRING: &str = "ÇüéâäàåçêëèïîìÄÅ";

            // The characters in INPUT_STRING encoded per the OEM codepage
            const INPUT_BYTES: [u8; 16] = [
                128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            ];

            // Test the conversion from the OEM codepage to UTF-8
            assert_eq!(
                bytes_to_str(INPUT_BYTES.to_vec()).unwrap().as_ref(),
                INPUT_STRING
            );

            // The characters in INPUT_STRING encoded in UTF-16
            const INPUT_WORDS: [u16; 16] = [
                199, 252, 233, 226, 228, 224, 229, 231, 234, 235, 232, 239, 238, 236, 196, 197,
            ];

            // Test the conversion from UTF-16 to the OEM codepage
            assert_eq!(wide_char_to_multi_byte(&INPUT_WORDS).unwrap(), INPUT_BYTES);
        }
    }
}
