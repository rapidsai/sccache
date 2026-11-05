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

use super::utils::{get_file_mode, set_file_mode};
use crate::{errors::*, mock_command::ProcessOutput};
use fs_err as fs;
use std::{
    fmt,
    io::{self, Cursor, Read, Seek, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::FileOptions};

/// Cache object sourced by a file.
#[derive(Clone)]
pub struct FileObjectSource {
    /// Identifier for this object. Should be unique within a compilation unit.
    /// Note that a compilation unit is a single source file in C/C++ and a crate in Rust.
    pub key: String,
    /// Path parent dir (or cwd if parent is None)
    pub dir: PathBuf,
    /// Absolute path to the file.
    pub path: PathBuf,
    /// Whether the file must be present on disk and is essential for the compilation.
    pub optional: bool,
}

/// Result of a cache lookup.
#[derive(Eq, PartialEq)]
pub enum Cache<T> {
    /// Result was found in cache.
    Hit(T),
    /// Result was not found in cache.
    Miss,
}

impl<T> Cache<T> {
    pub fn into_option(self) -> Option<T> {
        self.into()
    }
}

impl<T> From<Cache<T>> for Option<T> {
    fn from(val: Cache<T>) -> Self {
        match val {
            Cache::Hit(data) => Some(data),
            Cache::Miss => None,
        }
    }
}

impl<T> fmt::Debug for Cache<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Cache::Hit(_) => write!(f, "Cache::Hit(...)"),
            Cache::Miss => write!(f, "Cache::Miss"),
        }
    }
}

/// Trait objects can't be bounded by more than one non-builtin trait.
pub trait BufReadSeek: std::io::BufRead + Seek + Send {}

impl<T: std::io::BufRead + Seek + Send> BufReadSeek for T {}

/// Data stored in the compiler cache.
pub struct CacheRead {
    zip: ZipArchive<Box<dyn BufReadSeek>>,
}

/// Represents a failure to decompress stored object data.
#[derive(Debug)]
pub struct DecompressionFailure {
    message: String,
}

impl std::fmt::Display for DecompressionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for DecompressionFailure {}

/// Represents a failure to decompress stored object data.
#[derive(Debug)]
pub struct UnexpectedFileSize(pub u64, pub u64);

impl std::fmt::Display for UnexpectedFileSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Expected {} bytes, but unpacked {}", self.0, self.1)
    }
}

impl std::error::Error for UnexpectedFileSize {}

impl TryFrom<opendal::Buffer> for CacheRead {
    type Error = anyhow::Error;

    fn try_from(entry: opendal::Buffer) -> Result<Self> {
        CacheRead::from(OpendalBufferCursor::new(entry))
    }
}

impl CacheRead {
    /// Create a cache entry from `reader`.
    pub fn from<R>(reader: R) -> Result<CacheRead>
    where
        R: BufReadSeek + 'static,
    {
        let z = ZipArchive::new(Box::new(reader) as Box<dyn BufReadSeek>)
            .context("Failed to parse cache entry")?;
        Ok(CacheRead { zip: z })
    }

    /// Get an object from this cache entry at `name` and write it to `to`.
    /// If the file has stored permissions, return them.
    pub fn get_object<T>(&mut self, name: &str, to: &mut T) -> Result<Option<u32>>
    where
        T: Write,
    {
        let file = self.zip.by_name(name).map_err(|_| DecompressionFailure {
            message: format!("Failed to decompress {name:?}"),
        })?;
        if file.compression() != CompressionMethod::Stored {
            bail!(DecompressionFailure {
                message: format!(
                    "Compression method must be {:?}, received {:?}",
                    CompressionMethod::Stored,
                    file.compression()
                )
            });
        }
        let mode = file.unix_mode();
        zstd::stream::copy_decode(file, to).map_err(|_| DecompressionFailure {
            message: format!("Failed to decompress {name:?}"),
        })?;
        Ok(mode)
    }

    /// Get the stdout from this cache entry, if it exists.
    pub fn get_stdout(&mut self) -> Vec<u8> {
        self.get_bytes("stdout")
    }

    /// Get the stderr from this cache entry, if it exists.
    pub fn get_stderr(&mut self) -> Vec<u8> {
        self.get_bytes("stderr")
    }

    pub fn get_bytes(&mut self, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        drop(self.get_object(name, &mut bytes));
        bytes
    }

    pub fn into_inner(self) -> Box<dyn BufReadSeek> {
        self.zip.into_inner()
    }

    pub async fn extract_objects<T>(
        mut self,
        objects: T,
        pool: &tokio::runtime::Handle,
    ) -> Result<()>
    where
        T: IntoIterator<Item = FileObjectSource> + Send + Sync + 'static,
    {
        crate::util::spawn_blocking_on(pool, move || {
            for FileObjectSource {
                key,
                dir,
                path,
                optional,
            } in objects
            {
                if is_path_null(&path) {
                    // For unix, this is just a fast path to discard such outputs,
                    // so it is not an issue if `is_path_null` has false-negatives.
                    // But for Windows, since `NUL` looks like a relative path, the
                    // temporary file creation logic would happily succeed, creating
                    // a temp file in the CWD, but then the subsequent `persist`
                    // would fail with `ERROR_ALREADY_EXISTS`, since `NUL` always
                    // exists and cannot be `replaced`, so we really need to
                    // short-circuit more of such cases on Windows.
                    debug!("Skipping output to {}", path.display());
                    continue;
                }
                // Write the cache entry to a tempfile and then atomically
                // move it to its final location so that other rustc invocations
                // happening in parallel don't see a partially-written file.
                match (crate::util::tempfile_in(dir), optional) {
                    (Ok(mut tmp), _) => {
                        match (self.get_object(&key, &mut tmp), optional) {
                            (Ok(mode), _) => {
                                match tmp.persist(&path) {
                                    Ok(file) => {
                                        if let Some(mode) = mode {
                                            set_file_mode(&path, mode)?;
                                        }
                                        // Sync immediately so other readers see the file.
                                        // This is important for nvcc's .module_id files,
                                        // which are read during many pararllel compilations.
                                        file.sync_all().map(|_| file)
                                    }
                                    Err(err) => Err(err.error),
                                }?;
                            }
                            (Err(e), false) => return Err(e),
                            // skip if no object found and it's optional
                            (Err(_), true) => continue,
                        }
                    }
                    (Err(e), false) => {
                        // Fall back to writing directly to the final location
                        warn!("Failed to create temp file on the same file system: {e}");
                        let mut f = std::fs::File::create(&path)?;
                        // `optional` is false in this branch, so do not ignore errors
                        let mode = self.get_object(&key, &mut f)?;
                        if let Some(mode) = mode
                            && let Err(e) = set_file_mode(path.as_path(), mode)
                        {
                            // Here we ignore errors from setting file mode because
                            // if we could not create a temp file in the same directory,
                            // we probably can't set the mode either (e.g. /dev/stuff)
                            warn!("Failed to reset file mode: {e}");
                        }
                    }
                    // skip if no object found and it's optional
                    (Err(_), true) => continue,
                }
            }
            Ok(())
        })
        .await?
    }
}

#[cfg(unix)]
fn is_path_null(path: &Path) -> bool {
    path == Path::new("/dev/null")
}

#[cfg(windows)]
fn is_path_null(path: &Path) -> bool {
    // For Windows, it appears that `NUL` with whatever extension is also a blackhole
    // (at least for `CreateFileX`), so it does not suffice to check for an exact match
    // Also note that gcc, cl.exe, et al. append a correct extension automatically even
    // if the user asks for output to `NUL`.
    let Some(stem) = path.file_stem() else {
        return false;
    };
    stem.eq_ignore_ascii_case("NUL")
}

/// Data to be stored in the compiler cache.
pub struct CacheWrite {
    zip: ZipWriter<io::Cursor<Vec<u8>>>,
}

impl CacheWrite {
    /// Create a new, empty cache entry.
    pub fn new() -> CacheWrite {
        CacheWrite {
            zip: ZipWriter::new(io::Cursor::new(vec![])),
        }
    }

    /// Create a new cache entry populated with the contents of `objects`.
    pub async fn from_objects<T>(
        objects: T,
        output: ProcessOutput,
        pool: &tokio::runtime::Handle,
    ) -> Result<(Vec<u8>, ProcessOutput, Duration)>
    where
        T: IntoIterator<Item = FileObjectSource> + Send + Sync + 'static,
    {
        let start_create_artifact = Instant::now();

        crate::util::spawn_blocking_on(pool, move || {
            let mut entry = CacheWrite::new();
            for FileObjectSource {
                key,
                path,
                optional,
                ..
            } in objects
            {
                let file = fs::File::open(&path)
                    .with_context(|| format!("failed to open file `{path:?}`"));
                match (file, optional) {
                    (Ok(mut file), _) => {
                        let mode = get_file_mode(&file)?;
                        entry.put_object(&key, &mut file, mode).with_context(|| {
                            format!("failed to put object `{path:?}` in cache entry")
                        })?;
                    }
                    (Err(e), false) => return Err(e),
                    (Err(_), true) => continue,
                }
            }

            entry.put_stdout(&output.stdout)?;
            entry.put_stderr(&output.stderr)?;

            Ok((entry.finish()?, output, start_create_artifact.elapsed()))
        })
        .await?
    }

    /// Add an object containing the contents of `from` to this cache entry at `name`.
    /// If `mode` is `Some`, store the file entry with that mode.
    pub fn put_object<T>(&mut self, name: &str, from: &mut T, mode: Option<u32>) -> Result<()>
    where
        T: Read,
    {
        // We're going to declare the compression method as "stored",
        // but we're actually going to store zstd-compressed blobs.
        let opts = FileOptions::default().compression_method(CompressionMethod::Stored);
        let opts = if let Some(mode) = mode {
            opts.unix_permissions(mode)
        } else {
            opts
        };
        self.zip
            .start_file(name, opts)
            .context("Failed to start cache entry object")?;

        let compression_level = std::env::var("SCCACHE_CACHE_ZSTD_LEVEL")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(zstd::DEFAULT_COMPRESSION_LEVEL);
        zstd::stream::copy_encode(from, &mut self.zip, compression_level)?;
        Ok(())
    }

    pub fn put_stdout(&mut self, bytes: &[u8]) -> Result<()> {
        self.put_bytes("stdout", bytes)
    }

    pub fn put_stderr(&mut self, bytes: &[u8]) -> Result<()> {
        self.put_bytes("stderr", bytes)
    }

    fn put_bytes(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        if !bytes.is_empty() {
            let mut cursor = Cursor::new(bytes);
            return self.put_object(name, &mut cursor, None);
        }
        Ok(())
    }

    /// Finish writing data to the cache entry writer, and return the data.
    pub fn finish(self) -> Result<Vec<u8>> {
        let CacheWrite { mut zip } = self;
        let cur = zip.finish().context("Failed to finish cache entry zip")?;
        Ok(cur.into_inner())
    }
}

impl Default for CacheWrite {
    fn default() -> Self {
        Self::new()
    }
}

struct OpendalBufferCursor {
    data: opendal::Buffer,
    view: opendal::Buffer,
    pos: u64,
}

impl OpendalBufferCursor {
    pub fn new(buffer: opendal::Buffer) -> Self {
        Self {
            view: buffer.slice(..),
            data: buffer,
            pos: 0,
        }
    }
}

impl io::Read for OpendalBufferCursor {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let size = self.view.read(buf)?;
        self.seek(io::SeekFrom::Current(size as i64))?;
        Ok(size)
    }
}

impl io::BufRead for OpendalBufferCursor {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.view.fill_buf()
    }

    fn consume(&mut self, amount: usize) {
        self.pos += amount as u64;
    }
}

impl Seek for OpendalBufferCursor {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let end = self.data.len() as u64;
        match pos {
            io::SeekFrom::Start(pos) => {
                self.pos = pos.min(end);
            }
            io::SeekFrom::Current(pos) => {
                if pos.is_negative() {
                    self.pos = self.pos.saturating_sub(pos.unsigned_abs());
                } else {
                    self.pos = self.pos.saturating_add(pos as u64).min(end);
                }
            }
            io::SeekFrom::End(pos) => {
                if pos.is_negative() {
                    self.pos = end.saturating_sub(pos.unsigned_abs());
                } else {
                    self.pos = end.saturating_add(pos as u64).min(end);
                }
            }
        }
        self.view = self.data.slice(self.pos as usize..);
        Ok(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn test_extract_object_to_devnull_works() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .worker_threads(1)
            .build()
            .unwrap();

        let pool = runtime.handle();

        let cache_data = CacheWrite::new();
        let cache_read =
            CacheRead::from(std::io::Cursor::new(cache_data.finish().unwrap())).unwrap();

        let objects = vec![FileObjectSource {
            key: "test_key".to_string(),
            dir: PathBuf::from("/dev"),
            path: PathBuf::from("/dev/null"),
            optional: false,
        }];

        let result = crate::util::block_on(&runtime, cache_read.extract_objects(objects, pool));
        assert!(result.is_ok(), "Extracting to /dev/null should succeed");
    }

    #[cfg(unix)]
    #[test]
    fn test_extract_object_to_dev_fd_something() {
        // Open a pipe, write to `/dev/fd/{fd}` and check the other end that the correct data was written.
        use std::os::fd::AsRawFd;
        use tokio::io::AsyncReadExt;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .worker_threads(1)
            .build()
            .unwrap();
        let pool = runtime.handle();
        let mut cache_data = CacheWrite::new();
        let data = b"test data";
        cache_data.put_bytes("test_key", data).unwrap();
        let cache_read =
            CacheRead::from(std::io::Cursor::new(cache_data.finish().unwrap())).unwrap();
        crate::util::block_on(&runtime, async {
            let (sender, mut receiver) = tokio::net::unix::pipe::pipe().unwrap();
            let sender_fd = sender.into_blocking_fd().unwrap();
            let raw_fd = sender_fd.as_raw_fd();
            let fd_path = PathBuf::from(format!("/dev/fd/{raw_fd}"));
            let objects = vec![FileObjectSource {
                key: "test_key".to_string(),
                dir: fd_path.parent().unwrap().to_path_buf(),
                path: fd_path.clone(),
                optional: false,
            }];
            // On FreeBSD, `/dev/fd/{fd}` does not always exist (i.e. without mounting `fdescfs`), so we skip this test if we get `ENOENT`.
            if !fd_path.exists() {
                info!(
                    "Skipping test_extract_object_to_dev_fd_something because /dev/fd/{raw_fd} does not exist"
                );
                return;
            }
            let result = cache_read.extract_objects(objects, pool).await;
            assert!(
                result.is_ok(),
                "Extracting to /dev/fd/{raw_fd} should succeed"
            );
            let mut buf = vec![0; data.len()];
            let n = receiver.read_exact(&mut buf).await.unwrap();
            assert_eq!(n, data.len(), "Read the correct number of bytes");
            assert_eq!(buf, data, "Read the correct data from /dev/fd/{raw_fd}");
        });
    }

    #[test]
    fn test_extract_object_to_non_writable_path() {
        // See `test_extract_object_to_dev_fd_something`: we still cannot cover all platforms by the other tests. Here we test a more portable case of creating a file and making its parent directory non-writable, in which case we should still be able to extract the object successfully.

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .worker_threads(1)
            .build()
            .unwrap();

        let pool = runtime.handle();

        let mut cache_data = CacheWrite::new();
        cache_data.put_bytes("test_key", b"real_test_data").unwrap();
        let cache_read =
            CacheRead::from(std::io::Cursor::new(cache_data.finish().unwrap())).unwrap();

        let tmpdir = tempfile::tempdir().unwrap();
        let target_path = tmpdir.path().join("test_file");
        std::fs::write(&target_path, b"test").unwrap();
        // The current Rust fs permissions API is kind of awkward...
        let mut perm = tmpdir.path().metadata().unwrap().permissions();
        perm.set_readonly(true);
        std::fs::set_permissions(tmpdir.path(), perm.clone()).unwrap();
        // Note that this doesn't guarantee that the a new file cannot be created anymore.
        // For example, as documented in `std::fs::Permissions::set_readonly`, the
        // `FILE_ATTRIBUTE_READONLY` attribute on Windows is entirely ignored for directories.
        // std::fs::File::create(tmpdir.path().join("another_file")).unwrap_err();

        let objects = vec![FileObjectSource {
            key: "test_key".to_string(),
            dir: target_path.parent().unwrap().to_path_buf(),
            path: target_path.clone(),
            optional: false,
        }];

        let result = crate::util::block_on(&runtime, cache_read.extract_objects(objects, pool));
        assert!(
            result.is_ok(),
            "Extracting to the target path should succeed"
        );
        // Test the content; make sure the old content is overwritten
        let content = std::fs::read(&target_path).unwrap();
        assert_eq!(
            content, b"real_test_data",
            "Extracted content should be correct"
        );

        // `tempfile` needs us to reset permissions for cleanup to work
        #[allow(
            clippy::permissions_set_readonly_false,
            reason = "The affected directory is immediately deleted with no security implications"
        )]
        perm.set_readonly(false);
        std::fs::set_permissions(tmpdir.path(), perm).unwrap();
        tmpdir.close().unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn test_extract_object_to_nul_works() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .worker_threads(1)
            .build()
            .unwrap();

        let pool = runtime.handle();

        let cache_data = CacheWrite::new();
        let cache_read =
            CacheRead::from(std::io::Cursor::new(cache_data.finish().unwrap())).unwrap();

        let objects = vec![FileObjectSource {
            key: "test_key".to_string(),
            dir: Path::new(".").to_path_buf(),
            path: PathBuf::from("NUL"),
            optional: false,
        }];

        let result = crate::util::block_on(&runtime, cache_read.extract_objects(objects, pool));
        assert!(result.is_ok(), "Extracting to NUL should succeed");
    }
}
