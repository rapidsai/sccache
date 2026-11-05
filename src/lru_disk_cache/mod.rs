pub mod lru_cache;

use fs::File;
use fs_err as fs;
use std::borrow::Borrow;
use std::boxed::Box;
use std::collections::hash_map::RandomState;
use std::error::Error as StdError;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::hash::BuildHasher;
use std::io;
use std::io::prelude::*;
use std::path::{Path, PathBuf};

use filetime::{FileTime, set_file_times};
pub use lru_cache::{LruCache, Meter};
use tempfile::{NamedTempFile, TempDir};
use walkdir::WalkDir;

use crate::cache::BufReadSeek;
use crate::util::OsStrExt;

const TEMPFILE_PREFIX: &str = ".sccachetmp";

struct FileSize;

/// Given a tuple of (path, filesize), use the filesize for measurement.
impl<K> Meter<K, u64> for FileSize {
    type Measure = usize;
    fn measure<Q: ?Sized>(&self, _: &Q, v: &u64) -> usize
    where
        K: Borrow<Q>,
    {
        *v as usize
    }
}

/// Normalize key `abcdef` into `a/b/abcdef`
fn normalize_key<K: AsRef<OsStr>>(key: K) -> PathBuf {
    // TODO: Do we care if &OsStr key has non-UTF8 characters?
    let str = key.as_ref().to_string_lossy();
    let mut chars = str.chars();
    PathBuf::new()
        .join(chars.next().unwrap_or('_').to_string())
        .join(chars.next().unwrap_or('_').to_string())
        .join(key.as_ref())
}

fn get_entry_size<P: AsRef<Path>>(path: P) -> u64 {
    WalkDir::new(path.as_ref())
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .map(|meta| meta.len())
        .sum()
}

fn remove_entry_from_disk<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if path.is_dir() {
        fs::remove_dir_all(path).map_err(Error::Io)
    } else if path.is_file() || path.is_symlink() {
        fs::remove_file(path).map_err(Error::Io)
    } else {
        Ok(())
    }
}

pub async fn remove_entry_from_disk_async<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if path.is_dir() {
        fs::remove_dir_all(path).map_err(Error::Io)
    } else if path.is_file() || path.is_symlink() {
        fs::remove_file(path).map_err(Error::Io)
    } else {
        Ok(())
    }
}

/// Return an iterator of `(path, size)` of entries under `path` sorted by ascending last-modified
/// time, such that the oldest modified entry is returned first.
fn get_all_entries<P: AsRef<Path>>(path: P) -> Box<dyn Iterator<Item = (PathBuf, u64)>> {
    let mut entries: Vec<_> = WalkDir::new(path.as_ref())
        // min 1/max 3 because:
        //   <path>/a/b/abcdef
        // 0 -----^ ^ ^ ^
        // 1 -------' | |
        // 2 ---------' |
        // 3 -----------'
        .min_depth(1)
        .max_depth(3)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            if e.depth() == 3
                || e.file_type().is_file()
                || e.file_name().starts_with(TEMPFILE_PREFIX)
            {
                Some(e)
            } else {
                None
            }
        })
        .filter_map(|e| e.metadata().ok().map(|m| (e.path().to_owned(), m)))
        .filter_map(|(path, meta)| {
            meta.modified()
                .ok()
                .map(|mtime| (mtime, get_entry_size(&path), path))
        })
        .collect();
    // Sort by last-modified-time, so oldest entry first.
    entries.sort_by_key(|k| k.0);
    Box::new(entries.into_iter().map(|(_mtime, size, path)| (path, size)))
}

/// An LRU cache of files on disk.
pub struct LruDiskCache<S: BuildHasher = RandomState> {
    lru: LruCache<OsString, u64, S, FileSize>,
    root: PathBuf,
    pending: Vec<OsString>,
    pending_size: u64,
}

/// Errors returned by this crate.
#[derive(Debug)]
pub enum Error {
    /// The file was too large to fit in the cache.
    FileTooLarge,
    /// The file was not in the cache.
    FileNotInCache,
    /// An IO Error occurred.
    Io(io::Error),
    /// An unknown Error occurred.
    Any(anyhow::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Any(e) => write!(f, "{e}"),
            Error::FileTooLarge => write!(f, "File too large"),
            Error::FileNotInCache => write!(f, "File not in cache"),
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Any(e) => e.source(),
            Error::FileTooLarge => None,
            Error::FileNotInCache => None,
            Error::Io(e) => Some(e),
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Error {
        Error::Io(e)
    }
}

impl From<anyhow::Error> for Error {
    fn from(e: anyhow::Error) -> Error {
        match e.downcast::<io::Error>() {
            Ok(e) => Error::Io(e),
            Err(e) => Error::Any(e),
        }
    }
}

/// A convenience `Result` type
pub type Result<T, E = Error> = std::result::Result<T, E>;

enum AddFile<'a> {
    AbsPath(PathBuf),
    RelPath(&'a Path),
}

pub struct LruDiskCacheAddEntry {
    file: NamedTempFile,
    key: OsString,
    size: u64,
}

impl LruDiskCacheAddEntry {
    pub fn as_path(&self) -> &Path {
        self.file.path()
    }
    pub fn as_file_mut(&mut self) -> &mut std::fs::File {
        self.file.as_file_mut()
    }
}

pub struct LruDiskCacheDirEntry {
    file: TempDir,
    key: OsString,
    size: u64,
}

impl LruDiskCacheDirEntry {
    pub fn as_path(&self) -> &Path {
        self.file.path()
    }
}

impl LruDiskCache {
    /// Create an `LruDiskCache` that stores files in `path`, limited to `size` bytes.
    ///
    /// Existing files in `path` will be stored with their last-modified time from the filesystem
    /// used as the order for the recency of their use. Any files that are individually larger
    /// than `size` bytes will be removed.
    ///
    /// The cache is not observant of changes to files under `path` from external sources, it
    /// expects to have sole maintence of the contents.
    pub fn new<T>(path: T, size: u64) -> Result<Self>
    where
        PathBuf: From<T>,
    {
        LruDiskCache {
            lru: LruCache::with_meter(size, FileSize),
            root: PathBuf::from(path),
            pending: vec![],
            pending_size: 0,
        }
        .init()
    }

    /// Return the current size of all the files in the cache.
    pub fn size(&self) -> u64 {
        self.lru.size() + self.pending_size
    }

    /// Return the count of entries in the cache.
    pub fn len(&self) -> usize {
        self.lru.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lru.len() == 0
    }

    /// Return the maximum size of the cache.
    pub fn capacity(&self) -> u64 {
        self.lru.capacity()
    }

    /// Return the path in which the cache is stored.
    pub fn path(&self) -> &Path {
        self.root.as_path()
    }

    /// Make a path to the cache entry with key `key`
    /// Normalize key `abcdef` into `a/b/abcdef`
    pub fn key_to_rel_path<K: AsRef<OsStr>>(&self, key: K) -> PathBuf {
        // Always &OsStr, so normalize_key is compiled once whatever K is.
        normalize_key(key.as_ref())
    }

    /// Return the path that `key` would be stored at.
    pub fn key_to_abs_path<K: AsRef<OsStr>>(&self, key: K) -> PathBuf {
        self.rel_to_abs_path(self.key_to_rel_path(key))
    }

    /// Canonicalize the relative path for `key`.
    pub fn rel_to_abs_path<K: AsRef<Path>>(&self, rel_path: K) -> PathBuf {
        self.root.join(rel_path)
    }

    /// Scan `self.root` for existing files and store them.
    fn init(mut self) -> Result<Self> {
        fs::create_dir_all(&self.root)?;
        for (path, size) in get_all_entries(&self.root) {
            if path
                .file_name()
                .expect("Bad path?")
                .starts_with(TEMPFILE_PREFIX)
            {
                remove_entry_from_disk(&path).unwrap_or_else(|e| {
                    error!("LruDiskCache: Error removing temporary entry {path:?}:\n{e:?}");
                });
            } else if !self.can_store(size) {
                warn!(
                    "LruDiskCache: Deleting entry that is too large for the cache at max_size={size}b: {path:?}"
                );
                remove_entry_from_disk(&path).unwrap_or_else(|e| {
                    error!("LruDiskCache: Error removing entry {path:?}:\n{e:?}");
                });
            } else {
                self.add_file(AddFile::AbsPath(path.clone()), size)
                    .unwrap_or_else(|e| {
                        error!("LruDiskCache: Error adding entry {path:?}:\n{e:?}");
                    });
            }
        }
        Ok(self)
    }

    /// Returns `true` if the disk cache can store a file of `size` bytes.
    pub fn can_store(&self, size: u64) -> bool {
        size <= self.lru.capacity()
    }

    fn make_space(&mut self, size: u64) -> Result<()> {
        if !self.can_store(size) {
            return Err(Error::FileTooLarge);
        }
        // TODO: ideally LRUCache::insert would give us back the entries it had to remove.
        while self.size() + size > self.capacity() {
            let (rel_path, _) = self.lru.remove_lru().expect("Unexpectedly empty cache!");
            let abs_path = self.rel_to_abs_path(rel_path);
            // TODO: check that files are removable during `init`, so that this is only
            // due to outside interference.
            remove_entry_from_disk(&abs_path).unwrap_or_else(|e| {
                if let Error::Io(e) = e {
                    // Sometimes the file has already been removed
                    // this seems to happen when the max cache size has been reached
                    // https://github.com/mozilla/sccache/issues/2092
                    if e.kind() == std::io::ErrorKind::NotFound {
                        debug!(
                            "Error removing file from cache as it was not found: `{abs_path:?}`"
                        );
                    } else {
                        error!("LruDiskCache: Error removing entry {abs_path:?}:\n{e:?}");
                    }
                }
            });
        }
        Ok(())
    }

    async fn make_space_async(&mut self, size: u64) -> Result<()> {
        if !self.can_store(size) {
            return Err(Error::FileTooLarge);
        }
        // TODO: ideally LRUCache::insert would give us back the entries it had to remove.
        while self.size() + size > self.capacity() {
            let (rel_path, _) = self.lru.remove_lru().expect("Unexpectedly empty cache!");
            let abs_path = self.rel_to_abs_path(rel_path);
            // TODO: check that files are removable during `init`, so that this is only
            // due to outside interference.
            remove_entry_from_disk_async(&abs_path)
                .await
                .unwrap_or_else(|e| {
                    error!("LruDiskCache: Error removing entry {abs_path:?}:\n{e:?}");
                });
        }
        Ok(())
    }

    /// Add the file at `path` of size `size` to the cache.
    fn add_file(&mut self, addfile_path: AddFile<'_>, size: u64) -> Result<()> {
        let rel_path = match addfile_path {
            AddFile::AbsPath(ref p) => p.strip_prefix(&self.root).expect("Bad path?").as_os_str(),
            AddFile::RelPath(p) => p.as_os_str(),
        };
        self.make_space(size)?;
        self.lru.insert(rel_path.to_owned(), size);
        Ok(())
    }

    fn insert_by<K: AsRef<OsStr>, F: FnOnce(&Path) -> io::Result<()>>(
        &mut self,
        key: K,
        size: Option<u64>,
        by: F,
    ) -> Result<()> {
        if let Some(size) = size
            && !self.can_store(size)
        {
            return Err(Error::FileTooLarge);
        }
        let rel_path = self.key_to_rel_path(key);
        let abs_path = self.rel_to_abs_path(&rel_path);
        fs::create_dir_all(abs_path.parent().expect("Bad path?"))?;
        by(&abs_path)?;
        let size = match size {
            Some(size) => size,
            None => fs::metadata(&abs_path)?.len(),
        };
        self.add_file(AddFile::RelPath(&rel_path), size)
            .map_err(|e| {
                error!("LruDiskCache: Failed to insert entry {rel_path:?}:\n{e:?}");
                remove_entry_from_disk(&abs_path).unwrap_or_else(|e| {
                    error!(
                        "LruDiskCache: Error removing entry for failed insertion {abs_path:?}:\n{e:?}"
                    );
                });
                e
            })
    }

    fn cleanup_pending<'a, K: AsRef<OsStr> + 'a>(&mut self, key: K, size: u64) {
        self.pending
            .iter()
            .position(|k| k == key.as_ref())
            .map(|i| self.pending.remove(i))
            .unwrap();
        self.pending_size -= size;
    }

    /// Add an entry by calling `with_fn` with the path for `key`.
    pub async fn insert_with<K, F, Fut>(
        &mut self,
        key: K,
        size: u64,
        with_fn: F,
    ) -> Result<(PathBuf, u64)>
    where
        K: AsRef<OsStr>,
        F: FnOnce(&Path) -> Fut,
        Fut: std::future::Future<Output = io::Result<u64>>,
    {
        self.make_space_async(size).await?;
        let rel_path = self.key_to_rel_path(key);
        let abs_path = self.rel_to_abs_path(&rel_path);
        match with_fn(&abs_path).await {
            Ok(_) => {
                self.lru.insert(rel_path.into_os_string(), size);
                Ok((abs_path, size))
            }
            Err(err) => {
                error!("LruDiskCache: Failed to insert entry {rel_path:?}:\n{err:?}");
                remove_entry_from_disk_async(&abs_path).await.unwrap_or_else(|e| {
                    error!(
                        "LruDiskCache: Error removing entry for failed insertion {abs_path:?}:\n{e:?}"
                    );
                });
                Err(Error::Io(err))
            }
        }
    }

    /// Add a file with `bytes` as its contents to the cache at path `key`.
    pub fn insert_bytes<K: AsRef<OsStr>>(&mut self, key: K, bytes: &[u8]) -> Result<()> {
        self.insert_by(key, Some(bytes.len() as u64), |path| {
            let mut f = File::create(path)?;
            f.write_all(bytes)?;
            Ok(())
        })
    }

    /// Add an existing file at `path` to the cache at path `key`.
    pub fn insert_file<K: AsRef<OsStr>, P: AsRef<OsStr>>(&mut self, key: K, path: P) -> Result<()> {
        let size = fs::metadata(path.as_ref())?.len();
        self.insert_by(key, Some(size), |new_path| {
            fs::rename(path.as_ref(), new_path).or_else(|_| {
                warn!("fs::rename failed, falling back to copy!");
                fs::copy(path.as_ref(), new_path)?;
                remove_entry_from_disk(path.as_ref()).unwrap_or_else(|e| {
                    error!(
                        "LruDiskCache: Error removing entry for failed insertion {:?}:\n{e:?}",
                        path.as_ref()
                    );
                });
                Ok(())
            })
        })
    }

    /// Prepare the insertion of a file at path `key`. The resulting entry must be
    /// committed with `LruDiskCache::commit`.
    pub async fn prepare_add<'a, K: AsRef<OsStr> + 'a>(
        &mut self,
        key: K,
        size: u64,
    ) -> Result<LruDiskCacheAddEntry> {
        // Ensure we have enough space for the advertized space.
        self.make_space_async(size).await?;
        let key = self.key_to_rel_path(key).into_os_string();
        self.pending.push(key.clone());
        self.pending_size += size;
        crate::util::async_fs::create_dir_all(&self.root).await?;
        crate::util::tempfile_with_prefix_in(&self.root, TEMPFILE_PREFIX)
            .map(|file| LruDiskCacheAddEntry { file, key, size })
            .map_err(Into::into)
    }

    /// Prepare the insertion of a directory at path `key`. The resulting entry must be
    /// committed with `LruDiskCache::commit_dir`.
    pub async fn prepare_dir<'a, K: AsRef<OsStr> + 'a>(
        &mut self,
        key: K,
        size: u64,
    ) -> Result<LruDiskCacheDirEntry> {
        // Ensure we have enough space for the advertized space.
        self.make_space_async(size).await?;
        let key = self.key_to_rel_path(key).into_os_string();
        self.pending.push(key.clone());
        self.pending_size += size;
        crate::util::async_fs::create_dir_all(&self.root).await?;
        crate::util::tempdir_with_prefix_in(&self.root, TEMPFILE_PREFIX)
            .map(|file| LruDiskCacheDirEntry { file, key, size })
            .map_err(Into::into)
    }

    /// Commit an entry coming from `LruDiskCache::prepare_add`.
    pub async fn commit(&mut self, entry: LruDiskCacheAddEntry) -> Result<()> {
        let LruDiskCacheAddEntry {
            mut file,
            key,
            size,
        } = entry;
        file.flush()?;
        let real_size = file.as_file().metadata()?.len();
        // If the file is larger than the size that had been advertized, ensure
        // we have enough space for it.
        self.make_space_async(real_size.saturating_sub(size))
            .await?;
        self.cleanup_pending(&key, size);
        let abs_path = self.rel_to_abs_path(&key);
        if let Some(parent) = abs_path.parent() {
            crate::util::async_fs::create_dir_all(parent).await?;
        }
        file.persist(&abs_path).map_err(|e| e.error)?;
        self.lru.insert(key, real_size);
        trace!("LruDiskCache::commit({abs_path:?})");
        Ok(())
    }

    /// Commit an entry coming from `LruDiskCache::prepare_dir`.
    pub async fn commit_dir(
        &mut self,
        entry: Result<(u64, &LruDiskCacheDirEntry), &LruDiskCacheDirEntry>,
    ) -> Result<(PathBuf, u64)> {
        match entry {
            Err(entry) => {
                self.cleanup_pending(&entry.key, entry.size);
                Err(Error::FileNotInCache)
            }
            Ok((real_size, entry)) => {
                self.cleanup_pending(&entry.key, entry.size);
                let abs_path = self.rel_to_abs_path(&entry.key);
                if let Some(parent) = abs_path.parent() {
                    crate::util::async_fs::create_dir_all(parent).await?;
                }
                // If the dir is larger than the size that had been advertized, ensure
                // we have enough space for it.
                self.make_space_async(real_size.saturating_sub(entry.size))
                    .await?;
                crate::util::async_fs::rename(entry.as_path(), &abs_path).await?;
                self.lru.insert(entry.key.clone(), real_size);
                Ok((abs_path, real_size))
            }
        }
    }

    /// Return `true` if a file with path `key` is in the cache. Entries created
    /// by `LruDiskCache::prepare_add` but not yet committed return `false`.
    pub fn contains_key<K: AsRef<OsStr>>(&self, key: K) -> bool {
        self.lru.contains_key(self.key_to_rel_path(key).as_os_str())
    }

    /// Get the size of an LRU entry at `key`, if one exists.
    /// Updates the LRU state of the entry if present.
    ///
    /// If the LRU doesn't have an entry for `key`, but there is a file or dir
    /// on disk at that path, an entry for `key` is inserted and returned.
    /// This can occur when two sccache processes share the same cache dir.
    ///
    /// Entries created by `LruDiskCache::prepare_add` but not yet committed return
    /// `Err(Error::FileNotInCache)`.
    pub fn get_size<K: AsRef<OsStr>>(&mut self, key: K) -> Result<u64> {
        let rel_path = self.key_to_rel_path(&key);
        let abs_path = self.rel_to_abs_path(&rel_path);
        let path_key = rel_path.as_os_str();
        let size = if abs_path.try_exists().is_ok_and(|e| e) {
            if let Some(size) = self.lru.get(path_key) {
                Ok(*size)
            } else {
                let size = get_entry_size(&abs_path);
                self.insert_by(&key, Some(size), |_| Ok(())).map(|_| size)
            }
        } else {
            let _ = self.lru.remove(path_key);
            return Err(Error::FileNotInCache);
        };

        size.and_then(|size| {
            let t = FileTime::now();
            set_file_times(&abs_path, t, t)?;
            Ok(size)
        })
    }

    /// Get an opened `File` for `key`, if one exists and can be opened. Updates the LRU state
    /// of the file if present. Avoid using this method if at all possible, prefer `.get`.
    /// Entries created by `LruDiskCache::prepare_add` but not yet committed return
    /// `Err(Error::FileNotInCache)`.
    pub fn get_file<K: AsRef<OsStr>>(&mut self, key: K) -> Result<File> {
        self.get_size(key.as_ref())
            .and_then(|_| File::open(self.key_to_abs_path(key)).map_err(Into::into))
    }

    /// Get an opened readable and seekable handle to the file at `key`, if one exists and can
    /// be opened. Updates the LRU state of the file if present.
    /// Entries created by `LruDiskCache::prepare_add` but not yet committed return
    /// `Err(Error::FileNotInCache)`.
    pub fn get<K: AsRef<OsStr>>(&mut self, key: K) -> Result<Box<dyn BufReadSeek>> {
        self.get_file(key)
            .map(|f| Box::new(std::io::BufReader::new(f)) as Box<dyn BufReadSeek>)
    }

    /// Return the absolute path for `key` if it is present in the cache, without opening the
    /// file. Updates the LRU eviction order. Returns `None` if the key is absent.
    pub fn get_abs_path<K: AsRef<OsStr>>(&mut self, key: K) -> Option<PathBuf> {
        let key = key.as_ref();
        let path = self.key_to_abs_path(key);
        self.lru.get(key).map(|_| path)
    }

    /// Remove the given key from the cache.
    pub async fn remove<K: AsRef<OsStr>>(&mut self, key: K) -> Result<()> {
        let rel_path = self.key_to_rel_path(key);
        match self.lru.remove(rel_path.as_os_str()) {
            Some(_) => {
                let abs_path = self.rel_to_abs_path(rel_path);
                remove_entry_from_disk_async(&abs_path).await.map_err(|e| {
                    error!("LruDiskCache: Error removing entry {abs_path:?}:\n{e:?}");
                    e
                })
            }
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fs::{self, File};
    use super::{Error, LruDiskCache, LruDiskCacheAddEntry, get_all_entries, normalize_key};
    use crate::test::utils::*;

    use filetime::{FileTime, set_file_times};
    use std::io::{self, Read, Write};
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    struct TestFixture {
        /// Temp directory.
        pub tempdir: TempDir,
    }

    fn create_file<T: AsRef<Path>, F: FnOnce(File) -> io::Result<()>>(
        dir: &Path,
        path: T,
        fill_contents: F,
    ) -> io::Result<PathBuf> {
        let b = dir.join(normalize_key(path.as_ref()));
        fs::create_dir_all(b.parent().unwrap())?;
        let f = fs::File::create(&b)?;
        fill_contents(f)?;
        dunce::canonicalize(b)
    }

    /// Set the last modified time of `path` backwards by `seconds` seconds.
    fn set_mtime_back<T: AsRef<Path>>(path: T, seconds: usize) {
        let m = fs::metadata(path.as_ref()).unwrap();
        let t = FileTime::from_last_modification_time(&m);
        let t = FileTime::from_unix_time(t.unix_seconds() - seconds as i64, t.nanoseconds());
        set_file_times(path, t, t).unwrap();
    }

    fn read_all<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {
        let mut v = vec![];
        r.read_to_end(&mut v)?;
        Ok(v)
    }

    impl TestFixture {
        pub fn new() -> TestFixture {
            TestFixture {
                tempdir: crate::util::temp_dir().unwrap(),
            }
        }

        pub fn tmp(&self) -> &Path {
            self.tempdir.path()
        }

        pub fn create_file<T: AsRef<Path>>(&self, path: T, size: usize) -> PathBuf {
            create_file(self.tempdir.path(), path, |mut f| {
                f.write_all(&vec![0; size])
            })
            .unwrap()
        }
    }

    #[test]
    fn test_empty_dir() {
        let f = TestFixture::new();
        LruDiskCache::new(f.tmp(), 1024).unwrap();
    }

    #[test]
    fn test_missing_root() {
        let f = TestFixture::new();
        LruDiskCache::new(f.tmp().join(normalize_key("not-here")), 1024).unwrap();
    }

    #[test]
    fn test_some_existing_files() {
        let f = TestFixture::new();
        f.create_file("file1", 10);
        f.create_file("file2", 10);
        let c = LruDiskCache::new(f.tmp(), 20).unwrap();
        assert_eq!(c.size(), 20);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn test_existing_file_too_large() {
        let f = TestFixture::new();
        // Create files explicitly in the past.
        set_mtime_back(f.create_file("file1", 10), 10);
        set_mtime_back(f.create_file("file2", 10), 5);
        let c = LruDiskCache::new(f.tmp(), 15).unwrap();
        assert_eq!(c.size(), 10);
        assert_eq!(c.len(), 1);
        assert!(!c.contains_key("file1"));
        assert!(c.contains_key("file2"));
    }

    #[test]
    fn test_existing_files_lru_mtime() {
        let f = TestFixture::new();
        // Create files explicitly in the past.
        set_mtime_back(f.create_file("file1", 10), 5);
        set_mtime_back(f.create_file("file2", 10), 10);
        let mut c = LruDiskCache::new(f.tmp(), 25).unwrap();
        assert_eq!(c.size(), 20);
        c.insert_bytes("file3", &[0; 10]).unwrap();
        assert_eq!(c.size(), 20);
        // The oldest file on disk should have been removed.
        assert!(!c.contains_key("file2"));
        assert!(c.contains_key("file1"));
    }

    #[test]
    fn test_insert_bytes() {
        let f = TestFixture::new();
        let mut c = LruDiskCache::new(f.tmp(), 25).unwrap();
        c.insert_bytes("abc", &[0; 10]).unwrap();
        assert!(c.contains_key("abc"));
        c.insert_bytes("abd", &[0; 10]).unwrap();
        assert_eq!(c.size(), 20);
        // Adding this third file should put the cache above the limit.
        c.insert_bytes("xyz", &[0; 10]).unwrap();
        assert_eq!(c.size(), 20);
        // The least-recently-used file should have been removed.
        assert!(!c.contains_key("abc"));
        assert!(!f.tmp().join(normalize_key("abc")).exists());
    }

    #[test]
    fn test_insert_bytes_exact() {
        // Test that files adding up to exactly the size limit works.
        let f = TestFixture::new();
        let mut c = LruDiskCache::new(f.tmp(), 20).unwrap();
        c.insert_bytes("file1", &[1; 10]).unwrap();
        c.insert_bytes("file2", &[2; 10]).unwrap();
        assert_eq!(c.size(), 20);
        c.insert_bytes("file3", &[3; 10]).unwrap();
        assert_eq!(c.size(), 20);
        assert!(!c.contains_key("file1"));
    }

    #[test]
    fn test_add_get_lru() {
        let f = TestFixture::new();
        {
            let mut c = LruDiskCache::new(f.tmp(), 25).unwrap();
            c.insert_bytes("file1", &[1; 10]).unwrap();
            c.insert_bytes("file2", &[2; 10]).unwrap();
            // Get the file to bump its LRU status.
            assert_eq!(
                read_all(&mut c.get("file1").unwrap()).unwrap(),
                vec![1u8; 10]
            );
            // Adding this third file should put the cache above the limit.
            c.insert_bytes("file3", &[3; 10]).unwrap();
            assert_eq!(c.size(), 20);
            // The least-recently-used file should have been removed.
            assert!(!c.contains_key("file2"));
        }
        // Get rid of the cache, to test that the LRU persists on-disk as mtimes.
        // This is hacky, but mtime resolution on my mac with HFS+ is only 1 second, so we either
        // need to have a 1 second sleep in the test (boo) or adjust the mtimes back a bit so
        // that updating one file to the current time actually works to make it newer.
        set_mtime_back(f.tmp().join(normalize_key("file1")), 5);
        set_mtime_back(f.tmp().join(normalize_key("file3")), 5);
        {
            let mut c = LruDiskCache::new(f.tmp(), 25).unwrap();
            // Bump file1 again.
            c.get("file1").unwrap();
        }
        // Now check that the on-disk mtimes were updated and used.
        {
            let mut c = LruDiskCache::new(f.tmp(), 25).unwrap();
            assert!(c.contains_key("file1"));
            assert!(c.contains_key("file3"));
            assert_eq!(c.size(), 20);
            // Add another file to bump out the least-recently-used.
            c.insert_bytes("file4", &[4; 10]).unwrap();
            assert_eq!(c.size(), 20);
            assert!(!c.contains_key("file3"));
            assert!(c.contains_key("file1"));
        }
    }

    #[test]
    fn test_insert_bytes_too_large() {
        let f = TestFixture::new();
        let mut c = LruDiskCache::new(f.tmp(), 1).unwrap();
        match c.insert_bytes("abc", &[0; 2]) {
            Err(Error::FileTooLarge) => {}
            x => panic!("Unexpected result: {x:?}"),
        }
    }

    #[test]
    fn test_insert_file() {
        let f = TestFixture::new();
        let p1 = f.create_file("file1", 10);
        let p2 = f.create_file("file2", 10);
        let p3 = f.create_file("file3", 10);
        let mut c = LruDiskCache::new(f.tmp().join("cache"), 25).unwrap();
        c.insert_file("file1", &p1).unwrap();
        assert_eq!(c.len(), 1);
        c.insert_file("file2", &p2).unwrap();
        assert_eq!(c.len(), 2);
        // Get the file to bump its LRU status.
        assert_eq!(
            read_all(&mut c.get("file1").unwrap()).unwrap(),
            vec![0u8; 10]
        );
        // Adding this third file should put the cache above the limit.
        c.insert_file("file3", &p3).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c.size(), 20);
        // The least-recently-used file should have been removed.
        assert!(!c.contains_key("file2"));
        assert!(!p1.exists());
        assert!(!p2.exists());
        assert!(!p3.exists());
    }

    #[test]
    fn test_prepare_and_commit() {
        let f = TestFixture::new();
        let cache_dir = f.tmp();
        let mut c = LruDiskCache::new(cache_dir, 25).unwrap();
        let mut tmp = c.prepare_add("abc", 10).wait().unwrap();
        // An entry added but not committed doesn't count, except for the
        // (reserved) size of the disk cache.
        assert!(!c.contains_key("abc"));
        assert_eq!(c.size(), 10);
        assert_eq!(c.lru.size(), 0);
        tmp.as_file_mut().write_all(&[0; 10]).unwrap();
        c.commit(tmp).wait().unwrap();
        // Once committed, the file appears.
        assert!(c.contains_key("abc"));
        assert_eq!(c.size(), 10);
        assert_eq!(c.lru.size(), 10);

        let mut tmp = c.prepare_add("abd", 10).wait().unwrap();
        assert_eq!(c.size(), 20);
        assert_eq!(c.lru.size(), 10);
        // Even though we haven't committed the second file, preparing for
        // the addition of the third one should put the cache above the
        // limit and trigger cleanup.
        let mut tmp2 = c.prepare_add("xyz", 10).wait().unwrap();
        assert_eq!(c.size(), 20);
        assert_eq!(c.lru.size(), 0);
        // At this point, we expect the first entry to have been removed entirely.
        assert!(!c.contains_key("abc"));
        assert!(!f.tmp().join(normalize_key("abc")).exists());
        tmp.as_file_mut().write_all(&[0; 10]).unwrap();
        tmp2.as_file_mut().write_all(&[0; 10]).unwrap();
        c.commit(tmp).wait().unwrap();
        assert_eq!(c.size(), 20);
        assert_eq!(c.lru.size(), 10);
        c.commit(tmp2).wait().unwrap();
        assert_eq!(c.size(), 20);
        assert_eq!(c.lru.size(), 20);

        let mut tmp = c.prepare_add("abc", 5).wait().unwrap();
        assert_eq!(c.size(), 25);
        assert_eq!(c.lru.size(), 20);
        // Committing a file bigger than the promised size should properly
        // handle the case where the real size makes the cache go over the limit.
        tmp.as_file_mut().write_all(&[0; 10]).unwrap();
        c.commit(tmp).wait().unwrap();
        assert_eq!(c.size(), 20);
        assert_eq!(c.lru.size(), 20);
        assert!(!c.contains_key("abd"));
        assert!(!f.tmp().join(normalize_key("abd")).exists());

        // If for some reason, the cache still contains a temporary file on
        // initialization, the temporary file is removed.
        let LruDiskCacheAddEntry { file, .. } = c.prepare_add("abd", 5).wait().unwrap();
        let (_, path) = file.keep().unwrap();
        std::mem::drop(c);
        // Ensure that the temporary file is indeed there.
        assert!(get_all_entries(cache_dir).any(|(file, _)| file == path));
        LruDiskCache::new(cache_dir, 25).unwrap();
        // The temporary file should not be there anymore.
        assert!(get_all_entries(cache_dir).all(|(file, _)| file != path));
    }

    #[tokio::test]
    async fn test_remove() {
        let f = TestFixture::new();
        let p1 = f.create_file("file1", 10);
        let p2 = f.create_file("file2", 10);
        let p3 = f.create_file("file3", 10);
        let mut c = LruDiskCache::new(f.tmp().join("cache"), 25).unwrap();
        c.insert_file("file1", &p1).unwrap();
        c.insert_file("file2", &p2).unwrap();
        c.remove("file1").await.unwrap();
        c.insert_file("file3", &p3).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c.size(), 20);

        // file1 should have been removed.
        assert!(!c.contains_key("file1"));
        assert!(!f.tmp().join("cache").join(normalize_key("file1")).exists());
        assert!(f.tmp().join("cache").join(normalize_key("file2")).exists());
        assert!(f.tmp().join("cache").join(normalize_key("file3")).exists());
        assert!(!p1.exists());
        assert!(!p2.exists());
        assert!(!p3.exists());

        let p4 = f.create_file("file1", 10);
        c.insert_file("file1", &p4).unwrap();
        assert_eq!(c.len(), 2);
        // file2 should have been removed.
        assert!(c.contains_key("file1"));
        assert!(!c.contains_key("file2"));
        assert!(!f.tmp().join("cache").join(normalize_key("file2")).exists());
        assert!(!p4.exists());
    }
}
