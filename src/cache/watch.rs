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
use notify_debouncer_full as notify;
use notify_debouncer_full::{
    DebounceEventResult, RecommendedCache, new_debouncer,
    notify::{RecommendedWatcher, RecursiveMode},
};
use std::{ffi::OsString, future::Future, path::PathBuf, pin::Pin, sync::Arc, time::Duration};

use crate::{
    cache::{Cache, Storage},
    config::CacheMode,
    errors::*,
};

pub struct WatchStorage {
    basedirs: Vec<Vec<u8>>,
    cache_type_name: &'static str,
    storage: Arc<futures::lock::Mutex<Arc<dyn Storage>>>,
    #[allow(dead_code)]
    watcher: notify::Debouncer<RecommendedWatcher, RecommendedCache>,
}

impl WatchStorage {
    pub async fn from<T>(create: T, paths: &[PathBuf]) -> Result<Arc<dyn Storage>>
    where
        T: Fn() -> Pin<Box<dyn Future<Output = Result<Arc<dyn Storage>>> + Send>> + Send + 'static,
    {
        let storage = create().await?;
        if !storage.enabled() {
            return Ok(storage);
        }
        let paths = Self::dirs_and_paths(paths);
        if paths.is_empty() {
            trace!(
                "No paths to watch for {} storage",
                storage.cache_type_name()
            );
            return Ok(storage);
        }
        let basedirs = storage.basedirs().to_vec();
        let cache_type_name = storage.cache_type_name();
        let storage = Arc::new(futures::lock::Mutex::new(storage));
        let watcher = Self::watch(create, storage.clone(), paths)?;
        Ok(Arc::new(Self {
            basedirs,
            cache_type_name,
            storage,
            watcher,
        }))
    }

    fn dirs_and_paths(paths: &[PathBuf]) -> Vec<(PathBuf, Vec<OsString>)> {
        trace!("Filtering for {paths:?}");
        use itertools::Itertools;
        paths
            .iter()
            .filter_map(|p| {
                p.parent()
                    .filter(|dir| dir.exists())
                    .map(|dir| dir.to_owned())
                    .map(|dir| (p, dir))
            })
            .filter_map(|(p, dir)| p.file_name().map(|name| (dir, name.to_owned())))
            .sorted_by_key(|(dir, _)| dir.clone())
            .chunk_by(|(dir, _)| dir.clone())
            .into_iter()
            .map(|(dir, names)| {
                (
                    dir,
                    names
                        .into_iter()
                        .map(|(_, filename)| filename)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    }

    fn watch<T>(
        create: T,
        storage: Arc<futures::lock::Mutex<Arc<dyn Storage>>>,
        dirs: Vec<(PathBuf, Vec<OsString>)>,
    ) -> Result<notify::Debouncer<RecommendedWatcher, RecommendedCache>>
    where
        T: Fn() -> Pin<Box<dyn Future<Output = Result<Arc<dyn Storage>>> + Send>> + Send + 'static,
    {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<DebounceEventResult>();

        let mut debouncer = new_debouncer(
            Duration::from_secs(1),
            Duration::from_secs(1).into(),
            move |res: DebounceEventResult| {
                let _ = tx.send(res);
            },
        )?;

        for (dir, _) in dirs.iter() {
            debouncer.watch(dir, RecursiveMode::NonRecursive)?;
        }

        crate::util::spawn_detached(async move {
            use itertools::Itertools;
            use notify_debouncer_full::notify::{
                EventKind,
                event::{AccessKind, AccessMode, CreateKind, ModifyKind},
            };
            use tokio_stream::{StreamExt, wrappers::UnboundedReceiverStream};

            let name = storage.lock().await.cache_type_name().to_owned();

            for (dir, _) in dirs.iter() {
                debug!("{name} storage watching for changes in dir: {dir:?}");
            }

            let paths = dirs
                .into_iter()
                .flat_map(|(parent, filenames)| {
                    filenames
                        .into_iter()
                        .map(|filename| parent.join(filename))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            debug!("{name} storage watching for changes to files: {paths:?}");

            let changes = UnboundedReceiverStream::new(rx)
                .filter_map(|res| {
                    res.inspect_err(|err| error!("[WatchStorage::watch]: Notify error: {err:?}"))
                        .ok()
                })
                .filter_map(|events| {
                    let events = events
                        .into_iter()
                        .filter(|event| {
                            matches!(
                                event.kind,
                                EventKind::Access(AccessKind::Close(AccessMode::Write))
                                    | EventKind::Create(CreateKind::File)
                                    | EventKind::Modify(ModifyKind::Data(_))
                                    | EventKind::Remove(_)
                            )
                        })
                        .filter_map(|mut event| {
                            event.paths = paths
                                .iter()
                                .filter_map(|path| {
                                    if event.paths.contains(path) {
                                        Some(path.clone())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>();

                            if event.paths.is_empty() {
                                None
                            } else {
                                Some(event)
                            }
                        })
                        .collect::<Vec<_>>();

                    if events.is_empty() {
                        None
                    } else {
                        Some(events)
                    }
                })
                .chunks_timeout(20, Duration::from_secs(5))
                .map(|events| {
                    let events = events.into_iter().flatten().collect::<Vec<_>>();

                    trace!("[WatchStorage::watch]: Notify events: {events:?}");

                    let paths = events
                        .iter()
                        .flat_map(|event| event.paths.iter().map(|p| p.as_path()))
                        .unique()
                        .collect::<Vec<_>>();

                    info!("[WatchStorage::watch]: Recreating storage after changes to {paths:?}");

                    events
                });

            futures::pin_mut!(changes);

            while changes.next().await.is_some() {
                let mut guard = storage.lock().await;
                match create().await {
                    Ok(storage) => *guard = storage,
                    Err(err) => {
                        error!("Failed to recreate storage: {err:?}");
                    }
                }
                drop(guard);
            }
        });

        Ok(debouncer)
    }

    async fn inner(&self) -> Arc<dyn Storage> {
        // clone() so the lock is dropped immediately, otherwise concurrent
        // operations are serialized until their Future completes.
        self.storage.lock().await.clone()
    }
}

#[async_trait]
impl Storage for WatchStorage {
    async fn get(&self, key: &str) -> Result<Cache<opendal::Buffer>> {
        self.inner().await.get(key).await
    }

    async fn del(&self, key: &str) -> Result<()> {
        self.inner().await.del(key).await
    }

    async fn has(&self, key: &str) -> bool {
        self.inner().await.has(key).await
    }

    async fn put(&self, key: &str, entry: opendal::Buffer) -> Result<Duration> {
        self.inner().await.put(key, entry).await
    }

    async fn size(&self, key: &str) -> Result<u64> {
        self.inner().await.size(key).await
    }

    /// Check the cache capability.
    async fn check(&self) -> Result<CacheMode> {
        self.inner().await.check().await
    }

    /// Get the storage location.
    async fn location(&self) -> String {
        self.inner().await.location().await
    }

    /// Get the cache backend type name (e.g., "disk", "redis", "s3").
    /// Used for statistics and display purposes.
    fn cache_type_name(&self) -> &'static str {
        self.cache_type_name
    }

    /// Get the current storage usage, if applicable.
    async fn current_size(&self) -> Result<Option<u64>> {
        self.inner().await.current_size().await
    }

    /// Get the maximum storage size, if applicable.
    async fn max_size(&self) -> Result<Option<u64>> {
        self.inner().await.max_size().await
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
