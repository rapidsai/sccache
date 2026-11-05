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
    cache::{GetPathResult, cache::Storage, cache_io::Cache},
    client::ServerConnection,
    config::CacheMode,
    errors::*,
    protocol::{Request, Response, StorageHandshakeInfo},
};

use async_trait::async_trait;
use bytes::Bytes;
use memmap2::Mmap;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

/// `Storage` implementation that forwards all cache operations to the sccache
/// daemon over the existing IPC connection.  Used by CLI processes in
/// client-side mode.
///
/// `ServerConnection` is synchronous and non-`Clone`, so it lives behind a
/// `Mutex` and every RPC dispatches via `crate::util::spawn_blocking`.  The
/// lock is held only for the duration of a single blocking call, never across
/// an `.await` point.
pub struct IpcStorage {
    conn: Arc<Mutex<ServerConnection>>,
    handshake: StorageHandshakeInfo,
}

impl IpcStorage {
    /// Connect to the daemon and perform the `StorageHandshake` RPC.
    /// Returns an `IpcStorage` that can be used as an `Arc<dyn Storage>`.
    pub fn connect(conn: Arc<Mutex<ServerConnection>>) -> Result<Self> {
        let resp = conn.lock().unwrap().request(Request::StorageHandshake)?;
        let handshake = match resp {
            Response::StorageHandshake(info) => info,
            other => bail!("IpcStorage: unexpected handshake response: {other:?}"),
        };
        Ok(Self { conn, handshake })
    }

    /// Return a clone of the underlying connection handle so callers can send
    /// additional RPCs (e.g., `RecordStats`) after the storage is no longer
    /// needed.
    pub fn conn(&self) -> Arc<Mutex<ServerConnection>> {
        Arc::clone(&self.conn)
    }

    async fn rpc(&self, req: Request) -> Result<Response> {
        let conn = Arc::clone(&self.conn);
        crate::util::spawn_blocking(move || conn.lock().unwrap().request(req))
            .await
            .context("spawn_blocking panicked")?
    }
}

#[async_trait]
impl Storage for IpcStorage {
    async fn get(&self, key: &str) -> Result<Cache<opendal::Buffer>> {
        match self.get_path(key).await {
            GetPathResult::Found(path) => {
                let file = std::fs::File::open(&path).with_context(|| {
                    format!("[IpcStorage::get({key:?})]: open {}", path.display())
                })?;

                Ok(Cache::Hit(
                    Bytes::from_owner(unsafe { Mmap::map(&file) }?).into(),
                ))
            }
            GetPathResult::Miss => Ok(Cache::Miss),
            // Backend doesn't support paths (S3, Redis, …); fall back to bytes over IPC.
            GetPathResult::Unsupported => {
                let resp = self
                    .rpc(Request::StorageGetRaw {
                        key: key.to_owned(),
                    })
                    .await?;

                match resp {
                    Response::StorageGetRaw(opt) => {
                        Ok(opt.map_or_else(|| Cache::Miss, |data| Cache::Hit(data.into())))
                    }
                    other => bail!("[IpcStorage::get({key:?})]: unexpected response: {other:?}"),
                }
            }
        }
    }

    /// Delete the cache entry for `key`.
    async fn del(&self, key: &str) -> Result<()> {
        match self
            .rpc(Request::StorageDelPath {
                key: key.to_owned(),
            })
            .await
        {
            Ok(Response::StorageDelPath(res)) => res.map_err(|msg| anyhow!(msg)),
            Err(err) => bail!(err),
            other => bail!("[IpcStorage::del({key:?})]: unexpected response: {other:?}"),
        }
    }

    /// Check if the cache has an entry for `key`.
    ///
    /// If the entry is successfully found in the cache, return true.
    /// If an error occurs, or the entry is not found in the cache, return false.
    async fn has(&self, key: &str) -> bool {
        matches!(self.get_path(key).await, GetPathResult::Found(_))
    }

    async fn size(&self, key: &str) -> Result<u64> {
        match self.get_path(key).await {
            GetPathResult::Found(path) => std::fs::metadata(path)
                .map(|meta| meta.len())
                .map_err(Into::into),
            _ => Ok(0),
        }
    }

    async fn get_path(&self, key: &str) -> GetPathResult {
        match self
            .rpc(Request::StorageGetPath {
                key: key.to_owned(),
            })
            .await
        {
            Ok(Response::StorageGetPath(result)) => result,
            _ => GetPathResult::Unsupported,
        }
    }

    async fn put(&self, key: &str, data: opendal::Buffer) -> Result<Duration> {
        let resp = self
            .rpc(Request::StoragePutRaw {
                key: key.to_owned(),
                data: data.to_vec(),
            })
            .await?;
        match resp {
            Response::StoragePutRaw(Ok(())) => Ok(Duration::ZERO),
            Response::StoragePutRaw(Err(e)) => {
                bail!("[IpcStorage::put({key:?})]: daemon error: {e}")
            }
            other => bail!("[IpcStorage::put({key:?})]: unexpected response: {other:?}"),
        }
    }

    async fn check(&self) -> Result<CacheMode> {
        Ok(self.handshake.cache_mode)
    }

    async fn location(&self) -> String {
        self.handshake.location.clone()
    }

    fn cache_type_name(&self) -> &'static str {
        "ipc"
    }

    async fn current_size(&self) -> Result<Option<u64>> {
        Ok(None)
    }

    async fn max_size(&self) -> Result<Option<u64>> {
        Ok(self.handshake.max_size)
    }

    /// Return whether the storage is enabled.
    /// Currently only NoStorage impl returns false.
    fn enabled(&self) -> bool {
        self.handshake.enabled
    }

    fn basedirs(&self) -> &[Vec<u8>] {
        &self.handshake.basedirs
    }
}
