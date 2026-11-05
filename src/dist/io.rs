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

use bytes::Buf;

use crate::{
    cache::{Cache, Storage},
    dist::{
        RunJobStatus, Toolchain,
        http::bincode_deserialize,
        job_inputs_key, job_result_key, job_status_key,
        metrics::{JobsMetrics, ToolchainsMetrics},
        v3,
        v3::RunJobResponse,
    },
    errors::*,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct Jobs {
    metrics: JobsMetrics,
    storage: Arc<dyn Storage>,
}

impl Jobs {
    pub fn new(storage: Arc<dyn Storage>, metrics: JobsMetrics) -> Self {
        Self { metrics, storage }
    }

    /// Check if a job inputs exists
    pub async fn missing_job_inputs<'a, I: Iterator<Item = &'a str>>(
        &self,
        inputs: I,
    ) -> Option<Vec<String>> {
        let missing_inputs = crate::util::join_all(inputs.into_iter().map(|digest| async move {
            if !self.has_job_inputs(digest).await {
                Some(digest.to_owned())
            } else {
                None
            }
        }))
        .await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        if missing_inputs.is_empty() {
            None
        } else {
            Some(missing_inputs)
        }
    }

    /// Check if a job inputs exists
    pub async fn has_job_inputs(&self, job_id: &str) -> bool {
        // Record has_job_inputs time
        let _timer = self.metrics.has_job_inputs_timer();
        self.storage.has(&job_inputs_key(job_id)).await
    }

    /// Check if a job result exists
    pub async fn has_job_result(&self, job_id: &str) -> bool {
        // Record has_job_result time
        let _timer = self.metrics.has_job_result_timer();
        self.storage.has(&job_result_key(job_id)).await
    }

    /// Check if a job status exists
    pub async fn has_job_status(&self, job_id: &str) -> bool {
        // Record has_job_status time
        let _timer = self.metrics.has_job_status_timer();
        self.storage.has(&job_status_key(job_id)).await
    }

    /// Delete the job inputs
    pub async fn del_job_inputs(&self, job_id: &str) -> Result<()> {
        // Record del_job_inputs time
        let _timer = self.metrics.del_job_inputs_timer();
        self.storage
            .del(&job_inputs_key(job_id))
            .await
            .map_err(|e| {
                // Record del_job_inputs errors after retrying
                self.metrics.inc_del_job_inputs_error_count();
                tracing::warn!("[del_job_inputs({job_id})]: Error deleting job inputs: {e:?}");
                e
            })
    }

    /// Delete the job result
    pub async fn del_job_result(&self, job_id: &str) -> Result<()> {
        // Record del_job_result time
        let _timer = self.metrics.del_job_result_timer();
        self.storage
            .del(&job_result_key(job_id))
            .await
            .map_err(|e| {
                // Record del_job_result errors after retrying
                self.metrics.inc_del_job_result_error_count();
                tracing::warn!("[del_job_result({job_id})]: Error deleting job result: {e:?}");
                e
            })
    }

    /// Delete the job status
    pub async fn del_job_status(&self, job_id: &str) -> Result<()> {
        // Record del_job_status time
        let _timer = self.metrics.del_job_status_timer();
        self.storage
            .del(&job_status_key(job_id))
            .await
            .map_err(|e| {
                // Record del_job_status errors after retrying
                self.metrics.inc_del_job_status_error_count();
                tracing::warn!("[del_job_status({job_id})]: Error deleting job status: {e:?}");
                e
            })
    }

    /// Load the job inputs
    pub async fn get_job_inputs(&self, job_id: &str) -> Result<opendal::Buffer> {
        // Record get_job_inputs time
        let _timer = self.metrics.get_job_inputs_timer();
        self.storage
            .get(&job_inputs_key(job_id))
            .await
            .and_then(|res| match res {
                Cache::Hit(buffer) => Ok(buffer),
                _ => Err(anyhow!("Missing job inputs")),
            })
            .map_err(|err| {
                // Record get_job_inputs errors after retrying
                self.metrics.inc_get_job_inputs_error_count();
                tracing::warn!("[run_job({job_id})]: Error retrieving job inputs: {err:?}");
                err
            })
    }

    /// Load the job result
    pub async fn get_job_result(&self, job_id: &str) -> Result<v3::RunJobResponse> {
        // Record get_job_result time
        let _timer = self.metrics.get_job_result_timer();
        let result = self
            .storage
            .get(&job_result_key(job_id))
            .await
            .and_then(|res| match res {
                Cache::Hit(buffer) => Ok(buffer),
                _ => Err(anyhow!("Missing job result")),
            })
            .map_err(|err| {
                // Record get_job_result errors after retrying
                self.metrics.inc_get_job_result_error_count();
                tracing::warn!("[get_job_result({job_id})]: Error retrieving job result: {err:?}");
                err
            })?;

        // Deserialize the result
        bincode_deserialize(result).await.map_err(|err| {
            self.metrics.inc_get_job_result_error_count();
            tracing::warn!("[get_job_result({job_id})]: Error deserializing result: {err:?}");
            err
        })
    }

    /// Load the job status
    pub async fn get_job_status(&self, job_id: &str) -> Result<Option<RunJobStatus>> {
        // Record get_job_status time
        let _timer = self.metrics.get_job_status_timer();
        let result = self
            .storage
            .get(&job_status_key(job_id))
            .await
            .map(|res| match res {
                Cache::Miss => None,
                Cache::Hit(buffer) => Some(buffer),
            })
            .map_err(|err| {
                // Record get_job_status errors after retrying
                self.metrics.inc_get_job_status_error_count();
                tracing::warn!("[get_job_status({job_id})]: Error retrieving job status: {err:?}");
                err
            })?;

        if let Some(buffer) = result {
            // Deserialize the status
            bincode::deserialize_from(buffer.reader())
                .map(Some)
                .map_err(|err| {
                    self.metrics.inc_get_job_status_error_count();
                    tracing::warn!(
                        "[get_job_status({job_id})]: Error deserializing status: {err:?}"
                    );
                    err.into()
                })
        } else {
            Ok(None)
        }
    }

    /// Store the job inputs
    pub async fn put_job_inputs(&self, job_id: &str, inputs: opendal::Buffer) -> Result<()> {
        // Record put_job_inputs time
        let _timer = self.metrics.put_job_inputs_timer();
        self.storage
            .put(&job_inputs_key(job_id), inputs)
            .await
            .map(|_| ())
            .map_err(|err| {
                // Record put_job_inputs errors after retrying
                self.metrics.inc_put_job_inputs_error_count();
                tracing::warn!("[put_job_inputs({job_id})]: Error storing job inputs: {err:?}");
                err
            })
    }

    /// Store the job result
    pub async fn put_job_result(&self, job_id: &str, result: &RunJobResponse) -> Result<()> {
        // Record put_job_result load time after retrying
        let _timer = self.metrics.put_job_result_timer();
        let result = bincode::serialize(result).map_err(|err| {
            self.metrics.inc_put_job_result_error_count();
            tracing::warn!("[put_job_result({job_id})]: Error serializing result: {err:?}");
            err
        })?;
        self.storage
            .put(&job_result_key(job_id), result.into())
            .await
            .map(|_| ())
            .map_err(|err| {
                // Record put_job_result errors after retrying
                self.metrics.inc_put_job_result_error_count();
                tracing::warn!("[put_job_result({job_id})]: Error storing job result: {err:?}");
                err
            })
    }

    /// Store the job status
    pub async fn put_job_status(&self, job_id: &str, status: &RunJobStatus) -> Result<()> {
        // Record put_job_status time
        let _timer = self.metrics.put_job_status_timer();
        let status = bincode::serialize(status).map_err(|err| {
            self.metrics.inc_put_job_status_error_count();
            tracing::warn!("[put_job_status({job_id})]: Error serializing status: {err:?}");
            err
        })?;
        self.storage
            .put(&job_status_key(job_id), status.into())
            .await
            .map(|_| ())
            .map_err(|e| {
                // Record put_job_status errors after retrying
                self.metrics.inc_put_job_status_error_count();
                tracing::warn!("[put_job_status({job_id})]: Error storing job status: {e:?}");
                e
            })
    }
}

#[derive(Clone)]
pub struct Toolchains {
    metrics: ToolchainsMetrics,
    storage: Arc<dyn Storage>,
}

impl Toolchains {
    pub fn new(storage: Arc<dyn Storage>, metrics: ToolchainsMetrics) -> Self {
        Self { metrics, storage }
    }

    /// Check if a toolchain exists
    pub async fn has_toolchain(&self, toolchain: &Toolchain) -> bool {
        // Record has_has_toolchain time
        let _timer = self.metrics.has_toolchain_timer();
        self.storage.has(toolchain).await
    }

    /// Delete the toolchain archive
    pub async fn del_toolchain(&self, toolchain: &Toolchain) -> Result<()> {
        // Record del_toolchain time
        let _timer = self.metrics.del_toolchain_timer();
        // Delete the toolchain from toolchains storage (S3, GCS, etc.)
        self.storage.del(toolchain).await.map_err(|err| {
            tracing::error!("[del_toolchain({toolchain})]: Error deleting toolchain: {err:?}");
            err
        })
    }

    /// Load the toolchain archive
    pub async fn get_toolchain(&self, toolchain: &Toolchain) -> Result<opendal::Buffer> {
        let _timer = self.metrics.get_toolchain_timer();
        self.storage
            .get(toolchain)
            .await
            .and_then(|res| match res {
                Cache::Hit(buffer) => Ok(buffer),
                _ => Err(anyhow!("Missing toolchain")),
            })
            .map_err(|err| {
                // Record toolchain errors
                self.metrics.inc_get_toolchain_error_count();
                tracing::warn!("[get_toolchain({toolchain})]: Error loading toolchain: {err:?}");
                err
            })
    }

    /// Store the toolchain archive
    pub async fn put_toolchain(
        &self,
        toolchain: &Toolchain,
        toolchain_archive: opendal::Buffer,
    ) -> Result<()> {
        // Record put_toolchain time
        let _timer = self.metrics.put_toolchain_timer();
        self.storage
            .put(toolchain, toolchain_archive)
            .await
            .map(|_| ())
            .map_err(|err| {
                tracing::error!("[put_toolchain({toolchain})]: Error storing toolchain: {err:?}");
                err
            })
    }
}
