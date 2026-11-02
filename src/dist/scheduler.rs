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

use crate::{
    cache::Storage,
    dist::{
        self, JobInputs, JobStats, NewJobResponse, RunJobStatus, SchedulerService, SchedulerStatus,
        ServerStatus, StatusUpdate, SysStats, Toolchain,
        io::{Jobs, Toolchains},
        metrics::{JobsMetrics, Metrics, SchedulerMetrics, ToolchainsMetrics},
        v3::{NewJobResponse as NewJobResponseV3, RunJobRequest, RunJobResponse},
    },
    errors::*,
};
use async_trait::async_trait;
use futures::lock::Mutex;
use itertools::Itertools;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug)]
struct ServerInfo {
    // Last modified time (in the scheduler's reference frame)
    pub m_time: Instant,
    // Last modified time (in the server's reference frame)
    pub u_time: SystemTime,
    // Server performance stats
    pub info: SysStats,
    // Server job stats
    pub jobs: JobStats,
    // The age of the oldest active job (in seconds)
    pub max_job_age: u64,
    // The server-specific queue
    pub queue: String,
}

#[derive(Default)]
pub struct SchedulerBuilder {
    scheduler_id: Option<String>,
    jobs_storage: Option<Arc<dyn Storage>>,
    metrics: Option<Metrics>,
    tasks: Option<Arc<dyn dist::tasks::SchedulerTasks>>,
    toolchains_storage: Option<Arc<dyn Storage>>,
}

impl SchedulerBuilder {
    pub fn with_scheduler_id(self, scheduler_id: String) -> Self {
        Self {
            scheduler_id: Some(scheduler_id),
            ..self
        }
    }
    pub fn with_jobs_storage(self, jobs_storage: Arc<dyn Storage>) -> Self {
        Self {
            jobs_storage: Some(jobs_storage),
            ..self
        }
    }
    pub fn with_metrics(self, metrics: Metrics) -> Self {
        Self {
            metrics: Some(metrics),
            ..self
        }
    }
    pub fn with_tasks(self, tasks: Arc<dyn dist::tasks::SchedulerTasks>) -> Self {
        Self {
            tasks: Some(tasks),
            ..self
        }
    }
    pub fn with_toolchains_storage(self, toolchains_storage: Arc<dyn Storage>) -> Self {
        Self {
            toolchains_storage: Some(toolchains_storage),
            ..self
        }
    }
    pub fn build(self) -> Result<Arc<Scheduler>> {
        let SchedulerBuilder {
            scheduler_id,
            jobs_storage,
            metrics,
            tasks,
            toolchains_storage,
        } = self;

        let scheduler_id = scheduler_id
            .map(Ok)
            .unwrap_or_else(|| bail!("Missing required field `scheduler_id`"))?;
        let jobs_storage = jobs_storage
            .map(Ok)
            .unwrap_or_else(|| bail!("Missing required field `jobs_storage`"))?;
        let metrics = metrics
            .map(Ok)
            .unwrap_or_else(|| bail!("Missing required field `metrics`"))?;
        let tasks = tasks
            .map(Ok)
            .unwrap_or_else(|| bail!("Missing required field `tasks`"))?;
        let toolchains_storage = toolchains_storage
            .map(Ok)
            .unwrap_or_else(|| bail!("Missing required field `toolchains_storage`"))?;

        let (lifetime, _) = tokio::sync::watch::channel(true);

        let lifetime = Arc::new(lifetime);

        let jobs = Arc::new(Mutex::new(HashMap::new()));

        let jobs_io = Jobs::new(
            jobs_storage,
            JobsMetrics::new("sccache::scheduler", metrics.clone()),
        );

        let toolchains_io = Toolchains::new(
            toolchains_storage,
            ToolchainsMetrics::new("sccache::scheduler", metrics.clone()),
        );

        let metrics = SchedulerMetrics::new("sccache::scheduler", metrics);

        let scheduler = Arc::new(Scheduler {
            jobs_io,
            jobs: jobs.clone(),
            lifetime,
            metrics,
            scheduler_id,
            servers: Default::default(),
            tasks: tasks.clone(),
            toolchains_io,
        });

        scheduler.tasks.set_service(scheduler.clone())?;

        Ok(scheduler)
    }
}

type JobsMap = Mutex<HashMap<String, Vec<(usize, tokio::sync::oneshot::Sender<RunJobResponse>)>>>;

#[derive(Clone)]
pub struct Scheduler {
    jobs_io: Jobs,
    jobs: Arc<JobsMap>,
    lifetime: Arc<tokio::sync::watch::Sender<bool>>,
    metrics: SchedulerMetrics,
    scheduler_id: String,
    servers: Arc<Mutex<HashMap<String, ServerInfo>>>,
    tasks: Arc<dyn dist::tasks::SchedulerTasks>,
    toolchains_io: Toolchains,
}

impl Scheduler {
    pub fn builder() -> SchedulerBuilder {
        SchedulerBuilder::default()
    }

    pub async fn start(
        &self,
        handle: axum_server::Handle<SocketAddr>,
        server: impl futures::Future<Output = Result<()>>,
        heartbeat_interval: Duration,
        shutdown_timeout: Duration,
    ) -> Result<()> {
        // Start celery
        let celery = async {
            let res = self
                .tasks
                .app()
                .consume_from(&self.tasks.queues_to_consume())
                .await;
            tracing::info!("Celery shutdown");
            res.context("Celery error")
        };

        let sigint = async {
            use tokio::signal::unix::{SignalKind, signal};
            let mut sigint = signal(SignalKind::interrupt())?;
            let mut sigterm = signal(SignalKind::terminate())?;
            tokio::select! {
                biased;
                _ = sigint.recv() => {
                    tracing::info!("Received SIGINT");
                },
                _ = sigterm.recv() => {
                    tracing::info!("Received SIGTERM");
                },
            }
            Ok(())
        };

        let heartbeat = tokio::spawn({
            let this = self.clone();
            async move {
                loop {
                    if let Err(err) = futures::future::try_join(
                        this.poll_for_finished_jobs(),
                        this.send_status_to_servers(),
                    )
                    .await
                    {
                        break Err::<(), _>(err);
                    }
                    tokio::time::sleep(heartbeat_interval).await;
                }
            }
        });

        // Wait for Celery or sigint, then shutdown Axum and Celery
        let shutdown_celery = async move {
            // tokio::select! deterministically polls its futures in order, so
            // register celery first and sigint second so this sigint handler
            // doesn't override celery's internal sigint handler.
            //
            // This ensures celery progresses to "warm shutdown" mode, cancelling
            // the queue consumers, and doesn't attempt to pull more messages
            // from the queue while it is shutting down.
            let res = tokio::select! {
                biased;
                res = celery => res,
                res = sigint => res,
                res = heartbeat => res.map_err(Into::into).and_then(|res| res),
            };

            // Broadcast the scheduler is shutting down.
            self.lifetime.send_replace(false);

            tracing::info!(
                "Waiting {}s for graceful shutdown",
                shutdown_timeout.as_secs()
            );

            // Signal axum to shutdown
            handle.graceful_shutdown(Some(shutdown_timeout));

            res
        };

        // Wait for axum shutdown
        let shutdown_server = async move { server.await.context("Axum server error") };

        self.tasks.app().display_pretty().await;

        tracing::info!("Scheduler `{}` initialized", self.scheduler_id);

        // Wait for celery and/or server shutdown
        let shutdown_server = tokio::try_join!(shutdown_celery, shutdown_server);

        // Close the broker connection
        let shutdown_broker = self.tasks.app().close().await.context("Broker error");

        tracing::info!("Closed broker connection");

        shutdown_server
            .and(shutdown_broker)
            .inspect(|_| tracing::info!("Scheduler shutdown gracefully"))
            .inspect_err(|err| tracing::warn!("Scheduler did not shutdown gracefully: {err:#}"))
    }

    async fn poll_for_finished_jobs(&self) -> Result<()> {
        let finished_jobs =
            futures::future::join_all(self.jobs.lock().await.keys().map(|job_id| async {
                (job_id.clone(), self.jobs_io.has_job_result(job_id).await)
            }))
            .await
            .into_iter()
            .filter_map(|(job_id, has_result)| has_result.then_some(job_id));

        futures::future::try_join_all(
            finished_jobs.map(|job_id| async move { self.job_finished(&job_id, None).await }),
        )
        .await
        .map(|_| ())
    }

    async fn send_status_to_servers(&self) -> Result<()> {
        let status = self
            .get_status()
            .await
            .map(|status| self.status_update(status))?;

        let servers = self
            .servers
            .lock()
            .await
            .values()
            .map(|s| s.queue.clone())
            .collect::<Vec<_>>();

        futures::future::try_join_all(
            std::iter::once(self.tasks.update_status(status.clone(), None)).chain(
                servers
                    .iter()
                    .map(|send_to| self.tasks.update_status(status.clone(), Some(send_to))),
            ),
        )
        .await
        .map(|_| ())
        .map_err(Into::into)
    }

    fn status_update(&self, status: SchedulerStatus) -> StatusUpdate {
        let (_, cpu_usage, mem_avail, mem_total) = self.metrics.system_metrics();
        StatusUpdate {
            id: self.scheduler_id.clone(),
            queue: self.tasks.app().default_queue.clone(),
            info: dist::SysStats {
                cpu_usage,
                mem_avail,
                mem_total,
                num_cpus: status.info.num_cpus,
                occupancy: status.info.occupancy,
                pre_fetch: status.info.pre_fetch,
            },
            jobs: status.jobs,
            max_job_age: {
                let max_job_age = status
                    .servers
                    .iter()
                    .map(|s| s.max_job_age)
                    .max()
                    .unwrap_or(0);
                (
                    // microseconds as seconds
                    max_job_age.saturating_div(1_000_000),
                    // remainder microseconds as nanoseconds
                    (max_job_age % 1_000_000).saturating_mul(1_000) as u32,
                )
            },
            timestamp: {
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::from_secs(0))
                    .as_micros();
                (
                    // microseconds as seconds
                    timestamp.saturating_div(1_000_000) as u64,
                    // remainder microseconds as nanoseconds
                    (timestamp % 1_000_000).saturating_mul(1_000) as u32,
                )
            },
        }
    }

    fn prune_servers(servers: &mut HashMap<String, ServerInfo>) {
        let now = Instant::now();
        // Prune servers we haven't seen in 90s
        let timeout = Duration::from_secs(90);
        servers.retain(|_, server| now.duration_since(server.m_time) <= timeout);
    }

    async fn run_job_early_exit(
        &self,
        job_id: &str,
        inputs: &JobInputs,
        toolchain: &Toolchain,
    ) -> Result<Option<RunJobResponse>> {
        if self.jobs_io.has_job_result(job_id).await {
            match self.jobs_io.get_job_result(job_id).await {
                Ok(RunJobResponse::Complete { result, server_id }) => {
                    return Ok(Some(RunJobResponse::Complete { result, server_id }));
                }
                Ok(RunJobResponse::FatalError { message, server_id }) => {
                    return Ok(Some(RunJobResponse::FatalError { message, server_id }));
                }
                // All others, delete the bad result and retry run_job
                _ => {
                    let _ = self.jobs_io.del_job_result(job_id).await;
                }
            }
        }

        if !self.has_toolchain(toolchain).await {
            tracing::warn!("[run_job({job_id})]: Missing toolchain '{toolchain}'");
            return Ok(Some(RunJobResponse::MissingToolchain {
                server_id: self.scheduler_id.clone(),
            }));
        }

        match inputs {
            JobInputs::TarGZ { job_id } => {
                if !self.jobs_io.has_job_inputs(job_id).await {
                    tracing::warn!("[run_job({job_id})]: Missing job inputs");
                    return Ok(Some(RunJobResponse::MissingJobInputs {
                        server_id: self.scheduler_id.clone(),
                        inputs: vec![],
                    }));
                }
            }
            JobInputs::Manifest { files, .. } => {
                if let Some(missing_inputs) = self
                    .jobs_io
                    .missing_job_inputs(files.values().unique().map(|s| s.as_str()))
                    .await
                {
                    tracing::warn!("[run_job({job_id})]: Missing job inputs");
                    return Ok(Some(RunJobResponse::MissingJobInputs {
                        server_id: self.scheduler_id.clone(),
                        inputs: missing_inputs,
                    }));
                }
            }
        }

        Ok(None)
    }
}

#[async_trait]
impl SchedulerService for Scheduler {
    async fn get_status(&self) -> Result<SchedulerStatus> {
        let servers = {
            let mut servers = self.servers.lock().await;
            Self::prune_servers(&mut servers);

            let mut server_statuses = vec![];
            for (server_id, server) in servers.iter() {
                let u_time = server.m_time.elapsed().as_secs();
                server_statuses.push(ServerStatus {
                    id: server_id.clone(),
                    info: server.info.clone(),
                    jobs: server.jobs.clone(),
                    u_time,
                    max_job_age: if server.jobs.running == 0 {
                        server.max_job_age
                    } else {
                        server.max_job_age + u_time
                    },
                });
            }
            server_statuses
        };

        Ok(SchedulerStatus {
            info: Some(
                servers
                    .iter()
                    .fold(dist::SysStats::default(), |mut info, server| {
                        if server.info.cpu_usage.is_finite() {
                            info.cpu_usage += server.info.cpu_usage * server.info.num_cpus as f32;
                        }
                        info.mem_avail += server.info.mem_avail;
                        info.mem_total += server.info.mem_total;
                        info.num_cpus += server.info.num_cpus;
                        info.occupancy += server.info.occupancy;
                        info.pre_fetch += server.info.pre_fetch;
                        info
                    }),
            )
            .map(|mut info| {
                if !servers.is_empty() {
                    info.cpu_usage /= servers.iter().map(|s| s.info.num_cpus).sum::<usize>() as f32;
                }
                info
            })
            .unwrap(),
            jobs: dist::JobStats {
                accepted: servers
                    .iter()
                    .fold(0u64, |acc, server| acc.saturating_add(server.jobs.accepted)),
                finished: servers
                    .iter()
                    .fold(0u64, |acc, server| acc.saturating_add(server.jobs.finished)),
                loading: servers
                    .iter()
                    .fold(0u64, |acc, server| acc.saturating_add(server.jobs.loading)),
                pending: servers
                    .iter()
                    .fold(0u64, |acc, server| acc.saturating_add(server.jobs.pending)),
                running: servers
                    .iter()
                    .fold(0u64, |acc, server| acc.saturating_add(server.jobs.running)),
            },
            servers,
        })
    }

    async fn has_toolchain(&self, toolchain: &Toolchain) -> bool {
        self.toolchains_io.has_toolchain(toolchain).await
    }

    /// Upload toolchain to toolchains storage (S3, GCS, etc.)
    async fn put_toolchain(
        &self,
        toolchain: &Toolchain,
        toolchain_archive: opendal::Buffer,
    ) -> Result<()> {
        self.toolchains_io
            .put_toolchain(toolchain, toolchain_archive)
            .await
    }

    /// Delete the toolchain from toolchains storage (S3, GCS, etc.)
    async fn del_toolchain(&self, toolchain: &Toolchain) -> Result<()> {
        self.toolchains_io.del_toolchain(toolchain).await
    }

    async fn new_job(
        &self,
        toolchain: Toolchain,
        inputs: opendal::Buffer,
    ) -> Result<NewJobResponse> {
        let job_id = uuid::Uuid::new_v4().as_simple().to_string();
        let (has_toolchain, has_inputs) = futures::future::join(
            async { Ok::<bool, anyhow::Error>(self.has_toolchain(&toolchain).await) },
            async {
                if inputs.is_empty() {
                    Ok(())
                } else {
                    self.jobs_io.put_job_inputs(&job_id, inputs).await
                }
            },
        )
        .await;

        Ok(NewJobResponse {
            has_inputs: has_inputs.is_ok(),
            has_toolchain: has_toolchain.unwrap_or_default(),
            job_id,
            timeout: self.tasks.get_job_time_limit(),
        })
    }

    async fn new_job_with_inputs(
        &self,
        toolchain: Toolchain,
        inputs: Vec<String>,
    ) -> Result<NewJobResponseV3> {
        let (
            NewJobResponse {
                job_id,
                timeout,
                has_toolchain,
                ..
            },
            missing_inputs,
        ) = futures::future::try_join(
            async { self.new_job(toolchain, Default::default()).await },
            async {
                Ok(self
                    .jobs_io
                    .missing_job_inputs(inputs.iter().map(|s| s.as_str()))
                    .await
                    .unwrap_or_default())
            },
        )
        .await?;

        Ok(NewJobResponseV3 {
            job_id,
            timeout,
            has_toolchain,
            missing_inputs,
        })
    }

    async fn put_job(&self, job_id: &str, inputs: opendal::Buffer) -> Result<()> {
        self.jobs_io.put_job_inputs(job_id, inputs).await
    }

    async fn del_job(&self, job_id: &str) -> Result<()> {
        let _ = futures::future::join3(
            self.jobs_io.del_job_inputs(job_id),
            self.jobs_io.del_job_result(job_id),
            self.jobs_io.del_job_status(job_id),
        )
        .await;
        Ok(())
    }

    async fn run_job(
        &self,
        job_id: &str,
        RunJobRequest {
            command,
            inputs,
            outputs,
            toolchain,
            save_outputs_as_job_input,
        }: RunJobRequest,
    ) -> Result<RunJobResponse> {
        // Early exit if possible
        if let Some(res) = self.run_job_early_exit(job_id, &inputs, &toolchain).await? {
            return Ok(res);
        }

        let reply_to = &self.tasks.app().default_queue;

        let (tx, rx) = tokio::sync::oneshot::channel::<RunJobResponse>();

        let job = RunJob::new(
            self.jobs.clone(),
            self.jobs_io.clone(),
            job_id,
            reply_to,
            tx,
        )
        .await?;

        match job.status {
            // If the job was cancelled, return the result.
            RunJobStatus::Cancelled => self.jobs_io.get_job_result(job_id).await.or_else(|_| {
                Ok(RunJobResponse::MissingJobResult {
                    server_id: self.scheduler_id.clone(),
                })
            }),
            // If the job is active and we're the one who created it, start it.
            RunJobStatus::Active { ref schedulers } => {
                if schedulers.first() == Some(reply_to) {
                    self.tasks
                        .run_job(
                            job_id,
                            reply_to,
                            &toolchain,
                            &command,
                            inputs,
                            &outputs,
                            save_outputs_as_job_input,
                        )
                        .await
                        .map_err(anyhow::Error::new)?;
                }
                // Wait for the pending or running job
                rx.await.map_err(anyhow::Error::new)
            }
        }
    }

    async fn job_finished(&self, job_id: &str, status: Option<StatusUpdate>) -> Result<()> {
        let job = self.jobs.lock().await.remove(job_id);
        if let Some(callbacks) = job {
            let job_result = self
                .jobs_io
                .get_job_result(job_id)
                .await
                .unwrap_or_else(|_| RunJobResponse::MissingJobResult {
                    server_id: status
                        .as_ref()
                        .map(|status| status.id.as_str())
                        .unwrap_or(self.scheduler_id.as_str())
                        .to_owned(),
                });

            for (_, cb) in callbacks {
                let _ = cb.send(job_result.clone());
            }

            if let Some(status) = status {
                self.update_server_status(
                    status,
                    Some(matches!(job_result, RunJobResponse::Complete { .. })),
                )
                .await?;
            }
        }
        Ok(())
    }

    async fn update_server_status(
        &self,
        status: StatusUpdate,
        job_status: Option<bool>,
    ) -> Result<()> {
        if let Some(success) = job_status {
            if success {
                tracing::trace!("Received server success: {status:?}");
            } else {
                tracing::trace!("Received server failure: {status:?}");
            }
        }

        let mut servers = self.servers.lock().await;

        fn duration_from_micros((secs, nanos): (u64, u32)) -> Duration {
            Duration::new(secs, nanos)
        }

        // Insert or update the server info
        servers
            .entry(status.id.clone())
            .and_modify(|server| {
                // Convert to absolute durations since the Unix epoch
                let t1 = server.u_time.duration_since(UNIX_EPOCH).unwrap();
                let t2 = duration_from_micros(status.timestamp);
                // If this event is newer than the latest state, it is now the latest state.
                if t2 >= t1 {
                    server.info = status.info.clone();
                    server.jobs = status.jobs.clone();
                    server.queue = status.queue.clone();
                    server.m_time = Instant::now();
                    server.max_job_age = duration_from_micros(status.max_job_age).as_secs();
                    // Increment prev time with the difference between prev and next
                    server.u_time = server.u_time.checked_add(t2 - t1).unwrap();
                }
            })
            .or_insert_with(|| ServerInfo {
                info: status.info,
                jobs: status.jobs,
                queue: status.queue,
                m_time: Instant::now(),
                max_job_age: duration_from_micros(status.max_job_age).as_secs(),
                // Convert to absolute duration since the Unix epoch
                u_time: UNIX_EPOCH
                    .checked_add(duration_from_micros(status.timestamp))
                    .unwrap(),
            });

        Self::prune_servers(&mut servers);

        Ok(())
    }
}

#[derive(Clone)]
struct RunJob {
    job_idx: usize,
    status: RunJobStatus,
    state: Arc<(
        Arc<JobsMap>, // jobs
        Jobs,         // jobs_io
        String,       // job_id
        String,       // reply_to
    )>,
}

impl RunJob {
    async fn new(
        jobs: Arc<JobsMap>,
        jobs_io: Jobs,
        job_id: &str,
        reply_to: &str,
        tx: tokio::sync::oneshot::Sender<RunJobResponse>,
    ) -> Result<Self> {
        Self {
            job_idx: 0,
            status: RunJobStatus::Cancelled,
            state: Arc::new((jobs, jobs_io, job_id.to_owned(), reply_to.to_owned())),
        }
        .init(tx)
        .await
    }

    async fn init(mut self, tx: tokio::sync::oneshot::Sender<RunJobResponse>) -> Result<Self> {
        let Self {
            job_idx,
            status,
            state,
        } = &mut self;

        let (jobs, jobs_io, job_id, reply_to) = state.as_ref();

        {
            let mut jobs = jobs.lock().await;
            if let Some(callbacks) = jobs.get_mut(job_id) {
                *job_idx = callbacks.len();
                callbacks.push((*job_idx, tx));
            } else {
                jobs.insert(job_id.to_owned(), vec![(0, tx)]);
            }
        }

        loop {
            *status =
                jobs_io
                    .get_job_status(job_id)
                    .await?
                    .unwrap_or_else(|| RunJobStatus::Active {
                        schedulers: Vec::with_capacity(1),
                    });

            if let RunJobStatus::Cancelled = status {
                // If we caught a Cancelled job before the build server
                // culls it, transition it back from Cancelled to Active
                *status = RunJobStatus::Active {
                    schedulers: Vec::with_capacity(1),
                };
            }

            if let RunJobStatus::Active { schedulers } = status {
                if schedulers.contains(reply_to) {
                    // Stop once we're in the schedulers list
                    break;
                }
                // Add ourselves to the schedulers list
                schedulers.push(reply_to.clone());
            }

            // Write the status back and loop again to ensure the
            // status update wasn't clobbered by other schedulers
            // or build servers
            jobs_io.put_job_status(job_id, status).await?;
        }

        Ok(self)
    }

    async fn drop(
        job_idx: usize,
        state: Arc<(
            Arc<JobsMap>, // jobs
            Jobs,         // jobs_io
            String,       // job_id
            String,       // reply_to
        )>,
    ) -> Result<()> {
        let job_idx = &job_idx;
        let (jobs, jobs_io, job_id, reply_to) = state.as_ref();

        {
            let mut jobs = jobs.lock().await;
            if let Some(callbacks) = jobs.get_mut(job_id) {
                callbacks.retain(|(key, _)| key != job_idx);
                if callbacks.is_empty() {
                    jobs.remove(job_id);
                }
            } else {
                jobs.remove(job_id);
            }
        }

        loop {
            let mut status = jobs_io
                .get_job_status(job_id)
                .await?
                .unwrap_or(RunJobStatus::Cancelled);

            match &mut status {
                RunJobStatus::Active { schedulers } => {
                    if schedulers.contains(reply_to) {
                        // Remove ourselves from the schedulers list
                        schedulers.retain(|scheduler| scheduler != reply_to);
                        // Write the status back and loop again to ensure the
                        // status update wasn't clobbered by other schedulers
                        // or build servers
                        jobs_io.put_job_status(job_id, &status).await?;
                    } else if schedulers.is_empty() {
                        // If we were the last interested scheduler, cancel
                        // the job. Write the status back and loop again to
                        // ensure either that the status was updated to
                        // Cancelled, or it was set to Active by someone
                        // else (without us in the schedulers list).
                        jobs_io
                            .put_job_status(job_id, &RunJobStatus::Cancelled)
                            .await?;
                    } else {
                        // There are other interested parties, so break
                        // once we're not in the schedulers list anymore
                        break;
                    }
                }
                // Break once the job status is cancelled
                RunJobStatus::Cancelled => break,
            }
        }

        Ok(())
    }
}

impl Drop for RunJob {
    fn drop(&mut self) {
        tokio::spawn(Self::drop(self.job_idx, self.state.clone()));
    }
}
