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

use async_trait::async_trait;

use celery::{
    Celery,
    error::CeleryError,
    protocol::MessageContentType,
    task::{AsyncResult, Task},
};

use std::sync::Arc;

use crate::{
    config::dist::MessageBroker,
    dist::{
        CompileCommand, JobInputs, SchedulerService, ServerService, StatusUpdate, Toolchain,
        scheduler_to_servers_queue, server_to_schedulers_queue, to_scheduler_queue,
        to_server_queue,
    },
    errors::*,
};

use tokio::sync::OnceCell;

static SCHEDULER: OnceCell<Arc<dyn SchedulerService>> = OnceCell::const_new();
static SERVER: OnceCell<Arc<dyn ServerService>> = OnceCell::const_new();

const MESSAGE_BROKER_ERROR_TEXT: &str = "\
    The sccache-dist scheduler and servers communicate via an external message
    broker, either an AMQP v0.9.1 implementation (like RabbitMQ) or Redis.

    All major CSPs provide managed AMQP or Redis services, or you can deploy
    RabbitMQ or Redis as part of your infrastructure.

    For local development, you can install RabbitMQ/Redis services locally or
    run their containers.

    More details can be found in in the sccache-dist documentation at:
    https://github.com/mozilla/sccache/blob/main/docs/Distributed.md#message-brokers";

pub struct Tasks {
    app: Arc<Celery>,
    job_time_limit: u32,
    queues_to_consume: Vec<String>,
}

impl Tasks {
    pub async fn new(
        builder: celery::CeleryBuilder,
        queues_to_consume: Vec<String>,
    ) -> Result<Self> {
        Ok(Self {
            job_time_limit: u32::MAX,
            app: Arc::new(builder.build().await.map_err(|err| {
                let err_message = match err {
                    CeleryError::BrokerError(err) => err.to_string(),
                    err => err.to_string(),
                };
                anyhow!("{err_message}\n\n{MESSAGE_BROKER_ERROR_TEXT}")
            })?),
            queues_to_consume,
        })
    }

    pub async fn scheduler(
        id: &str,
        prefetch_count: u16,
        job_time_limit: u32,
        message_broker: Option<MessageBroker>,
    ) -> Result<Arc<dyn SchedulerTasks>> {
        let default_queue = to_scheduler_queue(id);
        let scheduler_to_servers = scheduler_to_servers_queue();
        let server_to_schedulers = server_to_schedulers_queue();
        let mut this = Tasks::new(
            Tasks::celery(id, prefetch_count, message_broker)?
                // Tasks the scheduler sends
                .task_route(task_impls::run_job::NAME, &scheduler_to_servers)
                .task_route(task_impls::scheduler_status::NAME, &scheduler_to_servers)
                // Tasks the scheduler receives
                .task_route(task_impls::server_status::NAME, &server_to_schedulers)
                // Instruct the broker to delete this queue when the scheduler disconnects
                .broker_declare_exclusive_queue(&default_queue)
                .default_queue(&default_queue),
            vec![default_queue, server_to_schedulers],
        )
        .await?;

        this.set_job_time_limit(job_time_limit);

        // Tasks the scheduler receives
        this.app.register_task::<task_impls::job_finished>().await?;
        this.app
            .register_task::<task_impls::server_status>()
            .await?;

        Ok(Arc::new(this))
    }

    pub async fn server(
        id: &str,
        prefetch_count: u16,
        message_broker: Option<MessageBroker>,
    ) -> Result<Arc<dyn ServerTasks>> {
        let default_queue = to_server_queue(id);
        let scheduler_to_servers = scheduler_to_servers_queue();
        let server_to_schedulers = server_to_schedulers_queue();
        let this = Tasks::new(
            Tasks::celery(id, prefetch_count, message_broker)?
                // Tasks the server sends
                .task_route(task_impls::job_finished::NAME, &server_to_schedulers)
                .task_route(task_impls::server_status::NAME, &server_to_schedulers)
                // Tasks the server receives
                .task_route(task_impls::scheduler_status::NAME, &scheduler_to_servers)
                // Instruct the broker to delete this queue when the server disconnects
                .broker_declare_exclusive_queue(&default_queue)
                .default_queue(&default_queue),
            vec![default_queue, scheduler_to_servers],
        )
        .await?;

        // Tasks the server receives
        this.app.register_task::<task_impls::run_job>().await?;
        this.app
            .register_task::<task_impls::scheduler_status>()
            .await?;

        Ok(Arc::new(this))
    }

    fn celery(
        id: &str,
        prefetch_count: u16,
        message_broker: Option<MessageBroker>,
    ) -> Result<celery::CeleryBuilder> {
        if let Some(message_broker) = message_broker {
            let scheduler_to_servers = scheduler_to_servers_queue();
            let server_to_schedulers = server_to_schedulers_queue();
            Ok(celery::CeleryBuilder::new(id, &message_broker.addr)
                // Retry connecting to the broker
                .broker_connection_retry(message_broker.connection_retry)
                .broker_connection_max_retries(message_broker.max_retries)
                .broker_connection_retry_delay(message_broker.retry_delay)
                .broker_connection_timeout(message_broker.connection_timeout)
                // Queues with no consumers should be deleted after 60s
                .broker_set_queue_expire_time(&scheduler_to_servers, 60 * 1000)
                .broker_set_queue_expire_time(&server_to_schedulers, 60 * 1000)
                // Prefetch messages
                .prefetch_count(prefetch_count)
                // These tasks are sent to these queues
                .task_route(task_impls::run_job::NAME, &scheduler_to_servers)
                // MessagePack is faster than JSON/Yaml/pickle etc.
                .task_content_type(MessageContentType::MsgPack)
                // Don't retry any tasks
                .task_max_retries(0)
                // Don't delay retrying failed tasks
                .task_min_retry_delay(0)
                .task_max_retry_delay(0)
                // Don't retry tasks that fail with unexpected errors
                .task_retry_for_unexpected(false))
        } else {
            bail!("Missing required message broker configuration!\n\n{MESSAGE_BROKER_ERROR_TEXT}")
        }
    }
}

#[async_trait]
pub trait AppTasks: Send + Sync {
    fn app(&self) -> &Arc<celery::Celery>;
    fn queues_to_consume(&self) -> Vec<&str>;
}

#[async_trait]
impl AppTasks for Tasks {
    fn app(&self) -> &Arc<Celery> {
        &self.app
    }

    fn queues_to_consume(&self) -> Vec<&str> {
        self.queues_to_consume
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
    }
}

#[async_trait]
pub trait SchedulerTasks: AppTasks + Send + Sync {
    fn set_service(&self, scheduler: Arc<dyn SchedulerService>) -> Result<()> {
        SCHEDULER
            .set(scheduler.clone())
            .map_err(|err| anyhow!("{err:#}"))
    }

    fn get_job_time_limit(&self) -> u32;
    fn set_job_time_limit(&mut self, job_time_limit: u32);

    async fn update_status(
        &self,
        status: StatusUpdate,
        send_to: Option<&str>,
    ) -> std::result::Result<AsyncResult, CeleryError> {
        self.app()
            .send_task(if let Some(send_to) = send_to {
                task_impls::scheduler_status::new(status).with_queue(send_to)
            } else {
                task_impls::scheduler_status::new(status)
            })
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_job(
        &self,
        job_id: &str,
        reply_to: &str,
        toolchain: &Toolchain,
        command: &CompileCommand,
        inputs: JobInputs,
        outputs: &[String],
        save_outputs_as_job_input: bool,
    ) -> std::result::Result<AsyncResult, CeleryError>;
}

#[async_trait]
impl SchedulerTasks for Tasks {
    fn get_job_time_limit(&self) -> u32 {
        self.job_time_limit
    }

    fn set_job_time_limit(&mut self, job_time_limit: u32) {
        self.job_time_limit = job_time_limit;
    }

    async fn run_job(
        &self,
        job_id: &str,
        reply_to: &str,
        toolchain: &Toolchain,
        command: &CompileCommand,
        inputs: JobInputs,
        outputs: &[String],
        save_outputs_as_job_input: bool,
    ) -> std::result::Result<AsyncResult, CeleryError> {
        self.app()
            .send_task(
                task_impls::run_job::new(
                    job_id.to_owned(),
                    reply_to.to_owned(),
                    toolchain.to_owned(),
                    command.to_owned(),
                    inputs,
                    outputs.to_owned(),
                    save_outputs_as_job_input,
                )
                .with_time_limit(self.job_time_limit.saturating_sub(30))
                .with_expires_in(self.job_time_limit.saturating_sub(30)),
            )
            .await
    }
}

#[async_trait]
pub trait ServerTasks: AppTasks + Send + Sync {
    fn set_service(&self, server: Arc<dyn ServerService>) -> Result<()> {
        SERVER.set(server.clone()).map_err(|err| anyhow!("{err:#}"))
    }

    async fn update_status(
        &self,
        server_status: StatusUpdate,
        send_to: Option<&str>,
    ) -> std::result::Result<AsyncResult, CeleryError> {
        self.app()
            .send_task(if let Some(send_to) = send_to {
                task_impls::server_status::new(server_status).with_queue(send_to)
            } else {
                task_impls::server_status::new(server_status)
            })
            .await
    }

    async fn job_finished(
        &self,
        job_id: &str,
        send_to: &str,
        server_status: StatusUpdate,
    ) -> std::result::Result<AsyncResult, CeleryError> {
        self.app()
            .send_task(
                task_impls::job_finished::new(job_id.to_owned(), server_status).with_queue(send_to),
            )
            .await
    }
}

impl ServerTasks for Tasks {}

#[allow(non_local_definitions)]
mod task_impls {
    use super::{SCHEDULER, SERVER};
    use crate::{
        dist::{
            CompileCommand, JobInputs, RunJobError, SchedulerService, ServerService, StatusUpdate,
            Toolchain, v3::RunJobResponse,
        },
        errors::*,
    };
    use celery::prelude::*;
    use futures::FutureExt;
    use std::sync::Arc;

    fn scheduler_service<'a>() -> Result<&'a Arc<dyn SchedulerService>> {
        SCHEDULER.get().ok_or_else(|| {
            let err = anyhow!("sccache-dist scheduler is not initialized");
            tracing::error!("{err:?}");
            err
        })
    }

    fn server_service<'a>() -> Result<&'a Arc<dyn ServerService>> {
        SERVER.get().ok_or_else(|| {
            let err = anyhow!("sccache-dist server is not initialized");
            tracing::error!("{err:?}");
            err
        })
    }

    #[celery::task(
        acks_late = true,
        on_failure = run_job_on_failure,
        on_success = run_job_on_success,
    )]
    pub async fn run_job(
        job_id: String,
        reply_to: String,
        toolchain: Toolchain,
        command: CompileCommand,
        inputs: JobInputs,
        outputs: Vec<String>,
        save_outputs_as_job_input: bool,
    ) -> TaskResult<RunJobResponse> {
        tracing::trace!(
            "[run_job({job_id}, {}, {:?}, {:?}, {outputs:?})]",
            toolchain.archive_id,
            command.executable,
            command.arguments,
        );

        server_service()
            .map(|svc| {
                svc.run_job(
                    &job_id,
                    &reply_to,
                    toolchain,
                    command,
                    inputs,
                    outputs,
                    save_outputs_as_job_input,
                )
            })
            .unwrap_or_else(|err| futures::future::err(err).boxed())
            .await
            .map_err(|err| match err.downcast::<RunJobError>() {
                Ok(RunJobError::MissingJobInputs { inputs }) => TaskError::ExpectedError(
                    serde_json::to_string(&RunJobError::MissingJobInputs { inputs }).unwrap(),
                ),
                // Other RunJobError types
                Ok(err) => TaskError::ExpectedError(serde_json::to_string(&err).unwrap()),
                Err(err) => TaskError::UnexpectedError(err.to_string()),
            })
    }

    async fn run_job_on_failure(task: &run_job, err: &TaskError) {
        let job_id = &task.request().params.job_id;

        let err = match err {
            // The client can choose to retry these or compile locally.
            // Matching strings because that's the only type in TaskError.
            TaskError::ExpectedError(msg) => match serde_json::from_str(msg) {
                Ok(err) => err,
                _ => RunJobError::Retryable(msg.to_owned()),
            },
            TaskError::UnexpectedError(msg) => RunJobError::Retryable(msg.to_owned()),
            // Report task timeouts as a fatal errors, since this means the
            // compilation took longer than the scheduler's `job_time_limit`.
            //
            // In this case, retrying the compilation on a different server
            // won't finish any sooner, so we report a fatal error so clients
            // see their build fail and can either increase the job time limit,
            // or apply source optimizations so their files compile in a
            // reasonable amount of time.
            TaskError::TimeoutError => RunJobError::Fatal(format!("Job {job_id} timed out")),
            TaskError::Retry(_) => RunJobError::Fatal(format!("Job {job_id} retries exceeded")),
        };

        if let Err(err) = server_service()
            .map(|svc| svc.notify_run_job_err(job_id, err).boxed())
            .unwrap_or_else(|err| futures::future::err(err).boxed())
            .await
        {
            tracing::error!("[run_job_on_failure({job_id})]: Error reporting job failure: {err:#}");
        }
    }

    async fn run_job_on_success(task: &run_job, res: &RunJobResponse) {
        let job_id = &task.request().params.job_id;

        if let Err(err) = server_service()
            .map(|svc| svc.notify_run_job_res(job_id, res).boxed())
            .unwrap_or_else(|err| futures::future::err(err).boxed())
            .await
        {
            tracing::error!("[run_job_on_success({job_id})]: Error reporting job success: {err:#}");
        }
    }

    #[celery::task]
    pub async fn job_finished(job_id: String, status: StatusUpdate) -> TaskResult<()> {
        tracing::trace!("[job_finished({job_id}, {status:?})]");

        scheduler_service()
            .map(|svc| svc.job_finished(&job_id, Some(status)))
            .unwrap_or_else(|err| futures::future::err(err).boxed())
            .await
            .map_err(|e| {
                tracing::error!("[job_finished({job_id})]: Failed with unexpected error: {e:#}");
                TaskError::UnexpectedError(format!(
                    "Job {job_id} failed with unexpected error: {e:#}"
                ))
            })
    }

    #[celery::task]
    pub async fn server_status(status: StatusUpdate) -> TaskResult<()> {
        let id = status.id.clone();

        scheduler_service()
            .map(|svc| svc.update_server_status(status, None))
            .unwrap_or_else(|err| futures::future::err(err).boxed())
            .await
            .map_err(|e| {
                tracing::error!("[server_status({id})]: Failed with unexpected error: {e:#}");
                TaskError::UnexpectedError(format!(
                    "Task server_status for {id} failed with unexpected error: {e:#}"
                ))
            })
    }

    #[celery::task]
    pub async fn scheduler_status(status: StatusUpdate) -> TaskResult<()> {
        let id = status.id.clone();

        server_service()
            .map(|svc| svc.update_scheduler_status(status))
            .unwrap_or_else(|err| futures::future::err(err).boxed())
            .await
            .map_err(|e| {
                tracing::error!("[scheduler_status({id})]: Failed with unexpected error: {e:#}");
                TaskError::UnexpectedError(format!(
                    "Task scheduler_status for {id} failed with unexpected error: {e:#}"
                ))
            })
    }

    #[cfg(test)]
    mod test {
        use super::server_status;
        use crate::dist::StatusUpdate;
        use celery::protocol::{Message, MessageContentType};

        // celery silently falls back to JSON when a content type's feature is off, and sccache
        // only enables its MessagePack support.
        #[test]
        fn task_messages_round_trip_as_msgpack() {
            let status = StatusUpdate {
                id: "server-1".into(),
                ..Default::default()
            };
            let task = server_status::new(status).with_content_type(MessageContentType::MsgPack);
            let message = Message::try_from(task).unwrap();
            assert_eq!(message.properties.content_type, "application/x-msgpack");
            let (params, _) = message.body::<server_status>().unwrap().parts();
            assert_eq!(params.status.id, "server-1");
        }
    }
}
