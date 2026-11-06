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

use std::{
    borrow::Cow,
    collections::BTreeMap,
    io,
    net::SocketAddr,
    str::FromStr,
    sync::{Arc, atomic::AtomicU64},
    time::{Duration, Instant},
};

use crate::{config, errors::*};

use metrics::SharedString;
use metrics_exporter_dogstatsd::{AggregationMode, DogStatsDBuilder};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

pub struct CountRecorder {
    name: SharedString,
    labels: Option<Arc<BTreeMap<String, String>>>,
}

impl Drop for CountRecorder {
    fn drop(&mut self) {
        self.labels.as_ref().map_or_else(
            || {
                metrics::counter!(self.name.clone()).increment(1);
            },
            |labels| {
                metrics::counter!(self.name.clone(), labels.as_ref()).increment(1);
            },
        );
    }
}

#[derive(Default)]
pub struct GaugeRecorder {
    name: SharedString,
    labels: Option<Arc<BTreeMap<String, String>>>,
    value: AtomicU64,
}

pub struct GaugeRecorderIncrement<'a> {
    name: SharedString,
    labels: &'a Option<Arc<BTreeMap<String, String>>>,
    value: &'a AtomicU64,
}

impl Drop for GaugeRecorderIncrement<'_> {
    fn drop(&mut self) {
        self.value.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        self.labels.as_ref().map_or_else(
            || {
                metrics::gauge!(self.name.clone()).decrement(1);
            },
            |labels| {
                metrics::gauge!(self.name.clone(), labels.as_ref()).decrement(1);
            },
        );
    }
}

impl GaugeRecorder {
    pub fn new(name: SharedString, labels: Option<Arc<BTreeMap<String, String>>>) -> Self {
        Self {
            name,
            labels,
            value: AtomicU64::new(0),
        }
    }

    pub fn increment(&self) -> GaugeRecorderIncrement<'_> {
        let value = &self.value;
        value.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.labels.as_ref().map_or_else(
            || {
                metrics::gauge!(self.name.clone()).increment(1);
            },
            |labels| {
                metrics::gauge!(self.name.clone(), labels.as_ref()).increment(1);
            },
        );
        GaugeRecorderIncrement {
            name: self.name.clone(),
            labels: &self.labels,
            value,
        }
    }

    pub fn value(&self) -> u64 {
        self.value.load(std::sync::atomic::Ordering::SeqCst)
    }
}

pub struct HistoRecorder {
    name: SharedString,
    value: f64,
    labels: Option<Arc<BTreeMap<String, String>>>,
}

impl Drop for HistoRecorder {
    fn drop(&mut self) {
        self.labels.as_ref().map_or_else(
            || {
                metrics::histogram!(self.name.clone()).record(self.value);
            },
            |labels| {
                metrics::histogram!(self.name.clone(), labels.as_ref()).record(self.value);
            },
        );
    }
}

pub struct TimeRecorder {
    name: SharedString,
    start: Instant,
    labels: Option<Arc<BTreeMap<String, String>>>,
}

impl Drop for TimeRecorder {
    fn drop(&mut self) {
        self.labels.as_ref().map_or_else(
            || {
                metrics::histogram!(self.name.clone()).record(self.start.elapsed());
            },
            |labels| {
                metrics::histogram!(self.name.clone(), labels.as_ref())
                    .record(self.start.elapsed());
            },
        );
    }
}

#[derive(Clone)]
pub struct Metrics {
    global_labels: Arc<BTreeMap<String, String>>,
    inner: Arc<dyn MetricsInner>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            global_labels: Default::default(),
            inner: Arc::new(NoopMetrics {}),
        }
    }
}

tokio::task_local! {
    static SCOPED_LABELS: Arc<BTreeMap<String, String>>;
}

impl Metrics {
    pub fn new(
        config: config::dist::Metrics,
        global_labels: BTreeMap<String, String>,
    ) -> Result<Self> {
        match config {
            config::dist::Metrics::None => Ok(Self {
                global_labels: Arc::new(global_labels),
                inner: Arc::new(NoopMetrics {}),
            }),
            config::dist::Metrics::Dogstatsd(config) => Ok(Self {
                global_labels: Arc::new(global_labels),
                inner: Arc::new(DogStatsDMetrics::new(config)?),
            }),
            config::dist::Metrics::Prometheus(config) => Ok(Self {
                global_labels: Arc::new(Default::default()),
                inner: Arc::new(PrometheusMetrics::new(config, global_labels)?),
            }),
        }
    }

    pub fn scoped_labels(&self) -> Option<Arc<BTreeMap<String, String>>> {
        SCOPED_LABELS.try_get().ok()
    }

    pub fn scope_with_labels<F>(
        &self,
        local_labels: &BTreeMap<String, String>,
        f: F,
    ) -> tokio::task::futures::TaskLocalFuture<Arc<BTreeMap<String, String>>, F>
    where
        F: Future,
    {
        let mut labels = BTreeMap::new();
        for (k, v) in self.global_labels.iter() {
            labels.insert(k.clone(), v.clone());
        }
        for (k, v) in local_labels.iter() {
            labels.insert(k.clone(), v.clone());
        }
        SCOPED_LABELS.scope(Arc::new(labels), f)
    }

    pub fn render(&self) -> String {
        self.inner.render()
    }

    pub fn render_to_write(&self, writer: Box<dyn io::Write>) -> io::Result<()> {
        self.inner.render_to_write(writer)
    }

    pub fn listen_path(&self) -> Option<String> {
        self.inner.listen_path()
    }

    pub fn gauge<S: Into<SharedString>>(&self, name: S) -> GaugeRecorder {
        GaugeRecorder {
            name: name.into(),
            labels: self.scoped_labels(),
            value: AtomicU64::new(0),
        }
    }

    pub fn count<S: Into<SharedString>>(&self, name: S) -> CountRecorder {
        CountRecorder {
            name: name.into(),
            labels: self.scoped_labels(),
        }
    }

    pub fn histo<S: Into<SharedString>, T: metrics::IntoF64>(
        &self,
        name: S,
        value: T,
    ) -> HistoRecorder {
        HistoRecorder {
            name: name.into(),
            value: value.into_f64(),
            labels: self.scoped_labels(),
        }
    }

    pub fn timer<S: Into<SharedString>>(&self, name: S) -> TimeRecorder {
        TimeRecorder {
            name: name.into(),
            start: Instant::now(),
            labels: self.scoped_labels(),
        }
    }

    pub fn increment_counter<S: Into<SharedString>>(&self, name: S, value: u64) {
        if let Some(labels) = self.scoped_labels().as_ref() {
            metrics::counter!(name, labels.as_ref()).increment(value);
        } else {
            metrics::counter!(name).increment(value);
        }
    }
}

trait MetricsInner: Send + Sync {
    fn render(&self) -> String;
    fn render_to_write(&self, writer: Box<dyn io::Write>) -> io::Result<()>;
    fn listen_path(&self) -> Option<String>;
}

struct NoopMetrics {}

impl MetricsInner for NoopMetrics {
    fn render(&self) -> String {
        String::new()
    }
    fn render_to_write(&self, _: Box<dyn io::Write>) -> io::Result<()> {
        Ok(())
    }
    fn listen_path(&self) -> Option<String> {
        None
    }
}

struct DogStatsDMetrics {}

impl DogStatsDMetrics {
    pub fn new(config: config::dist::DogStatsD) -> Result<Self> {
        let mut builder = DogStatsDBuilder::default();

        builder = builder.with_remote_address(config.addr)?;

        if let Some(write_timeout_ms) = config.write_timeout_ms {
            builder = builder.with_write_timeout(Duration::from_millis(write_timeout_ms));
        }
        if let Some(maximum_payload_length) = config.maximum_payload_length_bytes {
            builder = builder.with_maximum_payload_length(maximum_payload_length)?;
        }
        if let Some(aggregation_mode) = config.aggregation_mode {
            builder = builder.with_aggregation_mode(match aggregation_mode {
                config::dist::DogStatsDAggregationMode::Aggressive => AggregationMode::Aggressive,
                config::dist::DogStatsDAggregationMode::Conservative => {
                    AggregationMode::Conservative
                }
            });
        }
        if let Some(write_timeout_ms) = config.flush_interval_ms {
            builder = builder.with_flush_interval(Duration::from_millis(write_timeout_ms));
        }
        if let Some(telemetry) = config.telemetry {
            builder = builder.with_telemetry(telemetry);
        }
        if let Some(histogram_sampling) = config.histogram_sampling {
            builder = builder.with_histogram_sampling(histogram_sampling);
        }
        if let Some(histogram_reservoir_size) = config.histogram_reservoir_size_bytes {
            builder = builder.with_histogram_reservoir_size(histogram_reservoir_size);
        }
        if let Some(histograms_as_distributions) = config.histograms_as_distributions {
            builder = builder.send_histograms_as_distributions(histograms_as_distributions);
        }

        metrics::set_global_recorder(builder.build()?)?;

        Ok(Self {})
    }
}

impl MetricsInner for DogStatsDMetrics {
    fn render(&self) -> String {
        String::new()
    }
    fn render_to_write(&self, _: Box<dyn io::Write>) -> io::Result<()> {
        Ok(())
    }
    fn listen_path(&self) -> Option<String> {
        None
    }
}

struct PrometheusMetrics {
    inner: PrometheusHandle,
    listen_path: Option<String>,
    #[allow(unused)]
    exporter: crate::util::AbortOnDropHandle<()>,
}

impl PrometheusMetrics {
    pub fn new(
        config: config::dist::Prometheus,
        global_labels: BTreeMap<String, String>,
    ) -> Result<Self> {
        let builder = global_labels
            .iter()
            .fold(PrometheusBuilder::new(), |builder, (key, val)| {
                builder.add_global_label(key, val)
            });

        let (recorder, exporter, listen_path) = match config {
            config::dist::Prometheus::ListenAddr {
                addr,
                idle_timeout_secs,
            } => {
                let addr = addr.unwrap_or(SocketAddr::from_str("0.0.0.0:9000")?);
                let (recorder, exporter) = builder
                    .idle_timeout(
                        metrics_util::MetricKindMask::ALL,
                        idle_timeout_secs.map(Duration::from_secs),
                    )
                    .with_http_listener(addr)
                    .build()?;
                tracing::info!("Listening for metrics at {addr}");
                (recorder, exporter, None)
            }
            config::dist::Prometheus::ListenPath {
                path,
                idle_timeout_secs,
            } => {
                let path = path.clone().unwrap_or("/metrics".to_owned());
                let (recorder, exporter) = builder
                    .idle_timeout(
                        metrics_util::MetricKindMask::ALL,
                        idle_timeout_secs.map(Duration::from_secs),
                    )
                    .build()?;
                tracing::info!("Listening for metrics at {path}");
                (recorder, exporter, Some(path))
            }
            config::dist::Prometheus::PushGateway {
                ref endpoint,
                interval_ms,
                username,
                password,
                http_method,
                idle_timeout_secs,
            } => {
                let interval = Duration::from_millis(interval_ms);
                let (recorder, exporter) = builder
                    .set_bucket_duration(interval)?
                    .idle_timeout(
                        metrics_util::MetricKindMask::ALL,
                        idle_timeout_secs.map(Duration::from_secs),
                    )
                    .with_push_gateway(
                        endpoint,
                        interval,
                        username.clone(),
                        password.clone(),
                        http_method
                            .clone()
                            .map(|m| m.to_uppercase() == "POST")
                            .unwrap_or_default(),
                    )?
                    .build()?;
                tracing::info!(
                    "Pushing metrics to {endpoint} every {}s",
                    interval.as_secs_f64()
                );
                (recorder, exporter, None)
            }
        };

        let handle = recorder.handle();

        metrics::set_global_recorder(recorder)?;

        Ok(Self {
            inner: handle,
            listen_path,
            exporter: crate::util::spawn(async move {
                if let Err(err) = exporter.await {
                    tracing::error!("Prometheus exporter terminated with error: {err:?}");
                }
            }),
        })
    }
}

impl MetricsInner for PrometheusMetrics {
    fn render(&self) -> String {
        self.inner.render()
    }
    fn render_to_write(&self, mut writer: Box<dyn io::Write>) -> io::Result<()> {
        self.inner.render_to_write(&mut writer)?;
        writer.flush()?;
        Ok(())
    }
    fn listen_path(&self) -> Option<String> {
        self.listen_path.clone()
    }
}

// Takes a slice so the metrics structs below share one `BTreeMap` collect (and
// the sort inside it) instead of instantiating one per array iterator type.
fn describe_metrics(
    prefix: &str,
    defs: &[(&'static str, metrics::Unit, &'static str)],
) -> BTreeMap<&'static str, Cow<'static, str>> {
    defs.iter()
        .map(|&(name, unit, desc)| {
            let metric_name = Cow::<'static, str>::Owned(format!("{prefix}::{name}"));
            if matches!(unit, metrics::Unit::Count) {
                metrics::describe_counter!(metric_name.clone(), unit, desc);
            } else {
                metrics::describe_histogram!(metric_name.clone(), unit, desc);
            }
            (name, metric_name)
        })
        .collect()
}

const HAS_JOB_INPUTS_TIME: &str = "has_job_inputs_time";
const HAS_JOB_RESULT_TIME: &str = "has_job_result_time";
const HAS_JOB_STATUS_TIME: &str = "has_job_status_time";

const DEL_JOB_INPUTS_TIME: &str = "del_job_inputs_time";
const DEL_JOB_RESULT_TIME: &str = "del_job_result_time";
const DEL_JOB_STATUS_TIME: &str = "del_job_status_time";

const GET_JOB_INPUTS_TIME: &str = "get_job_inputs_time";
const GET_JOB_RESULT_TIME: &str = "get_job_result_time";
const GET_JOB_STATUS_TIME: &str = "get_job_status_time";

const PUT_JOB_INPUTS_TIME: &str = "put_job_inputs_time";
const PUT_JOB_RESULT_TIME: &str = "put_job_result_time";
const PUT_JOB_STATUS_TIME: &str = "put_job_status_time";

const DEL_JOB_INPUTS_ERROR_COUNT: &str = "del_job_inputs_error_count";
const DEL_JOB_RESULT_ERROR_COUNT: &str = "del_job_result_error_count";
const DEL_JOB_STATUS_ERROR_COUNT: &str = "del_job_status_error_count";

const GET_JOB_INPUTS_ERROR_COUNT: &str = "get_job_inputs_error_count";
const GET_JOB_RESULT_ERROR_COUNT: &str = "get_job_result_error_count";
const GET_JOB_STATUS_ERROR_COUNT: &str = "get_job_status_error_count";

const PUT_JOB_INPUTS_ERROR_COUNT: &str = "put_job_inputs_error_count";
const PUT_JOB_RESULT_ERROR_COUNT: &str = "put_job_result_error_count";
const PUT_JOB_STATUS_ERROR_COUNT: &str = "put_job_status_error_count";

#[derive(Clone)]
pub struct JobsMetrics {
    metrics: Metrics,
    m_names: BTreeMap<&'static str, Cow<'static, str>>,
}

impl Default for JobsMetrics {
    fn default() -> Self {
        Self::new("sccache", Default::default())
    }
}

impl JobsMetrics {
    pub fn new(prefix: &str, metrics: Metrics) -> Self {
        use metrics::Unit::{Count, Seconds};
        let m_names = [
            (
                HAS_JOB_INPUTS_TIME,
                Seconds,
                "The time to check if each job's inputs exists.",
            ),
            (
                HAS_JOB_RESULT_TIME,
                Seconds,
                "The time to check if each job's result exists.",
            ),
            (
                HAS_JOB_STATUS_TIME,
                Seconds,
                "The time to check if each job's status exists.",
            ),
            (
                GET_JOB_INPUTS_TIME,
                Seconds,
                "The time to load each job's inputs.",
            ),
            (
                GET_JOB_RESULT_TIME,
                Seconds,
                "The time to load each job's result.",
            ),
            (
                GET_JOB_STATUS_TIME,
                Seconds,
                "The time to load each job's status.",
            ),
            (
                DEL_JOB_INPUTS_TIME,
                Seconds,
                "The time to delete each job's inputs.",
            ),
            (
                DEL_JOB_RESULT_TIME,
                Seconds,
                "The time to delete each job's result.",
            ),
            (
                DEL_JOB_STATUS_TIME,
                Seconds,
                "The time to delete each job's status.",
            ),
            (
                PUT_JOB_INPUTS_TIME,
                Seconds,
                "The time to store each job's inputs.",
            ),
            (
                PUT_JOB_RESULT_TIME,
                Seconds,
                "The time to store each job's result.",
            ),
            (
                PUT_JOB_STATUS_TIME,
                Seconds,
                "The time to store each job's status.",
            ),
            (
                DEL_JOB_INPUTS_ERROR_COUNT,
                Count,
                "The number of errors raised deleting job inputs.",
            ),
            (
                DEL_JOB_RESULT_ERROR_COUNT,
                Count,
                "The number of errors raised deleting job results.",
            ),
            (
                DEL_JOB_STATUS_ERROR_COUNT,
                Count,
                "The number of errors raised deleting job statuses.",
            ),
            (
                GET_JOB_INPUTS_ERROR_COUNT,
                Count,
                "The number of errors raised while loading job inputs.",
            ),
            (
                GET_JOB_RESULT_ERROR_COUNT,
                Count,
                "The number of errors raised while loading job results.",
            ),
            (
                GET_JOB_STATUS_ERROR_COUNT,
                Count,
                "The number of errors raised while loading job statuses.",
            ),
            (
                PUT_JOB_INPUTS_ERROR_COUNT,
                Count,
                "The number of errors raised storing job inputs.",
            ),
            (
                PUT_JOB_RESULT_ERROR_COUNT,
                Count,
                "The number of errors raised storing job results.",
            ),
            (
                PUT_JOB_STATUS_ERROR_COUNT,
                Count,
                "The number of errors raised storing job statuses.",
            ),
        ];
        let m_names = describe_metrics(prefix, &m_names);

        Self { metrics, m_names }
    }

    fn m_name(&self, name: &str) -> Cow<'static, str> {
        self.m_names.get(name).unwrap().clone()
    }

    pub fn has_job_inputs_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(HAS_JOB_INPUTS_TIME))
    }

    pub fn has_job_result_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(HAS_JOB_RESULT_TIME))
    }

    pub fn has_job_status_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(HAS_JOB_STATUS_TIME))
    }

    pub fn get_job_inputs_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(GET_JOB_INPUTS_TIME))
    }

    pub fn get_job_result_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(GET_JOB_RESULT_TIME))
    }

    pub fn get_job_status_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(GET_JOB_STATUS_TIME))
    }

    pub fn del_job_inputs_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(DEL_JOB_INPUTS_TIME))
    }

    pub fn del_job_result_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(DEL_JOB_RESULT_TIME))
    }

    pub fn del_job_status_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(DEL_JOB_STATUS_TIME))
    }

    pub fn put_job_inputs_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(PUT_JOB_INPUTS_TIME))
    }

    pub fn put_job_result_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(PUT_JOB_RESULT_TIME))
    }

    pub fn put_job_status_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(PUT_JOB_STATUS_TIME))
    }

    pub fn inc_del_job_inputs_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(DEL_JOB_INPUTS_ERROR_COUNT))
    }

    pub fn inc_del_job_result_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(DEL_JOB_RESULT_ERROR_COUNT))
    }

    pub fn inc_del_job_status_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(DEL_JOB_STATUS_ERROR_COUNT))
    }

    pub fn inc_get_job_inputs_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(GET_JOB_INPUTS_ERROR_COUNT))
    }

    pub fn inc_get_job_result_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(GET_JOB_RESULT_ERROR_COUNT))
    }

    pub fn inc_get_job_status_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(GET_JOB_STATUS_ERROR_COUNT))
    }

    pub fn inc_put_job_inputs_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(PUT_JOB_INPUTS_ERROR_COUNT))
    }

    pub fn inc_put_job_result_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(PUT_JOB_RESULT_ERROR_COUNT))
    }

    pub fn inc_put_job_status_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(PUT_JOB_STATUS_ERROR_COUNT))
    }
}

const HAS_TOOLCHAIN_TIME: &str = "has_toolchain_time";
const DEL_TOOLCHAIN_TIME: &str = "del_toolchain_time";
const GET_TOOLCHAIN_TIME: &str = "get_toolchain_time";
const PUT_TOOLCHAIN_TIME: &str = "put_toolchain_time";
const DEL_TOOLCHAIN_ERROR_COUNT: &str = "del_toolchain_error_count";
const GET_TOOLCHAIN_ERROR_COUNT: &str = "get_toolchain_error_count";
const PUT_TOOLCHAIN_ERROR_COUNT: &str = "put_toolchain_error_count";

const TOOLCHAIN_LOAD_TIME: &str = "toolchain::load_time";
const TOOLCHAIN_LOAD_INFLATED_TIME: &str = "toolchain::load_inflated_time";
const TOOLCHAIN_LOAD_DEFLATED_TIME: &str = "toolchain::load_deflated_time";
const TOOLCHAIN_LOAD_INFLATED_SIZE_TIME: &str = "toolchain::load_inflated_size_time";
const TOOLCHAIN_UNPACK_INFLATED_TIME: &str = "toolchain::unpack_inflated_time";

#[derive(Clone)]
pub struct ToolchainsMetrics {
    metrics: Metrics,
    m_names: BTreeMap<&'static str, Cow<'static, str>>,
}

impl ToolchainsMetrics {
    pub fn new(prefix: &str, metrics: Metrics) -> Self {
        use metrics::Unit::{Count, Seconds};
        let m_names = [
            (
                HAS_TOOLCHAIN_TIME,
                Seconds,
                "The time to check if a toolchain exists.",
            ),
            (
                DEL_TOOLCHAIN_TIME,
                Seconds,
                "The time to delete each toolchain.",
            ),
            (
                GET_TOOLCHAIN_TIME,
                Seconds,
                "The time to load each toolchain.",
            ),
            (
                PUT_TOOLCHAIN_TIME,
                Seconds,
                "The time to store each toolchain.",
            ),
            (
                DEL_TOOLCHAIN_ERROR_COUNT,
                Count,
                "The number of errors raised deleting toolchains.",
            ),
            (
                GET_TOOLCHAIN_ERROR_COUNT,
                Count,
                "The number of errors raised loading toolchains.",
            ),
            (
                PUT_TOOLCHAIN_ERROR_COUNT,
                Count,
                "The number of errors raised storing toolchains.",
            ),
            (
                TOOLCHAIN_LOAD_TIME,
                Seconds,
                "The time to load a toolchain", //
            ),
            (
                TOOLCHAIN_LOAD_INFLATED_TIME,
                Seconds,
                "The time to load, inflate, and unpack a toolchain",
            ),
            (
                TOOLCHAIN_LOAD_DEFLATED_TIME,
                Seconds,
                "The time to load a deflated toolchain",
            ),
            (
                TOOLCHAIN_LOAD_INFLATED_SIZE_TIME,
                Seconds,
                "The time to calculate the inflated size of a toolchain",
            ),
            (
                TOOLCHAIN_UNPACK_INFLATED_TIME,
                Seconds,
                "The time to inflate and unpack a toolchain",
            ),
        ];
        let m_names = describe_metrics(prefix, &m_names);

        Self { metrics, m_names }
    }

    fn m_name(&self, name: &str) -> Cow<'static, str> {
        self.m_names.get(name).unwrap().clone()
    }

    pub fn has_toolchain_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(HAS_TOOLCHAIN_TIME))
    }

    pub fn get_toolchain_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(GET_TOOLCHAIN_TIME))
    }

    pub fn del_toolchain_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(DEL_TOOLCHAIN_TIME))
    }

    pub fn put_toolchain_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(PUT_TOOLCHAIN_TIME))
    }

    pub fn inc_del_toolchain_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(DEL_TOOLCHAIN_ERROR_COUNT))
    }

    pub fn inc_get_toolchain_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(GET_TOOLCHAIN_ERROR_COUNT))
    }

    pub fn inc_put_toolchain_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(PUT_TOOLCHAIN_ERROR_COUNT))
    }

    pub fn load_toolchain_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(TOOLCHAIN_LOAD_TIME))
    }

    pub fn load_inflated_toolchain_timer(&self) -> TimeRecorder {
        self.metrics
            .timer(self.m_name(TOOLCHAIN_LOAD_INFLATED_TIME))
    }

    pub fn load_deflated_toolchain_timer(&self) -> TimeRecorder {
        self.metrics
            .timer(self.m_name(TOOLCHAIN_LOAD_DEFLATED_TIME))
    }

    pub fn unpack_inflated_toolchain_timer(&self) -> TimeRecorder {
        self.metrics
            .timer(self.m_name(TOOLCHAIN_UNPACK_INFLATED_TIME))
    }
}

const NUM_CPUS: &str = "num_cpus";
const CPU_USAGE_RATIO: &str = "cpu_usage_ratio";
const MEM_AVAIL_BYTES: &str = "mem_avail_bytes";
const MEM_TOTAL_BYTES: &str = "mem_total_bytes";
const MEM_USED_BYTES: &str = "mem_used_bytes";

#[derive(Clone)]
pub struct SysinfoMetrics {
    m_names: BTreeMap<&'static str, Cow<'static, str>>,
    metrics: Metrics,
    num_cpus: usize,
    sysinfo: Arc<std::sync::Mutex<(Instant, sysinfo::System)>>,
}

impl SysinfoMetrics {
    pub fn new(prefix: &str, metrics: Metrics, num_cpus: Option<usize>) -> Self {
        use metrics::Unit::{Bytes, Count, Percent};
        let m_names = [
            (
                NUM_CPUS,
                Count,
                "The total number of CPUs.", //
            ),
            (
                CPU_USAGE_RATIO,
                Percent,
                "The current system CPU usage percent (0-100).",
            ),
            (
                MEM_AVAIL_BYTES,
                Bytes,
                "The amount of free system memory.", //
            ),
            (
                MEM_TOTAL_BYTES,
                Bytes,
                "The total amount of system memory.", //
            ),
            (
                MEM_USED_BYTES,
                Bytes,
                "The amount of used system memory.", //
            ),
        ];
        let m_names = describe_metrics(prefix, &m_names);

        let sysinfo = sysinfo::System::new_with_specifics(
            sysinfo::RefreshKind::nothing()
                .with_cpu(sysinfo::CpuRefreshKind::nothing().with_cpu_usage())
                .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
        );

        let num_cpus = num_cpus
            .or_else(|| sysinfo.physical_core_count())
            .unwrap_or_default();

        Self {
            metrics,
            m_names,
            num_cpus,
            sysinfo: Arc::new(std::sync::Mutex::new((Instant::now(), sysinfo))),
        }
    }

    fn m_name(&self, name: &str) -> Cow<'static, str> {
        self.m_names.get(name).unwrap().clone()
    }

    pub fn system_metrics(&self) -> (usize, f32, u64, u64) {
        let mut pair = self.sysinfo.lock().unwrap();

        if Instant::now().duration_since(pair.0) > sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
            pair.0 = Instant::now();
            pair.1
                .refresh_cpu_specifics(sysinfo::CpuRefreshKind::nothing().with_cpu_usage());
        }

        pair.1
            .refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());

        let cpu_usage = pair.1.global_cpu_usage();
        let mem_avail = pair.1.available_memory();
        let mem_total = pair.1.total_memory();
        self.metrics
            .histo(self.m_name(NUM_CPUS), self.num_cpus as f64);
        self.metrics.histo(self.m_name(CPU_USAGE_RATIO), cpu_usage);
        self.metrics
            .histo(self.m_name(MEM_AVAIL_BYTES), mem_avail as f64);
        self.metrics
            .histo(self.m_name(MEM_TOTAL_BYTES), mem_total as f64);
        self.metrics.histo(
            self.m_name(MEM_USED_BYTES),
            mem_total.saturating_sub(mem_avail) as f64,
        );

        (self.num_cpus, cpu_usage, mem_avail, mem_total)
    }
}

#[derive(Clone)]
pub struct SchedulerMetrics {
    #[allow(unused)]
    metrics: Metrics,
    sysinfo: SysinfoMetrics,
}

impl SchedulerMetrics {
    pub fn new(prefix: &str, metrics: Metrics) -> Self {
        let sysinfo = SysinfoMetrics::new(prefix, metrics.clone(), None);

        Self { metrics, sysinfo }
    }

    pub fn system_metrics(&self) -> (usize, f32, u64, u64) {
        self.sysinfo.system_metrics()
    }
}

const JOB_BUILD_ERROR_COUNT: &str = "job_build_error_count";
const JOB_ACCEPTED_COUNT: &str = "job_accepted_count";
const JOB_LOADED_COUNT: &str = "job_loaded_count";
const JOB_FINISHED_COUNT: &str = "job_finished_count";
const JOB_PENDING_COUNT: &str = "job_pending_count";
const JOB_LOADING_COUNT: &str = "job_loading_count";
const LOAD_JOB_TIME: &str = "load_job_time";
const RUN_BUILD_TIME: &str = "run_build_time";
const RUN_JOB_TIME: &str = "run_job_time";

#[derive(Clone)]
pub struct ServerMetrics {
    pub jobs_accepted: Arc<AtomicU64>,
    pub jobs_loaded: Arc<AtomicU64>,
    pub jobs_finished: Arc<AtomicU64>,
    pub jobs_cancelled: Arc<AtomicU64>,
    pub jobs_pending: Arc<GaugeRecorder>,
    pub jobs_loading: Arc<GaugeRecorder>,
    metrics: Metrics,
    m_names: BTreeMap<&'static str, Cow<'static, str>>,
    sysinfo: SysinfoMetrics,
}

impl ServerMetrics {
    pub fn new(prefix: &str, metrics: Metrics, num_cpus: usize) -> Self {
        use metrics::Unit::{Count, Seconds};
        let m_names = [
            (
                JOB_LOADING_COUNT,
                Count,
                "The number of accepted jobs for which this server is loading inputs and toolchains.",
            ),
            (
                JOB_PENDING_COUNT,
                Count,
                "The number of accepted jobs that are fully loaded and queued to run/are currently running.",
            ),
            (
                JOB_BUILD_ERROR_COUNT,
                Count,
                "The number of errors raised while running job builds.",
            ),
            (
                JOB_ACCEPTED_COUNT,
                Count,
                "The total number of jobs accepted by this server (but not yet loaded or run).",
            ),
            (
                JOB_LOADED_COUNT,
                Count,
                "The total number of jobs loaded by this server (but not yet run).",
            ),
            (
                JOB_FINISHED_COUNT,
                Count,
                "The total number of jobs accepted, loaded, and run by this server.",
            ),
            (
                LOAD_JOB_TIME,
                Seconds,
                "The time to load each job's inputs and toolchains.",
            ),
            (RUN_BUILD_TIME, Seconds, "The time to run each job's build."),
            (
                RUN_JOB_TIME,
                Seconds,
                "The time to load and build each job.",
            ),
        ];
        let m_names = describe_metrics(prefix, &m_names);

        let jobs_pending = Arc::new(metrics.gauge(m_names.get(JOB_PENDING_COUNT).unwrap().clone()));
        let jobs_loading = Arc::new(metrics.gauge(m_names.get(JOB_LOADING_COUNT).unwrap().clone()));
        let sysinfo = SysinfoMetrics::new(prefix, metrics.clone(), num_cpus.into());

        Self {
            m_names,
            metrics,
            jobs_pending,
            jobs_loading,
            jobs_accepted: Default::default(),
            jobs_loaded: Default::default(),
            jobs_finished: Default::default(),
            jobs_cancelled: Default::default(),
            sysinfo,
        }
    }

    fn m_name(&self, name: &str) -> Cow<'static, str> {
        self.m_names.get(name).unwrap().clone()
    }

    pub fn system_metrics(&self) -> (usize, f32, u64, u64) {
        self.sysinfo.system_metrics()
    }

    pub fn inc_job_build_error_count(&self) -> CountRecorder {
        self.metrics.count(self.m_name(JOB_BUILD_ERROR_COUNT))
    }

    pub fn inc_job_accepted_count(&self) -> CountRecorder {
        self.jobs_accepted
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.metrics.count(self.m_name(JOB_ACCEPTED_COUNT))
    }

    pub fn inc_job_loaded_count(&self) -> CountRecorder {
        self.jobs_loaded
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.metrics.count(self.m_name(JOB_LOADED_COUNT))
    }

    pub fn inc_job_finished_count(&self) -> CountRecorder {
        self.jobs_finished
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.metrics.count(self.m_name(JOB_FINISHED_COUNT))
    }

    pub fn load_job_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(LOAD_JOB_TIME))
    }

    pub fn run_build_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(RUN_BUILD_TIME))
    }

    pub fn run_job_timer(&self) -> TimeRecorder {
        self.metrics.timer(self.m_name(RUN_JOB_TIME))
    }
}

#[cfg(test)]
mod test {
    use metrics::{
        Counter, Gauge, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
    };
    use std::sync::Mutex;

    #[derive(Default)]
    struct Descriptions(Mutex<Vec<(&'static str, String, Option<Unit>, String)>>);

    impl Descriptions {
        fn push(&self, kind: &'static str, key: KeyName, unit: Option<Unit>, desc: SharedString) {
            let entry = (kind, key.as_str().to_owned(), unit, desc.into_owned());
            self.0.lock().unwrap().push(entry);
        }
    }

    impl Recorder for Descriptions {
        fn describe_counter(&self, key: KeyName, unit: Option<Unit>, desc: SharedString) {
            self.push("counter", key, unit, desc);
        }
        fn describe_gauge(&self, key: KeyName, unit: Option<Unit>, desc: SharedString) {
            self.push("gauge", key, unit, desc);
        }
        fn describe_histogram(&self, key: KeyName, unit: Option<Unit>, desc: SharedString) {
            self.push("histogram", key, unit, desc);
        }
        fn register_counter(&self, _: &Key, _: &Metadata<'_>) -> Counter {
            Counter::noop()
        }
        fn register_gauge(&self, _: &Key, _: &Metadata<'_>) -> Gauge {
            Gauge::noop()
        }
        fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> Histogram {
            Histogram::noop()
        }
    }

    #[test]
    fn describe_metrics_describes_counts_as_counters_and_the_rest_as_histograms() {
        let recorder = Descriptions::default();
        let names = metrics::with_local_recorder(&recorder, || {
            super::describe_metrics(
                "sccache::test",
                &[
                    ("b", Unit::Count, "Counts b."),
                    ("a", Unit::Seconds, "Times a."),
                ],
            )
        });
        assert_eq!(
            names.into_iter().collect::<Vec<_>>(),
            [
                ("a", "sccache::test::a".into()),
                ("b", "sccache::test::b".into())
            ]
        );
        let described = |kind, name: &str, unit, desc: &str| {
            (kind, name.to_owned(), Some(unit), desc.to_owned())
        };
        assert_eq!(
            recorder.0.into_inner().unwrap(),
            [
                described("counter", "sccache::test::b", Unit::Count, "Counts b."),
                described("histogram", "sccache::test::a", Unit::Seconds, "Times a."),
            ]
        );
    }
}
