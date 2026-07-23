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
    collections::BTreeMap,
    io,
    net::SocketAddr,
    str::FromStr,
    sync::{Arc, atomic::AtomicU64},
    time::{Duration, Instant},
};

use crate::{
    config::{
        DogStatsDAggregationMode, DogStatsDMetricsConfig, MetricsConfigs, PrometheusMetricsConfig,
    },
    errors::*,
};

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
    pub fn new(config: MetricsConfigs, global_labels: BTreeMap<String, String>) -> Result<Self> {
        if let Some(config) = config.dogstatsd {
            Ok(Self {
                global_labels: Arc::new(global_labels),
                inner: Arc::new(DogStatsDMetrics::new(config)?),
            })
        } else if let Some(config) = config.prometheus {
            Ok(Self {
                global_labels: Arc::new(Default::default()),
                inner: Arc::new(PrometheusMetrics::new(config, global_labels)?),
            })
        } else {
            Ok(Self {
                global_labels: Arc::new(global_labels),
                inner: Arc::new(NoopMetrics {}),
            })
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
    pub fn new(config: DogStatsDMetricsConfig) -> Result<Self> {
        let mut builder = DogStatsDBuilder::default();

        builder = builder.with_remote_address(config.remote_addr)?;

        if let Some(write_timeout) = config.write_timeout {
            builder = builder.with_write_timeout(Duration::from_millis(write_timeout));
        }
        if let Some(maximum_payload_length) = config.maximum_payload_length {
            builder = builder.with_maximum_payload_length(maximum_payload_length)?;
        }
        if let Some(aggregation_mode) = config.aggregation_mode {
            builder = builder.with_aggregation_mode(match aggregation_mode {
                DogStatsDAggregationMode::Aggressive => AggregationMode::Aggressive,
                DogStatsDAggregationMode::Conservative => AggregationMode::Conservative,
            });
        }
        if let Some(flush_interval) = config.flush_interval {
            builder = builder.with_flush_interval(Duration::from_millis(flush_interval));
        }
        if let Some(telemetry) = config.telemetry {
            builder = builder.with_telemetry(telemetry);
        }
        if let Some(histogram_sampling) = config.histogram_sampling {
            builder = builder.with_histogram_sampling(histogram_sampling);
        }
        if let Some(histogram_reservoir_size) = config.histogram_reservoir_size {
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
    exporter: tokio_util::task::AbortOnDropHandle<()>,
}

impl PrometheusMetrics {
    pub fn new(
        config: PrometheusMetricsConfig,
        global_labels: BTreeMap<String, String>,
    ) -> Result<Self> {
        let builder = global_labels
            .iter()
            .fold(PrometheusBuilder::new(), |builder, (key, val)| {
                builder.add_global_label(key, val)
            });

        let (recorder, exporter, listen_path) = match config {
            PrometheusMetricsConfig::ListenAddr {
                ref addr,
                ref idle_timeout_secs,
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
            PrometheusMetricsConfig::ListenPath {
                ref path,
                ref idle_timeout_secs,
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
            PrometheusMetricsConfig::PushGateway {
                ref endpoint,
                ref interval,
                ref username,
                ref password,
                ref http_method,
                ref idle_timeout_secs,
            } => {
                let interval = Duration::from_millis(interval.unwrap_or(10_000));
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
