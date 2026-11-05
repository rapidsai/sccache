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

use crate::config::utils::deserialize_bool;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub mod client;
#[cfg(feature = "dist-server")]
pub mod scheduler;
#[cfg(feature = "dist-server")]
pub mod server;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Keepalive {
    #[serde(
        default = "defaults::default_true",
        deserialize_with = "deserialize_bool"
    )]
    pub enabled: bool,
    #[serde(default = "defaults::default_keepalive_interval")]
    pub interval: u64,
    #[serde(default = "defaults::default_keepalive_timeout")]
    pub timeout: u64,
}

// Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
impl Default for Keepalive {
    fn default() -> Self {
        Self {
            enabled: defaults::default_true(),
            interval: defaults::default_keepalive_interval(),
            timeout: defaults::default_keepalive_timeout(),
        }
    }
}

#[cfg(feature = "dist-server")]
pub use dist_server::*;

#[cfg(feature = "dist-server")]
mod dist_server {
    use super::*;

    #[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
    pub struct MessageBroker {
        #[serde(default)]
        pub addr: String,
        #[serde(default = "defaults::default_broker_connection_timeout")]
        pub connection_timeout: u32,
        #[serde(default = "defaults::default_true")]
        pub connection_retry: bool,
        #[serde(default = "defaults::default_broker_max_retries")]
        pub max_retries: u32,
        #[serde(default = "defaults::default_broker_retry_delay")]
        pub retry_delay: u32,
    }

    // Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
    impl Default for MessageBroker {
        fn default() -> Self {
            Self {
                addr: String::new(),
                connection_timeout: defaults::default_broker_connection_timeout(),
                connection_retry: defaults::default_true(),
                max_retries: defaults::default_broker_max_retries(),
                retry_delay: defaults::default_broker_retry_delay(),
            }
        }
    }

    impl From<MessageBroker> for String {
        fn from(MessageBroker { addr, .. }: MessageBroker) -> Self {
            addr
        }
    }

    impl From<String> for MessageBroker {
        fn from(addr: String) -> Self {
            Self {
                addr,
                ..Default::default()
            }
        }
    }

    impl<'a> From<&'a str> for MessageBroker {
        fn from(addr: &'a str) -> Self {
            Self {
                addr: addr.into(),
                ..Default::default()
            }
        }
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Metrics {
        #[default]
        None,
        Dogstatsd(DogStatsD),
        Prometheus(Prometheus),
    }

    #[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
    pub struct DogStatsD {
        #[serde(alias = "remote_addr")]
        pub addr: String,

        // Write timeout (in milliseconds) for forwarding metrics to dogstatsd
        #[serde(default)]
        pub write_timeout_ms: Option<u64>,

        // Maximum payload length (in bytes) for forwarding metrics
        #[serde(default, alias = "maximum_payload_length")]
        pub maximum_payload_length_bytes: Option<usize>,

        // The aggregation mode for the exporter
        #[serde(default)]
        pub aggregation_mode: Option<DogStatsDAggregationMode>,

        // The flush interval of the aggregator
        #[serde(default, alias = "flush_interval")]
        pub flush_interval_ms: Option<u64>,

        // Whether or not to enable telemetry for the exporter
        #[serde(default)]
        pub telemetry: Option<bool>,

        // Whether or not to enable histogram sampling
        #[serde(default)]
        pub histogram_sampling: Option<bool>,

        // The reservoir size (in bytes) for histogram sampling
        #[serde(default, alias = "histogram_reservoir_size")]
        pub histogram_reservoir_size_bytes: Option<usize>,

        // Whether or not to send histograms as distributions
        #[serde(default)]
        pub histograms_as_distributions: Option<bool>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(rename_all = "lowercase")]
    pub enum DogStatsDAggregationMode {
        Aggressive,
        Conservative,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(tag = "type", rename_all = "lowercase")]
    pub enum Prometheus {
        #[serde(rename = "bind")]
        ListenAddr {
            addr: Option<std::net::SocketAddr>,
            idle_timeout_secs: Option<u64>,
        },
        #[serde(rename = "path")]
        ListenPath {
            path: Option<String>,
            idle_timeout_secs: Option<u64>,
        },
        #[serde(rename = "push")]
        PushGateway {
            endpoint: String,
            // Interval (in milliseconds) to push metrics to prometheus
            #[serde(default = "defaults::default_prometheus_push_gateway_interval")]
            interval_ms: u64,
            username: Option<String>,
            password: Option<String>,
            http_method: Option<String>,
            idle_timeout_secs: Option<u64>,
        },
    }
}

pub mod defaults {
    use super::*;

    pub use crate::config::defaults::default_true;

    pub fn default_broker_connection_timeout() -> u32 {
        Duration::from_secs(5).as_secs() as u32
    }

    pub fn default_broker_max_retries() -> u32 {
        5
    }

    pub fn default_broker_retry_delay() -> u32 {
        Duration::from_secs(5).as_secs() as u32
    }

    pub fn default_keepalive_interval() -> u64 {
        Duration::from_secs(20).as_secs()
    }

    pub fn default_keepalive_timeout() -> u64 {
        Duration::from_secs(600).as_secs()
    }

    // Default to 15s
    pub fn default_heartbeat_interval() -> u64 {
        Duration::from_secs(15).as_millis() as u64
    }

    pub fn default_shutdown_timeout() -> u64 {
        Duration::from_secs(10).as_secs()
    }

    pub fn default_prometheus_push_gateway_interval() -> u64 {
        Duration::from_secs(10).as_secs()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn defaults_match_serde_defaults() {
        assert_eq!(Keepalive::default(), serde_json::from_str("{}").unwrap());
        #[cfg(feature = "dist-server")]
        assert_eq!(
            MessageBroker::default(),
            serde_json::from_str("{}").unwrap()
        );
    }
}
