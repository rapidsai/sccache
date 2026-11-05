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
use crate::config::{
    dist::Keepalive,
    utils::{deserialize_bool, deserialize_size_from_str},
};

use serde::{Deserialize, Serialize};
use serde_with::{PickFirst, TryFromInto, serde_as};
use std::path::PathBuf;

#[serde_as]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct Config {
    // dist-client authentication configuration
    #[serde(default)]
    pub auth: Auth,

    /// Local toolchains cache dir
    #[serde(default = "defaults::default_dist_cache_dir")]
    pub cache_dir: PathBuf,

    /// Whether to fallback to local compile
    #[serde(
        default = "defaults::default_true",
        deserialize_with = "deserialize_bool"
    )]
    pub fallback_to_local_compile: bool,

    // `max_retries = 0` = never retry failed dist-compiles
    // `max_retries = 1` = retry failed dist-compiles once
    // `max_retries = inf` = retry all failed dist-compiles (never compile locally)
    #[serde_as(as = "PickFirst<(TryFromInto<String>, TryFromInto<f64>)>")]
    #[serde(default = "defaults::default_max_retries")]
    pub max_retries: MaxRetries,

    // Configuration for the dist reqwest client
    #[serde(default)]
    pub net: Networking,

    #[cfg(any(feature = "dist-client", feature = "dist-server"))]
    #[serde(default, alias = "scheduler_url")]
    pub url: Option<crate::config::utils::HTTPUrl>,

    #[cfg(not(any(feature = "dist-client", feature = "dist-server")))]
    #[serde(default, alias = "scheduler_url")]
    pub url: Option<String>,

    /// Custom toolchains
    #[serde(default)]
    pub toolchains: Vec<Toolchain>,

    #[serde(
        default = "defaults::default_disk_cache_size",
        deserialize_with = "deserialize_size_from_str"
    )]
    pub toolchain_cache_size: u64,
}

// Same as the #[serde(default)]s, see `defaults_match_serde_defaults`.
impl Default for Config {
    fn default() -> Self {
        Self {
            auth: Auth::default(),
            cache_dir: defaults::default_dist_cache_dir(),
            fallback_to_local_compile: defaults::default_true(),
            max_retries: defaults::default_max_retries(),
            net: Networking::default(),
            url: None,
            toolchains: vec![],
            toolchain_cache_size: defaults::default_disk_cache_size(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Auth {
    #[serde(rename = "token")]
    Token { token: String },
    #[serde(rename = "oauth2_code_grant_pkce")]
    Oauth2CodeGrantPKCE {
        client_id: String,
        auth_url: String,
        token_url: String,
    },
    #[serde(rename = "oauth2_implicit")]
    Oauth2Implicit { client_id: String, auth_url: String },
}

impl Default for Auth {
    fn default() -> Self {
        Auth::Token {
            token: crate::config::INSECURE_DIST_CLIENT_TOKEN.to_owned(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MaxRetries {
    Inf,
    Count(u32),
}

impl Default for MaxRetries {
    fn default() -> Self {
        Self::Count(0)
    }
}

impl TryFrom<String> for MaxRetries {
    type Error = anyhow::Error;

    fn try_from(count: String) -> Result<Self, Self::Error> {
        count.parse::<f64>().map(Into::into).map_err(Into::into)
    }
}

impl From<MaxRetries> for String {
    fn from(max_retries: MaxRetries) -> Self {
        match max_retries {
            MaxRetries::Inf => "inf".into(),
            MaxRetries::Count(c) => c.to_string(),
        }
    }
}

impl From<f64> for MaxRetries {
    fn from(count: f64) -> Self {
        if count.is_infinite() {
            MaxRetries::Inf
        } else {
            MaxRetries::Count(count.clamp(0f64, u32::MAX as f64) as u32)
        }
    }
}

impl From<MaxRetries> for f64 {
    fn from(max_retries: MaxRetries) -> Self {
        match max_retries {
            MaxRetries::Inf => f64::INFINITY,
            MaxRetries::Count(c) => c as f64,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Networking {
    pub connect_timeout: u32,
    pub request_timeout: u32,
    #[serde(deserialize_with = "deserialize_bool")]
    pub connection_pool: bool,
    pub max_connections: u32,
    pub keepalive: Keepalive,
}

impl Default for Networking {
    fn default() -> Self {
        Self {
            // Default timeout for connections to an sccache-dist server
            connect_timeout: 30,
            // Default timeout for compile requests to an sccache-dist server.
            // Users should set their load balancer's idle timeout to match or
            // exceed this value.
            request_timeout: 600,
            // Default to using reqwest's HTTP/2 connection pool
            connection_pool: true,
            // Default to using 8 separate reqwest clients to maximize HTTP/2 throughput
            max_connections: 8,
            keepalive: Default::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Toolchain {
    #[serde(rename = "no_dist")]
    NoDist { compiler_executable: PathBuf },
    #[serde(rename = "path_override")]
    PathOverride {
        compiler_executable: PathBuf,
        archive: PathBuf,
        archive_compiler_executable: String,
    },
}

pub mod defaults {
    use super::MaxRetries;

    pub use crate::config::defaults::{
        default_disk_cache_size, default_dist_cache_dir, default_true,
    };

    pub fn default_max_retries() -> MaxRetries {
        MaxRetries::Count(0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn defaults_match_serde_defaults() {
        assert_eq!(Config::default(), serde_json::from_str("{}").unwrap());
    }
}
