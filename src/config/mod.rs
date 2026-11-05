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

use crate::errors::*;
use itertools::Itertools;
use serde::{de, ser};
use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

#[macro_use]
pub mod utils;

pub mod cache;
pub mod client;
pub mod dist;

pub use cache::{
    Cache,              //
    Cache as CacheType, //
    CacheMode,          //
    Caches,             //
    WriteErrorPolicy,   //
};

pub use client::{CachedConfig, CachedFileConfig, Config as ClientConfig};

const ORGANIZATION: &str = "Mozilla";
const APP_NAME: &str = "sccache";
const DIST_APP_NAME: &str = "sccache-dist-client";
const TEN_GIGS: u64 = 10 * 1024 * 1024 * 1024;
pub const INSECURE_DIST_CLIENT_TOKEN: &str = "dangerously_insecure_client";

// The loaders below take the collected vars, so that they and serde_env's parsing are compiled
// once per config type rather than again for every caller's iterator type.
fn collect_vars<I, S>(vars: I) -> Vec<(OsString, OsString)>
where
    I: IntoIterator<Item = (S, S)>,
    S: AsRef<OsStr>,
{
    vars.into_iter()
        .map(|(key, val)| (key.as_ref().to_owned(), val.as_ref().to_owned()))
        .collect()
}

trait Loadable<D>
where
    D: de::DeserializeOwned + ser::Serialize + Valid + Clone + Default + PartialEq,
{
    fn env_prefix<'a>() -> Option<&'a str> {
        None
    }

    fn merge(file: &D, vars: &D) -> Result<D> {
        Self::merge_impl(file, vars)
    }

    fn merge_impl(file: &D, vars: &D) -> Result<D> {
        let mut conf = Default::default();

        // Diff the default and the file conf
        let file_diff = if file != &conf {
            Some(Self::json_diff(&conf, file)?)
        } else {
            None
        };

        // Diff the default and the vars conf
        let vars_diff = if vars != &conf {
            Some(Self::json_diff(&conf, vars)?)
        } else {
            None
        };

        // Apply the file conf diff first, then the vars conf diff to override it
        for diff in [file_diff, vars_diff].into_iter().flatten() {
            let mut value = Self::to_json_value(&conf)?;
            serde_patch::apply_mut(&mut value, diff)?;
            conf = Self::from_json_impl(&value.to_string())?;
        }

        conf.validate_merged(file)
    }

    fn json_diff(from: &D, to: &D) -> Result<String> {
        let diff = serde_patch::diff(&Self::to_json_value(from)?, &Self::to_json_value(to)?)?;
        Ok(diff.to_string())
    }

    // The merge goes through JSON text because serde_json::to_value/from_value would compile
    // another Serialize and Deserialize of the whole config tree, while to_string_pretty
    // (--show-config) and from_json_impl (JSON config files) are compiled anyway.
    fn to_json_value(conf: &D) -> Result<serde_json::Value> {
        Ok(serde_json::to_string_pretty(conf)?.parse()?)
    }

    fn from_envs_and_path<I, S, O, P>(vars: I, path: O) -> Result<D>
    where
        I: IntoIterator<Item = (S, S)>,
        S: AsRef<OsStr>,
        O: Into<Option<P>>,
        P: AsRef<Path>,
    {
        let path: Option<P> = path.into();
        Self::from_var_list_and_path(&collect_vars(vars), path.as_ref().map(AsRef::as_ref))
    }

    fn from_var_list_and_path(vars: &[(OsString, OsString)], path: Option<&Path>) -> Result<D> {
        // Read config from the environment
        let conf = Self::from_var_list(vars)?;

        // Maybe read config from the config file
        if let Some(file) = Self::from_path_or_conf(vars, path) {
            Self::merge(&(file?), &conf)
        } else {
            conf.validate_vars()
        }
    }

    #[cfg(test)]
    fn from_vars<I, S>(vars: I) -> Result<D>
    where
        I: IntoIterator<Item = (S, S)>,
        S: AsRef<OsStr>,
    {
        Self::from_var_list(&collect_vars(vars))
    }

    fn from_var_list(vars: &[(OsString, OsString)]) -> Result<D> {
        Self::from_vars_impl(vars)?.validate_vars()
    }

    fn from_vars_impl(vars: &[(OsString, OsString)]) -> Result<D> {
        serde_env::from_iter(Self::select_vars(vars)?)
            .map(|d: DeserializeWithPaths<D>| d.value)
            .map_err(|e| VarsError::from((Self::env_prefix(), e)).into())
    }

    fn from_path_or_conf(vars: &[(OsString, OsString)], path: Option<&Path>) -> Option<Result<D>> {
        path.and_then(|path| {
            // If path was explicitly provided, try to read it only
            Self::from_path(path)
        })
        .or_else(|| {
            // If path was not explicitly provided, try to read from SCCACHE_CONF
            // If SCCACHE_CONF is not defined, try to read the ProjectDirectories
            Self::from_envs(vars)
        })
    }

    fn from_envs(vars: &[(OsString, OsString)]) -> Option<Result<D>> {
        let path = if let Some(prefix) = Self::env_prefix() {
            let env_var = format!("{prefix}_CONF");
            vars.iter().find_map(|(key, val)| {
                if key.as_os_str() == env_var.as_str() {
                    Some(PathBuf::from(val))
                } else {
                    None
                }
            })
        } else {
            None
        };

        // If the SCCACHE_CONF envvar exists, attempt to load it
        if let Some(path) = path {
            Self::from_path(&path)
        } else {
            // The directories crate changed the location of `config_dir` on macos in version 3,
            // so we also check the config in `preference_dir` (new in that version), which
            // corresponds to the old location, for compatibility with older setups.
            let dirs = directories::ProjectDirs::from("", ORGANIZATION, APP_NAME)
                .expect("Unable to determine config dirs");

            let dirs = [
                // If the new location exists, use that.
                dirs.config_dir().join("config"),
                dirs.config_dir().join("config.toml"),
                dirs.config_dir().join("config.json"),
                // If the old location exists, use that.
                dirs.config_dir().join("config"),
                dirs.config_dir().join("config.toml"),
                dirs.config_dir().join("config.json"),
            ];

            dirs.iter().find_map(|path| Self::from_path(path))
        }
    }

    fn from_path(path: &Path) -> Option<Result<D>> {
        use fs_err::File;
        use std::io::Read;

        let mut file = File::open(path)
            .inspect_err(|e| trace!("Couldn't open config file: {e}"))
            .ok()?;

        let mut data = String::new();
        file.read_to_string(&mut data)
            .inspect_err(|e| warn!("Failed to read config file: {e}"))
            .ok()?;

        let conf = match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => Self::from_json(&data).with_context(|| {
                format!("Failed to load json config file from {}", path.display())
            }),
            _ => Self::from_toml(&data).with_context(|| {
                format!("Failed to load toml config file from {}", path.display())
            }),
        };

        Some(conf)
    }

    fn from_json(json: &str) -> Result<D> {
        Self::from_json_impl(json).and_then(|conf| conf.validate_file())
    }

    fn from_json_impl(json: &str) -> Result<D> {
        serde_json::from_str(json)
            .map(|d: DeserializeWithPaths<D>| d.value)
            .map_err(|e| FileError::from(anyhow!(e)).into())
    }

    fn from_toml(data: &str) -> Result<D> {
        Self::from_toml_impl(data).and_then(|conf| conf.validate_file())
    }

    fn from_toml_impl(data: &str) -> Result<D> {
        toml::from_str(data)
            .map(|d: DeserializeWithPaths<D>| d.value)
            .map_err(|e| FileError::from(anyhow!(e)).into())
    }

    fn select_vars(vars: &[(OsString, OsString)]) -> Result<Vec<(String, String)>> {
        use crate::util::os_str_to_str;

        let prefix_ = Self::env_prefix().map(|prefix| {
            if prefix.ends_with("_") {
                prefix.into()
            } else {
                format!("{prefix}_")
            }
        });

        vars.iter()
            .map(|(key, val)| {
                let k = os_str_to_str(key.as_os_str())?;
                let v = os_str_to_str(val.as_os_str())?;
                if let Some(prefix_) = prefix_.as_deref() {
                    Ok(k.strip_prefix(prefix_)
                        .map(|k| (k.to_owned(), v.into_owned())))
                } else {
                    Ok(None)
                }
            })
            .flatten_ok()
            .try_collect::<_, Vec<_>, _>()
    }
}

trait Valid {
    fn validate_vars(self) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn validate_file(self) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn validate_merged(self, orig: &Self) -> Result<Self>
    where
        Self: std::marker::Sized;
}

struct DeserializeWithPaths<T: de::DeserializeOwned> {
    value: T,
}

impl<'de, T: de::DeserializeOwned> de::Deserialize<'de> for DeserializeWithPaths<T> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        match serde_path_to_error::deserialize(deserializer) {
            Ok(value) => Ok(DeserializeWithPaths { value }),
            Err(err) => {
                let path = err.path().to_string();
                let err = err.into_inner().to_string();
                Err(de::Error::custom(format!("{path}<-- path : err -->{err}")))
            }
        }
    }
}

#[derive(Debug)]
struct FileError(String);

impl std::error::Error for FileError {}

impl std::fmt::Display for FileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<anyhow::Error> for FileError {
    fn from(err: anyhow::Error) -> Self {
        let err = err.to_string();

        if let Some((path, err)) = err.split_once("<-- path : err -->") {
            let prefix = path;

            let mut pos = 0;
            let mut msg = String::new();
            let lhs = memchr::memmem::find_iter(err.as_bytes(), "{{");
            let rhs = memchr::memmem::find_iter(err.as_bytes(), "}}");
            for (lhs, rhs) in lhs.zip(rhs) {
                // take everything before {{
                msg.push_str(&err[pos..lhs]);
                // take everything between {{ and }}
                let path = &err[lhs + 2..rhs];
                msg.push_str(&format!("{prefix}.{path}"));
                // move past the }}
                pos = rhs + 2;
            }

            // If nothing was replaced, report the envvar that caused the error
            if pos == 0 {
                // take everything past the last }}
                msg.push_str(&format!("{prefix}: "));
            }

            // take everything past the last }}, or push the full error message
            if pos < err.len() {
                msg.push_str(&err[pos..]);
            }

            Self(msg)
        } else {
            Self(err)
        }
    }
}

#[derive(Debug)]
struct VarsError(String);

impl std::error::Error for VarsError {}

impl std::fmt::Display for VarsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'a> From<(Option<&'a str>, serde_env::Error)> for VarsError {
    fn from((prefix, err): (Option<&'a str>, serde_env::Error)) -> Self {
        let err = err.to_string();

        if let Some((path, err)) = err.split_once("<-- path : err -->") {
            let path = path.to_uppercase().replace('.', "_");

            let prefix = if let Some(prefix) = prefix {
                format!("{prefix}_{path}")
            } else {
                path
            };

            let mut pos = 0;
            let mut msg = String::new();
            let lhs = memchr::memmem::find_iter(err.as_bytes(), "{{");
            let rhs = memchr::memmem::find_iter(err.as_bytes(), "}}");
            for (lhs, rhs) in lhs.zip(rhs) {
                // take everything before {{
                msg.push_str(&err[pos..lhs]);
                // take everything between {{ and }}
                let path = &err[lhs + 2..rhs];
                let path = path.to_uppercase().replace('.', "_");
                msg.push_str(&format!("{prefix}_{path}"));
                // move past the }}
                pos = rhs + 2;
            }

            // If nothing was replaced, report the envvar that caused the error
            if pos == 0 {
                // take everything past the last }}
                msg.push_str(&format!("{prefix}: "));
            }

            // take everything past the last }}, or push the full error message
            if pos < err.len() {
                msg.push_str(&err[pos..]);
            }

            Self(msg)
        } else {
            Self(err)
        }
    }
}

pub mod defaults {
    use super::*;

    pub fn default_disk_cache_size() -> u64 {
        TEN_GIGS
    }

    // Unfortunately this means that nothing else can use the sccache cache dir as
    // this top level directory is used directly to store sccache cached objects...
    pub fn default_disk_cache_dir() -> PathBuf {
        directories::ProjectDirs::from("", ORGANIZATION, APP_NAME)
            .expect("Unable to retrieve disk cache directory")
            .cache_dir()
            .to_owned()
    }

    // ...whereas subdirectories are used of this one
    pub fn default_dist_cache_dir() -> PathBuf {
        directories::ProjectDirs::from("", ORGANIZATION, DIST_APP_NAME)
            .expect("Unable to retrieve dist cache directory")
            .cache_dir()
            .to_owned()
    }

    pub fn default_true() -> bool {
        true
    }
}
