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

use crate::{
    config::{APP_NAME, ORGANIZATION, client::Basedirs},
    errors::*,
};

use itertools::Itertools;
use serde::{Deserialize, de, ser};
use std::{
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

#[cfg(any(feature = "dist-client", feature = "dist-server"))]
pub use dist::*;

#[cfg(any(feature = "dist-client", feature = "dist-server"))]
mod dist {

    use super::*;
    use serde::Serialize;

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct HTTPUrl(reqwest::Url);

    impl HTTPUrl {
        pub fn from_url(u: reqwest::Url) -> Self {
            HTTPUrl(u)
        }
        pub fn to_url(&self) -> reqwest::Url {
            self.0.clone()
        }
    }

    impl FromStr for HTTPUrl {
        type Err = anyhow::Error;
        fn from_str(url: &str) -> std::result::Result<Self, Self::Err> {
            parse_http_url(url).map(Self)
        }
    }

    impl Serialize for HTTPUrl {
        fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: ser::Serializer,
        {
            serializer.serialize_str(self.0.as_str())
        }
    }

    impl<'a> Deserialize<'a> for HTTPUrl {
        fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
        where
            D: de::Deserializer<'a>,
        {
            use serde::de::Error;
            let helper = String::deserialize(deserializer)?;
            let url = parse_http_url(helper).map_err(D::Error::custom)?;
            Ok(HTTPUrl(url))
        }
    }

    fn parse_http_url<K: AsRef<str>>(str: K) -> Result<reqwest::Url> {
        use std::net::SocketAddr;
        let url = str.as_ref();
        let url = if let Ok(sa) = url.parse::<SocketAddr>() {
            warn!("Url {url} has no scheme, assuming http");
            reqwest::Url::parse(&format!("http://{sa}"))
        } else {
            reqwest::Url::parse(url)
        }?;
        if url.scheme() != "http" && url.scheme() != "https" {
            bail!("url not http or https")
        }
        // TODO: relative url handling just hasn't been implemented and tested
        if url.path() != "/" {
            bail!("url has a relative path (currently unsupported)")
        }
        Ok(url)
    }
}

fn parse_size(val: &str) -> Option<u64> {
    let multiplier = match val.chars().last().map(|v| v.to_ascii_uppercase()) {
        Some('K') => 1024,
        Some('M') => 1024 * 1024,
        Some('G') => 1024 * 1024 * 1024,
        Some('T') => 1024 * 1024 * 1024 * 1024,
        _ => 1,
    };
    let val = if multiplier > 1 && !val.is_empty() {
        val.split_at(val.len() - 1).0
    } else {
        val
    };
    u64::from_str(val).ok().map(|size| size * multiplier)
}

pub struct DeserializeBool {
    value: Option<bool>,
}

impl From<DeserializeBool> for bool {
    fn from(value: DeserializeBool) -> Self {
        value.value.unwrap_or_default()
    }
}

impl From<DeserializeBool> for Option<bool> {
    fn from(value: DeserializeBool) -> Self {
        value.value
    }
}

impl<'de> de::Deserialize<'de> for DeserializeBool {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        Ok(DeserializeBool {
            value: deserialize_bool_option(deserializer)?,
        })
    }
}

// The visit_str bodies call helpers like this one so that their code isn't compiled again for
// every deserializer.
fn parse_bool(v: &str) -> std::result::Result<bool, String> {
    match v.to_lowercase().as_str() {
        "true" | "on" | "1" => Ok(true),
        "false" | "off" | "0" => Ok(false),
        _ => Err(format!(
            "invalid value: '{v}', expected 'true', 'on', '1', 'false', 'off' or '0'"
        )),
    }
}

pub fn deserialize_bool<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: de::Deserializer<'de>,
{
    struct BoolVisitor;

    impl<'de> de::Visitor<'de> for BoolVisitor {
        type Value = bool;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "'true', 'on', '1', 'false', 'off' or '0'")
        }

        fn visit_bool<E>(self, v: bool) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(v)
        }

        fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            parse_bool(v).map_err(E::custom)
        }

        fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(false)
        }
    }

    deserializer.deserialize_any(BoolVisitor)
}

pub fn deserialize_bool_option<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<bool>, D::Error>
where
    D: de::Deserializer<'de>,
{
    struct BoolOptionVisitor;

    impl<'de> de::Visitor<'de> for BoolOptionVisitor {
        type Value = Option<bool>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "'true', 'on', '1', 'false', 'off' or '0'")
        }

        fn visit_bool<E>(self, v: bool) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v))
        }

        fn visit_i8<E>(self, v: i8) -> std::prelude::v1::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v == 1))
        }

        fn visit_f32<E>(self, v: f32) -> std::prelude::v1::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(v == 1.0))
        }

        fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            if v.is_empty() {
                Ok(None)
            } else {
                parse_bool(v).map(Some).map_err(E::custom)
            }
        }

        fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            Ok(Some(deserialize_bool(deserializer)?))
        }

        fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> std::prelude::v1::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(None)
        }
    }

    deserializer.deserialize_any(BoolOptionVisitor)
}

fn basedirs_from_str(dirs: &str, delim: char) -> std::result::Result<Basedirs, String> {
    use crate::util::bytes_to_path;

    if dirs.is_empty() {
        return Ok(Basedirs::default());
    }
    dirs.split(delim)
        .filter_map(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.as_bytes())
            }
        })
        .map(|b| bytes_to_path(b).map(|b| b.into_owned()))
        .try_collect::<_, Vec<_>, _>()
        .map_err(Into::into)
        .and_then(Basedirs::try_from)
        .map_err(|e| format!("{e:#}"))
}

pub fn deserialize_basedirs<'de, D>(deserializer: D) -> std::result::Result<Basedirs, D::Error>
where
    D: de::Deserializer<'de>,
{
    struct BasedirsVisitor(char);

    impl<'de> de::Visitor<'de> for BasedirsVisitor {
        type Value = Basedirs;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "Expected a list of paths, or a string of paths delimited by '{}'",
                self.0
            )
        }

        fn visit_str<E>(self, dirs: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            basedirs_from_str(dirs, self.0).map_err(E::custom)
        }

        fn visit_seq<A>(self, seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            <Vec<PathBuf>>::deserialize(de::value::SeqAccessDeserializer::new(seq)).and_then(
                |basedirs| {
                    Basedirs::try_from(basedirs).map_err(|e| de::Error::custom(format!("{e:#}")))
                },
            )
        }

        fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Basedirs::default())
        }
    }

    #[cfg(target_os = "windows")]
    let delim = ';';
    #[cfg(not(target_os = "windows"))]
    let delim = ':';

    deserializer.deserialize_any(BasedirsVisitor(delim))
}

pub fn serialize_basedirs<S>(
    basedirs: &Basedirs,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: ser::Serializer,
{
    if basedirs.is_none() {
        serializer.serialize_none()
    } else {
        serializer.serialize_some(AsRef::<[PathBuf]>::as_ref(basedirs))
    }
}

pub fn deserialize_size_from_str<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: de::Deserializer<'de>,
{
    struct StringOrU64Visitor;

    impl de::Visitor<'_> for StringOrU64Visitor {
        type Value = u64;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a string with size suffix (like '20G') or a u64")
        }

        fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            parse_size(value).ok_or_else(|| E::custom(format!("Invalid size value: {value}")))
        }

        fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value)
        }

        fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            if value < 0 {
                Err(E::custom("negative values not supported"))
            } else {
                Ok(value as u64)
            }
        }
    }

    deserializer.deserialize_any(StringOrU64Visitor)
}

pub struct DeserializeListOfStrings {
    value: Vec<String>,
}

impl From<DeserializeListOfStrings> for Vec<String> {
    fn from(value: DeserializeListOfStrings) -> Self {
        value.value
    }
}

impl<'de> de::Deserialize<'de> for DeserializeListOfStrings {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        Ok(DeserializeListOfStrings {
            value: deserialize_string_or_list(deserializer)?,
        })
    }
}

pub fn deserialize_string_or_list<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    use serde::de::{self, SeqAccess, Visitor};
    use std::fmt;

    struct StringOrList;

    impl<'de> Visitor<'de> for StringOrList {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("string or list of strings")
        }

        fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value.to_owned()])
        }

        fn visit_seq<A>(self, seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(StringOrList)
}

pub struct DeserializeCommandList {
    value: Vec<String>,
}

impl From<DeserializeCommandList> for Vec<String> {
    fn from(value: DeserializeCommandList) -> Self {
        value.value
    }
}

impl<'de> de::Deserialize<'de> for DeserializeCommandList {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        Ok(DeserializeCommandList {
            value: deserialize_command_or_list(deserializer)?,
        })
    }
}

pub fn deserialize_command_or_list<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    use serde::de::{self, SeqAccess, Visitor};
    use std::fmt;

    struct StringOrList;

    impl<'de> Visitor<'de> for StringOrList {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("string or list of strings")
        }

        fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            if let Some(cmd) = crate::util::split_quoted_shell_str(value) {
                Ok(cmd)
            } else {
                Err(E::custom("Error parsing command"))
            }
        }

        fn visit_seq<A>(self, seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(StringOrList)
}

pub struct DeserializeMapOfStringsToStrings {
    value: HashMap<String, String>,
}

impl From<DeserializeMapOfStringsToStrings> for HashMap<String, String> {
    fn from(value: DeserializeMapOfStringsToStrings) -> Self {
        value.value
    }
}

impl<'de> de::Deserialize<'de> for DeserializeMapOfStringsToStrings {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        Ok(DeserializeMapOfStringsToStrings {
            value: deserialize_string_or_seq_to_map(deserializer)?,
        })
    }
}

fn parse_env_assignments(value: &str) -> Option<HashMap<String, String>> {
    crate::util::split_quoted_shell_str(value).map(|vars| {
        vars.iter()
            .filter_map(|s| s.trim().split_once("="))
            .map(|(key, val)| {
                (
                    key.trim().to_owned(),
                    val.trim()
                        .trim_start_matches('"')
                        .trim_end_matches('"')
                        .to_owned(),
                )
            })
            .collect()
    })
}

pub fn deserialize_string_or_seq_to_map<'de, D>(
    deserializer: D,
) -> std::result::Result<HashMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    use serde::de::{self, MapAccess, SeqAccess, Visitor};
    use std::fmt;

    struct StringOrList;

    impl<'de> Visitor<'de> for StringOrList {
        type Value = HashMap<String, String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("string or list of strings")
        }

        fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            parse_env_assignments(value)
                .ok_or_else(|| E::custom("Error splitting variables string"))
        }

        fn visit_map<A>(self, seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            Deserialize::deserialize(de::value::MapAccessDeserializer::new(seq))
        }

        fn visit_seq<A>(self, seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            <Vec<(String, String)>>::deserialize(de::value::SeqAccessDeserializer::new(seq))
                .map(|vars| vars.into_iter().collect::<Self::Value>())
        }
    }

    deserializer.deserialize_any(StringOrList)
}

pub fn string_from_env_var(env_var_name: &str) -> Option<String> {
    std::env::var(env_var_name).ok().filter(|s| !s.is_empty())
}

pub fn number_from_env_var<A: std::str::FromStr>(env_var_name: &str) -> Option<Result<A>>
where
    <A as FromStr>::Err: fmt::Debug,
{
    let value = string_from_env_var(env_var_name)?;

    value
        .parse::<A>()
        .map_err(|err| anyhow!("{env_var_name} value is invalid: {err:?}"))
        .into()
}

pub fn bool_from_env_var(env_var_name: &str) -> Result<Option<bool>> {
    string_from_env_var(env_var_name)
        .map(|value| match value.to_lowercase().as_str() {
            "true" | "on" | "1" => Ok(true),
            "false" | "off" | "0" => Ok(false),
            _ => bail!("{env_var_name} must be 'true', 'on', '1', 'false', 'off' or '0'."),
        })
        .transpose()
}

// The directories crate changed the location of `config_dir` on macos in version 3,
// so we also check the config in `preference_dir` (new in that version), which
// corresponds to the old location, for compatibility with older setups.
pub fn config_file(env_var: &str, leaf: &str) -> PathBuf {
    if let Some(env_value) = std::env::var_os(env_var) {
        return env_value.into();
    }
    let dirs = directories::ProjectDirs::from("", ORGANIZATION, APP_NAME)
        .expect("Unable to get config directory");
    // If the new location exists, use that.
    let path = dirs.config_dir().join(leaf);
    if path.exists() {
        return path;
    }
    // If the old location exists, use that.
    let path = dirs.preference_dir().join(leaf);
    if path.exists() {
        return path;
    }
    // Otherwise, use the new location.
    dirs.config_dir().join(leaf)
}

// If the file doesn't exist or we can't read it, log the issue and proceed. If the
// config exists but doesn't parse then something is wrong - return an error.
pub fn try_read_config_file<T: de::DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    use fs_err::File;
    use std::io::Read;

    debug!("Attempting to read config file at {path:?}");
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            debug!("Couldn't open config file: {e}");
            return Ok(None);
        }
    };

    let mut string = String::new();
    match file.read_to_string(&mut string) {
        Ok(_) => (),
        Err(e) => {
            warn!("Failed to read config file: {e}");
            return Ok(None);
        }
    }

    let res = if path.extension().is_some_and(|e| e == "json") {
        serde_json::from_str(&string)
            .with_context(|| format!("Failed to load json config file from {}", path.display()))?
    } else {
        toml::from_str(&string)
            .with_context(|| format!("Failed to load toml config file from {}", path.display()))?
    };

    Ok(Some(res))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub(crate) struct _Ignored {}

macro_rules! impl_add_for_config {
    ($klass:ident) => {
        impl std::ops::Add<&$klass> for &$klass {
            type Output = Result<$klass>;

            fn add(self, rhs: &$klass) -> Self::Output {
                <$klass as Loadable<$klass>>::merge(self, rhs)
            }
        }

        impl std::ops::Add<&$klass> for $klass {
            type Output = Result<$klass>;

            fn add(self, rhs: &$klass) -> Self::Output {
                <$klass as Loadable<$klass>>::merge(&self, rhs)
            }
        }

        impl std::ops::Add<$klass> for &$klass {
            type Output = Result<$klass>;

            fn add(self, rhs: $klass) -> Self::Output {
                <$klass as Loadable<$klass>>::merge(self, &rhs)
            }
        }

        impl std::ops::Add<$klass> for $klass {
            type Output = Result<$klass>;

            fn add(self, rhs: $klass) -> Self::Output {
                <$klass as Loadable<$klass>>::merge(&self, &rhs)
            }
        }

        impl std::ops::AddAssign<&$klass> for $klass {
            fn add_assign(&mut self, rhs: &$klass) {
                if let Ok(conf) = <$klass as Loadable<$klass>>::merge(self, rhs) {
                    *self = conf;
                }
            }
        }

        impl std::ops::AddAssign<$klass> for $klass {
            fn add_assign(&mut self, rhs: $klass) {
                if let Ok(conf) = <$klass as Loadable<$klass>>::merge(self, &rhs) {
                    *self = conf;
                }
            }
        }
    };
}

#[cfg(test)]
mod test {
    use super::super::TEN_GIGS;
    use super::*;

    #[test]
    fn test_parse_size() {
        assert_eq!(None, parse_size(""));
        assert_eq!(None, parse_size("bogus value"));
        assert_eq!(Some(100), parse_size("100"));
        assert_eq!(Some(2048), parse_size("2K"));
        assert_eq!(Some(2048), parse_size("2k"));
        assert_eq!(Some(10 * 1024 * 1024), parse_size("10M"));
        assert_eq!(Some(TEN_GIGS), parse_size("10G"));
        assert_eq!(Some(1024 * TEN_GIGS), parse_size("10T"));
    }

    #[test]
    fn test_string_from_env_var() {
        let var_name = "TEST_SCCACHE_VAR";
        for value in [None, Some(""), Some("foo")] {
            match value {
                None => unsafe {
                    std::env::remove_var(var_name);
                },
                Some(value) => unsafe {
                    std::env::set_var(var_name, value);
                },
            }
            let result = string_from_env_var(var_name);
            unsafe {
                std::env::remove_var(var_name);
            }

            let expected = match value {
                None | Some("") => None,
                Some(value) => Some(value.to_string()),
            };

            assert_eq!(result, expected);
        }
    }
}
