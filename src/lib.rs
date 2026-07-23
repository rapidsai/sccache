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

#![deny(rust_2018_idioms)]
#![allow(
    clippy::type_complexity,
    clippy::new_without_default,
    clippy::blocks_in_conditions
)]
#![recursion_limit = "256"]

#[macro_use]
extern crate log;
// To get macros in scope, this has to be first.
#[cfg(test)]
#[macro_use]
mod test;

#[macro_use]
pub mod errors;

pub mod cache;
mod client;
mod cmdline;
mod commands;
pub mod compiler;
pub mod config;
pub mod dist;
mod jobserver;
pub mod lru_disk_cache;
pub mod mock_command;
mod net;
mod protocol;
pub mod server;
#[doc(hidden)]
#[macro_use]
pub mod util;

use std::env;

/// VERSION is the pkg version of sccache.
///
/// This version is safe to be used in cache services to indicate the version
/// that sccache ie.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Used to denote the environment variable that controls
/// logging for sccache, and sccache-dist.
pub const LOGGING_ENV: &str = "SCCACHE_LOG";
pub const SERVER_LOG_ENV: &str = "SCCACHE_SERVER_LOG";

pub fn main() {
    // We only care if it's `1`
    init_logging(env::var("SCCACHE_START_SERVER").is_ok_and(|x| x == "1"));

    let command = match cmdline::try_parse() {
        Ok(cmd) => cmd,
        Err(e) => match e.downcast::<clap::error::Error>() {
            // If the error is from clap then let them handle formatting and exiting
            Ok(clap_err) => clap_err.exit(),
            Err(some_other_err) => {
                println!("sccache: {some_other_err}");
                for source in some_other_err.chain().skip(1) {
                    println!("sccache: caused by: {source}");
                }
                std::process::exit(1);
            }
        },
    };

    std::process::exit(match commands::run_command(command) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("sccache: error: {e}");
            for e in e.chain().skip(1) {
                eprintln!("sccache: caused by: {e}");
            }
            2
        }
    });
}

fn init_logging(is_server: bool) {
    //
    // If starting the server, first check if SCCACHE_SERVER_LOG is set
    //
    // This allows users to configure server logging independently from
    // client logs.
    //
    // A user doesn't necessarily want to see each sccache client's logs
    // in their `make -j` stderr output, but does want to see the server
    // logs redirected to $SCCACHE_ERROR_LOG.
    //
    [is_server.then_some(SERVER_LOG_ENV), Some(LOGGING_ENV)]
        .into_iter()
        .filter_map(|var| var.filter(|var| env::var(var).is_ok()))
        .map(|env| {
            let mut builder = env_logger::Builder::from_env(env);
            // Enable millisecond precision timestamps if SCCACHE_LOG_MILLIS is set
            if env::var("SCCACHE_LOG_MILLIS").is_ok() {
                builder.format_timestamp_millis();
            }
            builder.try_init()
        })
        .take_while(|res| res.is_err())
        .for_each(drop);
}

pub fn compiler_version(exe: &std::path::Path) -> crate::errors::Result<String> {
    use crate::compiler::get_compiler_info;
    use crate::jobserver::Client;
    use crate::mock_command::{CommandCreatorSync, ProcessCommandCreator};
    use tokio::runtime::Runtime;

    let cwd = env::current_dir()?;
    let runtime = Runtime::new()?;
    let jobserver = Client::new_num(1);
    let creator = ProcessCommandCreator::new(&jobserver);
    let handle = runtime.handle().clone();
    runtime.block_on(async move {
        get_compiler_info(creator, exe, &cwd, &[], &[], &handle, None)
            .await?
            .0
            .version()
            .ok_or(crate::errors::anyhow!(
                "Could not determine compiler version (exe={exe:?})"
            ))
    })
}
