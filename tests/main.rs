//! The integration tests, built as one binary. Each module tests one area of
//! sccache, using the servers, compilers and stats helpers from `harness`.
//!
//! The `preprocessor_cache` tests take any `SccacheClient`, and `dist` runs
//! them with a dist client too, so both kinds of client are checked the same
//! way.

#![deny(rust_2018_idioms)]

#[macro_use]
extern crate log;

#[macro_use]
mod harness;

mod c;
#[cfg(not(any(target_os = "macos", target_os = "freebsd")))]
mod cuda;
mod deps;
#[cfg(feature = "dist-tests")]
mod dist;
mod hip;
#[cfg(feature = "dist-tests")]
mod oauth;
mod rust;
mod server;
