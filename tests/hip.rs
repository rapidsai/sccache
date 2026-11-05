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

//! Compiling HIP with clang.

use crate::harness::{
    client::{SccacheClient, make_sccache_client},
    compilers::{Compiler, adv_key_kind, compile_hip_cmdline, find_hip_compiler},
    copy_to_tempdir, init_logger,
};
use assert_cmd::prelude::*;
use fs_err as fs;
use std::path::Path;

use crate::c::OUTPUT;

const INPUT_FOR_HIP_A: &str = "test_a.hip";
const INPUT_FOR_HIP_B: &str = "test_b.hip";

fn test_hip_compiles(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
    target_arches: &[String],
) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    println!("test_hip_compiles: {name} {target_arches:?}");
    // Compile multiple source files.
    copy_to_tempdir(&[INPUT_FOR_HIP_A, INPUT_FOR_HIP_B], tempdir);

    let out_file = tempdir.join(OUTPUT);
    trace!("compile A");
    client
        .cmd()
        .args(compile_hip_cmdline(
            name,
            exe,
            INPUT_FOR_HIP_A,
            OUTPUT,
            target_arches,
            [],
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    trace!("request stats");
    let stats = client.stats().unwrap();
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(0, stats.cache_hits.all());
    assert_eq!(1, stats.cache_misses.all());
    assert_eq!(&1, stats.cache_misses.get("HIP").unwrap());
    let adv_hip_key = adv_key_kind("hip", compiler.name);
    assert_eq!(&1, stats.cache_misses.get_adv(&adv_hip_key).unwrap());

    trace!("compile A again");
    fs::remove_file(&out_file).unwrap();
    client
        .cmd()
        .args(compile_hip_cmdline(
            name,
            exe,
            INPUT_FOR_HIP_A,
            OUTPUT,
            target_arches,
            [],
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    trace!("request stats");
    let stats = client.stats().unwrap();
    assert_eq!(2, stats.compile_requests);
    assert_eq!(2, stats.requests_executed);
    assert_eq!(1, stats.cache_hits.all());
    assert_eq!(1, stats.cache_misses.all());
    assert_eq!(&1, stats.cache_hits.get("HIP").unwrap());
    assert_eq!(&1, stats.cache_misses.get("HIP").unwrap());
    let adv_hip_key = adv_key_kind("hip", compiler.name);
    assert_eq!(&1, stats.cache_hits.get_adv(&adv_hip_key).unwrap());
    assert_eq!(&1, stats.cache_misses.get_adv(&adv_hip_key).unwrap());

    // By compiling another input source we verify that the pre-processor
    // phase is correctly running and outputting text
    trace!("compile B");
    client
        .cmd()
        .args(compile_hip_cmdline(
            name,
            exe,
            INPUT_FOR_HIP_B,
            OUTPUT,
            target_arches,
            [],
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    trace!("request stats");
    let stats = client.stats().unwrap();
    assert_eq!(3, stats.compile_requests);
    assert_eq!(3, stats.requests_executed);
    assert_eq!(1, stats.cache_hits.all());
    assert_eq!(2, stats.cache_misses.all());
    assert_eq!(&1, stats.cache_hits.get("HIP").unwrap());
    assert_eq!(&2, stats.cache_misses.get("HIP").unwrap());
    let adv_hip_key = adv_key_kind("hip", compiler.name);
    assert_eq!(&1, stats.cache_hits.get_adv(&adv_hip_key).unwrap());
    assert_eq!(&2, stats.cache_misses.get_adv(&adv_hip_key).unwrap());
}

fn run_sccache_hip_command_tests(client: &SccacheClient, compiler: Compiler, tempdir: &Path) {
    client.zero_stats();
    test_hip_compiles(client, &compiler, tempdir, &["gfx900".into()]);
    client.zero_stats();
    test_hip_compiles(
        client,
        &compiler,
        tempdir,
        &["gfx900".into(), "gfx1030".into()],
    );
    // test_proper_lang_stat_tracking(compiler, tempdir);
}

#[test]
#[cfg(any(unix, target_env = "msvc"))]
fn test_hip_sccache_command() {
    init_logger();

    if let Some(compiler) = find_hip_compiler() {
        let (tempdir, client) = make_sccache_client();
        run_sccache_hip_command_tests(&client.start(), compiler, tempdir.path());
    }
}
