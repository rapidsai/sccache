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

//! Compiling C and C++ with a local server: cache hits and misses, and the
//! compiler flags sccache has to handle specially.

use crate::harness::{
    client::{SccacheClient, with_local_server},
    compilers::{Compiler, adv_key_kind, compile_cmdline},
    copy_to_tempdir, write_source,
};
use assert_cmd::prelude::*;
use fs::File;
use fs_err as fs;
use pastey::paste;
use predicates::prelude::*;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::Path;

//TODO: could test gcc when targeting mingw.

pub const INPUT: &str = "test.c";
const INPUT_CLANG_MULTICALL: &str = "test_clang_multicall.c";
const INPUT_WITH_WHITESPACE: &str = "test_whitespace.c";
const INPUT_WITH_WHITESPACE_ALT: &str = "test_whitespace_alt.c";
const INPUT_ERR: &str = "test_err.c";
const INPUT_MACRO_EXPANSION: &str = "test_macro_expansion.c";
const INPUT_WITH_DEFINE: &str = "test_with_define.c";
pub const OUTPUT: &str = "test.o";
#[cfg(unix)]
const NULL_PATH: &str = "/dev/null";
#[cfg(windows)]
const NULL_PATH: &str = "NUL";
#[cfg(unix)]
const DEV_STDOUT: &str = "/dev/stdout";

fn test_basic_compile(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    println!("test_basic_compile: {name}");
    // Compile a source file.
    copy_to_tempdir(&[INPUT, INPUT_ERR], tempdir);

    let out_file = tempdir.join(OUTPUT);
    let adv_key = adv_key_kind(compiler.lang().as_str(), compiler.name);

    trace!("compile");
    client
        .cmd()
        .args(compile_cmdline(name, exe, INPUT, OUTPUT, []))
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
    assert_eq!(&1, stats.cache_misses.get("C/C++").unwrap());
    assert_eq!(&1, stats.cache_misses.get_adv(&adv_key).unwrap());
    trace!("compile");
    fs::remove_file(&out_file).unwrap();
    client
        .cmd()
        .args(compile_cmdline(name, exe, INPUT, OUTPUT, []))
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
    assert_eq!(&1, stats.cache_hits.get("C/C++").unwrap());
    assert_eq!(&1, stats.cache_misses.get("C/C++").unwrap());
    assert_eq!(&1, stats.cache_hits.get_adv(&adv_key).unwrap());
    assert_eq!(&1, stats.cache_misses.get_adv(&adv_key).unwrap());
}

/// Writes a cache hit into a special file like /dev/null, next to which no
/// temporary file can be created. `test_basic_compile` cached the object.
fn test_basic_compile_into(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
    output: &str,
) {
    let Compiler {
        name,
        exe,
        env_vars,
    } = compiler;
    println!("test_basic_compile_into: {name} -o {output}");
    client.zero_stats();
    // Compile a source file.
    copy_to_tempdir(&[INPUT, INPUT_ERR], tempdir);

    let adv_key = adv_key_kind(compiler.lang().as_str(), compiler.name);

    trace!("compile");
    client
        .cmd()
        .args(compile_cmdline(name, exe, INPUT, output, Vec::new()))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    trace!("request stats");
    let stats = client.stats().unwrap();
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(1, stats.cache_hits.all());
    assert_eq!(0, stats.cache_misses.all());
    assert!(stats.cache_misses.get("C/C++").is_none());
    assert!(stats.cache_misses.get_adv(&adv_key).is_none());
    trace!("compile");
    client
        .cmd()
        .args(compile_cmdline(name, exe, INPUT, output, Vec::new()))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    trace!("request stats");
    let stats = client.stats().unwrap();
    assert_eq!(2, stats.compile_requests);
    assert_eq!(2, stats.requests_executed);
    assert_eq!(2, stats.cache_hits.all());
    assert_eq!(0, stats.cache_misses.all());
    assert_eq!(&2, stats.cache_hits.get("C/C++").unwrap());
    assert!(stats.cache_misses.get("C/C++").is_none());
    assert_eq!(&2, stats.cache_hits.get_adv(&adv_key).unwrap());
    assert!(stats.cache_misses.get_adv(&adv_key).is_none());
}

fn test_cache_preprocessor_invocation(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;

    println!("test_cache_preprocessor_invocation: {name}");
    copy_to_tempdir(&[INPUT], tempdir);

    trace!("compile");

    client.zero_stats();

    client
        .cmd()
        .arg(exe)
        .arg("-E")
        .arg(INPUT)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    trace!("request stats");
    let stats = client.stats().unwrap();
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(0, stats.not_cached.len());
    assert_eq!(None, stats.not_cached.get("-E"));
}

fn test_msvc_deps(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    // Check that -deps works.
    trace!("compile with /sourceDependencies");
    let mut args = compile_cmdline(name, exe, INPUT, OUTPUT, []);
    args.push("/sourceDependenciestest.o.json".into());
    client
        .cmd()
        .args(&args)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    // Check the contents
    let f = File::open(tempdir.join("test.o.json")).expect("Failed to open dep file");
    // MSVC deps files are JSON, which we can validate properties of, but will be
    // subtly different on different systems (Windows SDK version, for example)
    let deps: serde_json::Value = serde_json::from_reader(f).expect("Failed to read dep file");
    let source = deps["Data"]["Source"].as_str().expect("No source found");
    let source = Path::new(source).file_name().expect("No source file name");
    assert_eq!(source, INPUT);

    let includes = deps["Data"]["Includes"]
        .as_array()
        .expect("No includes found");
    assert_ne!(includes.len(), 0);
}

fn test_msvc_responsefile(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) {
    let Compiler {
        name: _,
        exe,
        env_vars,
        ..
    } = compiler;

    let out_file = tempdir.join(OUTPUT);
    let cmd_file_name = "test_msvc.rsp";
    {
        let mut file = File::create(tempdir.join(cmd_file_name)).unwrap();
        let content = format!("-c {INPUT} -Fo{OUTPUT}");
        file.write_all(content.as_bytes()).unwrap();
    }

    let args = vec_from!(OsString, exe, &format!("@{cmd_file_name}"));
    client
        .cmd()
        .args(&args)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();

    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    fs::remove_file(&out_file).unwrap();
}

fn test_gcc_mp_werror(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    trace!("test -MP with -Werror");
    let mut args = compile_cmdline(name, exe, INPUT_ERR, OUTPUT, []);
    args.extend(vec_from!(
        OsString, "-MD", "-MP", "-MF", "foo.pp", "-Werror"
    ));
    // This should fail, but the error should be from the #error!
    client
        .cmd()
        .args(&args)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .failure()
        .stderr(
            predicates::str::contains("to generate dependencies you must specify either -M or -MM")
                .from_utf8()
                .not(),
        );
}

fn test_gcc_fprofile_generate_source_changes(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    trace!("test -fprofile-generate with different source inputs");
    client.zero_stats();
    const SRC: &str = "source.c";
    write_source(
        tempdir,
        SRC,
        "/*line 1*/
#ifndef UNDEFINED
/*unused line 1*/
#endif

int main(int argc, char** argv) {
  return 0;
}
",
    );
    let mut args = compile_cmdline(name, exe, SRC, OUTPUT, []);
    args.extend(vec_from!(OsString, "-fprofile-generate"));
    trace!("compile source.c (1)");
    client
        .cmd()
        .args(&args)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    let stats = client.stats().unwrap();
    assert_eq!(0, stats.cache_hits.all());
    assert_eq!(1, stats.cache_misses.all());
    assert_eq!(&1, stats.cache_misses.get("C/C++").unwrap());
    // Compile the same source again to ensure we can get a cache hit.
    trace!("compile source.c (2)");
    client
        .cmd()
        .args(&args)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    let stats = client.stats().unwrap();
    assert_eq!(1, stats.cache_hits.all());
    assert_eq!(1, stats.cache_misses.all());
    assert_eq!(&1, stats.cache_hits.get("C/C++").unwrap());
    assert_eq!(&1, stats.cache_misses.get("C/C++").unwrap());
    // Now write out a slightly different source file that will preprocess to the same thing,
    // modulo line numbers. This should not be a cache hit because line numbers are important
    // with -fprofile-generate.
    write_source(
        tempdir,
        SRC,
        "/*line 1*/
#ifndef UNDEFINED
/*unused line 1*/
/*unused line 2*/
#endif

int main(int argc, char** argv) {
  return 0;
}
",
    );
    trace!("compile source.c (3)");
    client
        .cmd()
        .args(&args)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    let stats = client.stats().unwrap();
    assert_eq!(1, stats.cache_hits.all());
    assert_eq!(2, stats.cache_misses.all());
    assert_eq!(&1, stats.cache_hits.get("C/C++").unwrap());
    assert_eq!(&2, stats.cache_misses.get("C/C++").unwrap());
}

/* test case like this:
    echo "int test(){}" > test.cc
    mkdir o1 o2
    sccache g++ -c -g -gsplit-dwarf test.cc -o test1.o
    sccache g++ -c -g -gsplit-dwarf test.cc -o test1.o   --- > cache hit
    sccache g++ -c -g -gsplit-dwarf test.cc -o test2.o   --- > cache miss
    strings test2.o |grep test2.dwo
*/
fn test_split_dwarf_object_generate_output_dir_changes(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    trace!("test -g -gsplit-dwarf with different output");
    client.zero_stats();
    const SRC: &str = "source.c";
    write_source(tempdir, SRC, "int test(){}");
    let mut args = compile_cmdline(name, exe.clone(), SRC, "test1.o", []);
    args.extend(vec_from!(OsString, "-g"));
    args.extend(vec_from!(OsString, "-gsplit-dwarf"));
    trace!("compile source.c (1)");
    client
        .cmd()
        .args(&args)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    let stats = client.stats().unwrap();
    assert_eq!(0, stats.cache_hits.all());
    assert_eq!(1, stats.cache_misses.all());
    assert_eq!(&1, stats.cache_misses.get("C/C++").unwrap());
    // Compile the same source again to ensure we can get a cache hit.
    trace!("compile source.c (2)");
    client
        .cmd()
        .args(&args)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    let stats = client.stats().unwrap();
    assert_eq!(1, stats.cache_hits.all());
    assert_eq!(1, stats.cache_misses.all());
    assert_eq!(&1, stats.cache_hits.get("C/C++").unwrap());
    assert_eq!(&1, stats.cache_misses.get("C/C++").unwrap());
    // Compile the same source again with different output
    // to ensure we can force generate new object file.
    let mut args2 = compile_cmdline(name, exe, SRC, "test2.o", []);
    args2.extend(vec_from!(OsString, "-g"));
    args2.extend(vec_from!(OsString, "-gsplit-dwarf"));
    trace!("compile source.c (2)");
    client
        .cmd()
        .args(&args2)
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    let stats = client.stats().unwrap();
    assert_eq!(1, stats.cache_hits.all());
    assert_eq!(2, stats.cache_misses.all());
    assert_eq!(&1, stats.cache_hits.get("C/C++").unwrap());
    assert_eq!(&2, stats.cache_misses.get("C/C++").unwrap());
}

fn test_gcc_clang_no_warnings_from_macro_expansion(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    println!("test_gcc_clang_no_warnings_from_macro_expansion: {name}");
    // Compile a source file.
    copy_to_tempdir(&[INPUT_MACRO_EXPANSION], tempdir);

    trace!("compile");
    client
        .cmd()
        .args(
            [
                &compile_cmdline(name, exe, INPUT_MACRO_EXPANSION, OUTPUT, [])[..],
                &vec_from!(OsString, "-Wunreachable-code")[..],
            ]
            .concat(),
        )
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success()
        .stderr(predicates::str::contains("warning:").from_utf8().not());
}

fn test_compile_with_define(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    println!("test_compile_with_define: {name}");
    // Compile a source file.
    copy_to_tempdir(&[INPUT_WITH_DEFINE], tempdir);

    trace!("compile");
    client
        .cmd()
        .args(
            [
                &compile_cmdline(name, exe, INPUT_WITH_DEFINE, OUTPUT, [])[..],
                &vec_from!(OsString, "-DSCCACHE_TEST_DEFINE")[..],
            ]
            .concat(),
        )
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success()
        .stderr(predicates::str::contains("warning:").from_utf8().not());
}

fn test_gcc_clang_depfile(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    println!("test_gcc_clang_depfile: {name}");
    copy_to_tempdir(&[INPUT], tempdir);
    fs::copy(tempdir.join(INPUT), tempdir.join("same-content.c")).unwrap();

    trace!("compile");
    client
        .cmd()
        .args(compile_cmdline(name, exe.clone(), INPUT, OUTPUT, []))
        .args(vec_from!(OsString, "-MD", "-MF", "first.d"))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            "same-content.c",
            "same-content.o",
            [],
        ))
        .args(vec_from!(OsString, "-MD", "-MF", "second.d"))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    let mut first = String::new();
    let mut second = String::new();
    File::open(tempdir.join("first.d"))
        .unwrap()
        .read_to_string(&mut first)
        .unwrap();
    File::open(tempdir.join("second.d"))
        .unwrap()
        .read_to_string(&mut second)
        .unwrap();
    assert_ne!(first, second);
}

fn test_clang_multicall(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    println!("test_clang_multicall: {name}");
    // Compile a source file.
    copy_to_tempdir(&[INPUT_CLANG_MULTICALL], tempdir);

    println!("compile clang_multicall");
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            INPUT_CLANG_MULTICALL,
            OUTPUT,
            [],
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
}

fn test_clang_cache_whitespace_normalization(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
    hit: bool,
) {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    println!("test_clang_cache_whitespace_normalization: {name}");
    debug!("expecting hit: {hit}");
    // Compile a source file.
    copy_to_tempdir(&[INPUT_WITH_WHITESPACE, INPUT_WITH_WHITESPACE_ALT], tempdir);
    client.zero_stats();

    debug!("compile whitespace");
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            INPUT_WITH_WHITESPACE,
            OUTPUT,
            [],
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    debug!("request stats");
    let stats = client.stats().unwrap();
    assert_eq!(1, stats.compile_requests);
    assert_eq!(1, stats.requests_executed);
    assert_eq!(0, stats.cache_hits.all());
    assert_eq!(1, stats.cache_misses.all());

    debug!("compile whitespace_alt");
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            INPUT_WITH_WHITESPACE_ALT,
            OUTPUT,
            [],
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    debug!("request stats (expecting cache hit)");
    if hit {
        let stats = client.stats().unwrap();
        assert_eq!(2, stats.compile_requests);
        assert_eq!(2, stats.requests_executed);
        assert_eq!(1, stats.cache_hits.all());
        assert_eq!(1, stats.cache_misses.all());
    } else {
        let stats = client.stats().unwrap();
        assert_eq!(2, stats.compile_requests);
        assert_eq!(2, stats.requests_executed);
        assert_eq!(0, stats.cache_hits.all());
        assert_eq!(2, stats.cache_misses.all());
    }
}

pub fn run_sccache_command_tests(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) {
    test_basic_compile(client, compiler, tempdir);
    test_basic_compile_into(client, compiler, tempdir, NULL_PATH);
    #[cfg(unix)]
    test_basic_compile_into(client, compiler, tempdir, DEV_STDOUT);
    test_compile_with_define(client, compiler, tempdir);
    if matches!(compiler.name, "cl") {
        test_msvc_deps(client, compiler, tempdir);
        test_msvc_responsefile(client, compiler, tempdir);
    }
    if matches!(compiler.name, "gcc" | "nvc" | "nvc++") {
        test_gcc_mp_werror(client, compiler, tempdir);
    }
    if matches!(compiler.name, "gcc") {
        test_gcc_fprofile_generate_source_changes(client, compiler, tempdir);
    }
    if matches!(compiler.name, "clang" | "clang++" | "gcc" | "nvc" | "nvc++") {
        test_gcc_clang_depfile(client, compiler, tempdir);
    }
    if matches!(compiler.name, "clang" | "clang++" | "gcc") {
        test_gcc_clang_no_warnings_from_macro_expansion(client, compiler, tempdir);
        test_split_dwarf_object_generate_output_dir_changes(client, compiler, tempdir);
    }
    if matches!(compiler.name, "clang++") {
        test_clang_multicall(client, compiler, tempdir);
    }

    test_cache_preprocessor_invocation(client, compiler, tempdir);
    test_clang_cache_whitespace_normalization(client, compiler, tempdir, false);
}

/// The assembler is what turns the compiler's output into the object file, so a
/// different assembler reporting the same version still has to be a cache miss.
/// Preprocessor cache mode is the interesting half of this: a hit there hands
/// back the object cache key stored with the entry, so the assembler has to be
/// part of that key too.
#[test]
#[cfg(all(unix, not(target_os = "macos")))]
fn test_assembler_affects_cache() -> std::result::Result<(), anyhow::Error> {
    use crate::harness::{client::make_sccache_client, compilers::find_compiler, init_logger};
    use fs_err::os::unix::fs::OpenOptionsExt;
    use std::env;
    use std::process::Command;
    use which::which_in;

    init_logger();

    // GCC always assembles by spawning `as`, so it needs no flag to opt into
    // this, unlike clang.
    let compiler = match find_compiler("gcc") {
        Some(compiler) => compiler,
        None => {
            warn!("No gcc found, skipping test");
            return Ok(());
        }
    };

    let (tempdir, client) = make_sccache_client();
    let tempdir_path = tempdir.path();

    let assembler = Command::new(&compiler.exe)
        .arg("-print-prog-name=as")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .and_then(|name| which_in(name, env::var_os("PATH"), env::current_dir().unwrap()).ok());

    let assembler = match assembler {
        Some(assembler) => assembler,
        None => {
            warn!("No assembler found, skipping test");
            return Ok(());
        }
    };

    // Something that assembles identically and reports the same version, but
    // isn't the same file, the way a distro rebuild wouldn't be.
    let bindir = tempdir_path.join("bin");
    fs::create_dir_all(&bindir).unwrap();

    let mut wrapper = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o777)
        .open(bindir.join("as"))
        .unwrap();
    writeln!(wrapper, "#!/bin/sh\nexec {} \"$@\"", assembler.display()).unwrap();
    drop(wrapper);

    let path = env::var_os("PATH").unwrap();
    let mut wrapper_first = OsString::from(&bindir);
    wrapper_first.push(":");
    wrapper_first.push(&path);

    copy_to_tempdir(&[INPUT], tempdir_path);

    // The compiler is only looked at once per server, so anything that wants a
    // fresh look at the assembler needs a fresh server.
    let compile = |client: SccacheClient, tempdir: &Path, path: &OsString, expected_hits: u64| {
        fs::remove_file(tempdir.join(OUTPUT)).ok();

        client
            .cmd()
            .args(compile_cmdline(
                compiler.name,
                &compiler.exe,
                INPUT,
                OUTPUT,
                Vec::new(),
            ))
            .current_dir(tempdir)
            .env("PATH", path)
            .assert()
            .success();

        let stats = client.stats()?;
        assert_eq!(expected_hits, stats.cache_hits.all());
        assert_eq!(1 - expected_hits, stats.cache_misses.all());

        client.stop();

        Ok::<SccacheClient, anyhow::Error>(client)
    };

    let client = compile(client.start(), tempdir_path, &path, 0)?;
    let client = compile(client.start(), tempdir_path, &wrapper_first, 0)?;
    // Back to the assembler the cache was populated with.
    let client = compile(client.start(), tempdir_path, &path, 1)?;
    // stop_local_daemon();

    let _ = client;

    Ok(())
}

macro_rules! test_sccache_command {
    ($compiler:ident, $compiler_name:expr) => {
        paste! {
            #[test]
            #[cfg(any(unix, target_env = "msvc"))]
            fn [<test_sccache_command_ $compiler>] () {
                with_local_server($compiler_name, run_sccache_command_tests);
            }
        }
    };
}

#[cfg(unix)]
test_sccache_command!(gcc, "gcc");
#[cfg(unix)]
test_sccache_command!(clang, "clang");
#[cfg(unix)]
test_sccache_command!(clangxx, "clang++");
#[cfg(unix)]
test_sccache_command!(nvc, "nvc");
#[cfg(unix)]
test_sccache_command!(nvcxx, "nvc++");

#[cfg(target_env = "msvc")]
test_sccache_command!(msvc, "cl");

// Test that invoking (e.g.) a symlink called "gcc" to "sccache" results in sccache invoking gcc
#[test]
#[cfg(unix)]
fn test_symlinked_exe() {
    use crate::harness::{client::make_sccache_client, compilers::find_compilers};
    use std::os::unix::fs as unix_fs;
    use std::process::Command;

    let compilers = find_compilers();
    if compilers.is_empty() {
        warn!("No compilers found, skipping test");
        return;
    }

    let (tempdir, client) = make_sccache_client();
    let tempdir_path = tempdir.path();

    let get_version = |compiler: &mut Command| {
        compiler
            .arg("--version")
            .current_dir(tempdir_path)
            .output()
            .unwrap()
            .stdout
    };

    let mut cmd = client.cmd();
    let sccache_path = cmd.get_program().to_os_string();
    let sccache_version = get_version(&mut cmd);

    for compiler in compilers {
        trace!(
            "Testing sccache symlink to {} ({:?})",
            compiler.name, compiler.exe
        );
        let compiler_version =
            get_version(Command::new(&compiler.exe).envs(compiler.env_vars.clone()));
        assert_ne!(compiler_version, sccache_version);

        let exe_filename = Path::new(&compiler.exe).file_name().unwrap();
        let sccache_compiler_alias = tempdir_path.join(exe_filename);
        unix_fs::symlink(&sccache_path, &sccache_compiler_alias).unwrap();

        let compiler_alias_version = get_version(
            client
                .command(&sccache_compiler_alias)
                .envs(compiler.env_vars),
        );
        assert_eq!(compiler_version, compiler_alias_version);

        fs::remove_file(&sccache_compiler_alias).unwrap();
    }
}
