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

//! The deps cache: dependencies cache entries have to notice when a
//! header changes, and when `__TIME__`, `__DATE__` or `__TIMESTAMP__` make an
//! object depend on when or where it was compiled.
//!
//! These take any client, so `crate::dist` also checks that a dist client gets
//! the same cache hits and misses.

use crate::harness::{
    client::{SccacheClient, with_local_server},
    compilers::{Compiler, compile_cmdline},
    stats::{AdditionalStats, ExpectedStats},
    write_source,
};
use assert_cmd::prelude::*;
use itertools::Itertools;
use pastey::paste;
use sccache::{compiler::SourceFileDeps, errors::*};
use std::{ffi::OsString, path::Path};

/// A dist client has a second dependencies cache entry for each local one,
/// listing the files a build server needs, with a result for each object.
fn assert_num_deps_cache_entries_and_results<
    I: IntoIterator<Item = (OsString, SourceFileDeps)> + Clone,
>(
    dist: bool,
    entries: I,
    exp_num_entries: usize,
    exp_num_results: &[usize],
) {
    let (exp_num_entries, exp_num_results) = if dist {
        (
            exp_num_entries * 2,
            exp_num_results.iter().flat_map(|n| [*n, *n]).collect(),
        )
    } else {
        (exp_num_entries, exp_num_results.to_vec())
    };
    // Assert the number of deps cache entries matches the expected number of entries
    assert_eq!(exp_num_entries, entries.clone().into_iter().count());
    // Assert the deps cache entry now has the expected number of results
    assert_eq!(
        exp_num_results,
        entries
            .into_iter()
            .map(|(_, e)| e.iter().count())
            .sorted()
            .collect::<Vec<_>>()
    );
}

/// A dist client's dependencies cache entries are only read when it
/// compiles, so a cache hit doesn't change their LRU order.
fn assert_deps_cache_entries_result_orders_changed<
    I: IntoIterator<Item = (OsString, SourceFileDeps)> + Clone,
>(
    dist: bool,
    old: I,
    new: I,
) {
    // Ensure same number of deps cache entries
    assert_eq!(
        new.clone().into_iter().count(),
        old.clone().into_iter().count()
    );

    let old = old
        .into_iter()
        .sorted_by(|a, b| a.0.cmp(&b.0))
        .collect::<Vec<_>>();
    let new = new
        .into_iter()
        .sorted_by(|a, b| a.0.cmp(&b.0))
        .collect::<Vec<_>>();

    let mut unchanged = 0;

    for ((old_key, old), (new_key, new)) in old.iter().zip(new.iter()) {
        // Assert deps cache entry keys match
        assert_eq!(old_key, new_key);

        if dist && old.iter().eq(new.iter()) {
            unchanged += 1;
            continue;
        }

        let old_back = old.iter().next_back();
        let new_back = new.iter().next_back();

        // Assert the new deps cache entry result is at the back of the LRU
        assert_ne!((old_key, old_back), (new_key, new_back));

        let new_back = new_back.unwrap();
        // Assert all but the last deps cache entry results are the same
        assert_eq!(
            old.iter().filter(|x| x.0 != new_back.0).collect::<Vec<_>>(),
            new.iter().filter(|x| x.0 != new_back.0).collect::<Vec<_>>()
        );
    }

    if dist {
        // Assert only the local entries changed
        assert_eq!(old.len(), unchanged * 2);
    }
}

fn assert_deps_cache_entries_are_identical<
    I: IntoIterator<Item = (OsString, SourceFileDeps)> + Clone,
>(
    old: I,
    new: I,
) {
    // Ensure same number of deps cache entries
    assert_eq!(
        new.clone().into_iter().count(),
        old.clone().into_iter().count()
    );
    let old = old.into_iter().sorted_by(|a, b| a.0.cmp(&b.0));
    let new = new.into_iter().sorted_by(|a, b| a.0.cmp(&b.0));
    for ((old_key, old), (new_key, new)) in old.zip(new) {
        // Assert deps cache entry keys match
        assert_eq!(old_key, new_key);
        // Assert the deps cache entry results are the same
        assert_eq!(
            old.iter().collect::<Vec<_>>(),
            new.iter().collect::<Vec<_>>()
        );
    }
}

pub fn test_dependency_changes(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
) -> Result<()> {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;

    println!("test_dependency_changes: {name}");

    let src = "pp_test.c";
    let obj = "pp_test.c.o";
    let dep_a = "pp_test_a.h";
    let dep_b = "pp_test_b.h";

    let kind = compiler.kind();
    let lang = compiler.lang();

    client.zero_stats();
    client.clear_disk_cache()?;

    let mut stats = ExpectedStats::new(client);

    let mut compile_and_verify = |additional_stats: AdditionalStats| -> Result<()> {
        let mut cmd = client.cmd();
        cmd.current_dir(tempdir)
            .envs(env_vars.clone())
            .args(compile_cmdline(name, exe, src, obj, []))
            .assert()
            .success();
        stats.assert_added(additional_stats);
        Ok(())
    };

    {
        // Write source file
        write_source(
            tempdir,
            src,
            &format!(
                r#"
#include <stdio.h>
#include "{dep_a}"
int main(int argc, char** argv) {{
    return a();
}}
"#
            ),
        );

        // Write dependency A
        write_source(tempdir, dep_a, r#"int a() { return 0; }"#);

        // Compile and populate deps and objs cache entries
        compile_and_verify(AdditionalStats::miss(kind, lang))?;

        // Compile again to ensure deps cache hit
        compile_and_verify(AdditionalStats::hit(kind, lang))?;
    }

    // Change dep_a and recompile
    {
        write_source(
            tempdir,
            dep_a,
            &format!(
                r#"
#include "{dep_b}"
int a() {{ return b(); }}
"#
            ),
        );

        // Write dependency B
        write_source(tempdir, dep_b, r#"int b() { return 1; }"#);

        // Should be two object hashes now
        // A dist client still has the dependencies it found for the unchanged
        // source file, and they all still exist
        compile_and_verify(AdditionalStats::miss(kind, lang).reusing_dist_deps())?;

        // Compile again to ensure deps cache hit
        // Should still be two object hashes
        compile_and_verify(AdditionalStats::hit(kind, lang))?;
    }

    // Change dep_b and recompile
    {
        write_source(tempdir, dep_b, r#"int b() { return 2; }"#);

        // Should be three object hashes now
        compile_and_verify(AdditionalStats::miss(kind, lang).reusing_dist_deps())?;

        // Compile again to ensure deps cache hit
        // Should still be three object hashes
        compile_and_verify(AdditionalStats::hit(kind, lang))?;
    }

    // Revert dep_b to ensure deps cache hit
    {
        // Write dependency B
        write_source(tempdir, dep_b, r#"int b() { return 1; }"#);

        // Should still be three object hashes
        compile_and_verify(AdditionalStats::hit(kind, lang))?;
    }

    // Revert dep_a to ensure deps cache hit
    {
        // Write dependency A
        write_source(tempdir, dep_a, r#"int a() { return 0; }"#);

        // Should still be three object hashes
        compile_and_verify(AdditionalStats::hit(kind, lang))?;
    }

    Ok(())
}

pub fn test_time_macros(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) -> Result<()> {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;

    println!("test_time_macros: {name}");

    let src = "time_macros_test.c";
    let obj = "time_macros_test.c.o";
    let dep_a = "time_macros_test_a.h";
    let dep_b = "time_macros_test_b.h";

    let kind = compiler.kind();
    let lang = compiler.lang();

    client.zero_stats();
    client.clear_disk_cache()?;

    let dist = client.is_dist();
    let mut stats = ExpectedStats::new(client);

    let mut compile_and_verify = |additional_stats: AdditionalStats| -> Result<()> {
        let mut cmd = client.cmd();
        cmd.current_dir(tempdir)
            .envs(env_vars.clone())
            .args(compile_cmdline(name, exe, src, obj, []))
            .assert()
            .success();
        stats.assert_added(additional_stats);
        Ok(())
    };

    // Test __TIME__ macro used in main source file
    {
        // Write source file with __TIME__ macro
        write_source(
            tempdir,
            src,
            r#"
#include <stdio.h>
int main(int argc, char** argv) {{
    printf("time: %s", __TIME__);
    return 0;
}}
    "#,
        );

        // Compile and populate deps and objs cache entries
        compile_and_verify(AdditionalStats {
            // The source file itself expands __TIME__, so this is known
            // without running the preprocessor.
            preprocessed: None,
            ..AdditionalStats::not_cacheable()
        })?;

        // Wait 2s and compile again to ensure deps cache miss
        std::thread::sleep(std::time::Duration::from_secs(2));

        compile_and_verify(AdditionalStats {
            // The source file itself expands __TIME__, so this is known
            // without running the preprocessor.
            preprocessed: None,
            ..AdditionalStats::not_cacheable()
        })?;

        // Assert the number of deps cache entries is still 0
        assert_eq!(
            0,
            client.dependencies_cache_entries().into_iter().count() as u64
        );
    }

    let mut entries_a;
    let mut entries_b;

    // Test time macro used in dependency
    {
        write_source(
            tempdir,
            src,
            &format!(
                r#"
#include "{dep_a}"
int main(int argc, char** argv) {{
    a();
    return 0;
}}
"#
            ),
        );

        // Write dependency A with __TIME__ macro
        write_source(
            tempdir,
            dep_a,
            r#"
#include <stdio.h>
void a() { printf("time: %s", __TIME__); }
"#,
        );

        // Compile and populate deps and objs cache entries
        compile_and_verify(AdditionalStats::not_cacheable())?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 0
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 0, &[]);

        // Wait 2s and compile again to ensure deps cache miss
        std::thread::sleep(std::time::Duration::from_secs(2));

        compile_and_verify(AdditionalStats::not_cacheable())?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 0
        assert_num_deps_cache_entries_and_results(dist, entries_b.clone(), 0, &[]);
    }

    // Test time macro used in dependency of a dependency
    {
        write_source(
            tempdir,
            dep_a,
            &format!(
                r#"
#include "{dep_b}"
void a() {{ b(); }}
"#
            ),
        );

        // Write dependency B with __TIME__ macro
        write_source(
            tempdir,
            dep_b,
            r#"
#include <stdio.h>
void b() { printf("time: %s", __TIME__); }
"#,
        );

        // Compile and populate deps and objs cache entries
        compile_and_verify(AdditionalStats::not_cacheable())?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 0
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 0, &[]);

        // Wait 2s and compile again to ensure deps cache miss
        std::thread::sleep(std::time::Duration::from_secs(2));

        compile_and_verify(AdditionalStats::not_cacheable())?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 0
        assert_num_deps_cache_entries_and_results(dist, entries_b.clone(), 0, &[]);
    }

    // Test removing __TIME__ macro from dependency B
    {
        // Remove __TIME__ macro from dependency B
        write_source(tempdir, dep_b, r#"void b() {}"#);

        // Compile and populate deps and objs cache entries
        compile_and_verify(AdditionalStats::miss(kind, lang))?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is 1 (with 1 result)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 1, &[1]);
        // // Verify the new deps cache entry result is at the back of the LRU
        // assert_deps_cache_entries_result_orders_changed(dist, entries_b.clone(), entries_a.clone());

        // Wait 2s and compile again to ensure deps cache hit
        std::thread::sleep(std::time::Duration::from_secs(2));

        compile_and_verify(AdditionalStats::hit(kind, lang))?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 1 (with 1 result)
        assert_num_deps_cache_entries_and_results(dist, entries_b.clone(), 1, &[1]);
        // Verify the deps cache entry results are in the same order
        assert_deps_cache_entries_are_identical(entries_b.clone(), entries_a.clone());
    }

    Ok(())
}

pub fn test_date_macros(client: &SccacheClient, compiler: &Compiler, tempdir: &Path) -> Result<()> {
    if (compiler.name == "clang" || compiler.name == "clang++")
        && !compiler
            .version()
            .and_then(|clang_v| {
                version_compare::compare_to(clang_v, "16.0.0", version_compare::Cmp::Ge)
                    .map_err(|_| anyhow!("clang too old"))
            })
            .unwrap_or_default()
    {
        return Ok(());
    }

    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;

    println!("test_date_macros: {name}");

    let src = "date_macros_test.c";
    let obj = "date_macros_test.c.o";
    let dep_a = "date_macros_test_a.h";
    let dep_b = "date_macros_test_b.h";

    let kind = compiler.kind();
    let lang = compiler.lang();

    client.zero_stats();
    client.clear_disk_cache()?;

    let dist = client.is_dist();
    let mut stats = ExpectedStats::new(client);

    let mut compile_and_verify = |envs: &[(OsString, OsString)], additional_stats| -> Result<()> {
        let mut cmd = client.cmd();
        cmd.current_dir(tempdir)
            .envs(envs.to_vec())
            .args(compile_cmdline(name, exe, src, obj, []))
            .assert()
            .success();
        stats.assert_added(additional_stats);
        Ok(())
    };

    let s_epoch_today = chrono::Local::now().to_utc();
    let s_epoch_tomorrow = s_epoch_today
        .checked_add_days(chrono::Days::new(1))
        .unwrap();
    let s_epoch_today = s_epoch_today.timestamp().to_string();
    let s_epoch_tomorrow = s_epoch_tomorrow.timestamp().to_string();

    let mut entries_a;
    let mut entries_b;

    // Test __DATE__ macro used in main source file
    {
        // Write source file with __DATE__ macro
        write_source(
            tempdir,
            src,
            r#"
#include <stdio.h>
int main(int argc, char** argv) {{
    printf("date: %s", __DATE__);
    return 0;
}}
    "#,
        );

        // Compile "today" and populate deps and objs disk caches
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_today.as_str().into())],
            ]
            .concat(),
            AdditionalStats::miss(kind, lang),
        )?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is now 1 (with 1 result)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 1, &[1]);

        // Compile again to ensure deps cache hit
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_today.as_str().into())],
            ]
            .concat(),
            AdditionalStats::hit(kind, lang),
        )?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 1 (with 1 result)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());

        // Compile "tomorrow" to ensure cache miss
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_tomorrow.as_str().into())],
            ]
            .concat(),
            AdditionalStats::miss(kind, lang),
        )?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is now 2 (with 1 result each)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 2, &[1, 1]);

        // Compile "tomorrow" again to ensure deps cache hit
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_tomorrow.as_str().into())],
            ]
            .concat(),
            AdditionalStats::hit(kind, lang),
        )?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 2 (with 1 result each)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());
    }

    // Test date macro used in dependency
    {
        write_source(
            tempdir,
            src,
            &format!(
                r#"
#include "{dep_a}"
int main(int argc, char** argv) {{
    a();
    return 0;
}}
"#
            ),
        );

        // Write dependency A with __DATE__ macro
        write_source(
            tempdir,
            dep_a,
            r#"
#include <stdio.h>
void a() { printf("date: %s", __DATE__); }
"#,
        );

        // Compile "today" and populate deps and objs disk caches
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_today.as_str().into())],
            ]
            .concat(),
            AdditionalStats::miss(kind, lang),
        )?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is now 3 (with 1 result each)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 1]);

        // Compile again to ensure deps cache hit
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_today.as_str().into())],
            ]
            .concat(),
            AdditionalStats::hit(kind, lang),
        )?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with 1 result each)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());

        // Compile "tomorrow" to ensure cache miss
        // A dist client still has the dependencies it found for the unchanged
        // source file, and they all still exist
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_tomorrow.as_str().into())],
            ]
            .concat(),
            AdditionalStats::miss(kind, lang).reusing_dist_deps(),
        )?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 2] results)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 2]);

        // Compile "tomorrow" again to ensure deps cache hit
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_tomorrow.as_str().into())],
            ]
            .concat(),
            AdditionalStats::hit(kind, lang),
        )?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 2] results)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());
    }

    // Test date macro used in dependency of a dependency
    {
        write_source(
            tempdir,
            dep_a,
            &format!(
                r#"
#include "{dep_b}"
void a() {{ b(); }}
"#
            ),
        );

        // Write dependency B with __DATE__ macro
        write_source(
            tempdir,
            dep_b,
            r#"
#include <stdio.h>
void b() { printf("date: %s", __DATE__); }
"#,
        );

        // Compile "today" and populate deps and objs disk caches
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_today.as_str().into())],
            ]
            .concat(),
            AdditionalStats::miss(kind, lang).reusing_dist_deps(),
        )?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 3] results)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 3]);

        // Compile "today" again to ensure deps cache hit
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_today.as_str().into())],
            ]
            .concat(),
            AdditionalStats::hit(kind, lang),
        )?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 3] results)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());

        // Compile "tomorrow" to ensure cache miss
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_tomorrow.as_str().into())],
            ]
            .concat(),
            AdditionalStats::miss(kind, lang).reusing_dist_deps(),
        )?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 4] results)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 4]);

        // Compile "today" again to ensure deps cache hit
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_today.as_str().into())],
            ]
            .concat(),
            AdditionalStats::hit(kind, lang),
        )?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 4] results)
        assert_num_deps_cache_entries_and_results(dist, entries_b.clone(), 3, &[1, 1, 4]);
        // Verify the new deps cache entry result is at the back of the LRU
        assert_deps_cache_entries_result_orders_changed(
            dist,
            entries_a
                .clone()
                .into_iter()
                .filter(|(_, e)| e.iter().count() > 1)
                .collect::<Vec<_>>(),
            entries_b
                .clone()
                .into_iter()
                .filter(|(_, e)| e.iter().count() > 1)
                .collect::<Vec<_>>(),
        );

        // Compile "tomorrow" again to ensure deps cache hit
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_tomorrow.as_str().into())],
            ]
            .concat(),
            AdditionalStats::hit(kind, lang),
        )?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 4] results)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 4]);
        // Verify the new deps cache entry result is at the back of the LRU
        assert_deps_cache_entries_result_orders_changed(
            dist,
            entries_b
                .clone()
                .into_iter()
                .filter(|(_, e)| e.iter().count() > 1)
                .collect::<Vec<_>>(),
            entries_a
                .clone()
                .into_iter()
                .filter(|(_, e)| e.iter().count() > 1)
                .collect::<Vec<_>>(),
        );
    }

    // Test removing __DATE__ from dependency B
    {
        // Write dependency B without __DATE__ macro
        write_source(tempdir, dep_b, r#"void b() {}"#);

        // Compile "today" and populate deps and objs disk caches
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_today.as_str().into())],
            ]
            .concat(),
            AdditionalStats::miss(kind, lang).reusing_dist_deps(),
        )?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 5] results)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 5]);

        // Compile "today" again to ensure deps cache hit
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_today.as_str().into())],
            ]
            .concat(),
            AdditionalStats::hit(kind, lang),
        )?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 5] results)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());

        // Compile "tomorrow" again to ensure deps cache hit
        compile_and_verify(
            &[
                &env_vars[..],
                &[("SOURCE_DATE_EPOCH".into(), s_epoch_tomorrow.as_str().into())],
            ]
            .concat(),
            AdditionalStats::hit(kind, lang),
        )?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 5] results)
        assert_deps_cache_entries_are_identical(entries_b.clone(), entries_a.clone());
    }

    Ok(())
}

pub fn test_timestamp_macros(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
) -> Result<()> {
    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;

    println!("test_timestamp_macros: {name}");

    let src = "timestamp_macro_test.c";
    let obj = "timestamp_macro_test.c.o";
    let dep_a = "timestamp_macro_test_a.h";
    let dep_b = "timestamp_macro_test_b.h";

    let kind = compiler.kind();
    let lang = compiler.lang();

    client.zero_stats();
    client.clear_disk_cache()?;

    let dist = client.is_dist();
    let mut stats = ExpectedStats::new(client);

    let mut compile_and_verify = |additional_stats: AdditionalStats| -> Result<()> {
        let mut cmd = client.cmd();
        cmd.current_dir(tempdir)
            .envs(env_vars.clone())
            .args(compile_cmdline(name, exe, src, obj, []))
            .assert()
            .success();
        stats.assert_added(additional_stats);
        Ok(())
    };

    let mut entries_a;
    let mut entries_b;

    // Test __TIMESTAMP__ macro used in main source file
    {
        // Write source file with __TIMESTAMP__ macro
        write_source(
            tempdir,
            src,
            r#"
#include <stdio.h>
int main(int argc, char** argv) {{
    printf("timestamp 1: %s", __TIMESTAMP__);
    return 0;
}}
    "#,
        );

        // Compile and populate deps and objs cache entries
        compile_and_verify(AdditionalStats::miss(kind, lang))?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is now 1 (with 1 result)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 1, &[1]);

        // Compile again to ensure deps cache hit
        compile_and_verify(AdditionalStats::hit(kind, lang))?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 1 (with 1 result)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());

        // Wait 2s, update source file with __TIMESTAMP__ macro,
        // then compile again to ensure deps cache miss
        std::thread::sleep(std::time::Duration::from_secs(2));

        write_source(
            tempdir,
            src,
            r#"
#include <stdio.h>
int main(int argc, char** argv) {{
    printf("timestamp 2: %s", __TIMESTAMP__);
    return 0;
}}
    "#,
        );

        compile_and_verify(AdditionalStats::miss(kind, lang))?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is now 2 (with 1 result each)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 2, &[1, 1]);

        // Compile again to ensure deps cache hit
        compile_and_verify(AdditionalStats::hit(kind, lang))?;

        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 2 (with 1 result each)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());
    }

    // Test timestamp macro used in dependency
    {
        std::thread::sleep(std::time::Duration::from_secs(2));

        write_source(
            tempdir,
            src,
            &format!(
                r#"
#include "{dep_a}"
int main(int argc, char** argv) {{
    a();
    return 0;
}}
"#
            ),
        );

        // Write dependency A with __TIMESTAMP__ macro
        write_source(
            tempdir,
            dep_a,
            r#"
#include <stdio.h>
void a() { printf("timestamp: %s", __TIMESTAMP__); }
"#,
        );

        // Compile and populate deps and objs cache entries
        compile_and_verify(AdditionalStats::miss(kind, lang))?;

        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is now 3 (with 1 result each)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 1]);

        // Compile again to ensure deps cache hit
        compile_and_verify(AdditionalStats::hit(kind, lang))?;

        // Assert the number of deps cache entries is still 3
        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with 1 result each)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());

        // Wait 2s, update dependency A with __TIMESTAMP__ macro,
        // then compile again to ensure deps cache miss
        std::thread::sleep(std::time::Duration::from_secs(2));

        write_source(
            tempdir,
            dep_a,
            r#"
#include <stdio.h>
void a() { printf("timestamp: %s", __TIMESTAMP__); }
"#,
        );

        // A dist client still has the dependencies it found for the unchanged
        // source file, and they all still exist
        compile_and_verify(AdditionalStats::miss(kind, lang).reusing_dist_deps())?;

        // Assert the number of deps cache entries is still 3
        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 2] results)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 2]);

        // Compile again to ensure deps cache hit
        compile_and_verify(AdditionalStats::hit(kind, lang))?;

        // Assert the number of deps cache entries is still 3
        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 2] results)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());
    }

    // Test timestamp macro used in dependency of a dependency
    {
        std::thread::sleep(std::time::Duration::from_secs(2));

        write_source(
            tempdir,
            dep_a,
            &format!(
                r#"
#include "{dep_b}"
void a() {{ b(); }}
"#
            ),
        );

        // Write dependency B with __TIMESTAMP__ macro
        write_source(
            tempdir,
            dep_b,
            r#"
#include <stdio.h>
void b() { printf("timestamp: %s", __TIMESTAMP__); }
"#,
        );

        // Compile and populate deps and objs cache entries
        compile_and_verify(AdditionalStats::miss(kind, lang).reusing_dist_deps())?;

        // Assert the number of deps cache entries is still 3
        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 3] results)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 3]);

        // Compile again to ensure deps cache hit
        compile_and_verify(AdditionalStats::hit(kind, lang))?;

        // Assert the number of deps cache entries is still 3
        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 3] results)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());

        // Wait 2s, update dependency B with __TIMESTAMP__ macro,
        // then compile again to ensure deps cache miss
        std::thread::sleep(std::time::Duration::from_secs(2));

        let prev_mtime = std::fs::metadata(tempdir.join(dep_b))?.modified()?;

        write_source(
            tempdir,
            dep_b,
            r#"
#include <stdio.h>
void b() { printf("timestamp: %s", __TIMESTAMP__); }
"#,
        );

        compile_and_verify(AdditionalStats::miss(kind, lang).reusing_dist_deps())?;

        // Assert the number of deps cache entries is still 3
        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 4] results)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 4]);

        // Compile again to ensure deps cache hit
        compile_and_verify(AdditionalStats::hit(kind, lang))?;

        // Assert the number of deps cache entries is still 3
        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 4] results)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());

        // Reset the mtime and compile again to validate deps cache hits
        // Don't error if this fails due to permissions issues (i.e. in Windows CI)
        let mtime_reset = std::fs::File::open(tempdir.join(dep_b))
            .and_then(|f| f.set_modified(prev_mtime).map(|_| f))
            .and_then(|f| f.sync_all());

        if mtime_reset.is_ok() {
            // Compile again to ensure deps cache hit
            compile_and_verify(AdditionalStats::hit(kind, lang))?;

            entries_a = client.dependencies_cache_entries();
            // Verify the number of deps cache entries is still 3 (with [1, 1, 4] results)
            assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 4]);
            // Verify the new deps cache entry result is at the back of the LRU
            assert_deps_cache_entries_result_orders_changed(
                dist,
                entries_b
                    .clone()
                    .into_iter()
                    .filter(|(_, e)| e.iter().count() > 1)
                    .collect::<Vec<_>>(),
                entries_a
                    .clone()
                    .into_iter()
                    .filter(|(_, e)| e.iter().count() > 1)
                    .collect::<Vec<_>>(),
            );
        }
    }

    // Test removing __TIMESTAMP__ from dependency B
    {
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Write dependency B without __TIMESTAMP__ macro
        write_source(tempdir, dep_b, r#"void b() {}"#);

        // Compile and populate deps and objs cache entries
        compile_and_verify(AdditionalStats::miss(kind, lang).reusing_dist_deps())?;

        // Assert the number of deps cache entries is still 3
        entries_a = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 5] results)
        assert_num_deps_cache_entries_and_results(dist, entries_a.clone(), 3, &[1, 1, 5]);

        // Compile again to ensure deps cache hit
        compile_and_verify(AdditionalStats::hit(kind, lang))?;

        // Assert the number of deps cache entries is still 3
        entries_b = client.dependencies_cache_entries();
        // Verify the number of deps cache entries is still 3 (with [1, 1, 5] results)
        assert_deps_cache_entries_are_identical(entries_a.clone(), entries_b.clone());
    }

    Ok(())
}

macro_rules! sccache_macro_tests {
    ($compiler:ident, $compiler_name:expr) => {
        paste! {
            #[test]
            fn [<test_sccache_dependency_changes_ $compiler>] () {
                with_local_server($compiler_name, test_dependency_changes)
                    .transpose()
                    .unwrap();
            }

            #[test]
            fn [<test_sccache_macros_time_ $compiler>] () {
                with_local_server($compiler_name, test_time_macros)
                    .transpose()
                    .unwrap();
            }

            #[test]
            fn [<test_sccache_macros_date_ $compiler>] () {
                // MSVC and NVHPC don't respect SOURCE_DATE_EPOCH
                if !matches!($compiler_name, "cl" | "nvc" | "nvc++") {
                    with_local_server($compiler_name, test_date_macros)
                        .transpose()
                        .unwrap();
                }
            }

            #[test]
            fn [<test_sccache_macros_timestamp_ $compiler>] () {
                with_local_server($compiler_name, test_timestamp_macros)
                    .transpose()
                    .unwrap();
            }
        }
    };
}

#[cfg(unix)]
sccache_macro_tests!(gcc, "gcc");
#[cfg(unix)]
sccache_macro_tests!(clang, "clang");
#[cfg(unix)]
sccache_macro_tests!(clangxx, "clang++");
#[cfg(unix)]
sccache_macro_tests!(nvc, "nvc");
#[cfg(unix)]
sccache_macro_tests!(nvcxx, "nvc++");

#[cfg(target_env = "msvc")]
sccache_macro_tests!(msvc, "cl");
