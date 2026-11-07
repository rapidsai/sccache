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

//! Compiling CUDA with nvcc and clang: the stats for nvcc's steps, and
//! cache hits for each of them.

use crate::c::{INPUT, OUTPUT};
use crate::harness::{
    client::{SccacheClient, make_sccache_client},
    compilers::{
        Compiler, compile_cmdline, compile_cuda_cmdline, find_compiler, find_cuda_compiler,
    },
    copy_to_tempdir, init_logger,
    stats::{AdditionalStats, ExpectedStats},
    write_source,
};
use assert_cmd::prelude::*;
use fs_err as fs;
use pastey::paste;
use sccache::compiler::{CCompilerKind, Language};
use std::ffi::OsString;
use std::path::Path;
use test_case::test_case;

const INPUT_FOR_CUDA_A: &str = "test_a.cu";
const INPUT_FOR_CUDA_A_COPY: &str = "test_a_copy.cu";
const INPUT_FOR_CUDA_A_INTERNAL_LINKAGE: &str = "test_a_internal_linkage.cu";
const INPUT_FOR_CUDA_B: &str = "test_b.cu";
const INPUT_FOR_CUDA_C: &str = "test_c.cu";

fn test_nvcc_cuda_compiles(
    client: &SccacheClient,
    compiler: &Compiler,
    host_compiler: &Compiler,
    tempdir: &Path,
    with_debug_flags: bool,
    with_rdc: bool,
) {
    let mut extra_args = vec![format!("-ccbin={}", host_compiler.exe.to_string_lossy()).into()];

    if with_debug_flags {
        extra_args.push("-G".into());
    }

    if with_rdc {
        extra_args.push("-rdc=true".into());
    }

    extra_args.push("-t=10".into());

    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;

    println!("test_nvcc_cuda_compiles: {name}/{}", host_compiler.name);

    let host_compiler_kind = host_compiler.name.into();

    // Compile multiple source files.
    copy_to_tempdir(
        &[
            INPUT_FOR_CUDA_A,
            INPUT_FOR_CUDA_A_COPY,
            INPUT_FOR_CUDA_A_INTERNAL_LINKAGE,
            INPUT_FOR_CUDA_B,
        ],
        tempdir,
    );

    let build_dir = Path::new("build");
    fs::create_dir_all(tempdir.join(build_dir)).unwrap();

    let mut stats = ExpectedStats::new(client);

    let mut run_cuda_test = |compile_flag: &str,
                             input: &Path,
                             output: &Path,
                             extra_args: &[OsString],
                             additional_stats: AdditionalStats| {
        client
            .cmd()
            .args(compile_cuda_cmdline(
                name,
                exe,
                compile_flag,
                input.to_string_lossy().as_ref(),
                output.to_string_lossy().as_ref(),
                extra_args,
            ))
            .current_dir(tempdir)
            .envs(env_vars.clone())
            .assert()
            .success();

        assert!(
            fs::metadata(tempdir.join(output))
                .map(|m| m.len() > 0)
                .unwrap()
        );

        fs::remove_file(tempdir.join(output)).unwrap();

        stats.assert_added(additional_stats);
    };

    trace!("compile A ptx");
    run_cuda_test(
        "-ptx",
        Path::new(INPUT_FOR_CUDA_A), // relative path for input
        &build_dir.join("test.ptx"), // relative path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(3),
            preprocessed: Some(1),
            cache_writes: Some(2),
            compilations: Some(3),
            compile_requests: Some(1),
            requests_executed: Some(3),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
            ]),
            ..Default::default()
        },
    );

    trace!("compile A cubin");
    run_cuda_test(
        "-cubin",
        Path::new(INPUT_FOR_CUDA_A), // relative path for input
        &tempdir.join(build_dir).join("test.cubin"), // absolute path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(4),
            preprocessed: Some(1),
            cache_writes: Some(2),
            compilations: Some(3),
            compile_requests: Some(1),
            requests_executed: Some(4),
            cache_hits: Some(vec![(CCompilerKind::Cicc, Language::Ptx, 1)]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            ..Default::default()
        },
    );

    trace!("compile A");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_A), // relative path for input
        &build_dir.join(OUTPUT),     // relative path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(5),
            preprocessed: Some(1),
            cache_writes: Some(3 + with_debug_flags as u64),
            compilations: Some(6 + with_debug_flags as u64),
            compile_requests: Some(1),
            requests_executed: Some(7),
            cache_hits: Some(vec![(
                CCompilerKind::Ptxas,
                Language::Cubin,
                !with_debug_flags as u64,
            )]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (
                    CCompilerKind::Ptxas,
                    Language::Cubin,
                    with_debug_flags as u64,
                ),
            ]),
            ..Default::default()
        },
    );

    trace!("compile A (cached)");
    run_cuda_test(
        "-c",
        &tempdir.join(INPUT_FOR_CUDA_A), // absolute path for input
        &tempdir.join(build_dir).join(OUTPUT), // absolute path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(1),
            compile_requests: Some(1),
            requests_executed: Some(1),
            cache_hits: Some(vec![(CCompilerKind::Nvcc, Language::Cuda, 1)]),
            ..Default::default()
        },
    );

    // Compile a copy of `test_a.cu` to ensure we get cache hits for identical PTX across different files.
    trace!("compile A (copy)");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_A_COPY), // relative path for input
        &build_dir.join(OUTPUT),          // relative path for output
        &extra_args,
        // Since `test_a_copy.cu` is a copy of `test_a.cu`, its PTX will be identical when *not* using -G.
        // But -G causes cudafe++ and cicc to embed the source path their output, and we get cache misses.
        AdditionalStats {
            hashed: Some(5),
            preprocessed: Some(1),
            cache_writes: Some(3 + with_debug_flags as u64),
            compilations: Some(6 + with_debug_flags as u64),
            compile_requests: Some(1),
            requests_executed: Some(7),
            cache_hits: Some(vec![(
                CCompilerKind::Ptxas,
                Language::Cubin,
                !with_debug_flags as u64,
            )]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (
                    CCompilerKind::Ptxas,
                    Language::Cubin,
                    with_debug_flags as u64,
                ),
            ]),
            ..Default::default()
        },
    );

    trace!("compile A (copy) (cached)");
    run_cuda_test(
        "-c",
        &tempdir.join(INPUT_FOR_CUDA_A_COPY), // absolute path for input
        &tempdir.join(build_dir).join(OUTPUT), // absolute path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(1),
            preprocessed: Some(0),
            compile_requests: Some(1),
            requests_executed: Some(1),
            cache_hits: Some(vec![(CCompilerKind::Nvcc, Language::Cuda, 1)]),
            ..Default::default()
        },
    );

    trace!("compile A (internal linkage)");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_A_INTERNAL_LINKAGE), // relative path for input
        &build_dir.join(OUTPUT),                      // relative path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(5),
            preprocessed: Some(1),
            cache_writes: Some(4),
            compilations: Some(7),
            compile_requests: Some(1),
            requests_executed: Some(7),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            ..Default::default()
        },
    );

    trace!("compile A (internal linkage) (cached)");
    run_cuda_test(
        "-c",
        &tempdir.join(INPUT_FOR_CUDA_A_INTERNAL_LINKAGE), // absolute path for input
        &tempdir.join(build_dir).join(OUTPUT),            // absolute path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(1),
            preprocessed: Some(0),
            compile_requests: Some(1),
            requests_executed: Some(1),
            cache_hits: Some(vec![(CCompilerKind::Nvcc, Language::Cuda, 1)]),
            ..Default::default()
        },
    );

    // By compiling another input source we verify that the pre-processor
    // phase is correctly running and outputting text
    trace!("compile B");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_B), // relative path for input
        &build_dir.join(OUTPUT),     // relative path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(5),
            preprocessed: Some(1),
            cache_writes: Some(4),
            compilations: Some(7),
            compile_requests: Some(1),
            requests_executed: Some(7),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            ..Default::default()
        },
    );

    trace!("compile B (cached)");
    run_cuda_test(
        "-c",
        &tempdir.join(INPUT_FOR_CUDA_B), // absolute path for input
        &tempdir.join(build_dir).join(OUTPUT), // absolute path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(1),
            preprocessed: Some(0),
            compile_requests: Some(1),
            requests_executed: Some(1),
            cache_hits: Some(vec![(CCompilerKind::Nvcc, Language::Cuda, 1)]),
            ..Default::default()
        },
    );

    // Test to ensure #2299 doesn't regress (https://github.com/mozilla/sccache/issues/2299)
    let test_2299_src_name = "test_2299.cu";
    let test_2299_out_name = "test_2299.cu.o";
    // Two versions of the source with different contents inside the #ifndef __CUDA_ARCH__
    let test_2299_cu_src_1 = r#"
#ifndef __CUDA_ARCH__
static const auto x = 5;
#endif
int main(int argc, char** argv) {
  return 0;
}
"#;
    let test_2299_cu_src_2 = r#"
#ifndef __CUDA_ARCH__
static const auto x = "5";
#endif
int main(int argc, char** argv) {
  return 0;
}
"#;
    write_source(tempdir, test_2299_src_name, test_2299_cu_src_1);
    run_cuda_test(
        "-c",
        Path::new(test_2299_src_name),       // relative path for input
        &build_dir.join(test_2299_out_name), // relative path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(5),
            preprocessed: Some(1),
            cache_writes: Some(4),
            compilations: Some(7),
            compile_requests: Some(1),
            requests_executed: Some(7),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            ..Default::default()
        },
    );

    write_source(tempdir, test_2299_src_name, test_2299_cu_src_2);
    trace!("compile test_2299.cu (2)");
    run_cuda_test(
        "-c",
        Path::new(test_2299_src_name),       // relative path for input
        &build_dir.join(test_2299_out_name), // relative path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(5),
            preprocessed: Some(1),
            cache_writes: Some(2),
            compilations: Some(5),
            compile_requests: Some(1),
            requests_executed: Some(7),
            cache_hits: Some(vec![
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
            ]),
            ..Default::default()
        },
    );

    // Recompile the original version again to ensure only cache hits
    write_source(tempdir, test_2299_src_name, test_2299_cu_src_1);
    trace!("compile test_2299.cu (1) (cached)");
    run_cuda_test(
        "-c",
        &tempdir.join(test_2299_src_name), // absolute path for input
        &tempdir.join(build_dir).join(test_2299_out_name), // absolute path for output
        &extra_args,
        AdditionalStats {
            hashed: Some(1),
            preprocessed: Some(0),
            compile_requests: Some(1),
            requests_executed: Some(1),
            cache_hits: Some(vec![(CCompilerKind::Nvcc, Language::Cuda, 1)]),
            ..Default::default()
        },
    );

    // Precompile sm_86 PTX and cubin so their cache entries potentially have a different .module_id file
    trace!("compile A cubin sm_86");
    run_cuda_test(
        "-cubin",
        Path::new(INPUT_FOR_CUDA_A), // relative path for input
        &build_dir.join(OUTPUT),     // relative path for output
        &[
            extra_args.as_slice(),
            &["-gencode=arch=compute_86,code=[sm_86]".into()],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(4),
            preprocessed: Some(1),
            cache_writes: Some(3),
            compilations: Some(4),
            compile_requests: Some(1),
            requests_executed: Some(4),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            ..Default::default()
        },
    );

    // Precompile sm_86 PTX and cubin so their cache entries potentially have a different .module_id file
    trace!("compile A (internal linkage) cubin sm_86");
    run_cuda_test(
        "-cubin",
        Path::new(INPUT_FOR_CUDA_A_INTERNAL_LINKAGE), // relative path for input
        &build_dir.join(OUTPUT),                      // relative path for output
        &[
            extra_args.as_slice(),
            &["-gencode=arch=compute_86,code=[sm_86]".into()],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(4),
            preprocessed: Some(1),
            cache_writes: Some(3),
            compilations: Some(4),
            compile_requests: Some(1),
            requests_executed: Some(4),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            ..Default::default()
        },
    );

    // Test compiling a file whose PTX yields a cache hit for a cubin from another file (`test_a.cu`)
    trace!("compile B cubin sm_86");
    run_cuda_test(
        "-cubin",
        Path::new(INPUT_FOR_CUDA_B), // relative path for input
        &build_dir.join(OUTPUT),     // relative path for output
        &[
            extra_args.as_slice(),
            &["-gencode=arch=compute_86,code=[sm_86]".into()],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(4),
            preprocessed: Some(1),
            cache_writes: Some(2 + with_debug_flags as u64),
            compilations: Some(3 + with_debug_flags as u64),
            compile_requests: Some(1),
            requests_executed: Some(4),
            cache_hits: Some(vec![(
                CCompilerKind::Ptxas,
                Language::Cubin,
                !with_debug_flags as u64,
            )]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (
                    CCompilerKind::Ptxas,
                    Language::Cubin,
                    with_debug_flags as u64,
                ),
            ]),
            ..Default::default()
        },
    );

    // Test compiling a multiarch object where the PTX and cubin for one of the archs is cached
    trace!("compile A sm_80,sm_86");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_A), // relative path for input
        &build_dir.join(OUTPUT),     // relative path for output
        &[
            extra_args.as_slice(),
            &[
                "-gencode=arch=compute_80,code=[sm_80]".into(),
                "-gencode=arch=compute_86,code=[compute_86,sm_86]".into(),
            ],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(7),
            preprocessed: Some(1),
            cache_writes: Some(5 + with_debug_flags as u64),
            compilations: Some(9 + with_debug_flags as u64),
            compile_requests: Some(1),
            requests_executed: Some(10),
            cache_hits: Some(vec![(
                CCompilerKind::Ptxas,
                Language::Cubin,
                !with_debug_flags as u64,
            )]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 2),
                (
                    CCompilerKind::Ptxas,
                    Language::Cubin,
                    1 + with_debug_flags as u64,
                ),
            ]),
            ..Default::default()
        },
    );

    // Test compiling a multiarch object of a different source file, but
    // whose device code is the same as a previously-compiled files'
    trace!("compile A (copy) sm_80,sm_86");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_A_COPY), // relative path for input
        &build_dir.join(OUTPUT),          // relative path for output
        &[
            extra_args.as_slice(),
            &[
                "-gencode=arch=compute_80,code=[sm_80]".into(),
                "-gencode=arch=compute_86,code=[compute_86,sm_86]".into(),
            ],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(7),
            preprocessed: Some(1),
            cache_writes: Some(4 + 2 * with_debug_flags as u64),
            compilations: Some(8 + 2 * with_debug_flags as u64),
            compile_requests: Some(1),
            requests_executed: Some(10),
            cache_hits: Some(vec![(
                CCompilerKind::Ptxas,
                Language::Cubin,
                2 * !with_debug_flags as u64,
            )]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 2),
                (
                    CCompilerKind::Ptxas,
                    Language::Cubin,
                    2 * with_debug_flags as u64,
                ),
            ]),
            ..Default::default()
        },
    );

    // Test compiling a multiarch object where the PTX and cubin for one of the archs is cached
    trace!("compile A (internal linkage) sm_80,sm_86");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_A_INTERNAL_LINKAGE), // relative path for input
        &build_dir.join(OUTPUT),                      // relative path for output
        &[
            extra_args.as_slice(),
            &[
                "-gencode=arch=compute_80,code=[sm_80]".into(),
                "-gencode=arch=compute_86,code=[compute_86,sm_86]".into(),
            ],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(7),
            preprocessed: Some(1),
            cache_writes: Some(5 + with_debug_flags as u64),
            compilations: Some(9 + with_debug_flags as u64),
            compile_requests: Some(1),
            requests_executed: Some(10),
            cache_hits: Some(vec![(
                CCompilerKind::Ptxas,
                Language::Cubin,
                !with_debug_flags as u64,
            )]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 2),
                (
                    CCompilerKind::Ptxas,
                    Language::Cubin,
                    1 + with_debug_flags as u64,
                ),
            ]),
            ..Default::default()
        },
    );

    trace!("compile B sm_80,sm_86");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_B), // relative path for input
        &build_dir.join(OUTPUT),     // relative path for output
        &[
            extra_args.as_slice(),
            &[
                "-gencode=arch=compute_80,code=[sm_80]".into(),
                "-gencode=arch=compute_86,code=[compute_86,sm_86]".into(),
            ],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(7),
            preprocessed: Some(1),
            cache_writes: Some(5 + with_debug_flags as u64),
            compilations: Some(9 + with_debug_flags as u64),
            compile_requests: Some(1),
            requests_executed: Some(10),
            cache_hits: Some(vec![(
                CCompilerKind::Ptxas,
                Language::Cubin,
                !with_debug_flags as u64,
            )]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 2),
                (
                    CCompilerKind::Ptxas,
                    Language::Cubin,
                    1 + with_debug_flags as u64,
                ),
            ]),
            ..Default::default()
        },
    );

    // Test that compiling a single-arch object where the arch is a subset of
    // a previous multi-arch compilation produces cache hits on the underlying
    // PTX and cubin compilations.
    trace!("compile A sm_80");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_A), // relative path for input
        &build_dir.join(OUTPUT),     // relative path for output
        &[
            extra_args.as_slice(),
            &["-gencode=arch=compute_80,code=[compute_80,sm_80]".into()],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(5),
            preprocessed: Some(1),
            cache_writes: Some(2),
            compilations: Some(5),
            compile_requests: Some(1),
            requests_executed: Some(7),
            cache_hits: Some(vec![
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
            ]),
            ..Default::default()
        },
    );

    trace!("compile B sm_80");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_B), // relative path for input
        &build_dir.join(OUTPUT),     // relative path for output
        &[
            extra_args.as_slice(),
            &["-gencode=arch=compute_80,code=[compute_80,sm_80]".into()],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(5),
            preprocessed: Some(1),
            cache_writes: Some(2),
            compilations: Some(5),
            compile_requests: Some(1),
            requests_executed: Some(7),
            cache_hits: Some(vec![
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
            ]),
            ..Default::default()
        },
    );

    // Test that compiling a single-arch cubin where the arch is a subset of
    // a previous multi-arch compilation produces cache hits on the underlying
    // PTX and cubin compilations.
    trace!("compile A cubin sm_80");
    run_cuda_test(
        "-cubin",
        &tempdir.join(INPUT_FOR_CUDA_A), // absolute path for input
        &tempdir.join(build_dir).join("test.cubin"), // absolute path for output
        &[
            extra_args.as_slice(),
            &["-gencode=arch=compute_80,code=[sm_80]".into()],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(4),
            preprocessed: Some(1),
            cache_writes: Some(2 + with_debug_flags as u64),
            compilations: Some(3 + with_debug_flags as u64),
            compile_requests: Some(1),
            requests_executed: Some(4),
            cache_hits: Some(vec![(
                CCompilerKind::Ptxas,
                Language::Cubin,
                !with_debug_flags as u64,
            )]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (
                    CCompilerKind::Ptxas,
                    Language::Cubin,
                    with_debug_flags as u64,
                ),
            ]),
            ..Default::default()
        },
    );

    trace!("compile B cubin sm_80");
    run_cuda_test(
        "-cubin",
        &tempdir.join(INPUT_FOR_CUDA_B), // absolute path for input
        &tempdir.join(build_dir).join("test.cubin"), // absolute path for output
        &[
            extra_args.as_slice(),
            &["-gencode=arch=compute_80,code=[sm_80]".into()],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(4),
            preprocessed: Some(1),
            cache_writes: Some(2 + with_debug_flags as u64),
            compilations: Some(3 + with_debug_flags as u64),
            compile_requests: Some(1),
            requests_executed: Some(4),
            cache_hits: Some(vec![(
                CCompilerKind::Ptxas,
                Language::Cubin,
                !with_debug_flags as u64,
            )]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
                (
                    CCompilerKind::Ptxas,
                    Language::Cubin,
                    with_debug_flags as u64,
                ),
            ]),
            ..Default::default()
        },
    );

    // Test that compiling sm80 PTX and assembling as an sm86 cubin
    // yields a cache hit for the PTX and cache miss for the cubin
    trace!("compile A compute_80,sm_86");
    run_cuda_test(
        "-c",
        Path::new(INPUT_FOR_CUDA_A), // relative path for input
        &build_dir.join(OUTPUT),     // relative path for output
        &[
            extra_args.as_slice(),
            &["-gencode=arch=compute_80,code=sm_86".into()],
        ]
        .concat(),
        AdditionalStats {
            hashed: Some(5),
            preprocessed: Some(1),
            cache_writes: Some(2),
            compilations: Some(5),
            compile_requests: Some(1),
            requests_executed: Some(7),
            cache_hits: Some(vec![
                (CCompilerKind::CudaFE, Language::CudaFE, 1),
                (CCompilerKind::Cicc, Language::Ptx, 1),
            ]),
            cache_misses: Some(vec![
                (CCompilerKind::Nvcc, Language::Cuda, 1),
                (CCompilerKind::Ptxas, Language::Cubin, 1),
            ]),
            ..Default::default()
        },
    );

    if !cfg!(target_os = "windows") {
        // Test compiling an executable (`nvcc -x cu test_a.cu -o test_a`)
        trace!("compile A to executable");
        run_cuda_test(
            "",
            Path::new(INPUT_FOR_CUDA_A), // relative path for input
            &build_dir.join("test_a"),   // relative path for output
            &[
                extra_args.as_slice(),
                &[
                    "-gencode=arch=compute_80,code=[sm_80]".into(),
                    "-gencode=arch=compute_86,code=[compute_86,sm_86]".into(),
                ],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(11),
                preprocessed: Some(3),
                cache_writes: Some(2),
                compilations: Some(5),
                compile_requests: Some(1),
                non_cacheable_compilations: Some(1),
                requests_executed: Some(11),
                cache_hits: Some(vec![
                    (CCompilerKind::CudaFE, Language::CudaFE, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 2),
                    (CCompilerKind::Ptxas, Language::Cubin, 2),
                ]),
                cache_misses: Some(vec![(host_compiler_kind, Language::Cxx, 2)]),
                ..Default::default()
            },
        );

        // Test compiling an executable (`nvcc -x cu test_a_copy.cu -o test_a_copy`)
        trace!("compile A (copy) to executable");
        run_cuda_test(
            "",
            Path::new(INPUT_FOR_CUDA_A_COPY), // relative path for input
            &build_dir.join("test_a_copy"),   // relative path for output
            &[
                extra_args.as_slice(),
                &[
                    "-gencode=arch=compute_80,code=[sm_80]".into(),
                    "-gencode=arch=compute_86,code=[compute_86,sm_86]".into(),
                ],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(11),
                preprocessed: Some(3),
                cache_writes: Some(2),
                compilations: Some(5),
                compile_requests: Some(1),
                non_cacheable_compilations: Some(1),
                requests_executed: Some(11),
                cache_hits: Some(vec![
                    (CCompilerKind::CudaFE, Language::CudaFE, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 2),
                    (CCompilerKind::Ptxas, Language::Cubin, 2),
                ]),
                cache_misses: Some(vec![(host_compiler.name.into(), Language::Cxx, 2)]),
                ..Default::default()
            },
        );

        // Test compiling an executable (`nvcc -x cu test_a_internal_linkage.cu -o test_a_internal_linkage`)
        trace!("compile A (internal linkage) to executable");
        run_cuda_test(
            "",
            Path::new(INPUT_FOR_CUDA_A_INTERNAL_LINKAGE), // relative path for input
            &build_dir.join("test_a_internal_linkage"),   // relative path for output
            &[
                extra_args.as_slice(),
                &[
                    "-gencode=arch=compute_80,code=[sm_80]".into(),
                    "-gencode=arch=compute_86,code=[compute_86,sm_86]".into(),
                ],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(11),
                preprocessed: Some(3),
                cache_writes: Some(2),
                compilations: Some(5),
                compile_requests: Some(1),
                non_cacheable_compilations: Some(1),
                requests_executed: Some(11),
                cache_hits: Some(vec![
                    (CCompilerKind::CudaFE, Language::CudaFE, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 2),
                    (CCompilerKind::Ptxas, Language::Cubin, 2),
                ]),
                cache_misses: Some(vec![(host_compiler_kind, Language::Cxx, 2)]),
                ..Default::default()
            },
        );

        // Test compiling an executable (`nvcc -x cu test_b.cu -o test_b`)
        trace!("compile B to executable");
        run_cuda_test(
            "",
            Path::new(INPUT_FOR_CUDA_B), // relative path for input
            &build_dir.join("test_b"),   // relative path for output
            &[
                extra_args.as_slice(),
                &[
                    "-gencode=arch=compute_80,code=[sm_80]".into(),
                    "-gencode=arch=compute_86,code=[compute_86,sm_86]".into(),
                ],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(11),
                preprocessed: Some(3),
                cache_writes: Some(2),
                compilations: Some(5),
                compile_requests: Some(1),
                non_cacheable_compilations: Some(1),
                requests_executed: Some(11),
                cache_hits: Some(vec![
                    (CCompilerKind::CudaFE, Language::CudaFE, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 2),
                    (CCompilerKind::Ptxas, Language::Cubin, 2),
                ]),
                cache_misses: Some(vec![(host_compiler_kind, Language::Cxx, 2)]),
                ..Default::default()
            },
        );
    }

    // Test lto (CUDA 12.8+)
    if !with_debug_flags
        && with_rdc
        && compiler
            .version()
            .and_then(|nvcc_v| {
                version_compare::compare_to(nvcc_v, "12.8.0", version_compare::Cmp::Ge)
                    .map_err(|_| sccache::errors::anyhow!("nvcc too old"))
            })
            .unwrap_or_default()
    {
        trace!("compile A ltoir");
        run_cuda_test(
            "-ltoir",
            Path::new(INPUT_FOR_CUDA_A),   // relative path for input
            &build_dir.join("test.ltoir"), // relative path for output
            &[
                extra_args.as_slice(),
                // needed to make -ltoir work
                &["-dlto".into()],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(3),
                preprocessed: Some(1),
                cache_writes: Some(2),
                compilations: Some(3),
                compile_requests: Some(1),
                requests_executed: Some(3),
                cache_misses: Some(vec![
                    (CCompilerKind::Nvcc, Language::Cuda, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 1),
                ]),
                ..Default::default()
            },
        );
        // Precompile lto_86 so its cache entry potentially has a different .module_id
        // file when hydrated for other multiarch compilations
        trace!("compile A lto_86");
        run_cuda_test(
            "-ltoir",
            &tempdir.join(INPUT_FOR_CUDA_A), // absolute path for input
            &tempdir.join(build_dir).join("test.ltoir"), // absolute path for output
            &[
                extra_args.as_slice(),
                &[
                    "-arch=compute_86".into(),
                    "-code=lto_86".into(),
                    // needed to make -ltoir work
                    "-dlto".into(),
                ],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(3),
                preprocessed: Some(1),
                cache_writes: Some(2),
                compilations: Some(3),
                compile_requests: Some(1),
                requests_executed: Some(3),
                cache_misses: Some(vec![
                    (CCompilerKind::Nvcc, Language::Cuda, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 1),
                ]),
                ..Default::default()
            },
        );

        // Test compiling a multiarch object
        trace!("compile A lto_80,lto_86");
        run_cuda_test(
            "-c",
            Path::new(INPUT_FOR_CUDA_A), // relative path for input
            &build_dir.join(OUTPUT),     // relative path for output
            &[
                extra_args.as_slice(),
                &[
                    "-gencode=arch=compute_80,code=[lto_80]".into(),
                    "-gencode=arch=compute_86,code=[lto_86]".into(),
                ],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(5),
                preprocessed: Some(1),
                cache_writes: Some(3),
                compilations: Some(7),
                compile_requests: Some(1),
                requests_executed: Some(8),
                cache_hits: Some(vec![(CCompilerKind::CudaFE, Language::CudaFE, 1)]),
                cache_misses: Some(vec![
                    (CCompilerKind::Nvcc, Language::Cuda, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 2),
                ]),
                ..Default::default()
            },
        );

        // Test compiling a multiarch object and embedding PTX where the LtoIR for one of the archs is cached
        trace!("compile A [lto_80] [compute_86,lto_86]");
        run_cuda_test(
            "-c",
            Path::new(INPUT_FOR_CUDA_A), // relative path for input
            &build_dir.join(OUTPUT),     // relative path for output
            &[
                extra_args.as_slice(),
                &[
                    "-gencode=arch=compute_80,code=[lto_80]".into(),
                    "-gencode=arch=compute_86,code=[compute_86,lto_86]".into(),
                ],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(5),
                preprocessed: Some(1),
                cache_writes: Some(2),
                compilations: Some(6),
                compile_requests: Some(1),
                requests_executed: Some(8),
                cache_hits: Some(vec![
                    (CCompilerKind::CudaFE, Language::CudaFE, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 1),
                ]),
                cache_misses: Some(vec![
                    (CCompilerKind::Nvcc, Language::Cuda, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 1),
                ]),
                ..Default::default()
            },
        );

        // Test compiling a multiarch object and embedding PTX where the LtoIR for one of the archs is cached
        trace!("compile A arch=lto_80,code=lto_80");
        run_cuda_test(
            "-dc",
            Path::new(INPUT_FOR_CUDA_A), // relative path for input
            Path::new("test.ltoir.o"),   // relative path for output
            &[
                extra_args.as_slice(),
                &["-gencode=arch=lto_80,code=lto_80".into()],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(4),
                preprocessed: Some(1),
                cache_writes: Some(1),
                compilations: Some(4),
                compile_requests: Some(1),
                requests_executed: Some(6),
                cache_hits: Some(vec![
                    (CCompilerKind::CudaFE, Language::CudaFE, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 1),
                ]),
                cache_misses: Some(vec![(CCompilerKind::Nvcc, Language::Cuda, 1)]),
                ..Default::default()
            },
        );
    }

    // Test Blackwell arch modifiers (CUDA 13.1+)
    if compiler
        .version()
        .and_then(|nvcc_v| {
            version_compare::compare_to(nvcc_v, "13.1.0", version_compare::Cmp::Ge)
                .map_err(|_| sccache::errors::anyhow!("nvcc too old"))
        })
        .unwrap_or_default()
    {
        trace!("compile A sm_120f+uumn");
        run_cuda_test(
            "-c",
            Path::new(INPUT_FOR_CUDA_A), // relative path for input
            &build_dir.join(OUTPUT),     // relative path for output
            &[
                extra_args.as_slice(),
                &["-gencode=arch=compute_120f+uumn,code=[compute_120f+uumn,sm_120f+uumn]".into()],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(5),
                preprocessed: Some(1),
                cache_writes: Some(4),
                compilations: Some(7),
                compile_requests: Some(1),
                requests_executed: Some(7),
                cache_misses: Some(vec![
                    (CCompilerKind::CudaFE, Language::CudaFE, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 1),
                    (CCompilerKind::Ptxas, Language::Cubin, 1),
                    (CCompilerKind::Nvcc, Language::Cuda, 1),
                ]),
                ..Default::default()
            },
        );

        trace!("compile A sm_80,sm_120f+uumn");
        run_cuda_test(
            "-c",
            Path::new(INPUT_FOR_CUDA_A), // relative path for input
            &build_dir.join(OUTPUT),     // relative path for output
            &[
                extra_args.as_slice(),
                &[
                    "-gencode=arch=compute_80,code=[sm_80]".into(),
                    "-gencode=arch=compute_120f+uumn,code=[compute_120f+uumn,sm_120f+uumn]".into(),
                ],
            ]
            .concat(),
            AdditionalStats {
                hashed: Some(7),
                preprocessed: Some(1),
                cache_writes: Some(1),
                compilations: Some(5),
                compile_requests: Some(1),
                requests_executed: Some(10),
                cache_hits: Some(vec![
                    (CCompilerKind::CudaFE, Language::CudaFE, 1),
                    (CCompilerKind::Cicc, Language::Ptx, 2),
                    (CCompilerKind::Ptxas, Language::Cubin, 2),
                ]),
                cache_misses: Some(vec![(CCompilerKind::Nvcc, Language::Cuda, 1)]),
                ..Default::default()
            },
        );
    }
}

fn test_nvcc_proper_lang_stat_tracking(
    client: &SccacheClient,
    compiler: &Compiler,
    host_compiler: &Compiler,
    tempdir: &Path,
    with_debug_flags: bool,
    with_rdc: bool,
) {
    let mut stats = ExpectedStats::new(client);

    let mut extra_args = vec![format!("-ccbin={}", host_compiler.exe.to_string_lossy()).into()];

    if with_debug_flags {
        extra_args.push("--device-debug".into());
    }

    if with_rdc {
        extra_args.push("-rdc=true".into());
    }

    extra_args.push("-t=10".into());

    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;

    println!(
        "test_nvcc_proper_lang_stat_tracking: {name}/{}",
        host_compiler.name
    );
    // Compile multiple source files.
    copy_to_tempdir(&[INPUT_FOR_CUDA_C, INPUT], tempdir);

    let out_file = tempdir.join(OUTPUT);

    trace!("compile CUDA C");
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            INPUT_FOR_CUDA_C,
            OUTPUT,
            extra_args.clone(),
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();

    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    fs::remove_file(&out_file).unwrap();

    stats.assert_added(AdditionalStats {
        cache_writes: Some(4),
        hashed: Some(5),
        preprocessed: Some(1),
        compilations: Some(7),
        compile_requests: Some(1),
        requests_executed: Some(7),
        cache_misses: Some(vec![
            (CCompilerKind::Nvcc, Language::Cuda, 1),
            (CCompilerKind::CudaFE, Language::CudaFE, 1),
            (CCompilerKind::Cicc, Language::Ptx, 1),
            (CCompilerKind::Ptxas, Language::Cubin, 1),
        ]),
        ..Default::default()
    });

    trace!("compile CUDA C");
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            INPUT_FOR_CUDA_C,
            OUTPUT,
            extra_args.clone(),
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();

    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    fs::remove_file(&out_file).unwrap();

    stats.assert_added(AdditionalStats {
        hashed: Some(1),
        preprocessed: Some(0),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_hits: Some(vec![(CCompilerKind::Nvcc, Language::Cuda, 1)]),
        ..Default::default()
    });

    trace!("compile C");
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            INPUT,
            OUTPUT,
            extra_args.clone(),
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();

    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    fs::remove_file(&out_file).unwrap();

    stats.assert_added(AdditionalStats {
        cache_writes: Some(1),
        hashed: Some(2),
        preprocessed: Some(1),
        compilations: Some(2),
        compile_requests: Some(1),
        requests_executed: Some(2),
        cache_misses: Some(vec![(CCompilerKind::Nvcc, Language::C, 1)]),
        ..Default::default()
    });

    trace!("compile C");
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            INPUT,
            OUTPUT,
            extra_args.clone(),
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();

    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    fs::remove_file(&out_file).unwrap();

    stats.assert_added(AdditionalStats {
        hashed: Some(1),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_hits: Some(vec![(CCompilerKind::Nvcc, Language::C, 1)]),
        ..Default::default()
    });
}

fn run_sccache_nvcc_cuda_command_tests(
    client: &SccacheClient,
    compiler: &Compiler,
    host_compiler: &Compiler,
    tempdir: &Path,
    with_debug_flags: bool,
    with_rdc: bool,
) {
    test_nvcc_cuda_compiles(
        client,
        compiler,
        host_compiler,
        tempdir,
        with_debug_flags,
        with_rdc,
    );
    test_nvcc_proper_lang_stat_tracking(
        client,
        compiler,
        host_compiler,
        tempdir,
        with_debug_flags,
        with_rdc,
    );
}

fn test_clang_cuda_compiles(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
    with_debug_flags: bool,
    with_rdc: bool,
) {
    let mut stats = ExpectedStats::new(client);

    let mut extra_args = vec![];

    if with_debug_flags {
        extra_args.push("-g".into());
        extra_args.push("--cuda-noopt-device-debug".into());
    }

    if with_rdc {
        extra_args.push("-fgpu-rdc".into());
    }

    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;
    println!("test_clang_cuda_compiles: {name}");
    // Compile multiple source files.
    copy_to_tempdir(&[INPUT_FOR_CUDA_A, INPUT_FOR_CUDA_B], tempdir);

    let out_file = tempdir.join(OUTPUT);
    trace!("compile A");
    client
        .cmd()
        .args(compile_cuda_cmdline(
            name,
            exe,
            "-c",
            INPUT_FOR_CUDA_A,
            OUTPUT,
            &extra_args,
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    fs::remove_file(&out_file).unwrap();
    stats.assert_added(AdditionalStats {
        cache_writes: Some(1),
        hashed: Some(2),
        preprocessed: Some(1),
        compilations: Some(1),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_misses: Some(vec![(CCompilerKind::Clang, Language::Cuda, 1)]),
        ..Default::default()
    });

    trace!("compile A");
    client
        .cmd()
        .args(compile_cuda_cmdline(
            name,
            exe,
            "-c",
            INPUT_FOR_CUDA_A,
            OUTPUT,
            &extra_args,
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    fs::remove_file(&out_file).unwrap();
    stats.assert_added(AdditionalStats {
        hashed: Some(1),
        preprocessed: Some(0),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_hits: Some(vec![(CCompilerKind::Clang, Language::Cuda, 1)]),
        ..Default::default()
    });

    // By compiling another input source we verify that the pre-processor
    // phase is correctly running and outputting text
    trace!("compile B");
    client
        .cmd()
        .args(compile_cuda_cmdline(
            name,
            exe,
            "-c",
            INPUT_FOR_CUDA_B,
            OUTPUT,
            &extra_args,
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    fs::remove_file(&out_file).unwrap();
    stats.assert_added(AdditionalStats {
        cache_writes: Some(1),
        hashed: Some(2),
        preprocessed: Some(1),
        compilations: Some(1),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_misses: Some(vec![(CCompilerKind::Clang, Language::Cuda, 1)]),
        ..Default::default()
    });

    trace!("compile B");
    client
        .cmd()
        .args(compile_cuda_cmdline(
            name,
            exe,
            "-c",
            INPUT_FOR_CUDA_B,
            OUTPUT,
            &extra_args,
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    assert!(fs::metadata(&out_file).map(|m| m.len() > 0).unwrap());
    fs::remove_file(&out_file).unwrap();
    stats.assert_added(AdditionalStats {
        hashed: Some(1),
        preprocessed: Some(0),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_hits: Some(vec![(CCompilerKind::Clang, Language::Cuda, 1)]),
        ..Default::default()
    });
}

fn test_clang_proper_lang_stat_tracking(
    client: &SccacheClient,
    compiler: &Compiler,
    tempdir: &Path,
    with_debug_flags: bool,
    with_rdc: bool,
) {
    let mut stats = ExpectedStats::new(client);

    let mut extra_args = vec![];

    if with_debug_flags {
        extra_args.push("-g".into());
        extra_args.push("--cuda-noopt-device-debug".into());
    }

    if with_rdc {
        extra_args.push("-fgpu-rdc".into());
    }

    let Compiler {
        name,
        exe,
        env_vars,
        ..
    } = compiler;

    println!("test_clang_proper_lang_stat_tracking: {name}");
    // Compile multiple source files.
    copy_to_tempdir(&[INPUT_FOR_CUDA_C, INPUT], tempdir);

    let out_file = tempdir.join(OUTPUT);
    trace!("compile CUDA A");
    client
        .cmd()
        .args(compile_cuda_cmdline(
            name,
            exe,
            "-c",
            INPUT_FOR_CUDA_C,
            OUTPUT,
            &extra_args,
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    fs::remove_file(&out_file).unwrap();
    stats.assert_added(AdditionalStats {
        hashed: Some(2),
        preprocessed: Some(1),
        cache_writes: Some(1),
        compilations: Some(1),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_misses: Some(vec![(CCompilerKind::Clang, Language::Cuda, 1)]),
        ..Default::default()
    });

    trace!("compile CUDA A");
    client
        .cmd()
        .args(compile_cuda_cmdline(
            name,
            exe,
            "-c",
            INPUT_FOR_CUDA_C,
            OUTPUT,
            &extra_args,
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    fs::remove_file(&out_file).unwrap();
    stats.assert_added(AdditionalStats {
        hashed: Some(1),
        preprocessed: Some(0),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_hits: Some(vec![(CCompilerKind::Clang, Language::Cuda, 1)]),
        ..Default::default()
    });

    trace!("compile C++ A");
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            INPUT,
            OUTPUT,
            extra_args.clone(),
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    fs::remove_file(&out_file).unwrap();
    stats.assert_added(AdditionalStats {
        hashed: Some(2),
        preprocessed: Some(1),
        cache_writes: Some(1),
        compilations: Some(1),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_misses: Some(vec![(CCompilerKind::Clang, Language::Cxx, 1)]),
        ..Default::default()
    });

    trace!("compile C++ A");
    client
        .cmd()
        .args(compile_cmdline(
            name,
            exe,
            INPUT,
            OUTPUT,
            extra_args.clone(),
        ))
        .current_dir(tempdir)
        .envs(env_vars.clone())
        .assert()
        .success();
    fs::remove_file(&out_file).unwrap();
    stats.assert_added(AdditionalStats {
        hashed: Some(1),
        preprocessed: Some(0),
        compile_requests: Some(1),
        requests_executed: Some(1),
        cache_hits: Some(vec![(CCompilerKind::Clang, Language::Cxx, 1)]),
        ..Default::default()
    });
}

fn run_sccache_clang_cuda_command_tests(
    client: &SccacheClient,
    compiler: &Compiler,
    _host_compiler: &Compiler,
    tempdir: &Path,
    with_debug_flags: bool,
    with_rdc: bool,
) {
    test_clang_cuda_compiles(client, compiler, tempdir, with_debug_flags, with_rdc);
    test_clang_proper_lang_stat_tracking(client, compiler, tempdir, with_debug_flags, with_rdc);
}

macro_rules! test_cuda_sccache_command {
    ($cuda_compiler:ident, $host_compiler:ident, $cuda_compiler_name:expr, $host_compiler_name:expr) => {
        paste! {
            #[test_case(false, false ; "device_debug=false, rdc=false")]
            #[test_case(false, true ; "device_debug=false, rdc=true")]
            #[test_case(true, false ; "device_debug=true, rdc=false")]
            #[test_case(true, true ; "device_debug=true, rdc=true")]
            fn [<test_cuda_sccache_command_ $cuda_compiler _ $host_compiler>] (
                with_debug_flags: bool,
                with_rdc: bool,
            ) {
                init_logger();
                if let Some(cuda_compiler) = find_cuda_compiler($cuda_compiler_name)
                    && let Some(host_compiler) = find_compiler($host_compiler_name)
                {
                    let (tempdir, client) = make_sccache_client();
                    [<run_sccache_ $cuda_compiler _cuda_command_tests>](
                        &client.start(),
                        &cuda_compiler,
                        &host_compiler,
                        tempdir.path(),
                        with_debug_flags,
                        with_rdc,
                    )
                }
            }
        }
    };
}

// Linux
#[cfg(target_os = "linux")]
test_cuda_sccache_command!(nvcc, gcc, "nvcc", "gcc");
#[cfg(target_os = "linux")]
test_cuda_sccache_command!(nvcc, clang, "nvcc", "clang++");
#[cfg(target_os = "linux")]
test_cuda_sccache_command!(nvcc, nvcxx, "nvcc", "nvc++");
#[cfg(target_os = "linux")]
test_cuda_sccache_command!(clang, clang, "clang++", "clang++");

// Windows
#[cfg(target_env = "msvc")]
test_cuda_sccache_command!(nvcc, cl, "nvcc", "cl");
#[cfg(target_env = "msvc")]
test_cuda_sccache_command!(clang, clang, "clang++", "clang++");
