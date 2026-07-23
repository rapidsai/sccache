// Copyright 2016 Mozilla Foundation
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    compiler::{
        Cacheable, CompileCommandImpl, CompilerArguments,
        args::*,
        c::{
            ArtifactDescriptor, CCompilerImpl, CCompilerKind, DepfilePath, OutDir, ParsedArguments,
            PreprocessorOutput,
        },
        gcc::{self, ArgData::*},
        msvc::from_local_codepage,
    },
    counted_array, dist,
    errors::*,
    mock_command::{CommandCreatorSync, ProcessOutput, RunCommand},
    server::SccacheService,
    util::{
        HASH_BUFFER_SIZE, OsStrExt, SCCACHE_TMPDIR, read_line_batches,
        resolve_compiler_avoiding_wrapper, run_input_output, split_quoted_shell_str, tempdir_in,
    },
};
use async_trait::async_trait;
use futures::{FutureExt, Stream, StreamExt, TryFutureExt};
use itertools::Itertools;
use regex::Regex;
use std::{
    collections::HashMap,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    sync::{Arc, LazyLock},
};
use which::which_in;

static HAS_SM_IN_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| regex_static::regex!(r"^(.*)\.sm_([0-9A-Za-z]+)\.(.*)$"));
static HAS_COMPUTE_IN_NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| regex_static::regex!(r"^(.*)\.compute_([0-9A-Za-z]+)\.(.*)$"));
static ARG_HAS_FILE_WITH_EXTENSION_RE: LazyLock<Regex> =
    LazyLock::new(|| regex_static::regex!(r"-.*=(.*)"));

static NVCC_TMPDIR: LazyLock<std::result::Result<PathBuf, std::io::Error>> = LazyLock::new(|| {
    SCCACHE_TMPDIR
        .as_ref()
        .map(|p| p.join("nvcc"))
        .map_err(|e| e.kind().into())
        .and_then(|tmpdir| {
            std::fs::create_dir_all(&tmpdir)?;
            Ok(tmpdir)
        })
});

/// A unit struct on which to implement `CCompilerImpl`.
#[derive(Clone, Debug)]
pub enum NvccHostCompiler {
    Clang,
    Gcc,
    Msvc,
    Nvhpc,
}

impl NvccHostCompiler {
    fn preprocessor_flag(&self) -> String {
        match self {
            NvccHostCompiler::Msvc => "-P",
            _ => "-E",
        }
        .to_owned()
    }
}

#[derive(Clone, Debug)]
pub struct Nvcc {
    pub archs_all: Vec<String>,
    pub archs_major: Vec<String>,
    pub archs_native: Vec<String>,
    pub host_compiler: NvccHostCompiler,
    pub host_compiler_version: Option<String>,
    pub version: Option<String>,
}

impl Nvcc {
    pub async fn read_implicit_specfiles<T>(
        host_compiler: &NvccHostCompiler,
        creator: &mut T,
        exe: &Path,
        arguments: &[OsString],
        env_vars: &[(OsString, OsString)],
        v_flag: &str,
    ) -> Result<Vec<PathBuf>>
    where
        T: CommandCreatorSync,
    {
        if matches!(host_compiler, NvccHostCompiler::Gcc) {
            gcc::Gcc::read_implicit_specfiles(creator, exe, arguments, env_vars, v_flag).await
        } else {
            Ok(vec![])
        }
    }

    pub async fn read_all_archs<T>(
        creator: &mut T,
        exe: &Path,
        env_vars: &[(OsString, OsString)],
        host_compiler: &NvccHostCompiler,
    ) -> Result<Vec<String>>
    where
        T: CommandCreatorSync,
    {
        Self::read_archs(creator, exe, env_vars, host_compiler, "all").await
    }

    pub async fn read_major_archs<T>(
        creator: &mut T,
        exe: &Path,
        env_vars: &[(OsString, OsString)],
        host_compiler: &NvccHostCompiler,
    ) -> Result<Vec<String>>
    where
        T: CommandCreatorSync,
    {
        Self::read_archs(creator, exe, env_vars, host_compiler, "all-major").await
    }

    pub async fn read_native_archs<T>(
        creator: &mut T,
        exe: &Path,
        env_vars: &[(OsString, OsString)],
        host_compiler: &NvccHostCompiler,
    ) -> Result<Vec<String>>
    where
        T: CommandCreatorSync,
    {
        Self::read_archs(creator, exe, env_vars, host_compiler, "native").await
    }

    async fn read_archs<T>(
        creator: &mut T,
        exe: &Path,
        env_vars: &[(OsString, OsString)],
        host_compiler: &NvccHostCompiler,
        meta_arch: &str,
    ) -> Result<Vec<String>>
    where
        T: CommandCreatorSync,
    {
        let mut env_vars = env_vars.to_vec();
        let mut nvcc_internal_files = HashMap::<String, String>::new();
        select_nvcc_subcommands(
            creator,
            exe,
            Path::new("."),
            &mut env_vars,
            &[
                "-x".into(),
                "cu".into(),
                "-c".into(),
                "x.cu".into(),
                "-o".into(),
                "x.cu.o".into(),
                format!("-arch={meta_arch}").into(),
            ],
            &NvccCompileFlag::Device,
            host_compiler,
            &mut nvcc_internal_files,
            Path::new("x.cu.o"),
            |exe, _| exe == "ptxas",
        )
        .await?
        .into_iter()
        .filter_map(|(_, _, args)| {
            for arg in args {
                if arg.starts_with("-arch=") {
                    return arg
                        .split_once("sm_")
                        .map(|(_, arch)| format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
                        .map(Ok);
                }
            }
            Some(Err(anyhow!(
                "ptxas command didn't include an `-arch=` argument"
            )))
        })
        .try_collect()
    }
}

#[async_trait]
impl CCompilerImpl for Nvcc {
    fn kind(&self) -> CCompilerKind {
        CCompilerKind::Nvcc
    }
    fn plusplus(&self) -> bool {
        false
    }
    fn version(&self) -> Option<String> {
        let host_ver = self
            .host_compiler_version
            .as_ref()
            .map(|host_ver| format!(" ({:?} {host_ver})", &self.host_compiler))
            .unwrap_or_else(|| format!(" ({:?} <unknown>)", &self.host_compiler))
            .to_lowercase();
        self.version
            .as_ref()
            .map(|nvcc_ver| format!("{nvcc_ver}{host_ver}"))
    }
    fn parse_arguments(
        &self,
        arguments: &[OsString],
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
    ) -> CompilerArguments<ParsedArguments> {
        let mut arguments = arguments.to_vec();

        if let Some(flags) = env_vars
            .iter()
            .find(|(k, _)| k == "NVCC_PREPEND_FLAGS")
            .and_then(|(_, p)| p.to_str())
        {
            arguments = split_quoted_shell_str(flags)
                .unwrap_or_default()
                .iter()
                .map(|s| s.clone().into_arg_os_string())
                .chain(arguments.iter().cloned())
                .collect::<Vec<_>>();
        }

        if let Some(flags) = env_vars
            .iter()
            .find(|(k, _)| k == "NVCC_APPEND_FLAGS")
            .and_then(|(_, p)| p.to_str())
        {
            arguments.extend(
                split_quoted_shell_str(flags)
                    .unwrap_or_default()
                    .iter()
                    .map(|s| s.clone().into_arg_os_string()),
            );
        }

        // Rewrite `nvcc -arch=all|all-major|native` to the real client archs,
        // otherwise users who do `-arch=native` with different GPUs will get
        // cache hits for objects that don't include their GPU arch.

        for flag in ["-arch=", "--gpu-architecture="] {
            if let Some(idx) = arguments.iter().position(|x| x.starts_with(flag)) {
                let meta_arch = arguments[idx]
                    .strip_prefix(flag)
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                let real_archs = match meta_arch {
                    "all" => &self.archs_all,
                    "all-major" => &self.archs_major,
                    "native" => &self.archs_native,
                    _ => continue,
                };
                if !real_archs.is_empty() {
                    arguments.splice(idx..(idx + 1), dist::strings_to_osstrings(real_archs));
                }
            }
        }

        for flag in ["-arch", "--gpu-architecture"] {
            if let Some(idx) = arguments.iter().position(|x| x == flag) {
                let meta_arch = arguments
                    .get(idx + 1)
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                let real_archs = match meta_arch {
                    "all" => &self.archs_all,
                    "all-major" => &self.archs_major,
                    "native" => &self.archs_native,
                    _ => continue,
                };
                if !real_archs.is_empty() {
                    arguments.splice(idx..(idx + 2), dist::strings_to_osstrings(real_archs));
                }
            }
        }

        let parsed_args = gcc::parse_arguments(
            &arguments,
            cwd,
            (&gcc::ARGS[..], &ARGS[..]),
            false,
            self.kind(),
        );

        match parsed_args {
            CompilerArguments::Ok(mut parsed_args) => {
                // Drop jobserver flags
                parsed_args.unhashed_args.retain(|arg| {
                    !matches!(arg.to_str(), Some("--jobserver") | Some("-jobserver"))
                });

                let compile_flag = parsed_args.compilation_flag.as_os_str().into();

                if !matches!(compile_flag, NvccCompileFlag::Executable) {
                    // Add the compilation flag to `parsed_args.common_args` so
                    // it's considered when computing the hash.
                    //
                    // Consider the following cases:
                    //  $ sccache nvcc x.cu -o x.bin
                    //  $ sccache nvcc x.cu -o x.cu.o -c
                    //  $ sccache nvcc x.cu -o x.ptx -ptx
                    //  $ sccache nvcc x.cu -o x.cubin -cubin
                    //
                    // The preprocessor output for all four are identical, so
                    // without including the compilation flag in the hasher's
                    // inputs, the same hash would be generated for all four.
                    parsed_args
                        .common_args
                        .push(parsed_args.compilation_flag.clone());
                }

                // Canonicalize here so the absolute path to the input is in the
                // preprocessor output instead of paths relative to `cwd`.
                //
                // Since cicc's input is the post-processed source run through cudafe++'s
                // transforms, its cache key is sensitive to the preprocessor output. The
                // preprocessor embeds the name of the input file in comments, so without
                // canonicalizing here, cicc will get cache misses on otherwise identical
                // inputs that should produce cache hits.
                let input = cwd.join(&parsed_args.input);
                if input.exists() {
                    parsed_args.input = dunce::canonicalize(input).unwrap();
                }

                // Use the object dir as the `out_dir` if the compile flags include `-c` and `-objtemp`
                if matches!(compile_flag, NvccCompileFlag::Device)
                    && parsed_args
                        .unhashed_args
                        .iter()
                        // Search for `-objtemp` or `--objdir-as-tempdir`
                        .any(|arg| arg == "-objtemp" || arg == "--objdir-as-tempdir") &&
                    // Get the output object's parent dir
                    let Some(dir) = parsed_args
                            .outputs
                            .get("obj")
                            .map(|obj| cwd.join(&obj.path))
                            .and_then(|mut p| p.pop().then_some(p))
                            .or_else(|| Some(cwd.to_owned()))
                {
                    // Set out_dir to the parent of the object file
                    parsed_args.out_dir = Some(Arc::new(OutDir::Dir(dir)));
                }
                // Otherwise if `-c` or `-objtemp` weren't passed, set `out_dir` to a tmpdir
                // Don't needlessly create the tempdir if running `parse_arguments()` unit tests
                else if !cfg!(test) {
                    parsed_args.out_dir = Some(Arc::new(OutDir::Tmp(
                        NVCC_TMPDIR
                            .as_ref()
                            .map_err(anyhow::Error::new)
                            .and_then(tempdir_in)
                            .context("Failed to create tempdir")
                            .unwrap(),
                    )));
                }

                CompilerArguments::Ok(parsed_args)
            }
            CompilerArguments::CannotCache(_, _) | CompilerArguments::NotCompilation => parsed_args,
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn preprocess<T>(
        &self,
        service: &SccacheService<T>,
        creator: &T,
        executable: &Path,
        parsed_args: &ParsedArguments,
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
        rewrite_includes_only: bool,
        generate_dependencies: bool,
        _include_line_numbers: bool,
    ) -> Result<PreprocessorOutput>
    where
        T: CommandCreatorSync,
    {
        let mut env_vars = env_vars
            .iter()
            .filter(|(k, _)| k != "NVCC_PREPEND_FLAGS" && k != "NVCC_APPEND_FLAGS")
            .cloned()
            .collect::<Vec<_>>();

        let compile_flag = parsed_args.compilation_flag.as_os_str().into();

        // Only return dependencies if requested, and only for compilations that don't produce executables
        let dependencies = if (generate_dependencies || !parsed_args.dependency_args.is_empty())
            && !matches!(compile_flag, NvccCompileFlag::Executable)
        {
            self.generate_dependencies(creator, executable, parsed_args, cwd, &env_vars)
                .await?
                .map(|depfile| {
                    gcc::parse_dependencies(cwd.to_owned(), parsed_args.input.clone(), depfile)
                        .map_ok(|dependencies| {
                            dependencies
                                .into_iter()
                                .sorted()
                                .unique()
                                .collect::<Vec<_>>()
                        })
                        .boxed()
                })
        } else {
            None
        };

        // Use `nvcc --dryrun` to extract all the host and device preprocessor commands.
        // Run each preprocessor command, hash the output, then concatenate all the hashes.
        // We don't care what the preprocessor output is (we can't compile it), we care that
        // it's unique to this input.
        //
        // We have to run all of them, because `nvcc -E` isn't sufficient to capture all
        // possible modifications:
        // ```
        //
        // /* host side */
        // #ifndef __CUDA_ARCH__
        // static_assert(false, "not nvcc")
        // #endif
        //
        // /* device side */
        // __global__ void gpu_kernel() {
        //   #if __CUDA_ARCH__ >= 700
        //   sm70_logic();
        //   #elseif __CUDA_ARCH__ >= 600
        //   sm60_logic();
        //   #else
        //   fallback_logic();
        //   #endif
        // }
        // ```
        //
        // See https://github.com/mozilla/sccache/issues/2299

        let output = {
            let (arguments, output_path, _, _) = parsed_to_nvcc_args(cwd, parsed_args);

            let mut nvcc_internal_files = HashMap::<String, String>::new();
            let preprocessor_flag = self.host_compiler.preprocessor_flag();

            // Gather the nvcc preprocessor subcommands
            // * one preprocessor call for the host (CPU) code
            // * one preprocessor call for device (GPU) code, per device architecture defined by the user
            trace!("[{}]: nvcc preprocess", parsed_args.output_pretty());
            let commands = select_nvcc_subcommands(
                creator,
                executable,
                cwd,
                &mut env_vars,
                &[
                    &arguments,
                    &[
                        "--keep-dir".into(),
                        parsed_args.out_dir.as_ref().as_ref().unwrap().path().into(),
                    ][..],
                ]
                .concat(),
                &compile_flag,
                &self.host_compiler,
                &mut nvcc_internal_files,
                &output_path,
                |_, args| args.contains(&preprocessor_flag),
            )
            .await?
            .into_iter()
            .map(|(_, exe, args)| (exe, dist::strings_to_osstrings(&args)))
            .collect::<Vec<_>>();

            // Get the CCompiler impl for each preprocessor subcommand
            let commands_and_compilers = preprocessor_commands_and_compilers(
                service.to_owned(),
                env_vars.clone(),
                cwd.to_owned(),
                self.host_compiler.clone(),
                commands,
            );

            // Run the `compiler.preprocess()` method on each CCompiler impl and return a `Stream<Stream<Result<Bytes, ProcessError>>>`
            let outputs = preprocess_for_compilers(
                service.to_owned(),
                creator.to_owned(),
                env_vars.clone(),
                parsed_args.clone(),
                cwd.to_owned(),
                rewrite_includes_only,
                commands_and_compilers,
            );

            // Flatten the `Stream<Stream<Result<Bytes, ProcessError>>>` -> `Stream<Result<Bytes, ProcessError>>`
            flatten_preprocessor_outputs(outputs).boxed()
        };

        Ok(PreprocessorOutput::Output(output, dependencies))
    }

    async fn generate_dependencies<T>(
        &self,
        creator: &T,
        executable: &Path,
        parsed_args: &ParsedArguments,
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
    ) -> Result<Option<DepfilePath>>
    where
        T: CommandCreatorSync,
    {
        gcc::generate_dependencies(creator, executable, parsed_args, cwd, env_vars, self.kind())
            .await
            .map(Some)
    }

    fn generate_compile_commands(
        &self,
        _path_transformer: &mut dist::PathTransformer,
        executable: &Path,
        parsed_args: &ParsedArguments,
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
        _rewrite_includes_only: bool,
        hash_key: &str,
    ) -> Result<(
        impl CompileCommandImpl,
        Option<dist::CompileCommand>,
        Cacheable,
    )> {
        generate_compile_commands(
            parsed_args,
            executable,
            cwd,
            env_vars,
            &self.host_compiler,
            hash_key,
        )
    }
}

fn preprocessor_commands_and_compilers<T>(
    service: SccacheService<T>,
    env_vars: Vec<(OsString, OsString)>,
    cwd: PathBuf,
    host_compiler: NvccHostCompiler,
    commands: Vec<(PathBuf, Vec<OsString>)>,
) -> impl Stream<
    Item = Result<(
        PathBuf,
        Vec<OsString>,
        Option<PathBuf>,
        Box<dyn std::any::Any + Send + Sync>,
    )>,
>
where
    T: CommandCreatorSync,
{
    async_stream::try_stream! {
        for (exe, args) in commands {
            yield preprocessor_command_and_compiler(
                &service,
                &env_vars,
                &cwd,
                &host_compiler,
                exe,
                args
            ).await?
        }
    }
}

async fn preprocessor_command_and_compiler<T>(
    service: &SccacheService<T>,
    env_vars: &[(OsString, OsString)],
    cwd: &Path,
    host_compiler: &NvccHostCompiler,
    exe: PathBuf,
    mut args: Vec<OsString>,
) -> Result<(
    PathBuf,
    Vec<OsString>,
    Option<PathBuf>,
    Box<dyn std::any::Any + Send + Sync>,
)>
where
    T: CommandCreatorSync,
{
    let compiler = service.compiler_info(&exe, cwd, &args, env_vars).await?;

    let output = parse_args_simple(args.iter().cloned(), cwd)
        .outputs
        .get_mut("obj")
        .map(|o| std::mem::take(&mut o.path));

    // Fix the args so they look like a compilation to `compiler.parse_args`
    let mut has_compile_flag = false;

    if matches!(host_compiler, NvccHostCompiler::Msvc) {
        for arg in args.iter_mut() {
            let bs = arg.as_encoded_bytes();

            match bs.len() {
                0 | 1 => {}
                2 if matches!(&bs[0..1], b"-" | b"/") => {
                    match &bs[1..2] {
                        b"P" => {
                            has_compile_flag = true;
                            // Replace /P -> /c
                            let mut bs = bs.to_vec();
                            bs[1] = b'c';
                            *arg = unsafe { OsString::from_encoded_bytes_unchecked(bs) };
                            continue;
                        }
                        b"c" => {
                            has_compile_flag = true;
                        }
                        _ => {}
                    }
                }
                _ if matches!(&bs[0..1], b"-" | b"/") => {
                    if matches!(&bs[1..3], b"Fi") {
                        // Rename /Fi to /Fo
                        let mut bs = bs.to_vec();
                        bs[2] = b'o';
                        *arg = unsafe { OsString::from_encoded_bytes_unchecked(bs) };
                        continue;
                    }
                }
                _ => {}
            }
        }
        // Remove /FS
        args.retain(|arg| arg != "/FS" && arg != "-FS");
    } else {
        for arg in args.iter_mut() {
            if arg == "-E" {
                // Replace -E -> -c (GCC/Clang/NVHPC)
                *arg = "-c".into();
            } else if arg == "-c" {
                has_compile_flag = true;
            }
        }
    }

    if !has_compile_flag {
        // add -c to ensure parse_arguments() thinks its a compilation
        args.push("-c".into());
    }

    Ok((
        exe,
        args,
        output,
        // Insane hack, see below
        compiler.into_any(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn preprocess_for_compilers<T, S>(
    service: SccacheService<T>,
    creator: T,
    env_vars: Vec<(OsString, OsString)>,
    parsed_args: ParsedArguments,
    cwd: PathBuf,
    rewrite_includes_only: bool,
    commands_and_compilers: S,
) -> impl Stream<Item = Result<PreprocessorOutput>>
where
    T: CommandCreatorSync,
    S: Stream<
        Item = Result<(
            PathBuf,
            Vec<OsString>,
            Option<PathBuf>,
            Box<dyn std::any::Any + Send + Sync>,
        )>,
    >,
{
    async_stream::try_stream! {
        for await command_and_compiler in commands_and_compilers {

            let (exe, args, output, compiler) = command_and_compiler?;

            yield preprocess_for_compiler(
                &service,
                &creator,
                &env_vars,
                &cwd,
                rewrite_includes_only,
                &exe,
                &args,
                output,
                compiler,
            ).await
                .map_err(|err| {
                    debug!("[{}]: {err:?}", parsed_args.output_pretty());
                    err
                })?;
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn preprocess_for_compiler<T>(
    service: &SccacheService<T>,
    creator: &T,
    env_vars: &[(OsString, OsString)],
    cwd: &Path,
    rewrite_includes_only: bool,
    exe: &Path,
    args: &[OsString],
    output: Option<PathBuf>,
    compiler: Box<dyn std::any::Any + Send + Sync>,
) -> Result<PreprocessorOutput>
where
    T: CommandCreatorSync,
{
    use crate::compiler::c::CCompiler;
    use crate::compiler::clang::Clang;
    use crate::compiler::gcc::Gcc;
    use crate::compiler::msvc::Msvc;
    use crate::compiler::nvhpc::Nvhpc;

    macro_rules! parse_args_and_preprocess {
        ($( $klass:ident ),* ) => {{
            futures::future::err(anyhow!("Failed to preprocess for host compiler {exe:?}"))
            $(
                .or_else(|err| async {
                    if let Some(c) = compiler.downcast_ref::<CCompiler<$klass>>().map(|c| c.compiler()) {
                        let mut parsed_arguments = match c.parse_arguments(&args, &cwd, &env_vars) {
                            CompilerArguments::Ok(args) => args,
                            err => bail!("Failed to parse arguments: {exe:?} {args:?}\n{err:?}"),
                        };
                        c.preprocess(
                            &service,
                            &creator,
                            &exe,
                            &mut parsed_arguments,
                            &cwd,
                            &env_vars,
                            rewrite_includes_only,
                            false, // generate_dependencies
                            true, // include_line_numbers
                        )
                        .await
                    } else {
                        Err(err)
                    }
                })
            )*
        }}
    }

    // This is a super wild hack, but it works.
    //
    // Downcast the detected Box<Compiler<_>> into one of its concrete implementations
    // so we can call its `parse_arguments()` and `preprocess()` functions directly.
    //
    // The Compiler<_> trait is implemented for both C and Rust. It doesn't provide
    // a function to preprocess the source code, because that isn't how Rust works.
    // Instead, it parses valid arguments into a CompilerHasher<_>, which provides
    // a function to generate a hash key *somehow*.
    //
    // Both CCompiler and CCompilerHasher are generic over a concrete CCompilerImpl
    // type, because CCompilerImpl has async functions with generic parameters and
    // can't be safely made into a trait object (i.e. `CCompiler<dyn CCompilerImpl>`
    // doens't work).
    //
    // To get around this, we attempt to downcast Box<Compiler<_>> to each of the
    // known concrete host compiler implementations and use the one that succeeds,
    // `Box<CCompiler<Gcc>> || Box<CCompiler<Clang>> || Box<CCompiler<Msvc>` etc.
    //
    // Note: This list must be expanded if support for additional nvcc host
    // compilers is added in the future.
    match parse_args_and_preprocess!(Gcc, Clang, Nvhpc, Msvc).await? {
        PreprocessorOutput::File(path, deps) => Ok(PreprocessorOutput::File(path, deps)),
        PreprocessorOutput::Output(src, deps) => {
            if let Some(output) = output {
                let src = Box::pin(async_stream::try_stream! {
                    use tokio::{
                        fs::File,
                        io::{AsyncWriteExt, BufWriter},
                    };
                    let mut out = BufWriter::new(File::create(output).await?);
                    for await bytes in src {
                        let bytes = bytes?;
                        out.write_all(&bytes[..]).await?;
                        yield bytes;
                    }
                    out.flush().await?;
                });
                Ok(PreprocessorOutput::Output(src, deps))
            } else {
                Ok(PreprocessorOutput::Output(src, deps))
            }
        }
    }
}

fn flatten_preprocessor_outputs<S>(
    preprocessor_outputs: S,
) -> impl Stream<Item = Result<bytes::BytesMut>>
where
    S: Stream<Item = Result<PreprocessorOutput>>,
{
    async_stream::try_stream! {
        for await preprocessor_output in preprocessor_outputs {
            match preprocessor_output? {
                PreprocessorOutput::File(path, _) => {
                    let src = tokio::fs::File::open(path).await?;
                    let src = read_line_batches(src, HASH_BUFFER_SIZE);
                    for await bytes in src {
                        yield bytes?;
                    }
                }
                PreprocessorOutput::Output(src, _) => {
                    for await bytes in src {
                        yield bytes?;
                    }
                },
            }
        }
    }
}

fn generate_compile_commands(
    parsed_args: &ParsedArguments,
    executable: &Path,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
    host_compiler: &NvccHostCompiler,
    _hash_key: &str,
) -> Result<(NvccCompileCommand, Option<dist::CompileCommand>, Cacheable)> {
    let env_vars = env_vars
        .iter()
        .filter(|(k, _)| k != "NVCC_PREPEND_FLAGS" && k != "NVCC_APPEND_FLAGS")
        .cloned()
        .collect::<Vec<_>>();

    let (arguments, output_path, keep_dir, num_parallel) = parsed_to_nvcc_args(cwd, parsed_args);
    let compile_flag = parsed_args.compilation_flag.as_os_str().into();
    let out_dir = parsed_args.out_dir.as_ref().unwrap().clone();

    // Don't cache compilations that produce executables.
    let cacheable = if compile_flag != NvccCompileFlag::Executable {
        Cacheable::Yes
    } else {
        Cacheable::No
    };

    let command = NvccCompileCommand {
        out_dir,
        keep_dir,
        num_parallel,
        executable: executable.to_owned(),
        arguments,
        compile_flag,
        env_vars,
        cwd: cwd.to_owned(),
        host_compiler: host_compiler.clone(),
        // Only here so we can include it in logs
        output_path,
    };

    Ok((command, None, cacheable))
}

fn parsed_to_nvcc_args(
    cwd: &Path,
    parsed_args: &ParsedArguments,
) -> (
    Vec<OsString>,   // nvcc arguments
    PathBuf,         // output path
    Option<PathBuf>, // keep_dir
    usize,           // num_parallel
) {
    let mut unhashed_args = parsed_args.unhashed_args.clone();

    let keep_dir = {
        let mut keep = false;
        let mut out_dir = None;
        let mut keep_dir = None;
        // Remove all occurrences of `-keep|-keep-dir|-objtemp`, but
        // remember `--keep-dir` so we can copy the files to it later
        loop {
            if let Some(idx) = unhashed_args
                .iter()
                .position(|x| x == "-keep-dir" || x == "--keep-dir")
            {
                let dir = PathBuf::from(unhashed_args[idx + 1].as_os_str());
                let dir = cwd.join(dir);
                unhashed_args.splice(idx..(idx + 2), []);
                keep_dir = Some(dir);
                continue;
            } else if let Some(idx) = unhashed_args
                .iter()
                .position(|x| x == "-objtemp" || x == "--objdir-as-tempdir")
            {
                out_dir = parsed_args.out_dir.as_deref();
                unhashed_args.splice(idx..(idx + 1), []);
                continue;
            } else if let Some(idx) = unhashed_args.iter().position(|x| {
                x == "-keep" || x == "--keep" || x == "-save-temps" || x == "--save-temps"
            }) {
                keep = true;
                unhashed_args.splice(idx..(idx + 1), []);
                continue;
            }
            break;
        }
        // Match nvcc behavior where intermediate files are kept if:
        // * Only `-keep` is specified (files copied to cwd or `-objtemp`)
        // * Both `-keep -keep-dir=<dir>` are specified (files copied to <dir>)
        // nvcc does _not_ keep intermediate files if `-keep-dir=` is specified without `-keep`
        keep.then(|| {
            keep_dir
                .or_else(|| out_dir.map(|dir| dir.path().to_owned()))
                .or_else(|| Some(cwd.to_path_buf()))
        })
        .flatten()
    };

    let num_parallel = {
        let mut num_parallel = 1;
        // Remove all occurrences of `-t=` or `--threads` because it's incompatible with --dryrun
        // Prefer the last occurrence of `-t=` or `--threads` to match nvcc behavior
        loop {
            if let Some(idx) = unhashed_args.iter().position(|x| x.starts_with("-t")) {
                let arg = unhashed_args.get(idx);
                if let Some(arg) = arg.and_then(|arg| arg.to_str()) {
                    let range = if arg.contains('=') {
                        3..arg.len()
                    } else {
                        2..arg.len()
                    };
                    if let Ok(arg) = arg[range].parse::<usize>() {
                        num_parallel = arg;
                    }
                }
                unhashed_args.splice(idx..(idx + 1), []);
                continue;
            }
            if let Some(idx) = unhashed_args.iter().position(|x| x == "--threads") {
                let arg = unhashed_args.get(idx + 1);
                if let Some(arg) = arg.and_then(|arg| arg.to_str())
                    && let Ok(arg) = arg.parse::<usize>()
                {
                    num_parallel = arg;
                }
                unhashed_args.splice(idx..(idx + 2), []);
                continue;
            }
            break;
        }
        num_parallel
    };

    let mut args = vec![];

    if let Some(lang) = parsed_args
        .language
        .as_compiler_str(CCompilerKind::Nvcc.into())
    {
        args.extend(vec!["-x".into(), lang.into()]);
    }

    let output = &parsed_args
        .outputs
        .get("obj")
        .context("Missing object file output")
        .unwrap()
        .path;

    args.extend_from_slice(&[
        "-o".into(),
        // Canonicalize the output path if the compile flag indicates we won't
        // produce an object file. Since we run cicc and ptxas in a temp dir,
        // but we run the host compiler in `cwd` (the dir from which sccache was
        // executed), cicc/ptxas `-o` argument should point at the real out path
        // that's potentially relative to `cwd`.
        if matches!(
            parsed_args.compilation_flag.as_os_str().into(),
            NvccCompileFlag::Device | NvccCompileFlag::Executable
        ) {
            output.clone().into()
        } else {
            cwd.join(output).into()
        },
    ]);

    args.extend_from_slice(&parsed_args.preprocessor_args);
    args.extend_from_slice(&unhashed_args);
    args.extend_from_slice(&parsed_args.common_args);
    args.extend_from_slice(&parsed_args.arch_args);
    if parsed_args.double_dash_input {
        args.push("--".into());
    }

    args.push(parsed_args.input.as_path().into());

    (args, output.clone(), keep_dir, num_parallel.max(1))
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum NvccCompileFlag {
    Cubin,
    Device,
    Executable,
    Fatbin,
    LtoIR,
    OptixIR,
    Ptx,
    TileBC,
}

impl From<&OsStr> for NvccCompileFlag {
    fn from(flag: &OsStr) -> Self {
        match flag.to_str() {
            // compile to executable
            Some("") => NvccCompileFlag::Executable,
            // compile to object
            Some("-c") | Some("--compile")
            // compile to object with -rdc=true
            | Some("-dc") | Some("--device-c")
            // compile to object with -rdc=false
            | Some("-dw") | Some("--device-w")
            => NvccCompileFlag::Device,
            Some("-cubin") | Some("--cubin")
            | Some("-tilecubin") | Some("--tilecubin")
            => NvccCompileFlag::Cubin,
            Some("-fatbin") | Some("--fatbin")
            | Some("-tilefatbin") | Some("--tilefatbin")
            => NvccCompileFlag::Fatbin,
            Some("-ltoir") | Some("--ltoir") => NvccCompileFlag::LtoIR,
            Some("-optix-ir") | Some("--optix-ir") => NvccCompileFlag::OptixIR,
            Some("-ptx") | Some("--ptx") => NvccCompileFlag::Ptx,
            Some("-tilebc") | Some("--tilebc") => NvccCompileFlag::TileBC,
            _ => unreachable!()
        }
    }
}

#[derive(Clone, Debug)]
struct NvccCompileCommand {
    out_dir: Arc<OutDir>,
    keep_dir: Option<PathBuf>,
    num_parallel: usize,
    executable: PathBuf,
    arguments: Vec<OsString>,
    compile_flag: NvccCompileFlag,
    env_vars: Vec<(OsString, OsString)>,
    cwd: PathBuf,
    host_compiler: NvccHostCompiler,
    output_path: PathBuf,
}

#[async_trait]
impl CompileCommandImpl for NvccCompileCommand {
    fn get_executable(&self) -> PathBuf {
        self.executable.clone()
    }
    fn get_arguments(&self) -> Vec<OsString> {
        self.arguments.clone()
    }
    fn get_env_vars(&self) -> Vec<(OsString, OsString)> {
        self.env_vars.clone()
    }
    fn get_cwd(&self) -> PathBuf {
        self.cwd.clone()
    }

    async fn execute<T>(&self, service: &SccacheService<T>, creator: &T) -> Result<ProcessOutput>
    where
        T: CommandCreatorSync,
    {
        let NvccCompileCommand {
            out_dir,
            keep_dir,
            executable,
            arguments,
            compile_flag,
            env_vars,
            cwd,
            host_compiler,
            num_parallel,
            output_path,
            ..
        } = self;

        let pending = service.increment_pending_compilations();

        let (mut nvcc_internal_files, nvcc_subcommand_groups) =
            group_nvcc_subcommands_by_compilation_stage(
                creator,
                executable,
                arguments,
                compile_flag,
                cwd,
                out_dir.path(),
                keep_dir.clone(),
                env_vars,
                host_compiler,
                output_path,
            )
            .await?;

        drop(pending);

        let maybe_keep_temps_then_clean = || async move {
            let _active = service.increment_active_compilations();

            // Move and/or delete nvcc's internal files.
            //
            // If the caller passed `-keep` or `-keep-dir`, copy the internal
            // files to the requested location. We do this because we override
            // `-keep` and `-keep-dir` in our `nvcc --dryrun` call.
            //
            // Renames the files back to the original names nvcc gave them.

            // Move nvcc internal files to `--keep-dir`, but only if it isn't the same as `out_dir`
            if let Some(dst_dir) = keep_dir.as_deref() {
                let src_dir = out_dir.as_ref().path();

                // Ensure `--keep-dir` exists
                tokio::fs::create_dir_all(dst_dir).await?;

                // Rename files back to original nvcc names
                for (old, new) in nvcc_internal_files.drain() {
                    let src = src_dir.join(new);
                    let dst = dst_dir.join(old);
                    if dst != src && src.exists() {
                        match tokio::fs::rename(&src, &dst).await {
                            Ok(_) => {}
                            Err(err) => match err.kind() {
                                std::io::ErrorKind::CrossesDevices => {
                                    tokio::fs::copy(&src, &dst).await?;
                                    tokio::fs::remove_file(&src).await?;
                                }
                                _ => bail!(err),
                            },
                        }
                    }
                }

                return Ok(());
            }

            // If `-objtemp` was passed without `-keep`, remove the nvcc internal files from `out_dir`
            if let OutDir::Dir(obj_dir) = out_dir.as_ref() {
                for path in nvcc_internal_files
                    .drain()
                    .flat_map(|(old, new)| [obj_dir.join(old), obj_dir.join(new)])
                    .filter(|path| path.exists())
                {
                    tokio::fs::remove_file(path).await?;
                }
            }

            Ok(())
        };

        let mut output = ProcessOutput::default();

        let n = nvcc_subcommand_groups.len();
        let cuda_front_end_range = if n > 0 { 0..1 } else { 0..0 };
        let device_compile_range = if n > 2 { 1..n - 1 } else { 0..0 };

        let num_parallel = device_compile_range.len().min(*num_parallel).max(1);

        let run_command_groups =
            |mut output: ProcessOutput, chunks: Vec<Vec<Vec<NvccGeneratedSubcommand>>>| async {
                for chunk in chunks {
                    let results = futures::future::join_all(chunk.into_iter().map(|commands| {
                        run_nvcc_subcommands_group(service, creator, commands, output_path)
                    }))
                    .await;

                    for result in results {
                        output = aggregate_output(output, result.unwrap_or_else(error_to_output));
                    }

                    if !output.success() {
                        output.stdout.shrink_to_fit();
                        output.stderr.shrink_to_fit();
                        return Err(anyhow!(ProcessError(output)));
                    }
                }
                Ok(output)
            };

        let mut cudafe_commands_group = nvcc_subcommand_groups;
        let mut device_compile_groups = cudafe_commands_group.split_off(cuda_front_end_range.len());
        let final_host_compiler_group = device_compile_groups.split_off(device_compile_range.len());

        for command_groups in [
            // Run the `cudafe++` command group first
            if cudafe_commands_group.is_empty() {
                vec![]
            } else {
                vec![cudafe_commands_group]
            },
            // Compile multiple device architectures in parallel when `nvcc -t=N` is specified
            {
                let mut chunks = vec![];
                let mut head = device_compile_groups;
                loop {
                    if head.is_empty() {
                        break;
                    }
                    let tail = head.split_off(head.len().min(num_parallel));
                    chunks.push(head);
                    head = tail;
                }
                chunks
            },
            // Run the final host compile and link command group
            if final_host_compiler_group.is_empty() {
                vec![]
            } else {
                vec![final_host_compiler_group]
            },
        ] {
            if !command_groups.is_empty() {
                output = match run_command_groups(output, command_groups).await {
                    Ok(output) => output,
                    Err(err) => return maybe_keep_temps_then_clean().await.and(Err(err)),
                };
            }
        }

        output.stdout.shrink_to_fit();
        output.stderr.shrink_to_fit();
        maybe_keep_temps_then_clean().await?;
        Ok(output)
    }
}

#[derive(Debug)]
enum NvccSubcommandKind {
    Preprocess,
    Compile,
    Other,
}

#[derive(Debug)]
struct NvccGeneratedSubcommand {
    exe: PathBuf,
    args: Vec<OsString>,
    cwd: PathBuf,
    kind: NvccSubcommandKind,
    env_vars: Vec<(OsString, OsString)>,
    output: Option<PathBuf>,
}

impl std::fmt::Display for NvccGeneratedSubcommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, r#"cd "{}"; "#, self.cwd.display())?;
        if cfg!(target_os = "windows") {
            write!(f, "& ")?;
        }
        write!(f, r#""{}""#, self.exe.display())?;
        for arg in self.args.iter() {
            write!(f, " {arg:?}")?;
        }
        std::fmt::Result::Ok(())
    }
}
fn is_nvcc_exe(exe: &str, _: &[String]) -> bool {
    matches!(
        exe,
        "cudafe++" | "cicc" | "ptxas" | "tileiras" | "nvlink" | "fatbinary"
    )
}

#[allow(clippy::too_many_arguments)]
async fn group_nvcc_subcommands_by_compilation_stage<T>(
    creator: &T,
    executable: &Path,
    arguments: &[OsString],
    compile_flag: &NvccCompileFlag,
    cwd: &Path,
    out: &Path,
    keep_dir: Option<PathBuf>,
    env_vars: &[(OsString, OsString)],
    host_compiler: &NvccHostCompiler,
    output_path: &Path,
) -> Result<(HashMap<String, String>, Vec<Vec<NvccGeneratedSubcommand>>)>
where
    T: CommandCreatorSync,
{
    // Run `nvcc --dryrun` twice to ensure the commands are correct
    // relative to the directory where they're run.
    //
    // All the "nvcc" commands (cudafe++, cicc, ptxas, nvlink, fatbinary)
    // are run in the out dir, so their arguments should be relative to
    // the out dir, e.g. `cudafe++ [...] "x.cpp4.ii"`
    //
    // All the host compiler invocations are run in the original `cwd` where
    // sccache was invoked. Arguments will be relative to the cwd, except
    // any arguments that reference nvcc-generated files should be absolute
    // to the out dir, e.g. `gcc -E [...] x.cu -o /out/dir/x.cpp4.ii`

    // Roughly equivalent to:
    // ```shell
    //   cat <(nvcc --dryrun --keep                                       \
    //       | nl -n ln -s ' ' -w 1                                       \
    //       | grep -P    "^[0-9]+ (cicc|ptxas|cudafe|nvlink|fatbinary)") \
    //                                                                    \
    //       <(nvcc --dryrun --keep --keep-dir /out/dir                   \
    //       | nl -n ln -s ' ' -w 1                                       \
    //       | grep -P -v "^[0-9]+ (cicc|ptxas|cudafe|nvlink|fatbinary)") \
    //                                                                    \
    //   | sort -k 1n
    // ```

    let mut env_vars_1 = env_vars.to_vec();
    let mut env_vars_2 = env_vars.to_vec();

    let mut nvcc_internal_files = HashMap::<String, String>::new();

    // Get the nvcc compile command lines with paths relative to `out`
    let nvcc_commands = select_nvcc_subcommands(
        creator,
        executable,
        cwd,
        &mut env_vars_1,
        arguments,
        compile_flag,
        host_compiler,
        &mut nvcc_internal_files,
        output_path,
        is_nvcc_exe,
    )
    .await?;

    // Get the host compile command lines with paths relative to `cwd` and absolute paths to `out`
    let host_commands = select_nvcc_subcommands(
        creator,
        executable,
        cwd,
        &mut env_vars_2,
        &[arguments, &["--keep-dir".into(), out.into()][..]].concat(),
        compile_flag,
        host_compiler,
        &mut nvcc_internal_files,
        output_path,
        |exe, args| !is_nvcc_exe(exe, args),
    )
    .await?;

    drop(env_vars_2);
    let env_vars = env_vars_1;

    // Merge the nvcc and host compiler commands into one list
    let (cudafe_has_gen_module_id_file_flag, all_commands) =
        merge_nvcc_and_host_compiler_commands(cwd, out, nvcc_commands, host_commands);

    // Create groups of commands that should be run sequential relative to each other,
    // but can optionally be run in parallel to other groups if the user requested via
    // `nvcc --threads`.
    let command_groups = create_nvcc_commands_graph(
        compile_flag,
        keep_dir,
        env_vars,
        host_compiler,
        cudafe_has_gen_module_id_file_flag,
        all_commands,
        output_path,
    );

    Ok((nvcc_internal_files, command_groups))
}

fn merge_nvcc_and_host_compiler_commands(
    cwd: &Path,
    out: &Path,
    nvcc_commands: Vec<(usize, PathBuf, Vec<String>)>,
    host_commands: Vec<(usize, PathBuf, Vec<String>)>,
) -> (bool, Vec<(PathBuf, PathBuf, Vec<String>)>) {
    let gen_module_id_file_flag = "--gen_module_id_file".to_owned();
    let module_id_file_name_flag = "--module_id_file_name".to_owned();

    let mut cudafe_has_gen_module_id_file_flag = false;

    // Now zip the two lists of commands again by sorting on original line index.
    // Transform to tuples that include the dir in which each command should run.
    let mut all_commands = nvcc_commands
        .iter()
        // Run cudafe++, nvlink, cicc, ptxas, tileiras, and fatbinary in `out`
        .map(|(idx, exe, args)| (idx, out, exe, args))
        .chain(
            host_commands
                .iter()
                // Run host preprocessing and compilation steps in `cwd`
                .map(|(idx, exe, args)| (idx, cwd, exe, args)),
        )
        .sorted_by(|a, b| Ord::cmp(&a.0, &b.0))
        .map(|(_, dir, exe, args)| (dir.to_owned(), exe.to_owned(), args.to_owned()))
        .collect::<Vec<_>>();

    // First pass over commands because in CTK < 12.0, `cudafe++` is at the end of the commands list,
    // but we need to set `cudafe_has_gen_module_id_file_flag` in order to adjust the cicc commands.
    for (_dir, exe, args) in all_commands.iter_mut() {
        if let Some("cudafe++") = exe.file_stem().and_then(|s| s.to_str()) {
            // Fix for CTK < 12.0:
            // Add `--gen_module_id_file` if the cudafe++ args include `--module_id_file_name`
            if let Some(idx) = args.iter().position(|x| x == &module_id_file_name_flag)
                && !args.contains(&gen_module_id_file_flag)
            {
                // Insert `--gen_module_id_file` just before `--module_id_file_name` to match nvcc behavior
                args.splice(idx..idx, [gen_module_id_file_flag.clone()]);
            }
            cudafe_has_gen_module_id_file_flag = args.contains(&gen_module_id_file_flag);
        }
    }

    (cudafe_has_gen_module_id_file_flag, all_commands)
}

fn create_nvcc_commands_graph(
    compile_flag: &NvccCompileFlag,
    keep_dir: Option<PathBuf>,
    env_vars: Vec<(OsString, OsString)>,
    host_compiler: &NvccHostCompiler,
    cudafe_has_gen_module_id_file_flag: bool,
    all_commands: Vec<(PathBuf, PathBuf, Vec<String>)>,
    output_path: &Path,
) -> Vec<Vec<NvccGeneratedSubcommand>> {
    // Create groups of commands that should be run sequential relative to each other,
    // but can optionally be run in parallel to other groups if the user requested via
    // `nvcc --threads`.

    let preprocessor_flag = OsString::from(host_compiler.preprocessor_flag());

    let mut cuda_front_end_group = Vec::<(ParsedArguments, NvccGeneratedSubcommand)>::new();
    let mut final_assembly_group = Vec::<(ParsedArguments, NvccGeneratedSubcommand)>::new();
    let mut device_compile_groups =
        HashMap::<PathBuf, Vec<(ParsedArguments, NvccGeneratedSubcommand)>>::new();

    for (dir, exe, args) in all_commands.iter() {
        let cmd = exe.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        if let Some((kind, group, parsed_args, env_vars)) = match cmd {
            "" => continue,
            // cudafe++ _must be_ cached, because the `.module_id` file is unique to each invocation (new in CTK 12.8)
            "cudafe++" => Some((
                NvccSubcommandKind::Compile,
                &mut cuda_front_end_group,
                parse_args_simple(args.iter().map(Into::into), dir),
                env_vars.clone(),
            )),
            // fatbinary and nvlink are not cacheable
            "fatbinary" | "nvlink" => Some((
                NvccSubcommandKind::Other,
                &mut final_assembly_group,
                parse_args_simple(args.iter().map(Into::into), dir),
                env_vars.clone(),
            )),
            // cicc, ptxas, and tileiras are cacheable
            "cicc" => Some(parse_args_simple(args.iter().map(Into::into), dir))
                .map(|parsed_args| {
                    fixup_cicc_args(
                        parsed_args,
                        keep_dir.as_deref(),
                        compile_flag,
                        cudafe_has_gen_module_id_file_flag,
                    )
                })
                .map(|parsed_args| {
                    (
                        NvccSubcommandKind::Compile,
                        device_compile_groups
                            .entry(parsed_args.input.clone())
                            .or_default(),
                        parsed_args,
                        env_vars.clone(),
                    )
                }),
            // Group these with their prior corresponding cicc commands
            "ptxas" | "tileiras" => {
                let cicc_output_flag = match cmd {
                    "ptxas" => "obj",
                    "tileiras" => "--tile_bc_file_name",
                    _ => unreachable!(),
                };
                // (ptxas | tileiras) ... <input>
                // (ptxas | tileiras) ... <input> -o <output>
                Some(parse_args_simple(args.iter().map(Into::into), dir))
                    // Ignore ptxas invocations without an output or input
                    .filter(|parsed_args| parsed_args.outputs.contains_key("obj"))
                    .filter(|parsed_args| !parsed_args.input.as_os_str().is_empty())
                    .and_then(|parsed_args| {
                        // Search for the prior corresponding cicc call
                        device_compile_groups
                            .values()
                            .find_map(|cmds| {
                                cmds.iter().rev().find_map(|(cicc, _)| {
                                    cicc.outputs
                                        .get(cicc_output_flag)
                                        .filter(|o| o.path == parsed_args.input)
                                        .map(|_| cicc.input.clone())
                                })
                            })
                            // Otherwise create a new device compile group
                            .or_else(|| Some(parsed_args.input.clone()))
                            .map(|input| {
                                (parsed_args, device_compile_groups.entry(input).or_default())
                            })
                    })
                    .map(|(parsed_args, group)| {
                        (
                            NvccSubcommandKind::Compile,
                            group,
                            parsed_args,
                            env_vars.clone(),
                        )
                    })
            }
            _ => {
                Some(parse_args_simple(args.iter().map(Into::into), dir))
                    // All generated host compiler commands include one of these defines.
                    // If one of these isn't present, this command is either a new binary
                    // in the CTK that we don't know about, or a line like `rm x_dlink.reg.c`
                    // that nvcc generates in certain cases.
                    .filter(|parsed_args| {
                        parsed_args.common_args.iter().any(|arg| {
                            arg.starts_with("-D__CUDA_ARCH__")
                                || arg.starts_with("-D__CUDA_ARCH_LIST__")
                                || arg.starts_with("-D__CUDACC__")
                                || arg.starts_with("-D__CUDACC_VER")
                                || arg.starts_with("-D__CUDACC_RDC__")
                                || arg.starts_with("-D__NVCC__")
                                || arg.starts_with("-lcudart")
                                || arg.starts_with("-lcudadevrt")
                        })
                    })
                    .and_then(|parsed_args| {
                        parsed_args
                            .outputs
                            .get("obj")
                            .map(|o| o.path.clone())
                            .map(|o| (parsed_args, o))
                    })
                    .map(|(parsed_args, output)| {
                        // Each host preprocessor step represents the start of a new command group
                        if parsed_args.common_args.contains(&preprocessor_flag) {
                            // If the output file ends with...
                            // * .cpp4.ii - cudafe++ input
                            // * .cpp1.ii - cicc/ptxas input
                            let group = if output
                                .file_name()
                                .expect("output is a named file")
                                .ends_with(".cpp4.ii")
                            {
                                &mut cuda_front_end_group
                            } else {
                                device_compile_groups.entry(output).or_default()
                            };
                            (
                                NvccSubcommandKind::Preprocess,
                                group,
                                parsed_args,
                                env_vars.clone(),
                            )
                        } else if compile_flag == &NvccCompileFlag::Executable {
                            // If building an executable, cache the host compiler objects.
                            (
                                NvccSubcommandKind::Compile,
                                &mut final_assembly_group,
                                parsed_args,
                                env_vars.clone(),
                            )
                        } else {
                            (
                                NvccSubcommandKind::Compile,
                                &mut final_assembly_group,
                                parsed_args,
                                // Don't cache this object because it will be cached using the outer nvcc object's hash_key
                                env_vars
                                    .iter()
                                    .chain(&[("SCCACHE_NO_CACHE".into(), "1".into())])
                                    .cloned()
                                    .collect::<Vec<_>>(),
                            )
                        }
                    })
            }
        } {
            let cmd = NvccGeneratedSubcommand {
                // Resolve compiler avoiding ccache wrappers to prevent double-caching.
                exe: resolve_compiler_avoiding_wrapper(exe, &env_vars),
                args: parsed_args.common_args.clone(),
                cwd: dir.to_owned(),
                env_vars,
                kind,
                output: parsed_args.outputs.get("obj").map(|o| dir.join(&o.path)),
            };

            trace!(
                "[{}]: transformed nvcc command: {cmd}",
                output_path.display(),
            );

            group.push((parsed_args, cmd));
        }
    }

    std::iter::empty()
        .chain([cuda_front_end_group])
        .chain(device_compile_groups.into_values())
        .chain([final_assembly_group])
        .map(|xs| xs.into_iter().map(|(_, c)| c).collect::<Vec<_>>())
        .collect::<Vec<_>>()
}

counted_array!(static SIMPLE_ARGS: [ArgInfo<gcc::ArgData>; _] = [
    take_arg!("--tile_bc_file_name", PathBuf, Separated, Output),
    take_arg!("-Fi", PathBuf, Concatenated, Output),
    take_arg!("-Fo", PathBuf, Concatenated, Output),
    take_arg!("-o", PathBuf, Separated, Output),
    take_arg!("-olto", OsString, Separated, PassThrough),
    take_arg!("/Fi", PathBuf, Concatenated, Output),
    take_arg!("/Fo", PathBuf, Concatenated, Output),
]);

fn parse_args_simple<I: IntoIterator<Item = OsString>>(args: I, cwd: &Path) -> ParsedArguments {
    let (input, mut outputs, args) =
        parse_input_outputs(args.into_iter(), &SIMPLE_ARGS[..], |arg| {
            match arg.get_data() {
                Some(Output(p)) => Some(p),
                _ => None,
            }
        });

    let input = input.map(|p| cwd.join(p)).unwrap_or_default();

    for (_, p) in outputs.iter_mut() {
        *p = cwd.join(&p);
    }

    if let Some(p) = None
        .or_else(|| outputs.remove("-o"))
        .or_else(|| outputs.remove("-Fo"))
        .or_else(|| outputs.remove("/Fo"))
        .or_else(|| outputs.remove("-Fi"))
        .or_else(|| outputs.remove("/Fi"))
        .or_else(|| {
            if outputs.len() == 1 {
                outputs.drain().next().map(|(_, v)| v)
            } else {
                None
            }
        })
    {
        outputs.insert("obj", p);
    }

    let outputs = outputs
        .drain()
        .map(|(key, path)| {
            (
                key,
                ArtifactDescriptor {
                    path,
                    optional: false,
                },
            )
        })
        .collect();

    ParsedArguments {
        input,
        outputs,
        common_args: args,
        ..Default::default()
    }
}

fn fixup_cicc_args(
    mut parsed_args: ParsedArguments,
    keep_dir: Option<&Path>,
    compile_flag: &NvccCompileFlag,
    cudafe_has_gen_module_id_file_flag: bool,
) -> ParsedArguments {
    let args = &mut parsed_args.common_args;

    // Fix for CTK < 12.0:
    if let Some(idx) = args.iter().position(|x| x == "--module_id_file_name") {
        // Remove `--gen_module_id_file` if cudafe++ already does it
        if cudafe_has_gen_module_id_file_flag {
            if let Some(idx) = args.iter().position(|x| x == "--gen_module_id_file") {
                args.splice(idx..idx + 1, []);
            }
        } else if !args.contains(&OsString::from("--gen_module_id_file")) {
            // Otherwise if `--module_id_file_name` exists, insert `--gen_module_id_file`
            args.splice(idx..idx, ["--gen_module_id_file".into()]);
        }
    }

    // Add or remove these flags:
    // * `--gen_c_file_name test_a.compute_XX.cudafe1.c`
    // * `--stub_file_name test_a.compute_XX.cudafe1.stub.c`
    // * `--gen_device_file_name test_a.compute_XX.cudafe1.gpu` (optional)
    //
    // This ensures the same `cicc` command is generated regardless
    // of whether the compilation flag is `-c`, `-ptx`, or `-cubin`

    // pa.input is absolute, so use the original form here instead
    let input = parsed_args
        .input
        .file_name()
        .and_then(|name| name.to_str())
        .expect("input is a named file");
    let mut input_index = args
        .iter()
        .position(|arg| arg.ends_with(input))
        .expect("input is in args");
    // foo.compute_52.cpp1.ii -> foo.compute_52.cpp1
    let input = Path::new(&args[input_index]);
    let input = if input.extension().filter(|&e| e == "ii").is_some() {
        input.with_extension("")
    } else {
        input.to_owned()
    };

    let is_compile = matches!(
        compile_flag,
        NvccCompileFlag::Device | NvccCompileFlag::Executable
    );

    for (flag, path) in [
        (
            "--stub_file_name",
            is_compile.then(|| input.with_extension("cudafe1.stub.c")),
        ),
        // Only generate `.cudafe1.{c,gpu}` files when `-keep` is on.
        //
        // nvcc's cicc commands  output them by default, but as far as
        // I can tell, they aren't used in the compilation pipeline.
        //
        // These files can be large (`.gpu` can be hundreds of MiB),
        // making them really expensive to cache and/or ship around
        // sccache-dist servers unnecessarily.
        //
        // `nvcc -h` says users can do `nvcc -c x.gpu -o x.cubin`
        // to compile a `.gpu` file directly to a cubin with cicc,
        // but I'm not sure how the user would get the `.gpu` file
        // without doing `nvcc -c x.cu -o x.o -keep` first.
        (
            "--gen_c_file_name",
            keep_dir
                .as_ref()
                .filter(|_| is_compile)
                .map(|_| input.with_extension("cudafe1.c")),
        ),
        (
            "--gen_device_file_name",
            keep_dir
                .as_ref()
                .filter(|_| is_compile)
                .map(|_| input.with_extension("cudafe1.gpu")),
        ),
    ] {
        match (path, args.iter().position(|x| x == flag)) {
            (Some(path), None) => {
                // Insert the flag and value before the input path
                args.splice(input_index..input_index, [flag.into(), path.into()]);
                input_index += 2;
            }
            (None, Some(flag_index)) => {
                // Remove the flag and value
                args.splice(flag_index..flag_index + 2, []);
                if flag_index < input_index {
                    input_index -= 2;
                }
            }
            _ => {}
        }
    }

    parsed_args
}

#[allow(clippy::too_many_arguments)]
async fn select_nvcc_subcommands<T, F>(
    creator: &T,
    executable: &Path,
    cwd: &Path,
    env_vars: &mut Vec<(OsString, OsString)>,
    arguments: &[OsString],
    compile_flag: &NvccCompileFlag,
    host_compiler: &NvccHostCompiler,
    nvcc_internal_files: &mut HashMap<String, String>,
    output_path: &Path,
    select_subcommand: F,
) -> Result<Vec<(usize, PathBuf, Vec<String>)>>
where
    F: Fn(&str, &[String]) -> bool,
    T: CommandCreatorSync,
{
    let mut nvcc_dryrun_cmd = creator.clone().new_command_sync(executable);

    nvcc_dryrun_cmd
        .args(&[arguments, &["--dryrun".into(), "--keep".into()][..]].concat())
        .env_clear()
        .current_dir(cwd)
        .envs(env_vars.iter().map(|(k, v)| (k, v)));

    trace!(
        "[{}]: nvcc dryrun command: {nvcc_dryrun_cmd}",
        output_path.display()
    );

    let lines = match run_input_output(nvcc_dryrun_cmd, None).await {
        Ok(output) => {
            #[cfg(unix)]
            let lines = output.stderr;
            #[cfg(windows)]
            let lines = output.stdout;
            // Avoid dropping Windows wide chars in paths
            from_local_codepage(lines)?
        }
        Err(err) => {
            match err.downcast::<ProcessError>() {
                Ok(ProcessError(mut output)) => {
                    // nvcc on windows outputs errors on stdout
                    if cfg!(target_os = "windows") {
                        output.stderr = output.stdout;
                    }
                    output.stdout = vec![];
                    bail!(ProcessError(output));
                }
                Err(err) => {
                    bail!(err);
                }
            }
        }
    };

    parse_nvcc_subcommands(
        cwd,
        env_vars,
        compile_flag,
        host_compiler,
        nvcc_internal_files,
        select_subcommand,
        &lines,
    )
}

fn parse_nvcc_subcommands<F>(
    cwd: &Path,
    env_vars: &mut Vec<(OsString, OsString)>,
    compile_flag: &NvccCompileFlag,
    host_compiler: &NvccHostCompiler,
    nvcc_internal_files: &mut HashMap<String, String>,
    select_subcommand: F,
    lines: &str,
) -> Result<Vec<(usize, PathBuf, Vec<String>)>>
where
    F: Fn(&str, &[String]) -> bool,
{
    let mut new_vars = HashMap::<String, (String, String)>::new();

    let lines = lines
        .split("\n")
        // Trim extra spaces/carriage returns
        .filter_map(|line| Some(line.trim()).filter(|&line| !line.is_empty()))
        // Select lines that match the `#$ ` prefix from nvcc --dryrun
        .filter_map(|line| line.strip_prefix("#$ ").filter(|&line| !line.is_empty()))
        // Parse the nvcc envvar and command lines
        .filter_map(|line| {
            fold_env_vars_or_split_into_exe_and_args(
                cwd,
                line,
                env_vars,
                &mut new_vars,
                host_compiler,
            )
        })
        .try_collect::<_, Vec<_>, _>()?;

    // Update the input env_vars with the ones from nvcc --dryrun
    for (var, (val, _)) in new_vars.iter() {
        env_vars.splice(
            if let Some(idx) = env_vars.iter().position(|(k, _)| k == var.as_str()) {
                idx..idx + 1
            } else {
                env_vars.len()..env_vars.len()
            },
            [(var.into(), val.into())],
        );
    }

    //
    // Remap nvcc's generated file names to deterministic names.
    // nvcc generates different file names depending on whether it's compiling one vs. many archs.
    //
    // For example, both of these commands generate PTX for sm60, so we should get a cicc cache hit:
    // 1. `nvcc -x cu -c x.cu -o x.cu.o -gencode=arch=compute_60,code=[compute_60,sm_60]`
    // 2. `nvcc -x cu -c x.cu -o x.cu.o -gencode=arch=compute_60,code=[sm_60] -gencode=arch=compute_70,code=[compute_70,sm_70]`
    //
    // The first command generates:
    // ```
    // cicc --gen_c_file_name x.cudafe1.c \
    //      --stub_file_name x.cudafe1.stub.c \
    //      --gen_device_file_name x.cudafe1.gpu \
    //      -o x.ptx
    // ```
    //
    // The second command generates:
    // ```
    // cicc --gen_c_file_name x.compute_60.cudafe1.c \
    //      --stub_file_name x.compute_60.cudafe1.stub.c \
    //      --gen_device_file_name x.compute_60.cudafe1.gpu \
    //      -o x.compute_60.ptx
    // ```
    //
    // The second command yields a false-positive cache miss because the names are different.
    //
    // This matters because CI jobs will often compile .cu files in "many-arch" mode, but devs
    // who have just one GPU locally prefer to only compile for their specific GPU arch. It is
    // preferrable if they can reuse the PTX populated by "many-arch" CI jobs.
    //
    // So to avoid this, we detect these "single-arch" compilations and rewrite the names to
    // match what nvcc generates for "many-arch" compilations.
    //
    let last_arch = lines
        .iter()
        // Reverse so we search from the end and work backwards
        .rev()
        .filter_map(|(exe, args)| {
            if let Some(exe_name) = exe.file_stem().and_then(|s| s.to_str())
                && is_nvcc_exe(exe_name, args)
            {
                return Some((exe_name, args));
            }
            None
        })
        .find_map(|(exe, args)| {
            if exe == "ptxas"
                && let Some(arch) = args
                    .iter()
                    .position(|arg| arg == "-o")
                    // Get the output file name
                    .and_then(|idx| args.get(idx + 1))
                    // Only if it ends with `.cubin`
                    .and_then(|val| val.strip_suffix(".cubin"))
                    // And has the string `sm_` in the name
                    .filter(|val| HAS_SM_IN_NAME_RE.is_match(val))
                    // * Split on `sm_`
                    // * Ignore everything before `.sm_`
                    // * Everything after `.sm_` is the arch
                    .and_then(|val| val.split("sm_").nth(1))
            {
                return Some(arch);
            }

            if let Some(arch) = args
                .iter()
                .position(|arg| arg == "-arch")
                // Get the -arch value
                .and_then(|idx| args.get(idx + 1))
                // Split on the first underscore
                .and_then(|val| val.split_once('_'))
                // Everything after the first underscore is the arch
                .map(|(_, arch)| arch)
            {
                return Some(arch);
            }

            None
        });

    let lines = if let Some(arch) = last_arch {
        remap_generated_filenames(arch, compile_flag, &lines, nvcc_internal_files)
    } else {
        lines
    };

    Ok(lines
        .into_iter()
        .enumerate()
        .filter_map(|(idx, (exe, args))| {
            if let Some(exe_name) = exe.file_stem().and_then(|s| s.to_str())
                && select_subcommand(exe_name, &args)
            {
                return Some((idx, exe, args));
            }
            None
        })
        .collect::<Vec<_>>())
}

fn fold_env_vars_or_split_into_exe_and_args(
    cwd: &Path,
    line: &str,
    env_vars: &[(OsString, OsString)],
    env_vars_nvcc: &mut HashMap<String, (String, String)>,
    host_compiler: &NvccHostCompiler,
) -> Option<Result<(PathBuf, Vec<String>)>> {
    // Intercept the environment variable lines and add them to env_vars_nvcc
    if let Some((var, val)) = line.split_once("=")
        && var.chars().all(|b| b.is_ascii_alphabetic() || b == '_')
    {
        env_vars_nvcc.entry(var.into()).or_insert_with(|| {
            (
                val.into(),
                if cfg!(target_os = "windows") {
                    // %CICC_PATH%
                    format!(r#"%{var}%"#)
                } else {
                    // $CICC_PATH
                    format!(r#"${var}"#)
                },
            )
        });

        return None;
    }

    // The rest of the lines are subcommands, so parse into a vec of [exe, args..]

    let line = if matches!(host_compiler, NvccHostCompiler::Msvc) {
        // Rewrite stdout to go to a file instead
        let line = line.replace(" > ", " -Fi");
        // Preprocess to a file instead of stdout
        let line = line.replace(" -E ", " -P ");
        // Unwrap double quoted commands that contain envvars
        if line.starts_with(r#""""#) {
            Some(line.as_str())
                .and_then(|line| line.strip_prefix(r#"""#))
                .and_then(|line| line.strip_suffix(r#"""#))
                .map(|line| line.to_owned())
                .unwrap_or(line)
        } else {
            line
        }
    } else {
        line.to_owned()
    };

    // Expand envvars in nvcc subcommands, i.e. "$CICC_PATH/cicc ..." or "%CICC_PATH%/cicc"
    let line = env_vars_nvcc
        .values()
        .fold(line, |line, (val, pat)| line.replace(pat, val));

    // Split line into exe and arguments, respecting platform quoting rules
    let (exe, args) = match split_quoted_shell_str(&line) {
        None => return Some(Err(anyhow!("Could not parse shell line"))),
        Some(mut exe_and_args) => {
            let args = exe_and_args.split_off(1);
            let exe = exe_and_args.drain(..).next().unwrap();
            (exe, args)
        }
    };

    // Find the full binary path
    let env_path = env_vars_nvcc
        .get("PATH")
        .map(|(v, _)| OsStr::new(v))
        .or_else(|| {
            env_vars
                .iter()
                .find(|(k, _)| k == "PATH")
                .map(|(_, v)| v.as_os_str())
        });

    let exe = which_in(&exe, env_path, cwd).unwrap_or_else(|_| PathBuf::from(exe));

    // Remove any shell stdio redirect arguments
    let (_, args) = args
        .into_iter()
        .fold((None::<String>, vec![]), |(prev, mut args), curr| {
            // If the previous arg was <, >, <0, 1>, or 2>, then it
            // redirected stdin/stdout/stderr to the current arg.
            // Discard both previous and current args.
            if let Some(prev) = prev
                && matches!(prev.as_str(), "<" | ">" | "<0" | "1>" | "2>")
            {
                return (None, args);
            }

            // If the arg is <, >, <0, 1>, or 2>, discard both the current and next args.
            if matches!(curr.as_str(), "<" | ">" | "<0" | "1>" | "2>") {
                return (Some(curr), args);
            }

            // Discard if the current arg starts with < or >.
            if !curr.is_empty() && matches!(&curr[0..1], "<" | ">") {
                return (None, args);
            }

            // Discard if the current arg starts with <0, 1> or 2>.
            if curr.len() > 1 && matches!(&curr[0..2], "<0" | "1>" | "2>") {
                return (None, args);
            }

            // Otherwise keep this arg
            args.push(curr);

            (None, args)
        });

    Some(Ok((exe, args)))
}

fn remap_generated_filenames(
    last_arch: &str,
    compile_flag: &NvccCompileFlag,
    lines: &[(PathBuf, Vec<String>)],
    nvcc_internal_files: &mut HashMap<String, String>,
) -> Vec<(PathBuf, Vec<String>)> {
    let nvcc_internal_file_extensions = [
        "_dlink.fatbin.c",
        "_dlink.fatbin",
        "_dlink.o",
        "_dlink.reg.c",
        ".cpp1.ii",
        ".cpp4.ii",
        ".cubin",
        ".cudafe1.c",
        ".cudafe1.cpp",
        ".cudafe1.gpu",
        ".cudafe1.stub.c",
        ".fatbin.c",
        ".fatbin",
        ".module_id",
        ".ltoir",
        ".optixir",
        ".ptx",
        ".tilebc",
    ];

    let mut extensions_to_rename = vec![
        ".cpp1.ii",
        ".cudafe1.c",
        ".cudafe1.cpp",
        ".cudafe1.gpu",
        ".cudafe1.stub.c",
    ];

    match compile_flag {
        // Rewrite PTX names if the compile flag is `-cubin`, `-ltoir`, or `-tilebc`
        NvccCompileFlag::Cubin | NvccCompileFlag::LtoIR | NvccCompileFlag::TileBC => {
            extensions_to_rename.push(".ptx");
        }
        // Rewrite PTX, cubin, ltoir, optixir, and tilebc names if the compile flag is `-c`, `-fatbin`, or `-tilefatbin`
        NvccCompileFlag::Device | NvccCompileFlag::Executable | NvccCompileFlag::Fatbin => {
            extensions_to_rename.push(".ptx");
            extensions_to_rename.push(".cubin");
            extensions_to_rename.push(".ltoir");
            extensions_to_rename.push(".optixir");
            extensions_to_rename.push(".tilebc");
        }
        _ => {}
    }

    lines
        .iter()
        .map(|(exe, args)| {
            (
                exe.clone(),
                args.iter()
                    .map(|arg| {
                        // Special case for MSVC's preprocess output file name flag
                        let arg_is_msvc_preprocessor_output = arg.starts_with("-Fi");

                        let arg = if arg_is_msvc_preprocessor_output {
                            arg.trim_start_matches("-Fi").to_owned()
                        } else {
                            arg.to_owned()
                        };

                        // If the argument doesn't start with `-` and is a file that
                        // ends in one of these extensions, rename the file to a
                        // stable name that includes the compute architecture.
                        let maybe_extension = if !arg.starts_with('-') {
                            nvcc_internal_file_extensions
                                .iter()
                                .find(|ext| arg.ends_with(*ext))
                                .copied()
                        } else {
                            None
                        };

                        // If the argument is a file that ends in one of the above extensions:
                        // * If it's our first time seeing this file, compute a stable name for it
                        // * If we've seen this file before, lookup its stable name in the hash map
                        //
                        // This ensures stable names are in cudafe++ output and #include directives,
                        // eliminating one source of false-positive cache misses.
                        let arg = match maybe_extension {
                            // nvcc generates different cubin names under different conditions:
                            // 1. `x.cubin` with one `-gencode` arg and *not* embedding sm_XX PTX
                            // 2. `x.sm_XX.cubin` with one `-gencode` arg and embedding sm_XX PTX
                            // 2. `x.compute_XX.cubin` with multiple `-gencode` args and *not* embedding sm_XX PTX
                            // 3. `x.compute_XX.sm_XX.cubin` with multiple `-gencode` args and embedding sm_XX PTX
                            //
                            // Since the output cubin is identical, rewrite the first three forms
                            // to match the fourth form so we get more cache hits.
                            Some(".cubin") => {
                                nvcc_internal_files
                                    .entry(arg)
                                    .or_insert_with_key(|arg| {
                                        if !extensions_to_rename
                                            .iter()
                                            .any(|ext| arg.ends_with(*ext))
                                        {
                                            return arg.to_owned();
                                        }
                                        let mut path = PathBuf::from(arg);

                                        let cubin_arch = path
                                            .file_name()
                                            .and_then(|name| name.to_str())
                                            .and_then(|name| {
                                                // Ignore everything before `.sm_` or `.compute_`
                                                // Take everything after `.sm_` or `.compute_`, i.e. `{arch}.cubin`
                                                let arch = if HAS_SM_IN_NAME_RE.is_match(name) {
                                                    name.split(".sm_").nth(1)
                                                } else if HAS_COMPUTE_IN_NAME_RE.is_match(name) {
                                                    name.split(".compute_").nth(1)
                                                } else {
                                                    return None;
                                                };

                                                arch.and_then(|arch| {
                                                    // Reverse chars so we split on the last dot
                                                    arch.chars()
                                                        .rev()
                                                        .collect::<String>()
                                                        .split_once('.')
                                                        // Now everything after the first dot is the
                                                        // arch number (but still reversed)
                                                        .map(|(_, arch)| {
                                                            // Reverse chars again to original order
                                                            arch.chars().rev().collect::<String>()
                                                        })
                                                })
                                            })
                                            .unwrap_or_else(|| last_arch.to_owned());

                                        // Add the `sm_{arch}` component if necessary
                                        if let Some(name) = path
                                            .file_name()
                                            .and_then(|name| name.to_str())
                                            .and_then(|name| {
                                                (!HAS_SM_IN_NAME_RE.is_match(name)).then_some(name)
                                            })
                                        {
                                            // test_a.cubin -> test_a
                                            // test_a.compute_60.cubin -> (test_a.compute_60, cubin)
                                            let name = name.strip_suffix(".cubin").unwrap();
                                            // test_a.cubin -> test_a.sm_60.cubin
                                            // test_a.compute_60.cubin -> test_a.compute_60.sm_60.cubin
                                            let name = format!("{name}.sm_{cubin_arch}.cubin");
                                            path.set_file_name(name);
                                        }

                                        // Add the `compute_{arch}` component if necessary
                                        if let Some(name) = path
                                            .file_name()
                                            .and_then(|name| name.to_str())
                                            .and_then(|name| {
                                                (!HAS_COMPUTE_IN_NAME_RE.is_match(name))
                                                    .then_some(name)
                                            })
                                        {
                                            // test_a.sm_60.cubin -> test_a
                                            let ext = format!(".sm_{cubin_arch}.cubin");
                                            let name = name.strip_suffix(&ext).unwrap();
                                            // test_a.sm_60.cubin -> test_a.compute_60.sm_60.cubin
                                            let name = format!("{name}.compute_{cubin_arch}{ext}");
                                            path.set_file_name(name);
                                        }
                                        path.into_os_string().into_string().unwrap()
                                    })
                                    .to_owned()
                            }
                            Some(ext) => {
                                let arg = nvcc_internal_files
                                    .entry(arg)
                                    .or_insert_with_key(|arg| {
                                        if !extensions_to_rename
                                            .iter()
                                            .any(|ext| arg.ends_with(*ext))
                                        {
                                            return arg.to_owned();
                                        }
                                        let mut path = PathBuf::from(arg);
                                        // Add the `compute_{arch}` component if necessary
                                        if let Some(name) = path
                                            .file_name()
                                            .and_then(|name| name.to_str())
                                            .and_then(|name| {
                                                (!HAS_COMPUTE_IN_NAME_RE.is_match(name))
                                                    .then_some(name)
                                            })
                                        {
                                            // test_a.cudafe1.c -> test_a
                                            let name = name.strip_suffix(ext).unwrap();
                                            // test_a.cudafe1.c -> test_a.compute_60.cudafe1.c
                                            let name = format!("{name}.compute_{last_arch}{ext}");
                                            path.set_file_name(name);
                                        }
                                        path.into_os_string().into_string().unwrap()
                                    })
                                    .to_owned();

                                // cicc generates additional "test_a.compute_XX.cpp1.int.c"
                                // and "test_a.compute_XX.cpp1.int.device.c" files that we
                                // should clean up at the end.
                                if ext == ".cpp1.ii"
                                    && let Some(name) = arg.strip_suffix(ext)
                                {
                                    for suf in ["cpp1.int.c", "cpp1.int.device.c"] {
                                        nvcc_internal_files.insert(
                                            format!("{name}.{suf}"),
                                            format!("{name}.{suf}"),
                                        );
                                    }
                                }

                                arg
                            }
                            None => {
                                // If the argument isn't a file name with one of our extensions,
                                // it may _reference_ files we've renamed. Go through and replace
                                // all old names with their new stable names.
                                //
                                // Sort by string length descending so we don't accidentally replace
                                // `test_a.cudafe1.cpp` with the new name for `test_a.cudafe1.c`.
                                //
                                // For example, if we have these renames:
                                //
                                //   test_a.cudafe1.cpp -> test_a.compute_70.cudafe1.cpp
                                //   test_a.cudafe1.c   -> test_a.compute_70.cudafe1.c
                                //
                                // `test_a.cudafe1.cpp` should be replaced with
                                // `test_a.compute_70.cudafe1.cpp`, not
                                // `test_a.compute_70.cudafe1.c`
                                //
                                let mut arg = arg.clone();
                                for (old, new) in nvcc_internal_files
                                    .iter()
                                    // Don't replace if old and new are different
                                    .filter(|&(old, new)| old != new)
                                    .filter(|&(old, _)| {
                                        //
                                        // Replace old -> new if:
                                        // * The argument ends with the old name
                                        // * The argument contains - but does not start with - the old name
                                        //
                                        // This ensures we don't replace `x.ltoir` -> `x.compute_XX.ltoir`
                                        // in `x.ltoir.o`, because even though `x.ltoir.o` contains and
                                        // starts with `x.ltoir`, it does not end with it.
                                        //
                                        // This will still replace `x.ltoir` -> `x.compute_XX.ltoir` in
                                        // the string `--image3=file=x.ltoir,kind=nvvm,sm=75`, because
                                        // even though it doesn't end with `x.ltoir`, it doesn't start
                                        // with and does contain and `x.ltoir`.
                                        //
                                        arg.ends_with(old)
                                            || (!arg.starts_with(old) && arg.contains(old))
                                    })
                                    .sorted_by(|a, b| b.0.len().cmp(&a.0.len()))
                                {
                                    arg = arg.replace(old, new);
                                }
                                arg
                            }
                        };

                        if arg_is_msvc_preprocessor_output {
                            format!("-Fi{arg}")
                        } else {
                            arg
                        }
                    })
                    .collect::<Vec<_>>()
                    .iter()
                    .map(|arg| {
                        // If the argument matches the form `-<arg>=<val>` and `val` is a
                        // filename or path that ends in one of these extensions, extract
                        // and track the filename as an internal nvcc file
                        if let Some(groups) = ARG_HAS_FILE_WITH_EXTENSION_RE.captures(arg.as_str())
                        {
                            let (_, [name]) = groups.extract();
                            if !nvcc_internal_files.contains_key(name)
                                && nvcc_internal_file_extensions
                                    .iter()
                                    .any(|ext| name.ends_with(*ext))
                            {
                                nvcc_internal_files.insert(name.to_owned(), name.to_owned());
                            }
                        }
                        arg.to_owned()
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>()
}

async fn run_nvcc_subcommand<T>(
    creator: &T,
    output_path: &Path,
    cmd: NvccGeneratedSubcommand,
) -> Result<ProcessOutput>
where
    T: CommandCreatorSync,
{
    let NvccGeneratedSubcommand {
        exe,
        args,
        cwd,
        env_vars,
        ..
    } = cmd;

    let mut cmd = creator.clone().new_command_sync(exe);

    cmd.args(&args).current_dir(cwd).env_clear().envs(env_vars);

    trace!("[{}]: run_nvcc_subcommand: {cmd}", output_path.display());

    run_input_output(cmd, None).await
}

async fn run_nvcc_subcommands_group<T>(
    service: &SccacheService<T>,
    creator: &T,
    commands: Vec<NvccGeneratedSubcommand>,
    output_path: &Path,
) -> Result<ProcessOutput>
where
    T: CommandCreatorSync,
{
    let mut output = ProcessOutput::default();

    for cmd in commands {
        let out = match cmd.kind {
            NvccSubcommandKind::Other => run_nvcc_subcommand(creator, output_path, cmd)
                .await
                .unwrap_or_else(error_to_output),
            NvccSubcommandKind::Preprocess => {
                if cmd
                    .output
                    .as_deref()
                    .filter(|output| output.exists())
                    .is_some()
                {
                    continue;
                }

                run_nvcc_subcommand(creator, output_path, cmd)
                    .await
                    .unwrap_or_else(error_to_output)
            }
            NvccSubcommandKind::Compile => {
                trace!(
                    "[{}]: run_commands_sequential: {cmd}",
                    output_path.display()
                );

                let NvccGeneratedSubcommand {
                    exe,
                    args,
                    cwd,
                    env_vars,
                    ..
                } = &cmd;

                let pending = service.increment_pending_compilations();

                match service.compiler_info(exe, cwd, args, env_vars).await {
                    Err(err) => error_to_output(err),
                    Ok(compiler) => match compiler.parse_arguments(args, cwd, env_vars) {
                        CompilerArguments::Ok(hasher) => service
                            .clone()
                            .start_compile_task(
                                compiler,
                                hasher,
                                cmd.args,
                                cmd.cwd,
                                cmd.env_vars,
                                pending,
                            )
                            .await
                            .map_or_else(error_to_output, |res| res.output),
                        _ => run_nvcc_subcommand(creator, output_path, cmd)
                            .await
                            .unwrap_or_else(error_to_output),
                    },
                }
            }
        };

        output = aggregate_output(output, out);

        if !output.success() {
            break;
        }
    }

    Ok(output)
}

fn aggregate_output(lhs: ProcessOutput, rhs: ProcessOutput) -> ProcessOutput {
    ProcessOutput {
        // On Unix, prioritize surfacing SIGSEV exit codes over others
        status: if cfg!(unix) && matches!(lhs.status.code(), Some(139)) {
            lhs.status
        } else if cfg!(unix) && matches!(rhs.status.code(), Some(139)) {
            rhs.status
        } else if !lhs.success() {
            lhs.status
        } else if !rhs.success() {
            rhs.status
        } else {
            Default::default()
        },
        stdout: [lhs.stdout, rhs.stdout].concat(),
        stderr: [lhs.stderr, rhs.stderr].concat(),
    }
}

fn error_to_output(err: Error) -> ProcessOutput {
    match err.downcast::<ProcessError>() {
        Ok(ProcessError(out)) => out,
        Err(err) => ProcessOutput::new(1, vec![], err.to_string().into_bytes()),
    }
}

counted_array!(pub static ARGS: [ArgInfo<gcc::ArgData>; _] = [
    //todo: refactor show_includes into dependency_args
    take_arg!("--Ofast-compile", OsString, CanBeSeparated, PassThrough),
    take_arg!("--Ofast-compile=", OsString, Concatenated, PassThrough),
    take_arg!("--Werror", OsString, CanBeSeparated(b'='), PreprocessorArgument),
    take_arg!("--archive-options", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("--compile", DoCompilation),
    take_arg!("--compiler-bindir", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("--compiler-options", OsString, CanBeSeparated(b'='), PreprocessorArgument),
    take_arg!("--compress-mode", OsString, CanBeSeparated, PassThrough),
    take_arg!("--compress-mode=", OsString, Concatenated, PassThrough),
    flag!("--cubin", DoCompilation),
    flag!("--cuda", NotCompilationFlag),
    take_arg!("--default-stream", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("--dependency-output", PathBuf, Separated, DepArgumentPath),
    flag!("--device-c", DoCompilation),
    flag!("--device-debug", PassThroughFlag),
    flag!("--device-link", NotCompilationFlag),
    flag!("--device-w", DoCompilation),
    take_arg!("--diag-suppress", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("--dlink-time-opt", PassThroughFlag),
    take_arg!("--dopt", OsString, CanBeSeparated, PassThrough),
    take_arg!("--dopt=", OsString, Concatenated, PassThrough),
    flag!("--expt-extended-lambda", PreprocessorArgumentFlag),
    flag!("--expt-relaxed-constexpr", PreprocessorArgumentFlag),
    flag!("--extended-lambda", PreprocessorArgumentFlag),
    flag!("--extensible-whole-program", PassThroughFlag),
    flag!("--fatbin", DoCompilation),
    take_arg!("--fdevice-time-trace", OsString, CanBeSeparated, TooHard),
    take_arg!("--fdevice-time-trace=", OsString, Concatenated, TooHard),
    flag!("--gen-opt-lto", PassThroughFlag),
    take_arg!("--generate-code", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("--generate-dependencies-with-compile", NeedDepTarget),
    flag!("--generate-nonsystem-dependencies-with-compile", NeedDepTarget),
    take_arg!("--gpu-architecture", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("--gpu-code", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("--include-path", PathBuf, CanBeSeparated(b'='), PreprocessorArgumentPath),
    flag!("--jobserver", UnhashedFlag),
    flag!("--keep", UnhashedFlag),
    take_arg!("--keep-dir", OsString, CanBeSeparated(b'='), Unhashed),
    take_arg!("--linker-options", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("--lto", NotCompilationFlag),
    flag!("--ltoir", DoCompilation),
    take_arg!("--maxrregcount", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("--no-compress", PassThroughFlag),
    flag!("--no-host-device-initializer-list", PreprocessorArgumentFlag),
    take_arg!("--nvlink-options", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("--objdir-as-tempdir", UnhashedFlag),
    take_arg!("--options-file", PathBuf, CanBeSeparated(b'='), ExtraHashFile),
    flag!("--optix-ir", DoCompilation),
    flag!("--profile", PassThroughFlag),
    flag!("--ptx", DoCompilation),
    take_arg!("--ptxas-options", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("--relocatable-device-code", OsString, CanBeSeparated(b'='), PreprocessorArgument),
    flag!("--save-temps", UnhashedFlag),
    take_arg!("--split-compile", OsString, CanBeSeparated, Unhashed),
    take_arg!("--split-compile-extended", OsString, CanBeSeparated, Unhashed),
    take_arg!("--split-compile-extended=", OsString, Concatenated, Unhashed),
    take_arg!("--split-compile=", OsString, Concatenated, Unhashed),
    take_arg!("--system-include", PathBuf, CanBeSeparated(b'='), PreprocessorArgumentPath),
    take_arg!("--threads", OsString, CanBeSeparated(b'='), Unhashed),
    flag!("--tilebc", DoCompilation),
    flag!("--tilecubin", DoCompilation),
    take_arg!("--tileiras-options", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("--time", OsString, CanBeSeparated, TooHard),
    take_arg!("--time=", OsString, Concatenated, TooHard),
    take_arg!("--x", OsString, CanBeSeparated(b'='), Language),

    flag!("-G", PassThroughFlag),
    take_arg!("-Ofc", OsString, CanBeSeparated, PassThrough),
    take_arg!("-Ofc=", OsString, Concatenated, PassThrough),
    take_arg!("-Werror", OsString, CanBeSeparated(b'='), PreprocessorArgument),
    take_arg!("-Xarchive", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-Xcompiler", OsString, CanBeSeparated(b'='), PreprocessorArgument),
    take_arg!("-Xlinker", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-Xnvlink", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-Xptxas", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-Xtileiras", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-arch", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-ccbin", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-code", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-compress-mode", OsString, CanBeSeparated, PassThrough),
    take_arg!("-compress-mode=", OsString, Concatenated, PassThrough),
    flag!("-cubin", DoCompilation),
    flag!("-cuda", NotCompilationFlag),
    flag!("-dc", DoCompilation),
    take_arg!("-default-stream", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("-dlink", NotCompilationFlag),
    flag!("-dlto", PassThroughFlag),
    take_arg!("-dopt", OsString, CanBeSeparated, PassThrough),
    take_arg!("-dopt=", OsString, Concatenated, PassThrough),
    flag!("-dw", DoCompilation),
    flag!("-ewp", PassThroughFlag),
    flag!("-expt-extended-lambda", PreprocessorArgumentFlag),
    flag!("-expt-relaxed-constexpr", PreprocessorArgumentFlag),
    flag!("-extended-lambda", PreprocessorArgumentFlag),
    flag!("-fatbin", DoCompilation),
    take_arg!("-fdevice-time-trace", OsString, CanBeSeparated, TooHard),
    take_arg!("-fdevice-time-trace=", OsString, Concatenated, TooHard),
    flag!("-gen-opt-lto", PassThroughFlag),
    take_arg!("-gencode", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-isystem", PathBuf, CanBeSeparated(b'='), PreprocessorArgumentPath),
    flag!("-jobserver", UnhashedFlag),
    flag!("-keep", UnhashedFlag),
    take_arg!("-keep-dir", OsString, CanBeSeparated(b'='), Unhashed),
    flag!("-lto", PassThroughFlag),
    flag!("-ltoir", DoCompilation),
    take_arg!("-maxrregcount", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("-no-compress", PassThroughFlag),
    flag!("-nohdinitlist", PreprocessorArgumentFlag),
    flag!("-objtemp", UnhashedFlag),
    flag!("-optix-ir", DoCompilation),
    flag!("-pg", PassThroughFlag),
    flag!("-ptx", DoCompilation),
    take_arg!("-rdc", OsString, CanBeSeparated(b'='), PreprocessorArgument),
    flag!("-save-temps", UnhashedFlag),
    take_arg!("-split-compile", OsString, CanBeSeparated, Unhashed),
    take_arg!("-split-compile-extended", OsString, CanBeSeparated, Unhashed),
    take_arg!("-split-compile-extended=", OsString, Concatenated, Unhashed),
    take_arg!("-split-compile=", OsString, Concatenated, Unhashed),
    take_arg!("-t", OsString, CanBeSeparated, Unhashed),
    take_arg!("-t=", OsString, Concatenated, Unhashed),
    flag!("-tilebc", DoCompilation),
    flag!("-tilecubin", DoCompilation),
    take_arg!("-time", OsString, CanBeSeparated, TooHard),
    take_arg!("-time=", OsString, Concatenated, TooHard),
    take_arg!("-x", OsString, CanBeSeparated(b'='), Language),
]);

#[cfg(test)]
mod test {
    use super::*;
    use crate::compiler::Language;
    use crate::compiler::*;

    fn parse_arguments_gcc(arguments: Vec<String>) -> CompilerArguments<ParsedArguments> {
        let arguments = arguments.iter().map(OsString::from).collect::<Vec<_>>();
        Nvcc {
            archs_all: vec![],
            archs_major: vec![],
            archs_native: vec![],
            host_compiler: NvccHostCompiler::Gcc,
            host_compiler_version: None,
            version: None,
        }
        .parse_arguments(&arguments, ".".as_ref(), &[])
    }
    fn parse_arguments_msvc(arguments: Vec<String>) -> CompilerArguments<ParsedArguments> {
        let arguments = arguments.iter().map(OsString::from).collect::<Vec<_>>();
        Nvcc {
            archs_all: vec![],
            archs_major: vec![],
            archs_native: vec![],
            host_compiler: NvccHostCompiler::Msvc,
            host_compiler_version: None,
            version: None,
        }
        .parse_arguments(&arguments, ".".as_ref(), &[])
    }
    fn parse_arguments_nvc(arguments: Vec<String>) -> CompilerArguments<ParsedArguments> {
        let arguments = arguments.iter().map(OsString::from).collect::<Vec<_>>();
        Nvcc {
            archs_all: vec![],
            archs_major: vec![],
            archs_native: vec![],
            host_compiler: NvccHostCompiler::Nvhpc,
            host_compiler_version: None,
            version: None,
        }
        .parse_arguments(&arguments, ".".as_ref(), &[])
    }

    macro_rules! parses {
        ( $( $s:expr ),* ) => {
            match parse_arguments_gcc(vec![ $( $s.to_string(), )* ]) {
                CompilerArguments::Ok(a) => a,
                o => panic!("Got unexpected parse result: {:?}", o),
            }
        }
    }
    macro_rules! parses_msvc {
        ( $( $s:expr ),* ) => {
            match parse_arguments_msvc(vec![ $( $s.to_string(), )* ]) {
                CompilerArguments::Ok(a) => a,
                o => panic!("Got unexpected parse result: {:?}", o),
            }
        }
    }
    macro_rules! parses_nvc {
        ( $( $s:expr ),* ) => {
            match parse_arguments_nvc(vec![ $( $s.to_string(), )* ]) {
                CompilerArguments::Ok(a) => a,
                o => panic!("Got unexpected parse result: {:?}", o),
            }
        }
    }

    #[test]
    fn test_parse_arguments_simple_c() {
        let a = parses!("-c", "foo.c", "-o", "foo.o");
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::C, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert!(a.unhashed_args.is_empty());
        assert_eq!(ovec!["-c"], a.common_args);
        assert_eq!(None, a.out_dir);
    }

    #[test]
    fn test_parse_arguments_simple_c_objtemp() {
        let a = parses!("-c", "foo.c", "-o", "foo.o", "-objtemp");
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::C, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-objtemp"], a.unhashed_args);
        assert_eq!(ovec!["-c"], a.common_args);
        assert_eq!(Some(Path::new(".")), a.out_dir.as_deref().map(|o| o.path()));
    }

    #[test]
    fn test_parse_arguments_simple_cu_gcc() {
        let a = parses!("-c", "foo.cu", "-o", "foo.o");
        assert_eq!(Some("foo.cu"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-c"], a.common_args);
    }

    #[test]
    fn test_parse_arguments_simple_cu_nvc() {
        let a = parses_nvc!("-c", "foo.cu", "-o", "foo.o");
        assert_eq!(Some("foo.cu"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-c"], a.common_args);
    }

    #[test]
    fn test_parse_arguments_simple_cu_msvc() {
        let a = parses_msvc!("-c", "foo.cu", "-o", "foo.o");
        assert_eq!(Some("foo.cu"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-c"], a.common_args);
    }

    #[test]
    fn test_parse_arguments_ccbin_no_path() {
        let a = parses!("-ccbin=gcc", "-c", "foo.cu", "-o", "foo.o");
        assert_eq!(Some("foo.cu"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-ccbin", "gcc", "-c"], a.common_args);
    }

    #[test]
    fn test_parse_arguments_ccbin_dir() {
        let a = parses!("-ccbin=/usr/bin/", "-c", "foo.cu", "-o", "foo.o");
        assert_eq!(Some("foo.cu"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-ccbin", "/usr/bin/", "-c"], a.common_args);
    }

    #[test]
    fn test_parse_arguments_dependency_output() {
        let a = parses!("--dependency-output", "foo.o.d", "-c", "foo.cu");
        assert_eq!(Some("foo.cu"), a.input.to_str());
        assert_eq!(ovec!["--dependency-output", "foo.o.d"], a.dependency_args);
    }

    #[test]
    fn test_parse_threads_argument_simple_cu() {
        let a = parses!(
            "-t1",
            "-t=2",
            "-t",
            "3",
            "--threads=1",
            "--threads=2",
            "-c",
            "foo.cu",
            "-o",
            "foo.o"
        );
        assert_eq!(Some("foo.cu"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(
            ovec!["-t1", "-t=2", "-t3", "--threads", "1", "--threads", "2"],
            a.unhashed_args
        );
    }

    #[test]
    fn test_parse_arguments_simple_c_as_cu() {
        let a = parses!("-x", "cu", "-c", "foo.c", "-o", "foo.o");
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-c"], a.common_args);
    }

    #[test]
    fn test_parse_arguments_dc_compile_flag() {
        let a = parses!("-x", "cu", "-dc", "foo.c", "-o", "foo.o");
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_eq!(Some("-dc"), a.compilation_flag.to_str());
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-dc"], a.common_args);
    }

    #[test]
    fn test_parse_arguments_fatbin_compile_flag() {
        let a = parses!("-x", "cu", "-fatbin", "foo.c", "-o", "foo.o");
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_eq!(Some("-fatbin"), a.compilation_flag.to_str());
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-fatbin"], a.common_args);
    }

    #[test]
    fn test_parse_arguments_cubin_compile_flag() {
        let a = parses!("-x", "cu", "-cubin", "foo.c", "-o", "foo.o");
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_eq!(Some("-cubin"), a.compilation_flag.to_str());
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(ovec!["-cubin"], a.common_args);
    }

    #[test]
    fn test_parse_arguments_values() {
        let a = parses!(
            "-c",
            "foo.cpp",
            "-fabc",
            "-I",
            "include-file",
            "-o",
            "foo.o",
            "--include-path",
            "include-file",
            "-isystem=/system/include/file",
            "-Werror",
            "cross-execution-space-call",
            "-Werror=all-warnings"
        );
        assert_eq!(Some("foo.cpp"), a.input.to_str());
        assert_eq!(Language::Cxx, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert_eq!(
            ovec![
                "-Iinclude-file",
                "--include-path",
                "include-file",
                "-isystem",
                "/system/include/file",
                "-Werror",
                "cross-execution-space-call",
                "-Werror",
                "all-warnings"
            ],
            a.preprocessor_args
        );
        assert!(a.dependency_args.is_empty());
        assert_eq!(ovec!["-fabc", "-c"], a.common_args);
    }

    #[test]
    fn test_parse_md_mt_flags_cu() {
        let a = parses!(
            "-x", "cu", "-c", "foo.c", "-fabc", "-MD", "-MT", "foo.o", "-MF", "foo.o.d", "-o",
            "foo.o"
        );
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_eq!(Some("-c"), a.compilation_flag.to_str());
        assert_eq!(Some("foo.o.d"), a.depfile.unwrap().to_str());
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            ),
            (
                "dep",
                ArtifactDescriptor {
                    path: "foo.o.d".into(),
                    optional: false,
                }
            )
        );
        assert_eq!(
            ovec!["-MD", "-MF", "foo.o.d", "-MT", "foo.o"],
            a.dependency_args
        );
        assert_eq!(ovec!["-fabc", "-c"], a.common_args);
    }

    #[test]
    fn test_parse_generate_code_flags() {
        let a = parses!(
            "-x",
            "cu",
            "--generate-code=arch=compute_61,code=sm_61",
            "-c",
            "foo.c",
            "-o",
            "foo.o"
        );
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(a.preprocessor_args.is_empty());
        assert_eq!(
            ovec!["--generate-code", "arch=compute_61,code=sm_61", "-c"],
            a.common_args
        );
    }

    #[test]
    fn test_parse_pass_to_host_flags() {
        let a = parses!(
            "-x=cu",
            "--generate-code=arch=compute_60,code=[sm_60,sm_61]",
            "-Xnvlink=--suppress-stack-size-warning",
            "-Xcompiler",
            "-fPIC,-fno-common",
            "-Xcompiler=-fvisibility=hidden",
            "-Xcompiler=-Wall,-Wno-unknown-pragmas,-Wno-unused-local-typedefs",
            "-Xcudafe",
            "--display_error_number",
            "-c",
            "foo.c",
            "-o",
            "foo.o"
        );
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert_eq!(
            ovec![
                "-Xcompiler",
                "-fPIC,-fno-common",
                "-Xcompiler",
                "-fvisibility=hidden",
                "-Xcompiler",
                "-Wall,-Wno-unknown-pragmas,-Wno-unused-local-typedefs"
            ],
            a.preprocessor_args
        );
        assert_eq!(
            ovec![
                "--generate-code",
                "arch=compute_60,code=[sm_60,sm_61]",
                "-Xnvlink",
                "--suppress-stack-size-warning",
                "-Xcudafe",
                "--display_error_number",
                "-c"
            ],
            a.common_args
        );
    }

    #[test]
    fn test_parse_no_capturing_of_xcompiler() {
        let a = parses!(
            "-x=cu",
            "-forward-unknown-to-host-compiler",
            "--expt-relaxed-constexpr",
            "-Xcompiler",
            "-pthread",
            "-std=c++14",
            "-c",
            "foo.c",
            "-o",
            "foo.o"
        );
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert_eq!(
            ovec!["--expt-relaxed-constexpr", "-Xcompiler", "-pthread"],
            a.preprocessor_args
        );
        assert_eq!(
            ovec!["-forward-unknown-to-host-compiler", "-std=c++14", "-c"],
            a.common_args
        );
    }

    #[test]
    fn test_parse_dlink_is_not_compilation() {
        assert_eq!(
            CompilerArguments::NotCompilation,
            parse_arguments_gcc(stringvec![
                "-forward-unknown-to-host-compiler",
                "--generate-code=arch=compute_50,code=[compute_50,sm_50,sm_52]",
                "-dlink",
                "main.cu.o",
                "-o",
                "device_link.o"
            ])
        );
        assert_eq!(
            CompilerArguments::NotCompilation,
            parse_arguments_nvc(stringvec![
                "-forward-unknown-to-host-compiler",
                "--generate-code=arch=compute_50,code=[compute_50,sm_50,sm_52]",
                "-dlink",
                "main.cu.o",
                "-o",
                "device_link.o"
            ])
        );
    }

    #[test]
    fn test_parse_cant_cache_flags() {
        assert_eq!(
            CompilerArguments::CannotCache("-E", None),
            parse_arguments_gcc(stringvec!["-x", "cu", "-c", "foo.c", "-o", "foo.o", "-E"])
        );
        assert_eq!(
            CompilerArguments::CannotCache("-E", None),
            parse_arguments_msvc(stringvec!["-x", "cu", "-c", "foo.c", "-o", "foo.o", "-E"])
        );
        assert_eq!(
            CompilerArguments::CannotCache("-E", None),
            parse_arguments_nvc(stringvec!["-x", "cu", "-c", "foo.c", "-o", "foo.o", "-E"])
        );

        assert_eq!(
            CompilerArguments::CannotCache("-M", None),
            parse_arguments_gcc(stringvec!["-x", "cu", "-c", "foo.c", "-o", "foo.o", "-M"])
        );
        assert_eq!(
            CompilerArguments::CannotCache("-M", None),
            parse_arguments_msvc(stringvec!["-x", "cu", "-c", "foo.c", "-o", "foo.o", "-M"])
        );
        assert_eq!(
            CompilerArguments::CannotCache("-M", None),
            parse_arguments_nvc(stringvec!["-x", "cu", "-c", "foo.c", "-o", "foo.o", "-M"])
        );

        // nvcc arg parsing is very permissive, so all these are valid and should yield CannotCache
        for arg in [
            "-fdevice-time-trace",
            "--fdevice-time-trace",
            "-time",
            "--time",
        ] {
            // {-,--}fdevice-time-trace
            assert_eq!(
                CompilerArguments::CannotCache(arg, None),
                parse_arguments_gcc(stringvec![arg])
            );
            // {-,--}fdevice-time-trace -
            assert_eq!(
                CompilerArguments::CannotCache(arg, None),
                parse_arguments_msvc(stringvec![arg, "-"])
            );
            // {-,--}fdevice-time-trace-
            assert_eq!(
                CompilerArguments::CannotCache(arg, None),
                parse_arguments_msvc(stringvec![format!("{arg}-")])
            );
            // {-,--}fdevice-time-trace flamegraph.json
            assert_eq!(
                CompilerArguments::CannotCache(arg, None),
                parse_arguments_nvc(stringvec![arg, "flamegraph.json"])
            );
        }

        for arg_with_separator in [
            "-fdevice-time-trace=",
            "--fdevice-time-trace=",
            "-time=",
            "--time=",
        ] {
            // {-,--}fdevice-time-trace=-
            assert_eq!(
                CompilerArguments::CannotCache(arg_with_separator, None),
                parse_arguments_msvc(stringvec![format!("{arg_with_separator}-")])
            );
            // {-,--}fdevice-time-trace=flamegraph.json
            assert_eq!(
                CompilerArguments::CannotCache(arg_with_separator, None),
                parse_arguments_nvc(stringvec![format!("{arg_with_separator}flamegraph.json")])
            );
        }
    }

    #[test]
    fn test_parse_device_debug_flag_short_cu() {
        let a = parses!(
            "-x", "cu", "-c", "foo.c", "-G", "-MD", "-MT", "foo.o", "-MF", "foo.o.d", "-o", "foo.o"
        );
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_eq!(Some("-c"), a.compilation_flag.to_str());
        assert_eq!(Some("foo.o.d"), a.depfile.unwrap().to_str());
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            ),
            (
                "dep",
                ArtifactDescriptor {
                    path: "foo.o.d".into(),
                    optional: false,
                }
            )
        );
        assert_eq!(
            ovec!["-MD", "-MF", "foo.o.d", "-MT", "foo.o"],
            a.dependency_args
        );
        assert_eq!(ovec!["-G", "-c"], a.common_args);
    }

    #[test]
    fn test_parse_device_debug_flag_long_cu() {
        let a = parses!(
            "-x",
            "cu",
            "-c",
            "foo.c",
            "--device-debug",
            "-MD",
            "-MT",
            "foo.o",
            "-MF",
            "foo.o.d",
            "-o",
            "foo.o"
        );
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cuda, a.language);
        assert_eq!(Some("-c"), a.compilation_flag.to_str());
        assert_eq!(Some("foo.o.d"), a.depfile.unwrap().to_str());
        assert_map_contains!(
            a.outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            ),
            (
                "dep",
                ArtifactDescriptor {
                    path: "foo.o.d".into(),
                    optional: false,
                }
            )
        );
        assert_eq!(
            ovec!["-MD", "-MF", "foo.o.d", "-MT", "foo.o"],
            a.dependency_args
        );
        assert_eq!(ovec!["--device-debug", "-c"], a.common_args);
    }

    #[test]
    fn test_rdc_ltoir_object() {
        drop(env_logger::try_init());

        let cwd = Path::new("");
        let out = Path::new("/tmp/out");
        let compile_flag = &NvccCompileFlag::Device;
        let host_compiler = &NvccHostCompiler::Gcc;
        let mut env_vars = vec![];
        let mut nvcc_internal_files = HashMap::<String, String>::new();

        // Get the nvcc compile command lines with paths relative to `out`
        let nvcc_commands = parse_nvcc_subcommands(
            cwd,
            &mut env_vars,
            compile_flag,
            host_compiler,
            &mut nvcc_internal_files,
            is_nvcc_exe,
            // nvcc -rdc=true -gencode=arch=lto_75,code=lto_75 -dc x.cu -o x.ltoir.o --dryrun --keep
            r#"
#$ gcc -D__CUDA_ARCH_LIST__=750 -E -x c++ -D__CUDACC__ -D__NVCC__ -D__CUDACC_RDC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=78 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=2 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "x.cpp4.ii"
#$ cudafe++ --c++17 --device-hidden-visibility --gnu_version=130300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed  --device-c  --m64 --parse_templates --gen_c_file_name "x.cudafe1.cpp" --stub_file_name "x.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "x.module_id" "x.cpp4.ii"
#$ gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__ -D__CUDACC_RDC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=78 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=2 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "x.cpp1.ii"
#$ "$CICC_PATH/cicc" --c++17 --device-hidden-visibility --gnu_version=130300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed  --device-c   -arch compute_75 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "x.fatbin.c"  -tused --module_id_file_name "x.module_id" --gen_c_file_name "x.cudafe1.c" --stub_file_name "x.cudafe1.stub.c" --gen_device_file_name "x.cudafe1.gpu"  "x.cpp1.ii" -lto -o "x.ltoir"
#$ fatbinary --create="x.fatbin" -64 --cmdline="--compile-only  " --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=nvvm,sm=75,file=x.ltoir" --embedded-fatbin="x.fatbin.c"  --device-c
#$ gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -Wno-psabi "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"   -m64 "x.cudafe1.cpp" -o "x.ltoir.o"
"#,
        ).unwrap();

        let host_commands = parse_nvcc_subcommands(
            cwd,
            &mut env_vars,
            compile_flag,
            host_compiler,
            &mut nvcc_internal_files,
            |exe, args| !is_nvcc_exe(exe, args),
            // nvcc -rdc=true -gencode=arch=lto_75,code=lto_75 -dc x.cu -o x.ltoir.o --dryrun --keep --keep-dir /tmp/out
            r#"
#$ gcc -D__CUDA_ARCH_LIST__=750 -E -x c++ -D__CUDACC__ -D__NVCC__ -D__CUDACC_RDC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=78 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=2 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "/tmp/out/x.cpp4.ii"
#$ cudafe++ --c++17 --device-hidden-visibility --gnu_version=130300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed  --device-c  --m64 --parse_templates --gen_c_file_name "/tmp/out/x.cudafe1.cpp" --stub_file_name "x.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "/tmp/out/x.module_id" "/tmp/out/x.cpp4.ii"
#$ gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__ -D__CUDACC_RDC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=78 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=2 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "/tmp/out/x.cpp1.ii"
#$ "$CICC_PATH/cicc" --c++17 --device-hidden-visibility --gnu_version=130300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed  --device-c   -arch compute_75 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "x.fatbin.c"  -tused --module_id_file_name "/tmp/out/x.module_id" --gen_c_file_name "/tmp/out/x.cudafe1.c" --stub_file_name "/tmp/out/x.cudafe1.stub.c" --gen_device_file_name "/tmp/out/x.cudafe1.gpu"  "/tmp/out/x.cpp1.ii" -lto -o "/tmp/out/x.ltoir"
#$ fatbinary --create="/tmp/out/x.fatbin" -64 --cmdline="--compile-only  " --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=nvvm,sm=75,file=/tmp/out/x.ltoir" --embedded-fatbin="/tmp/out/x.fatbin.c"  --device-c
#$ gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -Wno-psabi "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"   -m64 "/tmp/out/x.cudafe1.cpp" -o "x.ltoir.o"
"#,
        ).unwrap();

        // Merge the nvcc and host compiler commands into one list
        let (cudafe_has_gen_module_id_file_flag, all_commands) =
            merge_nvcc_and_host_compiler_commands(cwd, out, nvcc_commands, host_commands);

        let command_groups = create_nvcc_commands_graph(
            compile_flag,
            None,
            env_vars,
            host_compiler,
            cudafe_has_gen_module_id_file_flag,
            all_commands,
            Path::new("x.ltoir.o"),
        );

        let output = command_groups
            .last()
            .and_then(|cmds| cmds.last())
            .and_then(|cmd| cmd.output.as_deref());

        // Ensure `remap_generated_filenames` doesn't rewrite `-o x.ltoir.o` to `-o x.compute_75.ltoir.o`
        assert_eq!(output, Some(Path::new("x.ltoir.o")));
    }

    #[tokio::test]
    async fn test_parse_nvcc_lines_with_stdio_redirects() -> Result<()> {
        drop(env_logger::try_init());

        let mut nvcc_internal_files = HashMap::<String, String>::new();

        let commands = parse_nvcc_subcommands(
            Path::new(""),
            &mut vec![],
            &NvccCompileFlag::Device,
            &NvccHostCompiler::Gcc,
            &mut nvcc_internal_files,
            |_, _| true,
            // echo | nvcc -c -x cu --enable-tile - -o /tmp/x/x.cu.o --dryrun --keep --keep-dir /tmp/out
            r#"
#$ gcc -D__NV_TL__=__tile__ -D__NV_TL_BUILTIN__=__tile_builtin__ -D__CUDACC_TILE__=1 -D__CUDA_ARCH_LIST__=750 -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=0 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "/tmp/tmpxft_00000080_00000000-1_stdin" -o "/tmp/out/tmpxft_00000080_00000000-1_stdin.cpp4.ii"
#$ cudafe++ --c++17 --static-host-stub --device-hidden-visibility --gnu_version=140300 --display_error_number --orig_src_file_name "/tmp/tmpxft_00000080_00000000-1_stdin" --orig_src_path_name "/tmp/tmpxft_00000080_00000000-1_stdin" --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/out/tmpxft_00000080_00000000-1_stdin.cudafe1.cpp" --stub_file_name "tmpxft_00000080_00000000-1_stdin.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "/tmp/out/tmpxft_00000080_00000000-1_stdin.module_id" --enable-tile "/tmp/out/tmpxft_00000080_00000000-1_stdin.cpp4.ii"
#$ fatbinary --create="/tmp/out/tmpxft_00000080_00000000-1_stdin.tile.fatbin" -64 --embedded-fatbin="/tmp/out/tmpxft_00000080_00000000-1_stdin.tile.fatbin.c" --id-suffix="Alt"
#$ gcc -D__NV_TL__=__tile__ -D__NV_TL_BUILTIN__=__tile_builtin__ -D__CUDACC_TILE__=1 -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=0 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "/tmp/tmpxft_00000080_00000000-1_stdin" -o "/tmp/out/tmpxft_00000080_00000000-1_stdin.cpp1.ii"
#$ "$CICC_PATH/cicc" --c++17 --static-host-stub --device-hidden-visibility --gnu_version=140300 --display_error_number --orig_src_file_name "/tmp/tmpxft_00000080_00000000-1_stdin" --orig_src_path_name "/tmp/tmpxft_00000080_00000000-1_stdin" --allow_managed   -arch compute_75 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_00000080_00000000-1_stdin.fatbin.c" --include_alt_file_name "tmpxft_00000080_00000000-1_stdin.tile.fatbin.c" -tused --module_id_file_name "/tmp/out/tmpxft_00000080_00000000-1_stdin.module_id" --gen_c_file_name "/tmp/out/tmpxft_00000080_00000000-1_stdin.cudafe1.c" --stub_file_name "/tmp/out/tmpxft_00000080_00000000-1_stdin.cudafe1.stub.c" --gen_device_file_name "/tmp/out/tmpxft_00000080_00000000-1_stdin.cudafe1.gpu"  "/tmp/out/tmpxft_00000080_00000000-1_stdin.cpp1.ii" --enable-tile --tile_bc_file_name "/tmp/out/tmpxft_00000080_00000000-1_stdin.tilebc" -o "/tmp/out/tmpxft_00000080_00000000-1_stdin.ptx"
#$ ptxas -arch=sm_75 -m64  "/tmp/out/tmpxft_00000080_00000000-1_stdin.ptx"  -o "/tmp/out/tmpxft_00000080_00000000-1_stdin.sm_75.cubin"
#$ fatbinary --create="/tmp/out/tmpxft_00000080_00000000-1_stdin.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=75,file=/tmp/out/tmpxft_00000080_00000000-1_stdin.sm_75.cubin" "--image3=kind=ptx,sm=75,file=/tmp/out/tmpxft_00000080_00000000-1_stdin.ptx" --embedded-fatbin="/tmp/out/tmpxft_00000080_00000000-1_stdin.fatbin.c"
#$ gcc -D__CUDA_ARCH__=750 -D__CUDA_ARCH_LIST__=750 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -Wno-psabi "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"   -m64 "/tmp/out/tmpxft_00000080_00000000-1_stdin.cudafe1.cpp" -o "/tmp/x/x.cu.o"
"#,
        )?;

        assert!(
            commands
                .iter()
                .all(|(_, _, command)| command.iter().all(|arg| arg != ">"))
        );
        assert!(
            commands
                .iter()
                .all(|(_, _, command)| command.iter().all(|arg| !arg.starts_with("2>")))
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_group_cuda_tile_commands() -> Result<()> {
        drop(env_logger::try_init());

        let keep_dir = None;
        let mut env_vars = vec![];
        let cwd = Path::new("").to_owned();
        let out = Path::new("/tmp/out").to_owned();

        let compile_flag = &NvccCompileFlag::Device;
        let host_compiler = &NvccHostCompiler::Gcc;

        let mut nvcc_internal_files = HashMap::<String, String>::new();

        // Get the nvcc compile command lines with paths relative to `out`
        let nvcc_commands = parse_nvcc_subcommands(
            &out,
            &mut env_vars,
            compile_flag,
            host_compiler,
            &mut nvcc_internal_files,
            is_nvcc_exe,
            // nvcc -x cu --enable-tile  -gencode=arch=compute_80,code=[compute_80,sm_80] -gencode=arch=compute_86,code=[compute_86,sm_86] -c x.cu -o x.cu.o --dryrun --keep
            r#"
#$ gcc -D__NV_TL__=__tile__ -D__NV_TL_BUILTIN__=__tile_builtin__ -D__CUDACC_TILE__=1 -D__CUDA_ARCH_LIST__=800,860 -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=0 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "x.cpp4.ii"
#$ cudafe++ --c++17 --static-host-stub --device-hidden-visibility --gnu_version=140300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed  --m64 --parse_templates --gen_c_file_name "x.compute_86.cudafe1.cpp" --stub_file_name "x.compute_86.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "x.module_id" --enable-tile "x.cpp4.ii"
#$ gcc -D__NV_TL__=__tile__ -D__NV_TL_BUILTIN__=__tile_builtin__ -D__CUDACC_TILE__=1 -D__CUDA_ARCH__=800 -D__CUDA_ARCH_LIST__=800,860 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=0 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "x.compute_80.cpp1.ii"
#$ "$CICC_PATH/cicc" --c++17 --static-host-stub --device-hidden-visibility --gnu_version=140300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed   -arch compute_80 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "x.fatbin.c" --include_alt_file_name "x.tile.fatbin.c" -tused --module_id_file_name "x.module_id" --gen_c_file_name "x.compute_80.cudafe1.c" --stub_file_name "x.compute_80.cudafe1.stub.c" --gen_device_file_name "x.compute_80.cudafe1.gpu"  "x.compute_80.cpp1.ii" --enable-tile --tile_bc_file_name "x.compute_80.tilebc" -o "x.compute_80.ptx"
#$ tileiras --host-arch=x86_64 --host-os=linux -arch=sm_80  "x.compute_80.tilebc"  -o "x.compute_80.tile.cubin"
#$ gcc -D__NV_TL__=__tile__ -D__NV_TL_BUILTIN__=__tile_builtin__ -D__CUDACC_TILE__=1 -D__CUDA_ARCH__=860 -D__CUDA_ARCH_LIST__=800,860 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=0 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "x.compute_86.cpp1.ii"
#$ "$CICC_PATH/cicc" --c++17 --static-host-stub --device-hidden-visibility --gnu_version=140300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed   -arch compute_86 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "x.fatbin.c" --include_alt_file_name "x.tile.fatbin.c" -tused --module_id_file_name "x.module_id" --gen_c_file_name "x.compute_86.cudafe1.c" --stub_file_name "x.compute_86.cudafe1.stub.c" --gen_device_file_name "x.compute_86.cudafe1.gpu"  "x.compute_86.cpp1.ii" --enable-tile --tile_bc_file_name "x.compute_86.tilebc" -o "x.compute_86.ptx"
#$ tileiras --host-arch=x86_64 --host-os=linux -arch=sm_86  "x.compute_86.tilebc"  -o "x.compute_86.tile.cubin"
#$ fatbinary --create="x.tile.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=tileir,sm=80,file=x.compute_80.tilebc" "--image3=kind=elf,sm=80,file=x.compute_80.tile.cubin" "--image3=kind=tileir,sm=86,file=x.compute_86.tilebc" "--image3=kind=elf,sm=86,file=x.compute_86.tile.cubin" --embedded-fatbin="x.tile.fatbin.c" --id-suffix="Alt"
#$ ptxas -arch=sm_80 -m64  "x.compute_80.ptx"  -o "x.compute_80.sm_80.cubin"
#$ ptxas -arch=sm_86 -m64  "x.compute_86.ptx"  -o "x.compute_86.sm_86.cubin"
#$ fatbinary --create="x.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=ptx,sm=80,file=x.compute_80.ptx" "--image3=kind=elf,sm=80,file=x.compute_80.sm_80.cubin" "--image3=kind=ptx,sm=86,file=x.compute_86.ptx" "--image3=kind=elf,sm=86,file=x.compute_86.sm_86.cubin" --embedded-fatbin="x.fatbin.c"
#$ gcc -D__CUDA_ARCH__=860 -D__CUDA_ARCH_LIST__=800,860 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -Wno-psabi "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"   -m64 "x.compute_86.cudafe1.cpp" -o "x.cu.o"
"#,
        )?;

        // Get the host compile command lines with paths relative to `cwd` and absolute paths to `out`
        let host_commands = parse_nvcc_subcommands(
            &cwd,
            &mut env_vars,
            compile_flag,
            host_compiler,
            &mut nvcc_internal_files,
            |exe, args| !is_nvcc_exe(exe, args),
            // nvcc -x cu --enable-tile  -gencode=arch=compute_80,code=[compute_80,sm_80] -gencode=arch=compute_86,code=[compute_86,sm_86] -c x.cu -o x.cu.o --dryrun --keep --keep-dir /tmp/out
            r#"
#$ gcc -D__NV_TL__=__tile__ -D__NV_TL_BUILTIN__=__tile_builtin__ -D__CUDACC_TILE__=1 -D__CUDA_ARCH_LIST__=800,860 -E -x c++ -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=0 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "/tmp/out/x.cpp4.ii"
#$ cudafe++ --c++17 --static-host-stub --device-hidden-visibility --gnu_version=140300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed  --m64 --parse_templates --gen_c_file_name "/tmp/out/x.compute_86.cudafe1.cpp" --stub_file_name "x.compute_86.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "/tmp/out/x.module_id" --enable-tile "/tmp/out/x.cpp4.ii"
#$ gcc -D__NV_TL__=__tile__ -D__NV_TL_BUILTIN__=__tile_builtin__ -D__CUDACC_TILE__=1 -D__CUDA_ARCH__=800 -D__CUDA_ARCH_LIST__=800,860 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=0 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "/tmp/out/x.compute_80.cpp1.ii"
#$ "$CICC_PATH/cicc" --c++17 --static-host-stub --device-hidden-visibility --gnu_version=140300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed   -arch compute_80 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "x.fatbin.c" --include_alt_file_name "x.tile.fatbin.c" -tused --module_id_file_name "/tmp/out/x.module_id" --gen_c_file_name "/tmp/out/x.compute_80.cudafe1.c" --stub_file_name "/tmp/out/x.compute_80.cudafe1.stub.c" --gen_device_file_name "/tmp/out/x.compute_80.cudafe1.gpu"  "/tmp/out/x.compute_80.cpp1.ii" --enable-tile --tile_bc_file_name "/tmp/out/x.compute_80.tilebc" -o "/tmp/out/x.compute_80.ptx"
#$ tileiras --host-arch=x86_64 --host-os=linux -arch=sm_80  "/tmp/out/x.compute_80.tilebc"  -o "/tmp/out/x.compute_80.tile.cubin"
#$ gcc -D__NV_TL__=__tile__ -D__NV_TL_BUILTIN__=__tile_builtin__ -D__CUDACC_TILE__=1 -D__CUDA_ARCH__=860 -D__CUDA_ARCH_LIST__=800,860 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"    -D__CUDACC_VER_MAJOR__=13 -D__CUDACC_VER_MINOR__=4 -D__CUDACC_VER_BUILD__=0 -D__CUDA_API_VER_MAJOR__=13 -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 -include "cuda_runtime.h" -m64 "x.cu" -o "/tmp/out/x.compute_86.cpp1.ii"
#$ "$CICC_PATH/cicc" --c++17 --static-host-stub --device-hidden-visibility --gnu_version=140300 --display_error_number --orig_src_file_name "x.cu" --orig_src_path_name "x.cu" --allow_managed   -arch compute_86 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "x.fatbin.c" --include_alt_file_name "x.tile.fatbin.c" -tused --module_id_file_name "/tmp/out/x.module_id" --gen_c_file_name "/tmp/out/x.compute_86.cudafe1.c" --stub_file_name "/tmp/out/x.compute_86.cudafe1.stub.c" --gen_device_file_name "/tmp/out/x.compute_86.cudafe1.gpu"  "/tmp/out/x.compute_86.cpp1.ii" --enable-tile --tile_bc_file_name "/tmp/out/x.compute_86.tilebc" -o "/tmp/out/x.compute_86.ptx"
#$ tileiras --host-arch=x86_64 --host-os=linux -arch=sm_86  "/tmp/out/x.compute_86.tilebc"  -o "/tmp/out/x.compute_86.tile.cubin"
#$ fatbinary --create="/tmp/out/x.tile.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=tileir,sm=80,file=/tmp/out/x.compute_80.tilebc" "--image3=kind=elf,sm=80,file=/tmp/out/x.compute_80.tile.cubin" "--image3=kind=tileir,sm=86,file=/tmp/out/x.compute_86.tilebc" "--image3=kind=elf,sm=86,file=/tmp/out/x.compute_86.tile.cubin" --embedded-fatbin="/tmp/out/x.tile.fatbin.c" --id-suffix="Alt"
#$ ptxas -arch=sm_80 -m64  "/tmp/out/x.compute_80.ptx"  -o "/tmp/out/x.compute_80.sm_80.cubin"
#$ ptxas -arch=sm_86 -m64  "/tmp/out/x.compute_86.ptx"  -o "/tmp/out/x.compute_86.sm_86.cubin"
#$ fatbinary --create="/tmp/out/x.fatbin" -64 --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=ptx,sm=80,file=/tmp/out/x.compute_80.ptx" "--image3=kind=elf,sm=80,file=/tmp/out/x.compute_80.sm_80.cubin" "--image3=kind=ptx,sm=86,file=/tmp/out/x.compute_86.ptx" "--image3=kind=elf,sm=86,file=/tmp/out/x.compute_86.sm_86.cubin" --embedded-fatbin="/tmp/out/x.fatbin.c"
#$ gcc -D__CUDA_ARCH__=860 -D__CUDA_ARCH_LIST__=800,860 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -Wno-psabi "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   "-isystem" "/usr/local/cuda/bin/../targets/x86_64-linux/include/cccl"   -m64 "/tmp/out/x.compute_86.cudafe1.cpp" -o "x.cu.o"
"#,
        )?;

        // Merge the nvcc and host compiler commands into one list
        let (cudafe_has_gen_module_id_file_flag, all_commands) =
            merge_nvcc_and_host_compiler_commands(&cwd, &out, nvcc_commands, host_commands);

        let command_groups = create_nvcc_commands_graph(
            compile_flag,
            keep_dir,
            env_vars,
            host_compiler,
            cudafe_has_gen_module_id_file_flag,
            all_commands,
            Path::new("x.cu.o"),
        );

        assert!(command_groups[0][0].exe.as_os_str().contains("gcc"));
        assert!(command_groups[0][1].exe.as_os_str().contains("cudafe++"));

        assert!(command_groups[1][0].exe.as_os_str().contains("gcc"));
        assert!(command_groups[1][1].exe.as_os_str().contains("cicc"));
        assert!(command_groups[1][2].exe.as_os_str().contains("tileiras"));
        assert!(command_groups[1][3].exe.as_os_str().contains("ptxas"));

        assert!(command_groups[2][0].exe.as_os_str().contains("gcc"));
        assert!(command_groups[2][1].exe.as_os_str().contains("cicc"));
        assert!(command_groups[2][2].exe.as_os_str().contains("tileiras"));
        assert!(command_groups[2][3].exe.as_os_str().contains("ptxas"));

        assert!(command_groups[3][0].exe.as_os_str().contains("fatbinary"));
        assert!(command_groups[3][1].exe.as_os_str().contains("fatbinary"));
        assert!(command_groups[3][2].exe.as_os_str().contains("gcc"));

        Ok(())
    }
}
