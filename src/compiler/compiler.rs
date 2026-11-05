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

#[cfg(feature = "dist-client")]
use crate::{cache::UnexpectedFileSize, dist::pkg, lru_disk_cache};

use crate::{
    cache::{Cache, CacheRead, CacheWrite, DecompressionFailure, FileObjectSource, Storage},
    compiler::{
        args::*,
        c::{AnyCCompiler, CCompiler, CCompilerKind},
        cicc::Cicc,
        clang::Clang,
        cudafe::CudaFE,
        diab::Diab,
        gcc::Gcc,
        llc::Llc,
        msvc::{self, Msvc},
        nvcc::{Nvcc, NvccHostCompiler},
        nvhpc::Nvhpc,
        ptxas::Ptxas,
        rust::{Rust, RustupProxy},
        tasking_vx::TaskingVX,
        tileiras::Tileiras,
    },
    counted_array, dist,
    errors::*,
    mock_command::{CommandCreatorSync, ProcessOutput, RunCommand},
    server,
    util::{
        Digest, bytes_to_str, fmt_duration_as_secs, resolve_compiler_avoiding_wrapper,
        run_input_output,
    },
};

use async_trait::async_trait;
use filetime::FileTime;
use fs::File;
use fs_err as fs;
use futures::TryFutureExt;
use is_executable::IsExecutable;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::future::Future;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::str;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Can dylibs (shared libraries or proc macros) be distributed on this platform?
#[cfg(all(
    feature = "dist-client",
    any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(
            target_os = "windows",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "freebsd"
    )
))]
pub const CAN_DIST_DYLIBS: bool = true;
#[cfg(all(
    feature = "dist-client",
    not(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(
            target_os = "windows",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "freebsd"
    ))
))]
pub const CAN_DIST_DYLIBS: bool = false;

#[async_trait]
pub trait CompileCommand<T>: Send + Sync + 'static
where
    T: CommandCreatorSync,
{
    async fn execute(
        &self,
        service: &server::SccacheService<T>,
        creator: &T,
    ) -> Result<ProcessOutput>;

    fn get_executable(&self) -> PathBuf;
    fn get_arguments(&self) -> Vec<OsString>;
    fn get_env_vars(&self) -> Vec<(OsString, OsString)>;
    fn get_cwd(&self) -> PathBuf;
    fn box_clone(&self) -> Box<dyn CompileCommand<T>>;
}

impl<T: CommandCreatorSync> Clone for Box<dyn CompileCommand<T>> {
    fn clone(&self) -> Box<dyn CompileCommand<T>> {
        self.box_clone()
    }
}

#[derive(Debug, Clone)]
pub struct CompilerCommand<I>
where
    I: CompileCommandImpl,
{
    cmd: I,
}

impl<I> CompilerCommand<I>
where
    I: CompileCommandImpl,
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new<T>(cmd: I) -> Box<dyn CompileCommand<T>>
    where
        T: CommandCreatorSync,
    {
        Box::new(CompilerCommand { cmd }) as Box<dyn CompileCommand<T>>
    }
}

#[async_trait]
impl<T, I> CompileCommand<T> for CompilerCommand<I>
where
    T: CommandCreatorSync,
    I: CompileCommandImpl,
{
    fn get_executable(&self) -> PathBuf {
        self.cmd.get_executable()
    }
    fn get_arguments(&self) -> Vec<OsString> {
        self.cmd.get_arguments()
    }
    fn get_env_vars(&self) -> Vec<(OsString, OsString)> {
        self.cmd.get_env_vars()
    }
    fn get_cwd(&self) -> PathBuf {
        self.cmd.get_cwd()
    }

    async fn execute(
        &self,
        service: &server::SccacheService<T>,
        creator: &T,
    ) -> Result<ProcessOutput> {
        self.cmd.execute(service, creator).await
    }

    fn box_clone(&self) -> Box<dyn CompileCommand<T>> {
        CompilerCommand::<I>::new(self.cmd.clone())
    }
}

#[async_trait]
pub trait CompileCommandImpl: Send + Sync + Clone + 'static {
    fn get_executable(&self) -> PathBuf;
    fn get_arguments(&self) -> Vec<OsString>;
    fn get_env_vars(&self) -> Vec<(OsString, OsString)>;
    fn get_cwd(&self) -> PathBuf;

    async fn execute<T>(
        &self,
        service: &server::SccacheService<T>,
        creator: &T,
    ) -> Result<ProcessOutput>
    where
        T: CommandCreatorSync;
}

#[derive(Debug, Clone)]
pub struct SingleCompileCommand {
    pub arguments: Vec<OsString>,
    pub cwd: PathBuf,
    pub env_vars: Vec<(OsString, OsString)>,
    pub executable: PathBuf,
    pub out_pretty: String,
    /// Whether this compiler participates in the GNU make jobserver.
    ///
    /// Deliberately a field rather than a defaulted builder method: the
    /// compiler then makes every frontend answer, so a new one cannot
    /// silently get this wrong. Getting it wrong in the `true` direction
    /// costs a `fork` per compile; in the `false` direction it costs `rustc`
    /// its parallelism limit, which is what the jobserver exists to enforce.
    pub share_jobserver: bool,
}

impl fmt::Display for SingleCompileCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, r#"cd "{}"; "#, self.cwd.display())?;
        if cfg!(target_os = "windows") {
            write!(f, "& ")?;
        }
        write!(f, r#""{}""#, self.executable.display())?;
        for arg in self.arguments.iter() {
            write!(f, " {arg:?}")?;
        }
        fmt::Result::Ok(())
    }
}

#[async_trait]
impl CompileCommandImpl for SingleCompileCommand {
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

    async fn execute<T>(
        &self,
        service: &server::SccacheService<T>,
        creator: &T,
    ) -> Result<ProcessOutput>
    where
        T: CommandCreatorSync,
    {
        let SingleCompileCommand {
            arguments,
            cwd,
            env_vars,
            executable,
            out_pretty,
            share_jobserver,
        } = self;

        trace!("[{out_pretty}]: Compiling locally");

        let mut cmd = creator.clone().new_command_sync(executable);

        cmd.args(arguments)
            .env_clear()
            .current_dir(cwd)
            .envs(env_vars.iter().map(|(k, v)| (k, v)));

        if *share_jobserver {
            cmd.share_jobserver();
        }

        // Increment active_compilations after spawn() acquires the jobserver token
        let _active = service.increment_active_compilations();

        run_input_output(cmd, None).await
    }
}

/// Supported compilers.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum CompilerKind {
    /// A C compiler.
    C(CCompilerKind),
    /// A Rust compiler.
    Rust,
}

// Used for tests
// Need to be consistent with `Language`
#[allow(dead_code)]
const EXPECTED_LANGUAGE_COUNT: usize = 21;

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub enum Language {
    AssemblerToPreprocess,
    Assembler,
    #[default]
    C,
    Cxx,
    GenericHeader,
    CHeader,
    CPreprocessed,
    CxxHeader,
    CxxPreprocessed,
    ObjectiveC,
    ObjectiveCHeader,
    ObjectiveCPreprocessed,
    ObjectiveCxx,
    ObjectiveCxxPreprocessed,
    ObjectiveCxxHeader,
    Cuda,
    CudaFE,
    Ptx,
    Cubin,
    Rust,
    Hip,
    CxxModule,
}

impl Language {
    pub fn from_file_name(file: &Path) -> Option<Self> {
        match file.extension().and_then(|e| e.to_str()) {
            // gcc: https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html
            Some("s") => Some(Language::Assembler),
            Some("S") | Some("sx") => Some(Language::AssemblerToPreprocess),
            Some("c") => Some(Language::C),
            // Could be C or C++
            Some("h") => Some(Language::GenericHeader),
            Some("i") => Some(Language::CPreprocessed),
            Some("C") | Some("cc") | Some("cp") | Some("cpp") | Some("CPP") | Some("cxx")
            | Some("c++") => Some(Language::Cxx),
            Some("ii") => Some(Language::CxxPreprocessed),
            Some("H") | Some("hh") | Some("hp") | Some("hpp") | Some("HPP") | Some("hxx")
            | Some("h++") | Some("tcc") => Some(Language::CxxHeader),
            Some("cppm") | Some("ixx") => Some(Language::CxxModule),
            Some("m") => Some(Language::ObjectiveC),
            Some("mi") => Some(Language::ObjectiveCPreprocessed),
            Some("M") | Some("mm") => Some(Language::ObjectiveCxx),
            Some("mii") => Some(Language::ObjectiveCxxPreprocessed),
            Some("cu") => Some(Language::Cuda),
            Some("ptx") => Some(Language::Ptx),
            Some("cubin") => Some(Language::Cubin),
            // TODO cy
            Some("rs") => Some(Language::Rust),
            Some("hip") => Some(Language::Hip),
            e => {
                trace!("Unknown source extension: {}", e.unwrap_or("(None)"));
                None
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Language::AssemblerToPreprocess => "assemblerToPreprocess",
            Language::Assembler => "assembler",
            Language::C => "c",
            Language::CHeader => "cHeader",
            Language::CPreprocessed => "cPreprocessed",
            Language::Cxx => "c++",
            Language::CxxHeader => "c++Header",
            Language::CxxPreprocessed => "c++Preprocessed",
            Language::GenericHeader => "c/c++",
            Language::ObjectiveC | Language::ObjectiveCHeader => "objc",
            Language::ObjectiveCPreprocessed => "objcPreprocessed",
            Language::ObjectiveCxx | Language::ObjectiveCxxHeader => "objc++",
            Language::ObjectiveCxxPreprocessed => "objc++Preprocessed",
            Language::Cuda => "cuda",
            Language::CudaFE => "cuda",
            Language::Ptx => "ptx",
            Language::Cubin => "cubin",
            Language::Rust => "rust",
            Language::Hip => "hip",
            Language::CxxModule => "c++-module",
        }
    }

    pub fn from_compiler_str(kind: CompilerKind, lang: &str) -> Option<Self> {
        if lang == "rs" {
            return Some(Language::Rust);
        }
        match kind {
            CompilerKind::C(CCompilerKind::Nvhpc) => {
                // -x=c|cpp|c++|i|cpp-output|asm|assembler|ASM|assembler-with-cpp|none
                //     Specify the language for any following input files, instead of
                //     letting the compiler choose based on suffix. Turn off with -x none
                match lang {
                    "c" => Some(Language::C),
                    "c++" | "cpp" => Some(Language::Cxx),
                    "cpp-output" => Some(Language::CxxPreprocessed),
                    _ => None, // Let the compiler decide
                }
            }
            _ => match lang {
                // From https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html
                // and https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Driver/Types.def
                "assembler" => Some(Language::Assembler),
                "assembler-with-cpp" => Some(Language::AssemblerToPreprocess),
                "c" => Some(Language::C),
                "c++" => Some(Language::Cxx),
                "c-header" => Some(Language::CHeader),
                "cpp-output" => Some(Language::CPreprocessed),
                "c++-header" => Some(Language::CxxHeader),
                "c++-module" => Some(Language::CxxModule),
                "c++-cpp-output" => Some(Language::CxxPreprocessed),
                "objective-c" => Some(Language::ObjectiveC),
                "objective-c-header" => Some(Language::ObjectiveCHeader),
                "objective-c-cpp-output" => Some(Language::ObjectiveCPreprocessed),
                "objective-c++" => Some(Language::ObjectiveCxx),
                "objective-c++-header" => Some(Language::ObjectiveCxxHeader),
                "objective-c++-cpp-output" => Some(Language::ObjectiveCxxPreprocessed),
                "cuda" => match kind {
                    CompilerKind::C(CCompilerKind::Clang) => Some(Language::Cuda),
                    _ => None, // Let the compiler decide
                },
                "cu" => match kind {
                    CompilerKind::C(CCompilerKind::Clang)
                    | CompilerKind::C(CCompilerKind::Nvcc) => Some(Language::Cuda),
                    _ => None, // Let the compiler decide
                },
                "hip" => Some(Language::Hip),
                _ => None, // Let the compiler decide
            },
        }
    }

    pub fn as_compiler_str(&self, kind: CompilerKind) -> Option<&'static str> {
        match kind {
            CompilerKind::C(CCompilerKind::Nvhpc) => {
                // -x=c|cpp|c++|i|cpp-output|asm|assembler|ASM|assembler-with-cpp|none
                //     Specify the language for any following input files, instead of
                //     letting the compiler choose based on suffix. Turn off with -x none
                match self {
                    Language::AssemblerToPreprocess => Some("assembler-with-cpp"),
                    Language::Assembler => Some("assembler"),
                    Language::C => Some("c"),
                    Language::CPreprocessed => Some("cpp-output"),
                    Language::Cxx => Some("c++"),
                    Language::CxxPreprocessed => Some("cpp-output"),
                    _ => None, // Let the compiler decide
                }
            }
            _ => match self {
                Language::AssemblerToPreprocess => Some("assembler-with-cpp"),
                Language::Assembler => Some("assembler"),
                Language::C => Some("c"),
                Language::Cxx => Some("c++"),
                Language::CHeader => Some("c-header"),
                Language::CPreprocessed => Some("cpp-output"),
                Language::CxxHeader => Some("c++-header"),
                Language::CxxModule => Some("c++-module"),
                Language::CxxPreprocessed => Some("c++-cpp-output"),
                Language::ObjectiveC => Some("objective-c"),
                Language::ObjectiveCHeader => Some("objective-c-header"),
                Language::ObjectiveCPreprocessed => Some("objective-c-cpp-output"),
                Language::ObjectiveCxx => Some("objective-c++"),
                Language::ObjectiveCxxHeader => Some("objective-c++-header"),
                Language::ObjectiveCxxPreprocessed => Some("objective-c++-cpp-output"),
                Language::Cuda => match kind {
                    CompilerKind::C(CCompilerKind::Clang) => Some("cuda"),
                    CompilerKind::C(CCompilerKind::Nvcc) => Some("cu"),
                    _ => None, // Let the compiler decide
                },
                Language::CudaFE => None,
                Language::Ptx => None,
                Language::Cubin => None,
                Language::Rust => None, // Let the compiler decide
                Language::Hip => Some("hip"),
                Language::GenericHeader => None, // Let the compiler decide
            },
        }
    }

    pub fn needs_c_preprocessing(self) -> bool {
        !matches!(
            self,
            Language::Assembler
                | Language::CPreprocessed
                | Language::CxxPreprocessed
                | Language::ObjectiveCPreprocessed
                | Language::ObjectiveCxxPreprocessed
                | Language::CudaFE
                | Language::Ptx
                | Language::Cubin
                | Language::Rust
        )
    }

    pub fn is_c_like_header(self) -> bool {
        matches!(
            self,
            Language::CHeader
                | Language::CxxHeader
                | Language::ObjectiveCHeader
                | Language::ObjectiveCxxHeader
                | Language::GenericHeader
        )
    }

    pub fn to_c_preprocessed_language(self) -> Option<Language> {
        match self {
            Language::AssemblerToPreprocess => Some(Language::Assembler),
            Language::C => Some(Language::CPreprocessed),
            Language::Cxx => Some(Language::CxxPreprocessed),
            Language::ObjectiveC => Some(Language::ObjectiveCPreprocessed),
            Language::ObjectiveCxx => Some(Language::ObjectiveCxxPreprocessed),
            _ => None,
        }
    }
}

impl CompilerKind {
    pub fn lang_kind(&self, lang: &Language) -> String {
        match lang {
            Language::AssemblerToPreprocess | Language::Assembler => "Assembler",
            Language::C
            | Language::CHeader
            | Language::CPreprocessed
            | Language::Cxx
            | Language::CxxHeader
            | Language::CxxModule
            | Language::CxxPreprocessed
            | Language::GenericHeader
            | Language::ObjectiveC
            | Language::ObjectiveCHeader
            | Language::ObjectiveCPreprocessed
            | Language::ObjectiveCxx
            | Language::ObjectiveCxxHeader
            | Language::ObjectiveCxxPreprocessed => "C/C++",
            Language::Cuda => "CUDA",
            Language::CudaFE => "CUDA (Device code)",
            Language::Ptx => "PTX",
            Language::Cubin => "CUBIN",
            Language::Rust => "Rust",
            Language::Hip => "HIP",
        }
        .to_string()
    }
    pub fn lang_comp_kind(&self, lang: &Language) -> String {
        let textual_lang = lang.as_str().to_owned();
        match self {
            CompilerKind::C(CCompilerKind::Clang) => textual_lang + " [clang]",
            CompilerKind::C(CCompilerKind::Diab) => textual_lang + " [diab]",
            CompilerKind::C(CCompilerKind::Gcc) => textual_lang + " [gcc]",
            CompilerKind::C(CCompilerKind::Msvc) => textual_lang + " [msvc]",
            CompilerKind::C(CCompilerKind::Nvcc) => textual_lang + " [nvcc]",
            CompilerKind::C(CCompilerKind::CudaFE) => textual_lang + " [cudafe++]",
            CompilerKind::C(CCompilerKind::Cicc) => textual_lang + " [cicc]",
            CompilerKind::C(CCompilerKind::Ptxas) => textual_lang + " [ptxas]",
            CompilerKind::C(CCompilerKind::Nvhpc) => textual_lang + " [nvhpc]",
            CompilerKind::C(CCompilerKind::TaskingVX) => textual_lang + " [taskingvx]",
            CompilerKind::C(CCompilerKind::Tileiras) => textual_lang + " [tileiras]",
            CompilerKind::C(CCompilerKind::Llc) => textual_lang + " [llc]",
            CompilerKind::Rust => textual_lang,
        }
    }
}

#[cfg(feature = "dist-client")]
pub type DistPackagers = (
    Box<dyn pkg::InputsPackager>,
    Box<dyn pkg::ToolchainPackager>,
    Box<dyn OutputsRewriter>,
);

enum CacheLookupResult {
    Success(CompileResult, ProcessOutput),
    Miss(MissType),
}

/// An interface to a compiler for argument parsing.
pub trait Compiler<T>: Send + Sync + 'static
where
    T: CommandCreatorSync,
{
    /// Return the kind of compiler.
    fn kind(&self) -> CompilerKind;
    /// Return the compiler version reported by the compiler executable.
    fn version(&self) -> Option<String>;
    /// Retrieve a packager
    #[cfg(feature = "dist-client")]
    fn get_toolchain_packager(&self) -> Box<dyn pkg::ToolchainPackager>;
    /// Determine whether `arguments` are supported by this compiler.
    fn parse_arguments(
        &self,
        arguments: &[OsString],
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
        basedirs: &[Vec<u8>],
    ) -> CompilerArguments<Box<dyn CompilerHasher<T> + 'static>>;
    fn box_clone(&self) -> Box<dyn Compiler<T>>;
    #[cfg(test)]
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send + Sync>;
}

impl<T: CommandCreatorSync> Clone for Box<dyn Compiler<T>> {
    fn clone(&self) -> Box<dyn Compiler<T>> {
        self.box_clone()
    }
}

pub trait CompilerProxy<T>: Send + Sync + 'static
where
    T: CommandCreatorSync + Sized,
{
    /// Maps the executable to be used in `cwd` to the true, proxied compiler.
    ///
    /// Returns the absolute path to the true compiler and the timestamp of
    /// timestamp of the true compiler. Iff the resolution fails,
    /// the returned future resolves to an error with more information.
    fn resolve_proxied_executable(
        &self,
        creator: T,
        cwd: PathBuf,
        env_vars: &[(OsString, OsString)],
    ) -> Pin<Box<dyn Future<Output = Result<(PathBuf, FileTime)>> + Send + 'static>>;

    /// Create a clone of `Self` and puts it in a `Box`
    fn box_clone(&self) -> Box<dyn CompilerProxy<T>>;
}

impl<T: CommandCreatorSync> Clone for Box<dyn CompilerProxy<T>> {
    fn clone(&self) -> Box<dyn CompilerProxy<T>> {
        self.box_clone()
    }
}

pub struct GenerateHashKeyArgs<'a, T: CommandCreatorSync> {
    pub cache_control: CacheControl,
    pub creator: &'a T,
    pub cwd: &'a Path,
    pub env_vars: &'a [(OsString, OsString)],
    pub runtime: &'a tokio::runtime::Handle,
    pub service: &'a server::SccacheService<T>,
    pub dist_compile: bool,
}

/// An interface to a compiler for hash key generation, the result of
/// argument parsing.
#[async_trait]
pub trait CompilerHasher<T>: fmt::Debug + Send + 'static
where
    T: CommandCreatorSync,
{
    /// Given information about a compiler command, generate a hash key
    /// that can be used for cache lookups, as well as any additional
    /// information that can be reused for compilation if necessary.
    async fn generate_hash_key(
        self: Box<Self>,
        args: GenerateHashKeyArgs<'_, T>,
    ) -> Result<(
        HashResult<T>,
        Option<Pin<Box<dyn Future<Output = Result<()>> + Send>>>,
    )>;

    /// Return the state of any `--color` option passed to the compiler.
    fn color_mode(&self) -> ColorMode;

    /// A descriptive string about the file that we're going to be producing.
    ///
    /// This is primarily intended for debug logging and such, not for actual
    /// artifact generation.
    fn output_pretty(&self) -> Cow<'_, str>;

    fn box_clone(&self) -> Box<dyn CompilerHasher<T>>;

    fn language(&self) -> Language;

    fn get_executable(&self) -> &Path;
}

// An inherent method on the trait object rather than a provided trait method,
// which would be compiled again for every implementor.
impl<T: CommandCreatorSync> dyn CompilerHasher<T> {
    /// Look up a cached compile result in `storage`. If not found, run the
    /// compile and store the result.
    #[allow(clippy::too_many_arguments)]
    pub async fn get_cached_or_compile(
        self: Box<Self>,
        service: &server::SccacheService<T>,
        dist_client: Option<Arc<dyn dist::Client>>,
        creator: T,
        arguments: Vec<OsString>,
        cwd: PathBuf,
        env_vars: Vec<(OsString, OsString)>,
        cache_control: CacheControl,
        runtime: tokio::runtime::Handle,
        pending: crate::server::SccacheGaugeIncrement,
    ) -> Result<(CompileResult, ProcessOutput)> {
        let out_pretty = self.output_pretty().into_owned();

        // [<file>] get_cached_or_compile: "/path/to/exe <args...>"
        debug!(
            "[{out_pretty}]: get_cached_or_compile: {}",
            SingleCompileCommand {
                arguments: arguments.clone(),
                cwd: cwd.clone(),
                env_vars: env_vars.clone(),
                executable: self.get_executable().to_owned(),
                out_pretty: out_pretty.clone(),
                share_jobserver: false
            }
        );

        let dist_client = dist_client.filter(|_| {
            // Clients might want to set this when configuring
            // so e.g. the CMake compiler checks execute locally
            !env_vars.iter().any(|(k, v)| {
                k == "SCCACHE_NO_DIST_COMPILE"
                    && matches!(v.to_str().unwrap_or_default(), "true" | "on" | "1")
            })
        });

        let hash_time = Instant::now();

        let hash_result = self
            .generate_hash_key(GenerateHashKeyArgs {
                cache_control,
                creator: &creator,
                cwd: cwd.as_path(),
                env_vars: &env_vars,
                runtime: &runtime,
                service,
                dist_compile: dist_client.is_some(),
            })
            .await;

        let hash_time = hash_time.elapsed();

        debug!(
            "[{out_pretty}]: generate_hash_key took {}",
            fmt_duration_as_secs(&hash_time)
        );

        // Decrement pending_compilations as soon as we're done preprocessing
        drop(pending);

        let (hash_result, deps_cache_write) = match hash_result {
            Ok((hash_result, deps_cache_write)) => (
                hash_result,
                deps_cache_write.map(|deps_cache_write| {
                    // Immediately drive the dependency cache write so it
                    // runs concurrently to cache lookups and compilations
                    crate::util::spawn_detached_on(&runtime, deps_cache_write)
                }),
            ),
            Err(err) => {
                return match err.downcast::<ProcessError>() {
                    Ok(ProcessError(output)) => {
                        debug!("[{out_pretty}]: process error: {output:?}");
                        Ok((CompileResult::Error(hash_time), output))
                    }
                    Err(e) => Err(e),
                };
            }
        };

        // Instance to perform the cache lookup and prepare to compile
        let lookup_or_compile = CacheLookupOrCompile::new(
            cache_control,
            &creator,
            hash_result,
            &cwd,
            dist_client,
            out_pretty.clone(),
            runtime.clone(),
            service,
        );

        let lookup = lookup_or_compile
            .cache_lookup(
                service.storage.as_ref(),
                // Maximum time limit for the cache to respond before we
                // forge ahead ourselves with a compilation.
                service.cache_read_timeout,
            )
            .await;

        // Check the result of the cache lookup
        match lookup? {
            // Cache hit
            CacheLookupResult::Success(compile_result, output) => {
                if let Some(deps_cache_write) = deps_cache_write {
                    let _ = deps_cache_write.await;
                }
                Ok::<_, Error>((compile_result, output))
            }
            // Cache miss, compile
            CacheLookupResult::Miss(miss_type) => {
                let compile = lookup_or_compile.into_compile()?;
                // Do the compilation (either local or distributed)
                let (compile, duration) = {
                    let start = Instant::now();
                    let res = compile.into_result().await;
                    (res, start.elapsed())
                };

                let (hash_key, outputs, cacheable, dist_type, output) = compile?;

                if !output.success() {
                    trace!(
                        "[{out_pretty}]: Compiled in {}, but failed, not storing in cache",
                        fmt_duration_as_secs(&duration)
                    );
                    return Ok((CompileResult::CompileFailed(dist_type, duration), output));
                }
                if miss_type == MissType::ForcedNoCache {
                    // Do not cache
                    trace!(
                        "[{out_pretty}]: Compiled in {}, but not caching",
                        fmt_duration_as_secs(&duration)
                    );
                    return Ok((CompileResult::NotCached(dist_type, duration), output));
                }
                if hash_key.is_empty() || cacheable != Cacheable::Yes {
                    // Not cacheable
                    trace!(
                        "[{out_pretty}]: Compiled in {}, but not cacheable",
                        fmt_duration_as_secs(&duration)
                    );
                    return Ok((CompileResult::NotCacheable(dist_type, duration), output));
                }

                trace!(
                    "[{out_pretty}]: Compiled in {}, storing in cache",
                    fmt_duration_as_secs(&duration)
                );

                let (entry, output, duration) = CacheWrite::from_objects(outputs, output, &runtime)
                    .await
                    .context("failed to zip up compiler outputs")?;

                trace!(
                    "[{out_pretty}]: Created cache artifact in {}",
                    fmt_duration_as_secs(&duration)
                );

                let out_pretty2 = out_pretty.clone();
                let storage = service.storage.clone();
                // Try to finish storing the newly-written cache
                // entry. We'll get the result back elsewhere.
                let cache_write: Pin<Box<dyn Future<Output = Result<CacheWriteInfo>> + Send>> =
                    Box::pin(async move {
                        let start = Instant::now();
                        match storage.put(&hash_key, entry.into()).await {
                            Ok(_) => {
                                trace!("[{out_pretty2}]: Stored in cache successfully!");
                                Ok(CacheWriteInfo {
                                    object_file_pretty: out_pretty2,
                                    duration: start.elapsed(),
                                })
                            }
                            Err(e) => Err(e),
                        }
                    });

                let cache_write = Box::pin({
                    let out_pretty2 = out_pretty.clone();
                    let timeout = service.cache_write_timeout;
                    async move {
                        let start = Instant::now();
                        match tokio::time::timeout(timeout, cache_write).await {
                            Ok(res) => res,
                            Err(err) => {
                                trace!(
                                    "[{out_pretty2}]: Cache write timed out {}",
                                    fmt_duration_as_secs(&start.elapsed())
                                );
                                Err(anyhow::Error::new(err))
                            }
                        }
                    }
                });

                let cache_write = if let Some(deps_cache_write) = deps_cache_write {
                    Box::pin(async move {
                        // the deps cache write is allowed to error,
                        // we only care about the object cache write
                        futures::future::join(deps_cache_write, cache_write).await.1
                    })
                        as Pin<Box<dyn Future<Output = Result<CacheWriteInfo>> + Send>>
                } else {
                    cache_write
                };

                Ok((
                    CompileResult::CacheMiss(miss_type, dist_type, duration, cache_write),
                    output,
                ))
            }
        }
        .with_context(|| format!("Failed to cache `{out_pretty}`"))
    }
}

struct CacheLookupOrCompile<'a, T: CommandCreatorSync> {
    cache_control: CacheControl,
    command_creator: &'a T,
    compilation: Box<dyn Compilation<T>>,
    dist_client: Option<Arc<dyn dist::Client>>,
    hash_key: String,
    out_pretty: String,
    outputs: Vec<FileObjectSource>,
    runtime: tokio::runtime::Handle,
    sccache_service: &'a server::SccacheService<T>,
    weak_toolchain_key: String,
}

impl<'a, T> CacheLookupOrCompile<'a, T>
where
    T: CommandCreatorSync,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        cache_control: CacheControl,
        command_creator: &'a T,
        compilation: HashResult<T>,
        cwd: &Path,
        dist_client: Option<Arc<dyn dist::Client>>,
        out_pretty: String,
        runtime: tokio::runtime::Handle,
        sccache_service: &'a server::SccacheService<T>,
    ) -> Self {
        let HashResult {
            compilation,
            key: hash_key,
            weak_toolchain_key,
        } = compilation;

        trace!("[{out_pretty}]: Hash key: {hash_key}");

        let outputs = compilation
            .outputs()
            .map(|output| FileObjectSource {
                path: cwd.join(output.path),
                ..output
            })
            .collect::<Vec<_>>();

        let outputs = if compilation.is_locally_preprocessed() {
            // In this mode, cache entries are exclusively distinguished by their preprocessed
            // source contents. But two files may differ in their names and / or the names of
            // included files while still producing the same preprocessed output, so they get the
            // same cache entry. That entry will have wrong (file names) dependency informaton in
            // the dependency file except for the compilation unit that originally produced it.
            // Since we did local preprocessing, that should already have produced the dependency
            // file - just leave that one alone and don't overwrite it from the cache.
            outputs
                .iter()
                .filter(|fobj_source| !fobj_source.key.starts_with("dep")) // key "dep" means dependency file
                .cloned()
                .collect()
        } else {
            // In this mode, no local preprocessing was done, so the dependency file (if any)
            // has not been created. But in this mode, the cache key also includes a lot of
            // information about filenames (and less relevant here, file hashes), so it *is* safe
            // to restore the dependency file from the cache.
            outputs.clone()
        };

        Self {
            cache_control,
            command_creator,
            compilation,
            dist_client,
            hash_key,
            out_pretty,
            outputs,
            runtime,
            sccache_service,
            weak_toolchain_key,
        }
    }

    async fn cache_lookup(
        &self,
        storage: &dyn Storage,
        timeout: Duration,
    ) -> Result<CacheLookupResult> {
        let Self {
            cache_control,
            out_pretty,
            hash_key,
            ..
        } = self;

        let start = Instant::now();

        // If cache_control is `ForceNoCache`, `ForceRecache`, or `Nvcc`, don't check the cache.
        match cache_control {
            CacheControl::ForceNoCache => {
                trace!(
                    "[{out_pretty}]: Cache none in {}",
                    fmt_duration_as_secs(&start.elapsed())
                );
                Ok(CacheLookupResult::Miss(MissType::ForcedNoCache))
            }
            CacheControl::ForceRecache => {
                trace!(
                    "[{out_pretty}]: Cache recache in {}",
                    fmt_duration_as_secs(&start.elapsed())
                );
                Ok(CacheLookupResult::Miss(MissType::ForcedRecache))
            }
            _ if hash_key.is_empty() => Ok(CacheLookupResult::Miss(MissType::Normal)),
            _ => {
                let _pending = self.sccache_service.increment_pending_cache_lookups();
                CacheLookup::new(self, start, timeout, storage)
                    .into_result()
                    .await
            }
        }
    }

    #[cfg(not(feature = "dist-client"))]
    fn into_compile(self) -> Result<Compile<'a, T>> {
        let Self {
            command_creator,
            compilation,
            hash_key,
            out_pretty,
            outputs,
            sccache_service,
            ..
        } = self;

        let (compile_cmd, _dist_compile_cmd, cacheable) =
            compilation
                .generate_compile_commands()
                .context("Failed to generate compile commands")?;

        Ok(Compile {
            command_creator,
            hash_key,
            local: LocalCompile {
                compile_cmd,
                cacheable,
                outputs,
            },
            dist: None,
            out_pretty,
            sccache_service,
        })
    }

    #[cfg(feature = "dist-client")]
    fn into_compile(self) -> Result<Compile<'a, T>> {
        let Self {
            command_creator,
            compilation,
            dist_client,
            hash_key,
            out_pretty,
            outputs,
            sccache_service,
            weak_toolchain_key,
            ..
        } = self;

        let (compile_cmd, dist_compile_cmd, cacheable) = compilation
            .generate_compile_commands()
            .context("Failed to generate compile commands")?;

        let dist = dist_client.and_then(|dist_client| {
            dist_compile_cmd.map(|dist_compile_cmd| DistCompile {
                compilation,
                dist_client,
                dist_compile_cmd,
                out_pretty: out_pretty.clone(),
                path_transformer: Default::default(),
                weak_toolchain_key,
            })
        });

        Ok(Compile {
            command_creator,
            hash_key,
            local: LocalCompile {
                compile_cmd,
                cacheable,
                outputs,
            },
            dist,
            out_pretty,
            sccache_service,
        })
    }
}

struct CacheLookup<'a> {
    hash_key: String,
    out_pretty: String,
    outputs: Vec<FileObjectSource>,
    runtime: tokio::runtime::Handle,
    start: Instant,
    storage: &'a dyn Storage,
    timeout: Duration,
}

impl<'a> CacheLookup<'a> {
    fn new<T: CommandCreatorSync>(
        details: &CacheLookupOrCompile<'a, T>,
        start: Instant,
        timeout: Duration,
        storage: &'a dyn Storage,
    ) -> Self {
        Self {
            hash_key: details.hash_key.clone(),
            out_pretty: details.out_pretty.clone(),
            outputs: details.outputs.clone(),
            runtime: details.runtime.clone(),
            start,
            storage,
            timeout,
        }
    }

    async fn into_result(self) -> Result<CacheLookupResult> {
        let Self {
            out_pretty, start, ..
        } = &self;

        let fut = self.lookup_in_cache();
        let fut = tokio::time::timeout(self.timeout, fut);

        match fut.await {
            Ok(res) => res,
            Err(_) => {
                trace!(
                    "[{out_pretty}]: Cache timed out {}",
                    fmt_duration_as_secs(&start.elapsed())
                );
                Ok(CacheLookupResult::Miss(MissType::TimedOut))
            }
        }
    }

    async fn lookup_in_cache(&self) -> Result<CacheLookupResult> {
        let Self {
            hash_key,
            out_pretty,
            outputs,
            runtime,
            start,
            storage,
            ..
        } = self;

        match storage.get(hash_key).await {
            Ok(Cache::Hit(entry)) => {
                let duration = start.elapsed();
                trace!(
                    "[{out_pretty}]: Cache hit in {}",
                    fmt_duration_as_secs(&duration)
                );
                let mut entry = CacheRead::try_from(entry)?;
                let stdout = entry.get_stdout();
                let stderr = entry.get_stderr();
                match entry.extract_objects(outputs.clone(), runtime).await {
                    Ok(()) => Ok(CacheLookupResult::Success(
                        CompileResult::CacheHit(duration),
                        ProcessOutput::new(0, stdout, stderr),
                    )),
                    Err(e) => {
                        if let Some(err) = e.downcast_ref::<DecompressionFailure>() {
                            debug!("[{out_pretty}]: Failed to decompress object: {err:?}");
                            Ok(CacheLookupResult::Miss(MissType::CacheReadError))
                        } else {
                            Err(e)
                        }
                    }
                }
            }
            Ok(Cache::Miss) => {
                trace!(
                    "[{out_pretty}]: Cache miss in {}",
                    fmt_duration_as_secs(&start.elapsed())
                );
                Ok(CacheLookupResult::Miss(MissType::Normal))
            }
            Err(err) => {
                debug!(
                    "[{out_pretty}]: Cache read error in {}: {err}",
                    fmt_duration_as_secs(&start.elapsed())
                );
                Ok(CacheLookupResult::Miss(MissType::CacheReadError))
            }
        }
    }
}

struct Compile<'a, T: CommandCreatorSync> {
    command_creator: &'a T,
    dist: Option<DistCompile<T>>,
    hash_key: String,
    local: LocalCompile<T>,
    out_pretty: String,
    sccache_service: &'a server::SccacheService<T>,
}

impl<T> Compile<'_, T>
where
    T: CommandCreatorSync,
{
    #[cfg(not(feature = "dist-client"))]
    async fn into_result(
        self,
    ) -> Result<(
        String,
        Vec<FileObjectSource>,
        Cacheable,
        DistType,
        ProcessOutput,
    )> {
        let Self {
            command_creator,
            hash_key,
            sccache_service: service,
            ..
        } = self;

        let LocalCompile {
            ref compile_cmd,
            cacheable,
            outputs,
            ..
        } = self.local;

        Ok((
            hash_key,
            outputs,
            cacheable,
            DistType::NoDist,
            compile_cmd.execute(service, command_creator).await?,
        ))
    }

    #[cfg(feature = "dist-client")]
    async fn into_result(
        self,
    ) -> Result<(
        String,
        Vec<FileObjectSource>,
        Cacheable,
        DistType,
        ProcessOutput,
    )> {
        let Self {
            command_creator,
            hash_key,
            out_pretty,
            sccache_service: service,
            ..
        } = self;

        let LocalCompile {
            compile_cmd,
            cacheable,
            outputs,
            ..
        } = self.local;

        let mut dist_type = DistType::NoDist;

        if let Some(dist) = self.dist {
            let executable = compile_cmd.get_executable();
            let fallback_to_local = dist.dist_client.fallback_to_local_compile();

            // Do the distributed compilation
            let dist_compile_res = dist.into_result(&executable, &outputs, service).await;

            match dist_compile_res {
                // Distributed compilation succeeded
                Ok((dist_type, output)) => {
                    return Ok((hash_key, outputs, cacheable, dist_type, output));
                }
                // Fallback to local compile
                Err(e) if fallback_to_local => {
                    dist_type = DistType::Error;
                    // `{:#}` prints the error and the causes in a single line.
                    warn!("[{out_pretty}]: Could not perform distributed compile: {e:#}");
                }
                // Distributed compilation failed
                Err(e) => {
                    if e.downcast_ref::<HttpClientError>().is_some() {
                        return Err(e);
                    }

                    if let Some(lru_disk_cache::Error::FileTooLarge) =
                        e.downcast_ref::<lru_disk_cache::Error>()
                    {
                        return Err(anyhow!(
                            "Could not cache dist toolchain for {executable:?} locally.
                            Increase `toolchain_cache_size` or decrease the toolchain archive size.",
                        ));
                    }

                    return Err(DistClientError(e).into());
                }
            }
        }

        // Local compile
        Ok((
            hash_key,
            outputs,
            cacheable,
            dist_type,
            compile_cmd.execute(service, command_creator).await?,
        ))
    }
}

struct LocalCompile<T: CommandCreatorSync> {
    compile_cmd: Box<dyn CompileCommand<T>>,
    cacheable: Cacheable,
    outputs: Vec<FileObjectSource>,
}

struct DistCompile<T: CommandCreatorSync> {
    compilation: Box<dyn Compilation<T>>,
    dist_client: Arc<dyn dist::Client>,
    dist_compile_cmd: dist::CompileCommand,
    out_pretty: String,
    #[cfg(feature = "dist-client")]
    path_transformer: dist::PathTransformer,
    weak_toolchain_key: String,
}

#[cfg(feature = "dist-client")]
impl<T> DistCompile<T>
where
    T: CommandCreatorSync,
{
    async fn into_result(
        self,
        executable: &Path,
        outputs: &[FileObjectSource],
        service: &server::SccacheService<T>,
    ) -> Result<(DistType, ProcessOutput)> {
        use dist::v3::RunJobResponse;
        use itertools::Itertools;
        use std::io;
        use tokio_retry2::strategy::FibonacciBackoff;

        let Self {
            compilation,
            dist_client,
            dist_compile_cmd,
            ref out_pretty,
            mut path_transformer,
            weak_toolchain_key,
            ..
        } = self;

        use crate::util::path_to_str;

        let mut pending = Some(service.increment_pending_compilations());
        let mut uploading = None;

        let save_outputs_as_job_input = compilation.save_outputs_as_job_input();
        let mut dist_compile_cmd = dist_compile_cmd.as_dist(&mut path_transformer)?;

        let (inputs_packager, toolchain_packager, outputs_rewriter) =
            compilation.into_dist_packagers()?;

        let (inputs_manifest, hashed_inputs) = {
            trace!("[{out_pretty}]: Hashing dist inputs");

            let (contents, inputs) = inputs_packager
                .describe_inputs(&mut path_transformer)
                .await?;

            let (manifest, inputs) = dist_client.get_job_inputs(contents, inputs).await?;

            (manifest, inputs)
        };

        trace!("[{out_pretty}]: Identifying dist toolchain for {executable:?}");

        let (dist_toolchain, maybe_dist_compile_executable, packaged_toolchain) = dist_client
            .hash_toolchain(
                executable,
                &weak_toolchain_key,
                toolchain_packager,
                &mut path_transformer,
            )
            .await?;

        let tc_archive =
            maybe_dist_compile_executable.map(|(dist_compile_executable, archive_path)| {
                dist_compile_cmd.executable = dist_compile_executable;
                archive_path
            });

        let dist_output_paths = outputs
            .iter()
            .map(|output| {
                path_transformer
                    .as_dist_abs(&output.path)
                    .with_context(|| {
                        format!("Expected output path {:?} to be absolute", output.path)
                    })
                    .and_then(|p| {
                        path_to_str(p).with_context(|| {
                            format!("Failed to serialize output path {:?}", output.path)
                        })
                    })
                    .map(Into::into)
            })
            .try_collect::<_, Vec<_>, _>()
            .context("Failed to adapt an output path for distributed compile")?;

        let dist_retry_limit = dist_client.max_retries() + 1.0;
        let mut num_dist_attempts = 1.0f64;
        let mut missing_inputs = vec![];
        let mut has_toolchain = false;
        let mut job = None;

        let mut should_retry = |err: &Error,
                                job_id: Option<&str>,
                                server_id: Option<&str>,
                                missing_inputs: &[String]| {
            use itertools::Itertools;

            let out_pretty = Some(out_pretty.as_str())
                .iter()
                .chain(job_id.iter())
                .chain(server_id.iter())
                .join(", ");

            if num_dist_attempts == 1.0f64 && !missing_inputs.is_empty() {
                // Allow one retry if the local dist client mistakenly  believed that
                // the dist remote still has inputs that it previously reported it had.
                // If the job gets to a build server and it's missing some inputs, the
                // build server will reject with exactly the inputs that are missing,
                // which is the client's chance to send them again.
                num_dist_attempts += 1.0;
                true
            } else if num_dist_attempts < dist_retry_limit {
                info!(
                    "[{out_pretty}]: Distributed compilation error ({num_dist_attempts} of {dist_retry_limit}), retrying: {err:#}"
                );
                num_dist_attempts += 1.0;
                true
            } else {
                warn!("[{out_pretty}]: Could not run distributed compilation job: {err}");
                false
            }
        };

        let mut retry_delay = FibonacciBackoff::from_millis(1000)
            .max_delay(Duration::from_secs(10))
            .map(tokio_retry2::strategy::jitter);

        let dist_compile_res = loop {
            use futures::TryStreamExt;

            let job_id: &String;
            let timeout: u32;

            macro_rules! retry_or_bail {
                ($err:ident, $job_id:expr, $server_id:expr) => {{
                    if should_retry(&$err, $job_id, $server_id, &missing_inputs) {
                        tokio::time::sleep(retry_delay.next().unwrap()).await;
                        continue;
                    }
                    break Err($err);
                }};
                ($err:expr, $job_id:expr, $server_id:expr) => {{
                    let err = $err;
                    retry_or_bail!(err, $job_id, $server_id);
                }};
                ($err:expr, $job_id:expr) => {{
                    retry_or_bail!($err, $job_id, None);
                }};
            }

            (job_id, timeout) = if let Some((ref job_id, timeout)) = job {
                (job_id, timeout)
            } else {
                pending.get_or_insert_with(|| service.increment_pending_compilations());
                uploading.get_or_insert_with(|| service.increment_uploading_compilations());

                trace!("[{out_pretty}]: Requesting job allocation");

                match dist_client
                    .new_job(
                        dist_toolchain.clone(),
                        hashed_inputs.keys().cloned().collect(),
                    )
                    .await
                {
                    Ok(res) => {
                        trace!("[{out_pretty}, {}]: Received job allocation", res.job_id);
                        missing_inputs = res.missing_inputs;
                        has_toolchain = res.has_toolchain;
                        job = Some((res.job_id, res.timeout.min(dist_client.request_timeout())));
                        continue;
                    }
                    // Maybe retry network errors
                    Err(err) => retry_or_bail!(err, None),
                }
            };

            if !missing_inputs.is_empty() {
                pending.get_or_insert_with(|| service.increment_pending_compilations());
                uploading.get_or_insert_with(|| service.increment_uploading_compilations());

                debug!("[{out_pretty}, {job_id}]: Uploading job inputs: {missing_inputs:?}");

                match dist_client
                    .put_job(
                        missing_inputs
                            .iter()
                            .filter_map(|hash| {
                                hashed_inputs
                                    .get(hash)
                                    .map(|input| (hash.clone(), input.clone()))
                            })
                            .collect(),
                    )
                    .await
                    .map_err(|e| e.context("Could not submit job inputs"))
                {
                    Ok(_) => {
                        missing_inputs = vec![];
                    }
                    // Maybe retry network errors
                    Err(err) => retry_or_bail!(err, Some(job_id.as_str())),
                }
            }

            if !has_toolchain {
                pending.get_or_insert_with(|| service.increment_pending_compilations());
                uploading.get_or_insert_with(|| service.increment_uploading_compilations());

                debug!("[{out_pretty}, {job_id}]: Uploading toolchain: {dist_toolchain}");

                match dist_client
                    .put_toolchain(dist_toolchain.clone(), packaged_toolchain.clone())
                    .await
                    .map_err(|e| e.context("Could not submit toolchain"))
                {
                    Ok(dist::SubmitToolchainResult::Success) => {
                        has_toolchain = true;
                    }
                    Ok(dist::SubmitToolchainResult::Error { message }) => {
                        retry_or_bail!(anyhow!(message), Some(job_id.as_str()));
                    }
                    // Maybe retry network errors
                    Err(err) => retry_or_bail!(err, Some(job_id.as_str())),
                }
            }

            debug!("[{out_pretty}, {job_id}]: Running job");

            let run_job = dist_client.run_job(
                job_id,
                Duration::from_secs(timeout as u64),
                dist_toolchain.clone(),
                dist_compile_cmd.clone(),
                inputs_manifest.clone(),
                dist_output_paths.clone(),
                save_outputs_as_job_input,
            );

            let _ = (pending.take(), uploading.take());

            let active = service.increment_active_compilations();

            let run_job = run_job.await;

            drop(active);

            let (build_result, server_id) = match run_job {
                // Job completed, regardless of whether compilation succeeded or failed
                Ok(RunJobResponse::Complete { result, server_id }) => (result, server_id),
                // Job failed with an unrecoverable fatal error. These should be rare, since
                // there aren't many cases where we want to explicitly fail the build without
                // allowing the client to retry.
                //
                // The most likely source of these errors are when the compilation took longer
                // than the Scheduler's `job_time_limit` configuration. In this case, retrying
                // the compilation on a different server won't finish any sooner, so we report
                // a fatal error so clients see their build fail and can either increase the
                // job time limit, or apply source optimizations so their files compile in a
                // reasonable amount of time.
                //
                // Fatal errors are not retryable.
                Ok(RunJobResponse::FatalError { message, server_id }) => {
                    error!(
                        "[{out_pretty}, {job_id}, {server_id}]: Distributed compilation failed (fatal): {message}"
                    );
                    break Err(anyhow!(message));
                }
                // Disk errors, build process killed, server shutdown, etc.
                // Can be retried.
                Ok(RunJobResponse::RetryableError { message, server_id }) => {
                    debug!(
                        "[{out_pretty}, {job_id}, {server_id}]: Distributed compilation failed (retryable): {message:?}"
                    );
                    retry_or_bail!(
                        anyhow!(message),
                        Some(job_id.as_str()),
                        Some(server_id.as_str())
                    );
                }
                // Missing inputs (S3 cleared, Redis rebooted, etc.)
                // Can be retried.
                Ok(RunJobResponse::MissingJobInputs { server_id, inputs }) => {
                    // If we retry, send the inputs again
                    if inputs.is_empty() {
                        missing_inputs = hashed_inputs.keys().cloned().collect();
                    } else {
                        missing_inputs = inputs;
                    }
                    debug!(
                        "[{out_pretty}, {job_id}, {server_id}]: Missing distributed compilation job inputs"
                    );
                    retry_or_bail!(
                        anyhow!("Missing distributed compilation job inputs"),
                        Some(job_id.as_str()),
                        Some(server_id.as_str())
                    );
                }
                // Missing toolchain (S3 cleared, Redis rebooted, etc.)
                // Can be retried.
                Ok(RunJobResponse::MissingToolchain { server_id }) => {
                    // If we retry, send the toolchain again
                    has_toolchain = false;
                    debug!(
                        "[{out_pretty}, {job_id}, {server_id}]: Missing distributed compilation job toolchain"
                    );
                    retry_or_bail!(
                        anyhow!("Missing distributed compilation job toolchain"),
                        Some(job_id.as_str()),
                        Some(server_id.as_str())
                    );
                }
                // Missing result (S3 cleared, Redis rebooted, etc.),
                // or build server failed to write the job result due
                // to e.g. network error, server shutdown, etc.
                // Can be retried.
                Ok(RunJobResponse::MissingJobResult { server_id }) => {
                    debug!(
                        "[{out_pretty}, {job_id}, {server_id}]: Missing distributed compilation job result"
                    );
                    retry_or_bail!(
                        anyhow!("Missing distributed compilation job result"),
                        Some(job_id.as_str()),
                        Some(server_id.as_str())
                    );
                }
                // Other (e.g. client network, timeout, etc.) errors
                // Can be retried.
                Err(err) => retry_or_bail!(anyhow!(err), Some(job_id.as_str())),
            };

            debug!(
                "[{out_pretty}, {job_id}, {server_id}]: Fetched {:?}",
                build_result
                    .outputs
                    .iter()
                    .map(|(p, bs)| (p, bs.lens().to_string()))
                    .collect::<Vec<_>>()
            );

            if !build_result.output.success() {
                break Ok((DistType::Ok(server_id), build_result.output));
            }

            let mut outputs = outputs.iter().collect::<Vec<_>>();
            let build_outputs_count = build_result.outputs.len();
            let build_outputs = build_result.outputs.into_iter();

            struct BuildOutput {
                dir: PathBuf,
                path: PathBuf,
                data: dist::OutputData,
                expected_size: u64,
            }

            let mut output_paths = Vec::with_capacity(build_outputs_count);

            // Aggregate outputs into a list so if an error occurs while unpacking
            // an output, we can clean up everything that's been written so far.
            let unpack_result = build_outputs.map(|(path, data)| {
                let path = path_transformer.to_local(&path).with_context(|| {
                    format!("[{out_pretty}, {job_id}, {server_id}]: unable to transform output path {path}")
                });

                let path = match path {
                    Ok(path) => path,
                    Err(err) => return Err(err),
                };

                let out = BuildOutput {
                    expected_size: data.lens().actual,
                    data,
                    // Do this in the iterator adapter so we can borrow `outputs` as mutable
                    dir: outputs
                        .swap_remove(outputs.iter().position(|o| o.path == path).unwrap())
                        .dir
                        .clone(),
                    path,
                };

                output_paths.push(out.path.clone());

                Ok(out)
            });

            let unpack_result = futures::stream::iter(unpack_result);

            // Ensure the parent dirs exist
            let unpack_result = unpack_result.and_then(|out| async {
                let dir = out.dir.as_path();
                let path = out.path.as_path();

                crate::util::async_fs::try_exists(dir)
                    .and_then(|exists| async move {
                        if !exists {
                            crate::util::async_fs::create_dir_all(dir).await
                        } else {
                            Ok(())
                        }
                    })
                    .await
                    .with_context(|| format!("Failed to create output dir {path:?}"))?;

                Ok(out)
            });

            // Write the output file to a tempfile and then atomically
            // move it to its final location so that other compiler invocations
            // happening in parallel don't see a partially-written file.
            let unpack_result = unpack_result.and_then(|out| async move {
                let dir = out.dir.as_path();
                let path = out.path.as_path();
                let expected_size = out.expected_size;

                // Create the output tempfile
                let mut tmp = crate::util::tempfile_in(dir).with_context(|| {
                    format!("Failed to create tempfile for output path {path:?}")
                })?;

                let bytes_written = futures::io::copy_buf(
                    futures::io::AllowStdIo::new(std::io::BufReader::new(out.data.into_reader())),
                    &mut futures::io::AllowStdIo::new(std::io::BufWriter::new(tmp.as_file_mut())),
                )
                .await
                .with_context(|| format!("Failed to write output to {path:?}"))?;

                // Verify we wrote the number of bytes we expected to
                if bytes_written != expected_size {
                    return Err(UnexpectedFileSize(expected_size, bytes_written).into());
                }

                // Persist the tempfile to its real location
                match tmp.persist(path) {
                    Ok(file) => {
                        // Sync immediately so other readers see the file.
                        // This is important for nvcc's .module_id files,
                        // which are read during many pararllel compilations.
                        file.sync_all().map(|_| file)
                    }
                    Err(err) => Err(err.error),
                }
                .with_context(|| format!("Failed to persist tempfile to {path:?}"))?;

                Ok(())
            });

            let unpack_result = unpack_result.try_collect::<()>().await;

            // Rewrite Rust compilation outputs
            let unpack_result = unpack_result.and_then(|_| {
                let output_paths = output_paths.iter().map(|o| o.as_path()).collect::<Vec<_>>();
                let extra_inputs = tc_archive
                    .as_ref()
                    .map(|p| vec![p.as_path()])
                    .unwrap_or_default();
                outputs_rewriter
                    .handle_outputs(&path_transformer, &output_paths, &extra_inputs)
                    .with_context(|| "Failed to rewrite outputs from compile")
            });

            // Clean up on errors
            if let Err(err) = unpack_result {
                // Do our best to clean up. We may end up deleting a file that we just wrote over
                // the top of, but it's better to clean up too much than too little
                for path in output_paths.iter() {
                    if let Err(e) = crate::util::async_fs::remove_file(path).await
                        && e.kind() != io::ErrorKind::NotFound
                    {
                        debug!(
                            "[{out_pretty}, {job_id}, {server_id}]: {e} while attempting to remove {path:?}"
                        );
                    }
                }

                if err.downcast_ref::<UnexpectedFileSize>().is_some() {
                    debug!("[{out_pretty}, {job_id}, {server_id}]: {err:?}");
                    retry_or_bail!(err, Some(job_id.as_str()), Some(server_id.as_str()));
                } else {
                    break Err(err);
                }
            }

            break Ok((DistType::Ok(server_id), build_result.output));
        };

        // Inform the scheduler we've received and unpacked the result.
        // This is its chance to clean up after the job.
        //
        // It's possible a network error occurs and the scheduler isn't notified
        // it should delete job artifacts. To mitigate this, sccache-dist servers
        // should use storage backends that are configured to delete job artifacts
        // after some reasonable time, like 3-24hrs.
        //
        // Intentionally ignore errors so otherwise successful builds don't fail.
        //
        // Don't unwrap `dist_compile_res` until after this, so even if the job
        // failed or the client encounters errors downloading/unpacking the result,
        // we still call `del_job` to tell the scheduler to delete the job artifacts.
        if let Some((ref job_id, _)) = job {
            let _ = dist_client.del_job(job_id).await;
        }

        dist_compile_res
    }
}

impl<T: CommandCreatorSync> Clone for Box<dyn CompilerHasher<T>> {
    fn clone(&self) -> Box<dyn CompilerHasher<T>> {
        self.box_clone()
    }
}

/// An interface to a compiler for actually invoking compilation.
#[async_trait]
pub trait Compilation<T>: Send + Sync
where
    T: CommandCreatorSync,
{
    /// Given information about a compiler command, generate a command that can
    /// execute the compiler.
    fn generate_compile_commands(
        &self,
    ) -> Result<(
        Box<dyn CompileCommand<T>>,
        Option<dist::CompileCommand>,
        Cacheable,
    )>;

    /// Create a function that will create the inputs used to perform a distributed compilation
    #[cfg(feature = "dist-client")]
    fn into_dist_packagers(self: Box<Self>) -> Result<DistPackagers>;

    fn is_locally_preprocessed(&self) -> bool {
        true
    }

    fn save_outputs_as_job_input(&self) -> bool {
        false
    }

    /// Returns an iterator over the results of this compilation.
    ///
    /// Each item is a descriptive (and unique) name of the output paired with
    /// the path where it'll show up.
    fn outputs<'a>(&'a self) -> Box<dyn Iterator<Item = FileObjectSource> + 'a>;
}

#[cfg(feature = "dist-client")]
pub trait OutputsRewriter: Send + Sync {
    /// Perform any post-compilation handling of outputs, given a Vec of the dist_path and local_path
    fn handle_outputs(
        &self,
        path_transformer: &dist::PathTransformer,
        output_paths: &[&Path],
        extra_inputs: &[&Path],
    ) -> Result<()>;
}

#[cfg(feature = "dist-client")]
pub struct NoopOutputsRewriter;
#[cfg(feature = "dist-client")]
impl OutputsRewriter for NoopOutputsRewriter {
    fn handle_outputs(
        &self,
        _path_transformer: &dist::PathTransformer,
        _output_paths: &[&Path],
        _extra_inputs: &[&Path],
    ) -> Result<()> {
        Ok(())
    }
}

/// Result of generating a hash from a compiler command.
pub struct HashResult<T>
where
    T: CommandCreatorSync,
{
    /// The hash key of the inputs.
    pub key: String,
    /// An object to use for the actual compilation, if necessary.
    pub compilation: Box<dyn Compilation<T> + 'static>,
    /// A weak key that may be used to identify the toolchain
    pub weak_toolchain_key: String,
}

/// Possible results of parsing compiler arguments.
#[derive(Debug, PartialEq, Eq)]
pub enum CompilerArguments<T> {
    /// Commandline can be handled.
    Ok(T),
    /// Cannot cache this compilation.
    CannotCache(&'static str, Option<String>),
    /// This commandline is not a compile.
    NotCompilation,
}

macro_rules! cannot_cache {
    ($why:expr) => {
        return CompilerArguments::CannotCache($why, None)
    };
    ($why:expr, $extra_info:expr) => {
        return CompilerArguments::CannotCache($why, Some($extra_info))
    };
}

macro_rules! try_or_cannot_cache {
    ($arg:expr, $why:expr) => {{
        match $arg {
            Ok(arg) => arg,
            Err(e) => cannot_cache!($why, e.to_string()),
        }
    }};
}

/// Specifics about distributed compilation.
#[derive(Debug, PartialEq, Eq)]
pub enum DistType {
    /// Distribution was not enabled.
    NoDist,
    /// Distributed compile success.
    Ok(String),
    /// Distributed compile failed.
    Error,
}

/// Specifics about cache misses.
#[derive(Debug, PartialEq, Eq)]
pub enum MissType {
    /// The compilation was not found in the cache, nothing more.
    Normal,
    /// Do not cache the results of the compilation.
    ForcedNoCache,
    /// Cache lookup was overridden, recompilation was forced.
    ForcedRecache,
    /// Cache took too long to respond.
    TimedOut,
    /// Error reading from cache
    CacheReadError,
}

/// Information about a successful cache write.
pub struct CacheWriteInfo {
    pub object_file_pretty: String,
    pub duration: Duration,
}

/// The result of a compilation or cache retrieval.
pub enum CompileResult {
    /// An error made the compilation not possible.
    Error(Duration),
    /// Result was found in cache.
    CacheHit(Duration),
    /// Result was not found in cache.
    ///
    /// The `CacheWriteFuture` will resolve when the result is finished
    /// being stored in the cache.
    CacheMiss(
        MissType,
        DistType,
        Duration, // Compilation time
        Pin<Box<dyn Future<Output = Result<CacheWriteInfo>> + Send>>,
    ),
    /// Not in cache and do not cache the results of the compilation.
    NotCached(DistType, Duration),
    /// Not in cache, but the compilation result was determined to be not cacheable.
    NotCacheable(DistType, Duration),
    /// Not in cache, but compilation failed.
    CompileFailed(DistType, Duration),
}

/// The state of `--color` options passed to a compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ColorMode {
    Off,
    On,
    #[default]
    Auto,
}

/// Can't derive(Debug) because of `CacheWriteFuture`.
impl fmt::Debug for CompileResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            CompileResult::Error(ref d) => write!(f, "CompileResult::Error({d:?})"),
            CompileResult::CacheHit(ref d) => write!(f, "CompileResult::CacheHit({d:?})"),
            CompileResult::CacheMiss(ref m, ref dt, ref d, _) => {
                write!(f, "CompileResult::CacheMiss({d:?}, {m:?}, {dt:?}, _)")
            }
            CompileResult::NotCached(ref dt, ref d) => {
                write!(f, "CompileResult::NotCached({dt:?}, {d:?}_")
            }
            CompileResult::NotCacheable(ref dt, ref d) => {
                write!(f, "CompileResult::NotCacheable({dt:?}, {d:?}_")
            }
            CompileResult::CompileFailed(ref dt, ref d) => {
                write!(f, "CompileResult::CompileFailed({dt:?}, {d:?})")
            }
        }
    }
}

/// Can't use derive(PartialEq) because of the `CacheWriteFuture`.
impl PartialEq<CompileResult> for CompileResult {
    fn eq(&self, other: &CompileResult) -> bool {
        match (self, other) {
            (&CompileResult::Error(_), &CompileResult::Error(_)) => true,
            (&CompileResult::CacheHit(_), &CompileResult::CacheHit(_)) => true,
            (CompileResult::CacheMiss(m, dt, _, _), CompileResult::CacheMiss(n, dt2, _, _)) => {
                m == n && dt == dt2
            }
            (CompileResult::NotCached(dt, _), CompileResult::NotCached(dt2, _)) => dt == dt2,
            (CompileResult::NotCacheable(dt, _), CompileResult::NotCacheable(dt2, _)) => dt == dt2,
            (CompileResult::CompileFailed(dt, _), CompileResult::CompileFailed(dt2, _)) => {
                dt == dt2
            }
            _ => false,
        }
    }
}

/// Can this result be stored in cache?
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Cacheable {
    Yes,
    No,
}

/// Control of caching behavior.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CacheControl {
    /// Default caching behavior.
    Default,
    /// Do not cache the results of the compilation.
    ForceNoCache,
    /// Ignore existing cache entries, force recompilation.
    ForceRecache,
}

/// Creates a future that will write `contents` to `path` inside of a temporary
/// directory.
///
/// The future will resolve to the temporary directory and an absolute path
/// inside that temporary directory with a file that has the same filename as
/// `path` contains the `contents` specified.
///
/// Note that when the `TempDir` is dropped it will delete all of its contents
/// including the path returned.
pub async fn write_temp_file(
    pool: &tokio::runtime::Handle,
    path: &Path,
    contents: Vec<u8>,
) -> Result<(TempDir, PathBuf)> {
    let path = path.to_owned();
    crate::util::spawn_blocking_on(pool, move || {
        let dir = crate::util::temp_dir().context("Failed to create temp dir")?;
        let src = dir.path().join(path);
        let mut file = File::create(&src)?;
        file.write_all(&contents)?;
        Ok::<_, anyhow::Error>((dir, src))
    })
    .await?
    .context("failed to write temporary file")
}

/// Returns true if the given path looks like a program known to have
/// a rustc compatible interface.
fn is_rustc_like<P: AsRef<Path>>(p: P) -> bool {
    matches!(
        p.as_ref()
            .file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .as_deref(),
        Some("rustc") | Some("clippy-driver")
    )
}

/// Returns the tool name and nvcc version if the given
/// path looks like cudafe++, cicc, ptxas, or tileiras.
///
/// First assumes each tool is located relative to nvcc
/// in the standard CUDA toolkit layout, otherwise looks
/// up the first nvcc on PATH.
///
async fn is_internal_nvcc_tool<T, P: AsRef<Path>>(
    creator: T,
    path: P,
    cwd: &Path,
    env: &[(OsString, OsString)],
    pool: &tokio::runtime::Handle,
) -> Option<(String, String)>
where
    T: CommandCreatorSync,
{
    let mut path = path.as_ref().to_path_buf();

    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let (name, nvcc) = if stem.starts_with("cudafe++") {
        // nvcc is typically a sibling of cudafe++
        // (/usr/local/cuda/bin/cudafe++)
        path.set_file_name("nvcc");
        ("cudafe++", path)
    } else if stem.starts_with("cicc") {
        // nvcc is typically one level up and over from cicc
        // (/usr/local/cuda/nvvm/bin/cicc)
        path.pop();
        path.pop();
        path.pop();
        path.push("bin");
        path.push("nvcc");
        ("cicc", path)
    } else if stem.starts_with("ptxas") {
        // nvcc is typically a sibling of ptxas
        // (/usr/local/cuda/bin/ptxas)
        path.set_file_name("nvcc");
        ("ptxas", path)
    } else if stem.starts_with("tileiras") {
        // nvcc is typically a sibling of tileiras
        // (/usr/local/cuda/bin/tileiras)
        path.set_file_name("nvcc");
        ("tileiras", path)
    } else {
        // Path is not an nvcc internal tool
        return None;
    };

    let nvcc = if nvcc.is_executable() {
        Some(nvcc)
    } else {
        let env_path = env
            .iter()
            .find(|(k, _)| k == "PATH")
            .map(|(_, v)| v.as_os_str());

        crate::util::which_in(OsStr::new("nvcc"), env_path, cwd).ok()
    }
    // Resolve compiler avoiding ccache wrappers to prevent double-caching.
    .map(|nvcc| resolve_compiler_avoiding_wrapper(&nvcc, env));

    let version = if let Some(nvcc) = nvcc {
        if let Ok(nvcc) = detect_c_compiler(creator, nvcc, &[], cwd, env, pool.clone()).await {
            nvcc.version().unwrap_or_else(|| "unknown".to_owned())
        } else {
            "unknown".to_owned()
        }
    } else {
        "unknown".to_owned()
    };

    Some((name.to_owned(), version))
}

/// Returns true if the given path looks like llc
fn is_llvm_llc<P: AsRef<Path>>(p: P) -> bool {
    p.as_ref()
        .file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .filter(|s| s.starts_with("llc"))
        .is_some()
}

/// Returns true if the given path looks like a c compiler program
///
/// This does not check c compilers, it only report programs that are definitely not rustc
fn is_known_c_compiler<P: AsRef<Path>>(p: P) -> bool {
    matches!(
        p.as_ref()
            .file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .as_deref(),
        Some(
            "cc" | "c++"
                | "gcc"
                | "g++"
                | "clang"
                | "clang++"
                | "clang-cl"
                | "cl"
                | "nvc"
                | "nvc++"
                | "nvcc"
        )
    )
}

/// If `executable` is a known compiler, return `Some(Box<Compiler>)`.
async fn detect_compiler<T>(
    creator: T,
    executable: &Path,
    cwd: &Path,
    args: &[OsString],
    env: &[(OsString, OsString)],
    pool: &tokio::runtime::Handle,
    dist_archive: Option<PathBuf>,
) -> Result<(Box<dyn Compiler<T>>, Option<Box<dyn CompilerProxy<T>>>)>
where
    T: CommandCreatorSync,
{
    // Resolve compiler avoiding ccache wrappers to prevent double-caching.
    let executable = resolve_compiler_avoiding_wrapper(executable, env);

    // First, see if this looks like rustc.

    let maybe_rustc_executable = if is_rustc_like(&executable) {
        Some(executable.as_path())
    } else if env.iter().any(|(k, _)| k == OsStr::new("CARGO")) {
        // If not, detect the scenario where cargo is configured to wrap rustc with something other than sccache.
        // This happens when sccache is used as a `RUSTC_WRAPPER` and another tool is used as a
        // `RUSTC_WORKSPACE_WRAPPER`. In that case rustc will be the first argument rather than the command.
        //
        // The check for the `CARGO` env acts as a guardrail against false positives.
        // https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-reads
        args.iter()
            .next()
            .filter(|arg1| is_rustc_like(arg1))
            .map(Path::new)
    } else {
        None
    };

    let pool = pool.clone();

    let rustc_executable = if let Some(ref rustc_executable) = maybe_rustc_executable {
        rustc_executable
    } else if let Some((name, nvcc_version)) =
        is_internal_nvcc_tool(creator.clone(), &executable, cwd, env, &pool).await
    {
        trace!("Found {name} (version: {nvcc_version})");
        match name.as_str() {
            "cudafe++" => {
                return CCompiler::new(
                    AnyCCompiler::from(CudaFE {
                        version: Some(nvcc_version),
                    }),
                    executable,
                    vec![],
                )
                .await
                .map(|c| (Box::new(c) as Box<dyn Compiler<T>>, None));
            }
            "cicc" => {
                return CCompiler::new(
                    AnyCCompiler::from(Cicc {
                        version: Some(nvcc_version),
                    }),
                    executable,
                    vec![],
                )
                .await
                .map(|c| (Box::new(c) as Box<dyn Compiler<T>>, None));
            }
            "ptxas" => {
                return CCompiler::new(
                    AnyCCompiler::from(Ptxas {
                        version: Some(nvcc_version),
                    }),
                    executable,
                    vec![],
                )
                .await
                .map(|c| (Box::new(c) as Box<dyn Compiler<T>>, None));
            }
            "tileiras" => {
                return CCompiler::new(
                    AnyCCompiler::from(Tileiras {
                        version: Some(nvcc_version),
                    }),
                    executable,
                    vec![],
                )
                .await
                .map(|c| (Box::new(c) as Box<dyn Compiler<T>>, None));
            }
            _ => unreachable!(),
        }
    } else if is_llvm_llc(&executable) {
        let mut cmd = creator.clone().new_command_sync(&executable);
        cmd.arg("--version");

        let output = run_input_output(cmd, None).await?;

        let stdout = bytes_to_str(output.stdout).context("Failed to parse output")?;

        let version = stdout.lines().next().unwrap_or("unknown").to_string();

        trace!("Found llc  (version: {version})");

        return CCompiler::new(
            AnyCCompiler::from(Llc {
                version: Some(version),
            }),
            executable,
            vec![],
        )
        .await
        .map(|c| (Box::new(c) as Box<dyn Compiler<T>>, None));
    } else if is_known_c_compiler(&executable) {
        return detect_c_compiler(creator, executable, args, cwd, env, pool)
            .await
            .map(|c| (c, None));
    } else {
        // Even if it does not look like rustc like it might still be rustc driver
        // so we do full check
        executable.as_path()
    };

    match resolve_rust_compiler(
        creator.clone(),
        executable.as_path(),
        rustc_executable,
        env,
        cwd.to_path_buf(),
        dist_archive,
        pool.clone(),
    )
    .await
    {
        Ok(res) => Ok(res),
        Err(e) => {
            // in case we attempted to test for rustc while it didn't look like it, fallback to c compiler detection one last time
            if maybe_rustc_executable.is_none() {
                detect_c_compiler(creator, executable, args, cwd, env, pool)
                    .await
                    .map(|c| (c, None))
            } else {
                Err(e)
            }
        }
    }
}

/// Tries to verify that provided executable is really rust compiler
/// or rust driver
async fn resolve_rust_compiler<T>(
    creator: T,
    executable: &Path,
    rustc_executable: &Path,
    env: &[(OsString, OsString)],
    cwd: PathBuf,
    dist_archive: Option<PathBuf>,
    pool: tokio::runtime::Handle,
) -> Result<(Box<dyn Compiler<T>>, Option<Box<dyn CompilerProxy<T>>>)>
where
    T: CommandCreatorSync,
{
    let mut child = creator.clone().new_command_sync(executable);
    // We're wrapping rustc if the executable doesn't match the detected rustc_executable. In this case the wrapper
    // expects rustc as the first argument.
    if rustc_executable != executable {
        child.arg(rustc_executable);
    }

    child.env_clear().envs(env.to_vec()).args(&["-vV"]);

    let rustc_vv = run_input_output(child, None).await.map(|output| {
        if let Ok(stdout) = bytes_to_str(&output.stdout[..])
            && stdout.starts_with("rustc ")
        {
            Ok(stdout.into_owned())
        } else {
            Err(ProcessError(output))
        }
    })?;

    // rustc -vV verification status
    match rustc_vv {
        Ok(rustc_verbose_version) => {
            let proxy = RustupProxy::find_proxy_executable::<T>(
                rustc_executable,
                "rustup",
                creator.clone(),
                env,
            );
            let creator1 = creator.clone();
            let res = proxy.and_then(move |proxy| async move {
                match proxy {
                    Ok(Some(proxy)) => {
                        trace!("Found rustup proxy executable");
                        // take the pathbuf for rustc as resolved by the proxy
                        match proxy.resolve_proxied_executable(creator1, cwd, env).await {
                            Ok((resolved_path, _time)) => {
                                trace!("Resolved path with rustup proxy {:?}", resolved_path);
                                Ok((Some(proxy), resolved_path))
                            }
                            Err(e) => {
                                trace!("Could not resolve compiler with rustup proxy: {e}");
                                Ok((None, rustc_executable.to_path_buf()))
                            }
                        }
                    }
                    Ok(None) => {
                        trace!("Did not find rustup");
                        Ok((None, rustc_executable.to_path_buf()))
                    }
                    Err(e) => {
                        trace!("Did not find rustup due to {e}, compiling without proxy");
                        Ok((None, rustc_executable.to_path_buf()))
                    }
                }
            });

            let (proxy, resolved_rustc) = res
                .await
                .map(|(proxy, resolved_compiler_executable)| {
                    (
                        proxy
                            .map(Box::new)
                            .map(|x: Box<RustupProxy>| x as Box<dyn CompilerProxy<T>>),
                        resolved_compiler_executable,
                    )
                })
                .unwrap_or_else(|_e| {
                    trace!("Compiling rust without proxy");
                    (None, rustc_executable.to_path_buf())
                });

            debug!("Using rustc at path: {resolved_rustc:?}");

            Rust::new(
                creator,
                resolved_rustc,
                env,
                &rustc_verbose_version,
                dist_archive,
                pool,
            )
            .await
            .map(|c| {
                (
                    Box::new(c) as Box<dyn Compiler<T>>,
                    proxy as Option<Box<dyn CompilerProxy<T>>>,
                )
            })
        }
        Err(e) => Err(e).context("Failed to launch subprocess for compiler determination"),
    }
}

ArgData! {
    PassThroughFlag,
    PassThrough(OsString),
}

// Establish a set of compiler flags that are required for
// valid execution of the compiler even in preprocessor mode.
// If the requested compiler invocation has any of these arguments,
// propagate them when doing our compiler vendor detection
//
// Current known required flags:
// ccbin/compiler-bindir needed for nvcc
//  This flag specifies the host compiler to use otherwise
//  gcc is expected to exist on the PATH. So if gcc doesn't exist
//  compiler detection fails if we don't pass along the ccbin arg
counted_array!(static ARGS: [ArgInfo<ArgData>; _] = [
    flag!("--allow-unsupported-compiler", ArgData::PassThroughFlag),
    take_arg!("--compiler-bindir", OsString, CanBeConcatenated(b'='), ArgData::PassThrough),
    take_arg!("--cuda-gpu-arch", OsString, CanBeConcatenated(b'='), ArgData::PassThrough),
    take_arg!("--cuda-path", OsString, CanBeConcatenated(b'='), ArgData::PassThrough),
    flag!("--cuda-path-ignore-env", ArgData::PassThroughFlag),
    flag!("--dont-use-profile", ArgData::PassThroughFlag),
    flag!("--force-cl-env-setup", ArgData::PassThroughFlag),
    flag!("--forward-unknown-opts", ArgData::PassThroughFlag),
    flag!("--forward-unknown-to-host-compiler", ArgData::PassThroughFlag),
    flag!("--forward-unknown-to-host-linker", ArgData::PassThroughFlag),
    flag!("--no-cuda-version-check", ArgData::PassThroughFlag),
    flag!("--use-local-env", ArgData::PassThroughFlag),
    flag!("-allow-unsupported-compiler", ArgData::PassThroughFlag),
    take_arg!("-ccbin", OsString, CanBeConcatenated(b'='), ArgData::PassThrough),
    flag!("-force-cl-env-setup", ArgData::PassThroughFlag),
    flag!("-forward-unknown-opts", ArgData::PassThroughFlag),
    flag!("-forward-unknown-to-host-compiler", ArgData::PassThroughFlag),
    flag!("-forward-unknown-to-host-linker", ArgData::PassThroughFlag),
    flag!("-noprof", ArgData::PassThroughFlag),
    flag!("-use-local-env", ArgData::PassThroughFlag),
]);

pub fn compiler_info_args(arguments: &[OsString]) -> Vec<OsString> {
    let mut args = vec![];
    // Iterate over all the arguments for compilation and extract
    // any that are required for any valid execution of the compiler.
    // Allowing our compiler vendor detection to always properly execute
    for arg in ArgsIter::new(arguments.iter().cloned(), &ARGS[..]).flatten() {
        if arg.get_data().is_some() {
            args.extend(arg.iter_os_strings());
        }
    }
    args
}

/// Ask `executable` for the version of the assembler it would run.
///
/// Going through the compiler rather than looking for `as` ourselves means the
/// answer accounts for the search paths the compiler was built with.
async fn assembler_version<T>(
    creator: &T,
    executable: &Path,
    extra_args: &[&str],
    env: &[(OsString, OsString)],
) -> Option<String>
where
    T: CommandCreatorSync,
{
    let mut cmd = creator.clone().new_command_sync(executable);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .envs(env.iter().map(|s| (&s.0, &s.1)))
        .args(extra_args)
        .arg("-Wa,--version")
        // Empty input from stdin and the object to stdout, so that asking for a
        // version leaves nothing behind. The assembler prints its version and
        // exits without assembling, so the object never materializes.
        .arg("-x")
        .arg("assembler")
        .arg("-c")
        .arg("-")
        .arg("-o")
        .arg("-");

    trace!("assembler_version: {cmd}");

    let output = run_input_output(cmd, None).await.ok()?;

    if !output.status.success() {
        debug!(
            "Failed to get the assembler version: {}",
            bytes_to_str(output.stderr).ok()?
        );
        return None;
    }
    // The rest of what `--version` prints is copyright boilerplate.
    let version = bytes_to_str(&output.stdout)
        .ok()?
        .lines()
        .next()?
        .trim()
        .to_owned();

    if version.is_empty() {
        return None;
    }

    Some(version)
}

/// Ask `executable` where the assembler it would run lives.
async fn assembler_path<T>(
    creator: &T,
    executable: &Path,
    extra_args: &[&str],
    cwd: &Path,
    env: &[(OsString, OsString)],
) -> Option<PathBuf>
where
    T: CommandCreatorSync,
{
    let mut cmd = creator.clone().new_command_sync(executable);
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .envs(env.iter().map(|s| (&s.0, &s.1)))
        .args(extra_args)
        .arg("-print-prog-name=as");

    trace!("assembler_path: {cmd}");

    let output = run_input_output(cmd, None).await.ok()?;

    if !output.status.success() {
        return None;
    }

    let name = bytes_to_str(&output.stdout).ok()?;
    let name = name.trim();

    if name.is_empty() {
        return None;
    }

    // GCC answers with a bare name when it leaves finding the assembler to
    // $PATH, and clang does the same when there is nothing to find, so finish
    // the search the way the compiler will: against the client's $PATH.
    let paths = env
        .iter()
        .find(|(k, _)| k == "PATH")
        .map(|(_, v)| v.clone())
        .or_else(|| env::var_os("PATH"));

    crate::util::which_in(OsStr::new(name), paths.as_deref(), cwd).ok()
}

/// Identify the assembler `executable` hands the compilation off to.
///
/// GCC never assembles by itself, and clang doesn't either when its integrated
/// assembler is disabled, so without this the object file depends on a program
/// that nothing in the cache key accounts for. The assembler gets identified the
/// same way the compiler is, by both its contents and the version it reports:
/// the contents catch a rebuild that didn't bump the version, the version speaks
/// for wrappers like Apple's `as`, where the binary we find isn't what ends up
/// doing the work.
async fn detect_assembler<T>(
    creator: &T,
    executable: &Path,
    extra_args: &[&str],
    cwd: &Path,
    env: &[(OsString, OsString)],
) -> Option<String>
where
    T: CommandCreatorSync,
{
    let version = assembler_version(creator, executable, extra_args, env).await;
    let path = assembler_path(creator, executable, extra_args, cwd, env).await;
    let contents = match &path {
        Some(path) => Digest::from_file(path)
            .await
            .map_err(|e| debug!("Failed to hash assembler {}: {}", path.display(), e))
            .ok()
            .map(|digest| digest.finish()),
        None => None,
    };
    if version.is_none() && contents.is_none() {
        return None;
    }
    debug!("Found assembler {:?} at {:?}", version, path);

    let mut m = Digest::new();
    m.update(version.unwrap_or_default().as_bytes());
    m.update(contents.unwrap_or_default().as_bytes());
    Some(m.finish())
}

async fn detect_c_compiler<T, P>(
    mut creator: T,
    executable: P,
    arguments: &[OsString],
    cwd: &Path,
    env: &[(OsString, OsString)],
    pool: tokio::runtime::Handle,
) -> Result<Box<dyn Compiler<T>>>
where
    T: CommandCreatorSync,
    P: AsRef<Path>,
{
    // NVCC needs to be first as msvc, clang, or gcc could
    // be the underlying host compiler for nvcc
    // Both clang and clang-cl define _MSC_VER on Windows, so we first
    // check for MSVC, then check whether _MT is defined, which is the
    // difference between clang and clang-cl.
    // NVHPC needs a custom version line since `__VERSION__` evaluates
    // to the EDG version.
    //
    // We prefix the information we need with `compiler_id` and `compiler_version`
    // so that we can support compilers that insert pre-amble code even in `-E` mode
    let test = b"
#if defined(__NVCC__) && defined(__NVCOMPILER)
compiler_id=nvcc-nvhpc
compiler_version=__CUDACC_VER_MAJOR__ . __CUDACC_VER_MINOR__ . __CUDACC_VER_BUILD__
compiler_version=__NVCOMPILER_MAJOR__ . __NVCOMPILER_MINOR__ . __NVCOMPILER_PATCHLEVEL__
#elif defined(__NVCC__) && defined(_MSC_VER) && !defined(__clang__)
compiler_id=nvcc-msvc
compiler_version=__CUDACC_VER_MAJOR__ . __CUDACC_VER_MINOR__ . __CUDACC_VER_BUILD__
#if !defined(__VERSION__)
compiler_version=_MSC_VER
#endif
#elif defined(__NVCC__) && defined(__clang__)
compiler_id=nvcc-clang
compiler_version=__CUDACC_VER_MAJOR__ . __CUDACC_VER_MINOR__ . __CUDACC_VER_BUILD__
#elif defined(__NVCC__)
compiler_id=nvcc
compiler_version=__CUDACC_VER_MAJOR__ . __CUDACC_VER_MINOR__ . __CUDACC_VER_BUILD__
#elif defined(_MSC_VER) && !defined(__clang__)
compiler_id=msvc
#if !defined(__VERSION__)
compiler_version=_MSC_VER
#endif
#elif defined(_MSC_VER) && defined(_MT)
compiler_id=msvc-clang
#if !defined(__VERSION__)
compiler_version=_MSC_VER
#endif
#elif defined(__NVCOMPILER)
#if defined(__cplusplus)
compiler_id=nvc++
#else
compiler_id=nvc
#endif
compiler_version=__NVCOMPILER_MAJOR__ . __NVCOMPILER_MINOR__ . __NVCOMPILER_PATCHLEVEL__
#elif defined(__clang__) && defined(__cplusplus) && defined(__apple_build_version__)
compiler_id=apple-clang++
#elif defined(__clang__) && defined(__cplusplus)
compiler_id=clang++
#elif defined(__clang__) && defined(__apple_build_version__)
compiler_id=apple-clang
#elif defined(__clang__)
compiler_id=clang
#elif defined(__GNUC__) && defined(__cplusplus)
compiler_id=g++
#elif defined(__GNUC__)
compiler_id=gcc
#elif defined(__DCC__)
compiler_id=diab
#elif defined(__CTC__)
compiler_id=tasking_vx
#else
compiler_id=unknown
#endif
compiler_version=__VERSION__
"
    .to_vec();
    let (_tempdir, src) = write_temp_file(&pool, "testfile.c".as_ref(), test).await?;

    let executable = executable.as_ref().to_owned();

    let mut cmd = creator.clone().new_command_sync(&executable);
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .envs(env.iter().map(|s| (&s.0, &s.1)))
        .args(arguments)
        .arg("-E")
        .arg(&src);

    let output = run_input_output(cmd, None)
        .await
        .context("failed to read child output")?;

    let status = output.desc();
    let stdout = bytes_to_str(output.stdout).context("Failed to parse output")?;

    let mut lines = stdout.lines().filter_map(|line| {
        let line = line.trim();
        if let Some(compiler_id) = line.strip_prefix("compiler_id=") {
            Some(compiler_id)
        } else if let Some(compiler_version) = line.strip_prefix("compiler_version=") {
            Some(compiler_version)
        } else {
            None
        }
    });

    if let Some(kind) = lines.next() {
        let mut next_version = || {
            lines
                .next()
                // In case the compiler didn't expand the macro.
                .filter(|&line| line != "__VERSION__")
                .map(|s| s.trim_matches('"').to_owned())
                .or(Some(String::from("<unknown>")))
        };

        let version = next_version();
        match kind {
            "clang" | "clang++" | "apple-clang" | "apple-clang++" => {
                trace!("Found {kind} (version: {})", version.as_ref().unwrap());
                // Identity of the assembler used with -fno-integrated-as.
                let assembler_digest =
                    detect_assembler(&creator, &executable, &["-fno-integrated-as"], cwd, env)
                        .await;

                return CCompiler::new(
                    AnyCCompiler::from(Clang::new(
                        kind.ends_with("++"), // clangplusplus
                        // kind.starts_with("apple-"), // is_appleclang
                        version, // version
                        assembler_digest,
                        Clang::find_extra_gcc_toolchain_markers(&executable),
                    )),
                    executable,
                    vec![],
                )
                .await
                .map(|c| Box::new(c) as Box<dyn Compiler<T>>);
            }
            "diab" => {
                trace!("Found {kind} (version: {})", version.as_ref().unwrap());
                return CCompiler::new(AnyCCompiler::from(Diab { version }), executable, vec![])
                    .await
                    .map(|c| Box::new(c) as Box<dyn Compiler<T>>);
            }
            "gcc" | "g++" => {
                trace!("Found {kind} (version: {})", version.as_ref().unwrap());

                let assembler_digest = detect_assembler(&creator, &executable, &[], cwd, env).await;

                let gcc = Gcc::new_from_compiler(
                    &mut creator,
                    &executable,
                    arguments.iter().cloned(),
                    env.iter().cloned(),
                    kind == "g++", // gplusplus
                    version,
                    assembler_digest,
                )
                .await?;

                let specfiles = gcc.specfiles.clone();

                return CCompiler::new(AnyCCompiler::from(gcc), executable, specfiles)
                    .await
                    .map(|c| Box::new(c) as Box<dyn Compiler<T>>);
            }
            "msvc" | "msvc-clang" => {
                let is_clang = kind == "msvc-clang";
                trace!("Found {kind} (version: {})", version.as_ref().unwrap());
                let includes_prefix = msvc::detect_showincludes_prefix(
                    &creator,
                    executable.as_ref(),
                    is_clang,
                    env,
                    &pool,
                )
                .await?;
                trace!("showIncludes prefix: '{includes_prefix}'");

                return CCompiler::new(
                    AnyCCompiler::from(Msvc {
                        includes_prefix,
                        is_clang,
                        version,
                    }),
                    executable,
                    vec![],
                )
                .await
                .map(|c| Box::new(c) as Box<dyn Compiler<T>>);
            }
            "nvcc" | "nvcc-clang" | "nvcc-msvc" | "nvcc-nvhpc" => {
                let version = version.map(|s| s.replace(' ', ""));
                let mut host_compiler_version = next_version();

                let host_compiler = match kind {
                    "nvcc-clang" => NvccHostCompiler::Clang,
                    "nvcc-nvhpc" => {
                        host_compiler_version = host_compiler_version.map(|s| s.replace(' ', ""));
                        NvccHostCompiler::Nvhpc
                    }
                    "nvcc-msvc" => NvccHostCompiler::Msvc,
                    "nvcc" => NvccHostCompiler::Gcc,
                    &_ => NvccHostCompiler::Gcc,
                };

                trace!(
                    "Found {kind} (version: {}/{})",
                    version.as_ref().unwrap(),
                    host_compiler_version.as_ref().unwrap()
                );

                let nvcc = Nvcc::new_from_compiler(
                    &mut creator,
                    &executable,
                    arguments.iter().cloned(),
                    env.iter().cloned(),
                    version,
                    host_compiler,
                    host_compiler_version,
                )
                .await?;

                let hashes = crate::util::hash_all(&nvcc.nvcc_tools).await?;
                let nvcc = CCompiler::new(AnyCCompiler::from(nvcc), executable, vec![]).await?;
                return Ok(Box::new(nvcc.with_extra_hashes(hashes)) as Box<dyn Compiler<T>>);
            }
            "nvc++" | "nvc" => {
                let version = version.map(|s| s.replace(' ', ""));
                trace!("Found {kind} (version: {})", version.as_ref().unwrap());
                let native_arch = Nvhpc::read_native_arch(&mut creator, &executable, env).await;

                let openmpi = {
                    let mut cmd = creator.clone().new_command_sync(&executable);
                    cmd.stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .envs(env.iter().map(|s| (&s.0, &s.1)))
                        .args(arguments)
                        .arg("-dD")
                        .arg("-c")
                        .arg(&src);

                    trace!("{cmd}");

                    let stdout = match run_input_output(cmd, None).await {
                        Ok(output) => Some(output.stdout),
                        Err(err) => match err.downcast::<ProcessError>() {
                            Ok(ProcessError(output)) => Some(output.stdout),
                            Err(err) => {
                                warn!("{kind} error: {err:#}");
                                None
                            }
                        },
                    };

                    if let Some(stdout) = stdout.and_then(|stdout| bytes_to_str(stdout).ok()) {
                        stdout.contains("#define _REENTRANT")
                    } else {
                        false
                    }
                };

                return CCompiler::new(
                    AnyCCompiler::from(Nvhpc {
                        nvcplusplus: kind == "nvc++",
                        openmpi,
                        version,
                        native_arch,
                    }),
                    executable,
                    vec![],
                )
                .await
                .map(|c| Box::new(c) as Box<dyn Compiler<T>>);
            }
            "tasking_vx" => {
                trace!("Found Tasking VX");
                return CCompiler::new(AnyCCompiler::from(TaskingVX), executable, vec![])
                    .await
                    .map(|c| Box::new(c) as Box<dyn Compiler<T>>);
            }
            _ => (),
        }
    }

    let stderr = bytes_to_str(output.stderr).context("Failed to parse output")?;
    debug!("nothing useful in detection output {stdout:?}");
    debug!("compiler status: {status}");
    debug!("compiler stderr:\n{stderr}");

    bail!(stderr)
}

/// If `executable` is a known compiler, return a `Box<Compiler>` containing information about it.
pub async fn get_compiler_info<T>(
    creator: T,
    executable: &Path,
    cwd: &Path,
    args: &[OsString],
    env: &[(OsString, OsString)],
    pool: &tokio::runtime::Handle,
    dist_archive: Option<PathBuf>,
) -> Result<(Box<dyn Compiler<T>>, Option<Box<dyn CompilerProxy<T>>>)>
where
    T: CommandCreatorSync,
{
    let pool = pool.clone();
    detect_compiler(creator, executable, cwd, args, env, &pool, dist_archive).await
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        cache::disk::DiskCache,
        compiler::c::ParseArgs,
        config::CacheMode,
        mock_command::*,
        server::SccacheService,
        test::{mock_storage::MockStorage, utils::*},
    };
    use fs::File;
    use std::io::{Cursor, Write};
    use std::sync::Arc;
    use std::time::Duration;
    use test_case::test_case;
    use tokio::runtime::Runtime;

    #[test]
    fn test_c_preprocessing_consistent() {
        // Must contain all possible values from `Language`.
        // Cannot use `enum_iterator` because it adds macro to the real enum,
        // and we don't want to add additional dependencies.
        let languages = vec![
            Language::AssemblerToPreprocess,
            Language::Assembler,
            Language::C,
            Language::Cxx,
            Language::GenericHeader,
            Language::CHeader,
            Language::CPreprocessed,
            Language::CxxHeader,
            Language::CxxPreprocessed,
            Language::ObjectiveC,
            Language::ObjectiveCHeader,
            Language::ObjectiveCPreprocessed,
            Language::ObjectiveCxx,
            Language::ObjectiveCxxPreprocessed,
            Language::ObjectiveCxxHeader,
            Language::Cuda,
            Language::CudaFE,
            Language::Ptx,
            Language::Cubin,
            Language::Rust,
            Language::Hip,
        ];

        assert_eq!(
            languages.len(),
            EXPECTED_LANGUAGE_COUNT,
            "update the list of supported languages"
        );

        for lang in languages {
            let maybe_processed = lang.to_c_preprocessed_language();

            if let Some(processed) = maybe_processed {
                assert!(
                    !processed.needs_c_preprocessing(),
                    "{processed:?} should not need preprocessing - it is a result of {lang:?} processing"
                );
            }

            if !lang.needs_c_preprocessing() {
                assert!(
                    maybe_processed.is_none(),
                    "{lang:?} should not be processed, but it produces the {maybe_processed:?} as a C preprocessing result"
                );

                assert!(
                    !lang.is_c_like_header(),
                    "{lang:?} should not be processed, but it is a C-like header and can be used as preprocessor input"
                );
            }
        }
    }

    #[test]
    fn test_detect_compiler_kind_gcc() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(1),
                "",
                "gcc: error: unrecognized command-line option '-vV'; did you mean '-v'?",
            )),
        );
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "\n\ncompiler_id=gcc", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c1 = detect_compiler(creator, &f.bins[0], f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::C(CCompilerKind::Gcc), c1.kind());

        // Now with implicit specfiles
        let creator = new_creator();
        let (_tempdir, specfile) =
            write_temp_file(pool, "file.spec".as_ref(), "specfile".as_bytes().to_vec())
                .wait()
                .unwrap();
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(1),
                "",
                "gcc: error: unrecognized command-line option '-vV'; did you mean '-v'?",
            )),
        );
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "\n\ncompiler_id=gcc", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                "",
                format!("Reading specs from {}", specfile.display()),
            )),
        );
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c2 = detect_compiler(creator, &f.bins[0], f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::C(CCompilerKind::Gcc), c2.kind());

        if let (Ok(c1), Ok(c2)) = (
            c1.into_any().downcast::<CCompiler<AnyCCompiler>>(),
            c2.into_any().downcast::<CCompiler<AnyCCompiler>>(),
        ) {
            assert_ne!(c1.executable_digest, c2.executable_digest);

            use crate::compiler::c::CCompilerImpl;

            let AnyCCompiler::Gcc(gcc) = c1.compiler() else {
                panic!("expected gcc, got {:?}", c1.compiler());
            };
            match gcc.parse_arguments(ParseArgs {
                arguments: ovec![
                    "-c",
                    "foo.cxx",
                    "-march=native",
                    "-mtune=native",
                    "-o",
                    "foo.o"
                ]
                .as_slice(),
                cwd: ".".as_ref(),
                env_vars: &[],
            }) {
                CompilerArguments::Ok(parsed_args) => {
                    assert_eq!(
                        parsed_args.arch_args,
                        ovec!["-march=znver2", "-mtune=znver2"]
                    );
                    assert!(!parsed_args.common_args.contains(&("-march=native".into())));
                    assert!(!parsed_args.common_args.contains(&("-mtune=native".into())));
                }
                _ => unreachable!("Failed to parse arguments"),
            }
        } else {
            unreachable!("Failed to downcast");
        }
    }

    #[test]
    fn test_detect_compiler_kind_clang() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(1),
                "",
                "clang: error: unknown argument: '-vV'",
            )),
        );
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=clang\n", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        let c = detect_compiler(creator, &f.bins[0], f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::C(CCompilerKind::Clang), c.kind());
    }

    #[test]
    fn test_detect_compiler_assembler_probes() {
        let f = TestFixture::new();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        for (kind, integrated_as) in [("gcc", false), ("clang", true)] {
            let compiler = f.mk_bin(kind).unwrap();
            let creator = new_creator();
            next_command(
                &creator,
                Ok(MockChild::new(
                    exit_status(0),
                    format!("compiler_id={}", kind),
                    "",
                )),
            );
            for (probe, output) in [
                ("-Wa,--version", "GNU assembler (GNU Binutils) 2.42"),
                ("-print-prog-name=as", "as"),
            ] {
                next_command_calls(&creator, move |args| {
                    assert!(args.iter().any(|arg| arg == probe), "{:?}", args);
                    // Clang answers for its own integrated assembler unless that
                    // one is turned off; gcc has none to turn off.
                    assert_eq!(
                        integrated_as,
                        args.iter().any(|arg| arg == "-fno-integrated-as"),
                        "{:?}",
                        args
                    );
                    Ok(MockChild::new(exit_status(0), output, ""))
                });
            }
            if kind == "gcc" {
                // Try to read gcc implicit specfiles
                next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
                // Try to read -march=native
                next_command(
                    &creator,
                    Ok(MockChild::new(
                        exit_status(0),
                        [
                            "  -mandroid                   		[disabled]",
                            "  -march=                     		znver2",
                            "  -masm=                      		att",
                            "  -mtune-ctrl=                		",
                            "  -mtune=                     		znver2",
                            "  -muclibc                    		[disabled]",
                        ]
                        .join("\n"),
                        "",
                    )),
                );
            }
            detect_compiler(creator, &compiler, f.tempdir.path(), &[], &[], pool, None)
                .wait()
                .unwrap();
        }
    }

    #[test]
    fn test_detect_compiler_must_be_clang() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        let clang = f.mk_bin("clang").unwrap();
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=clang\n", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        let c = detect_compiler(creator, &clang, f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::C(CCompilerKind::Clang), c.kind());
    }

    #[test]
    fn test_detect_compiler_vv_clang() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "clang: 13\n", "")),
        );
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=clang\n", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        let c = detect_compiler(creator, &f.bins[0], f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::C(CCompilerKind::Clang), c.kind());
    }

    #[test]
    fn test_detect_compiler_kind_msvc() {
        drop(env_logger::try_init());
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        let f = TestFixture::new();
        let srcfile = f.touch("test.h").unwrap();
        let mut s = srcfile.to_str().unwrap();
        if s.starts_with("\\\\?\\") {
            s = &s[4..];
        }
        let prefix = String::from("blah: ");
        let stderr = format!("{prefix}{s}\r\n");
        // Compiler detection output
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(1),
                "",
                "msvc-error: unknown argument: '-vV'",
            )),
        );
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "\ncompiler_id=msvc\n", "")),
        );
        // showincludes prefix detection output
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), String::new(), stderr)),
        );
        let c = detect_compiler(creator, &f.bins[0], f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::C(CCompilerKind::Msvc), c.kind());
    }

    #[test]
    fn test_detect_compiler_kind_nvcc() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        next_command(&creator, Ok(MockChild::new(exit_status(1), "", "no -vV")));
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=nvcc\n", "")),
        );
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        // Try to read nvcc all, all-major, and native archs lists
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        let c1 = detect_compiler(creator, &f.bins[0], f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::C(CCompilerKind::Nvcc), c1.kind());
    }

    #[test]
    fn test_detect_compiler_kind_nvhpc() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        next_command(&creator, Ok(MockChild::new(exit_status(1), "", "no -vV")));
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=nvc++\n", "")),
        );
        // Try to read native -tp= value
        next_command(&creator, Ok(MockChild::new(exit_status(0), "nvc++", "")));
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "ABM                 =1",
                    "ACCCGDEF            =-y 163 0xc0000000",
                    "DEFTPVAL            =-tp znver2",
                    "INFOTPVAL           =-tp znver2",
                    "TPVAL               =-tp znver2",
                    "TESTTPVAL           =znver2",
                    "USETPVAL            =-tp znver2",
                ]
                .join("\n"),
                "",
            )),
        );
        // Test whether NVHPC is mpicc/mpic++ (yes)
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(1), "#define _REENTRANT 1", "")),
        );
        let c = detect_compiler(creator, &f.bins[0], f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::C(CCompilerKind::Nvhpc), c.kind());
        let nvcxx = c
            .into_any()
            .downcast::<CCompiler<AnyCCompiler>>()
            .expect("CCompiler<AnyCCompiler>");
        let AnyCCompiler::Nvhpc(nvcxx) = nvcxx.compiler() else {
            panic!("expected nvhpc, got {:?}", nvcxx.compiler());
        };
        assert!(nvcxx.openmpi, "openmpi is true");
        assert!(nvcxx.nvcplusplus, "nvcplusplus is true");
    }

    #[test]
    fn test_detect_compiler_kind_rustc() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        // Windows uses bin, everything else uses lib. Just create both.
        fs::create_dir(f.tempdir.path().join("lib")).unwrap();
        fs::create_dir(f.tempdir.path().join("bin")).unwrap();
        let rustc = f.mk_bin("rustc.exe").unwrap();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        populate_rustc_command_mock(&creator, &f);
        let c = detect_compiler(creator, &rustc, f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::Rust, c.kind());
    }

    #[test]
    fn test_is_rustc_like() {
        drop(env_logger::try_init());
        assert!(is_rustc_like("rustc"));
        assert!(is_rustc_like("rustc.exe"));
        assert!(is_rustc_like("/path/to/rustc.exe"));
        assert!(is_rustc_like("/path/to/rustc"));
        assert!(is_rustc_like("/PATH/TO/RUSTC.EXE"));
        assert!(is_rustc_like("/Path/To/RustC.Exe"));
        assert!(is_rustc_like("/path/to/clippy-driver"));
        assert!(is_rustc_like("/path/to/clippy-driver.exe"));
        assert!(is_rustc_like("/PATH/TO/CLIPPY-DRIVER.EXE"));
        assert!(is_rustc_like("/Path/To/Clippy-Driver.Exe"));
        assert!(!is_rustc_like("rust"));
        assert!(!is_rustc_like("RUST"));
    }

    fn populate_rustc_command_mock(
        creator: &Arc<std::sync::Mutex<MockCommandCreator>>,
        f: &TestFixture,
    ) {
        // rustc --vV
        next_command(
            creator,
            Ok(MockChild::new(
                exit_status(0),
                "\
rustc 1.27.0 (3eda71b00 2018-06-19)
binary: rustc
commit-hash: 3eda71b00ad48d7bf4eef4c443e7f611fd061418
commit-date: 2018-06-19
host: x86_64-unknown-linux-gnu
release: 1.27.0
LLVM version: 6.0",
                "",
            )),
        );
        // rustc --print=sysroot
        let sysroot = f.tempdir.path().to_str().unwrap();
        next_command(creator, Ok(MockChild::new(exit_status(0), sysroot, "")));
        next_command(creator, Ok(MockChild::new(exit_status(0), sysroot, "")));
        next_command(creator, Ok(MockChild::new(exit_status(0), sysroot, "")));
    }

    #[test]
    fn test_detect_compiler_kind_rustc_workspace_wrapper() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        // Windows uses bin, everything else uses lib. Just create both.
        fs::create_dir(f.tempdir.path().join("lib")).unwrap();
        fs::create_dir(f.tempdir.path().join("bin")).unwrap();
        let rustc = f.mk_bin("rustc-workspace-wrapper").unwrap();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        populate_rustc_command_mock(&creator, &f);
        let c = detect_compiler(
            creator,
            &rustc,
            f.tempdir.path(),
            // Specifying an extension tests the ignoring
            &[OsString::from("rustc.exe")],
            &[(OsString::from("CARGO"), OsString::from("CARGO"))],
            pool,
            None,
        )
        .wait()
        .unwrap()
        .0;
        assert_eq!(CompilerKind::Rust, c.kind());

        // Test we don't detect rustc if the first arg is not rustc
        let creator = new_creator();
        next_command(&creator, Ok(MockChild::new(exit_status(1), "", "no -vV")));
        populate_rustc_command_mock(&creator, &f);
        assert!(
            detect_compiler(
                creator,
                &rustc,
                f.tempdir.path(),
                &[OsString::from("not-rustc")],
                &[(OsString::from("CARGO"), OsString::from("CARGO"))],
                pool,
                None,
            )
            .wait()
            .is_err()
        );

        // Test we detect rustc if the CARGO env is not defined
        let creator = new_creator();
        populate_rustc_command_mock(&creator, &f);
        assert!(
            detect_compiler(
                creator,
                &rustc,
                f.tempdir.path(),
                &[OsString::from("rustc")],
                &[],
                pool,
                None,
            )
            .wait()
            .is_ok()
        );
    }

    #[test]
    fn test_detect_compiler_kind_diab() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        next_command(&creator, Ok(MockChild::new(exit_status(1), "", "no -vV")));
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "\ncompiler_id=diab\n", "")),
        );
        let c = detect_compiler(creator, &f.bins[0], f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        assert_eq!(CompilerKind::C(CCompilerKind::Diab), c.kind());
    }

    #[test]
    fn test_detect_compiler_kind_unknown() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        next_command(&creator, Ok(MockChild::new(exit_status(1), "", "no -vV")));
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "something", "")),
        );
        assert!(
            detect_compiler(
                creator,
                "/foo/bar".as_ref(),
                f.tempdir.path(),
                &[],
                &[],
                pool,
                None
            )
            .wait()
            .is_err()
        );
    }

    #[test]
    fn test_detect_compiler_kind_process_fail() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        next_command(&creator, Ok(MockChild::new(exit_status(1), "", "no -vV")));
        next_command(&creator, Ok(MockChild::new(exit_status(1), "", "")));
        assert!(
            detect_compiler(
                creator,
                "/foo/bar".as_ref(),
                f.tempdir.path(),
                &[],
                &[],
                pool,
                None
            )
            .wait()
            .is_err()
        );
    }

    #[test]
    fn test_compiler_version_affects_hash() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let clang = f.mk_bin("clang").unwrap();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        let storage = Arc::new(MockStorage::new(None, true));
        let service = SccacheService::mock_with_storage(storage, pool.clone());
        let arguments = ovec!["-c", "foo.c", "-o", "foo.o", "-MD"];
        let cwd = f.tempdir.path();
        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();

        let results: Vec<_> = [11, 12]
            .iter()
            .map(|version| {
                let output = format!("compiler_id=clang\ncompiler_version=\"{version}.0.0\"");
                next_command(&creator, Ok(MockChild::new(exit_status(0), output, "")));
                next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
                let c = detect_compiler(
                    creator.clone(),
                    &clang,
                    f.tempdir.path(),
                    &[],
                    &[],
                    pool,
                    None,
                )
                .wait()
                .unwrap()
                .0;
                // The generate_hash_key preprocessor invocation.
                let dep = f.tempdir.path().join("foo.d");
                next_command_calls(&creator, move |_| {
                    // Pretend to preprocess something.
                    let mut f = File::create(&dep)?;
                    f.write_all(b"foo.o :")?;
                    Ok(MockChild::new(exit_status(0), "", ""))
                });
                let hasher = match c.parse_arguments(&arguments, cwd, &[], &[]) {
                    CompilerArguments::Ok(h) => h,
                    o => panic!("Bad result from parse_arguments: {o:?}"),
                };
                hasher
                    .generate_hash_key(GenerateHashKeyArgs {
                        cache_control: CacheControl::Default,
                        creator: &creator,
                        cwd,
                        env_vars: &[],
                        runtime: pool,
                        service: &service,
                        dist_compile: false,
                    })
                    .wait()
                    .unwrap()
                    .0
            })
            .collect();
        assert_eq!(results.len(), 2);
        assert_ne!(results[0].key, results[1].key);
    }

    #[test]
    fn test_assembler_affects_hash() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let clang = f.mk_bin("clang").unwrap();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        let storage = Arc::new(MockStorage::new(None, true));
        let service = SccacheService::mock_with_storage(storage, pool.clone());
        let cwd = f.tempdir.path();
        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();

        let key = |version: &str, path: &str, arguments: &[OsString]| {
            next_command(
                &creator,
                Ok(MockChild::new(
                    exit_status(0),
                    "compiler_id=clang\ncompiler_version=\"16.0.0\"",
                    "",
                )),
            );
            next_assembler(&creator, version, path);
            let c = detect_compiler(
                creator.clone(),
                &clang,
                f.tempdir.path(),
                &[],
                &[],
                pool,
                None,
            )
            .wait()
            .unwrap()
            .0;
            // Generate dependencies
            next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
            let hasher = match c.parse_arguments(arguments, cwd, &[], &[]) {
                CompilerArguments::Ok(h) => h,
                o => panic!("Bad result from parse_arguments: {:?}", o),
            };
            hasher
                .generate_hash_key(GenerateHashKeyArgs {
                    cache_control: CacheControl::Default,
                    creator: &creator,
                    cwd,
                    env_vars: &[],
                    runtime: pool,
                    service: &service,
                    dist_compile: false,
                })
                .wait()
                .unwrap()
                .0
                .key
        };

        // Two assemblers reporting the same version, one of them a rebuild.
        let mk_as = |name, contents: &'static str| {
            mk_bin_contents(f.tempdir.path(), name, |mut f| {
                f.write_all(contents.as_bytes())
            })
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned()
        };
        let as1 = mk_as("as1", "assembler");
        let as2 = mk_as("as2", "assembler, rebuilt");
        let (as1, as2) = (as1.as_str(), as2.as_str());
        let version = "GNU assembler (GNU Binutils) 2.42";

        let external = ovec!["-c", "foo.c", "-o", "foo.o", "-fno-integrated-as"];
        assert_ne!(
            key(version, as1, &external),
            key("GNU assembler (GNU Binutils) 2.44", as1, &external)
        );
        assert_ne!(key(version, as1, &external), key(version, as2, &external));

        // With the integrated assembler, which is the default, no external
        // assembler runs and none of this is our business.
        let integrated = ovec!["-c", "foo.c", "-o", "foo.o"];
        assert_eq!(
            key(version, as1, &integrated),
            key("GNU assembler (GNU Binutils) 2.44", as2, &integrated)
        );
    }

    #[test]
    fn test_common_args_affects_hash() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        let storage = Arc::new(MockStorage::new(None, true));
        let service = SccacheService::mock_with_storage(storage, pool.clone());
        let output = "compiler_id=clang\ncompiler_version=\"16.0.0\"";
        let arguments = [
            ovec!["-c", "foo.c", "-o", "foo.o", "-DHELLO"],
            ovec!["-c", "foo.c", "-o", "foo.o", "-DHI"],
            ovec!["-c", "foo.c", "-o", "foo.o"],
        ];
        let cwd = f.tempdir.path();

        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();

        let results: Vec<_> = arguments
            .iter()
            .map(|argument| {
                next_command(
                    &creator,
                    Ok(MockChild::new(
                        exit_status(1),
                        "",
                        "clang: error: unknown argument: '-vV'",
                    )),
                );
                next_command(&creator, Ok(MockChild::new(exit_status(0), output, "")));
                next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
                let c = detect_compiler(
                    creator.clone(),
                    &f.bins[0],
                    f.tempdir.path(),
                    &[],
                    &[],
                    pool,
                    None,
                )
                .wait()
                .unwrap()
                .0;
                // The generate_hash_key preprocessor invocation.
                next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
                let hasher = match c.parse_arguments(argument, cwd, &[], &[]) {
                    CompilerArguments::Ok(h) => h,
                    o => panic!("Bad result from parse_arguments: {o:?}"),
                };
                hasher
                    .generate_hash_key(GenerateHashKeyArgs {
                        cache_control: CacheControl::Default,
                        creator: &creator,
                        cwd,
                        env_vars: &[],
                        runtime: pool,
                        service: &service,
                        dist_compile: false,
                    })
                    .wait()
                    .unwrap()
                    .0
            })
            .collect();

        assert_eq!(results.len(), 3);
        assert_ne!(results[0].key, results[1].key);
        assert_ne!(results[1].key, results[2].key);
        assert_ne!(results[0].key, results[2].key);
    }

    #[test]
    fn test_preprocessed_file_works_without_preprocessor_call() {
        drop(env_logger::try_init());
        let f = TestFixture::new();
        let clang = f.mk_bin("clang").unwrap();
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        let storage = Arc::new(MockStorage::new(None, true));
        let service = SccacheService::mock_with_storage(storage, pool.clone());
        let output = "compiler_id=clang\ncompiler_version=\"16.0.0\"";
        let cwd = f.tempdir.path();

        let results: Vec<_> = ["foo.c", "foo.i"]
            .iter()
            .map(|file| {
                let arguments = ovec!["-c", file, "-o", "foo.o"];

                // Write a dummy input file so the dependencies cache can work
                std::fs::write(f.tempdir.path().join(file), "int foo(void) { return 0; }").unwrap();

                next_command(&creator, Ok(MockChild::new(exit_status(0), output, "")));
                next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
                let c = detect_compiler(
                    creator.clone(),
                    &clang,
                    f.tempdir.path(),
                    &[],
                    &[],
                    pool,
                    None,
                )
                .wait()
                .unwrap()
                .0;

                // Only run the preprocessor on the non-preprocessed file
                if !file.ends_with("i") {
                    next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
                }
                let hasher = match c.parse_arguments(&arguments, cwd, &[], &[]) {
                    CompilerArguments::Ok(h) => h,
                    o => panic!("Bad result from parse_arguments: {o:?}"),
                };
                hasher
                    .generate_hash_key(GenerateHashKeyArgs {
                        cache_control: CacheControl::Default,
                        creator: &creator,
                        cwd,
                        env_vars: &[],
                        runtime: pool,
                        service: &service,
                        dist_compile: false,
                    })
                    .wait()
                    .unwrap()
                    .0
            })
            .collect();
        assert_eq!(results.len(), 2);
        assert_ne!(results[0].key, results[1].key);
    }

    #[test]
    fn test_get_compiler_info() {
        drop(env_logger::try_init());
        let creator = new_creator();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle();
        let f = TestFixture::new();
        // Pretend to be GCC.
        let gcc = f.mk_bin("gcc").unwrap();
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=gcc", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c = get_compiler_info(creator, &gcc, f.tempdir.path(), &[], &[], pool, None)
            .wait()
            .unwrap()
            .0;
        // digest of an empty file.
        assert_eq!(CompilerKind::C(CCompilerKind::Gcc), c.kind());
    }

    #[test]
    fn test_compiler_get_cached_or_compile() {
        drop(env_logger::try_init());
        let pending = crate::server::SccacheGauge::default();
        let creator = new_creator();
        let f = TestFixture::new();
        let gcc = f.mk_bin("gcc").unwrap();
        let runtime = Runtime::new().unwrap();
        let pool = runtime.handle().clone();
        let storage = Arc::new(DiskCache::new(
            f.tempdir.path().join("cache"),
            u64::MAX,
            CacheMode::ReadWrite,
            vec![],
        ));
        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();
        let service = SccacheService::mock_with_storage(storage.clone(), pool.clone());

        // Pretend to be GCC.
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=gcc", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c = get_compiler_info(
            creator.clone(),
            &gcc,
            f.tempdir.path(),
            &[],
            &[],
            &pool,
            None,
        )
        .wait()
        .unwrap()
        .0;
        // The generate_hash_key preprocessor invocation.
        let dep = f.tempdir.path().join("foo.d");
        let d = dep.clone();
        next_command_calls(&creator, move |_| {
            // Pretend to preprocess something.
            let mut f = File::create(&d)?;
            f.write_all(b"foo.o :")?;
            Ok(MockChild::new(exit_status(0), "", ""))
        });
        // The compiler invocation.
        const COMPILER_STDOUT: &[u8] = b"compiler stdout";
        const COMPILER_STDERR: &[u8] = b"compiler stderr";
        let obj = f.tempdir.path().join("foo.o");
        let o = obj.clone();
        next_command_calls(&creator, move |_| {
            // Pretend to compile something.
            let mut f = File::create(&o)?;
            f.write_all(b"file contents")?;
            Ok(MockChild::new(
                exit_status(0),
                COMPILER_STDOUT,
                COMPILER_STDERR,
            ))
        });
        let cwd = f.tempdir.path();
        let arguments = ovec!["-c", "foo.c", "-o", "foo.o", "-MD"];
        let hasher = match c.parse_arguments(&arguments, cwd, &[], &[]) {
            CompilerArguments::Ok(h) => h,
            o => panic!("Bad result from parse_arguments: {o:?}"),
        };
        let hasher2 = hasher.clone();
        let (cached, res) = crate::util::block_on(&runtime, async {
            hasher2
                .get_cached_or_compile(
                    &service,
                    None,
                    creator.clone(),
                    arguments.clone(),
                    cwd.to_path_buf(),
                    vec![],
                    CacheControl::Default,
                    pool.clone(),
                    pending.increment(),
                )
                .await
        })
        .unwrap();
        // Ensure that the dependency file was created.
        assert!(fs::metadata(&dep).map(|m| m.len() > 0).unwrap());
        // Ensure that the object file was created.
        assert!(fs::metadata(&obj).map(|m| m.len() > 0).unwrap());
        match cached {
            CompileResult::CacheMiss(MissType::Normal, DistType::NoDist, _, f) => {
                // wait on cache write future so we don't race with it!
                f.wait().unwrap();
            }
            _ => panic!("Unexpected compile result: {cached:?}"),
        }
        assert_eq!(0, res.code().unwrap());
        assert_eq!(COMPILER_STDOUT, res.stdout.as_slice());
        assert_eq!(COMPILER_STDERR, res.stderr.as_slice());
        // Now compile again, which should be a cache hit.
        fs::remove_file(&obj).unwrap();
        // The preprocessor invocation.
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // There should be no actual compiler invocation.
        let hasher2 = hasher.clone();
        let (cached, res) = crate::util::block_on(&runtime, async {
            hasher2
                .get_cached_or_compile(
                    &service,
                    None,
                    creator,
                    arguments,
                    cwd.to_path_buf(),
                    vec![],
                    CacheControl::Default,
                    pool,
                    pending.increment(),
                )
                .await
        })
        .unwrap();
        // Ensure that the object file was created.
        assert!(fs::metadata(&obj).map(|m| m.len() > 0).unwrap());
        assert_eq!(CompileResult::CacheHit(Duration::new(0, 0)), cached);
        assert_eq!(0, res.code().unwrap());
        assert_eq!(COMPILER_STDOUT, res.stdout.as_slice());
        assert_eq!(COMPILER_STDERR, res.stderr.as_slice());
    }

    #[test_case(true ; "server has toolchain")]
    #[test_case(false ; "server needs toolchain")]
    #[cfg(feature = "dist-client")]
    fn test_compiler_get_cached_or_compile_dist(server_already_has_toolchain: bool) {
        drop(env_logger::try_init());
        let pending = crate::server::SccacheGauge::default();
        let creator = new_creator();
        let f = TestFixture::new();
        let gcc = f.mk_bin("gcc").unwrap();
        let runtime = Runtime::new().unwrap();
        let pool = runtime.handle().clone();
        let storage = Arc::new(DiskCache::new(
            f.tempdir.path().join("cache"),
            u64::MAX,
            CacheMode::ReadWrite,
            vec![],
        ));
        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();
        // Pretend to be GCC.
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=gcc", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c = get_compiler_info(
            creator.clone(),
            &gcc,
            f.tempdir.path(),
            &[],
            &[],
            &pool,
            None,
        )
        .wait()
        .unwrap()
        .0;
        // The generate_hash_key preprocessor invocation.
        let dep = f.tempdir.path().join("foo.d");
        let d = dep.clone();
        next_command_calls(&creator, move |_| {
            // Pretend to preprocess something.
            let mut f = File::create(&d)?;
            f.write_all(b"foo.o :")?;
            Ok(MockChild::new(exit_status(0), "", ""))
        });
        // The inputs_packager preprocessor invocation.
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // The compiler invocation.
        const COMPILER_STDOUT: &[u8] = b"compiler stdout";
        const COMPILER_STDERR: &[u8] = b"compiler stderr";
        let obj = f.tempdir.path().join("foo.o");
        // Dist client will do the compilation
        let dist_client = test_dist::OneshotClient::new(
            0,
            COMPILER_STDOUT.to_owned(),
            COMPILER_STDERR.to_owned(),
            server_already_has_toolchain,
        );
        let service = server::SccacheService::mock_with_dist_client(
            dist_client.clone(),
            storage.clone(),
            pool.clone(),
        );

        let cwd = f.tempdir.path();
        let arguments = ovec!["-c", "foo.c", "-o", "foo.o", "-MD"];
        let hasher = match c.parse_arguments(&arguments, cwd, &[], &[]) {
            CompilerArguments::Ok(h) => h,
            o => panic!("Bad result from parse_arguments: {o:?}"),
        };
        let hasher2 = hasher.clone();
        let (cached, res) = crate::util::block_on(&runtime, async {
            hasher2
                .get_cached_or_compile(
                    &service,
                    Some(dist_client.clone()),
                    creator.clone(),
                    arguments.clone(),
                    cwd.to_path_buf(),
                    vec![],
                    CacheControl::Default,
                    pool.clone(),
                    pending.increment(),
                )
                .await
        })
        .unwrap();
        // Ensure that the dependency file was created.
        assert!(fs::metadata(&dep).map(|m| m.len() > 0).unwrap());
        // Ensure that the object file was created.
        assert!(fs::metadata(&obj).map(|m| m.len() > 0).unwrap());
        match cached {
            CompileResult::CacheMiss(MissType::Normal, DistType::Ok(_), _, f) => {
                // wait on cache write future so we don't race with it!
                f.wait().unwrap();
            }
            _ => panic!("Unexpected compile result: {cached:?}"),
        }
        assert_eq!(0, res.code().unwrap());
        assert_eq!(COMPILER_STDOUT, res.stdout.as_slice());
        assert_eq!(COMPILER_STDERR, res.stderr.as_slice());
        // Now compile again, which should be a cache hit.
        fs::remove_file(&obj).unwrap();
        // The generate_hash_key preprocessor invocation.
        let d = dep.clone();
        next_command_calls(&creator, move |_| {
            // Pretend to preprocess something.
            let mut f = File::create(&d)?;
            f.write_all(b"foo.o :")?;
            Ok(MockChild::new(exit_status(0), "", ""))
        });
        // There should be no actual compiler invocation.
        let hasher2 = hasher.clone();
        let (cached, res) = crate::util::block_on(&runtime, async {
            hasher2
                .get_cached_or_compile(
                    &service,
                    Some(dist_client.clone()),
                    creator,
                    arguments,
                    cwd.to_path_buf(),
                    vec![],
                    CacheControl::Default,
                    pool,
                    pending.increment(),
                )
                .await
        })
        .unwrap();
        // Ensure that the object file was created.
        assert!(fs::metadata(&obj).map(|m| m.len() > 0).unwrap());
        assert_eq!(CompileResult::CacheHit(Duration::new(0, 0)), cached);
        assert_eq!(0, res.code().unwrap());
        assert_eq!(COMPILER_STDOUT, res.stdout.as_slice());
        assert_eq!(COMPILER_STDERR, res.stderr.as_slice());
    }

    /// Test that a cache read that results in an error is treated as a cache
    /// miss.
    #[test]
    fn test_compiler_get_cached_or_compile_cache_error() {
        drop(env_logger::try_init());
        let pending = crate::server::SccacheGauge::default();
        let creator = new_creator();
        let f = TestFixture::new();
        let gcc = f.mk_bin("gcc").unwrap();
        let runtime = Runtime::new().unwrap();
        let pool = runtime.handle().clone();
        let storage = Arc::new(MockStorage::new(None, true));
        let service = SccacheService::mock_with_storage(storage.clone(), pool.clone());

        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();
        // Pretend to be GCC.
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=gcc", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c = get_compiler_info(
            creator.clone(),
            &gcc,
            f.tempdir.path(),
            &[],
            &[],
            &pool,
            None,
        )
        .wait()
        .unwrap()
        .0;
        // The generate_hash_key preprocessor invocation.
        let dep = f.tempdir.path().join("foo.d");
        let d = dep.clone();
        next_command_calls(&creator, move |_| {
            // Pretend to preprocess something.
            let mut f = File::create(&d)?;
            f.write_all(b"foo.o :")?;
            Ok(MockChild::new(exit_status(0), "", ""))
        });
        // The compiler invocation.
        const COMPILER_STDOUT: &[u8] = b"compiler stdout";
        const COMPILER_STDERR: &[u8] = b"compiler stderr";
        let obj = f.tempdir.path().join("foo.o");
        let o = obj.clone();
        next_command_calls(&creator, move |_| {
            // Pretend to compile something.
            let mut f = File::create(&o)?;
            f.write_all(b"file contents")?;
            Ok(MockChild::new(
                exit_status(0),
                COMPILER_STDOUT,
                COMPILER_STDERR,
            ))
        });
        let cwd = f.tempdir.path();
        let arguments = ovec!["-c", "foo.c", "-o", "foo.o", "-MD"];
        let hasher = match c.parse_arguments(&arguments, cwd, &[], &[]) {
            CompilerArguments::Ok(h) => h,
            o => panic!("Bad result from parse_arguments: {o:?}"),
        };
        // The cache will return an error.
        storage.next_get(Err(anyhow!("Source Error")));
        storage.next_get(Err(anyhow!("Object Error")));
        let (cached, res) = crate::util::block_on(
            &runtime,
            hasher.get_cached_or_compile(
                &service,
                None,
                creator,
                arguments.clone(),
                cwd.to_path_buf(),
                vec![],
                CacheControl::Default,
                pool,
                pending.increment(),
            ),
        )
        .unwrap();
        // Ensure that the dependency file was created.
        assert!(fs::metadata(&dep).map(|m| m.len() > 0).unwrap());
        // Ensure that the object file was created.
        assert!(fs::metadata(&obj).map(|m| m.len() > 0).unwrap());
        match cached {
            CompileResult::CacheMiss(MissType::CacheReadError, DistType::NoDist, _, f) => {
                // wait on cache write future so we don't race with it!
                let _ = f.wait();
            }
            _ => panic!("Unexpected compile result: {cached:?}"),
        }

        assert_eq!(0, res.code().unwrap());
        assert_eq!(COMPILER_STDOUT, res.stdout.as_slice());
        assert_eq!(COMPILER_STDERR, res.stderr.as_slice());
    }

    #[test]
    /// Test that cache read timing is recorded.
    fn test_compiler_get_cached_or_compile_cache_get_timing() {
        drop(env_logger::try_init());
        let pending = crate::server::SccacheGauge::default();
        let creator = new_creator();
        let f = TestFixture::new();
        let gcc = f.mk_bin("gcc").unwrap();
        let runtime = Runtime::new().unwrap();
        let pool = runtime.handle().clone();
        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();
        // Make our storage wait 2ms for each get/put operation.
        let storage_delay = Duration::from_millis(2);
        let storage = Arc::new(MockStorage::new(Some(storage_delay), true));
        let service = SccacheService::mock_with_storage(storage.clone(), pool.clone());
        // Pretend to be GCC.
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=gcc", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c = get_compiler_info(
            creator.clone(),
            &gcc,
            f.tempdir.path(),
            &[],
            &[],
            &pool,
            None,
        )
        .wait()
        .unwrap()
        .0;
        // The generate_hash_key preprocessor invocation.
        let dep = f.tempdir.path().join("foo.d");
        let d = dep.clone();
        next_command_calls(&creator, move |_| {
            // Pretend to preprocess something.
            let mut f = File::create(&d)?;
            f.write_all(b"foo.o :")?;
            Ok(MockChild::new(exit_status(0), "", ""))
        });
        // The compiler invocation.
        const COMPILER_STDOUT: &[u8] = b"compiler stdout";
        const COMPILER_STDERR: &[u8] = b"compiler stderr";
        let dep_file: &[u8] = b"dependency file";
        let obj_file: &[u8] = &[1, 2, 3, 4];
        // A cache entry to hand out
        let mut cachewrite = CacheWrite::new();
        cachewrite
            .put_stdout(COMPILER_STDOUT)
            .expect("Failed to store stdout");
        cachewrite
            .put_stderr(COMPILER_STDERR)
            .expect("Failed to store stderr");
        cachewrite
            .put_object("dep", &mut Cursor::new(dep_file), None)
            .expect("Failed to store cache object");
        cachewrite
            .put_object("obj", &mut Cursor::new(obj_file), None)
            .expect("Failed to store cache object");

        let entry = cachewrite.finish().expect("Failed to finish cache entry");

        let cwd = f.tempdir.path();
        let arguments = ovec!["-c", "foo.c", "-o", "foo.o", "-MD"];
        let hasher = match c.parse_arguments(&arguments, cwd, &[], &[]) {
            CompilerArguments::Ok(h) => h,
            o => panic!("Bad result from parse_arguments: {o:?}"),
        };
        storage.next_get(Err(anyhow!("Source error")));
        storage.next_get(Ok(Cache::Hit(entry.into())));
        let (cached, _res) = crate::util::block_on(
            &runtime,
            hasher.get_cached_or_compile(
                &service,
                None,
                creator,
                arguments.clone(),
                cwd.to_path_buf(),
                vec![],
                CacheControl::Default,
                pool,
                pending.increment(),
            ),
        )
        .unwrap();
        // Ensure that the dependency file was created.
        assert!(fs::metadata(&dep).map(|m| m.len() > 0).unwrap());
        match cached {
            CompileResult::CacheHit(duration) => {
                assert!(duration >= storage_delay);
            }
            _ => panic!("Unexpected compile result: {cached:?}"),
        }
    }

    #[test]
    fn test_compiler_get_cached_or_compile_force_recache() {
        drop(env_logger::try_init());
        let pending = crate::server::SccacheGauge::default();
        let creator = new_creator();
        let f = TestFixture::new();
        let gcc = f.mk_bin("gcc").unwrap();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(2)
            .build()
            .unwrap();
        let pool = runtime.handle().clone();
        let storage = Arc::new(DiskCache::new(
            f.tempdir.path().join("cache"),
            u64::MAX,
            CacheMode::ReadWrite,
            vec![],
        ));
        let service = SccacheService::mock_with_storage(storage.clone(), pool.clone());
        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();
        // Pretend to be GCC.
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=gcc", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c = get_compiler_info(
            creator.clone(),
            &gcc,
            f.tempdir.path(),
            &[],
            &[],
            &pool,
            None,
        )
        .wait()
        .unwrap()
        .0;
        const COMPILER_STDOUT: &[u8] = b"compiler stdout";
        const COMPILER_STDERR: &[u8] = b"compiler stderr";
        // The compiler should be invoked twice, since we're forcing
        // recaching.
        let obj = f.tempdir.path().join("foo.o");
        let dep = f.tempdir.path().join("foo.d");
        for _ in 0..2 {
            // The generate_hash_key preprocessor invocation.
            let d = dep.clone();
            next_command_calls(&creator, move |_| {
                // Pretend to preprocess something.
                let mut f = File::create(&d)?;
                f.write_all(b"foo.o :")?;
                Ok(MockChild::new(exit_status(0), "", ""))
            });
            // The compiler invocation.
            let o = obj.clone();
            next_command_calls(&creator, move |_| {
                // Pretend to compile something.
                let mut f = File::create(&o)?;
                f.write_all(b"file contents")?;
                Ok(MockChild::new(
                    exit_status(0),
                    COMPILER_STDOUT,
                    COMPILER_STDERR,
                ))
            });
        }
        let cwd = f.tempdir.path();
        let arguments = ovec!["-c", "foo.c", "-o", "foo.o", "-MD"];
        let hasher = match c.parse_arguments(&arguments, cwd, &[], &[]) {
            CompilerArguments::Ok(h) => h,
            o => panic!("Bad result from parse_arguments: {o:?}"),
        };
        let hasher2 = hasher.clone();
        let (cached, res) = crate::util::block_on(&runtime, async {
            hasher2
                .get_cached_or_compile(
                    &service,
                    None,
                    creator.clone(),
                    arguments.clone(),
                    cwd.to_path_buf(),
                    vec![],
                    CacheControl::Default,
                    pool.clone(),
                    pending.increment(),
                )
                .await
        })
        .unwrap();
        // Ensure that the dependency file was created.
        assert!(fs::metadata(&dep).map(|m| m.len() > 0).unwrap());
        // Ensure that the object file was created.
        assert!(fs::metadata(&obj).map(|m| m.len() > 0).unwrap());
        match cached {
            CompileResult::CacheMiss(MissType::Normal, DistType::NoDist, _, f) => {
                // wait on cache write future so we don't race with it!
                f.wait().unwrap();
            }
            _ => panic!("Unexpected compile result: {cached:?}"),
        }
        assert_eq!(0, res.code().unwrap());
        assert_eq!(COMPILER_STDOUT, res.stdout.as_slice());
        assert_eq!(COMPILER_STDERR, res.stderr.as_slice());
        // Now compile again, but force recaching.
        fs::remove_file(&dep).unwrap();
        fs::remove_file(&obj).unwrap();
        let (cached, res) = hasher
            .get_cached_or_compile(
                &service,
                None,
                creator,
                arguments,
                cwd.to_path_buf(),
                vec![],
                CacheControl::ForceRecache,
                pool,
                pending.increment(),
            )
            .wait()
            .unwrap();
        // Ensure that the dependency file was created.
        assert!(fs::metadata(&dep).map(|m| m.len() > 0).unwrap());
        // Ensure that the object file was created.
        assert!(fs::metadata(&obj).map(|m| m.len() > 0).unwrap());
        match cached {
            CompileResult::CacheMiss(MissType::ForcedRecache, DistType::NoDist, _, f) => {
                // wait on cache write future so we don't race with it!
                f.wait().unwrap();
            }
            _ => panic!("Unexpected compile result: {cached:?}"),
        }
        assert_eq!(0, res.code().unwrap());
        assert_eq!(COMPILER_STDOUT, res.stdout.as_slice());
        assert_eq!(COMPILER_STDERR, res.stderr.as_slice());
    }

    #[test]
    fn test_compiler_get_cached_or_compile_preprocessor_error() {
        drop(env_logger::try_init());
        let pending = crate::server::SccacheGauge::default();
        let creator = new_creator();
        let f = TestFixture::new();
        let gcc = f.mk_bin("gcc").unwrap();
        let runtime = single_threaded_runtime();
        let pool = runtime.handle().clone();
        let storage = Arc::new(DiskCache::new(
            f.tempdir.path().join("cache"),
            u64::MAX,
            CacheMode::ReadWrite,
            vec![],
        ));
        let service = SccacheService::mock_with_storage(storage.clone(), pool.clone());

        // Pretend to be GCC.  Also inject a fake object file that the subsequent
        // preprocessor failure should remove.
        // let obj = f.tempdir.path().join("foo.o");
        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();
        // let o = obj.clone();
        next_command_calls(&creator, move |_| {
            // let mut f = File::create(&o)?;
            // f.write_all(b"file contents")?;
            Ok(MockChild::new(exit_status(0), "compiler_id=gcc", ""))
        });
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c = get_compiler_info(
            creator.clone(),
            &gcc,
            f.tempdir.path(),
            &[],
            &[],
            &pool,
            None,
        )
        .wait()
        .unwrap()
        .0;
        // // We should now have a fake object file.
        // assert!(fs::metadata(&obj).is_ok());
        // The preprocessor invocation.
        const PREPROCESSOR_STDERR: &[u8] = b"something went wrong";
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(1), b"", PREPROCESSOR_STDERR)),
        );
        let cwd = f.tempdir.path();
        let arguments = ovec!["-c", "foo.c", "-o", "foo.o"];
        let hasher = match c.parse_arguments(&arguments, cwd, &[], &[]) {
            CompilerArguments::Ok(h) => h,
            o => panic!("Bad result from parse_arguments: {o:?}"),
        };
        let (cached, res) = crate::util::block_on(&runtime, async {
            hasher
                .get_cached_or_compile(
                    &service,
                    None,
                    creator,
                    arguments,
                    cwd.to_path_buf(),
                    vec![],
                    CacheControl::Default,
                    pool,
                    pending.increment(),
                )
                .await
        })
        .unwrap();
        assert_eq!(cached, CompileResult::Error(Default::default()));
        assert_eq!(1, res.code().or(res.signal()).unwrap());
        // Shouldn't get anything on stdout, since that would just be preprocessor spew!
        assert_eq!("", str::from_utf8(&res.stdout).unwrap());
        assert_eq!(
            str::from_utf8(PREPROCESSOR_STDERR),
            str::from_utf8(&res.stderr)
        );
        // // Errors in preprocessing should remove the object file.
        // assert!(fs::metadata(&obj).is_err());
    }

    #[test]
    #[cfg(feature = "dist-client")]
    fn test_compiler_get_cached_or_compile_dist_error() {
        drop(env_logger::try_init());
        let pending = crate::server::SccacheGauge::default();
        let creator = new_creator();
        let f = TestFixture::new();
        let gcc = f.mk_bin("gcc").unwrap();
        let runtime = Runtime::new().unwrap();
        let pool = runtime.handle().clone();
        let dist_clients = vec![
            test_dist::ErrorPutToolchainClient::new(),
            test_dist::ErrorAllocJobClient::new(),
            test_dist::ErrorSubmitToolchainClient::new(),
            test_dist::ErrorRunJobClient::new(),
        ];
        // Write a dummy input file so the dependencies cache can work
        std::fs::write(f.tempdir.path().join("foo.c"), "whatever").unwrap();
        let storage = Arc::new(DiskCache::new(
            f.tempdir.path().join("cache"),
            u64::MAX,
            CacheMode::ReadWrite,
            vec![],
        ));
        // Pretend to be GCC.
        next_command(
            &creator,
            Ok(MockChild::new(exit_status(0), "compiler_id=gcc", "")),
        );
        next_assembler(&creator, "GNU assembler (GNU Binutils) 2.42", "");
        // Try to read gcc implicit specfiles
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        // Try to read -march=native
        next_command(
            &creator,
            Ok(MockChild::new(
                exit_status(0),
                [
                    "  -mandroid                   		[disabled]",
                    "  -march=                     		znver2",
                    "  -masm=                      		att",
                    "  -mtune-ctrl=                		",
                    "  -mtune=                     		znver2",
                    "  -muclibc                    		[disabled]",
                ]
                .join("\n"),
                "",
            )),
        );
        let c = get_compiler_info(
            creator.clone(),
            &gcc,
            f.tempdir.path(),
            &[],
            &[],
            &pool,
            None,
        )
        .wait()
        .unwrap()
        .0;
        const COMPILER_STDOUT: &[u8] = b"compiler stdout";
        const COMPILER_STDERR: &[u8] = b"compiler stderr";
        // The compiler should be invoked twice, since we're forcing
        // recaching.
        let obj = f.tempdir.path().join("foo.o");
        let dep = f.tempdir.path().join("foo.d");
        for _ in dist_clients.iter() {
            // The generate_hash_key preprocessor invocation.
            let d = dep.clone();
            next_command_calls(&creator, move |_| {
                // Pretend to preprocess something.
                let mut f = File::create(&d)?;
                f.write_all(b"foo.o :")?;
                Ok(MockChild::new(exit_status(0), "preprocessor", ""))
            });
            // The inputs_packager preprocessor invocation.
            next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
            // The compiler invocation.
            let o = obj.clone();
            next_command_calls(&creator, move |_| {
                // Pretend to compile something.
                let mut f = File::create(&o)?;
                f.write_all(b"file contents")?;
                Ok(MockChild::new(
                    exit_status(0),
                    COMPILER_STDOUT,
                    COMPILER_STDERR,
                ))
            });
        }
        let cwd = f.tempdir.path();
        let arguments = ovec!["-c", "foo.c", "-o", "foo.o", "-MD"];
        let hasher = match c.parse_arguments(&arguments, cwd, &[], &[]) {
            CompilerArguments::Ok(h) => h,
            o => panic!("Bad result from parse_arguments: {o:?}"),
        };
        // All these dist clients will fail, but should still result in successful compiles
        for dist_client in dist_clients {
            let service = server::SccacheService::mock_with_dist_client(
                dist_client.clone(),
                storage.clone(),
                pool.clone(),
            );

            if obj.is_file() {
                fs::remove_file(&obj).unwrap();
            }
            if dep.is_file() {
                fs::remove_file(&dep).unwrap();
            }
            let hasher = hasher.clone();
            let (cached, res) = hasher
                .get_cached_or_compile(
                    &service,
                    Some(dist_client.clone()),
                    creator.clone(),
                    arguments.clone(),
                    cwd.to_path_buf(),
                    vec![],
                    CacheControl::ForceRecache,
                    pool.clone(),
                    pending.increment(),
                )
                .wait()
                .expect("Does not error if storage put fails. qed");
            // Ensure that the dependency file was created.
            assert!(fs::metadata(&dep).map(|m| m.len() > 0).unwrap());
            // Ensure that the object file was created.
            assert!(fs::metadata(&obj).map(|m| m.len() > 0).unwrap());
            match cached {
                CompileResult::CacheMiss(MissType::ForcedRecache, DistType::Error, _, f) => {
                    // wait on cache write future so we don't race with it!
                    f.wait().unwrap();
                }
                _ => panic!("Unexpected compile result: {cached:?}"),
            }
            assert_eq!(0, res.code().unwrap());
            assert_eq!(
                String::from_utf8_lossy(COMPILER_STDOUT),
                String::from_utf8_lossy(&res.stdout)
            );
            assert_eq!(
                String::from_utf8_lossy(COMPILER_STDERR),
                String::from_utf8_lossy(&res.stderr)
            );
        }
    }
}

#[cfg(test)]
#[cfg(feature = "dist-client")]
mod test_dist {
    use crate::dist::{
        self, CompileCommand, JobInput, JobInputs, OutputData, SchedulerStatus,
        SubmitToolchainResult, Toolchain,
        v3::{NewJobResponse, RunJobResponse},
    };
    use crate::dist::{
        BuildResult, PathTransformer,
        pkg::{ArchivePackager, ToolchainPackager},
    };
    use crate::mock_command::ProcessOutput;
    use async_trait::async_trait;
    use std::{
        collections::{BTreeMap, HashMap},
        io::Write,
        path::{Path, PathBuf},
        sync::{Arc, atomic::AtomicBool},
        time::Duration,
    };

    use crate::errors::*;

    struct PanicArchivePackager;

    #[async_trait]
    impl ArchivePackager for PanicArchivePackager {
        async fn compute_hash(&self) -> Result<String> {
            panic!("ArchivePackager::compute_hash should not have called packager")
        }
        fn compress(&self, _: &mut (dyn Write + Send)) -> Result<()> {
            panic!("ArchivePackager::package should not have called compress")
        }
        fn write_tar_gz(&self, _: &mut (dyn Write + Send)) -> Result<()> {
            panic!("ArchivePackager::write_compressed_tar should not have called packager")
        }
    }

    pub struct ErrorPutToolchainClient;
    impl ErrorPutToolchainClient {
        #[allow(clippy::new_ret_no_self)]
        pub fn new() -> Arc<dyn dist::Client> {
            Arc::new(ErrorPutToolchainClient)
        }
    }
    #[async_trait]
    impl dist::Client for ErrorPutToolchainClient {
        async fn get_job_inputs(
            &self,
            contents: dist::ArchiveContents,
            inputs: BTreeMap<String, JobInput>,
        ) -> Result<(dist::JobInputs, BTreeMap<String, JobInput>)> {
            dist::http::Client::get_job_inputs(Default::default(), contents, inputs).await
        }
        async fn new_job(&self, _: Toolchain, _: Vec<String>) -> Result<NewJobResponse> {
            unreachable!()
        }
        async fn put_job(&self, _: HashMap<String, JobInput>) -> Result<()> {
            unreachable!()
        }
        async fn del_job(&self, _: &str) -> Result<()> {
            Ok(())
        }
        async fn get_status(&self) -> Result<SchedulerStatus> {
            unreachable!()
        }
        async fn put_toolchain(
            &self,
            _: Toolchain,
            _: Option<Arc<dyn ArchivePackager>>,
        ) -> Result<SubmitToolchainResult> {
            unreachable!()
        }
        async fn run_job(
            &self,
            _: &str,
            _: Duration,
            _: Toolchain,
            _: CompileCommand,
            _: JobInputs,
            _: Vec<String>,
            _: bool,
        ) -> Result<RunJobResponse> {
            unreachable!()
        }
        async fn hash_toolchain(
            &self,
            _: &Path,
            _: &str,
            _: Box<dyn ToolchainPackager>,
            _: &mut PathTransformer,
        ) -> Result<(
            Toolchain,
            Option<(String, PathBuf)>,
            Option<Arc<dyn ArchivePackager>>,
        )> {
            Err(anyhow!("MOCK: hash_toolchain failure"))
        }
        fn fallback_to_local_compile(&self) -> bool {
            true
        }
        fn max_retries(&self) -> f64 {
            0f64
        }
        fn request_timeout(&self) -> u32 {
            300
        }
        async fn get_custom_toolchain(&self, _exe: &Path) -> Option<PathBuf> {
            None
        }
    }

    pub struct ErrorAllocJobClient {
        tc: Toolchain,
    }
    impl ErrorAllocJobClient {
        #[allow(clippy::new_ret_no_self)]
        pub fn new() -> Arc<dyn dist::Client> {
            Arc::new(Self {
                tc: Toolchain {
                    archive_id: "somearchiveid".to_owned(),
                },
            })
        }
    }
    #[async_trait]
    impl dist::Client for ErrorAllocJobClient {
        async fn get_job_inputs(
            &self,
            contents: dist::ArchiveContents,
            inputs: BTreeMap<String, JobInput>,
        ) -> Result<(dist::JobInputs, BTreeMap<String, JobInput>)> {
            dist::http::Client::get_job_inputs(Default::default(), contents, inputs).await
        }
        async fn new_job(&self, tc: Toolchain, _: Vec<String>) -> Result<NewJobResponse> {
            assert_eq!(self.tc, tc);
            Err(anyhow!("MOCK: alloc job failure"))
        }
        async fn put_job(&self, _: HashMap<String, JobInput>) -> Result<()> {
            unreachable!()
        }
        async fn del_job(&self, _: &str) -> Result<()> {
            Ok(())
        }
        async fn get_status(&self) -> Result<SchedulerStatus> {
            unreachable!()
        }
        async fn put_toolchain(
            &self,
            _: Toolchain,
            _: Option<Arc<dyn ArchivePackager>>,
        ) -> Result<SubmitToolchainResult> {
            unreachable!()
        }
        async fn run_job(
            &self,
            _: &str,
            _: Duration,
            _: Toolchain,
            _: CompileCommand,
            _: JobInputs,
            _: Vec<String>,
            _: bool,
        ) -> Result<RunJobResponse> {
            unreachable!()
        }
        async fn hash_toolchain(
            &self,
            _: &Path,
            _: &str,
            _: Box<dyn ToolchainPackager>,
            _: &mut PathTransformer,
        ) -> Result<(
            Toolchain,
            Option<(String, PathBuf)>,
            Option<Arc<dyn ArchivePackager>>,
        )> {
            Ok((self.tc.clone(), None, Some(Arc::new(PanicArchivePackager))))
        }
        fn fallback_to_local_compile(&self) -> bool {
            true
        }
        fn max_retries(&self) -> f64 {
            0f64
        }
        fn request_timeout(&self) -> u32 {
            300
        }
        async fn get_custom_toolchain(&self, _exe: &Path) -> Option<PathBuf> {
            None
        }
    }

    pub struct ErrorSubmitToolchainClient {
        has_started: AtomicBool,
        tc: Toolchain,
    }
    impl ErrorSubmitToolchainClient {
        #[allow(clippy::new_ret_no_self)]
        pub fn new() -> Arc<dyn dist::Client> {
            Arc::new(Self {
                has_started: AtomicBool::default(),
                tc: Toolchain {
                    archive_id: "somearchiveid".to_owned(),
                },
            })
        }
    }

    #[async_trait]
    impl dist::Client for ErrorSubmitToolchainClient {
        async fn get_job_inputs(
            &self,
            contents: dist::ArchiveContents,
            inputs: BTreeMap<String, JobInput>,
        ) -> Result<(dist::JobInputs, BTreeMap<String, JobInput>)> {
            dist::http::Client::get_job_inputs(Default::default(), contents, inputs).await
        }
        async fn new_job(&self, tc: Toolchain, _: Vec<String>) -> Result<NewJobResponse> {
            assert!(
                !self
                    .has_started
                    .swap(true, std::sync::atomic::Ordering::SeqCst)
            );
            assert_eq!(self.tc, tc);

            Ok(NewJobResponse {
                has_toolchain: false,
                missing_inputs: vec![],
                job_id: "job_id".into(),
                timeout: 10,
            })
        }
        async fn put_job(&self, _: HashMap<String, JobInput>) -> Result<()> {
            unreachable!()
        }
        async fn del_job(&self, _: &str) -> Result<()> {
            Ok(())
        }
        async fn get_status(&self) -> Result<SchedulerStatus> {
            unreachable!("fn do_get_status is not used for this test. qed")
        }
        async fn put_toolchain(
            &self,
            tc: Toolchain,
            _: Option<Arc<dyn ArchivePackager>>,
        ) -> Result<SubmitToolchainResult> {
            assert_eq!(self.tc, tc);
            Err(anyhow!("MOCK: submit toolchain failure"))
        }
        async fn run_job(
            &self,
            _: &str,
            _: Duration,
            _: Toolchain,
            _: CompileCommand,
            _: JobInputs,
            _: Vec<String>,
            _: bool,
        ) -> Result<RunJobResponse> {
            unreachable!("fn run_job is not used for this test. qed")
        }
        async fn hash_toolchain(
            &self,
            _: &Path,
            _: &str,
            _: Box<dyn ToolchainPackager>,
            _: &mut PathTransformer,
        ) -> Result<(
            Toolchain,
            Option<(String, PathBuf)>,
            Option<Arc<dyn ArchivePackager>>,
        )> {
            Ok((self.tc.clone(), None, Some(Arc::new(PanicArchivePackager))))
        }
        fn fallback_to_local_compile(&self) -> bool {
            true
        }
        fn max_retries(&self) -> f64 {
            0f64
        }
        fn request_timeout(&self) -> u32 {
            300
        }
        async fn get_custom_toolchain(&self, _exe: &Path) -> Option<PathBuf> {
            None
        }
    }

    pub struct ErrorRunJobClient {
        has_started: AtomicBool,
        tc: Toolchain,
    }
    impl ErrorRunJobClient {
        #[allow(clippy::new_ret_no_self)]
        pub fn new() -> Arc<dyn dist::Client> {
            Arc::new(Self {
                has_started: AtomicBool::default(),
                tc: Toolchain {
                    archive_id: "somearchiveid".to_owned(),
                },
            })
        }
    }

    #[async_trait]
    impl dist::Client for ErrorRunJobClient {
        async fn get_job_inputs(
            &self,
            contents: dist::ArchiveContents,
            inputs: BTreeMap<String, JobInput>,
        ) -> Result<(dist::JobInputs, BTreeMap<String, JobInput>)> {
            dist::http::Client::get_job_inputs(Default::default(), contents, inputs).await
        }
        async fn new_job(&self, tc: Toolchain, _: Vec<String>) -> Result<NewJobResponse> {
            assert!(
                !self
                    .has_started
                    .swap(true, std::sync::atomic::Ordering::SeqCst)
            );
            assert_eq!(self.tc, tc);

            Ok(NewJobResponse {
                has_toolchain: false,
                missing_inputs: vec![],
                job_id: "job_id".into(),
                timeout: 10,
            })
        }
        async fn put_job(&self, _: HashMap<String, JobInput>) -> Result<()> {
            unreachable!()
        }
        async fn del_job(&self, _: &str) -> Result<()> {
            Ok(())
        }
        async fn get_status(&self) -> Result<SchedulerStatus> {
            unreachable!()
        }
        async fn put_toolchain(
            &self,
            tc: Toolchain,
            _: Option<Arc<dyn ArchivePackager>>,
        ) -> Result<SubmitToolchainResult> {
            assert_eq!(self.tc, tc);
            Ok(SubmitToolchainResult::Success)
        }
        async fn run_job(
            &self,
            job_id: &str,
            timeout: Duration,
            tc: Toolchain,
            command: CompileCommand,
            _: JobInputs,
            _: Vec<String>,
            _: bool,
        ) -> Result<RunJobResponse> {
            assert_eq!(job_id, "job_id");
            assert_eq!(timeout, Duration::from_secs(10));
            assert_eq!(self.tc, tc);
            assert_eq!(command.executable, "/overridden/compiler");
            Err(anyhow!("MOCK: run job failure"))
        }
        async fn hash_toolchain(
            &self,
            _: &Path,
            _: &str,
            _: Box<dyn ToolchainPackager>,
            _: &mut PathTransformer,
        ) -> Result<(
            Toolchain,
            Option<(String, PathBuf)>,
            Option<Arc<dyn ArchivePackager>>,
        )> {
            Ok((
                self.tc.clone(),
                Some((
                    "/overridden/compiler".to_owned(),
                    PathBuf::from("somearchiveid"),
                )),
                Some(Arc::new(PanicArchivePackager)),
            ))
        }
        fn fallback_to_local_compile(&self) -> bool {
            true
        }
        fn max_retries(&self) -> f64 {
            0f64
        }
        fn request_timeout(&self) -> u32 {
            300
        }
        async fn get_custom_toolchain(&self, _exe: &Path) -> Option<PathBuf> {
            None
        }
    }

    pub struct OneshotClient {
        has_started: AtomicBool,
        tc: Toolchain,
        output: ProcessOutput,
        has_toolchain: bool,
    }

    impl OneshotClient {
        #[allow(clippy::new_ret_no_self)]
        pub fn new(
            code: i32,
            stdout: Vec<u8>,
            stderr: Vec<u8>,
            has_toolchain: bool,
        ) -> Arc<dyn dist::Client> {
            Arc::new(Self {
                has_started: AtomicBool::default(),
                has_toolchain,
                tc: Toolchain {
                    archive_id: "somearchiveid".to_owned(),
                },
                output: ProcessOutput::new(code as i64, stdout, stderr),
            })
        }
    }

    #[async_trait]
    impl dist::Client for OneshotClient {
        async fn get_job_inputs(
            &self,
            contents: dist::ArchiveContents,
            inputs: BTreeMap<String, JobInput>,
        ) -> Result<(dist::JobInputs, BTreeMap<String, JobInput>)> {
            dist::http::Client::get_job_inputs(Default::default(), contents, inputs).await
        }
        async fn new_job(&self, tc: Toolchain, _: Vec<String>) -> Result<NewJobResponse> {
            assert!(
                !self
                    .has_started
                    .swap(true, std::sync::atomic::Ordering::SeqCst)
            );
            assert_eq!(self.tc, tc);
            Ok(NewJobResponse {
                missing_inputs: vec![],
                has_toolchain: self.has_toolchain,
                job_id: "job_id".into(),
                timeout: 10,
            })
        }
        async fn put_job(&self, _: HashMap<String, JobInput>) -> Result<()> {
            unreachable!()
        }
        async fn del_job(&self, _: &str) -> Result<()> {
            Ok(())
        }
        async fn get_status(&self) -> Result<SchedulerStatus> {
            unreachable!("fn do_get_status is not used for this test. qed")
        }
        async fn put_toolchain(
            &self,
            tc: Toolchain,
            _: Option<Arc<dyn ArchivePackager>>,
        ) -> Result<SubmitToolchainResult> {
            if self.has_toolchain {
                panic!("DistClient::put_toolchain should not have been called!")
            }
            assert_eq!(self.tc, tc);

            Ok(SubmitToolchainResult::Success)
        }
        async fn run_job(
            &self,
            job_id: &str,
            timeout: Duration,
            tc: Toolchain,
            command: CompileCommand,
            _: JobInputs,
            outputs: Vec<String>,
            _: bool,
        ) -> Result<RunJobResponse> {
            assert_eq!(job_id, "job_id");
            assert_eq!(timeout, Duration::from_secs(10));
            assert_eq!(self.tc, tc);
            assert_eq!(command.executable, "/overridden/compiler");

            let outputs = outputs
                .into_iter()
                .map(|name| {
                    let data = format!("some data in {name}");
                    let data = OutputData::try_from_reader(data.as_bytes()).unwrap();
                    (name, data)
                })
                .collect();
            let result = RunJobResponse::Complete {
                server_id: "server_id".into(),
                result: BuildResult {
                    output: self.output.clone(),
                    outputs,
                },
            };
            Ok(result)
        }
        async fn hash_toolchain(
            &self,
            _: &Path,
            _: &str,
            _: Box<dyn ToolchainPackager>,
            _: &mut PathTransformer,
        ) -> Result<(
            Toolchain,
            Option<(String, PathBuf)>,
            Option<Arc<dyn ArchivePackager>>,
        )> {
            Ok((
                self.tc.clone(),
                Some((
                    "/overridden/compiler".to_owned(),
                    PathBuf::from("somearchiveid"),
                )),
                Some(Arc::new(PanicArchivePackager)),
            ))
        }
        fn fallback_to_local_compile(&self) -> bool {
            true
        }
        fn max_retries(&self) -> f64 {
            0f64
        }
        fn request_timeout(&self) -> u32 {
            300
        }
        async fn get_custom_toolchain(&self, _exe: &Path) -> Option<PathBuf> {
            None
        }
    }
}
