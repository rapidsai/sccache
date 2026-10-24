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

use crate::{
    cache::{Cache, FileObjectSource, Storage},
    compiler::{
        CacheControl, Cacheable, ColorMode, Compilation, CompileCommand, CompileCommandImpl,
        Compiler, CompilerArguments, CompilerHasher, CompilerKind, GenerateHashKeyArgs, HashResult,
        Language,
        util::{
            DEPS_VERSION, Dependencies, HiddenIncludes, SourceFileCache, SourceFileDeps,
            SourceFileEntry,
        },
    },
    dist,
    errors::*,
    mock_command::{CommandCreatorSync, ProcessOutput},
    server::SccacheService,
    util::{Digest, HashToDigest, apply_basedirs_to_args, fmt_duration_as_secs},
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    ffi::{OsStr, OsString},
    fmt,
    hash::Hash,
    io::Write,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};
use tokio::io::AsyncWriteExt;

#[cfg(feature = "dist-client")]
use crate::{
    compiler::{DistPackagers, NoopOutputsRewriter},
    dist::pkg,
};

/// A generic implementation of the `Compiler` trait for C/C++ compilers.
#[derive(Clone)]
pub struct CCompiler<I>
where
    I: CCompilerImpl,
{
    executable: PathBuf,
    #[cfg(test)]
    pub executable_digest: String,
    #[cfg(not(test))]
    executable_digest: String,
    extra_hashes: Vec<String>,
    compiler: I,
}

/// A generic implementation of the `CompilerHasher` trait for C/C++ compilers.
#[derive(Debug, Clone)]
pub struct CCompilerHasher<I>
where
    I: CCompilerImpl,
{
    parsed_args: ParsedArguments,
    executable: PathBuf,
    executable_digest: String,
    compiler: I,
    extra_hashes: Vec<String>,
}

/// Artifact produced by a C/C++ compiler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactDescriptor {
    /// Path to the artifact.
    pub path: PathBuf,
    /// Whether the artifact is an optional object file.
    pub optional: bool,
}

#[derive(Debug)]
pub enum OutDir {
    Dir(PathBuf),
    Tmp(tempfile::TempDir),
}

impl OutDir {
    pub fn path(&self) -> &Path {
        match self {
            Self::Dir(dir) => dir.as_path(),
            Self::Tmp(dir) => dir.path(),
        }
    }
}

impl Eq for OutDir {}

impl PartialEq for OutDir {
    fn eq(&self, other: &Self) -> bool {
        self.path() == other.path()
    }
}

impl AsRef<Path> for OutDir {
    fn as_ref(&self) -> &Path {
        self.path()
    }
}

pub struct ParseArgs<'a> {
    pub arguments: &'a [OsString],
    pub cwd: &'a Path,
    pub env_vars: &'a [(OsString, OsString)],
}

#[derive(Clone, Copy)]
pub struct GenerateDependenciesArgs<'a, T: CommandCreatorSync> {
    pub creator: &'a T,
    pub executable: &'a Path,
    pub parsed_args: &'a ParsedArguments,
    pub cwd: &'a Path,
    pub env_vars: &'a [(OsString, OsString)],
    pub tolerate_failure: bool,
}

pub struct GenerateCompileCommandsArgs<'a> {
    pub executable: &'a Path,
    pub parsed_args: &'a ParsedArguments,
    pub cwd: &'a Path,
    pub env_vars: &'a [(OsString, OsString)],
}

/// The results of parsing a compiler commandline.
#[allow(dead_code)]
#[derive(Default, Debug, Clone)]
pub struct ParsedArguments {
    /// The input source file.
    pub input: PathBuf,
    /// The input source file absolute path.
    pub input_absolute: PathBuf,
    /// Whether to prepend the input with `--`
    pub double_dash_input: bool,
    /// The type of language used in the input source file.
    pub language: Language,
    /// The flag required to compile for the given language
    pub compilation_flag: OsString,
    /// Whether C++20 modules are enabled
    pub cxx20_modules: bool,
    /// The file in which to generate dependencies.
    pub depfile: Option<PathBuf>,
    /// The file in which to generate module dependencies.
    pub structured_depfile: Option<PathBuf>,
    /// The format of the structured dependencies file.
    pub structured_depfile_format: Option<OsString>,
    /// Output files and whether it's optional, keyed by a simple name, like "obj".
    pub outputs: HashMap<&'static str, ArtifactDescriptor>,
    /// Commandline arguments for dependency generation.
    pub dependency_args: Vec<OsString>,
    /// Commandline arguments for the preprocessor (not including common_args).
    pub preprocessor_args: Vec<OsString>,
    /// Commandline arguments for the preprocessor or the compiler.
    pub common_args: Vec<OsString>,
    /// Commandline arguments for the compiler that specify the architecture given
    pub arch_args: Vec<OsString>,
    /// Commandline arguments for the preprocessor or the compiler that don't affect the computed hash.
    pub unhashed_args: Vec<OsString>,
    /// Extra unhashed files that need to be sent along with dist compiles.
    pub extra_dist_files: Vec<PathBuf>,
    /// Extra files that need to have their contents hashed.
    pub extra_hash_files: Vec<PathBuf>,
    /// Whether the compiler runs a separate assembler to produce the object file.
    pub uses_external_assembler: bool,
    /// The -scanDependencies depfile in MSVC
    pub msvc_scan_depfile: Option<PathBuf>,
    /// The -sourceDependencies:directives depfile in MSVC
    pub msvc_directives_depfile: Option<PathBuf>,
    /// Whether or not the `-showIncludes` argument is passed on MSVC
    pub msvc_show_includes: bool,
    /// Whether the compilation is generating profiling or coverage data.
    pub profile_generate: bool,
    /// The color mode.
    pub color_mode: ColorMode,
    /// Path to the scratch directory to use for this compilation (nvcc)
    pub out_dir: Option<Arc<OutDir>>,
}

impl std::cmp::Eq for ParsedArguments {}

impl std::cmp::PartialEq for ParsedArguments {
    fn eq(&self, other: &Self) -> bool {
        let ParsedArguments {
            input,
            input_absolute,
            double_dash_input,
            language,
            compilation_flag,
            cxx20_modules,
            depfile,
            structured_depfile,
            structured_depfile_format,
            outputs,
            dependency_args,
            preprocessor_args,
            common_args,
            arch_args,
            unhashed_args,
            extra_dist_files,
            extra_hash_files,
            uses_external_assembler,
            msvc_scan_depfile,
            msvc_directives_depfile,
            msvc_show_includes,
            profile_generate,
            color_mode,
            out_dir,
        } = self;
        input == &other.input
            && input_absolute == &other.input_absolute
            && double_dash_input == &other.double_dash_input
            && language == &other.language
            && compilation_flag == &other.compilation_flag
            && cxx20_modules == &other.cxx20_modules
            && depfile == &other.depfile
            && structured_depfile == &other.structured_depfile
            && structured_depfile_format == &other.structured_depfile_format
            && outputs == &other.outputs
            && dependency_args == &other.dependency_args
            && preprocessor_args == &other.preprocessor_args
            && common_args == &other.common_args
            && arch_args == &other.arch_args
            && unhashed_args == &other.unhashed_args
            && extra_dist_files == &other.extra_dist_files
            && extra_hash_files == &other.extra_hash_files
            && uses_external_assembler == &other.uses_external_assembler
            && msvc_scan_depfile == &other.msvc_scan_depfile
            && msvc_directives_depfile == &other.msvc_directives_depfile
            && msvc_show_includes == &other.msvc_show_includes
            && profile_generate == &other.profile_generate
            && color_mode == &other.color_mode
            && out_dir == &other.out_dir
    }
}

impl ParsedArguments {
    pub fn output_pretty(&self) -> Cow<'_, str> {
        self.outputs
            .get("obj")
            .map(|o| o.path.as_os_str())
            .and_then(|s| s.to_str().map(Cow::Borrowed))
            .unwrap_or(Cow::Borrowed("Unknown filename"))
    }
}

/// A generic implementation of the `Compilation` trait for C/C++ compilers.
#[derive(Clone, Debug)]
struct CCompilation<I: CCompilerImpl, T: CommandCreatorSync> {
    parsed_args: ParsedArguments,
    is_locally_preprocessed: bool,
    executable: PathBuf,
    compiler: I,
    cwd: PathBuf,
    obj_key: String,
    env_vars: Vec<(OsString, OsString)>,
    dist_files: Vec<PathBuf>,
    deps_args: Option<GetDependenciesArgs<I, T>>,
}

/// Supported C compilers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CCompilerKind {
    /// GCC
    Gcc,
    /// clang
    Clang,
    /// Diab
    Diab,
    /// Microsoft Visual C++
    Msvc,
    /// NVIDIA CUDA compiler
    Nvcc,
    /// NVIDIA CUDA front-end
    CudaFE,
    /// NVIDIA CUDA optimizer and PTX generator
    Cicc,
    /// NVIDIA CUDA PTX assembler
    Ptxas,
    /// NVIDIA hpc c, c++ compiler
    Nvhpc,
    /// Tasking VX
    TaskingVX,
    /// NVIDIA CUDA Tile IR assembler
    Tileiras,
    // LLVM system compiler
    Llc,
}

impl From<&str> for CCompilerKind {
    fn from(kind: &str) -> Self {
        match kind {
            "clang" | "clang++" | "clang-cl" => CCompilerKind::Clang,
            "diab" => CCompilerKind::Diab,
            "gcc" | "g++" => CCompilerKind::Gcc,
            "cl" => CCompilerKind::Msvc,
            "nvcc" => CCompilerKind::Nvcc,
            "cudafe++" => CCompilerKind::CudaFE,
            "cicc" => CCompilerKind::Cicc,
            "ptxas" => CCompilerKind::Ptxas,
            "nvc" | "nvc++" => CCompilerKind::Nvhpc,
            "taskingvx" => CCompilerKind::TaskingVX,
            "tileiras" => CCompilerKind::Tileiras,
            "llc" => CCompilerKind::Llc,
            _ => unreachable!(),
        }
    }
}

impl From<CCompilerKind> for CompilerKind {
    fn from(kind: CCompilerKind) -> Self {
        CompilerKind::C(kind)
    }
}

impl From<&CCompilerKind> for CompilerKind {
    fn from(kind: &CCompilerKind) -> Self {
        CompilerKind::C(*kind)
    }
}

/// An interface to a specific C compiler.
#[async_trait]
pub trait CCompilerImpl: Clone + fmt::Debug + Send + Sync + 'static {
    /// Return the kind of compiler.
    fn kind(&self) -> CCompilerKind;
    /// Return true iff this is g++ or clang++.
    fn plusplus(&self) -> bool;
    /// Return the compiler version reported by the compiler executable.
    fn version(&self) -> Option<String>;
    /// Return paths to extra files that should be included in the dist toolchain.
    fn extra_dist_files(&self) -> impl Iterator<Item = PathBuf> {
        std::iter::empty()
    }
    fn assembler_digest(&self) -> Option<&str> {
        None
    }
    /// Determine whether `arguments` are supported by this compiler.
    fn parse_arguments(&self, args: ParseArgs<'_>) -> CompilerArguments<ParsedArguments>;

    /// Run the C preprocessor to generate the dependencies file.
    async fn generate_dependencies<T>(
        &self,
        args: GenerateDependenciesArgs<'_, T>,
    ) -> Result<(Vec<PathBuf>, Duration)>
    where
        T: CommandCreatorSync;

    /// Generate a command that can be used to invoke the C compiler to perform
    /// the compilation.
    fn generate_compile_commands(
        &self,
        args: GenerateCompileCommandsArgs<'_>,
    ) -> Result<(
        impl CompileCommandImpl,
        Option<dist::CompileCommand>,
        Cacheable,
    )>;
}

impl<I> CCompiler<I>
where
    I: CCompilerImpl,
{
    pub async fn new(
        compiler: I,
        executable: PathBuf,
        extra_hash_files: Vec<PathBuf>,
    ) -> Result<CCompiler<I>> {
        trace!(
            "[CCompiler::new]: compiler={compiler:?}, executable={executable:?}, extra_hash_files={extra_hash_files:?}"
        );

        let paths = std::iter::once(executable.as_path())
            .chain(extra_hash_files.iter().map(|p| p.as_path()))
            .filter(|path| path.exists())
            .map(Ok);

        let executable_digest = futures::stream::iter(paths)
            .try_fold(Digest::new(), |digest, path| digest.with_file(path))
            .await
            .map(|mut digest| {
                if let Some(version) = compiler.version() {
                    digest.update(version.as_bytes());
                }
                digest.finish()
            })?;

        Ok(CCompiler {
            compiler,
            executable,
            executable_digest,
            extra_hashes: vec![],
        })
    }

    fn extract_rocm_arg(args: &ParsedArguments, flag: &str) -> Option<PathBuf> {
        args.common_args.iter().find_map(|arg| match arg.to_str() {
            Some(sarg) if sarg.starts_with(flag) => {
                Some(PathBuf::from(sarg[arg.len()..].to_string()))
            }
            _ => None,
        })
    }

    fn extract_rocm_env(env_vars: &[(OsString, OsString)], name: &str) -> Option<PathBuf> {
        env_vars.iter().find_map(|(k, v)| match v.to_str() {
            Some(path) if k == name => Some(PathBuf::from(path.to_string())),
            _ => None,
        })
    }

    // See https://clang.llvm.org/docs/HIPSupport.html for details regarding the
    // order in which the environment variables and command-line arguments control the
    // directory to search for bitcode libraries.
    fn search_hip_device_libs(
        args: &ParsedArguments,
        env_vars: &[(OsString, OsString)],
    ) -> Vec<PathBuf> {
        let rocm_path_arg: Option<PathBuf> = Self::extract_rocm_arg(args, "--rocm-path=");
        let hip_device_lib_path_arg: Option<PathBuf> =
            Self::extract_rocm_arg(args, "--hip-device-lib-path=");
        let rocm_path_env: Option<PathBuf> = Self::extract_rocm_env(env_vars, "ROCM_PATH");
        let hip_device_lib_path_env: Option<PathBuf> =
            Self::extract_rocm_env(env_vars, "HIP_DEVICE_LIB_PATH");

        let hip_device_lib_path: PathBuf = hip_device_lib_path_arg
            .or(hip_device_lib_path_env)
            .or(rocm_path_arg.map(|path| path.join("amdgcn").join("bitcode")))
            .or(rocm_path_env.map(|path| path.join("amdgcn").join("bitcode")))
            // This is the default location in official AMD packages and containers.
            .unwrap_or(PathBuf::from("/opt/rocm/amdgcn/bitcode"));

        hip_device_lib_path
            .read_dir()
            .ok()
            .map(|f| {
                let mut device_libs = f
                    .flatten()
                    .filter(|f| f.path().extension().is_some_and(|ext| ext == "bc"))
                    .map(|f| f.path())
                    .collect::<Vec<_>>();
                device_libs.sort_unstable();
                device_libs
            })
            .unwrap_or_default()
    }

    #[allow(unused)]
    pub fn compiler(&self) -> &I {
        &self.compiler
    }

    pub fn with_extra_hashes(mut self, extra_hashes: Vec<String>) -> Self {
        self.extra_hashes.extend(extra_hashes);
        self
    }
}

impl<T: CommandCreatorSync, I: CCompilerImpl> Compiler<T> for CCompiler<I> {
    fn kind(&self) -> CompilerKind {
        CompilerKind::C(self.compiler.kind())
    }
    fn version(&self) -> Option<String> {
        self.compiler.version()
    }
    #[cfg(feature = "dist-client")]
    fn get_toolchain_packager(&self) -> Box<dyn pkg::ToolchainPackager> {
        Box::new(CToolchainPackager {
            env_vars: vec![],
            executable: self.executable.clone(),
            extra_files: self.compiler.extra_dist_files().collect(),
            kind: self.compiler.kind(),
        })
    }
    fn parse_arguments(
        &self,
        arguments: &[OsString],
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
        basedirs: &[Vec<u8>],
    ) -> CompilerArguments<Box<dyn CompilerHasher<T> + 'static>> {
        match self.compiler.parse_arguments(ParseArgs {
            arguments,
            cwd,
            env_vars,
        }) {
            CompilerArguments::Ok(mut args) => {
                // Canonicalize here so the absolute path to the input is in the
                // preprocessor output instead of paths relative to `cwd`.
                args.input_absolute = cwd.join(&args.input);
                if args.input_absolute.exists() {
                    args.input_absolute =
                        dunce::canonicalize(&args.input_absolute).unwrap_or(args.input_absolute);
                }

                // Make input path and arguments relative to basedirs
                args.input = {
                    let mut input = vec![args.input_absolute.as_os_str().into()];
                    if let Err(err) = apply_basedirs_to_args(cwd, input.as_mut_slice(), basedirs) {
                        return CompilerArguments::CannotCache(
                            "base dirs",
                            Some(format!("{err:#}")),
                        );
                    }
                    input.drain(..).next().unwrap().into()
                };

                for args in [
                    &mut args.common_args,
                    &mut args.dependency_args,
                    &mut args.preprocessor_args,
                ] {
                    if let Err(err) = apply_basedirs_to_args(cwd, args, basedirs) {
                        return CompilerArguments::CannotCache(
                            "base dirs",
                            Some(format!("{err:#}")),
                        );
                    }
                }

                // Handle SCCACHE_EXTRAFILES
                for (k, v) in env_vars.iter() {
                    if k.as_os_str() == OsStr::new("SCCACHE_EXTRAFILES") {
                        args.extra_dist_files.extend(std::env::split_paths(&v));
                        args.extra_hash_files.extend(std::env::split_paths(&v));
                    }
                }

                // Handle cache invalidation for the ROCm device bitcode libraries. Every HIP
                // object links in some LLVM bitcode libraries (.bc files), so in some sense
                // every HIP object compilation has a direct dependency on those bitcode
                // libraries.
                //
                // The bitcode libraries are unlikely to change **except** when a ROCm version
                // changes, so for correctness we should take these bitcode libraries into
                // account by adding them to `extra_hash_files`.
                //
                // In reality, not every available bitcode library is needed, but that is
                // too much to handle on our side so we just hash every bitcode library we find.
                if args.language == Language::Hip {
                    args.extra_hash_files
                        .extend(Self::search_hip_device_libs(&args, env_vars));
                }

                CompilerArguments::Ok(Box::new(CCompilerHasher {
                    parsed_args: args,
                    executable: self.executable.clone(),
                    executable_digest: self.executable_digest.clone(),
                    compiler: self.compiler.clone(),
                    extra_hashes: self.extra_hashes.clone(),
                }))
            }
            CompilerArguments::CannotCache(why, extra_info) => {
                CompilerArguments::CannotCache(why, extra_info)
            }
            CompilerArguments::NotCompilation => CompilerArguments::NotCompilation,
        }
    }

    fn box_clone(&self) -> Box<dyn Compiler<T>> {
        Box::new((*self).clone())
    }

    #[cfg(test)]
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send + Sync> {
        self
    }
}

#[derive(Clone, Debug)]
struct GetDependenciesArgs<I, T>
where
    I: CCompilerImpl,
    T: CommandCreatorSync,
{
    cache_control: CacheControl,
    compile_timestamp: DateTime<Utc>,
    compiler: I,
    creator: T,
    cwd: PathBuf,
    dist_compile: bool,
    env_vars: Vec<(OsString, OsString)>,
    executable_digest: String,
    executable: PathBuf,
    extra_hashes: Vec<String>,
    hidden_includes: HiddenIncludes,
    parsed_args: ParsedArguments,
    runtime: tokio::runtime::Handle,
    service: crate::server::SccacheService<T>,
    use_async_io: bool,
}

struct ResolvedDependencies {
    dependencies: Dependencies,
    depsfile: SourceFileDeps,
    obj_key: String,
    write_to_storage: bool,
}

#[async_trait]
impl<T, I> CompilerHasher<T> for CCompilerHasher<I>
where
    T: CommandCreatorSync,
    I: CCompilerImpl,
{
    fn get_executable(&self) -> &Path {
        self.executable.as_path()
    }

    async fn generate_hash_key(
        self: Box<Self>,
        args: GenerateHashKeyArgs<'_, T>,
    ) -> Result<(
        HashResult<T>,
        Option<Pin<Box<dyn Future<Output = Result<()>> + Send>>>,
    )> {
        let GenerateHashKeyArgs {
            creator,
            cwd,
            env_vars,
            cache_control,
            service,
            runtime,
            use_async_io,
            ..
        } = args;

        let mut hash_files = vec![];
        let out_pretty = self.parsed_args.output_pretty();

        hash_files.push(cwd.join(&self.parsed_args.input));

        // If the compiler has support for it, the expansion of __DATE__ will change
        // according to the value of SOURCE_DATE_EPOCH. If the compiler doesn't support
        // it (i.e. MSVC), this envvar shouldn't be defined.
        let compile_timestamp = env_vars
            .iter()
            .find(|(k, _)| k == "SOURCE_DATE_EPOCH")
            .and_then(|(_, v)| v.as_os_str().to_str())
            .and_then(|v| v.parse().ok())
            .and_then(DateTime::from_timestamp_secs)
            .unwrap_or_else(Utc::now);

        // Only get dependencies if the file requires preprocessing
        let (deps, deps_args, hidden_includes) =
            if self.parsed_args.language.needs_c_preprocessing() {
                let start = Instant::now();

                let deps_args = GetDependenciesArgs {
                    cache_control,
                    compile_timestamp,
                    compiler: self.compiler.clone(),
                    creator: creator.clone(),
                    cwd: cwd.to_owned(),
                    dist_compile: false,
                    env_vars: env_vars.to_owned(),
                    executable_digest: self.executable_digest.clone(),
                    executable: self.executable.clone(),
                    extra_hashes: self.extra_hashes.clone(),
                    hidden_includes: HiddenIncludes::default(),
                    parsed_args: self.parsed_args.clone(),
                    runtime: runtime.clone(),
                    service: service.clone(),
                    use_async_io,
                };

                let (deps, hidden_includes) = get_dependencies(&deps_args).await?;

                hash_files.extend(
                    deps.dependencies
                        .sources
                        .iter()
                        .map(|meta| PathBuf::from(&meta.path)),
                );

                trace!(
                    "[{out_pretty}]: get deps time: {}",
                    fmt_duration_as_secs(&start.elapsed())
                );

                (Some(deps), Some(deps_args), hidden_includes)
            } else {
                (None, None, Default::default())
            };

        let obj_key = if CacheControl::ForceNoCache == cache_control {
            // Skip hashing if we're not going to do any cache reads/writes
            // In this mode, the cache key doesn't matter, so return empty string
            String::new()
        } else if let Some(deps) = deps.as_ref().filter(|deps| !deps.obj_key.is_empty()) {
            deps.obj_key.clone()
        } else {
            let start = Instant::now();

            let obj_key = HashKeyParams::new(&self.parsed_args)
                .with_async_io(use_async_io)
                .with_runtime(runtime)
                // Hash arch, common, and preprocessor args
                .with_args(&self.parsed_args.arch_args[..])
                .with_args(&self.parsed_args.common_args[..])
                .with_args(&self.parsed_args.preprocessor_args[..])
                .with_compile_timestamp(compile_timestamp)
                .with_cwd(cwd)
                .with_env_vars(env_vars)
                .with_digests(
                    &std::iter::once(&self.executable_digest)
                        .chain(self.extra_hashes.iter())
                        .map(|s| s.as_str())
                        .chain(
                            self.compiler
                                .assembler_digest()
                                .filter(|_| self.parsed_args.uses_external_assembler),
                        )
                        .collect::<Vec<_>>(),
                )
                .with_input_files(&hash_files[..])
                .with_input_files(&self.parsed_args.extra_hash_files[..])
                // Hash the input path for CUDA compilations because nvcc's
                // .module_id is different for files in different directories, even
                // if their contents and dependencies are identical
                .with_hashed_input_path(if matches!(self.parsed_args.language, Language::Cuda) {
                    Some(self.parsed_args.input.as_path())
                } else {
                    None
                })
                .with_plusplus(self.compiler.plusplus())
                .with_version(CACHE_VERSION)
                .compute()
                .await?
                .unwrap_or_default();

            trace!(
                "[{out_pretty}]: comp hash key time: {}",
                fmt_duration_as_secs(&start.elapsed())
            );

            // Track average hash time
            {
                let duration = start.elapsed();
                let mut stats = service.stats.lock().await;
                stats.hashed += 1;
                stats.hasher_duration += duration;
            }

            obj_key
        };

        let cache_write = deps
            .filter(|deps| {
                !obj_key.is_empty() && (obj_key != deps.obj_key || deps.write_to_storage)
            })
            .map(|deps| {
                Box::pin(put_dependencies(
                    obj_key.clone(),
                    deps,
                    service.storage.clone(),
                    out_pretty.into_owned(),
                    service.cache_write_timeout,
                )) as Pin<Box<dyn Future<Output = Result<()>> + Send>>
            });

        // A compiler binary may be a symlink to another and so has the same digest, but that means
        // the toolchain will not contain the correct path to invoke the compiler! Add the compiler
        // executable path to try and prevent this
        let weak_toolchain_key = format!(
            "{}-{}",
            self.executable.to_string_lossy(),
            self.executable_digest
        );

        Ok((
            HashResult {
                key: obj_key.clone(),
                compilation: Box::new(CCompilation {
                    compiler: self.compiler,
                    cwd: cwd.to_owned(),
                    obj_key,
                    dist_files: hash_files,
                    env_vars: env_vars.to_vec(),
                    executable: self.executable,
                    is_locally_preprocessed: true,
                    parsed_args: self.parsed_args,
                    deps_args: deps_args.map(|mut deps_args| {
                        deps_args.hidden_includes = hidden_includes;
                        deps_args
                    }),
                }),
                weak_toolchain_key,
            },
            cache_write,
        ))
    }

    fn color_mode(&self) -> ColorMode {
        self.parsed_args.color_mode
    }

    fn output_pretty(&self) -> Cow<'_, str> {
        self.parsed_args.output_pretty()
    }

    fn box_clone(&self) -> Box<dyn CompilerHasher<T>> {
        Box::new((*self).clone())
    }

    fn language(&self) -> Language {
        self.parsed_args.language
    }
}

async fn get_dependencies<I, T>(
    args: &GetDependenciesArgs<I, T>,
) -> Result<(ResolvedDependencies, HiddenIncludes)>
where
    I: CCompilerImpl,
    T: CommandCreatorSync,
{
    use itertools::Itertools;

    let GetDependenciesArgs {
        compiler,
        creator,
        cwd,
        env_vars,
        executable_digest,
        executable,
        extra_hashes,
        hidden_includes,
        parsed_args,
        runtime,
        service,
        ..
    } = args;

    let GetDependenciesArgs {
        compile_timestamp,
        cache_control,
        use_async_io,
        dist_compile,
        ..
    } = *args;

    let out_pretty = parsed_args.output_pretty();

    let deps_key = if CacheControl::ForceNoCache == cache_control {
        // Skip hashing if we're not going to do any cache reads/writes
        // In this mode, the cache key doesn't matter, so return empty string
        None
    } else {
        let start = Instant::now();

        let deps_key = HashKeyParams::new(parsed_args)
            .with_runtime(runtime)
            .with_async_io(use_async_io)
            // Hash arch, common, dependency, and preprocessor args
            .with_args(&parsed_args.arch_args[..])
            .with_args(&parsed_args.common_args[..])
            .with_args(&parsed_args.dependency_args[..])
            .with_args(&parsed_args.preprocessor_args[..])
            .with_compile_timestamp(compile_timestamp)
            .with_cwd(cwd)
            .with_env_vars(env_vars)
            .with_digests(
                &std::iter::once(executable_digest)
                    .chain(extra_hashes.iter())
                    .map(|s| s.as_str())
                    .chain(
                        compiler
                            .assembler_digest()
                            .filter(|_| parsed_args.uses_external_assembler),
                    )
                    .chain([if dist_compile { "1" } else { "0" }])
                    .collect::<Vec<_>>(),
            )
            .with_input_files(std::slice::from_ref(&parsed_args.input_absolute))
            .with_input_files(&parsed_args.extra_hash_files[..])
            // Hash the input file path, otherwise:
            // - a/r.h exists.
            // - a/x.c has #include "r.h".
            // - b/x.c is identical to a/x.c.
            // - Compiling a/x.c records a/r.h in the dependencies cache entry.
            // - Compiling b/x.c results in a false cache hit since a/x.c and b/x.c
            // share dependencies cache entries and a/r.h exists.
            //
            // Do not strip basedirs from the input file path.
            //
            // If the compiler, args, and input file are identical to another
            // compilation in a different base dir, then that would compute the
            // same dependencies cache entry key as the compilation in the other
            // basedir.
            //
            // That could lead to loading the wrong object from cache if their
            // dependencies are different, which they almost certainly will be.
            //
            // Example:
            // ```shell
            // cat <<"EOF" >a/value.h
            // #define VALUE 42
            // EOF
            //
            // cat <<"EOF" >b/value.h
            // #define VALUE 43
            // EOF
            //
            // $ cat <<"EOF" | tee {a,b}/input.cpp >/dev/null
            // #include "value.h"
            // int result() { return VALUE; }
            // EOF
            //
            // $ cat <<"EOF" | tee {a,b}/main.cpp >/dev/null
            // #include <cstdio>
            // int result();
            // int main() {
            //     printf("%d\n", result());
            //     return 0;
            // }
            // EOF
            //
            // $ export SCCACHE_BASEDIRS="$(pwd)/a:$(pwd)/b"
            // $ sccache gcc -c a/input.cpp -o a/input.o
            // $ sccache gcc -c b/input.cpp -o b/input.o # < error: cache hit returns a/input.o
            // $ gcc a/input.o a/main.cpp -o a/main
            // $ gcc b/input.o b/main.cpp -o b/main
            //
            // # correct:
            // $ a/main
            // > 42
            //
            // # incorrect, should be 43:
            // $ b/main
            // > 42
            // ```
            .with_hashed_input_path(parsed_args.input_absolute.as_path())
            .with_plusplus(compiler.plusplus())
            .with_version(DEPS_VERSION)
            .compute()
            .await?;

        trace!(
            "[{out_pretty}]: deps hash key time: {}",
            fmt_duration_as_secs(&start.elapsed())
        );

        // Track average hash time
        {
            let duration = start.elapsed();
            let mut stats = service.stats.lock().await;
            stats.hashed += 1;
            stats.hasher_duration += duration;
        }

        deps_key
    };

    let force_recache_dependencies = env_vars.iter().any(|(k, v)| {
        k == "SCCACHE_RECACHE_DEPENDENCIES"
            && matches!(v.to_str().unwrap_or_default(), "true" | "on" | "1")
    });

    let maybe_depsfile = if let Some(deps_key) = deps_key.as_deref().filter(|deps_key| {
        debug!("[{out_pretty}]: Deps key: {deps_key}");
        !force_recache_dependencies
    }) {
        let start = Instant::now();
        tokio::time::timeout(
            service.cache_read_timeout,
            SourceFileDeps::get(service.storage.as_ref(), deps_key),
        )
        .await
        .map_err(|err| {
            debug!(
                "[{out_pretty}, {deps_key}]: Deps cache read timeout {:?}",
                start.elapsed()
            );
            err.into()
        })
        .and_then(|res| {
            res.inspect_err(|err| {
                debug!("[{out_pretty}, {deps_key}]: Deps cache lookup error: {err:#}");
            })
        })
        .ok()
        .and_then(|res| match res {
            Cache::Miss => None,
            Cache::Hit(depsfile) => Some(depsfile),
        })
    } else {
        None
    };

    let (maybe_depsfile, obj_key_and_deps) = if CacheControl::ForceRecache != cache_control
        && let Some(mut depsfile) = maybe_depsfile
    {
        trace!(
            "[{out_pretty}, {deps_key}]: Searching deps cache entry",
            deps_key = depsfile.key
        );

        let start = Instant::now();

        let (depsfile, needs_update, obj_key_and_deps) = if use_async_io {
            let (needs_update, obj_key_and_deps) = depsfile
                .find_dependencies(
                    &SOURCE_FILE_CACHE,
                    compile_timestamp,
                    dist_compile,
                    &out_pretty,
                )
                .await;
            (depsfile, needs_update, obj_key_and_deps)
        } else {
            let (depsfile, needs_update, obj_key_and_deps) = runtime
                .spawn_blocking({
                    let out_pretty = out_pretty.clone().into_owned();
                    move || {
                        let (needs_update, obj_key_and_deps) = depsfile.find_dependencies_sync(
                            &SOURCE_FILE_CACHE,
                            compile_timestamp,
                            dist_compile,
                            &out_pretty,
                        );
                        (depsfile, needs_update, obj_key_and_deps)
                    }
                })
                .await?;
            (depsfile, needs_update, obj_key_and_deps)
        };

        trace!(
            "[{out_pretty}]: find deps time: {}",
            fmt_duration_as_secs(&start.elapsed())
        );

        (
            Some(depsfile),
            obj_key_and_deps.map(|(obj_key, dependencies, hidden_includes)| {
                (obj_key, dependencies, hidden_includes, needs_update)
            }),
        )
    } else {
        (maybe_depsfile, None)
    };

    let depsfile = if let Some(depsfile) = maybe_depsfile {
        depsfile
    } else {
        SourceFileDeps::new(
            deps_key.unwrap_or_default(),
            SOURCE_FILE_CACHE
                .get_sync(&parsed_args.input_absolute, compile_timestamp)?
                .meta
                .clone(),
        )
    };

    if let Some((obj_key, dependencies, hidden_includes, write_to_storage)) = obj_key_and_deps {
        // Hydrate the depfiles from the dependencies cache entry

        let has_all_required_files = if dist_compile {
            true
        } else {
            let mut depfile_outputs = parsed_args
                .outputs
                .iter()
                .filter(|(name, _)| name.starts_with("dep"))
                .map(|(name, desc)| (*name, desc))
                .collect::<BTreeMap<_, _>>();

            let mut depfile_futures = vec![];

            for (name, data) in dependencies.outputs.iter() {
                if let Some(desc) = depfile_outputs.remove(name.as_str()) {
                    depfile_futures.push(async {
                        tokio::fs::File::options()
                            .write(true)
                            .create(true)
                            .truncate(true)
                            .open(cwd.join(&desc.path))
                            .await?
                            .write_all(data.as_bytes())
                            .await
                    });
                }
            }

            let has_all_required_depfiles = depfile_futures.is_empty()
                || futures::future::try_join_all(depfile_futures).await.is_ok();

            has_all_required_depfiles
                && (depfile_outputs.is_empty()
                    || depfile_outputs.values().all(|desc| desc.optional))
        };

        if has_all_required_files {
            trace!(
                "[{out_pretty}, {deps_key}]: Found valid deps cache entry",
                deps_key = depsfile.key
            );
            return Ok((
                ResolvedDependencies {
                    obj_key,
                    depsfile,
                    dependencies,
                    write_to_storage,
                },
                hidden_includes,
            ));
        }
    }

    // Discover dependencies and generate depfiles
    //
    //
    // When SCCACHE_NO_CACHE=1:
    // * Do not generate dependencies when computing the object hash
    // * We must generate all dependencies if doing a dist compile
    //
    let (sources, outputs) = if !depsfile.key.is_empty() || dist_compile {
        let start = Instant::now();

        let mut sources;
        let mut deps = BTreeSet::new();

        let gen_deps_args = GenerateDependenciesArgs {
            creator,
            executable,
            parsed_args,
            cwd,
            env_vars,
            tolerate_failure: false,
        };

        let mut hidden_includes_input = None;
        let mut hidden_includes = hidden_includes.clone();

        loop {
            hidden_includes_input = if !dist_compile || hidden_includes.is_empty() {
                None
            } else {
                trace!("[{out_pretty}]: found hidden includes: {hidden_includes:?}");
                // trace!("[{out_pretty}]:\n{}", hidden_includes.to_source_file());

                let temp = hidden_includes_input
                    .map(Ok)
                    .unwrap_or_else(crate::util::temppath)?;

                let mut f = fs_err::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(&temp)?;

                f.write_all(hidden_includes.to_source_file().as_bytes())?;

                Some(temp)
            };

            let gen_deps_args = if !dist_compile {
                &gen_deps_args
            } else {
                &GenerateDependenciesArgs {
                    parsed_args: &ParsedArguments {
                        input: hidden_includes_input
                            .as_deref()
                            .unwrap_or(parsed_args.input.as_path())
                            .to_owned(),
                        dependency_args: vec![],
                        depfile: None,
                        msvc_scan_depfile: None,
                        msvc_directives_depfile: None,
                        structured_depfile: None,
                        structured_depfile_format: None,
                        msvc_show_includes: false,
                        profile_generate: false,
                        ..parsed_args.clone()
                    },
                    tolerate_failure: hidden_includes_input.is_some(),
                    ..gen_deps_args.clone()
                }
            };

            let input = gen_deps_args.parsed_args.input.as_path();

            let (dependencies, duration) = compiler
                .generate_dependencies(gen_deps_args.clone())
                .await?;

            trace!(
                "[{out_pretty}]: compiler generate_dependencies: {}",
                fmt_duration_as_secs(&duration)
            );

            let dependencies = dependencies
                .into_iter()
                .filter(|path| path != input)
                .map(|path| cwd.join(path))
                .collect::<Vec<_>>();

            let deps_len = deps.len();
            deps.extend(dependencies);
            let found_all = deps.len() == deps_len;
            let dependencies = deps.iter().cloned().collect::<Vec<_>>();

            sources = if use_async_io {
                futures::future::try_join_all(dependencies.into_iter().map(|path| async {
                    if path.exists() {
                        SOURCE_FILE_CACHE
                            .get(path, compile_timestamp)
                            .await
                            .map(Some)
                    } else {
                        Ok(None)
                    }
                }))
                .await?
                .into_iter()
                .flatten()
                .collect::<Vec<Arc<SourceFileEntry>>>()
            } else {
                runtime
                    .spawn_blocking(move || {
                        dependencies
                            .into_iter()
                            .filter_map(|path| {
                                if path.exists() {
                                    Some(SOURCE_FILE_CACHE.get_sync(path, compile_timestamp))
                                } else {
                                    None
                                }
                            })
                            .try_collect()
                    })
                    .await??
            };

            // Break early if we don't need to find the additional hidden dependencies
            // for distributed compilation, or if we tried and didn't find any new ones.
            if found_all || !dist_compile {
                break;
            }

            hidden_includes =
                sources
                    .iter()
                    .fold(HiddenIncludes::default(), |mut hidden_includes, file| {
                        hidden_includes.extend([&file.hidden_includes]);
                        hidden_includes
                    });

            // If there's no more hidden dependencies to find, we're done
            if hidden_includes.is_empty() {
                break;
            }

            hidden_includes = hidden_includes.dedupe();
        }

        let outputs = if !depsfile.key.is_empty() && !dist_compile {
            futures::future::try_join_all(parsed_args.outputs.iter().filter_map(|(name, desc)| {
                if name.starts_with("dep") {
                    Some(async {
                        tokio::fs::read_to_string(cwd.join(&desc.path))
                            .await
                            .or_else(|err| {
                                if desc.optional {
                                    Ok(String::new())
                                } else {
                                    Err(err)
                                }
                            })
                            .map(|file| (name.to_string(), file))
                    })
                } else {
                    None
                }
            }))
            .await?
        } else {
            vec![]
        };

        // Track average preprocessor time
        {
            let duration = start.elapsed();
            let mut stats = service.stats.lock().await;
            stats.preprocessed += 1;
            stats.preprocessor_duration += duration;
        }

        (sources, outputs)
    } else {
        (vec![], vec![])
    };

    let hidden_includes = sources
        .iter()
        .fold(HiddenIncludes::default(), |mut hidden_includes, file| {
            hidden_includes.extend([&file.hidden_includes]);
            hidden_includes
        })
        .dedupe();

    let dependencies = Dependencies {
        sources: sources.into_iter().map(|file| file.meta.clone()).collect(),
        outputs: outputs.into_iter().collect::<BTreeMap<_, _>>(),
    };

    Ok((
        ResolvedDependencies {
            write_to_storage: !depsfile.key.is_empty(),
            obj_key: String::new(),
            dependencies,
            depsfile,
        },
        hidden_includes,
    ))
}

async fn put_dependencies(
    new_key: String,
    ResolvedDependencies {
        mut depsfile,
        dependencies,
        obj_key: old_key,
        ..
    }: ResolvedDependencies,
    storage: Arc<dyn Storage>,
    out_pretty: String,
    cache_write_timeout: Duration,
) -> Result<()> {
    let start = Instant::now();

    if new_key != old_key {
        depsfile.add_revision(&new_key, dependencies, &out_pretty);
    }

    let fut = depsfile.put(storage.as_ref());

    let duration = match tokio::time::timeout(cache_write_timeout, fut).await {
        Ok(res) => match res {
            Ok(duration) => duration,
            Err(err) => {
                // Don't fail if updating the deps cache entry fails, just log it
                debug!("[{out_pretty}]: Failed to update deps cache entry: {err:#}");
                start.elapsed()
            }
        },
        Err(_) => {
            // Don't fail if updating the deps cache entry fails, just log it
            debug!(
                "[{out_pretty}]: Deps cache write timeout ({:?})",
                start.elapsed()
            );
            start.elapsed()
        }
    };

    trace!(
        "[{out_pretty}]: put deps time: {}",
        fmt_duration_as_secs(&duration)
    );

    Ok(())
}

#[derive(Debug, Clone)]
struct CCompilerCommand<C, I, T>
where
    C: CCompilerImpl,
    I: CompileCommandImpl,
    T: CommandCreatorSync,
{
    cmd: I,
    compilation: CCompilation<C, T>,
    __: std::marker::PhantomData<T>,
}

impl<C, I, T> CCompilerCommand<C, I, T>
where
    C: CCompilerImpl,
    I: CompileCommandImpl,
    T: CommandCreatorSync,
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new(cmd: I, compilation: CCompilation<C, T>) -> Box<dyn CompileCommand<T>> {
        Box::new(CCompilerCommand {
            cmd,
            compilation,
            __: std::marker::PhantomData {},
        }) as Box<dyn CompileCommand<T>>
    }
}

#[async_trait]
impl<C, I, T> CompileCommand<T> for CCompilerCommand<C, I, T>
where
    C: CCompilerImpl,
    I: CompileCommandImpl,
    T: CommandCreatorSync,
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

    async fn execute(&self, service: &SccacheService<T>, creator: &T) -> Result<ProcessOutput> {
        self.cmd.execute(service, creator).await
    }

    fn box_clone(&self) -> Box<dyn CompileCommand<T>> {
        CCompilerCommand::<C, I, T>::new(self.cmd.clone(), self.compilation.clone())
    }
}

#[async_trait]
impl<T: CommandCreatorSync, I: CCompilerImpl> Compilation<T> for CCompilation<I, T> {
    fn generate_compile_commands(
        &self,
    ) -> Result<(
        Box<dyn CompileCommand<T>>,
        Option<dist::CompileCommand>,
        Cacheable,
    )> {
        let CCompilation {
            parsed_args,
            executable,
            compiler,
            cwd,
            env_vars,
            ..
        } = self;

        compiler
            .generate_compile_commands(GenerateCompileCommandsArgs {
                executable,
                parsed_args,
                cwd,
                env_vars,
            })
            .map(|(command, dist_command, cacheable)| {
                (
                    CCompilerCommand::new(command, self.clone()),
                    dist_command,
                    cacheable,
                )
            })
    }

    #[cfg(feature = "dist-client")]
    fn into_dist_packagers(self: Box<Self>) -> Result<DistPackagers> {
        let toolchain_packager = Box::new(CToolchainPackager {
            env_vars: self.env_vars.clone(),
            executable: self.executable.clone(),
            extra_files: self.compiler.extra_dist_files().collect(),
            kind: self.compiler.kind(),
        });

        let outputs_rewriter = Box::new(NoopOutputsRewriter);

        Ok((self, toolchain_packager, outputs_rewriter))
    }

    fn is_locally_preprocessed(&self) -> bool {
        self.is_locally_preprocessed
    }

    fn outputs<'a>(&'a self) -> Box<dyn Iterator<Item = FileObjectSource> + 'a> {
        Box::new(self.parsed_args.outputs.iter().map(|(k, output)| {
            let dir = self
                .cwd
                .join(&output.path)
                .parent()
                .unwrap_or(self.cwd.as_path())
                .to_owned();
            FileObjectSource {
                key: k.to_string(),
                dir,
                path: output.path.clone(),
                optional: output.optional,
            }
        }))
    }
}

async fn get_dependencies_dist<I, T>(
    obj_key: &str,
    args: GetDependenciesArgs<I, T>,
) -> Result<Vec<PathBuf>>
where
    I: CCompilerImpl,
    T: CommandCreatorSync,
{
    let deps_time = Instant::now();
    let runtime = args.runtime.clone();
    let storage = args.service.storage.clone();
    let cache_write_timeout = args.service.cache_write_timeout;
    let out_pretty = args.parsed_args.output_pretty().into_owned();

    let (deps, _) = get_dependencies(&GetDependenciesArgs {
        dist_compile: true,
        ..args
    })
    .await?;

    let deps_time = deps_time.elapsed();
    trace!(
        "[{out_pretty}]: get dist deps time: {}",
        fmt_duration_as_secs(&deps_time)
    );

    let dist_files = deps
        .dependencies
        .sources
        .iter()
        .map(|meta| PathBuf::from(&meta.path))
        .collect::<Vec<_>>();

    if !obj_key.is_empty() && (obj_key != deps.obj_key || deps.write_to_storage) {
        // If the dependencies cache entry is new or was updated, write it back to
        // storage.
        // This happens if the LRU order changes, or if time macros were found and
        // the entry's includes were updated to match the current ctimes/mtimes.
        // See [`SourceFileDeps::test_dependencies`].
        //
        // Immediately drive the dependency cache write so it
        // runs concurrently to the compilation.
        runtime.spawn(put_dependencies(
            obj_key.to_owned(),
            deps,
            storage,
            out_pretty,
            cache_write_timeout,
        ));
    }

    Ok(dist_files)
}

#[cfg(feature = "dist-client")]
#[async_trait]
impl<I: CCompilerImpl, T: CommandCreatorSync> pkg::InputsPackager for CCompilation<I, T> {
    fn write_inputs(
        self: Box<Self>,
        path_transformer: &mut dist::PathTransformer,
        compressor: &mut (dyn std::io::Write + Send),
    ) -> Result<()> {
        let CCompilation {
            parsed_args,
            mut dist_files,
            env_vars,
            deps_args,
            ..
        } = *self;

        if let Some(deps_args) = deps_args {
            dist_files.extend(
                deps_args
                    .runtime
                    .clone()
                    .block_on(async { get_dependencies_dist(&self.obj_key, deps_args).await })?,
            );
        }

        // use itertools::Itertools;
        // trace!(
        //     "Dist inputs:\n{}",
        //     std::iter::empty()
        //         .chain(dist_files.iter().unique())
        //         .chain(parsed_args.extra_dist_files.iter())
        //         .unique()
        //         .map(|path| format!("  {}", path.display()))
        //         .collect::<Vec<_>>()
        //         .join("\n")
        // );

        let mut package = pkg::ArchiveBuilder::new(path_transformer);

        // Add the input files and dependencies
        for paths in [&dist_files[..], &parsed_args.extra_dist_files[..]] {
            for path in paths {
                package.add_path(&env_vars, path)?;
            }
        }

        // Finish and compress the archive
        package.build()?.compress(compressor)
    }

    async fn write_inputs_async(
        self: Box<Self>,
        path_transformer: &mut dist::PathTransformer,
        sink: &mut (dyn futures::io::AsyncWrite + Unpin + Send + Sync),
    ) -> Result<()> {
        let CCompilation {
            parsed_args,
            mut dist_files,
            env_vars,
            deps_args,
            ..
        } = *self;

        if let Some(deps_args) = deps_args {
            dist_files.extend(get_dependencies_dist(&self.obj_key, deps_args).await?);
        }

        // use itertools::Itertools;
        // trace!(
        //     "Dist inputs:\n{}",
        //     std::iter::empty()
        //         .chain(dist_files.iter().unique())
        //         .chain(parsed_args.extra_dist_files.iter())
        //         .unique()
        //         .map(|path| format!("  {}", path.display()))
        //         .collect::<Vec<_>>()
        //         .join("\n")
        // );

        let mut package = pkg::ArchiveBuilder::new(path_transformer);

        // Add the input files and dependencies
        for paths in [&dist_files[..], &parsed_args.extra_dist_files[..]] {
            for path in paths {
                package.add_path(&env_vars, path)?;
            }
        }

        // Finish and compress the archive
        package.build()?.build_async(sink).await
    }
}

// Helper to use -print-file-name and -print-prog-name to look up
// files by path.
pub fn named_file<P: AsRef<Path>>(exe: P, arg: &str) -> Option<PathBuf> {
    let executable = exe.as_ref();
    let mut output = std::process::Command::new(executable)
        .arg(arg)
        .output()
        .ok()?;
    debug!(
        "{} {arg} output:\n{}\n===\n{}",
        executable.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    if !output.status.success() {
        debug!("exit failure");
        return None;
    }

    // Remove the trailing newline (if present)
    if output.stdout.last() == Some(&b'\n') {
        output.stdout.pop();
    }

    // Create our PathBuf from the raw bytes.  Assume that relative
    // paths can be found via PATH.
    let path = crate::util::bytes_to_path(output.stdout).ok()?;
    if path.is_absolute() {
        Some(path.into_owned())
    } else {
        which::which(path.as_ref()).ok()
    }
}

#[cfg(feature = "dist-client")]
#[allow(unused)]
pub(crate) struct CToolchainPackager {
    env_vars: Vec<(OsString, OsString)>,
    executable: PathBuf,
    kind: CCompilerKind,
    extra_files: Vec<PathBuf>,
}

#[cfg(feature = "dist-client")]
#[allow(unused)]
impl CToolchainPackager {
    pub fn named_file<P: AsRef<Path>>(exe: P, arg: &str) -> Option<PathBuf> {
        named_file(exe, arg)
    }
}

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
#[async_trait]
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
impl pkg::ToolchainPackager for CToolchainPackager {
    async fn package(
        self: Box<Self>,
        path_transformer: &mut dist::PathTransformer,
    ) -> Result<Arc<dyn pkg::ArchivePackager>> {
        use itertools::Itertools;
        use tokio_util::compat::TokioAsyncReadCompatExt;

        debug!(
            "Packaging toolchain for executable {:?}",
            self.executable.display()
        );
        let mut package_builder = pkg::ArchiveBuilder::new(path_transformer);

        // Add gcc implicit specfiles
        for path in self.extra_files.iter() {
            package_builder.add_path(&self.env_vars, path)?;
        }

        // Helper to use -print-file-name and -print-prog-name to look up
        // files by path.
        let named_file = |arg: &str| Self::named_file(&self.executable, arg);

        // Helper to add a named file/program by to the package.
        // We ignore the case where the file doesn't exist, as we don't need it.
        let add_named_prog = |builder: &mut pkg::ArchiveBuilder<'_>, name: &str| -> Result<()> {
            if let Some(path) = named_file(&format!("-print-prog-name={name}")) {
                builder.add_executable_and_deps(&self.env_vars, &path)?;
            }
            Ok(())
        };
        let add_named_file = |builder: &mut pkg::ArchiveBuilder<'_>, name: &str| -> Result<()> {
            if let Some(path) = named_file(&format!("-print-file-name={name}")) {
                builder.add_file(&self.env_vars, path)?;
            }
            Ok(())
        };

        let mut add_default_files = || -> Result<()> {
            // Add basic |as| and |objcopy| programs.
            add_named_prog(&mut package_builder, "as")?;
            add_named_prog(&mut package_builder, "objcopy")?;

            // Linker configuration.
            if Path::new("/etc/ld.so.cache").is_file() {
                package_builder.add_file(&self.env_vars, "/etc/ld.so.cache")?;
            }

            Ok(())
        };

        // Compiler-specific handling
        match self.kind {
            CCompilerKind::Clang => {
                add_default_files()?;
                package_builder.add_executable_and_deps(&self.env_vars, &self.executable)?;
                // Clang uses internal header files, so add them.
                if let Some(limits_h) = named_file("-print-file-name=include/limits.h") {
                    info!("limits_h = {}", limits_h.display());
                    package_builder.add_dir_contents(&self.env_vars, limits_h.parent().unwrap())?;
                }
            }

            CCompilerKind::Gcc => {
                // Various external programs / files which may be needed by gcc
                add_default_files()?;
                package_builder.add_executable_and_deps(&self.env_vars, &self.executable)?;
                add_named_prog(&mut package_builder, "cc1")?;
                add_named_prog(&mut package_builder, "cc1plus")?;
                add_named_file(&mut package_builder, "liblto_plugin.so")?;
            }

            CCompilerKind::Nvhpc => {
                // Various programs called by the nvc/nvc++ front end.
                let _ = add_default_files();

                // Handle NVHPC's mpicc/mpic++ compiler wrappers
                let executable = if let Some(executable) = named_file("-showme:command") {
                    package_builder.add_executable_and_deps(&self.env_vars, &executable)?;

                    // Handle NVHPC's symlinks of `mpic++ -> bin/env.sh` where `env.sh`
                    // is a wrapper to the real `mpic++` executable at `bin/.bin/mpic++`
                    if let Err(orig_err) =
                        package_builder.add_executable_and_deps(&self.env_vars, &self.executable)
                    {
                        if let Some(exe_dir) = self.executable.parent() {
                            let dot_bin_dir = if let Ok(path) = self.executable.read_link() {
                                exe_dir.join(path).parent().map(|p| p.join(".bin"))
                            } else {
                                Some(exe_dir).map(|p| p.join(".bin"))
                            };
                            if let Some(real_exe) = dot_bin_dir
                                .and_then(|p| self.executable.file_name().map(|name| p.join(name)))
                            {
                                if !real_exe.exists() {
                                    return Err(orig_err);
                                }
                                // Symlink `bin/mpic++` -> `bin/.bin/mpic++`
                                package_builder.add_link(&self.executable, &real_exe)?;
                                package_builder
                                    .add_executable_and_deps(&self.env_vars, &real_exe)?;
                            }
                        } else {
                            return Err(orig_err);
                        }
                    }
                    executable
                } else {
                    // If `-showme:command` fails, self.executable is nvc/nvc++
                    package_builder.add_executable_and_deps(&self.env_vars, &self.executable)?;
                    self.executable.clone()
                };

                let (contents, dirs, mut files): (Vec<PathBuf>, Vec<PathBuf>, Vec<PathBuf>) = {
                    use futures::{AsyncBufReadExt, future, io::BufReader};
                    use std::process::Stdio;

                    let mut child = tokio::process::Command::new(&executable)
                        .arg("-show")
                        .stdin(Stdio::null())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()?;

                    let info = child.stdout.take().unwrap();
                    let info = BufReader::new(info.compat());

                    // DEFCPPINC      These dirs just need to exist
                    // DEFSTDINC      These dirs just need to exist
                    // GCCINC         These dirs just need to exist
                    // GPPDIR         These dirs just need to exist
                    // STDINC         These dirs just need to exist
                    // SYSTEMINC      These dirs just need to exist
                    // NVCOMPILER                                         =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7
                    // COMPBIN        Compiler binary directory           =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin
                    // COMPLIB        Compiler library directory          =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/lib
                    // CCOMPDIR       Directory containing the C compiler =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/tools
                    // CPPCOMPDIR     Directory containing the C compiler =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/tools
                    // COMPINCDIRFULL                                     =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/include /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/include-stdexec
                    // CUDAROOT       (sometimes not set)                 =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9
                    // DCUDADIR                                           =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/
                    // USECUDAROOT                                        =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/
                    // CUDADIR                                            =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/bin
                    // CUDALIBDIR                                         =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/lib64
                    // LLVMBINDIR     Directory containing LLVM tools     =/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/share/llvm/bin
                    // DEFCUDAVERSIONVAL                                  =12.9
                    // DEFCUDAVERSION                                     =12.9

                    let lines = info
                        .lines()
                        .try_filter(|line| future::ready(line.len() >= 14 && line.contains('=')))
                        .try_filter_map(|line| {
                            future::ok(match &line[0..14] {
                                "CCOMPDIR      " | // .
                                "COMPBIN       " | // .
                                "COMPINCDIRFULL" | // .
                                "COMPLIB       " | // .
                                "CPPCOMPDIR    " | // .
                                "CUDADIR       " | // .
                                "CUDALIBDIR    " | // .
                                "CUDAROOT      " | // .
                                "DCUDADIR      " | // .
                                "DEFCPPINC     " | // .
                                "DEFSTDINC     " | // .
                                "GCCINC        " | // .
                                "GPPDIR        " | // .
                                "LLVMBINDIR    " | // .
                                "NVCOMPILER    " | // .
                                "STDINC        " | // .
                                "SYSTEMINC     " | // .
                                "USECUDAROOT   " | // .
                                "DEFCUDAVERSION" => {
                                    Some(line)
                                }
                                _ => None,
                            })
                        })
                        .try_collect::<Vec<_>>()
                        .await?;

                    // Search for the CUDA version first
                    let cuda_version = lines.iter().find_map(|line| match &line[0..14] {
                        "DEFCUDAVERSION" => {
                            line.split_once('=').map(|(_, ver)| ver.trim().to_owned())
                        }
                        _ => None,
                    });

                    lines
                        .into_iter()
                        .filter_map(|line| {
                            let name_and_dirs = line.split_once('=').map(|(name, dirs)| {
                                // `name` is the left-hand side of the '=', i.e. "NAME    Description of NAME "
                                //
                                //  There's always a space after NAME so it's safe to use `split_once(' ')`,
                                //  but I'm being defensive in case a future version of NVHPC emits lines
                                //  without any spaces between the name and value (like "NAME=/foo/bar")
                                //
                                // `dirs` is the right-hand side of the '=', i.e. "/foo/bar /foo/baz"
                                let (name, _) = name.split_once(' ').unwrap_or((name, ""));
                                (
                                    name.trim().to_owned(),
                                    dirs.trim()
                                        .split(' ')
                                        .map(|s| s.trim())
                                        .map(PathBuf::from)
                                        .collect::<Vec<_>>(),
                                )
                            });

                            name_and_dirs.and_then(|(name, dirs)| {
                                match &name[..] {
                                    "DCUDADIR" | "USECUDAROOT" => {
                                        // Append the CUDA version to these paths since they don't have them.
                                        // This is important if CUDAROOT isn't set.
                                        cuda_version.as_ref().map(|cuda_version| {
                                            (
                                                name,
                                                dirs.into_iter()
                                                    .flat_map(|dir| {
                                                        [dir.as_path().join(cuda_version), dir]
                                                    })
                                                    .collect::<Vec<_>>(),
                                            )
                                        })
                                    }
                                    _ => Some((name, dirs)),
                                }
                            })
                        })
                        .map(|(name, dirs)| {
                            let mut contents = vec![];
                            let mut files = vec![];

                            match &name[..] {
                                "NVCOMPILER" => {
                                    files.extend(dirs.iter().flat_map(|root| {
                                        use crate::util::OsStrExt;
                                        walkdir::WalkDir::new(root)
                                            .follow_links(true)
                                            .same_file_system(true)
                                            .into_iter()
                                            .filter_map_ok(|e| {
                                                if e.file_type().is_file() {
                                                    let name = e.file_name();
                                                    None
                                                        // rcfiles
                                                        .or(name.ends_with("rc").then_some(true))
                                                        .or(name.ends_with(".bc").then_some(true))
                                                        // mpic++-wrapper-data.txt
                                                        .or(name
                                                            .ends_with("-wrapper-data.txt")
                                                            .then_some(true))
                                                        .or(name
                                                            .ends_with("help-opal-wrapper.txt")
                                                            .then_some(true))
                                                        .map(|_| e.into_path())
                                                } else {
                                                    None
                                                }
                                            })
                                            .filter_map(|r| r.ok())
                                    }));
                                }
                                "COMPBIN" => {
                                    contents.extend(dirs.iter().map(|root| root.join("rcfiles")));
                                }
                                "COMPLIB" => {
                                    files.extend(dirs.iter().flat_map(|root| {
                                        [
                                            "acc_init_link_cuda.o",
                                            "cuda_init_register_end.o",
                                            "init_pgpf.o",
                                            "libmem.il",
                                            "nvhpc.ld",
                                            "nvhpc.syms",
                                        ]
                                        .iter()
                                        .map(|name| root.join(name))
                                    }));
                                }
                                "CCOMPDIR" | "CPPCOMPDIR" => {
                                    // Add everything under `compilers/bin/tools`
                                    contents.extend_from_slice(dirs.as_slice());
                                }
                                "COMPINCDIRFULL" => {
                                    // NVHPC uses internal LLVM header files, so add them.
                                    contents.extend_from_slice(dirs.as_slice());
                                }
                                "CUDAROOT" | "DCUDADIR" | "USECUDAROOT" => {
                                    files
                                        .extend(dirs.iter().map(|root| root.join("nvvm/bin/cicc")));
                                    files.extend(dirs.iter().flat_map(|root| {
                                        ["include/cuda.h", "nvvm/lib64/libnvvm.so"]
                                            .iter()
                                            .map(|name| root.join(name))
                                    }));
                                }
                                "CUDADIR" => {
                                    files.extend(dirs.iter().flat_map(|root| {
                                        [
                                            "bin2c",
                                            "cudafe++",
                                            "fatbinary",
                                            "nvlink",
                                            "nvprune",
                                            "ptxas",
                                        ]
                                        .iter()
                                        .map(|name| root.join(name))
                                    }));
                                    files.extend(dirs.iter().map(|root| root.join("nvcc.profile")));
                                }
                                "CUDALIBDIR" => {}
                                "LLVMBINDIR" => {
                                    files.extend(dirs.iter().flat_map(|root| {
                                        ["llc", "opt", "llvm-as", "llvm-link", "llvm-mc"]
                                            .iter()
                                            .map(|name| root.join(name))
                                    }));
                                }
                                _ => {}
                            }

                            (contents, dirs, files)
                        })
                        .fold((vec![], vec![], vec![]), |mut acc, res| {
                            acc.0.extend(res.0);
                            acc.1.extend(res.1);
                            acc.2.extend(res.2);
                            acc
                        })
                };

                if let Ok(as_path) = which::which("as") {
                    files.push(as_path);
                }

                for path in dirs.into_iter().filter(|p| p.exists()) {
                    if path.is_dir() {
                        let _ = package_builder
                            .add_dir(path.clone())
                            .map_err(|e| trace!("add_dir error {path:?}: {e:?}"));
                    }
                }

                for path in files.into_iter().filter(|p| p.exists()) {
                    if path.is_file() || path.is_symlink() {
                        let _ = package_builder
                            .add_file(&self.env_vars, path.clone())
                            .map_err(|e| trace!("add_file error {path:?}: {e:?}"));
                    }
                }

                for path in contents.into_iter().filter(|p| p.exists()) {
                    if path.is_dir() {
                        let _ = package_builder
                            .add_dir_contents(&self.env_vars, &path)
                            .map_err(|e| trace!("add_dir_contents error {path:?}: {e:?}"));
                    }
                }
            }

            CCompilerKind::Msvc => {
                use crate::util::OsStrExt;
                // cl.exe
                package_builder.add_executable_and_deps(&self.env_vars, &self.executable)?;

                if let Some(dir) = self.executable.parent() {
                    // mspdbcmf.exe
                    // mspdbsrv.exe
                    for exe in [
                        format!("mspdbcmf{}", std::env::consts::EXE_SUFFIX),
                        format!("mspdbsrv{}", std::env::consts::EXE_SUFFIX),
                    ] {
                        package_builder.add_executable_and_deps(&self.env_vars, dir.join(exe))?;
                    }

                    let dll_prefixes = [
                        "c1",
                        "c2",
                        "mspdbcore",
                        "mspdbst",
                        "tbbmalloc",
                        // msobj140.dll
                        "msobj",
                        // mspdb140.dll
                        "mspdb",
                        // msvcp140.dll
                        "msvcp",
                        // vcruntime140.dll
                        // vcruntime140_1.dll
                        "vcruntime",
                        // 1033/clui.dll
                        "clui",
                        // 1033/mspdbcmfui.dll
                        "mspdbcmfui",
                    ];

                    let paths = walkdir::WalkDir::new(dir)
                        .min_depth(1)
                        .max_depth(2)
                        .into_iter()
                        .filter_map(|e| e.ok())
                        .filter_map(|e| {
                            if e.file_type().is_file()
                                && let file_name = e.file_name().to_ascii_lowercase()
                                && file_name.ends_with(std::env::consts::DLL_SUFFIX)
                            {
                                Some((file_name, e.path().to_owned()))
                            } else {
                                None
                            }
                        })
                        .filter_map(|(file_name, file_path)| {
                            if dll_prefixes
                                .iter()
                                .any(|prefix| file_name.starts_with(prefix))
                            {
                                Some(file_path)
                            } else {
                                None
                            }
                        });

                    for path in paths {
                        package_builder.add_executable_and_deps(&self.env_vars, &path)?;
                    }
                }
            }

            _ => {
                package_builder.add_executable_and_deps(&self.env_vars, &self.executable)?;
            }
        }

        // Return the builder so the archive can be lazily created, depending
        // on whether or not the scheduler reports it already has the toolchain
        package_builder.build()
    }
}

/// The cache is versioned by the inputs to `HashKeyParams`.
pub const CACHE_VERSION: &[u8] = b"14";

/// Environment variables that are factored into the cache key.
static CACHED_ENV_VARS: LazyLock<HashSet<&'static OsStr>> = LazyLock::new(|| {
    [
        // SCCACHE_C_CUSTOM_CACHE_BUSTER has no particular meaning behind it,
        // serving as a way for the user to factor custom data into the hash.
        // One can set it to different values for different invocations
        // to prevent cache reuse between them.
        "SCCACHE_C_CUSTOM_CACHE_BUSTER",
        "C_INCLUDE_PATH",
        "CCC_OVERRIDE_OPTIONS",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "INCLUDE",
        "IPHONEOS_DEPLOYMENT_TARGET",
        "MACOSX_DEPLOYMENT_TARGET",
        "OBJC_INCLUDE_PATH",
        "OBJCPLUS_INCLUDE_PATH",
        "QNX_HOST",
        "QNX_TARGET",
        "SDKROOT",
        "TVOS_DEPLOYMENT_TARGET",
        "WATCHOS_DEPLOYMENT_TARGET",
    ]
    .iter()
    .map(OsStr::new)
    .collect()
});

// cache for hashed source files
static SOURCE_FILE_CACHE: LazyLock<SourceFileCache> = LazyLock::new(|| {
    SourceFileCache::new(
        // TODO: capacity chosen arbitrarily, should measure instead
        50_000,
    )
});

/// Parameters for computing a hash key for C/C++ compilation caching.
///
/// Construct with required fields via [`HashKeyParams::new`], then add optional
/// fields with builder methods.
///
/// # Example
/// ```ignore
/// let hash = HashKeyParams::new(
///     "abc123",           // compiler_digest
///     &parsed_args,       // arguments
/// )
/// .with_extra_hashes(&extra)
/// .with_plusplus(true)
/// .compute();
/// ```
// #[derive(Debug, Clone)]
pub struct HashKeyParams<'a> {
    args: Vec<&'a OsString>,
    async_io: bool,
    compile_timestamp: DateTime<Utc>,
    cwd: &'a Path,
    env_vars: &'a [(OsString, OsString)],
    digests: &'a [&'a str],
    input_files: Vec<PathBuf>,
    parsed_args: &'a ParsedArguments,
    hashed_input_path: Option<&'a Path>,
    plusplus: bool,
    runtime: Option<&'a tokio::runtime::Handle>,
    version: &'a [u8],
}

impl<'a> HashKeyParams<'a> {
    /// Creates a new hash key parameter set with required fields.
    ///
    /// Optional fields have default values and can be set via `with_*` methods.
    ///
    /// # Arguments
    /// * `parsed_args` - Parsed compiler arguments
    /// * `preprocessor_output` - Preprocessed source to hash
    pub fn new(parsed_args: &'a ParsedArguments) -> Self {
        Self {
            args: vec![],
            async_io: true,
            compile_timestamp: Utc::now(),
            cwd: Path::new(""),
            env_vars: &[],
            digests: &[],
            input_files: vec![],
            parsed_args,
            hashed_input_path: None,
            plusplus: false,
            runtime: None,
            version: &[],
        }
    }

    /// Sets the environment variables to consider for caching.
    pub fn with_args(mut self, args: &'a [OsString]) -> Self {
        self.args.extend(args);
        self
    }

    pub fn with_async_io(mut self, async_io: bool) -> Self {
        self.async_io = async_io;
        self
    }

    /// Sets the compile timestamp to use when comparing __DATE__, __TIME__, and __TIMESTAMP__ macros
    pub fn with_compile_timestamp(mut self, compile_timestamp: DateTime<Utc>) -> Self {
        self.compile_timestamp = compile_timestamp;
        self
    }

    /// Sets the cwd to use when hashing.
    pub fn with_cwd(mut self, cwd: &'a Path) -> Self {
        self.cwd = cwd;
        self
    }

    /// Sets the environment variables to consider for caching.
    pub fn with_env_vars(mut self, env_vars: &'a [(OsString, OsString)]) -> Self {
        self.env_vars = env_vars;
        self
    }

    /// Sets digests to include in the cache key.
    pub fn with_digests(mut self, digests: &'a [&'a str]) -> Self {
        self.digests = digests;
        self
    }

    /// Sets the input file and dependencies to hash.
    pub fn with_input_files(mut self, input_files: &'a [PathBuf]) -> Self {
        self.input_files.extend(input_files.iter().cloned());
        self
    }

    /// Sets the optional input path to hash.
    pub fn with_hashed_input_path<P: Into<Option<&'a Path>>>(
        mut self,
        hashed_input_path: P,
    ) -> Self {
        self.hashed_input_path = hashed_input_path.into();
        self
    }

    /// Sets whether this is a C++ compiler (affects clang/clang++ distinction).
    pub fn with_plusplus(mut self, plusplus: bool) -> Self {
        self.plusplus = plusplus;
        self
    }

    pub fn with_runtime(mut self, runtime: &'a tokio::runtime::Handle) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Sets the cache entry version.
    pub fn with_version(mut self, version: &'a [u8]) -> Self {
        self.version = version;
        self
    }

    /// Computes the hash key based on the configured parameters.
    ///
    /// If `basedirs` are provided, paths in the preprocessor output will be normalized by
    /// stripping the longest matching basedir prefix. This enables cache hits across different
    /// absolute paths (similar to ccache's CCACHE_BASEDIR).
    ///
    /// # Note
    /// If you change any of the inputs to the hash, you should change `CACHE_VERSION`.
    pub async fn compute(self) -> Result<Option<String>> {
        let mut m = Digest::new();
        m.update(self.version);
        // Include all the digests (compiler, assembler, specfiles, etc.)
        for digest in self.digests {
            digest.hash(&mut HashToDigest { digest: &mut m });
        }
        // clang and clang++ have different behavior despite being byte-for-byte identical binaries, so
        // we have to incorporate that into the hash as well.
        m.update(&[self.plusplus as u8]);
        // Encode the color mode too, because that affects the cached stdout/stderr
        m.update(&[(self.parsed_args.color_mode != ColorMode::Off) as u8]);
        m.update(self.parsed_args.language.as_str().as_bytes());

        // Hash args
        for arg in self.args {
            arg.hash(&mut HashToDigest { digest: &mut m });
        }

        for (var, val) in self.env_vars.iter() {
            if CACHED_ENV_VARS.contains(var.as_os_str()) {
                var.hash(&mut HashToDigest { digest: &mut m });
                m.update(&b"="[..]);

                // Canonicalize the paths in CPATH and friends, otherwise we
                // can get false-positive dependencies cache hits when these
                // envvars traverse symlinks
                if var == "CPATH"
                    || var == "C_INCLUDE_PATH"
                    || var == "CPLUS_INCLUDE_PATH"
                    || var == "OBJC_INCLUDE_PATH"
                    || var == "OBJCPLUS_INCLUDE_PATH"
                {
                    std::env::join_paths(
                        std::env::split_paths(val)
                            .map(|path| self.cwd.join(path))
                            // Fallback to original path if we can't canonicalize
                            .map(|path| dunce::canonicalize(&path).unwrap_or(path)),
                    )?
                    .hash(&mut HashToDigest { digest: &mut m });
                } else {
                    val.hash(&mut HashToDigest { digest: &mut m });
                }
            }
        }

        if let Some(path) = self.hashed_input_path {
            path.hash(&mut HashToDigest { digest: &mut m });
        }

        let runtime = if let Some(runtime) = self.runtime {
            runtime
        } else {
            &tokio::runtime::Handle::current()
        };

        let entries = if self.async_io {
            // Hash the input and dependencies
            futures::future::try_join_all(self.input_files.into_iter().map(|path| async move {
                let path = self.cwd.join(path);
                let entry = SOURCE_FILE_CACHE.get(&path, self.compile_timestamp).await?;

                if entry.found_time {
                    // __TIME__ found, disable caching
                    debug!(
                        "[{}]: Found __TIME__ in '{}'",
                        self.parsed_args.output_pretty(),
                        path.display()
                    );
                    Ok::<_, anyhow::Error>(None)
                } else {
                    Ok(Some(entry.clone()))
                }
            }))
            .await?
        } else {
            // Canonicalize and hash the dependencies on a background thread
            runtime
                .spawn_blocking({
                    let cwd = self.cwd.to_owned();
                    let out_pretty = self.parsed_args.output_pretty().into_owned();
                    move || {
                        let mut entries = vec![];

                        for path in self.input_files {
                            let path = cwd.join(path.as_path());
                            let entry =
                                SOURCE_FILE_CACHE.get_sync(&path, self.compile_timestamp)?;

                            if entry.found_time {
                                // __TIME__ found, disable caching
                                debug!("[{out_pretty}]: Found __TIME__ in '{}'", path.display());
                                entries.push(None);
                            } else {
                                entries.push(Some(entry.clone()));
                            }
                        }

                        Ok::<_, anyhow::Error>(entries)
                    }
                })
                .await??
        };

        if entries.iter().any(|entry| entry.is_none()) {
            Ok(None)
        } else {
            for entry in entries.into_iter().flatten() {
                entry.meta.digest.hash(&mut HashToDigest { digest: &mut m });
            }
            Ok(Some(m.finish()))
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Deserialize, Serialize)]
pub struct P1689r5_JSON {
    pub version: u8,
    pub revision: u8,
    #[serde(default)]
    pub rules: Vec<P1689r5_Rule>,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Deserialize, Serialize)]
pub struct P1689r5_Rule {
    #[serde(rename = "primary-output")]
    pub primary_output: String,
    #[serde(default)]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub provides: Vec<P1689r5_Dep>,
    #[serde(default)]
    pub requires: Vec<P1689r5_Dep>,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Deserialize, Serialize)]
pub struct P1689r5_Dep {
    #[serde(default, rename = "logical-name")]
    pub logical_name: String,
    #[serde(default, rename = "source-path")]
    pub source_path: String,
    #[serde(default, rename = "lookup-method")]
    pub lookup_method: String,
    #[serde(default, rename = "unique-on-source-path")]
    pub unique_on_source_path: bool,
}

#[cfg(test)]
mod test {
    use futures::io::AllowStdIo;

    use crate::{
        test::utils::*,
        util::{HASH_BUFFER_SIZE, MAX_TIME_MACRO_HAYSTACK_LEN},
    };

    use super::*;

    #[test]
    fn test_same_content() {
        let args = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            language: Language::C,
            ..Default::default()
        };

        let f = TestFixture::new();
        let p = f.tempdir.path().join("source.c");
        std::fs::write(&p, "hello world").unwrap();
        let p = vec![p];

        let h1 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        let h2 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_plusplus_differs() {
        let args = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            language: Language::C,
            ..Default::default()
        };

        let f = TestFixture::new();
        let p = f.tempdir.path().join("source.c");
        std::fs::write(&p, "hello world").unwrap();
        let p = vec![p];

        let h1 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        let h2 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .with_plusplus(true)
            .compute()
            .wait()
            .unwrap();
        assert_neq!(h1, h2);
    }

    #[test]
    fn test_compiler_digest_differs() {
        let args = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            language: Language::C,
            ..Default::default()
        };

        let f = TestFixture::new();
        let p = f.tempdir.path().join("source.c");
        std::fs::write(&p, "hello world").unwrap();
        let p = vec![p];

        let h1 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        let h2 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["efgh"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        let h3 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["ijkl"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        assert_neq!(h1, h2);
        assert_neq!(h2, h3);
        // Not knowing the assembler must keep the key the compiler alone would give,
        // so that caches predating assembler tracking stay usable.
        let h4 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        assert_eq!(h1, h4);
    }

    #[test]
    fn test_header_differs() {
        let args = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            language: Language::C,
            ..Default::default()
        };

        let f = TestFixture::new();
        let p = f.tempdir.path().join("source.c");
        std::fs::write(&p, "hello world").unwrap();
        let p = vec![p];

        let h1 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();

        let args_cheader = ParsedArguments {
            language: Language::CHeader,
            ..args.clone()
        };

        let h2 = HashKeyParams::new(&args_cheader)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        assert_neq!(h1, h2);
    }

    #[test]
    fn test_plusplus_header_differs() {
        let args = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            language: Language::Cxx,
            ..Default::default()
        };

        let f = TestFixture::new();
        let p = f.tempdir.path().join("source.c");
        std::fs::write(&p, "hello world").unwrap();
        let p = vec![p];

        let h1 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .with_plusplus(true)
            .compute()
            .wait()
            .unwrap();

        let args_cxxheader = ParsedArguments {
            language: Language::CxxHeader,
            ..args.clone()
        };

        let h2 = HashKeyParams::new(&args_cxxheader)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .with_plusplus(true)
            .compute()
            .wait()
            .unwrap();
        assert_neq!(h1, h2);
    }

    #[test]
    fn test_hash_key_executable_contents_differs() {
        let args = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            language: Language::C,
            ..Default::default()
        };

        let f = TestFixture::new();
        let p = f.tempdir.path().join("source.c");
        std::fs::write(&p, "hello world").unwrap();
        let p = vec![p];

        let h1 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        let h2 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["wxyz"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        assert_neq!(h1, h2);
    }

    #[test]
    fn test_hash_key_args_differs() {
        let a = ParsedArguments {
            common_args: ovec!["a"],
            language: Language::C,
            ..Default::default()
        };

        let ab = ParsedArguments {
            common_args: ovec!["a", "b"],
            ..a.clone()
        };

        let abc = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            ..a.clone()
        };

        let xyz = ParsedArguments {
            common_args: ovec!["x", "y"],
            arch_args: ovec!["z"],
            ..a.clone()
        };

        // const PREPROCESSED: &[u8] = b"hello world";
        let f = TestFixture::new();
        let p = f.tempdir.path().join("source.c");
        std::fs::write(&p, "hello world").unwrap();
        let p = vec![p];

        let h_abc = HashKeyParams::new(&abc)
            .with_args(&abc.arch_args[..])
            .with_args(&abc.common_args[..])
            .with_args(&abc.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        let h_xyz = HashKeyParams::new(&xyz)
            .with_args(&xyz.arch_args[..])
            .with_args(&xyz.common_args[..])
            .with_args(&xyz.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        let h_ab = HashKeyParams::new(&ab)
            .with_args(&ab.arch_args[..])
            .with_args(&ab.common_args[..])
            .with_args(&ab.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();
        let h_a = HashKeyParams::new(&a)
            .with_args(&a.arch_args[..])
            .with_args(&a.common_args[..])
            .with_args(&a.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();

        assert_neq!(h_abc, h_xyz);
        assert_neq!(h_abc, h_ab);
        assert_neq!(h_abc, h_a);
    }

    #[test]
    fn test_hash_key_preprocessed_content_differs() {
        let args = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            language: Language::C,
            ..Default::default()
        };

        let f = TestFixture::new();
        let p1 = f.tempdir.path().join("source1.c");
        std::fs::write(&p1, "hello world").unwrap();
        let p1 = vec![p1];

        let p2 = f.tempdir.path().join("source2.c");
        std::fs::write(&p2, "goodbye").unwrap();
        let p2 = vec![p2];

        let h1 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p1)
            .compute()
            .wait()
            .unwrap();
        let h2 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p2)
            .compute()
            .wait()
            .unwrap();
        assert_neq!(h1, h2);
    }

    #[test]
    fn test_hash_key_env_var_differs() {
        let args = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            language: Language::C,
            ..Default::default()
        };

        let f = TestFixture::new();
        let p = f.tempdir.path().join("source.c");
        std::fs::write(&p, "hello world").unwrap();
        let p = vec![p];

        for var in CACHED_ENV_VARS.iter() {
            let h1 = HashKeyParams::new(&args)
                .with_args(&args.arch_args[..])
                .with_args(&args.common_args[..])
                .with_args(&args.preprocessor_args[..])
                .with_digests(&["abcd"])
                .with_input_files(&p)
                .compute()
                .wait()
                .unwrap();

            let vars1 = vec![(OsString::from(var), OsString::from("something"))];
            let h2 = HashKeyParams::new(&args)
                .with_args(&args.arch_args[..])
                .with_args(&args.common_args[..])
                .with_args(&args.preprocessor_args[..])
                .with_digests(&["abcd"])
                .with_input_files(&p)
                .with_env_vars(&vars1)
                .compute()
                .wait()
                .unwrap();

            let vars2 = vec![(OsString::from(var), OsString::from("something else"))];
            let h3 = HashKeyParams::new(&args)
                .with_args(&args.arch_args[..])
                .with_args(&args.common_args[..])
                .with_args(&args.preprocessor_args[..])
                .with_digests(&["abcd"])
                .with_input_files(&p)
                .with_env_vars(&vars2)
                .compute()
                .wait()
                .unwrap();

            assert_neq!(h1, h2);
            assert_neq!(h2, h3);
        }
    }

    #[test]
    fn test_extra_hash_data() {
        let args = ParsedArguments {
            common_args: ovec!["a", "b"],
            arch_args: ovec!["c"],
            language: Language::C,
            ..Default::default()
        };

        let extra_data = ["hello", "world"];

        let f = TestFixture::new();
        let p = f.tempdir.path().join("source.c");
        std::fs::write(&p, "hello world").unwrap();
        let p = vec![p];

        let h1 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_input_files(&p)
            .with_digests(&[&["abcd"], &extra_data[..]].concat())
            .compute()
            .wait()
            .unwrap();
        let h2 = HashKeyParams::new(&args)
            .with_args(&args.arch_args[..])
            .with_args(&args.common_args[..])
            .with_args(&args.preprocessor_args[..])
            .with_digests(&["abcd"])
            .with_input_files(&p)
            .compute()
            .wait()
            .unwrap();

        assert_neq!(h1, h2);
    }

    // #[test]
    // fn test_hash_key_basedirs() {
    //     let args = ParsedArguments {
    //         common_args: ovec!["a", "b"],
    //         arch_args: ovec!["c"],
    //         language: Language::C,
    //         ..Default::default()
    //     };
    //     let preprocessed1: &[u8] =
    //         b"# 1 \"/home/user1/project/src/main.c\"\nint main() { return 0; }";
    //     let preprocessed2: &[u8] =
    //         b"# 1 \"/home/user2/project/src/main.c\"\nint main() { return 0; }";
    //     let basedirs = [
    //         b"/home/user1/project".to_vec(),
    //         b"/home/user2/project".to_vec(),
    //     ];

    //     // Helper to compute with basedirs
    //     let hash_with_basedirs = |output: &[u8], dirs: &[Vec<u8>]| {
    //         HashKeyParams::new("abcd", &args, into_process_output_stream(output))
    //             .with_basedirs(dirs)
    //             .compute()
    //             .wait()
    //             .unwrap()
    //     };

    //     // Test 1: Same hash with different absolute paths when basedir is used
    //     let h1 = hash_with_basedirs(preprocessed1, &basedirs);
    //     let h2 = hash_with_basedirs(preprocessed2, &basedirs);
    //     assert_eq!(h1, h2);

    //     // Test 2: Same hash with single basedir that matches each
    //     assert_eq!(
    //         hash_with_basedirs(preprocessed1, &basedirs[..1]),
    //         hash_with_basedirs(preprocessed2, &basedirs[1..])
    //     );

    //     // Test 3: Different hashes without basedir
    //     assert_neq!(
    //         HashKeyParams::new("abcd", &args, into_process_output_stream(preprocessed1))
    //             .compute()
    //             .wait()
    //             .unwrap(),
    //         HashKeyParams::new("abcd", &args, into_process_output_stream(preprocessed2))
    //             .compute()
    //             .wait()
    //             .unwrap()
    //     );

    //     // Test 4: Doesn't work with trailing slash in basedir, they must be normalized in config
    //     let basedir_slash = b"/home/user1/project/".to_vec();
    //     let h_slash = hash_with_basedirs(preprocessed1, std::slice::from_ref(&basedir_slash));
    //     assert_neq!(h1, h_slash);

    //     // Test 5: Multiple basedirs - longest match wins
    //     let multi_basedirs = vec![
    //         b"/home/user1".to_vec(),
    //         b"/home/user1/project".to_vec(), // This should match (longest)
    //     ];
    //     assert_eq!(h1, hash_with_basedirs(preprocessed1, &multi_basedirs));

    //     // Test 6: Works for C++ files too
    //     let preprocessed_cpp1: &[u8] =
    //         b"# 1 \"/home/user1/project/src/main.cpp\"\nint main() { return 0; }";
    //     let preprocessed_cpp2: &[u8] =
    //         b"# 1 \"/home/user2/project/src/main.cpp\"\nint main() { return 0; }";

    //     let args_cxx = ParsedArguments {
    //         language: Language::Cxx,
    //         ..args.clone()
    //     };
    //     let h_cpp1 = HashKeyParams::new(
    //         "abcd",
    //         &args_cxx,
    //         into_process_output_stream(preprocessed_cpp1),
    //     )
    //     .with_plusplus(true)
    //     .with_basedirs(&basedirs)
    //     .compute()
    //     .wait()
    //     .unwrap();
    //     let h_cpp2 = HashKeyParams::new(
    //         "abcd",
    //         &args_cxx,
    //         into_process_output_stream(preprocessed_cpp2),
    //     )
    //     .with_plusplus(true)
    //     .with_basedirs(&basedirs)
    //     .compute()
    //     .wait()
    //     .unwrap();
    //     assert_eq!(h_cpp1, h_cpp2);
    // }

    // #[cfg(target_os = "windows")]
    // #[test]
    // fn test_hash_key_basedirs_windows_escaped_backslashes() {
    //     // On Windows, paths in preprocessor output are C string literals with
    //     // escaped backslashes (e.g. MSVC: #line 1 "D:\\dir\\file.c", clang:
    //     // # 1 "D:\\dir\\file.c"). The same source compiled from two different
    //     // directories must produce the same hash when both are basedirs.
    //     let args = ParsedArguments {
    //         common_args: ovec!["a", "b"],
    //         arch_args: ovec!["c"],
    //         language: Language::C,
    //         ..Default::default()
    //     };
    //     let preprocessed1 =
    //         b"#line 1 \"D:\\\\dev\\\\user1\\\\project\\\\src\\\\main.c\"\nint main() { return 0; }";
    //     let preprocessed2 =
    //         b"#line 1 \"E:\\\\work\\\\user2\\\\checkout\\\\src\\\\main.c\"\nint main() { return 0; }";
    //     // As normalized by Config: lowercase, forward slashes, trailing slash
    //     let basedirs = [
    //         b"d:/dev/user1/project/".to_vec(),
    //         b"e:/work/user2/checkout/".to_vec(),
    //     ];

    //     let hash_with_basedirs = |output: &[u8], dirs: &[Vec<u8>]| {
    //         HashKeyParams::new("abcd", &args, into_process_output_stream(output))
    //             .with_basedirs(dirs)
    //             .compute()
    //             .wait()
    //             .unwrap()
    //     };

    //     // Same hash with different absolute (escaped backslash) paths
    //     let h1 = hash_with_basedirs(preprocessed1, &basedirs);
    //     let h2 = hash_with_basedirs(preprocessed2, &basedirs);
    //     assert_eq!(h1, h2);

    //     // Different hashes without basedirs
    //     assert_neq!(
    //         HashKeyParams::new("abcd", &args, into_process_output_stream(preprocessed1))
    //             .compute()
    //             .wait()
    //             .unwrap(),
    //         HashKeyParams::new("abcd", &args, into_process_output_stream(preprocessed2))
    //             .compute()
    //             .wait()
    //             .unwrap()
    //     );
    // }

    #[test]
    fn test_language_from_file_name() {
        fn t(extension: &str, expected: Language) {
            let path_str = format!("input.{extension}");
            let path = Path::new(&path_str);
            let actual = Language::from_file_name(path);
            assert_eq!(actual, Some(expected));
        }

        t("s", Language::Assembler);
        t("S", Language::AssemblerToPreprocess);
        t("sx", Language::AssemblerToPreprocess);

        t("c", Language::C);

        t("i", Language::CPreprocessed);

        t("C", Language::Cxx);
        t("cc", Language::Cxx);
        t("cp", Language::Cxx);
        t("cpp", Language::Cxx);
        t("CPP", Language::Cxx);
        t("cxx", Language::Cxx);
        t("c++", Language::Cxx);

        t("ii", Language::CxxPreprocessed);

        t("h", Language::GenericHeader);

        t("hh", Language::CxxHeader);
        t("H", Language::CxxHeader);
        t("hp", Language::CxxHeader);
        t("hxx", Language::CxxHeader);
        t("hpp", Language::CxxHeader);
        t("HPP", Language::CxxHeader);
        t("h++", Language::CxxHeader);
        t("tcc", Language::CxxHeader);

        t("m", Language::ObjectiveC);

        t("mi", Language::ObjectiveCPreprocessed);

        t("M", Language::ObjectiveCxx);
        t("mm", Language::ObjectiveCxx);

        t("mii", Language::ObjectiveCxxPreprocessed);

        t("cu", Language::Cuda);
        t("hip", Language::Hip);
    }

    #[test]
    fn test_language_from_file_name_none() {
        fn t(extension: &str) {
            let path_str = format!("input.{extension}");
            let path = Path::new(&path_str);
            let actual = Language::from_file_name(path);
            let expected = None;
            assert_eq!(actual, expected);
        }

        // gcc parses file-extensions as case-sensitive
        t("Cp");
        t("Cpp");
        t("Hp");
        t("Hpp");
        t("Mm");
        t("Cu");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_empty_file() {
        let buf: Vec<u8> = vec![];
        let buf = AllowStdIo::new(&buf[..]);
        let hash = Digest::from_reader_with_macro_finders(buf).await.unwrap().0;
        assert_eq!(hash.finish(), Digest::new().finish());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_small_file_no_match() {
        let buf = b"This is a small file, which doesn't contain any time macros.";
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(!finder.found_time_macros());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_small_file_match() {
        let buf = b"__TIME__";
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(finder.found_time());
        assert!(!finder.found_timestamp());
        assert!(!finder.found_date());
        let buf = b"__DATE__";
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(!finder.found_time());
        assert!(!finder.found_timestamp());
        assert!(finder.found_date());
        let buf = b"__TIMESTAMP__";
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(!finder.found_time());
        assert!(finder.found_timestamp());
        assert!(!finder.found_date());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_small_file_match_multiple() {
        let buf = b"__TIMESTAMP____DATE____TIME__";
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(finder.found_time());
        assert!(finder.found_timestamp());
        assert!(finder.found_date());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_large_file_no_match() {
        let buf = vec![0; HASH_BUFFER_SIZE * 2];
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(!finder.found_time_macros());
        assert!(!finder.found_time());
        assert!(!finder.found_timestamp());
        assert!(!finder.found_date());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_large_file_match_no_overlap() {
        let mut buf = vec![0; HASH_BUFFER_SIZE * 2];
        buf.extend(b"__TIMESTAMP____DATE____TIME__");
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(finder.found_time());
        assert!(finder.found_timestamp());
        assert!(finder.found_date());
    }
    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_large_file_match_overlap() {
        let mut buf = vec![0; HASH_BUFFER_SIZE * 2];
        // Make the pattern overlap two buffer chunks to make sure we account for this
        let start = HASH_BUFFER_SIZE - MAX_TIME_MACRO_HAYSTACK_LEN / 2;
        buf[start..][..b"__TIMESTAMP__".len()].copy_from_slice(b"__TIMESTAMP__");
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(!finder.found_time());
        assert!(finder.found_timestamp());
        assert!(!finder.found_date());

        let mut buf = vec![0; HASH_BUFFER_SIZE * 2];
        // Make the pattern overlap two buffer chunks to make sure we account for this
        let start = HASH_BUFFER_SIZE - MAX_TIME_MACRO_HAYSTACK_LEN / 2;
        buf[start..][..b"__TIME__".len()].copy_from_slice(b"__TIME__");
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(finder.found_time());
        assert!(!finder.found_timestamp());
        assert!(!finder.found_date());

        let mut buf = vec![0; HASH_BUFFER_SIZE * 2];
        // Make the pattern overlap two buffer chunks to make sure we account for this
        let start = HASH_BUFFER_SIZE - MAX_TIME_MACRO_HAYSTACK_LEN / 2;
        buf[start..][..b"__DATE__".len()].copy_from_slice(b"__DATE__");
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(!finder.found_time());
        assert!(!finder.found_timestamp());
        assert!(finder.found_date());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_large_file_match_overlap_multiple_pages() {
        let mut buf = vec![0; HASH_BUFFER_SIZE * 3];
        // Make the patterns overlap buffer chunks twice to make sure we account for this
        let start = HASH_BUFFER_SIZE - MAX_TIME_MACRO_HAYSTACK_LEN / 2;
        buf[start..][..b"__TIME__".len()].copy_from_slice(b"__TIME__");
        let start = HASH_BUFFER_SIZE * 2 - MAX_TIME_MACRO_HAYSTACK_LEN / 2;
        buf[start..][..b"__DATE__".len()].copy_from_slice(b"__DATE__");
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(finder.found_time());
        assert!(!finder.found_timestamp());
        assert!(finder.found_date());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_large_file_match_overlap_multiple_pages_tiny() {
        drop(env_logger::try_init());
        let mut buf = vec![0; HASH_BUFFER_SIZE * 3];
        // Make the patterns overlap buffer chunks twice to make sure we account for this
        let start = HASH_BUFFER_SIZE - MAX_TIME_MACRO_HAYSTACK_LEN / 2;
        buf[start..][..b"__TIME__".len()].copy_from_slice(b"__TIME__");
        let start = HASH_BUFFER_SIZE * 2 - MAX_TIME_MACRO_HAYSTACK_LEN / 2;
        buf[start..][..b"__DATE__".len()].copy_from_slice(b"__DATE__");
        // Test overlap with the last chunk being less than the haystack
        buf.extend([0; MAX_TIME_MACRO_HAYSTACK_LEN / 2 + 1]);
        let start = HASH_BUFFER_SIZE * 3 - MAX_TIME_MACRO_HAYSTACK_LEN / 2;
        buf[start..][..b"__TIMESTAMP__".len()].copy_from_slice(b"__TIMESTAMP__");
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(finder.found_time_macros());
        assert!(finder.found_time());
        assert!(finder.found_timestamp());
        assert!(finder.found_date());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_find_time_macros_ghost_pattern() {
        // Check the (unlikely) case of a pattern being spread between the
        // start of a chunk and its end.
        let mut buf = vec![0; HASH_BUFFER_SIZE * 3];
        buf[HASH_BUFFER_SIZE..HASH_BUFFER_SIZE + b"__TI".len()].copy_from_slice(b"__TI");
        buf[HASH_BUFFER_SIZE * 2 - "ME__".len()..HASH_BUFFER_SIZE * 2].copy_from_slice(b"ME__");
        let buf = AllowStdIo::new(&buf[..]);
        let finder = Digest::from_reader_with_macro_finders(buf).await.unwrap().1;
        assert!(!finder.found_time_macros());
        assert!(!finder.found_time());
        assert!(!finder.found_timestamp());
        assert!(!finder.found_date());
    }
}
