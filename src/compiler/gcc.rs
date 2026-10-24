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
    compiler::{
        Cacheable, ColorMode, CompileCommandImpl, CompilerArguments, Language,
        SingleCompileCommand,
        args::*,
        c::{
            ArtifactDescriptor, CCompilerImpl, CCompilerKind, GenerateCompileCommandsArgs,
            GenerateDependenciesArgs, P1689r5_JSON, ParseArgs, ParsedArguments,
        },
        clang,
    },
    counted_array, dist,
    errors::*,
    mock_command::{CommandCreatorSync, RunCommand},
    util::{OsStrExt, bytes_to_str, run_input_output, run_input_output_timed},
};
use async_trait::async_trait;
use fs_err as fs;
use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet},
    env,
    ffi::{OsStr, OsString},
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

/// A struct on which to implement `CCompilerImpl`.
#[derive(Clone, Debug, Default)]
pub struct Gcc {
    pub gplusplus: bool,
    pub version: Option<String>,
    pub native_archs: Option<(String, String)>,
    pub specfiles: Vec<PathBuf>,
    pub assembler_digest: Option<String>,
    pragma_once_fix: Arc<PathBuf>,
}

impl Gcc {
    pub async fn new_from_compiler<
        C,
        Arg: AsRef<OsStr>,
        Args: Iterator<Item = Arg> + Clone,
        Exe: AsRef<Path>,
        Key: AsRef<OsStr>,
        Var: AsRef<OsStr>,
        Envs: Iterator<Item = (Key, Var)> + Clone,
    >(
        creator: &mut C,
        executable: Exe,
        args: Args,
        env_vars: Envs,
        gplusplus: bool,
        version: Option<String>,
        assembler_digest: Option<String>,
    ) -> Result<Self>
    where
        C: CommandCreatorSync,
    {
        #[cfg(unix)]
        let pragma_once_fix = Arc::new(
            crate::compiler::util::build_gcc_pragma_once_fix(
                executable.as_ref(),
                ([] as [&OsStr; 0]).into_iter(),
                env_vars.clone(),
            )
            .await
            .unwrap_or_else(|_| PathBuf::new()),
        );
        #[cfg(not(unix))]
        let pragma_once_fix = Arc::new(PathBuf::new());

        // Include gcc's implicit specfiles in the object hash
        let specfiles = Self::read_implicit_specfiles(
            creator,
            executable.as_ref(),
            args.clone(),
            env_vars.clone(),
            "-v",
        )
        .await?;

        let native_archs =
            Self::read_native_arch(creator, executable.as_ref(), env_vars.clone()).await;

        Ok(Self {
            assembler_digest,
            gplusplus,
            native_archs,
            specfiles,
            version,
            pragma_once_fix,
        })
    }

    pub async fn read_implicit_specfiles<
        C,
        Arg: AsRef<OsStr>,
        Args: Iterator<Item = Arg>,
        Exe: AsRef<Path>,
        Key: AsRef<OsStr>,
        Var: AsRef<OsStr>,
        Envs: Iterator<Item = (Key, Var)>,
    >(
        creator: &mut C,
        executable: Exe,
        args: Args,
        env_vars: Envs,
        v_flag: &str,
    ) -> Result<Vec<PathBuf>>
    where
        C: CommandCreatorSync,
    {
        use crate::compiler::write_temp_file;
        use bytes::Buf;
        use std::io::BufRead;

        let (_tempdir, testfile) = write_temp_file(
            &tokio::runtime::Handle::current(),
            "testfile.c".as_ref(),
            Default::default(),
        )
        .await?;

        let mut cmd = creator.new_command_sync(executable.as_ref());
        cmd.env_clear().envs(env_vars);
        for arg in args {
            cmd.arg(arg);
        }
        cmd.arg(v_flag).arg("-E").arg(&testfile);

        Ok(cmd
            .output()
            .await
            .unwrap_or_default()
            .stderr
            .reader()
            .lines()
            .filter_map_ok(|line| {
                line.split_once("Reading specs from ")
                    .map(|(_, path)| PathBuf::from(path.trim()))
            })
            .try_collect()
            .unwrap_or_default())
    }

    pub async fn read_native_arch<
        C,
        Exe: AsRef<Path>,
        Key: AsRef<OsStr>,
        Var: AsRef<OsStr>,
        Envs: Iterator<Item = (Key, Var)>,
    >(
        creator: &mut C,
        executable: Exe,
        env_vars: Envs,
    ) -> Option<(String, String)>
    where
        C: CommandCreatorSync,
    {
        use bytes::Buf;
        use itertools::Itertools;
        use std::io::BufRead;

        let mut cmd = creator.new_command_sync(executable.as_ref());
        cmd.env_clear()
            .envs(env_vars)
            .arg("-Q")
            .arg("-march=native")
            .arg("--help=target");

        let output = run_input_output(cmd, None).await.ok()?;

        output
            .stdout
            .reader()
            .lines()
            .map_while(|line| line.ok())
            .filter_map(|line| {
                let line = line.trim();
                if line.starts_with("-march=") || line.starts_with("-mtune=") {
                    line.split_once('=')
                        .map(|(_, arch)| arch.trim().to_owned())
                        .filter(|arch| !arch.is_empty())
                } else {
                    None
                }
            })
            .take(2)
            .collect_tuple()
    }
}

#[async_trait]
impl CCompilerImpl for Gcc {
    fn kind(&self) -> CCompilerKind {
        CCompilerKind::Gcc
    }
    fn plusplus(&self) -> bool {
        self.gplusplus
    }
    fn version(&self) -> Option<String> {
        self.version.clone()
    }
    fn extra_dist_files(&self) -> impl Iterator<Item = PathBuf> {
        self.specfiles.iter().cloned()
    }
    fn assembler_digest(&self) -> Option<&str> {
        self.assembler_digest.as_deref()
    }
    fn parse_arguments(
        &self,
        ParseArgs { arguments, cwd, .. }: ParseArgs<'_>,
    ) -> CompilerArguments<ParsedArguments> {
        let mut parsed_args =
            parse_arguments(arguments, cwd, &ARGS[..], self.gplusplus, self.kind());
        if let CompilerArguments::Ok(ref mut parsed_args) = parsed_args {
            // Rewrite -march=native to the current CPU architecture to
            // ensure the correct architecture is used if dist compiling
            if let Some((native_march, native_mtune)) = self.native_archs.as_ref() {
                for arch in parsed_args.arch_args.iter_mut() {
                    if arch == "-march=native" {
                        *arch = format!("-march={native_march}").into();
                    } else if arch == "-mtune=native" {
                        *arch = format!("-mtune={native_mtune}").into();
                    }
                }
            }
        }
        parsed_args
    }

    async fn generate_dependencies<T>(
        &self,
        GenerateDependenciesArgs {
            creator,
            executable,
            parsed_args,
            cwd,
            env_vars,
            tolerate_failure,
            ..
        }: GenerateDependenciesArgs<'_, T>,
    ) -> Result<(Vec<PathBuf>, Duration)>
    where
        T: CommandCreatorSync,
    {
        generate_dependencies(
            creator,
            executable,
            parsed_args,
            cwd,
            [("LD_PRELOAD".into(), "".into())]
                .iter()
                .chain(env_vars.iter())
                .cloned()
                .map(|(key, mut val)| {
                    //
                    if key == "LD_PRELOAD" {
                        if !val.is_empty() {
                            val.push(":");
                        }
                        if !self.pragma_once_fix.as_os_str().is_empty() {
                            val.push(self.pragma_once_fix.as_os_str());
                        }
                    }
                    (key, val)
                }),
            self.kind(),
            tolerate_failure,
        )
        .await
    }

    fn generate_compile_commands(
        &self,
        GenerateCompileCommandsArgs {
            executable,
            parsed_args,
            cwd,
            env_vars,
            ..
        }: GenerateCompileCommandsArgs<'_>,
    ) -> Result<(
        impl CompileCommandImpl,
        Option<dist::CompileCommand>,
        Cacheable,
    )> {
        generate_compile_commands(executable, parsed_args, cwd, env_vars, self.kind())
    }
}

ArgData! { pub
    TooHardFlag,
    TooHard(OsString),
    NotCompilationFlag,
    DiagnosticsColor(OsString),
    DiagnosticsColorFlag,
    NoDiagnosticsColorFlag,
    // Should only be necessary for -Xclang flags - unknown flags not hidden behind
    // that are assumed to not affect compilation
    PassThroughFlag,
    PassThrough(OsString),
    PassThroughPath(PathBuf),
    PreprocessorArgumentFlag,
    PreprocessorArgument(OsString),
    PreprocessorArgumentPath(PathBuf),
    NvhpcIncludeFile(PathBuf),
    NvhpcIncludePath(PathBuf),
    NvhpcSystemIncludePath(PathBuf),

    // Used for arguments that shouldn't affect the computed hash
    UnhashedFlag,
    Unhashed(OsString),
    DoCompilation,
    Output(PathBuf),
    NeedDepTarget,
    // Though you might think this should be a path as it's a Makefile target,
    // it's not treated as a path by the compiler - it's just written wholesale
    // (including any funny make syntax) into the dep file.
    DepTarget(OsString),
    DepArgumentPath(PathBuf),
    StructuredDepTarget(OsString),
    StructuredDepFormat(OsString),
    StructuredDepArgumentPath(PathBuf),
    Language(OsString),
    SplitDwarf,
    ProfileGenerate,
    ClangProfileUse(PathBuf),
    TestCoverage,
    Coverage,
    ModuleOnlyFlag,
    ExtraHashFile(PathBuf),
    // Only valid for clang, but this needs to be here since clang shares gcc's arg parsing.
    // For -fmodule-file which can be either "path" or "name=path"
    ExtraHashFileClangModuleFile(OsString),
    // For -fmodule-output
    ClangModuleOutput(OsString),
    XClang(OsString),
    Arch(OsString),
    PedanticFlag,
    Standard(OsString),
    SerializeDiagnostics(PathBuf),
    // C++20 modules
    EnableModules,
    ModuleMapper(OsString),
    // Only valid for clang, but this needs to be here since clang shares gcc's
    // arg parsing.
    IntegratedAs,
    NoIntegratedAs,
}

use self::ArgData::*;

// Mostly taken from https://github.com/ccache/ccache/blob/master/src/ccache/compopt.cpp#L52-L183
counted_array!(pub static ARGS: [ArgInfo<ArgData>; _] = [
    flag!("-", TooHardFlag),
    flag!("--coverage", Coverage),
    flag!("--help", NotCompilationFlag),
    take_arg!("--param", OsString, Separated, PassThrough),
    flag!("--save-temps", TooHardFlag),
    flag!("--save-temps=cwd", TooHardFlag),
    flag!("--save-temps=obj", TooHardFlag),
    take_arg!("--serialize-diagnostics", PathBuf, Separated, SerializeDiagnostics),
    take_arg!("--sysroot", PathBuf, Separated, PassThroughPath),
    flag!("--write-dependencies", NeedDepTarget),
    flag!("--write-user-dependencies", NeedDepTarget),
    take_arg!("-A", OsString, Separated, PassThrough),
    take_arg!("-B", PathBuf, CanBeSeparated, PassThroughPath),
    take_arg!("-D", OsString, CanBeSeparated, PreprocessorArgument),
    flag!("-E", DoCompilation), // TooHardFlag),
    take_arg!("-F", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-G", OsString, Separated, PassThrough),
    take_arg!("-I", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-L", OsString, Separated, PassThrough),
    flag!("-M", PassThroughFlag), // TooHardFlag),
    flag!("-MD", NeedDepTarget),
    take_arg!("-MF", PathBuf, Separated, DepArgumentPath),
    flag!("-MM", PassThroughFlag), // TooHardFlag),
    flag!("-MMD", NeedDepTarget),
    flag!("-MP", NeedDepTarget),
    take_arg!("-MQ", OsString, Separated, DepTarget),
    take_arg!("-MT", OsString, Separated, DepTarget),
    take_arg!("-O", OsString, Concatenated, PreprocessorArgument),
    flag!("-P", PassThroughFlag), // TooHardFlag),
    take_arg!("-U", OsString, CanBeSeparated, PassThrough),
    take_arg!("-V", OsString, Separated, PassThrough),
    flag!("-Werror=pedantic", PedanticFlag),
    take_arg!("-Wp", OsString, Concatenated(b','), PreprocessorArgument),
    flag!("-Wpedantic", PedanticFlag),
    take_arg!("-Xassembler", OsString, Separated, PassThrough),
    take_arg!("-Xlinker", OsString, Separated, PassThrough),
    take_arg!("-Xpreprocessor", OsString, Separated, PreprocessorArgument),
    take_arg!("-aux-info", OsString, Separated, PassThrough),
    take_arg!("-b", OsString, Separated, PassThrough),
    flag!("-c", DoCompilation),
    take_arg!("-fcallgraph-info", OsString, Concatenated(b'='), TooHard),
    take_arg!("-fdeps-file", PathBuf, Concatenated(b'='), StructuredDepArgumentPath),
    take_arg!("-fdeps-format", OsString, Concatenated(b'='), StructuredDepFormat),
    take_arg!("-fdeps-target", OsString, Concatenated(b'='), StructuredDepTarget),
    take_arg!("-fdiagnostics-color", OsString, Concatenated(b'='), DiagnosticsColor),
    take_arg!("-fmodule-mapper", OsString, Concatenated(b'='), ModuleMapper),
    flag!("-fmodules", TooHardFlag),
    flag!("-fmodules-ts", EnableModules),
    flag!("-fno-diagnostics-color", NoDiagnosticsColorFlag),
    flag!("-fno-profile-generate", TooHardFlag),
    flag!("-fno-profile-use", TooHardFlag),
    flag!("-fno-working-directory", PreprocessorArgumentFlag),
    take_arg!("-fplugin", OsString, Concatenated(b'='), TooHard),
    flag!("-fprofile-arcs", ProfileGenerate),
    flag!("-fprofile-generate", ProfileGenerate),
    take_arg!("-fprofile-use", OsString, Concatenated, TooHard),
    flag!("-frepo", TooHardFlag),
    flag!("-fstack-usage", TooHardFlag),
    flag!("-fsyntax-only", TooHardFlag),
    flag!("-ftest-coverage", TestCoverage),
    flag!("-fworking-directory", PreprocessorArgumentFlag),
    flag!("-gsplit-dwarf", SplitDwarf),
    take_arg!("-idirafter", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-iframework", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-imacros", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-imultilib", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-include", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-index-store-path", OsString, Separated, TooHard),
    take_arg!("-install_name", OsString, Separated, PassThrough),
    take_arg!("-iprefix", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-iquote", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-isysroot", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-isystem", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-ivfsstatcache", PathBuf, CanBeSeparated, PassThroughPath),
    take_arg!("-iwithprefix", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-iwithprefixbefore", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-march", OsString, Concatenated(b'='), Arch),
    take_arg!("-mtune", OsString, Concatenated(b'='), Arch),
    flag!("-nostdinc", PreprocessorArgumentFlag),
    flag!("-nostdinc++", PreprocessorArgumentFlag),
    take_arg!("-o", PathBuf, CanBeSeparated, Output),
    flag!("-pedantic", PedanticFlag),
    flag!("-pedantic-errors", PedanticFlag),
    flag!("-pipe", UnhashedFlag),
    flag!("-remap", PreprocessorArgumentFlag),
    flag!("-save-temps", TooHardFlag),
    flag!("-save-temps=cwd", TooHardFlag),
    flag!("-save-temps=obj", TooHardFlag),
    take_arg!("-specs", PathBuf, Concatenated(b'='), ExtraHashFile),
    take_arg!("-std", OsString, Concatenated(b'='), Standard),
    take_arg!("-stdlib", OsString, Concatenated(b'='), PreprocessorArgument),
    flag!("-trigraphs", PreprocessorArgumentFlag),
    take_arg!("-u", OsString, CanBeSeparated, PassThrough),
    take_arg!("-x", OsString, CanBeSeparated, Language),
    take_arg!("-z", OsString, CanBeSeparated, PassThrough),
    take_arg!("@", OsString, Concatenated, TooHard),
]);

/// Parse `arguments`, determining whether it is supported.
///
/// If any of the entries in `arguments` result in a compilation that
/// cannot be cached, return `CompilerArguments::CannotCache`.
/// If the commandline described by `arguments` is not compilation,
/// return `CompilerArguments::NotCompilation`.
/// Otherwise, return `CompilerArguments::Ok(ParsedArguments)`, with
/// the `ParsedArguments` struct containing information parsed from
/// `arguments`.
pub fn parse_arguments<S>(
    arguments: &[OsString],
    cwd: &Path,
    arg_info: S,
    plusplus: bool,
    kind: CCompilerKind,
) -> CompilerArguments<ParsedArguments>
where
    S: SearchableArgInfo<ArgData> + Clone,
{
    let mut output_arg = None;
    let mut module_output_path = None;
    let mut module_only_flag = false;
    let mut input_arg = None;
    let mut double_dash_input = false;
    let mut depfile = None;
    let mut structured_depfile = None;
    let mut structured_depfile_format = None;
    let mut dep_target = None;
    let mut structured_dep_target = None;
    let mut dep_flag = OsString::from("-MT");
    let mut structured_dep_flag = OsString::from("-fdeps-target");
    let mut cxx20_modules = false;
    let mut common_args = vec![];
    let mut arch_args = vec![];
    let mut unhashed_args = vec![];
    let mut preprocessor_args = vec![];
    let mut dependency_args = vec![];
    let mut extra_dist_files = vec![];
    let mut extra_hash_files = vec![];
    let mut compilation = kind == CCompilerKind::Nvcc;
    let mut multiple_input = false;
    let mut multiple_input_files = Vec::new();
    let mut split_dwarf = false;
    let mut need_explicit_dep_target = false;
    let mut need_explicit_structured_dep_target = false;
    let mut need_explicit_structured_dep_format = true;
    enum DepArgumentRequirePath {
        NotNeeded,
        Missing,
        Provided,
    }
    let mut need_explicit_dep_argument_path = DepArgumentRequirePath::NotNeeded;
    let mut need_explicit_structured_dep_argument_path = DepArgumentRequirePath::NotNeeded;
    let mut language = None;
    // Clang assembles in-process unless told otherwise; the other compilers
    // parsed here always hand off to a separate assembler.
    let mut integrated_assembler = kind == CCompilerKind::Clang;
    let mut compilation_flag = OsString::new();
    let mut profile_generate = false;
    let mut outputs_gcno = false;
    let mut xclangs: Vec<OsString> = vec![];
    let mut color_mode = ColorMode::Auto;
    let mut seen_arch = None;
    let mut serialize_diagnostics = None;
    let dont_cache_multiarch = env::var("SCCACHE_CACHE_MULTIARCH").is_err();

    // Custom iterator to expand `@` arguments which stand for reading a file
    // and interpreting it as a list of more arguments.
    let it = ExpandIncludeFile::new(cwd, arguments);

    let mut args_iter = ArgsIter::new(it, arg_info.clone());
    if kind == CCompilerKind::Clang {
        args_iter = args_iter.with_double_dashes();
    }
    for arg in args_iter {
        let arg = try_or_cannot_cache!(arg, "argument parse");
        // Check if the value part of this argument begins with '@'. If so, we either
        // failed to expand it, or it was a concatenated argument - either way, bail.
        // We refuse to cache concatenated arguments (like "-include@foo") because they're a
        // mess. See https://github.com/mozilla/sccache/issues/150#issuecomment-318586953
        match arg {
            Argument::WithValue(_, ref v, ArgDisposition::Separated)
            | Argument::WithValue(_, ref v, ArgDisposition::CanBeConcatenated(_))
            | Argument::WithValue(_, ref v, ArgDisposition::CanBeSeparated(_)) => {
                if v.clone().into_arg_os_string().starts_with("@") {
                    cannot_cache!("@");
                }
            }
            // Empirically, concatenated arguments appear not to interpret '@' as
            // an include directive, so just continue.
            Argument::WithValue(_, _, ArgDisposition::Concatenated(_))
            | Argument::Raw(_)
            | Argument::UnknownFlag(_)
            | Argument::Flag(_, _) => {}
        }

        match arg.get_data() {
            Some(NotCompilationFlag) => return CompilerArguments::NotCompilation,
            Some(TooHardFlag) | Some(TooHard(_)) => {
                cannot_cache!(arg.flag_str().expect("Can't be Argument::Raw/UnknownFlag",))
            }
            Some(ModuleOnlyFlag) => module_only_flag = true,
            Some(SplitDwarf) => split_dwarf = true,
            Some(DoCompilation) => {
                compilation = true;
                compilation_flag =
                    OsString::from(arg.flag_str().expect("Compilation flag expected"));
            }
            Some(EnableModules) => cxx20_modules = true,
            Some(ProfileGenerate) => profile_generate = true,
            Some(ClangProfileUse(path)) => {
                let path = clang::resolve_profile_use_path(path, cwd);
                extra_dist_files.push(path.clone());
                extra_hash_files.push(path);
            }
            Some(TestCoverage) => outputs_gcno = true,
            Some(Coverage) => {
                outputs_gcno = true;
                profile_generate = true;
            }
            Some(IntegratedAs) => integrated_assembler = true,
            Some(NoIntegratedAs) => integrated_assembler = false,
            Some(DiagnosticsColorFlag) => color_mode = ColorMode::On,
            Some(NoDiagnosticsColorFlag) => color_mode = ColorMode::Off,
            Some(DiagnosticsColor(value)) => {
                color_mode = match value.to_str().unwrap_or("auto") {
                    "" | "always" => ColorMode::On,
                    "never" => ColorMode::Off,
                    _ => ColorMode::Auto,
                };
            }
            Some(Output(p)) => output_arg = Some(p.clone()),
            Some(ClangModuleOutput(p)) => {
                if let Some(p) = p.strip_prefix("=") {
                    module_output_path = Some(Some(PathBuf::from(p)));
                } else if p.is_empty() {
                    module_output_path = Some(None);
                } else {
                    cannot_cache!(
                        "unknown module output format",
                        p.to_string_lossy().into_owned()
                    );
                }
            }
            Some(NeedDepTarget) => {
                need_explicit_dep_target = true;
                if let DepArgumentRequirePath::NotNeeded = need_explicit_dep_argument_path {
                    need_explicit_dep_argument_path = DepArgumentRequirePath::Missing;
                }
            }
            Some(DepTarget(s)) => {
                dep_flag = OsString::from(arg.flag_str().expect("Dep target flag expected"));
                dep_target = Some(s.clone());
            }
            Some(StructuredDepTarget(s)) => {
                structured_dep_flag =
                    OsString::from(arg.flag_str().expect("Structured dep target flag expected"));
                structured_dep_target = Some(s.clone());
            }
            Some(DepArgumentPath(path)) => {
                depfile = Some(path.clone());
                need_explicit_dep_argument_path = DepArgumentRequirePath::Provided;
            }
            Some(StructuredDepArgumentPath(path)) => {
                structured_depfile = Some(path.clone());
                // https://gcc.gnu.org/onlinedocs/gcc/C-Dialect-Options.html#index-fdeps-
                structured_depfile_format =
                    structured_depfile_format.or(Some(OsString::from("p1689r5")));
                need_explicit_structured_dep_argument_path = DepArgumentRequirePath::Provided;
            }
            Some(StructuredDepFormat(format)) => {
                need_explicit_structured_dep_target = true;
                need_explicit_structured_dep_format = false;
                if let DepArgumentRequirePath::NotNeeded =
                    need_explicit_structured_dep_argument_path
                {
                    need_explicit_structured_dep_argument_path = DepArgumentRequirePath::Missing;
                }
                structured_depfile_format = Some(format.clone());
            }
            Some(SerializeDiagnostics(path)) => {
                serialize_diagnostics = Some(path.clone());
            }
            Some(ExtraHashFile(_))
            | Some(PedanticFlag)
            | Some(Standard(_))
            | Some(ExtraHashFileClangModuleFile(_))
            | Some(PassThroughFlag)
            | Some(NvhpcIncludeFile(_))
            | Some(NvhpcIncludePath(_))
            | Some(NvhpcSystemIncludePath(_))
            | Some(PreprocessorArgumentFlag)
            | Some(PreprocessorArgument(_))
            | Some(PreprocessorArgumentPath(_))
            | Some(PassThrough(_))
            | Some(PassThroughPath(_))
            | Some(UnhashedFlag)
            | Some(Unhashed(_))
            | Some(ModuleMapper(_)) => {}
            Some(Language(lang)) => {
                language = match lang
                    .to_str()
                    .and_then(|lang| Language::from_compiler_str(kind.into(), lang))
                {
                    Some(language) => Some(language),
                    None => cannot_cache!("-x", lang.to_string_lossy().to_string()),
                };
            }
            Some(Arch(arch)) => {
                if kind == CCompilerKind::Clang {
                    match seen_arch {
                        Some(s) if &s != arch && dont_cache_multiarch => {
                            cannot_cache!(
                                "multiple different -arch, and SCCACHE_CACHE_MULTIARCH not set"
                            )
                        }
                        _ => {}
                    }
                    seen_arch = Some(arch.clone());
                }
            }
            Some(XClang(s)) => xclangs.push(s.clone()),
            None => match arg {
                Argument::Raw(ref val) if val == "--" => {
                    if input_arg.is_none() {
                        double_dash_input = true;
                    }
                }
                Argument::Raw(ref val) => {
                    if input_arg.is_some() {
                        multiple_input = true;
                        multiple_input_files.push(val.clone());
                    }
                    input_arg = Some(val.clone());
                }
                Argument::UnknownFlag(_) => {}
                _ => unreachable!(),
            },
        }
        let args = match arg.get_data() {
            Some(Standard(_)) => &mut preprocessor_args,
            Some(SplitDwarf)
            | Some(ModuleOnlyFlag)
            | Some(PedanticFlag)
            | Some(ProfileGenerate)
            | Some(ClangProfileUse(_))
            | Some(TestCoverage)
            | Some(Coverage)
            | Some(NoDiagnosticsColorFlag)
            | Some(PassThroughFlag)
            | Some(PassThrough(_))
            | Some(IntegratedAs)
            | Some(NoIntegratedAs)
            | Some(PassThroughPath(_))
            | Some(EnableModules)
            | Some(ModuleMapper(_))
            | Some(ClangModuleOutput(_)) => &mut common_args,
            Some(UnhashedFlag) | Some(Unhashed(_)) => &mut unhashed_args,
            Some(Arch(_)) => &mut arch_args,
            Some(ExtraHashFile(path)) => {
                extra_dist_files.push(cwd.join(path));
                extra_hash_files.push(cwd.join(path));
                &mut common_args
            }
            Some(ExtraHashFileClangModuleFile(val)) => {
                // -fmodule-file can be either "path" or "name=path"
                let val_str = val.to_string_lossy();
                let path = if let Some(idx) = val_str.find('=') {
                    cwd.join(&val_str[idx + 1..])
                } else {
                    cwd.join(val)
                };
                extra_dist_files.push(path.clone());
                extra_hash_files.push(path);
                &mut common_args
            }
            Some(PreprocessorArgument(val)) => {
                let flag = arg.flag_str().unwrap_or_default();
                if matches!(flag, "-Xpreprocessor" | "-Wp") {
                    // Parse out the dependency file and target names from
                    // `-Wp,-MD`, `-Wp,-MT`, and `-Wp,-MF`.
                    // Example:
                    //  * `-Wp,-MD,CMakeFiles/foo.dir/foo.cpp.o.d`
                    //  * `-Wp,-MT,CMakeFiles/foo.dir/foo.cpp.o`
                    //  * `-Wp,-MF,CMakeFiles/foo.dir/foo.cpp.o.d`
                    let mut preprocessor_arg = val.split(",");
                    let preprocessor_arg_arg = preprocessor_arg.next().and_then(|s| s.to_str());
                    let preprocessor_arg_val = preprocessor_arg.next();
                    match (preprocessor_arg_arg, preprocessor_arg_val) {
                        (Some("-MD"), Some(s)) |
                        (Some("-MF"), Some(s)) |
                        (Some("--write-dependencies"), Some(s)) => {
                            depfile = Some(s.into());
                            need_explicit_dep_argument_path = DepArgumentRequirePath::Provided;
                            &mut dependency_args
                        }
                        (Some("-MT"), Some(s)) |
                        (Some("-MQ"), Some(s)) => {
                            need_explicit_dep_target = false;
                            dep_target = Some(s.into());
                            &mut dependency_args
                        }
                        (Some(arg), _) => {
                            if let Some(arg) = arg_info.search(arg) {
                                match arg {
                                    ArgInfo::Flag(_, data) => {
                                        if matches!(data, TooHardFlag) {
                                            let msg = format!("Arg '{flag},{}' is too hard", val.display());
                                            warn!("{msg}");
                                            cannot_cache!(flag, msg)
                                        }
                                    }
                                    ArgInfo::TakeArg(_, parse, _) => {
                                        match parse(val.clone()) {
                                            Ok(TooHard(..)) => {
                                                let msg = format!("Arg '{flag},{}' is too hard", val.display());
                                                warn!("{msg}");
                                                cannot_cache!(flag, msg);
                                            },
                                            Err(ArgParseError::Other(err)) => {
                                                let msg = format!("Error parsing arg '{flag},{}': {err:#}", val.display());
                                                error!("{msg}");
                                                cannot_cache!(flag, msg);
                                            },
                                            Err(err) => {
                                                let msg = format!("Error parsing arg '{flag},{}': {err:#}", val.display());
                                                error!("{msg}");
                                                cannot_cache!(flag, msg)
                                            },
                                            _ => {},
                                        }
                                    }
                                }
                            }
                            &mut preprocessor_args
                        },
                        _ => {
                            &mut preprocessor_args
                        }
                    }
                } else {
                    &mut preprocessor_args
                }
            }
            Some(PreprocessorArgumentFlag) => {
                &mut preprocessor_args
            }
            Some(PreprocessorArgumentPath(path)) => {
                extra_dist_files.push(cwd.join(path));
                &mut preprocessor_args
            }
            Some(NvhpcIncludeFile(path)) => {
                extra_dist_files.push(path.clone());
                preprocessor_args.push("-include".into());
                preprocessor_args.push(path.into());
                continue;
            }
            Some(NvhpcIncludePath(path)) => {
                extra_dist_files.push(path.clone());
                preprocessor_args.push("-I".into());
                preprocessor_args.push(path.into());
                continue;
            }
            Some(NvhpcSystemIncludePath(path)) => {
                extra_dist_files.push(path.clone());
                preprocessor_args.push("-isystem".into());
                preprocessor_args.push(path.into());
                continue;
            }
            Some(DepArgumentPath(_))
            | Some(NeedDepTarget)
            | Some(StructuredDepFormat(_))
            | Some(StructuredDepArgumentPath(_)) => &mut dependency_args,
            Some(DoCompilation)
            // Skip `-fdiagnostics-color[=auto]` flags and handle in `generate_compile_commands`
            | Some(DiagnosticsColor(_))
            | Some(DiagnosticsColorFlag)
            | Some(Language(_))
            | Some(Output(_))
            | Some(XClang(_))
            | Some(DepTarget(_))
            | Some(StructuredDepTarget(_))
            | Some(SerializeDiagnostics(_)) => continue,
            Some(NotCompilationFlag) | Some(TooHardFlag) | Some(TooHard(_)) => unreachable!(),
            None => match arg {
                Argument::Raw(_) => continue,
                Argument::UnknownFlag(_) => &mut common_args,
                _ => unreachable!(),
            },
        };

        // Normalize attributes such as "-I foo", "-D FOO=bar", as
        // "-Ifoo", "-DFOO=bar", etc. and "-includefoo", "idirafterbar" as
        // "-include foo", "-idirafter bar", etc.
        let norm = match arg.flag_str() {
            Some(s) if s.len() == 2 => NormalizedDisposition::Concatenated,
            _ => NormalizedDisposition::Separated,
        };

        args.extend(arg.normalize(norm).iter_os_strings());
    }

    let xclang_it = ExpandIncludeFile::new(cwd, &xclangs);
    let mut follows_plugin_arg = false;
    for arg in ArgsIter::new(xclang_it, (&ARGS[..], &clang::ARGS[..])) {
        let arg = try_or_cannot_cache!(arg, "argument parse");
        let args = match arg.get_data() {
            Some(SplitDwarf)
            | Some(NvhpcIncludeFile(_))
            | Some(NvhpcIncludePath(_))
            | Some(NvhpcSystemIncludePath(_))
            | Some(ModuleOnlyFlag)
            | Some(PedanticFlag)
            | Some(Standard(_))
            | Some(ProfileGenerate)
            | Some(ClangProfileUse(_))
            | Some(TestCoverage)
            | Some(Coverage)
            | Some(DoCompilation)
            | Some(Language(_))
            | Some(Output(_))
            | Some(ClangModuleOutput(_))
            | Some(TooHardFlag)
            | Some(XClang(_))
            | Some(IntegratedAs)
            | Some(NoIntegratedAs)
            | Some(TooHard(_)) => cannot_cache!(
                arg.flag_str()
                    .unwrap_or("Can't handle complex arguments through clang",)
            ),
            Some(NotCompilationFlag) | None => match arg {
                Argument::Raw(_) if follows_plugin_arg => &mut common_args,
                Argument::Raw(flag) => cannot_cache!(
                    "Can't handle Raw arguments with -Xclang",
                    flag.to_str().unwrap_or("").to_string()
                ),
                Argument::UnknownFlag(flag) => {
                    cannot_cache!(
                        "Can't handle UnknownFlag arguments with -Xclang",
                        flag.to_str().unwrap_or("").to_string()
                    )
                }
                _ => unreachable!(),
            },
            Some(DiagnosticsColor(_))
            | Some(DiagnosticsColorFlag)
            | Some(NoDiagnosticsColorFlag)
            | Some(Arch(_))
            | Some(PassThrough(_))
            | Some(PassThroughFlag)
            | Some(PassThroughPath(_))
            | Some(SerializeDiagnostics(_))
            | Some(EnableModules)
            | Some(ModuleMapper(_)) => &mut common_args,
            Some(UnhashedFlag) | Some(Unhashed(_)) => &mut unhashed_args,
            Some(ExtraHashFile(path)) => {
                extra_dist_files.push(cwd.join(path));
                extra_hash_files.push(cwd.join(path));
                &mut common_args
            }
            Some(ExtraHashFileClangModuleFile(val)) => {
                // -fmodule-file can be either "path" or "name=path"
                let val_str = val.to_string_lossy();
                let path = if let Some(idx) = val_str.find('=') {
                    cwd.join(&val_str[idx + 1..])
                } else {
                    cwd.join(val)
                };
                extra_dist_files.push(path.clone());
                extra_hash_files.push(path);
                &mut common_args
            }
            Some(PreprocessorArgumentFlag) | Some(PreprocessorArgument(_)) => {
                &mut preprocessor_args
            }
            Some(PreprocessorArgumentPath(path)) => {
                extra_dist_files.push(cwd.join(path));
                &mut preprocessor_args
            }
            Some(DepArgumentPath(_))
            | Some(StructuredDepFormat(_))
            | Some(StructuredDepTarget(_))
            | Some(StructuredDepArgumentPath(_))
            | Some(DepTarget(_))
            | Some(NeedDepTarget) => &mut dependency_args,
        };
        follows_plugin_arg = match arg.flag_str() {
            Some(s) => s == "-plugin-arg",
            _ => false,
        };

        // Normalize attributes such as "-I foo", "-D FOO=bar", as
        // "-Ifoo", "-DFOO=bar", etc. and "-includefoo", "idirafterbar" as
        // "-include foo", "-idirafter bar", etc.
        let norm = match arg.flag_str() {
            Some(s) if s.len() == 2 => NormalizedDisposition::Concatenated,
            _ => NormalizedDisposition::Separated,
        };
        for arg in arg.normalize(norm).iter_os_strings() {
            args.push("-Xclang".into());
            args.push(arg);
        }
    }

    // We only support compilation.
    if !compilation {
        return CompilerArguments::NotCompilation;
    }

    // Can't cache compilations with multiple inputs.
    if multiple_input {
        cannot_cache!(
            "multiple input files",
            format!("{:?}", multiple_input_files)
        );
    }
    let input = match input_arg {
        Some(i) => i,
        // We can't cache compilation without an input.
        None => cannot_cache!("no input file"),
    };

    let language = match language {
        None => {
            let mut lang = Language::from_file_name(Path::new(&input));
            if let (Some(Language::C), true) = (lang, plusplus) {
                lang = Some(Language::Cxx);
            }
            lang
        }
        language => language,
    };
    let language = match language {
        Some(language) => language,
        None => cannot_cache!("unknown source language"),
    };
    let mut outputs = HashMap::new();
    let output = match output_arg {
        // We can't cache compilation that doesn't go to a file
        None => PathBuf::from(Path::new(&input).with_extension("o").file_name().unwrap()),
        Some(o) => o,
    };
    if split_dwarf {
        let dwo = output.with_extension("dwo");
        common_args.push(OsString::from(
            "-D_gsplit_dwarf_path=".to_owned() + dwo.to_str().unwrap(),
        ));
        // -gsplit-dwarf doesn't guarantee .dwo file if no -g is specified
        outputs.insert(
            "dwo",
            ArtifactDescriptor {
                path: dwo,
                optional: true,
            },
        );
    }
    if outputs_gcno {
        let gcno = output.with_extension("gcno");
        outputs.insert(
            "gcno",
            ArtifactDescriptor {
                path: gcno,
                optional: false,
            },
        );
        profile_generate = true;
    }

    // If the language doesn't need preprocessing, it doesn't generate a dependency file. See issue #2664
    if language.needs_c_preprocessing() {
        if need_explicit_dep_target {
            let dep_target = dep_target.unwrap_or_else(|| output.clone().into_os_string());
            dependency_args.push(dep_flag);
            dependency_args.push(dep_target);
        }

        if let DepArgumentRequirePath::Missing = need_explicit_dep_argument_path {
            let dep_path = Path::new(&output).with_extension("d");
            dependency_args.push(OsString::from("-MF"));
            dependency_args.push(dep_path.as_os_str().into());
            depfile = Some(dep_path);
        }

        if let Some(p) = depfile.as_ref() {
            outputs.insert(
                "dep",
                ArtifactDescriptor {
                    path: p.clone(),
                    optional: false,
                },
            );
        }

        if need_explicit_structured_dep_target {
            let dep_target =
                structured_dep_target.unwrap_or_else(|| output.clone().into_os_string());
            let mut arg = structured_dep_flag;
            arg.push("=");
            arg.push(dep_target);
            dependency_args.push(arg);
        }

        if let DepArgumentRequirePath::Missing = need_explicit_structured_dep_argument_path {
            let dep_path = Path::new(&output).with_extension("ddi");
            let mut arg = OsString::from("-fdeps-file=");
            arg.push(dep_path.as_os_str());
            dependency_args.push(arg);
            structured_depfile = Some(dep_path);
        }

        if let Some(p) = structured_depfile.as_ref() {
            outputs.insert(
                "dep:modules",
                ArtifactDescriptor {
                    path: p.clone(),
                    optional: false,
                },
            );
        }
    }

    if let Some(path) = serialize_diagnostics {
        outputs.insert(
            "dia",
            ArtifactDescriptor {
                path: path.clone(),
                optional: false,
            },
        );
    }

    if !module_only_flag {
        outputs.extend(module_output_path.into_iter().map(|p| {
            (
                "module",
                module_artifact_descriptor(Path::new(&input), &output, p.as_deref()),
            )
        }));
    }

    // Sometimes if this is precompiled this will be a module file,
    // however we call it an object because there MUST be an object file output.
    outputs.insert(
        "obj",
        ArtifactDescriptor {
            path: output,
            optional: false,
        },
    );

    let structured_depfile_format = if structured_depfile_format.is_none() {
        if structured_depfile.is_some() {
            Some(OsString::from("p1689r5"))
        } else {
            None
        }
    } else {
        structured_depfile_format
    };

    if need_explicit_structured_dep_format
        && let Some(format) = structured_depfile_format.as_deref()
    {
        let mut arg = OsString::from("-fdeps-format=");
        arg.push(format);
        dependency_args.push(arg);
    }

    CompilerArguments::Ok(ParsedArguments {
        input: input.into(),
        double_dash_input,
        language,
        compilation_flag,
        cxx20_modules,
        outputs,
        depfile,
        structured_depfile_format,
        structured_depfile,
        dependency_args,
        preprocessor_args,
        common_args,
        arch_args,
        unhashed_args,
        extra_dist_files,
        extra_hash_files,
        uses_external_assembler: !integrated_assembler,
        profile_generate,
        color_mode,
        ..Default::default()
    })
}

fn module_artifact_descriptor(
    input: &Path,
    output: &Path,
    module_output_path: Option<&Path>,
) -> ArtifactDescriptor {
    let empty_os_string = OsString::new();

    let path = module_output_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| {
            let input_file_name = input.file_name().unwrap_or(&empty_os_string);
            let mut path = output.with_file_name(input_file_name);
            let mut ext = path
                .extension()
                .map(|e| e.to_os_string())
                .unwrap_or_default();
            ext.push(".pcm");
            path.set_extension(ext);
            path
        });

    ArtifactDescriptor {
        path,
        optional: false,
    }
}

pub async fn generate_dependencies<
    T,
    Cwd: AsRef<Path>,
    Exe: AsRef<Path>,
    Key: AsRef<OsStr>,
    Var: AsRef<OsStr>,
    Envs: Iterator<Item = (Key, Var)> + Clone,
>(
    creator: &T,
    executable: Exe,
    parsed_args: &ParsedArguments,
    cwd: Cwd,
    env_vars: Envs,
    kind: CCompilerKind,
    tolerate_failure: bool,
) -> Result<(Vec<PathBuf>, Duration)>
where
    T: CommandCreatorSync,
{
    let mut elapsed = Duration::default();

    let generate_user_dependencies = parsed_args.dependency_args.iter().any(|arg| {
        arg.contains("-MMD")
            || arg.contains("--write-user-dependencies")
            || arg.contains("--generate-nonsystem-dependencies-with-compile")
    });

    let args_without_dependency_flags = {
        let mut iter = std::iter::empty()
            .chain(
                // Set the langauge flag
                parsed_args
                    .language
                    .as_compiler_str(kind.into())
                    .into_iter()
                    .flat_map(|lang| ["-x", lang])
                    .map(OsStr::new),
            )
            .chain(
                std::iter::empty()
                    .chain(parsed_args.preprocessor_args.iter())
                    .chain(parsed_args.arch_args.iter())
                    .chain(parsed_args.common_args.iter())
                    .map(|arg| arg.as_os_str()),
            )
            .filter(|&arg| arg != parsed_args.compilation_flag);

        let mut args_without_dependency_flags = vec![];

        // Filter out verbose flags (both gcc and nvcc) since they write to stdout on Windows
        while let Some(arg) = iter.next() {
            let next_arg = if arg == "-Xcompiler" {
                iter.next()
            } else {
                None
            };

            let maybe_verbose = next_arg.or(arg.strip_prefix("-Wp,")).unwrap_or(arg);
            if ["-v", "-v2", "--verbose", "--verbose2"]
                .into_iter()
                .any(|v| {
                    maybe_verbose == v
                        || maybe_verbose.contains(format!(",{v}"))
                        || maybe_verbose.contains(format!("{v},"))
                })
            {
                continue;
            } else {
                args_without_dependency_flags.extend(std::iter::once(arg).chain(next_arg));
            }
        }

        args_without_dependency_flags
    };

    // If `-MMD -MF <file>`, write that depfile out now. We need to return
    // dependencies with system includes to compute the hash key, but the
    // user requested dependencies *without* system includes be written
    // to `parsed_args.depfile`
    if generate_user_dependencies && parsed_args.depfile.is_some() {
        let args = args_without_dependency_flags
            .clone()
            .into_iter()
            .chain(
                std::iter::once(OsStr::new("-MM"))
                    .chain(parsed_args.dependency_args.iter().map(|arg| {
                        // Map all depflags to `-MM`
                        if arg.ends_with("-M")
                            || arg.contains("-MD")
                            || arg.contains("-MMD")
                            || arg.contains("--dependencies")
                            || arg.contains("--write-dependencies")
                            || arg.contains("--write-user-dependencies")
                            || arg.contains("--generate-dependencies-with-compile")
                            || arg.contains("--generate-nonsystem-dependencies-with-compile")
                        {
                            OsStr::new("-MM")
                        } else {
                            arg.as_os_str()
                        }
                    }))
                    .unique(),
            )
            .chain(parsed_args.double_dash_input.then_some(OsStr::new("--")))
            .chain(std::iter::once(parsed_args.input.as_os_str()));

        let mut cmd = creator.clone().new_command_sync(executable.as_ref());

        cmd.current_dir(cwd.as_ref())
            .env_clear()
            .envs(env_vars.clone());

        for arg in args {
            cmd.arg(arg);
        }

        trace!(
            "[{}]: user dependencies: {cmd}",
            parsed_args.output_pretty(),
        );

        let (_, duration) =
            run_input_output_timed(cmd, None)
                .await
                .or_else(|(err, duration)| match err.downcast::<ProcessError>() {
                    Ok(ProcessError(out)) => {
                        if tolerate_failure {
                            Ok((out, duration))
                        } else {
                            Err(ProcessError(out).into())
                        }
                    }
                    Err(err) => Err(err),
                })?;
        elapsed += duration;
    }

    let generate_module_dependencies = parsed_args.cxx20_modules
        && parsed_args.structured_depfile_format.is_some()
        && parsed_args.structured_depfile.is_some();

    let depfile = parsed_args
        .depfile
        .clone()
        .filter(|_| !generate_user_dependencies);

    let structured_depfile_format = parsed_args
        .structured_depfile_format
        .as_deref()
        .and_then(|s| s.to_str());

    let (deps_arg, depfile) = if matches!(kind, CCompilerKind::Nvhpc)
        || matches!(structured_depfile_format, Some("p1689r5"))
    {
        (
            OsStr::new("-MD"),
            depfile.or_else(|| {
                Some({
                    match parsed_args.outputs.get("obj") {
                        Some(obj) => obj.path.with_added_extension("d"),
                        None => parsed_args.input.with_added_extension("d"),
                    }
                })
            }),
        )
    } else {
        (OsStr::new("-M"), depfile)
    };

    let cmd = {
        let args = args_without_dependency_flags
            .into_iter()
            .chain(
                std::iter::once(deps_arg)
                    .chain(
                        parsed_args
                            .dependency_args
                            .iter()
                            // If this is generating the real dependency file,
                            // pass all the user's other dependency flags.
                            .filter(|_| depfile.is_some())
                            .map(|arg| {
                                // Map all depflags to `-M`
                                if arg.ends_with("-MM")
                                    || arg.contains("-MD")
                                    || arg.contains("-MMD")
                                    || arg.contains("--dependencies")
                                    || arg.contains("--write-dependencies")
                                    || arg.contains("--write-user-dependencies")
                                    || arg.contains("--generate-dependencies-with-compile")
                                    || arg
                                        .contains("--generate-nonsystem-dependencies-with-compile")
                                {
                                    deps_arg
                                } else {
                                    arg.as_os_str()
                                }
                            }),
                    )
                    .unique(),
            )
            .chain(
                (deps_arg == "-MD")
                    .then(|| {
                        [
                            OsStr::new("-E"),
                            OsStr::new("-P"),
                            OsStr::new("-o"),
                            if cfg!(windows) {
                                OsStr::new("nul")
                            } else {
                                OsStr::new("/dev/null")
                            },
                        ]
                    })
                    .into_iter()
                    .flatten(),
            )
            .chain(parsed_args.double_dash_input.then_some(OsStr::new("--")))
            .chain(std::iter::once(parsed_args.input.as_os_str()));

        let mut cmd = creator.clone().new_command_sync(executable.as_ref());

        cmd.current_dir(cwd.as_ref())
            .env_clear()
            .envs(env_vars.clone());

        for arg in args {
            cmd.arg(arg);
        }

        cmd
    };

    trace!("[{}]: dependencies: {cmd}", parsed_args.output_pretty());

    let (output, duration) =
        run_input_output_timed(cmd, None)
            .await
            .or_else(|(err, duration)| match err.downcast::<ProcessError>() {
                Ok(ProcessError(out)) => {
                    if tolerate_failure {
                        Ok((out, duration))
                    } else {
                        Err(ProcessError(out).into())
                    }
                }
                Err(err) => Err(err),
            })?;

    elapsed += duration;

    let start = Instant::now();
    let dependencies = [
        if generate_module_dependencies
            && let Some(depfile) = parsed_args.structured_depfile.as_deref()
        {
            parse_dependencies(
                &parsed_args.input,
                parsed_args.structured_depfile_format.as_deref(),
                &read_depfile(cwd.as_ref().join(depfile)).await?,
            )?
        } else {
            vec![]
        },
        parse_dependencies(
            &parsed_args.input,
            None,
            &if let Some(depfile) = depfile {
                read_depfile(cwd.as_ref().join(depfile)).await?
            } else {
                bytes_to_str(output.stdout)?
            },
        )?,
    ]
    .concat()
    .into_iter()
    .unique()
    .collect();

    elapsed += start.elapsed();

    Ok((dependencies, elapsed))
}

pub async fn read_depfile<'a, P: AsRef<Path>>(depfile: P) -> Result<std::borrow::Cow<'a, str>> {
    let depfile = depfile.as_ref();
    tokio::fs::read(depfile)
        .await
        .with_context(|| format!("{}", depfile.display()))
        // Avoid dropping Windows wide chars in paths
        .and_then(|lines| Ok(bytes_to_str(lines)?))
}

pub fn parse_dependencies<P: AsRef<Path>, S: AsRef<str>>(
    input: P,
    format: Option<&OsStr>,
    depfile: S,
) -> Result<Vec<PathBuf>> {
    if format.unwrap_or_default() == "p1689r5" {
        parse_p1689r5_dependenices(depfile)
    } else {
        parse_makefile_dependencies(input, depfile)
    }
}

pub fn parse_p1689r5_dependenices<S: AsRef<str>>(depfile: S) -> Result<Vec<PathBuf>> {
    serde_json::from_str::<P1689r5_JSON>(depfile.as_ref())
        .with_context(|| "Failed to parse p1689r5 json")
        .map(|json| {
            json.rules
                .iter()
                .flat_map(|rule| {
                    rule.requires
                        .iter()
                        .filter_map(|dep| {
                            if dep.source_path.is_empty() {
                                None
                            } else {
                                Some(PathBuf::from(&dep.source_path))
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        })
}

pub fn parse_makefile_dependencies<P: AsRef<Path>, S: AsRef<str>>(
    input: P,
    depfile_lines: S,
) -> Result<Vec<PathBuf>> {
    // trace!("lines:\n{}", depfile_lines.as_ref());

    let input_path = input.as_ref();

    let lines = depfile_lines
        .as_ref()
        .split("\n")
        // The path to the left of the colon is the `-MT <path>`, so take everything to the right.
        .filter_map(|line| {
            let line = if let Some((_, rest)) = line.split_once(":") {
                rest
            } else {
                line
            }
            // Trim whitespace and trailing backslashes + colons
            .trim()
            .trim_end_matches(r"\")
            .trim_end_matches(r":")
            .trim();

            (!line.is_empty()).then_some(line)
        });

    let paths = lines
        // .inspect(|line| trace!("line: {line}"))
        .flat_map(|line| shlex::split(line).unwrap_or_default())
        .map(PathBuf::from);

    let paths = paths
        .sorted()
        .unique()
        // Skip the input file path
        .filter_map(|path| {
            // Compare to the unmodified input file path
            if path == input_path {
                return None;
            }
            Some(path)
        })
        .collect::<Vec<_>>();

    Ok(paths)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_compile_commands(
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
    kind: CCompilerKind,
) -> Result<(
    SingleCompileCommand,
    Option<dist::CompileCommand>,
    Cacheable,
)> {
    let out_pretty = parsed_args.output_pretty();

    let input = parsed_args.input.as_path();
    let output = match parsed_args.outputs.get("obj") {
        Some(obj) => &obj.path,
        None => return Err(anyhow!("Missing object file output")),
    };

    let mut arguments: Vec<OsString> = vec![];

    // Pass the language explicitly as we might have gotten it from the
    // command line.
    let language = parsed_args.language.as_compiler_str(kind.into());
    if let Some(lang) = language {
        arguments.extend(vec!["-x".into(), lang.into()]);
    }

    arguments.extend_from_slice(&parsed_args.preprocessor_args);
    arguments.extend_from_slice(&parsed_args.unhashed_args);
    arguments.extend_from_slice(&parsed_args.common_args);
    arguments.extend_from_slice(&parsed_args.arch_args);

    if parsed_args.language.needs_c_preprocessing() {
        // Only add dependency args for files that need preprocessing
        arguments.extend_from_slice(&parsed_args.dependency_args);

        // Rewrite `-fdiagnostics-color[=auto]` to `fdiagnostics-color=always`,
        // otherwise gcc disables colored output because stdout is not a tty.
        // See: https://github.com/mozilla/sccache/pull/2758
        if parsed_args.color_mode != ColorMode::Off {
            arguments.push("-fdiagnostics-color=always".into());
        }
    }

    arguments.extend_from_slice(&["-o".into(), output.into()]);

    if !parsed_args.compilation_flag.is_empty() {
        arguments.push(parsed_args.compilation_flag.clone());
    }

    if parsed_args.double_dash_input {
        arguments.push("--".into());
    }
    arguments.push(input.into());

    #[cfg(feature = "dist-client")]
    let has_verbose_flag = arguments.contains(&OsString::from("-v"))
        || arguments.contains(&OsString::from("--verbose"));

    #[cfg(not(feature = "dist-client"))]
    let dist_command = None;
    #[cfg(feature = "dist-client")]
    // 1. Compilations with -v|--verbose must be run locally, since the verbose
    //    output is parsed by tools like CMake and must reflect the local toolchain
    // 2. ClangCUDA cannot be dist-compiled because Clang has separate host and
    //    device preprocessor outputs and cannot compile preprocessed CUDA files.
    let dist_command = if has_verbose_flag || parsed_args.language == Language::Cuda {
        None
    } else {
        (|| {
            use crate::util::path_to_str;
            Some(dist::CompileCommand {
                arguments: dist::osstrings_to_strings(&arguments)?,
                cwd: path_to_str(cwd).ok().map(Into::into)?,
                env_vars: dist::osstring_tuples_to_strings(env_vars)?,
                executable: path_to_str(executable).ok().map(Into::into)?,
            })
        })()
    };

    let command = SingleCompileCommand {
        arguments,
        cwd: cwd.to_owned(),
        env_vars: env_vars.to_owned(),
        executable: executable.to_owned(),
        out_pretty: out_pretty.to_string(),
        share_jobserver: false,
    };

    trace!("[{out_pretty}]: compile command: {command}");

    if let Some(ref dist_command) = dist_command {
        trace!("[{out_pretty}]: dist command: {dist_command}");
    }

    Ok((command, dist_command, Cacheable::Yes))
}

pub struct ExpandIncludeFile<'a> {
    cwd: &'a Path,
    stack: Vec<OsString>,
}

impl<'a> ExpandIncludeFile<'a> {
    pub fn new(cwd: &'a Path, args: &[OsString]) -> Self {
        ExpandIncludeFile {
            stack: args.iter().rev().map(|a| a.to_owned()).collect(),
            cwd,
        }
    }
}

impl Iterator for ExpandIncludeFile<'_> {
    type Item = OsString;

    fn next(&mut self) -> Option<OsString> {
        loop {
            let arg = self.stack.pop()?;
            let file = match arg.strip_prefix("@") {
                Some(arg) => self.cwd.join(arg),
                None => return Some(arg),
            };

            // According to gcc [1], @file means:
            //
            //     Read command-line options from file. The options read are
            //     inserted in place of the original @file option. If file does
            //     not exist, or cannot be read, then the option will be
            //     treated literally, and not removed.
            //
            //     Options in file are separated by whitespace. A
            //     whitespace character may be included in an option by
            //     surrounding the entire option in either single or double
            //     quotes. Any character (including a backslash) may be
            //     included by prefixing the character to be included with
            //     a backslash. The file may itself contain additional
            //     @file options; any such options will be processed
            //     recursively.
            //
            // So here we interpret any I/O errors as "just return this
            // argument". On a successful read the contents are tokenized using
            // the same rules GCC and Clang apply (quotes group an argument and
            // a backslash escapes the next character) and spliced into the
            // argument stream in place of the original `@file`. Because the
            // local and distributed commands are later reconstructed from the
            // parsed (expanded) arguments, the response file never needs to
            // exist on a remote machine, so such compilations can be both
            // cached and distributed.
            //
            // If the read fails we return the original `@file` argument, which
            // the parser treats as `TooHard` and refuses to cache, so we fall
            // back to invoking the compiler directly.
            //
            // [1]: https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html#Overall-Options
            let mut contents = String::new();
            let res = fs::File::open(&file).and_then(|mut f| f.read_to_string(&mut contents));
            if let Err(e) = res {
                debug!("failed to read @-file `{}`: {}", file.display(), e);
                return Some(arg);
            }
            let new_args = split_gnu_response_file_args(&contents);
            self.stack.extend(new_args.into_iter().rev());
        }
    }
}

/// Split the contents of a GCC/Clang `@response` file into arguments.
///
/// This mirrors `llvm::cl::TokenizeGNUCommandLine`, the routine Clang (and,
/// equivalently, GCC) uses to parse response files:
///
///  - Arguments are separated by whitespace.
///  - A single- or double-quoted string is part of a single argument; the
///    surrounding quotes are removed and may abut unquoted text (`a"b"c` is one
///    argument `abc`).
///  - A backslash escapes the following character, which is taken literally.
///    Backslashes are literal inside single quotes, but still escape inside
///    double quotes.
///  - Empty quoted strings (`""`) produce no argument. This matches Clang;
///    GCC's `buildargv` would instead emit an empty argument, but a bare `""`
///    argument is meaningless to a compiler so the difference is immaterial.
///
/// Whitespace is restricted to ASCII, matching the compilers' own tokenizers
/// (a non-ASCII Unicode space inside a filename must not split an argument).
fn split_gnu_response_file_args(contents: &str) -> Vec<OsString> {
    let mut args = Vec::new();
    let mut token = String::new();
    let mut chars = contents.chars();

    while let Some(c) = chars.next() {
        match c {
            c if c.is_ascii_whitespace() => {
                if !token.is_empty() {
                    args.push(OsString::from(std::mem::take(&mut token)));
                }
            }
            '\\' => {
                // A backslash escapes the next character, if any.
                if let Some(next) = chars.next() {
                    token.push(next);
                }
            }
            '\'' | '"' => {
                let quote = c;
                while let Some(qc) = chars.next() {
                    if qc == quote {
                        break;
                    }
                    // Backslash escapes inside double-quoted strings only.
                    if quote == '"' && qc == '\\' {
                        if let Some(next) = chars.next() {
                            token.push(next);
                        }
                    } else {
                        token.push(qc);
                    }
                }
            }
            c => token.push(c),
        }
    }

    if !token.is_empty() {
        args.push(OsString::from(token));
    }

    args
}

#[cfg(test)]
mod test {
    use itertools::assert_equal;
    use std::io::Write;

    use super::*;
    use crate::compiler::*;
    use crate::mock_command::*;
    use crate::server::SccacheService;
    use crate::test::mock_storage::MockStorage;
    use crate::test::utils::*;

    use temp_env::{with_var, with_var_unset};

    fn parse_arguments_(
        arguments: Vec<String>,
        plusplus: bool,
    ) -> CompilerArguments<ParsedArguments> {
        let args = arguments.iter().map(OsString::from).collect::<Vec<_>>();
        parse_arguments(&args, ".".as_ref(), &ARGS[..], plusplus, CCompilerKind::Gcc)
    }

    fn parse_arguments_clang(
        arguments: Vec<String>,
        plusplus: bool,
    ) -> CompilerArguments<ParsedArguments> {
        let args = arguments.iter().map(OsString::from).collect::<Vec<_>>();
        parse_arguments(
            &args,
            ".".as_ref(),
            (&ARGS[..], &clang::ARGS[..]),
            plusplus,
            CCompilerKind::Clang,
        )
    }

    #[test]
    fn test_parse_arguments_simple() {
        let args = stringvec!["-c", "foo.c", "-o", "foo.o"];
        let ParsedArguments {
            input,
            language,
            compilation_flag,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_eq!(Some("-c"), compilation_flag.to_str());
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert!(common_args.is_empty());
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_external_assembler() {
        // GCC has no integrated assembler, so `as` is always part of what
        // produced the object file.
        let args = stringvec!["-c", "foo.c", "-o", "foo.o"];
        match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => assert!(args.uses_external_assembler),
            o => panic!("Got unexpected parse result: {:?}", o),
        }
    }

    #[test]
    fn test_parse_arguments_default_name() {
        let args = stringvec!["-c", "foo.c"];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert!(common_args.is_empty());
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_default_outputdir() {
        let args = stringvec!["-c", "/tmp/foo.c"];
        let ParsedArguments { outputs, .. } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
    }

    #[test]
    fn test_parse_arguments_objc_header_language() {
        let args = stringvec!["-x", "objective-c-header", "-c", "foo.h", "-o", "foo.o"];
        let ParsedArguments {
            input,
            language,
            compilation_flag,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.h"), input.to_str());
        assert_eq!(Language::ObjectiveCHeader, language);
        assert_eq!(Some("-c"), compilation_flag.to_str());
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert!(common_args.is_empty());
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_split_dwarf() {
        let args = stringvec!["-gsplit-dwarf", "-c", "foo.cpp", "-o", "foo.o"];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.cpp"), input.to_str());
        assert_eq!(Language::Cxx, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            ),
            (
                "dwo",
                ArtifactDescriptor {
                    path: "foo.dwo".into(),
                    optional: true,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert!(
            common_args.contains(&"-gsplit-dwarf".into())
                && common_args.contains(&"-D_gsplit_dwarf_path=foo.dwo".into())
        );
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_linker_options() {
        let args = stringvec![
            // is basically the same as `-z deps`
            "-Wl,--unresolved-symbols=report-all",
            "-z",
            "call-nop=suffix-nop",
            "-z",
            "deps",
            "-c",
            "foo.c",
            "-o",
            "foo.o"
        ];

        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert_eq!(3, common_args.len());
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_coverage_outputs_gcno() {
        let args = stringvec!["--coverage", "-c", "foo.cpp", "-o", "foo.o"];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            profile_generate,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.cpp"), input.to_str());
        assert_eq!(Language::Cxx, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            ),
            (
                "gcno",
                ArtifactDescriptor {
                    path: PathBuf::from("foo.gcno"),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert_eq!(ovec!["--coverage"], common_args);
        assert!(!msvc_show_includes);
        assert!(profile_generate);
    }

    #[test]
    fn test_parse_arguments_depfile_for_raw_assembly_gcc() {
        let args = stringvec!["-c", "foo.s", "-o", "foo.o", "-MD", "-MF", "foo.d"];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.s"), input.to_str());
        assert_eq!(Language::Assembler, language);
        assert_equal(
            outputs,
            vec![(
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                },
            )],
        );
        assert!(preprocessor_args.is_empty());
    }

    #[test]
    fn test_parse_arguments_depfile_for_preprocessed_c_clang() {
        let args = stringvec!["-c", "foo.i", "-o", "foo.o", "-MD", "-MF", "foo.d"];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            ..
        } = match parse_arguments_clang(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.i"), input.to_str());
        assert_eq!(Language::CPreprocessed, language);
        assert_equal(
            outputs,
            vec![(
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                },
            )],
        );
        assert!(preprocessor_args.is_empty());
    }

    #[test]
    fn test_parse_arguments_test_coverage_outputs_gcno() {
        let args = stringvec!["-ftest-coverage", "-c", "foo.cpp", "-o", "foo.o"];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            profile_generate,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.cpp"), input.to_str());
        assert_eq!(Language::Cxx, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            ),
            (
                "gcno",
                ArtifactDescriptor {
                    path: PathBuf::from("foo.gcno"),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert_eq!(ovec!["-ftest-coverage"], common_args);
        assert!(!msvc_show_includes);
        assert!(profile_generate);
    }

    #[test]
    fn test_parse_arguments_profile_generate() {
        let args = stringvec!["-fprofile-generate", "-c", "foo.cpp", "-o", "foo.o"];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            profile_generate,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.cpp"), input.to_str());
        assert_eq!(Language::Cxx, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert_eq!(ovec!["-fprofile-generate"], common_args);
        assert!(!msvc_show_includes);
        assert!(profile_generate);
    }

    #[test]
    fn test_parse_arguments_extra() {
        let args = stringvec!["-c", "foo.cc", "-fabc", "-o", "foo.o", "-mxyz"];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.cc"), input.to_str());
        assert_eq!(Language::Cxx, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert_eq!(ovec!["-fabc", "-mxyz"], common_args);
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_values() {
        let args = stringvec![
            "-c", "foo.cxx", "-fabc", "-I", "include", "-o", "foo.o", "-include", "file"
        ];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.cxx"), input.to_str());
        assert_eq!(Language::Cxx, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert_eq!(ovec!["-Iinclude", "-include", "file"], preprocessor_args);
        assert_eq!(ovec!["-fabc"], common_args);
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_march_native() {
        let args = stringvec![
            "-c",
            "foo.cxx",
            "-march=native",
            "-mtune=native",
            "-o",
            "foo.o"
        ];
        let ParsedArguments {
            input,
            language,
            outputs,
            arch_args,
            common_args,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.cxx"), input.to_str());
        assert_eq!(Language::Cxx, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert_eq!(ovec!["-march=native", "-mtune=native"], arch_args);
        assert!(common_args.is_empty());
    }

    #[test]
    fn test_parse_arguments_preprocessor_args() {
        let args = stringvec![
            "-c",
            "foo.c",
            "-fabc",
            "-MF",
            "foo.o.d",
            "-o",
            "foo.o",
            "-MQ",
            "abc",
            "-nostdinc"
        ];
        let ParsedArguments {
            input,
            language,
            outputs,
            dependency_args,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            depfile,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_eq!(Some("foo.o.d"), depfile.unwrap().to_str());
        assert_map_contains!(
            outputs,
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
        assert_eq!(ovec!["-MF", "foo.o.d"], dependency_args);
        assert_eq!(ovec!["-nostdinc"], preprocessor_args);
        assert_eq!(ovec!["-fabc"], common_args);
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_unhashed() {
        let unhashed_flags = stringvec!["-pipe"];

        for flag in unhashed_flags {
            let args = stringvec!["-c", "foo.c", "-o", "foo.o", flag];

            let a = match parse_arguments_(args, false) {
                CompilerArguments::Ok(args) => args,
                o => panic!("Got unexpected parse result: {o:?} for flag: {flag:?}"),
            };

            assert_eq!(Language::C, a.language);
            assert!(a.preprocessor_args.is_empty());
            assert!(a.common_args.is_empty());
            assert_eq!(ovec![flag], a.unhashed_args,);
        }
    }

    #[test]
    fn test_parse_arguments_double_dash() {
        let args = stringvec!["-c", "-o", "foo.o", "--", "foo.c"];
        let ParsedArguments {
            input,
            double_dash_input,
            common_args,
            ..
        } = match parse_arguments_(args.clone(), false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        // GCC doesn't support double dashes. If we got one, we'll pass them
        // through to GCC for it to error out.
        assert!(!double_dash_input);
        assert_eq!(ovec!["--"], common_args);

        let ParsedArguments {
            input,
            double_dash_input,
            common_args,
            ..
        } = match parse_arguments_clang(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert!(double_dash_input);
        assert!(common_args.is_empty());

        let args = stringvec!["-c", "-o", "foo.o", "foo.c", "--"];
        let ParsedArguments {
            input,
            double_dash_input,
            common_args,
            ..
        } = match parse_arguments_clang(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        // Double dash after input file is ignored.
        assert!(!double_dash_input);
        assert!(common_args.is_empty());

        let args = stringvec!["-c", "-o", "foo.o", "foo.c", "--", "bar.c"];
        assert_eq!(
            CompilerArguments::CannotCache("multiple input files", Some("[\"bar.c\"]".to_string())),
            parse_arguments_clang(args, false)
        );

        let args = stringvec!["-c", "-o", "foo.o", "foo.c", "--", "-fPIC"];
        assert_eq!(
            CompilerArguments::CannotCache("multiple input files", Some("[\"-fPIC\"]".to_string())),
            parse_arguments_clang(args, false)
        );
    }

    #[test]
    fn test_parse_arguments_too_hard() {
        let too_hard_flags = stringvec![
            "-fstack-usage",
            "-fcallgraph-info",
            "-save-temps",
            "-save-temps=cwd",
            "-save-temps=obj",
            "--save-temps",
            "--save-temps=cwd",
            "--save-temps=obj"
        ];

        for flag in too_hard_flags {
            let args = stringvec!["-c", "foo.c", "-o", "foo.o", flag];

            match parse_arguments_(args, false) {
                CompilerArguments::CannotCache(reason, None) => {
                    assert_eq!(reason, &flag);
                }
                o => {
                    panic!("Got unexpected parse result: {o:?} for flag: {flag:?}");
                }
            }
        }
    }

    #[test]
    fn test_parse_arguments_explicit_dep_target() {
        let args = stringvec![
            "-c", "foo.c", "-MT", "depfile", "-fabc", "-MF", "foo.o.d", "-o", "foo.o"
        ];
        let ParsedArguments {
            input,
            language,
            outputs,
            dependency_args,
            msvc_show_includes,
            common_args,
            depfile,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_eq!(Some("foo.o.d"), depfile.unwrap().to_str());
        assert_map_contains!(
            outputs,
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
        assert_eq!(ovec!["-MF", "foo.o.d"], dependency_args);
        assert_eq!(ovec!["-fabc"], common_args);
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_explicit_dep_target_needed() {
        let args = stringvec![
            "-c", "foo.c", "-MT", "depfile", "-fabc", "-MF", "foo.o.d", "-o", "foo.o", "-MD"
        ];
        let ParsedArguments {
            input,
            language,
            outputs,
            dependency_args,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            depfile,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_eq!(Some("foo.o.d"), depfile.unwrap().to_str());
        assert_map_contains!(
            outputs,
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
            ovec!["-MF", "foo.o.d", "-MD", "-MT", "depfile"],
            dependency_args
        );
        assert!(preprocessor_args.is_empty());
        assert_eq!(ovec!["-fabc"], common_args);
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_explicit_mq_dep_target_needed() {
        let args = stringvec![
            "-c", "foo.c", "-MQ", "depfile", "-fabc", "-MF", "foo.o.d", "-o", "foo.o", "-MD"
        ];
        let ParsedArguments {
            input,
            language,
            outputs,
            dependency_args,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            depfile,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_eq!(Some("foo.o.d"), depfile.unwrap().to_str());
        assert_map_contains!(
            outputs,
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
            ovec!["-MF", "foo.o.d", "-MD", "-MQ", "depfile"],
            dependency_args
        );
        assert!(preprocessor_args.is_empty());
        assert_eq!(ovec!["-fabc"], common_args);
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_diagnostics_color() {
        fn get_color_mode(color_flag: &str) -> ColorMode {
            let args = stringvec!["-c", "foo.c", color_flag];
            match parse_arguments_(args, false) {
                CompilerArguments::Ok(args) => args.color_mode,
                o => panic!("Got unexpected parse result: {o:?}"),
            }
        }

        assert_eq!(get_color_mode("-fdiagnostics-color=always"), ColorMode::On);
        assert_eq!(get_color_mode("-fdiagnostics-color=never"), ColorMode::Off);
        assert_eq!(get_color_mode("-fdiagnostics-color=auto"), ColorMode::Auto);
        assert_eq!(get_color_mode("-fno-diagnostics-color"), ColorMode::Off);
        assert_eq!(get_color_mode("-fdiagnostics-color"), ColorMode::On);
    }

    #[test]
    fn color_mode_preprocess() {
        let args = stringvec!["-c", "foo.c", "-fdiagnostics-color"];
        let args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };

        assert!(!args.common_args.contains(&"-fdiagnostics-color".into()));
    }

    // #[test]
    // fn test_preprocess_cmd_rewrites_archs() {
    //     with_var("SCCACHE_CACHE_MULTIARCH", Some("1"), || {
    //         let args = stringvec!["-arch", "arm64", "-arch", "i386", "-c", "foo.cc"];
    //         let parsed_args = match parse_arguments_clang(args, false) {
    //             CompilerArguments::Ok(args) => args,
    //             o => panic!("Got unexpected parse result: {o:?}"),
    //         };
    //         let cmd = preprocess_cmd(
    //             MockCommand {
    //                 child: None,
    //                 args: vec![],
    //             },
    //             &parsed_args,
    //             Path::new(""),
    //             &[],
    //             &CCompilerKind::Clang,
    //             &[],
    //         );
    //         // make sure the architectures were rewritten to prepocessor defines
    //         let expected_args = ovec!["-x", "c++", "-D__arm64__=1", "-D__i386__=1", "-E", "foo.cc"];
    //         assert_eq!(cmd.args, expected_args);
    //     });
    // }

    // #[test]
    // fn test_preprocess_cmd_doesnt_rewrite_single_arch() {
    //     let args = stringvec!["-arch", "arm64", "-c", "foo.cc"];
    //     let parsed_args = match parse_arguments_clang(args, false) {
    //         CompilerArguments::Ok(args) => args,
    //         o => panic!("Got unexpected parse result: {o:?}"),
    //     };
    //     let cmd = preprocess_cmd(
    //         MockCommand {
    //             child: None,
    //             args: vec![],
    //         },
    //         &parsed_args,
    //         Path::new(""),
    //         &[],
    //         &CCompilerKind::Gcc,
    //         &[],
    //     );
    //     // make sure the architectures were rewritten to prepocessor defines
    //     let expected_args = ovec!["-x", "c++", "-arch", "arm64", "-o", "-", "-E", "foo.cc"];
    //     assert_eq!(cmd.args, expected_args);
    // }

    // #[test]
    // fn test_preprocess_double_dash_input() {
    //     let args = stringvec!["-c", "-o", "foo.o", "--", "foo.c"];
    //     let parsed_args = match parse_arguments_clang(args, false) {
    //         CompilerArguments::Ok(args) => args,
    //         o => panic!("Got unexpected parse result: {o:?}"),
    //     };
    //     let cmd = preprocess_cmd(
    //         MockCommand {
    //             child: None,
    //             args: vec![],
    //         },
    //         &parsed_args,
    //         Path::new(""),
    //         &[],
    //         &CCompilerKind::Clang,
    //         &[],
    //     );
    //     let expected_args = ovec!["-x", "c", "-E", "--", "foo.c"];
    //     assert_eq!(cmd.args, expected_args);
    // }

    // #[test]
    // fn pedantic_default() {
    //     let args = stringvec!["-pedantic", "-c", "foo.cc"];
    //     let parsed_args = match parse_arguments_(args, false) {
    //         CompilerArguments::Ok(args) => args,
    //         o => panic!("Got unexpected parse result: {o:?}"),
    //     };
    //     let cmd = preprocess_cmd(
    //         MockCommand {
    //             child: None,
    //             args: vec![],
    //         },
    //         &parsed_args,
    //         Path::new(""),
    //         &[],
    //         &CCompilerKind::Gcc,
    //         &[],
    //     );
    //     // disable with extensions enabled
    //     assert!(!cmd.args.contains(&"-fdirectives-only".into()));
    // }

    // #[test]
    // fn pedantic_std() {
    //     let args = stringvec!["-pedantic-errors", "-c", "-std=c++14", "foo.cc"];
    //     let parsed_args = match parse_arguments_(args, false) {
    //         CompilerArguments::Ok(args) => args,
    //         o => panic!("Got unexpected parse result: {o:?}"),
    //     };
    //     let cmd = preprocess_cmd(
    //         MockCommand {
    //             child: None,
    //             args: vec![],
    //         },
    //         &parsed_args,
    //         Path::new(""),
    //         &[],
    //         &CCompilerKind::Gcc,
    //         &[],
    //     );
    //     // no reason to disable it with no extensions enabled
    //     assert!(cmd.args.contains(&"-fdirectives-only".into()));
    // }

    // #[test]
    // fn pedantic_gnu() {
    //     let args = stringvec!["-pedantic-errors", "-c", "-std=gnu++14", "foo.cc"];
    //     let parsed_args = match parse_arguments_(args, false) {
    //         CompilerArguments::Ok(args) => args,
    //         o => panic!("Got unexpected parse result: {o:?}"),
    //     };
    //     let cmd = preprocess_cmd(
    //         MockCommand {
    //             child: None,
    //             args: vec![],
    //         },
    //         &parsed_args,
    //         Path::new(""),
    //         &[],
    //         &CCompilerKind::Gcc,
    //         &[],
    //     );
    //     // disable with extensions enabled
    //     assert!(!cmd.args.contains(&"-fdirectives-only".into()));
    // }

    #[test]
    fn test_parse_arguments_dep_target_needed() {
        let args = stringvec![
            "-c", "foo.c", "-fabc", "-MF", "foo.o.d", "-o", "foo.o", "-MD"
        ];
        let ParsedArguments {
            input,
            language,
            outputs,
            dependency_args,
            msvc_show_includes,
            common_args,
            depfile,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_eq!(Some("foo.o.d"), depfile.unwrap().to_str());
        assert_map_contains!(
            outputs,
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
            ovec!["-MF", "foo.o.d", "-MD", "-MT", "foo.o"],
            dependency_args
        );
        assert_eq!(ovec!["-fabc"], common_args);
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_dep_target_and_file_needed() {
        let args = stringvec!["-c", "foo/bar.c", "-fabc", "-o", "foo/bar.o", "-MMD"];
        let ParsedArguments {
            input,
            language,
            outputs,
            dependency_args,
            msvc_show_includes,
            common_args,
            depfile,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo/bar.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_eq!(Some("foo/bar.d"), depfile.unwrap().to_str());
        assert_map_contains!(
            outputs,
            (
                "dep",
                ArtifactDescriptor {
                    path: PathBuf::from("foo/bar.d"),
                    optional: false,
                }
            ),
            (
                "obj",
                ArtifactDescriptor {
                    path: PathBuf::from("foo/bar.o"),
                    optional: false,
                }
            )
        );
        assert_eq!(
            ovec!["-MMD", "-MT", "foo/bar.o", "-MF", "foo/bar.d"],
            dependency_args
        );
        assert_eq!(ovec!["-fabc"], common_args);
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_empty_args() {
        assert_eq!(
            CompilerArguments::NotCompilation,
            parse_arguments_(vec!(), false)
        );
    }

    #[test]
    fn test_parse_arguments_not_compile() {
        assert_eq!(
            CompilerArguments::NotCompilation,
            parse_arguments_(stringvec!["-o", "foo"], false)
        );
    }

    #[test]
    fn test_parse_arguments_too_many_inputs_single() {
        assert_eq!(
            CompilerArguments::CannotCache("multiple input files", Some("[\"bar.c\"]".to_string())),
            parse_arguments_(stringvec!["-c", "foo.c", "-o", "foo.o", "bar.c"], false)
        );
    }

    #[test]
    fn test_parse_arguments_too_many_inputs_multiple() {
        assert_eq!(
            CompilerArguments::CannotCache(
                "multiple input files",
                Some("[\"bar.c\", \"baz.c\"]".to_string())
            ),
            parse_arguments_(
                stringvec!["-c", "foo.c", "-o", "foo.o", "bar.c", "baz.c"],
                false
            )
        );
    }

    #[test]
    fn test_parse_arguments_link() {
        assert_eq!(
            CompilerArguments::NotCompilation,
            parse_arguments_(
                stringvec!["-shared", "foo.o", "-o", "foo.so", "bar.o"],
                false
            )
        );
    }

    #[test]
    fn test_parse_arguments_fplugin() {
        assert_eq!(
            CompilerArguments::CannotCache("-fplugin", None),
            parse_arguments_(
                stringvec!["-c", "foo.c", "-fplugin=plugin.so", "-o", "foo.o"],
                false
            )
        );
        assert_eq!(
            CompilerArguments::CannotCache("-fplugin", None),
            parse_arguments_(
                stringvec!["-c", "foo.c", "-fplugin=libcc1plugin", "-o", "foo.o"],
                false
            )
        );
    }

    #[test]
    fn test_parse_arguments_fcallgraph_info_with_markers() {
        assert_eq!(
            CompilerArguments::CannotCache("-fcallgraph-info", None),
            parse_arguments_(
                stringvec!["-c", "foo.c", "-fcallgraph-info=su,da", "-o", "foo.o"],
                false
            )
        );
    }

    #[test]
    fn test_parse_arguments_pgo() {
        assert_eq!(
            CompilerArguments::CannotCache("-fprofile-use", None),
            parse_arguments_(
                stringvec!["-c", "foo.c", "-fprofile-use", "-o", "foo.o"],
                false
            )
        );
        assert_eq!(
            CompilerArguments::CannotCache("-fprofile-use", None),
            parse_arguments_(
                stringvec!["-c", "foo.c", "-fprofile-use=file", "-o", "foo.o"],
                false
            )
        );
    }

    #[test]
    fn test_parse_arguments_cxx20_modules_supported() {
        // -fmodules-ts - enables C++20 modules support in GCC
        match parse_arguments_(
            stringvec!["-c", "foo.cpp", "-fmodules-ts", "-o", "foo.o"],
            false,
        ) {
            CompilerArguments::Ok(args) => assert!(args.cxx20_modules),
            o => panic!("Got unexpected parse result: {o:?}"),
        }

        // -fmodules - older/clang-style modules flag
        assert_eq!(
            CompilerArguments::CannotCache("-fmodules", None),
            parse_arguments_(
                stringvec!["-c", "foo.cpp", "-fmodules", "-o", "foo.o"],
                false
            )
        );
    }

    #[test]
    fn test_parse_arguments_response_file() {
        assert_eq!(
            CompilerArguments::CannotCache("@", None),
            parse_arguments_(stringvec!["-c", "foo.c", "@foo", "-o", "foo.o"], false)
        );
        assert_eq!(
            CompilerArguments::CannotCache("@", None),
            parse_arguments_(stringvec!["-c", "foo.c", "-o", "@foo"], false)
        );
    }

    #[test]
    fn test_parse_index_store_path() {
        assert_eq!(
            CompilerArguments::CannotCache("-index-store-path", None),
            parse_arguments_(
                stringvec![
                    "-c",
                    "foo.c",
                    "-index-store-path",
                    "index.store",
                    "-o",
                    "foo.o"
                ],
                false
            )
        );
    }

    #[test]
    fn test_parse_arguments_multiarch_cache_disabled() {
        with_var_unset("SCCACHE_CACHE_MULTIARCH", || {
            assert_eq!(
                CompilerArguments::CannotCache(
                    "multiple different -arch, and SCCACHE_CACHE_MULTIARCH not set",
                    None
                ),
                parse_arguments_clang(
                    stringvec![
                        "-fPIC", "-arch", "arm64", "-arch", "i386", "-o", "foo.o", "-c", "foo.cpp"
                    ],
                    false
                )
            );
        });
    }

    #[test]
    fn test_parse_arguments_multiple_arch() {
        match parse_arguments_clang(
            stringvec!["-arch", "arm64", "-o", "foo.o", "-c", "foo.cpp"],
            false,
        ) {
            CompilerArguments::Ok(_) => {}
            o => panic!("Got unexpected parse result: {o:?}"),
        }

        with_var("SCCACHE_CACHE_MULTIARCH", Some("1"), || {
            match parse_arguments_clang(
                stringvec![
                    "-arch", "arm64", "-arch", "arm64", "-o", "foo.o", "-c", "foo.cpp"
                ],
                false,
            ) {
                CompilerArguments::Ok(_) => {}
                o => panic!("Got unexpected parse result: {o:?}"),
            }

            let args = stringvec![
                "-fPIC", "-arch", "arm64", "-arch", "i386", "-o", "foo.o", "-c", "foo.cpp"
            ];
            let ParsedArguments {
                input,
                language,
                compilation_flag,
                outputs,
                preprocessor_args,
                msvc_show_includes,
                common_args,
                arch_args,
                ..
            } = match parse_arguments_clang(args, false) {
                CompilerArguments::Ok(args) => args,
                o => panic!("Got unexpected parse result: {o:?}"),
            };
            assert_eq!(Some("foo.cpp"), input.to_str());
            assert_eq!(Language::Cxx, language);
            assert_eq!(Some("-c"), compilation_flag.to_str());
            assert_map_contains!(
                outputs,
                (
                    "obj",
                    ArtifactDescriptor {
                        path: "foo.o".into(),
                        optional: false,
                    }
                )
            );
            assert!(preprocessor_args.is_empty());
            assert_eq!(ovec!["-fPIC"], common_args);
            assert_eq!(ovec!["-arch", "arm64", "-arch", "i386"], arch_args);
            assert!(!msvc_show_includes);
        });
    }

    #[test]
    fn at_signs() {
        let td = crate::util::temp_dir().unwrap();
        fs::File::create(td.path().join("foo"))
            .unwrap()
            .write_all(
                b"\
            -c foo.c -o foo.o\
        ",
            )
            .unwrap();
        let arg = format!("@{}", td.path().join("foo").display());
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(vec![arg], false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert!(common_args.is_empty());
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_split_gnu_response_file_args() {
        // Plain whitespace separation, including newlines and tabs.
        assert_eq!(
            ovec!["-c", "foo.c", "-o", "foo.o"],
            split_gnu_response_file_args("-c foo.c\n\t-o  foo.o\n")
        );
        // Quotes group an argument and are stripped.
        assert_eq!(
            ovec!["-I", "/a path/with spaces", "-DFOO=bar baz"],
            split_gnu_response_file_args("-I \"/a path/with spaces\" '-DFOO=bar baz'")
        );
        // Quotes may abut unquoted text.
        assert_eq!(
            ovec!["-DA=a b c"],
            split_gnu_response_file_args("-DA=\"a b c\"")
        );
        // Backslash escapes the next character outside quotes.
        assert_eq!(
            ovec!["a b", "c\\d"],
            split_gnu_response_file_args("a\\ b c\\\\d")
        );
        // Backslash escapes inside double quotes but not single quotes.
        assert_eq!(
            ovec!["a\"b", "c\\d"],
            split_gnu_response_file_args("\"a\\\"b\" 'c\\d'")
        );
        // Empty quoted strings produce no argument.
        assert_eq!(
            ovec!["-c", "foo.c"],
            split_gnu_response_file_args("-c \"\" foo.c")
        );
        // Empty or whitespace-only input produces no arguments.
        assert!(split_gnu_response_file_args("").is_empty());
        assert!(split_gnu_response_file_args("   \n\t").is_empty());
        // A trailing backslash with nothing to escape is dropped.
        assert_eq!(ovec!["foo"], split_gnu_response_file_args("foo\\"));
        // A trailing backslash at the end of a double-quoted string is dropped.
        assert_eq!(ovec!["x"], split_gnu_response_file_args("\"x\\"));
        // An unterminated quote consumes the rest of the input as one argument.
        assert_eq!(ovec!["abc"], split_gnu_response_file_args("\"abc"));
        assert_eq!(ovec!["a b"], split_gnu_response_file_args("'a b"));
    }

    #[test]
    fn test_parse_arguments_response_file_with_quotes() {
        // A response file containing quoted arguments (common with Clang) must
        // be expanded and remain cacheable.
        let td = tempfile::Builder::new()
            .prefix("sccache")
            .tempdir()
            .unwrap();
        fs::File::create(td.path().join("args"))
            .unwrap()
            .write_all(b"-c foo.c -o foo.o \"-DGREETING=hello world\"\n")
            .unwrap();
        let arg = format!("@{}", td.path().join("args").display());
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            ..
        } = match parse_arguments_(vec![arg], false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false
                }
            )
        );
        assert_eq!(ovec!["-DGREETING=hello world"], preprocessor_args);
    }

    #[test]
    fn test_compile_simple() {
        let creator = new_creator();
        let f = TestFixture::new();
        let parsed_args = ParsedArguments {
            input: "foo.c".into(),
            language: Language::C,
            compilation_flag: "-c".into(),
            outputs: vec![(
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                },
            )]
            .into_iter()
            .collect(),
            color_mode: ColorMode::Auto,
            ..Default::default()
        };
        let runtime = single_threaded_runtime();
        let storage = MockStorage::new(None, true);
        let storage: std::sync::Arc<MockStorage> = std::sync::Arc::new(storage);
        let service = SccacheService::mock_with_storage(storage, runtime.handle().clone());
        let compiler = &f.bins[0];
        // Compiler invocation.
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        let (command, dist_command, cacheable) = generate_compile_commands(
            compiler,
            &parsed_args,
            f.tempdir.path(),
            &[],
            CCompilerKind::Gcc,
        )
        .unwrap();
        #[cfg(feature = "dist-client")]
        assert!(dist_command.is_some());
        #[cfg(not(feature = "dist-client"))]
        assert!(dist_command.is_none());
        let _ = command.execute(&service, &creator).wait();
        assert_eq!(Cacheable::Yes, cacheable);
        // Ensure that we ran all processes.
        assert_eq!(0, creator.lock().unwrap().children.len());
    }

    #[test]
    fn test_compile_simple_verbose_short() {
        let creator = new_creator();
        let f = TestFixture::new();
        let parsed_args = ParsedArguments {
            input: "foo.c".into(),
            language: Language::C,
            compilation_flag: "-c".into(),
            outputs: vec![(
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                },
            )]
            .into_iter()
            .collect(),
            common_args: vec!["-v".into()],
            ..Default::default()
        };
        let runtime = single_threaded_runtime();
        let storage = MockStorage::new(None, true);
        let storage: std::sync::Arc<MockStorage> = std::sync::Arc::new(storage);
        let service = SccacheService::mock_with_storage(storage, runtime.handle().clone());
        let compiler = &f.bins[0];
        // Compiler invocation.
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        let (command, dist_command, cacheable) = generate_compile_commands(
            compiler,
            &parsed_args,
            f.tempdir.path(),
            &[],
            CCompilerKind::Gcc,
        )
        .unwrap();
        // -v should never generate a dist_command
        assert!(dist_command.is_none());
        let _ = command.execute(&service, &creator).wait();
        assert_eq!(Cacheable::Yes, cacheable);
        // Ensure that we ran all processes.
        assert_eq!(0, creator.lock().unwrap().children.len());
    }

    #[test]
    fn test_compile_simple_verbose_long() {
        let creator = new_creator();
        let f = TestFixture::new();
        let parsed_args = ParsedArguments {
            input: "foo.c".into(),
            language: Language::C,
            compilation_flag: "-c".into(),
            outputs: vec![(
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                },
            )]
            .into_iter()
            .collect(),
            common_args: vec!["--verbose".into()],
            ..Default::default()
        };
        let runtime = single_threaded_runtime();
        let storage = MockStorage::new(None, true);
        let storage: std::sync::Arc<MockStorage> = std::sync::Arc::new(storage);
        let service = SccacheService::mock_with_storage(storage, runtime.handle().clone());
        let compiler = &f.bins[0];
        // Compiler invocation.
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        let (command, dist_command, cacheable) = generate_compile_commands(
            compiler,
            &parsed_args,
            f.tempdir.path(),
            &[],
            CCompilerKind::Gcc,
        )
        .unwrap();
        // --verbose should never generate a dist_command
        assert!(dist_command.is_none());
        let _ = command.execute(&service, &creator).wait();
        assert_eq!(Cacheable::Yes, cacheable);
        // Ensure that we ran all processes.
        assert_eq!(0, creator.lock().unwrap().children.len());
    }

    #[test]
    fn test_compile_double_dash_input() {
        let args = stringvec!["-c", "-o", "foo.o", "--", "foo.c"];
        let parsed_args = match parse_arguments_clang(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        let f = TestFixture::new();
        let compiler = &f.bins[0];
        let (command, _, _) = generate_compile_commands(
            compiler,
            &parsed_args,
            f.tempdir.path(),
            &[],
            CCompilerKind::Clang,
        )
        .unwrap();
        let expected_args = ovec![
            "-x",
            "c",
            "-fdiagnostics-color=always",
            "-o",
            "foo.o",
            "-c",
            "--",
            "foo.c"
        ];
        assert_eq!(command.get_arguments(), expected_args);
    }

    #[test]
    fn test_parse_arguments_plusplus() {
        let args = stringvec!["-c", "foo.c", "-o", "foo.o"];
        let ParsedArguments {
            input,
            language,
            compilation_flag,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(args, true) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::Cxx, language);
        assert_eq!(Some("-c"), compilation_flag.to_str());
        assert_map_contains!(
            outputs,
            (
                "obj",
                ArtifactDescriptor {
                    path: "foo.o".into(),
                    optional: false,
                }
            )
        );
        assert!(preprocessor_args.is_empty());
        assert!(common_args.is_empty());
        assert!(!msvc_show_includes);
    }

    // #[test]
    // fn test_pch_explicit() {
    //     let args = stringvec!["-c", "-x", "c++-header", "pch.h", "-o", "pch.pch"];
    //     let parsed_args = match parse_arguments_(args, false) {
    //         CompilerArguments::Ok(args) => args,
    //         o => panic!("Got unexpected parse result: {o:?}"),
    //     };
    //     let cmd = preprocess_cmd(
    //         MockCommand {
    //             child: None,
    //             args: vec![],
    //         },
    //         &parsed_args,
    //         Path::new(""),
    //         &[],
    //         &CCompilerKind::Gcc,
    //         &[],
    //     );
    //     assert!(cmd.args.contains(&"-x".into()) && cmd.args.contains(&"c++-header".into()));
    // }

    // #[test]
    // fn test_pch_implicit() {
    //     let args = stringvec!["-c", "pch.hpp", "-o", "pch.pch"];
    //     let parsed_args = match parse_arguments_(args, false) {
    //         CompilerArguments::Ok(args) => args,
    //         o => panic!("Got unexpected parse result: {o:?}"),
    //     };
    //     let cmd = preprocess_cmd(
    //         MockCommand {
    //             child: None,
    //             args: vec![],
    //         },
    //         &parsed_args,
    //         Path::new(""),
    //         &[],
    //         &CCompilerKind::Gcc,
    //         &[],
    //     );
    //     assert!(cmd.args.contains(&"-x".into()) && cmd.args.contains(&"c++-header".into()));
    // }

    // #[test]
    // fn test_pch_generic() {
    //     let args = stringvec!["-c", "pch.h", "-o", "pch.pch"];
    //     let parsed_args = match parse_arguments_(args, false) {
    //         CompilerArguments::Ok(args) => args,
    //         o => panic!("Got unexpected parse result: {o:?}"),
    //     };
    //     let cmd = preprocess_cmd(
    //         MockCommand {
    //             child: None,
    //             args: vec![],
    //         },
    //         &parsed_args,
    //         Path::new(""),
    //         &[],
    //         &CCompilerKind::Gcc,
    //         &[],
    //     );
    //     assert!(!cmd.args.contains(&"-x".into()));
    // }

    #[test]
    fn test_hard_preprocessor_args() {
        let args = stringvec!["-c", "foo.c", "-o", "foo.o"];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert!(parsed_args.preprocessor_args.is_empty());

        let args = stringvec!["-c", "foo.c", "-o", "foo.o", r#"-Wp,-DFOO="something""#];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(
            parsed_args.preprocessor_args,
            vec![r#"-Wp,-DFOO="something""#]
        );
    }

    #[test]
    fn test_xpreprocessor_dependency_args() {
        let args = stringvec![
            "-c",
            "foo.c",
            "-fabc",
            "-Wp,-MD,foo.o.d",
            "-Wp,-MT,foo.o",
            "-Wp,-MF,foo.o.d",
            "-o",
            "foo.o"
        ];
        let ParsedArguments {
            input,
            language,
            outputs,
            dependency_args,
            msvc_show_includes,
            common_args,
            depfile,
            ..
        } = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_eq!(Some("foo.o.d"), depfile.unwrap().to_str());
        assert_map_contains!(
            outputs,
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
            ovec!["-Wp,-MD,foo.o.d", "-Wp,-MT,foo.o", "-Wp,-MF,foo.o.d"],
            dependency_args
        );
        assert_eq!(ovec!["-fabc"], common_args);
        assert!(!msvc_show_includes);
    }
}
