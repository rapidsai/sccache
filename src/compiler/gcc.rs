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
            ArtifactDescriptor, CCompilerImpl, CCompilerKind, DepfilePath, ParsedArguments,
            PreprocessorOutput,
        },
        clang,
        msvc::from_local_codepage,
        preprocessor_cache::normalize_path,
    },
    counted_array, dist,
    errors::*,
    mock_command::{CommandCreatorSync, RunCommand},
    server::SccacheService,
    util::{OsStrExt, run_input_output, run_input_stream_output, temppath},
};

use async_trait::async_trait;
use fs::File;
use fs_err as fs;
use futures::{FutureExt, StreamExt};
use itertools::Itertools;
use std::{
    collections::HashMap,
    env,
    ffi::OsString,
    io::Read,
    path::{Path, PathBuf},
};

/// A struct on which to implement `CCompilerImpl`.
#[derive(Clone, Debug, Default)]
pub struct Gcc {
    pub gplusplus: bool,
    pub version: Option<String>,
    pub native_archs: Option<(String, String)>,
}

impl Gcc {
    pub async fn read_implicit_specfiles<T>(
        creator: &mut T,
        exe: &Path,
        arguments: &[OsString],
        env_vars: &[(OsString, OsString)],
        v_flag: &str,
    ) -> Result<Vec<PathBuf>>
    where
        T: CommandCreatorSync,
    {
        use crate::compiler::write_temp_file;
        use bytes::Buf;
        use itertools::Itertools;
        use std::io::BufRead;

        let (_tempdir, src) = write_temp_file(
            &tokio::runtime::Handle::current(),
            "testfile.c".as_ref(),
            Default::default(),
        )
        .await?;

        let mut cmd = creator.new_command_sync(exe);
        cmd.env_clear()
            .envs(env_vars.iter().map(|s| (&s.0, &s.1)))
            .args(arguments)
            .arg(v_flag)
            .arg("-E")
            .arg(&src);

        let output = run_input_output(cmd, None).await.unwrap_or_default();

        Ok(output
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

    pub async fn read_native_arch<T>(
        creator: &mut T,
        exe: &Path,
        env_vars: &[(OsString, OsString)],
    ) -> Option<(String, String)>
    where
        T: CommandCreatorSync,
    {
        use bytes::Buf;
        use itertools::Itertools;
        use std::io::BufRead;

        let mut cmd = creator.new_command_sync(exe);
        cmd.env_clear()
            .envs(env_vars.iter().map(|s| (&s.0, &s.1)))
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
    fn parse_arguments(
        &self,
        arguments: &[OsString],
        cwd: &Path,
        _env_vars: &[(OsString, OsString)],
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
        include_line_numbers: bool,
    ) -> Result<PreprocessorOutput>
    where
        T: CommandCreatorSync,
    {
        let extra_preprocessor_flags = if include_line_numbers {
            vec![]
        } else {
            vec!["-P".into()]
        };

        preprocess(
            service,
            creator,
            executable,
            parsed_args,
            cwd,
            env_vars,
            self.kind(),
            rewrite_includes_only,
            generate_dependencies,
            &extra_preprocessor_flags,
        )
        .await
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
        generate_dependencies(creator, executable, parsed_args, cwd, env_vars, self.kind())
            .await
            .map(Some)
    }

    fn generate_compile_commands(
        &self,
        path_transformer: &mut dist::PathTransformer,
        executable: &Path,
        parsed_args: &ParsedArguments,
        cwd: &Path,
        env_vars: &[(OsString, OsString)],
        rewrite_includes_only: bool,
        _hash_key: &str,
    ) -> Result<(
        impl CompileCommandImpl,
        Option<dist::CompileCommand>,
        Cacheable,
    )> {
        generate_compile_commands(
            path_transformer,
            executable,
            parsed_args,
            cwd,
            env_vars,
            self.kind(),
            rewrite_includes_only,
        )
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
}

use self::ArgData::*;

// Mostly taken from https://github.com/ccache/ccache/blob/master/src/ccache/compopt.cpp#L52-L183
counted_array!(pub static ARGS: [ArgInfo<ArgData>; _] = [
    flag!("-", TooHardFlag),
    flag!("--coverage", Coverage),
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
    take_arg!("-D", OsString, CanBeSeparated, PassThrough),
    flag!("-E", TooHardFlag),
    take_arg!("-F", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-G", OsString, Separated, PassThrough),
    take_arg!("-I", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-L", OsString, Separated, PassThrough),
    flag!("-M", TooHardFlag),
    flag!("-MD", NeedDepTarget),
    take_arg!("-MF", PathBuf, Separated, DepArgumentPath),
    flag!("-MM", TooHardFlag),
    flag!("-MMD", NeedDepTarget),
    flag!("-MP", NeedDepTarget),
    take_arg!("-MQ", OsString, Separated, DepTarget),
    take_arg!("-MT", OsString, Separated, DepTarget),
    flag!("-P", TooHardFlag),
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
    flag!("-fplugin=libcc1plugin", TooHardFlag),
    flag!("-fprofile-arcs", ProfileGenerate),
    flag!("-fprofile-generate", ProfileGenerate),
    take_arg!("-fprofile-use", OsString, Concatenated, TooHard),
    flag!("-frepo", TooHardFlag),
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
    S: SearchableArgInfo<ArgData>,
{
    let mut output_arg = None;
    let mut module_output_path = None;
    let mut module_only_flag = false;
    let mut input_arg = None;
    let mut double_dash_input = false;
    let mut depfile = None;
    let mut dep_target = None;
    let mut dep_flag = OsString::from("-MT");
    let mut cxx20_modules = false;
    let mut common_args = vec![];
    let mut arch_args = vec![];
    let mut unhashed_args = vec![];
    let mut preprocessor_args = vec![];
    let mut dependency_args = vec![];
    let mut extra_hash_files = vec![];
    let mut extra_dist_files = vec![];
    let mut compilation = kind == CCompilerKind::Nvcc;
    let mut multiple_input = false;
    let mut multiple_input_files = Vec::new();
    let mut pedantic_flag = false;
    let mut language_extensions = true; // by default, GCC allows extensions
    let mut split_dwarf = false;
    let mut depfile_is_system_only = false;
    let mut need_explicit_dep_target = false;
    enum DepArgumentRequirePath {
        NotNeeded,
        Missing,
        Provided,
    }
    let mut need_explicit_dep_argument_path = DepArgumentRequirePath::NotNeeded;
    let mut language = None;
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

    let mut too_hard_for_preprocessor_cache_mode = vec![];

    let mut args_iter = ArgsIter::new(it, arg_info);
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
            Some(PedanticFlag) => pedantic_flag = true,
            // standard values vary, but extension values all start with "gnu"
            Some(Standard(version)) => language_extensions = version.starts_with("gnu"),
            Some(SplitDwarf) => split_dwarf = true,
            Some(DoCompilation) => {
                compilation = true;
                compilation_flag =
                    OsString::from(arg.flag_str().expect("Compilation flag expected"));
            }
            Some(EnableModules) => cxx20_modules = true,
            Some(ProfileGenerate) => profile_generate = true,
            Some(ClangProfileUse(path)) => {
                extra_hash_files.push(clang::resolve_profile_use_path(path, cwd));
            }
            Some(TestCoverage) => outputs_gcno = true,
            Some(Coverage) => {
                outputs_gcno = true;
                profile_generate = true;
            }
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
                if matches!(arg.flag_str(), Some("-MMD")) {
                    depfile_is_system_only = true;
                    too_hard_for_preprocessor_cache_mode.extend(arg.iter_os_strings());
                }
            }
            Some(DepTarget(s)) => {
                dep_flag = OsString::from(arg.flag_str().expect("Dep target flag expected"));
                dep_target = Some(s.clone());
            }
            Some(DepArgumentPath(path)) => {
                depfile = Some(path.clone());
                need_explicit_dep_argument_path = DepArgumentRequirePath::Provided;
            }
            Some(SerializeDiagnostics(path)) => {
                serialize_diagnostics = Some(path.clone());
            }
            Some(ExtraHashFile(_))
            | Some(ExtraHashFileClangModuleFile(_))
            | Some(PassThroughFlag)
            | Some(PreprocessorArgumentFlag)
            | Some(PreprocessorArgument(_))
            | Some(PreprocessorArgumentPath(_))
            | Some(PassThrough(_))
            | Some(PassThroughPath(_))
            | Some(UnhashedFlag)
            | Some(Unhashed(_))
            | Some(ModuleMapper(_))
            // TODO: parse the JSON instead of using `-MD -MF`?
            | Some(StructuredDepFormat(_))
            | Some(StructuredDepTarget(_))
            | Some(StructuredDepArgumentPath(_)) => {}
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
            Some(SplitDwarf)
            | Some(ModuleOnlyFlag)
            | Some(PedanticFlag)
            | Some(Standard(_))
            | Some(ProfileGenerate)
            | Some(ClangProfileUse(_))
            | Some(TestCoverage)
            | Some(Coverage)
            | Some(NoDiagnosticsColorFlag)
            | Some(PassThroughFlag)
            | Some(PassThrough(_))
            | Some(PassThroughPath(_))
            | Some(EnableModules)
            | Some(ModuleMapper(_))
            | Some(ClangModuleOutput(_)) => &mut common_args,
            Some(UnhashedFlag) | Some(Unhashed(_)) => &mut unhashed_args,
            Some(Arch(_)) => &mut arch_args,
            Some(ExtraHashFile(path)) => {
                extra_hash_files.push(cwd.join(path));
                &mut common_args
            }
            Some(ExtraHashFileClangModuleFile(val)) => {
                // -fmodule-file can be either "path" or "name=path"
                let val_str = val.to_string_lossy();
                let path = if let Some(idx) = val_str.find('=') {
                    PathBuf::from(&val_str[idx + 1..])
                } else {
                    PathBuf::from(val)
                };
                extra_hash_files.push(cwd.join(path));
                &mut common_args
            }
            Some(PreprocessorArgument(val)) => {
                if matches!(arg.flag_str(), Some("-Xpreprocessor") | Some("-Wp")) {
                    // `-Wp,-MD`, `-Wp,-MT`, and `-Wp,-MF` are not too hard for preprocessor cache mode.
                    // Instead, parse out the dependency file and target names, for example:
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
                        _ => {
                            too_hard_for_preprocessor_cache_mode.extend(arg.iter_os_strings());
                            &mut preprocessor_args
                        }
                    }
                } else {
                    &mut preprocessor_args
                }
            }
            Some(PreprocessorArgumentFlag) | Some(PreprocessorArgumentPath(_)) => {
                &mut preprocessor_args
            }
            Some(DepArgumentPath(_))
            | Some(NeedDepTarget)
            | Some(StructuredDepFormat(_))
            | Some(StructuredDepTarget(_))
            | Some(StructuredDepArgumentPath(_)) => &mut dependency_args,
            Some(DoCompilation)
            // Skip `-fdiagnostics-color[=auto]` flags and handle in `generate_compile_commands`
            | Some(DiagnosticsColor(_))
            | Some(DiagnosticsColorFlag)
            | Some(Language(_))
            | Some(Output(_))
            | Some(XClang(_))
            | Some(DepTarget(_))
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
                extra_hash_files.push(cwd.join(path));
                &mut common_args
            }
            Some(ExtraHashFileClangModuleFile(val)) => {
                // -fmodule-file can be either "path" or "name=path"
                let val_str = val.to_string_lossy();
                let path = if let Some(idx) = val_str.find('=') {
                    PathBuf::from(&val_str[idx + 1..])
                } else {
                    PathBuf::from(val)
                };
                extra_hash_files.push(cwd.join(path));
                &mut common_args
            }
            Some(PreprocessorArgumentFlag)
            | Some(PreprocessorArgument(_))
            | Some(PreprocessorArgumentPath(_)) => &mut preprocessor_args,
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
        l => l,
    };
    let language = match language {
        Some(l) => l,
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
    let suppress_rewrite_includes_only = match kind {
        CCompilerKind::Gcc => language_extensions && pedantic_flag,
        _ => false,
    };
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
        // If the language needs preprocessing, include the source file in the dist inputs.
        // This ensures gcc embeds the correct source and line numbers in warnings/errors.
        // See the docstring on the `PathTransformer::as_dist_input_path` impl for details.
        extra_dist_files.push(cwd.join(&input));

        if need_explicit_dep_target {
            dependency_args.push(dep_flag);
            dependency_args.push(dep_target.unwrap_or_else(|| output.clone().into_os_string()));
        }
        if let DepArgumentRequirePath::Missing = need_explicit_dep_argument_path {
            let dep_path = Path::new(&output).with_extension("d");
            depfile = Some(dep_path.clone());
            dependency_args.push(OsString::from("-MF"));
            dependency_args.push(dep_path.into_os_string());
        }

        if let Some(p) = depfile.as_ref().filter(|_| !depfile_is_system_only) {
            outputs.insert(
                "dep",
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

    CompilerArguments::Ok(ParsedArguments {
        input: input.into(),
        double_dash_input,
        language,
        compilation_flag,
        cxx20_modules,
        outputs,
        depfile,
        dependency_args,
        preprocessor_args,
        common_args,
        arch_args,
        unhashed_args,
        extra_dist_files,
        extra_hash_files,
        profile_generate,
        color_mode,
        suppress_rewrite_includes_only,
        too_hard_for_preprocessor_cache_mode,
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

#[allow(clippy::too_many_arguments)]
pub fn preprocess_cmd<T>(
    mut cmd: T,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
    kind: &CCompilerKind,
    rewrite_includes_only: bool,
    extra_preprocessor_flags: &[OsString],
) -> T
where
    T: RunCommand,
{
    cmd.current_dir(cwd).env_clear().envs(env_vars.to_vec());

    if let Some(language) = parsed_args.language.as_compiler_str(kind.into()) {
        cmd.arg("-x").arg(language);
    }

    if rewrite_includes_only {
        if parsed_args.suppress_rewrite_includes_only {
            trace!(
                "[{}]: preprocess: pedantic arguments disable rewrite_includes_only",
                parsed_args.output_pretty()
            );
        } else {
            match kind {
                CCompilerKind::Clang => {
                    cmd.arg("-frewrite-includes");
                }
                CCompilerKind::Gcc => {
                    cmd.arg("-fdirectives-only");
                }
                _ => {}
            }
        }
    }

    cmd.args(&parsed_args.preprocessor_args)
        //
        // Define the __PREPROCESS_ONLY__ macro when preprocessing so users can work
        // around clang __has_builtin() bugs when using distributed compilation.
        //
        // If an identifier with the same name as a builtin exists:
        // * __has_builtin() is true when preprocessing: https://godbolt.org/z/dsPbWGs5o
        // * __has_builtin() is false when compiling: https://godbolt.org/z/47azYd8so
        //
        .arg("-D__PREPROCESS_ONLY__")
        .args(&parsed_args.dependency_args)
        .args(&parsed_args.common_args)
        .args(
            (
                // Explicitly rewrite the -arch args to be preprocessor defines of the form
                // __arch__ so that they affect the preprocessor output but don't cause
                // clang to error.
                if kind == &CCompilerKind::Clang {
                    let rewritten_arch_args = parsed_args
                        .arch_args
                        .iter()
                        .filter(|&arg| arg.ne(clang::ARCH_FLAG))
                        .filter_map(|arg| {
                            arg.to_str()
                                .map(|arg_string| format!("-D__{arg_string}__=1").into())
                        })
                        .collect::<Vec<OsString>>();
                    let mut unique_rewritten = rewritten_arch_args.clone();
                    unique_rewritten.sort();
                    unique_rewritten.dedup();
                    if unique_rewritten.len() <= 1 {
                        // don't use rewritten arch args if there is only one arch
                        parsed_args.arch_args.clone()
                    } else {
                        trace!("-arch args before rewrite: {:?}", parsed_args.arch_args);
                        trace!("-arch args after rewrite:  {rewritten_arch_args:?}");
                        rewritten_arch_args
                    }
                } else {
                    parsed_args.arch_args.clone()
                }
            )
            .as_slice(),
        );

    // Some "gcc-like" compilers (QNX's qcc/q++) need an explicit `-o -`,
    // otherwise they write the preprocessed file to `<input>.i`
    if let CCompilerKind::Gcc = kind {
        cmd.arg("-o").arg("-");
    }

    cmd
        // older NVCC versions only support `-E` when it comes after preprocessor and common flags.
        .arg("-E")
        .args(extra_preprocessor_flags);

    if parsed_args.double_dash_input {
        cmd.arg("--");
    }

    cmd.arg(&parsed_args.input);

    cmd
}

#[allow(clippy::too_many_arguments)]
pub async fn preprocess<T>(
    _service: &SccacheService<T>,
    creator: &T,
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
    kind: CCompilerKind,
    rewrite_includes_only: bool,
    generate_dependencies: bool,
    extra_preprocessor_flags: &[OsString],
) -> Result<PreprocessorOutput>
where
    T: CommandCreatorSync,
{
    let cmd = preprocess_cmd(
        creator.clone().new_command_sync(executable),
        parsed_args,
        cwd,
        env_vars,
        &kind,
        rewrite_includes_only,
        extra_preprocessor_flags,
    );

    let (cmd, depfile) = if generate_dependencies {
        generate_dependencies_with_preprocess_cmd(
            creator,
            executable,
            parsed_args,
            cwd,
            env_vars,
            kind,
            cmd,
        )
        .await
        .map(|(cmd, depfile)| (cmd, Some(depfile)))?
    } else {
        (cmd, None)
    };

    trace!("[{}]: preprocess: {cmd}", parsed_args.output_pretty());
    trace!("[{}]: depfile: {depfile:?}", parsed_args.output_pretty());

    let output = run_input_stream_output(cmd, 8 * 1024, None).await?.boxed();

    let dependencies = depfile.map(|depfile| {
        parse_dependencies(cwd.to_owned(), parsed_args.input.clone(), depfile).boxed()
    });

    Ok(PreprocessorOutput::Output(output, dependencies))
}

pub async fn generate_dependencies<T>(
    creator: &T,
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
    kind: CCompilerKind,
) -> Result<DepfilePath>
where
    T: CommandCreatorSync,
{
    let depfile = if let Some(depfile) = parsed_args.depfile.as_deref() {
        DepfilePath::Path(cwd.join(depfile))
    } else {
        DepfilePath::Temp(temppath()?)
    };

    let cmd = generate_dependencies_cmd(
        creator,
        executable,
        parsed_args,
        cwd,
        env_vars,
        &kind,
        &depfile,
    );

    run_input_output(cmd, None).await?;

    Ok(depfile)
}

async fn generate_dependencies_with_preprocess_cmd<T>(
    creator: &T,
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
    kind: CCompilerKind,
    mut preprocess_command: T::Cmd,
) -> Result<(T::Cmd, DepfilePath)>
where
    T: CommandCreatorSync,
{
    let generate_deps_with_system_headers = parsed_args
        .dependency_args
        .iter()
        .any(|arg| arg.contains("-MD") || arg.contains("--write-dependencies"));

    let generate_deps_without_system_headers = parsed_args
        .dependency_args
        .iter()
        .any(|arg| arg.contains("-MMD") || arg.contains("--write-user-dependencies"));

    let depfile = if let Some(depfile) = parsed_args
        .depfile
        .as_deref()
        .filter(|_| generate_deps_with_system_headers)
        .filter(|_| !generate_deps_without_system_headers)
    {
        // If `-MD -MF <file>`, write all dependencies to <file> while preprocessing.
        DepfilePath::Path(cwd.join(depfile))
    } else {
        let temp = temppath()?;

        if generate_deps_without_system_headers {
            // If `-MMD` (with or without `-MF <file>`), generate and write all dependencies to a tempfile immediately.
            // We need dependencies with system includes for preprocessor cache mode, but the user potentially requested
            // dependencies *without* system includes be written to `parsed_args.depfile`. Those will be generated by the
            // preprocessor command already, so write deps with system includes to a tempfile so we can read them later.
            run_input_output(
                generate_dependencies_cmd(
                    creator,
                    executable,
                    parsed_args,
                    cwd,
                    env_vars,
                    &kind,
                    &temp,
                ),
                None,
            )
            .await?;
        } else {
            // If only `-MD`, or missing/invalid depflags, write all dependencies to a tempfile while preprocessing.
            if !generate_deps_with_system_headers {
                // Add missing `-MD` to preprocessor command
                if let CCompilerKind::Gcc = kind {
                    let mut md = OsString::new();
                    md.push("-Wp,-MD,");
                    md.push(temp.as_os_str());
                    preprocess_command.arg(md);
                } else {
                    preprocess_command.arg("-MD");
                }
            }
            // Add `-MF <temp>` to preprocessor command
            if let CCompilerKind::Gcc = kind {
                let mut mf = OsString::new();
                mf.push("-Wp,-MF,");
                mf.push(temp.as_os_str());
                preprocess_command.arg(mf);
            } else {
                preprocess_command.arg("-MF");
                preprocess_command.arg(&temp);
            }
        }

        DepfilePath::Temp(temp)
    };

    Ok((preprocess_command, depfile))
}

fn generate_dependencies_cmd<T>(
    creator: &T,
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
    kind: &CCompilerKind,
    depfile: &Path,
) -> T::Cmd
where
    T: CommandCreatorSync,
{
    // Replace existing dependency options with flags to generate and write all dependencies to `depfile`
    let mut dependency_args = vec![];

    if parsed_args.cxx20_modules {
        if let CCompilerKind::Gcc = kind {
            let mut md = OsString::new();
            md.push("-Wp,-MD,");
            md.push(depfile);
            dependency_args.push(md);
        } else {
            dependency_args.push("-MD".into());
        }
    } else {
        dependency_args.push("-M".into());
    }

    if let CCompilerKind::Gcc = kind {
        let mut mf = OsString::new();
        mf.push("-Wp,-MF,");
        mf.push(depfile);
        dependency_args.push(mf);
    } else {
        dependency_args.push("-MF".into());
        dependency_args.push(depfile.into());
    }

    let cmd = preprocess_cmd(
        creator.clone().new_command_sync(executable),
        &ParsedArguments {
            dependency_args,
            ..parsed_args.clone()
        },
        cwd,
        env_vars,
        kind,
        false,
        &[],
    );

    trace!("[{}]: dependencies: {}", parsed_args.output_pretty(), cmd);

    cmd
}

pub async fn parse_dependencies<P: AsRef<Path>>(
    cwd: P,
    input: P,
    depfile: DepfilePath,
) -> Result<Vec<PathBuf>> {
    let cwd = cwd.as_ref();
    let input_path = input.as_ref();

    let lines = tokio::fs::read(&depfile)
        .await
        // Avoid dropping Windows wide chars in paths
        .and_then(from_local_codepage)
        .with_context(|| format!("{depfile:?}"))?;

    let lines = lines
        .split("\n")
        .enumerate()
        // The path to the left of the colon is the `-MT <path>`, so take everything to the right.
        .map(|(idx, line)| {
            if idx == 0 {
                if let Some((_, rest)) = line.split_once(":") {
                    rest
                } else {
                    line
                }
            } else {
                line
            }
        })
        .map(|line| {
            // Trim whitespace and trailing backslashes + colons
            line.trim().trim_end_matches(r"\").trim_end_matches(r":")
        })
        .filter(|line| !line.is_empty());

    let paths = lines
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
            // Make relative paths absolute with respect to cwd
            Some(normalize_path(cwd.join(path)))
        })
        .collect::<Vec<_>>();

    Ok(paths)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_compile_commands(
    path_transformer: &mut dist::PathTransformer,
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
    kind: CCompilerKind,
    rewrite_includes_only: bool,
) -> Result<(
    SingleCompileCommand,
    Option<dist::CompileCommand>,
    Cacheable,
)> {
    // Unused arguments
    #[cfg(not(feature = "dist-client"))]
    {
        let _ = path_transformer;
        let _ = kind;
        let _ = rewrite_includes_only;
    }

    let out_pretty = parsed_args.output_pretty();

    let out_file = match parsed_args.outputs.get("obj") {
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
    arguments.extend(vec![
        parsed_args.compilation_flag.clone(),
        "-o".into(),
        out_file.into(),
    ]);

    let mut common_args = parsed_args.common_args.clone();

    arguments.extend_from_slice(&parsed_args.arch_args);
    arguments.extend_from_slice(&parsed_args.unhashed_args);
    arguments.extend_from_slice(&parsed_args.preprocessor_args);

    if parsed_args.language.needs_c_preprocessing() {
        // Only add dependency args for files that need preprocessing
        arguments.extend_from_slice(&parsed_args.dependency_args);

        // Rewrite `-fdiagnostics-color[=auto]` to `fdiagnostics-color=always`,
        // otherwise gcc disables colored output because stdout is not a tty.
        // See: https://github.com/mozilla/sccache/pull/2758
        if parsed_args.color_mode != ColorMode::Off {
            common_args.push("-fdiagnostics-color=always".into());
        }
    }

    arguments.extend_from_slice(&common_args);

    if parsed_args.double_dash_input {
        arguments.push("--".into());
    }
    arguments.push(parsed_args.input.clone().into());

    #[cfg(feature = "dist-client")]
    let has_verbose_flag = arguments.contains(&OsString::from("-v"))
        || arguments.contains(&OsString::from("--verbose"));

    let command = SingleCompileCommand {
        arguments,
        cwd: cwd.to_owned(),
        env_vars: env_vars.to_owned(),
        executable: executable.to_owned(),
        out_pretty: out_pretty.to_string(),
    };

    trace!("[{out_pretty}]: compile command: {command}");

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
            let command = dist::CompileCommand {
                cwd: path_transformer.as_dist_abs(cwd)?,
                env_vars: dist::osstring_tuples_to_strings(env_vars)?,
                executable: path_transformer.as_dist(executable)?,
                arguments: {
                    let mut language = language.map(|lang| lang.to_owned());

                    // https://gcc.gnu.org/onlinedocs/gcc-4.9.0/gcc/Overall-Options.html
                    if !rewrite_includes_only && !parsed_args.language.is_c_like_header() {
                        if let Some(processed_lang) =
                            parsed_args.language.to_c_preprocessed_language()
                        {
                            language = processed_lang
                                .as_compiler_str(kind.into())
                                .map(|lang| lang.to_owned());
                        } else if parsed_args.language.needs_c_preprocessing() {
                            if let CCompilerKind::Nvhpc = kind {
                                // -x=c|cpp|c++|i|cpp-output|asm|assembler|ASM|assembler-with-cpp|none
                                // Specify the language for any following input files, instead of letting
                                // the compiler choose based on suffix. Turn off with -x none
                                language = Some("none".into());
                            } else if let Some(lang) = language.as_mut() {
                                lang.push_str("-cpp-output");
                            }
                        }
                    }

                    let mut arguments: Vec<String> = vec![];
                    // Language needs to be before input
                    if let Some(lang) = &language {
                        arguments.extend_from_slice(&["-x".into(), lang.into()][..]);
                    }

                    arguments.extend(dist::osstrings_to_strings(&parsed_args.arch_args)?);
                    arguments.extend(dist::osstrings_to_strings(&common_args)?);

                    if let CCompilerKind::Gcc = kind {
                        // From https://gcc.gnu.org/onlinedocs/gcc/Preprocessor-Options.html:
                        //
                        // -fdirectives-only
                        //
                        //     [...]
                        //
                        //     With -fpreprocessed, predefinition of command line and most
                        //     builtin macros is disabled. Macros such as __LINE__, which
                        //     are contextually dependent, are handled normally. This
                        //     enables compilation of files previously preprocessed with -E
                        //     -fdirectives-only.
                        //
                        // Which is exactly what we do :-)
                        if rewrite_includes_only && !parsed_args.suppress_rewrite_includes_only {
                            arguments.push("-fdirectives-only".into());
                        }
                        arguments.push("-fpreprocessed".into());
                    }

                    // Escape hatch to work around compiler bugs when compiling preprocessed input.
                    // Sometimes code that doesn't warn in fused preprocess-compile mode does warn
                    // when dist-compiling the preprocessed file. That means local compiles will
                    // succeed, but the remote compile will warn (and then fail with `-Werror`).
                    //
                    // Since we can't predict all the warnings this may apply to, allow users to
                    // provide their own list like so:
                    // ```
                    // SCCACHE_DIST_DISABLE_WARNINGS=warning-1;warning-2
                    // ```
                    let warnings_to_ignore = if let Some(warnings) = env_vars
                        .iter()
                        .find(|(k, _)| k == "SCCACHE_DIST_DISABLE_WARNINGS")
                        .and_then(|(_, p)| p.to_str())
                    {
                        warnings
                            .split(";")
                            .filter(|warning| !warning.is_empty())
                            .collect::<Vec<_>>()
                    } else {
                        match kind {
                            CCompilerKind::Gcc => match parsed_args.language {
                                Language::Cxx | Language::ObjectiveC => {
                                    vec!["range-loop-construct"]
                                }
                                _ => vec![],
                            },
                            CCompilerKind::Clang => {
                                vec!["logical-op-parentheses", "reserved-user-defined-literal"]
                            }
                            _ => vec![],
                        }
                    };

                    arguments.extend_from_slice(
                        &warnings_to_ignore
                            .into_iter()
                            .map(|warning| format!("-Wno-{warning}"))
                            .collect::<Vec<_>>(),
                    );

                    arguments.extend_from_slice(
                        &[
                            parsed_args.compilation_flag.clone().into_string().ok()?,
                            if !parsed_args.language.needs_c_preprocessing() {
                                path_transformer.as_dist(&parsed_args.input)?
                            } else {
                                path_transformer.as_dist_input_path(&parsed_args.input)?
                            },
                            "-o".into(),
                            path_transformer.as_dist(out_file)?,
                        ][..],
                    );

                    arguments
                },
            };

            trace!("[{out_pretty}]: dist command: {command}");

            Some(command)
        })()
    };

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
            // argument". Currently we don't implement handling of arguments
            // with quotes, so if those are encountered we just pass the option
            // through literally anyway.
            //
            // At this time we interpret all `@` arguments above as non
            // cacheable, so if we fail to interpret this we'll just call the
            // compiler anyway.
            //
            // [1]: https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html#Overall-Options
            let mut contents = String::new();
            let res = File::open(&file).and_then(|mut f| f.read_to_string(&mut contents));
            if let Err(e) = res {
                debug!("failed to read @-file `{}`: {}", file.display(), e);
                return Some(arg);
            }
            if contents.contains('"') || contents.contains('\'') {
                return Some(arg);
            }
            let new_args = contents.split_whitespace().collect::<Vec<_>>();
            self.stack.extend(new_args.iter().rev().map(|s| s.into()));
        }
    }
}

#[cfg(test)]
mod test {
    use fs::File;
    use itertools::assert_equal;
    use std::io::Write;

    use super::*;
    use crate::compiler::*;
    use crate::mock_command::*;
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

    #[test]
    fn test_preprocess_cmd_rewrites_archs() {
        with_var("SCCACHE_CACHE_MULTIARCH", Some("1"), || {
            let args = stringvec!["-arch", "arm64", "-arch", "i386", "-c", "foo.cc"];
            let parsed_args = match parse_arguments_clang(args, false) {
                CompilerArguments::Ok(args) => args,
                o => panic!("Got unexpected parse result: {o:?}"),
            };
            let cmd = preprocess_cmd(
                MockCommand {
                    child: None,
                    args: vec![],
                },
                &parsed_args,
                Path::new(""),
                &[],
                &CCompilerKind::Clang,
                true,
                &[],
            );
            // make sure the architectures were rewritten to prepocessor defines
            let expected_args = ovec![
                "-x",
                "c++",
                "-frewrite-includes",
                "-D__PREPROCESS_ONLY__",
                "-D__arm64__=1",
                "-D__i386__=1",
                "-E",
                "foo.cc"
            ];
            assert_eq!(cmd.args, expected_args);
        });
    }

    #[test]
    fn test_preprocess_cmd_doesnt_rewrite_single_arch() {
        let args = stringvec!["-arch", "arm64", "-c", "foo.cc"];
        let parsed_args = match parse_arguments_clang(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        let cmd = preprocess_cmd(
            MockCommand {
                child: None,
                args: vec![],
            },
            &parsed_args,
            Path::new(""),
            &[],
            &CCompilerKind::Gcc,
            true,
            &[],
        );
        // make sure the architectures were rewritten to prepocessor defines
        let expected_args = ovec![
            "-x",
            "c++",
            "-fdirectives-only",
            "-D__PREPROCESS_ONLY__",
            "-arch",
            "arm64",
            "-o",
            "-",
            "-E",
            "foo.cc"
        ];
        assert_eq!(cmd.args, expected_args);
    }

    #[test]
    fn test_preprocess_double_dash_input() {
        let args = stringvec!["-c", "-o", "foo.o", "--", "foo.c"];
        let parsed_args = match parse_arguments_clang(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        let cmd = preprocess_cmd(
            MockCommand {
                child: None,
                args: vec![],
            },
            &parsed_args,
            Path::new(""),
            &[],
            &CCompilerKind::Clang,
            true,
            &[],
        );
        let expected_args = ovec![
            "-x",
            "c",
            "-frewrite-includes",
            "-D__PREPROCESS_ONLY__",
            "-E",
            "--",
            "foo.c"
        ];
        assert_eq!(cmd.args, expected_args);
    }

    #[test]
    fn pedantic_default() {
        let args = stringvec!["-pedantic", "-c", "foo.cc"];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        let cmd = preprocess_cmd(
            MockCommand {
                child: None,
                args: vec![],
            },
            &parsed_args,
            Path::new(""),
            &[],
            &CCompilerKind::Gcc,
            true,
            &[],
        );
        // disable with extensions enabled
        assert!(!cmd.args.contains(&"-fdirectives-only".into()));
    }

    #[test]
    fn pedantic_std() {
        let args = stringvec!["-pedantic-errors", "-c", "-std=c++14", "foo.cc"];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        let cmd = preprocess_cmd(
            MockCommand {
                child: None,
                args: vec![],
            },
            &parsed_args,
            Path::new(""),
            &[],
            &CCompilerKind::Gcc,
            true,
            &[],
        );
        // no reason to disable it with no extensions enabled
        assert!(cmd.args.contains(&"-fdirectives-only".into()));
    }

    #[test]
    fn pedantic_gnu() {
        let args = stringvec!["-pedantic-errors", "-c", "-std=gnu++14", "foo.cc"];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        let cmd = preprocess_cmd(
            MockCommand {
                child: None,
                args: vec![],
            },
            &parsed_args,
            Path::new(""),
            &[],
            &CCompilerKind::Gcc,
            true,
            &[],
        );
        // disable with extensions enabled
        assert!(!cmd.args.contains(&"-fdirectives-only".into()));
    }

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
        File::create(td.path().join("foo"))
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
        let storage = MockStorage::new(None, false);
        let storage: std::sync::Arc<MockStorage> = std::sync::Arc::new(storage);
        let service =
            SccacheService::mock_with_storage(storage.clone(), storage, runtime.handle().clone());
        let compiler = &f.bins[0];
        // Compiler invocation.
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        let mut path_transformer = dist::PathTransformer::new();
        let (command, dist_command, cacheable) = generate_compile_commands(
            &mut path_transformer,
            compiler,
            &parsed_args,
            f.tempdir.path(),
            &[],
            CCompilerKind::Gcc,
            false,
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
        let storage = MockStorage::new(None, false);
        let storage: std::sync::Arc<MockStorage> = std::sync::Arc::new(storage);
        let service =
            SccacheService::mock_with_storage(storage.clone(), storage, runtime.handle().clone());
        let compiler = &f.bins[0];
        // Compiler invocation.
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        let mut path_transformer = dist::PathTransformer::new();
        let (command, dist_command, cacheable) = generate_compile_commands(
            &mut path_transformer,
            compiler,
            &parsed_args,
            f.tempdir.path(),
            &[],
            CCompilerKind::Gcc,
            false,
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
        let storage = MockStorage::new(None, false);
        let storage: std::sync::Arc<MockStorage> = std::sync::Arc::new(storage);
        let service =
            SccacheService::mock_with_storage(storage.clone(), storage, runtime.handle().clone());
        let compiler = &f.bins[0];
        // Compiler invocation.
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        let mut path_transformer = dist::PathTransformer::new();
        let (command, dist_command, cacheable) = generate_compile_commands(
            &mut path_transformer,
            compiler,
            &parsed_args,
            f.tempdir.path(),
            &[],
            CCompilerKind::Gcc,
            false,
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
        let mut path_transformer = dist::PathTransformer::new();
        let (command, _, _) = generate_compile_commands(
            &mut path_transformer,
            compiler,
            &parsed_args,
            f.tempdir.path(),
            &[],
            CCompilerKind::Clang,
            false,
        )
        .unwrap();
        let expected_args = ovec![
            "-x",
            "c",
            "-c",
            "-o",
            "foo.o",
            "-fdiagnostics-color=always",
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

    #[test]
    fn test_pch_explicit() {
        let args = stringvec!["-c", "-x", "c++-header", "pch.h", "-o", "pch.pch"];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        let cmd = preprocess_cmd(
            MockCommand {
                child: None,
                args: vec![],
            },
            &parsed_args,
            Path::new(""),
            &[],
            &CCompilerKind::Gcc,
            true,
            &[],
        );
        assert!(cmd.args.contains(&"-x".into()) && cmd.args.contains(&"c++-header".into()));
    }

    #[test]
    fn test_pch_implicit() {
        let args = stringvec!["-c", "pch.hpp", "-o", "pch.pch"];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        let cmd = preprocess_cmd(
            MockCommand {
                child: None,
                args: vec![],
            },
            &parsed_args,
            Path::new(""),
            &[],
            &CCompilerKind::Gcc,
            true,
            &[],
        );
        assert!(cmd.args.contains(&"-x".into()) && cmd.args.contains(&"c++-header".into()));
    }

    #[test]
    fn test_pch_generic() {
        let args = stringvec!["-c", "pch.h", "-o", "pch.pch"];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        let cmd = preprocess_cmd(
            MockCommand {
                child: None,
                args: vec![],
            },
            &parsed_args,
            Path::new(""),
            &[],
            &CCompilerKind::Gcc,
            true,
            &[],
        );
        assert!(!cmd.args.contains(&"-x".into()));
    }

    #[test]
    fn test_too_hard_for_preprocessor_cache_mode() {
        let args = stringvec!["-c", "foo.c", "-o", "foo.o"];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert!(parsed_args.too_hard_for_preprocessor_cache_mode.is_empty());

        let args = stringvec!["-c", "foo.c", "-o", "foo.o", "-Xpreprocessor", "-M"];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(
            parsed_args.too_hard_for_preprocessor_cache_mode,
            vec!["-Xpreprocessor", "-M"]
        );

        let args = stringvec!["-c", "foo.c", "-o", "foo.o", r#"-Wp,-DFOO="something""#];
        let parsed_args = match parse_arguments_(args, false) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(
            parsed_args.too_hard_for_preprocessor_cache_mode,
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
