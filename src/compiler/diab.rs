// Copyright 2018 Mozilla Foundation
// Copyright 2018 Felix Obenhuber <felix@obenhuber.de>
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
        args::{
            ArgDisposition, ArgInfo, ArgToStringResult, ArgsIter, Argument, FromArg, IntoArg,
            NormalizedDisposition, PathTransformerFn, SearchableArgInfo,
        },
        c::{
            AnyCompileCommand, ArtifactDescriptor, CCompilerImpl, CCompilerKind,
            GenerateCompileCommandsArgs, GenerateDependenciesArgs, ParseArgs, ParsedArguments,
        },
        gcc,
    },
    counted_array, dist,
    errors::*,
    mock_command::{CommandCreatorSync, RunCommand},
    util::{OsStrExt, bytes_to_str, run_input_output_timed},
};
use async_trait::async_trait;
use fs_err as fs;
use itertools::Itertools;
use std::{
    collections::HashMap,
    ffi::{OsStr, OsString},
    io::Read,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Debug)]
pub struct Diab {
    pub version: Option<String>,
}

#[async_trait]
impl CCompilerImpl for Diab {
    fn kind(&self) -> CCompilerKind {
        CCompilerKind::Diab
    }

    fn plusplus(&self) -> bool {
        false
    }

    fn version(&self) -> Option<String> {
        self.version.clone()
    }

    fn parse_arguments(
        &self,
        ParseArgs { arguments, cwd, .. }: ParseArgs<'_>,
    ) -> CompilerArguments<ParsedArguments> {
        parse_arguments(arguments, cwd, &ARGS[..])
    }

    async fn generate_dependencies<T>(
        &self,
        GenerateDependenciesArgs {
            creator,
            executable,
            parsed_args,
            cwd,
            env_vars,
            ..
        }: GenerateDependenciesArgs<'_, T>,
    ) -> Result<(Vec<PathBuf>, Duration)>
    where
        T: CommandCreatorSync,
    {
        generate_dependencies(creator, executable, parsed_args, cwd, env_vars).await
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
        impl CompileCommandImpl + Into<AnyCompileCommand>,
        Option<dist::CompileCommand>,
        Cacheable,
    )> {
        generate_compile_commands(executable, parsed_args, cwd, env_vars)
    }
}

ArgData! { pub
    DoCompilation,
    Output(PathBuf),
    PassThrough(OsString),
    PreprocessorArgument(OsString),
    PreprocessorArgumentPath(PathBuf),
    DepArgumentFlag,
    DepArgument(OsString),
    DepArgumentPath(PathBuf),
    TooHardFlag,
    TooHard(OsString),
}

use self::ArgData::*;

counted_array!(pub static ARGS: [ArgInfo<ArgData>; _] = [
    flag!("-", TooHardFlag),
    flag!("-##", TooHardFlag),
    flag!("-###", TooHardFlag),
    take_arg!("-@", OsString, Concatenated, TooHard),
    take_arg!("-D", OsString, CanBeSeparated, PreprocessorArgument),
    flag!("-E", TooHardFlag),
    take_arg!("-I", PathBuf, CanBeSeparated, PreprocessorArgumentPath),
    take_arg!("-L", OsString, Separated, PassThrough),
    flag!("-P", TooHardFlag),
    flag!("-S", TooHardFlag),
    take_arg!("-U", OsString, CanBeSeparated, PreprocessorArgument),
    flag!("-V", TooHardFlag),
    flag!("-VV", TooHardFlag),
    take_arg!("-W", OsString, Separated, PassThrough),
    flag!("-X156", DepArgumentFlag),
    take_arg!("-X156=", OsString, Concatenated, DepArgument),
    flag!("-Xmake-dependency", DepArgumentFlag),
    flag!(
        "-Xmake-dependency-canonicalize-path-off",
        DepArgumentFlag
    ),
    take_arg!(
        "-Xmake-dependency-savefile",
        PathBuf,
        Concatenated(b'='),
        DepArgumentPath
    ),
    take_arg!(
        "-Xmake-dependency-target",
        OsString,
        Concatenated(b'='),
        DepArgument
    ),
    take_arg!("-Xmake-dependency=", OsString, Concatenated, DepArgument),
    flag!("-c", DoCompilation),
    take_arg!(
        "-include",
        PathBuf,
        CanBeSeparated,
        PreprocessorArgumentPath
    ),
    take_arg!("-l", OsString, Separated, PassThrough),
    take_arg!("-o", PathBuf, Separated, Output),
    take_arg!("-t", OsString, Separated, PassThrough),
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
) -> CompilerArguments<ParsedArguments>
where
    S: SearchableArgInfo<ArgData>,
{
    let mut common_args = vec![];
    let mut compilation = false;
    let mut compilation_flag = OsString::new();
    let mut input_arg = None;
    let mut multiple_input = false;
    let mut output_arg = None;
    let mut preprocessor_args = vec![];
    let mut dependency_args = vec![];
    let mut depfile = None;

    // Custom iterator to expand `@` arguments which stand for reading a file
    // and interpreting it as a list of more arguments.
    let it = ExpandAtArgs::new(cwd, arguments);

    for arg in ArgsIter::new(it, arg_info) {
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
            Some(TooHardFlag) | Some(TooHard(_)) => {
                cannot_cache!(arg.flag_str().expect("Can't be Argument::Raw/UnknownFlag",))
            }

            Some(DepArgument(_)) | Some(DepArgumentFlag) => {}
            Some(DepArgumentPath(p)) => depfile = Some(p.clone()),

            Some(DoCompilation) => {
                compilation = true;
                compilation_flag =
                    OsString::from(arg.flag_str().expect("Compilation flag expected"));
            }
            Some(Output(p)) => output_arg = Some(p.clone()),
            Some(PreprocessorArgument(_))
            | Some(PreprocessorArgumentPath(_))
            | Some(PassThrough(_)) => {}
            None => match arg {
                Argument::Raw(ref val) => {
                    if input_arg.is_some() {
                        multiple_input = true;
                    }
                    input_arg = Some(val.clone());
                }
                Argument::UnknownFlag(_) => {}
                _ => unreachable!(),
            },
        }
        let args = match arg.get_data() {
            Some(PassThrough(_)) => &mut common_args,
            Some(DepArgument(_)) | Some(DepArgumentFlag) | Some(DepArgumentPath(_)) => {
                &mut dependency_args
            }
            Some(PreprocessorArgument(_)) | Some(PreprocessorArgumentPath(_)) => {
                &mut preprocessor_args
            }
            Some(DoCompilation) | Some(Output(_)) => continue,
            Some(TooHardFlag) | Some(TooHard(_)) => unreachable!(),
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

    // We only support compilation.
    if !compilation {
        return CompilerArguments::NotCompilation;
    }
    // Can't cache compilations with multiple inputs.
    if multiple_input {
        cannot_cache!("multiple input files");
    }
    let input = match input_arg {
        Some(i) => i,
        // We can't cache compilation without an input.
        None => cannot_cache!("no input file"),
    };
    let language = match Language::from_file_name(Path::new(&input)) {
        Some(l) => l,
        None => cannot_cache!("unknown source language"),
    };

    let output = output_arg.unwrap_or_else(|| Path::new(&input).with_extension("o"));

    let mut outputs = HashMap::new();
    outputs.insert(
        "obj",
        ArtifactDescriptor {
            path: output,
            optional: false,
        },
    );

    if let Some(ref p) = depfile {
        outputs.insert(
            "dep",
            ArtifactDescriptor {
                path: p.clone(),
                optional: false,
            },
        );
    }

    CompilerArguments::Ok(ParsedArguments {
        input: input.into(),
        language,
        compilation_flag,
        depfile,
        outputs,
        dependency_args,
        preprocessor_args,
        common_args,
        // FIXME: Implement me.
        color_mode: ColorMode::Auto,
        ..Default::default()
    })
}

async fn generate_dependencies<T>(
    creator: &T,
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
) -> Result<(Vec<PathBuf>, Duration)>
where
    T: CommandCreatorSync,
{
    let mut elapsed = Duration::default();

    let generate_user_dependencies = parsed_args.dependency_args.iter().any(|arg| {
        if (arg.contains("-X156=") || arg.contains("-Xmake-dependency="))
            && let Some((_, mask)) = arg.split_once("=")
        {
            // http://bitsavers.informatik.uni-stuttgart.de/pdf/windRiver/Compiler_5.4_2007/wr_compiler_users_guide_mips_5.4.pdf
            // -Xmake-dependency=2
            //     Same as -Xmake-dependency=1, but show only files enclosed in double
            //     quotation marks (#include "filename").
            // -Xmake-dependency=6
            //     Show only files enclosed in double quotation marks (-Xmake-dependency=2);
            //     write output to a file, then continue with normal compilation
            //     (-Xmake-dependency=4).
            // -Xmake-dependency=a
            //     Show only files in double quotation marks (-Xmake-dependency=2) and
            //     output phony targets (-Xmake-dependency=8).
            // -Xmake-dependency=e
            //     Show only files enclosed in double quotation marks (-Xmake-dependency=2);
            //     output phony targets (-Xmake-dependency=8); write output to a file, then
            //     continue with normal compilation (-Xmake-dependency=4).
            matches!(mask.to_str(), Some("2" | "6" | "a" | "e"))
        } else {
            false
        }
    });

    let args_without_dependency_flags = std::iter::empty()
        .chain(parsed_args.preprocessor_args.iter())
        .chain(parsed_args.common_args.iter())
        .map(|arg| arg.as_os_str())
        .filter(|&arg| arg != parsed_args.compilation_flag)
        .collect::<Vec<_>>();

    if generate_user_dependencies && parsed_args.depfile.is_some() {
        let args = args_without_dependency_flags
            .clone()
            .into_iter()
            .chain(
                parsed_args
                    .dependency_args
                    .iter()
                    .map(|arg| {
                        if arg.contains("-X156=") || arg.contains("-Xmake-dependency=") {
                            OsStr::new("-Xmake-dependency=2")
                        } else {
                            arg.as_os_str()
                        }
                    })
                    .unique(),
            )
            .chain(std::iter::once(parsed_args.input.as_os_str()));

        let mut cmd = creator.clone().new_command_sync(executable);
        cmd.current_dir(cwd).env_clear().envs(env_vars.to_vec());

        for arg in args {
            cmd.arg(arg);
        }

        trace!(
            "[{}]: user dependencies: {cmd}",
            parsed_args.output_pretty(),
        );

        let (_, duration) = run_input_output_timed(cmd, None)
            .await
            .map_err(|(err, _)| err)?;

        elapsed += duration;
    }

    let depfile = parsed_args
        .depfile
        .as_deref()
        .filter(|_| !generate_user_dependencies);

    let cmd = {
        let args = args_without_dependency_flags
            .into_iter()
            .chain(
                std::iter::once(OsStr::new("-Xmake-dependency"))
                    .chain(depfile.iter().flat_map(|depfile| {
                        [
                            OsStr::new("-Xmake-dependency-savefile"),
                            depfile.as_os_str(),
                        ]
                    }))
                    .chain(
                        parsed_args
                            .dependency_args
                            .iter()
                            // If this is generating the real dependency file,
                            // pass all the user's other dependency flags.
                            .filter(|_| depfile.is_some())
                            .map(|arg| {
                                // Map all depflags to `-Xmake-dependency`
                                if arg.contains("-X156=") || arg.contains("-Xmake-dependency=") {
                                    OsStr::new("-Xmake-dependency")
                                } else {
                                    arg.as_os_str()
                                }
                            }),
                    )
                    .unique(),
            )
            .chain(std::iter::once(parsed_args.input.as_os_str()));

        let mut cmd = creator.clone().new_command_sync(executable);
        cmd.current_dir(cwd).env_clear().envs(env_vars.to_vec());

        for arg in args {
            cmd.arg(arg);
        }

        cmd
    };

    trace!("[{}]: dependencies: {cmd}", parsed_args.output_pretty());

    let (output, duration) = run_input_output_timed(cmd, None)
        .await
        .map_err(|(err, _)| err)?;

    elapsed += duration;

    gcc::parse_dependencies(
        &parsed_args.input,
        None,
        &if let Some(depfile) = depfile {
            gcc::read_depfile(cwd.join(depfile)).await?
        } else {
            bytes_to_str(output.stdout)?
        },
    )
    .map(|dependencies| (dependencies, elapsed))
}

pub fn generate_compile_commands(
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
) -> Result<(
    SingleCompileCommand,
    Option<dist::CompileCommand>,
    Cacheable,
)> {
    let out_pretty = parsed_args.output_pretty();
    let out_file = match parsed_args.outputs.get("obj") {
        Some(obj) => &obj.path,
        None => return Err(anyhow!("Missing object file output")),
    };

    let mut arguments = vec![];

    if !parsed_args.compilation_flag.is_empty() {
        arguments.push(parsed_args.compilation_flag.clone());
    }

    arguments.extend([
        parsed_args.input.clone().into(),
        "-o".into(),
        out_file.into(),
    ]);

    arguments.extend_from_slice(&parsed_args.preprocessor_args);
    arguments.extend_from_slice(&parsed_args.unhashed_args);
    arguments.extend_from_slice(&parsed_args.common_args);
    let command = SingleCompileCommand {
        arguments,
        cwd: cwd.to_owned(),
        env_vars: env_vars.to_owned(),
        executable: executable.to_owned(),
        out_pretty: out_pretty.to_string(),
        share_jobserver: false,
    };

    trace!("[{out_pretty}]: command: {command}");

    Ok((command, None, Cacheable::Yes))
}

pub struct ExpandAtArgs<'a> {
    cwd: &'a Path,
    stack: Vec<OsString>,
}

impl<'a> ExpandAtArgs<'a> {
    pub fn new(cwd: &'a Path, args: &[OsString]) -> Self {
        ExpandAtArgs {
            stack: args.iter().rev().map(|a| a.to_owned()).collect(),
            cwd,
        }
    }
}

impl Iterator for ExpandAtArgs<'_> {
    type Item = OsString;

    fn next(&mut self) -> Option<OsString> {
        loop {
            let arg = self.stack.pop()?;

            // Just return non @ arguments
            if !arg.starts_with("-@") {
                return Some(arg);
            }

            let value = match arg.strip_prefix("-@") {
                Some(arg) => arg,
                None => return Some(arg),
            };

            // Return options that produce additional output and are not cacheable
            if value.starts_with("E") || value.starts_with("O") || value.starts_with("@") {
                return Some(arg);
            }

            // According to diab [1], @file means:
            // Read command line options from either a file or an environment
            // variable. When -@name is encountered on the command line, the
            // driver first looks for an environment variable with the given
            // name and substitutes its value. If an environment variable is
            // not found then the driver tries to open a file with given name
            // and substitutes the contents of the file. If neither an
            // environment variable or a file can be found, an error message
            // is issued and the driver terminates.
            //
            // [1]: http://www.vxdev.com/docs/vx55man/diab5.0ppc/c-invoke.htm#3000619
            //
            // The environment variable feature is *not* supported by sccache
            // since this would raise the need for the clients environment
            // and not just env::var. This is technically possible, but
            // considered as a unneeded edge case for now.

            let mut contents = String::new();
            let file = self.cwd.join(value);
            let res = fs::File::open(file).and_then(|mut f| f.read_to_string(&mut contents));
            if res.is_err() {
                // Failed to read the file, so return the argument as it is.
                // This will result in a CannotCache.
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
    use super::{
        ARGS, Language, OsString, ParsedArguments, fs, generate_compile_commands, parse_arguments,
    };
    use crate::compiler::c::ArtifactDescriptor;
    use crate::compiler::*;
    use crate::mock_command::*;
    use crate::server;
    use crate::test::mock_storage::MockStorage;
    use crate::test::utils::*;
    use fs::File;
    use std::io::Write;

    fn parse_arguments_(arguments: Vec<String>) -> CompilerArguments<ParsedArguments> {
        let args = arguments.iter().map(OsString::from).collect::<Vec<_>>();
        parse_arguments(&args, ".".as_ref(), &ARGS[..])
    }

    #[test]
    fn test_parse_arguments_simple() {
        let args = stringvec!["-c", "foo.c", "-o", "foo.o"];
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(args) {
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
        } = match parse_arguments_(args) {
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
        } = match parse_arguments_(args) {
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
        } = match parse_arguments_(args) {
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
    fn test_parse_arguments_preprocessor_args() {
        let args = stringvec![
            "-c",
            "foo.c",
            "-fabc",
            "-Xmake-dependency",
            "-Xmake-dependency-canonicalize-path-off",
            "-Xmake-dependency-savefile=bar",
            "-Xmake-dependency-target=foo",
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
        } = match parse_arguments_(args) {
            CompilerArguments::Ok(args) => args,
            o => panic!("Got unexpected parse result: {o:?}"),
        };
        assert_eq!(Some("foo.c"), input.to_str());
        assert_eq!(Language::C, language);
        assert_eq!(Some("bar"), depfile.unwrap().to_str());
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
                    path: "bar".into(),
                    optional: false,
                }
            )
        );
        assert_eq!(
            ovec![
                "-Xmake-dependency",
                "-Xmake-dependency-canonicalize-path-off",
                "-Xmake-dependency-savefile=bar",
                "-Xmake-dependency-target=foo"
            ],
            dependency_args
        );
        assert_eq!(ovec!["-fabc"], common_args);
        assert!(!msvc_show_includes);
    }

    #[test]
    fn test_parse_arguments_empty_args() {
        assert_eq!(CompilerArguments::NotCompilation, parse_arguments_(vec![]));
    }

    #[test]
    fn test_parse_arguments_not_compile() {
        assert_eq!(
            CompilerArguments::NotCompilation,
            parse_arguments_(stringvec!["-o", "foo"])
        );
    }

    #[test]
    fn test_parse_arguments_too_many_inputs() {
        assert_eq!(
            CompilerArguments::CannotCache("multiple input files", None),
            parse_arguments_(stringvec!["-c", "foo.c", "-o", "foo.o", "bar.c"])
        );
    }

    #[test]
    fn test_parse_arguments_link() {
        assert_eq!(
            CompilerArguments::NotCompilation,
            parse_arguments_(stringvec!["-shared", "foo.o", "-o", "foo.so", "bar.o"])
        );
    }

    #[test]
    fn test_parse_dry_run() {
        assert_eq!(
            CompilerArguments::CannotCache("-##", None),
            parse_arguments_(stringvec!["-##", "-c", "foo.c"])
        );

        assert_eq!(
            CompilerArguments::CannotCache("-###", None),
            parse_arguments_(stringvec!["-###", "-c", "foo.c"])
        );
    }

    #[test]
    fn test_at_signs() {
        let cannot_cache = CompilerArguments::CannotCache("-@", None);
        assert_eq!(parse_arguments_(vec!["-@@foo".into()]), cannot_cache);
        assert_eq!(parse_arguments_(vec!["-@E=foo".into()]), cannot_cache);
        assert_eq!(parse_arguments_(vec!["-@E+foo".into()]), cannot_cache);
        assert_eq!(parse_arguments_(vec!["-@O=foo".into()]), cannot_cache);
        assert_eq!(parse_arguments_(vec!["-@O+foo".into()]), cannot_cache);
    }

    #[test]
    fn test_at_signs_file_not_readable() {
        let td = crate::util::temp_dir().unwrap();
        let arg = format!("-@{}", td.path().join("foo").display());
        // File foo doesn't exist.
        assert_eq!(
            parse_arguments_(vec![arg]),
            CompilerArguments::CannotCache("-@", None)
        );
    }

    #[test]
    fn test_at_signs_file() {
        let td = crate::util::temp_dir().unwrap();
        File::create(td.path().join("foo"))
            .unwrap()
            .write_all(b"-c foo.c -o foo.o")
            .unwrap();
        let arg = format!("-@{}", td.path().join("foo").display());
        let ParsedArguments {
            input,
            language,
            outputs,
            preprocessor_args,
            msvc_show_includes,
            common_args,
            ..
        } = match parse_arguments_(vec![arg]) {
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
            ..Default::default()
        };
        let runtime = single_threaded_runtime();
        let storage = MockStorage::new(None, true);
        let storage: std::sync::Arc<MockStorage> = std::sync::Arc::new(storage);
        let service = server::SccacheService::mock_with_storage(storage, runtime.handle().clone());
        let compiler = &f.bins[0];
        // Compiler invocation.
        next_command(&creator, Ok(MockChild::new(exit_status(0), "", "")));
        let (command, _, cacheable) =
            generate_compile_commands(compiler, &parsed_args, f.tempdir.path(), &[]).unwrap();
        let _ = command.execute(&service, &creator).wait();
        assert_eq!(Cacheable::Yes, cacheable);
        // Ensure that we ran all processes.
        assert_eq!(0, creator.lock().unwrap().children.len());
    }
}
