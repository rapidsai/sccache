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
        Cacheable, ColorMode, CompileCommandImpl, CompilerArguments,
        args::*,
        c::{
            CCompilerImpl, CCompilerKind, GenerateCompileCommandsArgs, GenerateDependenciesArgs,
            ParseArgs, ParsedArguments,
        },
        gcc::{self, ArgData::*},
    },
    errors::*,
    mock_command::{CommandCreatorSync, RunCommand},
    util::{bytes_to_str, run_input_output},
};
use crate::{counted_array, dist};
use async_trait::async_trait;
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    time::Duration,
};

/// A unit struct on which to implement `CCompilerImpl`.
#[derive(Clone, Debug)]
pub struct Nvhpc {
    /// true iff this is nvc++.
    pub nvcplusplus: bool,
    pub version: Option<String>,
    pub native_arch: Option<String>,
}

impl Nvhpc {
    pub async fn read_native_arch<T>(
        creator: &mut T,
        exe: &Path,
        env_vars: &[(OsString, OsString)],
    ) -> Option<String>
    where
        T: CommandCreatorSync,
    {
        use bytes::Buf;
        use std::io::BufRead;

        let mut cmd = creator.new_command_sync(exe);
        cmd.env_clear()
            .envs(env_vars.iter().map(|s| (&s.0, &s.1)))
            .arg("-showme:command");

        let exe = if let Ok(out) = run_input_output(cmd, None).await {
            which::which(
                // Remove the trailing newlines (if present)
                bytes_to_str(out.stdout).ok()?.trim(),
            )
            .ok()
            .unwrap_or_else(|| exe.to_path_buf())
        } else {
            exe.to_path_buf()
        };

        let mut cmd = creator.new_command_sync(&exe);
        cmd.env_clear()
            .envs(env_vars.iter().map(|s| (&s.0, &s.1)))
            .arg("-show");

        let output = run_input_output(cmd, None).await.ok()?;

        output
            .stdout
            .reader()
            .lines()
            .map_while(|line| line.ok())
            .find_map(|line| {
                let line = line.trim();
                // TESTTPVAL           =znver2
                if line.starts_with("TESTTPVAL") {
                    line.split_once('=')
                }
                // DEFTPVAL            =-tp znver2
                // INFOTPVAL           =-tp znver2
                // TPVAL               =-tp znver2
                // USETPVAL            =-tp znver2
                else if line.starts_with("DEFTPVAL")
                    || line.starts_with("INFOTPVAL")
                    || line.starts_with("TPVAL")
                    || line.starts_with("USETPVAL")
                {
                    line.split_once('=')
                        .and_then(|(_, args)| args.split_once(" "))
                } else {
                    None
                }
                .map(|(_, arch)| arch.trim())
                .filter(|arch| !arch.is_empty())
                .map(|arch| arch.to_owned())
            })
    }
}

#[async_trait]
impl CCompilerImpl for Nvhpc {
    fn kind(&self) -> CCompilerKind {
        CCompilerKind::Nvhpc
    }
    fn plusplus(&self) -> bool {
        self.nvcplusplus
    }
    fn version(&self) -> Option<String> {
        self.version.clone()
    }
    fn parse_arguments(
        &self,
        ParseArgs { arguments, cwd, .. }: ParseArgs<'_>,
    ) -> CompilerArguments<ParsedArguments> {
        let mut parsed_args = gcc::parse_arguments(
            arguments,
            cwd,
            (&gcc::ARGS[..], &ARGS[..]),
            self.nvcplusplus,
            self.kind(),
        );
        if let CompilerArguments::Ok(ref mut parsed_args) = parsed_args {
            // Insert or rewrite -tp=native to the current CPU architecture
            // to ensure the correct architecture is used if dist compiling
            if let Some(native_arch) = self.native_arch.as_ref() {
                if parsed_args.arch_args.is_empty() {
                    parsed_args
                        .arch_args
                        .push(format!("-tp={native_arch}").into());
                } else {
                    for arch in parsed_args.arch_args.iter_mut() {
                        if arch == "-tp=native" {
                            *arch = format!("-tp={native_arch}").into();
                        } else if arch == "-march=native" {
                            *arch = format!("-march={native_arch}").into();
                        }
                    }
                }
            }
            // nvc does not support -fdiagnostics-color=always
            if !self.nvcplusplus {
                parsed_args.color_mode = ColorMode::Off;
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
        gcc::generate_dependencies(
            creator,
            executable,
            parsed_args,
            cwd,
            env_vars.iter().cloned(),
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
        gcc::generate_compile_commands(executable, parsed_args, cwd, env_vars, self.kind())
    }
}

counted_array!(pub static ARGS: [ArgInfo<gcc::ArgData>; _] = [
    //todo: refactor show_includes into dependency_args
    take_arg!("--gcc-toolchain", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("--include-path", PathBuf, CanBeSeparated, NvhpcIncludePath),
    take_arg!("--include_directory", PathBuf, CanBeSeparated, NvhpcIncludePath),
    take_arg!("--linker-options", OsString, CanBeSeparated, PassThrough),
    flag!("--nvcchost", PassThroughFlag),
    take_arg!("--preinclude", PathBuf, CanBeSeparated, NvhpcIncludeFile),
    take_arg!("--system-include-path", PathBuf, CanBeSeparated, NvhpcSystemIncludePath),

    take_arg!("-Mconcur", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("-Mnostdlib", PreprocessorArgumentFlag),
    flag!("-Werror", PreprocessorArgumentFlag),
    take_arg!("-Werror=", OsString, Concatenated, PreprocessorArgument),
    take_arg!("-Xcompiler", OsString, CanBeSeparated(b'='), PreprocessorArgument),
    take_arg!("-Xfatbinary", OsString, CanBeSeparated, PassThrough),
    take_arg!("-Xlinker", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-Xnvlink", OsString, CanBeSeparated, PassThrough),
    take_arg!("-Xptxas", OsString, CanBeSeparated, PassThrough),
    take_arg!("-acc", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("-acclibs", PassThroughFlag),
    take_arg!("-c++", OsString, Concatenated, Standard),
    flag!("-c++libs", PassThroughFlag),
    flag!("-cuda", PreprocessorArgumentFlag),
    flag!("-cudaforlibs", PassThroughFlag),
    take_arg!("-cudalib", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("-fortranlibs", PassThroughFlag),
    flag!("-gopt", PassThroughFlag),
    take_arg!("-gpu", OsString, Concatenated(b'='), PassThrough),
    take_arg!("-mcmodel", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-mcpu", OsString, CanBeSeparated(b'='), PassThrough),
    flag!("-noswitcherror", PassThroughFlag),
    take_arg!("-ta", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-target", OsString, CanBeSeparated(b'='), PassThrough),
    take_arg!("-tp", OsString, CanBeSeparated(b'='), Arch),
    take_arg!("-x", OsString, CanBeSeparated(b'='), Language)
]);

#[cfg(test)]
mod test {
    use super::*;
    use crate::compiler::c::ArtifactDescriptor;
    use crate::compiler::{Language, *};

    fn parse_arguments_(arguments: Vec<String>) -> CompilerArguments<ParsedArguments> {
        let arguments = arguments.iter().map(OsString::from).collect::<Vec<_>>();
        Nvhpc {
            nvcplusplus: false,
            version: None,
            native_arch: None,
        }
        .parse_arguments(ParseArgs {
            arguments: &arguments,
            cwd: ".".as_ref(),
            env_vars: &[],
        })
    }

    macro_rules! parses {
        ( $( $s:expr ),* ) => {
            match parse_arguments_(vec![ $( $s.to_string(), )* ]) {
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
        assert!(a.common_args.is_empty());
    }

    #[test]
    fn test_parse_arguments_simple_cxx() {
        let a = parses!("-c", "foo.cxx", "-o", "foo.o");
        assert_eq!(Some("foo.cxx"), a.input.to_str());
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
        assert!(a.preprocessor_args.is_empty());
        assert!(a.common_args.is_empty());
    }

    #[test]
    fn test_parse_arguments_values() {
        let a = parses!(
            "-c",
            "-fabc",
            "--preinclude",
            "preinclude-file",
            "-I",
            "include-path1",
            "-o",
            "foo.o",
            "--include-path",
            "include-path2",
            "-isystem",
            "/system/include/file",
            "-gpu=ccnative",
            "-Werror=an_error",
            "-Werror",
            "foo.cpp"
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
                "-include",
                "preinclude-file",
                "-Iinclude-path1",
                "-I",
                "include-path2",
                "-isystem",
                "/system/include/file",
                "-Werror=an_error",
                "-Werror"
            ],
            a.preprocessor_args
        );
        assert!(a.dependency_args.is_empty());
        assert_eq!(ovec!["-fabc", "-gpu=ccnative"], a.common_args);
    }

    #[test]
    fn test_parse_md_mt_flags_cxx() {
        let a = parses!(
            "-x", "c++", "-c", "foo.c", "-fabc", "-MD", "-MT", "foo.o", "-MF", "foo.o.d", "-o",
            "foo.o"
        );
        assert_eq!(Some("foo.c"), a.input.to_str());
        assert_eq!(Language::Cxx, a.language);
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
        assert_eq!(ovec!["-fabc"], a.common_args);
    }

    #[test]
    fn test_parse_generate_code_flags() {
        let a = parses!(
            "-x",
            "c++",
            "-cuda",
            "-gpu=cc60,cc70",
            "-c",
            "foo.c",
            "-o",
            "foo.o"
        );
        assert_eq!(Some("foo.c"), a.input.to_str());
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
        assert_eq!(ovec!["-cuda"], a.preprocessor_args);
        assert_eq!(ovec!["-gpu=cc60,cc70"], a.common_args);
    }

    #[test]
    fn test_parse_generate_code_mem_flags() {
        let a = parses!(
            "-x",
            "c++",
            "-cuda",
            "-gpu=mem:separate",
            "-c",
            "foo.c",
            "-o",
            "foo.o"
        );
        assert_eq!(Some("foo.c"), a.input.to_str());
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
        assert_eq!(ovec!["-cuda"], a.preprocessor_args);
        assert_eq!(ovec!["-gpu=mem:separate"], a.common_args);
    }

    // #[test]
    // fn test_parse_cant_cache_flags() {
    //     assert_eq!(
    //         CompilerArguments::CannotCache("-E", None),
    //         parse_arguments_(stringvec!["-c", "foo.c", "-o", "foo.o", "-E"])
    //     );
    //     assert_eq!(
    //         CompilerArguments::CannotCache("-M", None),
    //         parse_arguments_(stringvec!["-c", "foo.c", "-o", "foo.o", "-M"])
    //     );
    // }
}
