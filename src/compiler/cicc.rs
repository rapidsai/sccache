// Copyright 2016 Mozilla Foundation
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        Cacheable, CompileCommandImpl, CompilerArguments, Language, SingleCompileCommand,
        args::*,
        c::{
            AnyCompileCommand, ArtifactDescriptor, CCompilerImpl, CCompilerKind,
            GenerateCompileCommandsArgs, GenerateDependenciesArgs, ParseArgs, ParsedArguments,
        },
    },
    counted_array, dist,
    errors::*,
    mock_command::CommandCreatorSync,
    util::OsStrExt,
};
use async_trait::async_trait;
use std::{
    collections::HashMap,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    time::Duration,
};

/// A unit struct on which to implement `CCompilerImpl`.
#[derive(Clone, Debug)]
pub struct Cicc {
    pub version: Option<String>,
}

#[async_trait]
impl CCompilerImpl for Cicc {
    fn kind(&self) -> CCompilerKind {
        CCompilerKind::Cicc
    }
    fn plusplus(&self) -> bool {
        true
    }
    fn version(&self) -> Option<String> {
        self.version.clone()
    }
    fn parse_arguments(
        &self,
        ParseArgs { arguments, cwd, .. }: ParseArgs<'_>,
    ) -> CompilerArguments<ParsedArguments> {
        parse_arguments(arguments, cwd, Language::Ptx, &ARGS[..])
    }
    async fn generate_dependencies<T>(
        &self,
        GenerateDependenciesArgs { .. }: GenerateDependenciesArgs<'_, T>,
    ) -> Result<(Vec<PathBuf>, Duration)>
    where
        T: CommandCreatorSync,
    {
        Ok((Vec::with_capacity(0), Duration::default()))
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
        generate_compile_commands(executable, parsed_args, cwd, env_vars, "-o")
    }
}

pub fn parse_arguments<S>(
    arguments: &[OsString],
    cwd: &Path,
    language: Language,
    arg_info: S,
) -> CompilerArguments<ParsedArguments>
where
    S: Clone + SearchableArgInfo<ArgData>,
{
    // Initial parse just to get the input and output
    let (input, mut outputs, mut args) = parse_input_outputs(
        arguments.iter().cloned(),
        arg_info.clone(),
        |arg| match arg.get_data() {
            Some(Output(p)) => Some(p),
            _ => None,
        },
    );

    let input = match input {
        Some(input) => input,
        // We can't cache compilation without an input.
        None => return CompilerArguments::CannotCache("no input file", None),
    };

    // Remove the input from the args list
    if let Some(idx) = args.iter().position(|arg| arg.ends_with(&input)) {
        args.splice(idx..idx + 1, []);
    }

    // Remove the output(s) from the args list
    for (flag, path) in outputs.iter() {
        let flag = OsStr::new(flag);
        let mut both = flag.to_owned();
        both.push(path);
        for (idx, arg) in args.iter().enumerate() {
            if arg == flag {
                // -o <output>
                // --tile_bc_file_name <output>
                // --module_id_file_name <module_id>
                args.splice(idx..idx + 2, []);
                break;
            } else if arg == &both {
                // -Fi<output>
                // -Fo<output>
                args.splice(idx..idx + 1, []);
                break;
            }
        }
    }

    if outputs.len() == 1 {
        let (_, p) = outputs.drain().next().unwrap();
        outputs.insert("obj", cwd.join(&p));
    } else if let Some(p) = outputs.remove("-o") {
        outputs.insert("obj", cwd.join(&p));
    }

    let mut outputs = outputs
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
        .collect::<HashMap<&'static str, ArtifactDescriptor>>();

    let (common_args, unhashed_args, gen_module_id_file, module_id_file_name) =
        ArgsIter::new(args.into_iter(), arg_info)
            .filter_map(|arg| arg.ok())
            .fold(
                (vec![], vec![], false, None),
                |(
                    mut common_args,
                    mut unhashed_args,
                    mut gen_module_id_file,
                    mut module_id_file_name,
                ),
                 arg| {
                    let args = match arg.get_data() {
                        Some(Unhashed(_)) => &mut unhashed_args,
                        Some(ExtraOutput(path)) => {
                            if let Some(flag) = arg.flag_str() {
                                outputs.insert(
                                    flag,
                                    ArtifactDescriptor {
                                        path: path.to_owned(),
                                        optional: false,
                                    },
                                );
                            }
                            &mut common_args
                        }
                        Some(GenModuleIdFileFlag) => {
                            gen_module_id_file = true;
                            &mut common_args
                        }
                        Some(ModuleIdFileName(path)) => {
                            module_id_file_name = Some(cwd.join(path));
                            &mut common_args
                        }
                        _ => &mut common_args,
                    };

                    args.extend(arg.iter_os_strings());

                    (
                        common_args,
                        unhashed_args,
                        gen_module_id_file,
                        module_id_file_name,
                    )
                },
            );

    let mut extra_dist_files = vec![];

    let gen_module_id_file = gen_module_id_file
        && [
            "--gen_c_file_name",
            "--stub_file_name",
            "--tile_bc_file_name",
        ]
        .iter()
        .any(|k| outputs.contains_key(k));

    match (gen_module_id_file, module_id_file_name) {
        (true, Some(path)) => {
            outputs.insert(
                "--module_id_file_name",
                ArtifactDescriptor {
                    path,
                    optional: false,
                },
            );
        }
        (false, Some(path)) if path.exists() => {
            extra_dist_files.push(path.clone());
        }
        _ => {}
    }

    CompilerArguments::Ok(ParsedArguments {
        input: input.into(),
        outputs,
        language,
        common_args,
        unhashed_args,
        extra_dist_files: extra_dist_files.clone(),
        extra_hash_files: extra_dist_files,
        ..Default::default()
    })
}

// pub async fn preprocess(cwd: &Path, parsed_args: &ParsedArguments) -> Result<PreprocessorOutput> {
//     Ok(PreprocessorOutput::File(cwd.join(&parsed_args.input), None))
// }

pub fn generate_compile_commands(
    executable: &Path,
    parsed_args: &ParsedArguments,
    cwd: &Path,
    env_vars: &[(OsString, OsString)],
    output_flag: &str,
) -> Result<(
    SingleCompileCommand,
    Option<dist::CompileCommand>,
    Cacheable,
)> {
    let language = parsed_args.language.as_str();
    let out_pretty = parsed_args.output_pretty();

    let input = parsed_args.input.as_path();
    let output = match parsed_args.outputs.get("obj") {
        Some(obj) => &obj.path,
        None => return Err(anyhow!("Missing {language} file output")),
    };

    let command = SingleCompileCommand {
        arguments: [
            &parsed_args.common_args[..],
            &parsed_args.unhashed_args[..],
            &[output_flag.into(), output.into(), input.into()],
        ]
        .concat(),
        cwd: cwd.to_owned(),
        env_vars: env_vars.to_owned(),
        executable: executable.to_owned(),
        out_pretty: out_pretty.to_string(),
        share_jobserver: false,
    };

    trace!("[{out_pretty}]: {language} command: {command}");

    #[cfg(not(feature = "dist-client"))]
    {
        Ok((command, None, Cacheable::Yes))
    }
    #[cfg(feature = "dist-client")]
    {
        let dist_command = (|| {
            use crate::util::path_to_str;

            let command = dist::CompileCommand {
                arguments: [
                    &dist::osstrings_to_strings(&parsed_args.common_args)?[..],
                    &[
                        output_flag.into(),
                        path_to_str(output).map(Into::into).ok()?,
                        path_to_str(input).map(Into::into).ok()?,
                    ],
                ]
                .concat(),
                cwd: path_to_str(cwd).map(Into::into).ok()?,
                env_vars: dist::osstring_tuples_to_strings(env_vars)?,
                executable: dunce::canonicalize(executable)
                    .and_then(path_to_str)
                    .map(Into::into)
                    .ok()?,
            };

            trace!("[{out_pretty}]: {language} dist_command: {command}");

            Some(command)
        })();

        Ok((command, dist_command, Cacheable::Yes))
    }
}

ArgData! { pub
    ExtraOutput(PathBuf),
    GenModuleIdFileFlag,
    ModuleIdFileName(PathBuf),
    Output(PathBuf),
    PassThroughFlag,
    PassThrough(OsString),
    Unhashed(OsString),
}

use self::ArgData::*;

counted_array!(pub static ARGS: [ArgInfo<ArgData>; _] = [
    flag!("--allow_managed", PassThroughFlag),
    take_arg!("--c++", OsString, Concatenated, PassThrough),
    flag!("--device-hidden-visibility", PassThroughFlag),
    flag!("--display_error_number", PassThroughFlag),
    flag!("--emit-optix-ir", PassThroughFlag),
    flag!("--enable-tile", PassThroughFlag),
    take_arg!("--gen_c_file_name", PathBuf, Separated, ExtraOutput),
    take_arg!("--gen_device_file_name", PathBuf, Separated, ExtraOutput),
    flag!("--gen_module_id_file", GenModuleIdFileFlag),
    take_arg!("--gnu_version", OsString, Concatenated(b'='), PassThrough),
    take_arg!("--include_alt_file_name", OsString, Separated, PassThrough),
    take_arg!("--include_file_name", OsString, Separated, PassThrough),
    take_arg!("--module_id_file_name", PathBuf, Separated, ModuleIdFileName),
    flag!("--no-version-ident", PassThroughFlag),
    take_arg!("--orig_src_file_name", OsString, Separated, PassThrough),
    take_arg!("--orig_src_path_name", OsString, Separated, PassThrough),
    flag!("--static-host-stub", PassThroughFlag),
    take_arg!("--stub_file_name", PathBuf, Separated, ExtraOutput),
    take_arg!("--tile_bc_file_name", PathBuf, Separated, ExtraOutput),
    take_arg!("-arch", OsString, Separated, PassThrough),
    take_arg!("-fmad", OsString, Concatenated(b'='), PassThrough),
    take_arg!("-ftz", OsString, Concatenated(b'='), PassThrough),
    flag!("-lto", PassThroughFlag),
    take_arg!("-m", OsString, Concatenated, PassThrough),
    take_arg!("-o", PathBuf, Separated, Output),
    take_arg!("-olto", PathBuf, Separated, ExtraOutput),
    take_arg!("-prec_div", OsString, Concatenated(b'='), PassThrough),
    take_arg!("-prec_sqrt", OsString, Concatenated(b'='), PassThrough),
    take_arg!("-t", OsString, Concatenated, PassThrough),
]);

#[cfg(test)]
mod test {
    use super::*;
    use crate::compiler::ptxas;

    fn os_strings(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn test_parse_cicc_input_before_simt_only() {
        let args = os_strings(&[
            "--device-c",
            "-arch",
            "compute_80",
            "--module_id_file_name",
            "kernel.module_id",
            "kernel.cpp1.ii",
            "--simt-only",
            "-o",
            "kernel.ptx",
        ]);

        let parsed = match parse_arguments(&args, ".".as_ref(), Language::Ptx, &ARGS[..]) {
            CompilerArguments::Ok(parsed) => parsed,
            other => panic!("Got unexpected parse result: {:?}", other),
        };

        assert_eq!(PathBuf::from("kernel.cpp1.ii"), parsed.input);
        assert_eq!(
            Some(&ArtifactDescriptor {
                path: PathBuf::from(".").join("kernel.ptx"),
                optional: false
            }),
            parsed.outputs.get("obj")
        );
    }

    #[test]
    fn test_parse_ptxas_input_by_ptx_extension() {
        let args = os_strings(&[
            "-arch=sm_80",
            "kernel.ptx",
            "--some-new-flag",
            "-o",
            "kernel.cubin",
        ]);

        let parsed = match parse_arguments(&args, ".".as_ref(), Language::Cubin, &ptxas::ARGS[..]) {
            CompilerArguments::Ok(parsed) => parsed,
            other => panic!("Got unexpected parse result: {:?}", other),
        };

        assert_eq!(PathBuf::from("kernel.ptx"), parsed.input);
        assert_eq!(
            Some(&ArtifactDescriptor {
                path: PathBuf::from(".").join("kernel.cubin"),
                optional: false
            }),
            parsed.outputs.get("obj")
        );
    }
}
