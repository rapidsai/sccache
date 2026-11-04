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
        Cacheable, CompileCommandImpl, CompilerArguments, Language,
        args::*,
        c::{
            AnyCompileCommand, CCompilerImpl, CCompilerKind, GenerateCompileCommandsArgs,
            GenerateDependenciesArgs, ParseArgs, ParsedArguments,
        },
        cicc,
    },
    counted_array, dist,
    errors::*,
    mock_command::CommandCreatorSync,
};

use async_trait::async_trait;

use std::{ffi::OsString, path::PathBuf, time::Duration};

/// A unit struct on which to implement `CCompilerImpl`.
#[derive(Clone, Debug)]
pub struct Llc {
    pub version: Option<String>,
}

#[async_trait]
impl CCompilerImpl for Llc {
    fn kind(&self) -> CCompilerKind {
        CCompilerKind::Llc
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
        cicc::parse_arguments(arguments, cwd, Language::Assembler, &ARGS[..])
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
        cicc::generate_compile_commands(executable, parsed_args, cwd, env_vars, "-o")
    }
}

use cicc::ArgData::*;

counted_array!(pub static ARGS: [ArgInfo<cicc::ArgData>; _] = [
    take_arg!("-o", PathBuf, Separated, Output),
]);
