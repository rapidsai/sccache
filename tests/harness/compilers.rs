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

use regex::Regex;
use sccache::{
    compiler::{CCompilerKind, Language},
    errors::*,
};
use std::{
    env,
    ffi::{OsStr, OsString},
    path::Path,
    sync::LazyLock,
};
use which::which_in;

#[derive(Clone)]
pub struct Compiler {
    pub name: &'static str,
    pub exe: OsString,
    pub env_vars: Vec<(OsString, OsString)>,
}

impl std::fmt::Debug for Compiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiler")
            .field("name", &self.name)
            .field("path", &self.exe)
            .finish()
    }
}

static COMPILER_VERSION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^.*?(?P<major>\d+).*?(?P<minor>\d+).*?(?P<patch>\d+).*?$").unwrap()
});

impl Compiler {
    pub fn version(&self) -> Result<String> {
        use sccache::compiler_version;

        let exe = Path::new(&self.exe);
        let ver = compiler_version(exe)?;

        let (major, minor, patch) = match COMPILER_VERSION_RE.captures(&ver) {
            Some(ver) => (
                ver.name("major")
                    .and_then(|s| s.as_str().parse::<usize>().ok())
                    .unwrap_or(0),
                ver.name("minor")
                    .and_then(|s| s.as_str().parse::<usize>().ok())
                    .unwrap_or(0),
                ver.name("patch")
                    .and_then(|s| s.as_str().parse::<usize>().ok())
                    .unwrap_or(0),
            ),
            None => bail!("Could not determine compiler version (exe={exe:?})"),
        };

        Ok(format!("{major}.{minor}.{patch}"))
    }

    pub fn kind(&self) -> CCompilerKind {
        self.name.into()
    }

    /// The language sccache reports for C sources compiled by this compiler.
    pub fn lang(&self) -> Language {
        if self.name.contains("++") {
            Language::Cxx
        } else {
            Language::C
        }
    }
}

// Test GCC + clang on non-OS X platforms.
#[cfg(all(unix, not(target_os = "macos")))]
pub const COMPILERS: &[&str] = &["gcc", "clang", "clang++", "nvc", "nvc++", "mpic++"];

// OS X ships a `gcc` that's just a clang wrapper, so only test clang there.
#[cfg(target_os = "macos")]
pub const COMPILERS: &[&str] = &["clang", "clang++"];

#[cfg(target_os = "linux")]
pub const CUDA_COMPILERS: &[&str] = &["nvcc", "clang++"];

#[cfg(target_os = "windows")]
pub const CUDA_COMPILERS: &[&str] = &["nvcc"];

#[cfg(unix)]
pub fn find_compilers() -> Vec<Compiler> {
    let cwd = env::current_dir().unwrap();
    COMPILERS
        .iter()
        .filter_map(|c| {
            which_in(c, env::var_os("PATH"), &cwd)
                .ok()
                .map(|full_path| Compiler {
                    name: c,
                    exe: full_path.as_path().into(),
                    env_vars: vec![],
                })
        })
        .collect::<Vec<_>>()
}

#[cfg(target_env = "msvc")]
pub fn find_compilers() -> Vec<Compiler> {
    let tool = cc::Build::new()
        .opt_level(1)
        .host("x86_64-pc-windows-msvc")
        .target("x86_64-pc-windows-msvc")
        .debug(false)
        .get_compiler();
    vec![Compiler {
        name: "cl",
        exe: tool.path().into(),
        env_vars: tool.env().to_vec(),
    }]
}

#[cfg(any(unix, target_env = "msvc"))]
pub fn find_compiler(name: &str) -> Option<Compiler> {
    find_compilers().into_iter().find(|c| c.name == name)
}

#[cfg(not(any(target_os = "macos", target_os = "freebsd")))]
pub fn find_cuda_compilers() -> Vec<Compiler> {
    use which::which;

    let cwd = env::current_dir().unwrap();

    let candidates = match env::var_os("NOTEST_CUDA_COMPILERS") {
        Some(nc) => {
            let ncs = nc.into_string().unwrap();
            let not_candidates = ncs.split(':').collect::<Vec<_>>();
            CUDA_COMPILERS
                .iter()
                .filter(|&c| !not_candidates.contains(c))
                .collect::<Vec<_>>()
        }
        None => CUDA_COMPILERS.iter().collect::<Vec<_>>(),
    };

    // CUDA compilers like clang don't come with all of the components for compilation.
    // To consider a machine to have any cuda compilers we rely on the existence of `nvcc`

    match which("nvcc") {
        Ok(_) => candidates
            .iter()
            .filter_map(|c| {
                which_in(c, env::var_os("PATH"), &cwd)
                    .ok()
                    .map(|full_path| Compiler {
                        name: c,
                        exe: full_path.as_path().into(),
                        env_vars: vec![],
                    })
            })
            .collect::<Vec<_>>(),
        Err(_) => {
            eprintln!(
                "unable to find `nvcc` in PATH={:?}",
                env::var_os("PATH").unwrap_or_default()
            );
            vec![]
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "freebsd")))]
pub fn find_cuda_compiler(name: &str) -> Option<Compiler> {
    find_cuda_compilers().into_iter().find(|c| c.name == name)
}

// We detect the HIP Clang compiler through 2 methods:
// 1. If the env var HIP_CLANG_PATH is set, try $HIP_CLANG_PATH/clang. This is the same behavior as
//    hipcc, but is rarely know, so we have another option.
// 2. If the env var ROCM_PATH is set, try $ROCM_PATH/llvm/bin/clang. This is the location in
//    AMD's official debian packages.
// 3. Otherwise, just bail.
pub fn find_hip_compiler() -> Option<Compiler> {
    let env_vars: Vec<(OsString, OsString)> = env::vars_os().collect();

    if let Ok(hip_clang_path) = env::var("HIP_CLANG_PATH") {
        let clang_path = Path::new(&hip_clang_path).join("clang");

        if let Ok(true) = clang_path.try_exists() {
            return Some(Compiler {
                name: "clang",
                exe: clang_path.as_path().into(),
                env_vars,
            });
        }
    }
    if let Ok(rocm_path) = env::var("ROCM_PATH") {
        let clang_path = Path::new(&rocm_path).join("llvm").join("bin").join("clang");

        if let Ok(true) = clang_path.try_exists() {
            return Some(Compiler {
                name: "hip",
                exe: clang_path.as_path().into(),
                env_vars,
            });
        }
    }
    None
}

pub fn adv_key_kind(lang: &str, compiler: &str) -> String {
    let language = lang.to_owned();
    match compiler {
        "clang" | "clang++" | "clang-cl" => language + " [clang]",
        "gcc" | "g++" => language + " [gcc]",
        "cl" => language + " [msvc]",
        "nvc" | "nvc++" => language + " [nvhpc]",
        "nvcc" => match lang {
            "cudafe++" => "cuda [cudafe++]".to_owned(),
            "ptx" => language + " [cicc]",
            "cubin" => language + " [ptxas]",
            _ => language + " [nvcc]",
        },
        _ => {
            trace!("Unknown compiler type: {compiler}");
            language + "unknown"
        }
    }
}

// TODO: This will fail if gcc/clang is actually a ccache wrapper, as it is the
// default case on Fedora, e.g.
pub fn compile_cmdline<T: AsRef<OsStr>>(
    compiler: &str,
    exe: T,
    input: &str,
    output: &str,
    extra_args: impl IntoIterator<Item = OsString>,
) -> Vec<OsString> {
    let mut arg = match compiler {
        "gcc" | "clang" | "clang++" | "nvc" | "nvc++" => {
            vec_from!(OsString, exe.as_ref(), "-c", input, "-o", output)
        }
        "nvcc" => {
            vec_from!(
                OsString,
                exe.as_ref(),
                "-allow-unsupported-compiler",
                "-c",
                input,
                "-o",
                output
            )
        }
        "cl" => vec_from!(OsString, exe, "-c", input, format!("-Fo{}", output)),
        _ => panic!("Unsupported compiler: {compiler}"),
    };
    arg.extend(extra_args);
    arg
}

// TODO: This will fail if gcc/clang is actually a ccache wrapper, as it is the
// default case on Fedora, e.g.
#[cfg(not(any(target_os = "macos", target_os = "freebsd")))]
pub fn compile_cuda_cmdline<T: AsRef<OsStr>>(
    compiler: &str,
    exe: T,
    compile_flag: &str,
    input: &str,
    output: &str,
    extra_args: &[OsString],
) -> Vec<OsString> {
    let mut args = match compiler {
        "nvcc" => vec_from!(
            OsString,
            exe.as_ref(),
            "-allow-unsupported-compiler",
            "-x",
            "cu",
            compile_flag,
            input,
            "-o",
            output
        ),
        "clang++" => {
            vec_from!(
                OsString,
                exe,
                "-x",
                "cuda",
                compile_flag,
                input,
                "--cuda-gpu-arch=sm_75",
                format!(
                    "--cuda-path={}",
                    env::var_os("CUDA_PATH")
                        .or(env::var_os("CUDA_HOME"))
                        .unwrap_or("/usr/local/cuda".into())
                        .to_string_lossy()
                ),
                "--no-cuda-version-check",
                // work around for clang-cuda on windows-2019 (https://github.com/microsoft/STL/issues/2359)
                "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                "-o",
                output
            )
        }
        _ => panic!("Unsupported compiler: {compiler}"),
    };
    if !extra_args.is_empty() {
        args.append(&mut extra_args.to_vec());
    }
    args.iter()
        .filter(|x| !x.is_empty())
        .cloned()
        .collect::<Vec<_>>()
}

// TODO: This will fail if gcc/clang is actually a ccache wrapper, as it is the
// default case on Fedora, e.g.
//
// archs is a list of GPU architectures to compile for.
pub fn compile_hip_cmdline<T: AsRef<OsStr>>(
    compiler: &str,
    exe: T,
    input: &str,
    output: &str,
    archs: &[String],
    extra_args: impl IntoIterator<Item = OsString>,
) -> Vec<OsString> {
    let mut arg = match compiler {
        "clang" => {
            vec_from!(OsString, exe, "-x", "hip", "-c", input, "-o", output)
        }
        _ => panic!("Unsupported compiler: \"{compiler}\""),
    };
    for arch in archs {
        arg.push(format!("--offload-arch={arch}").into());
    }
    arg.extend(extra_args);
    arg
}
