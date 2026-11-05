#![allow(dead_code)]

use fs_err as fs;
use regex::Regex;
use std::{
    env,
    ffi::{OsStr, OsString},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::LazyLock,
};

use serde::Serialize;
use which::which_in;

use sccache::{
    compiler::{CCompilerKind, CompilerKind, Language},
    errors::*,
    mock_command::ProcessOutput,
    server::ServerStats,
};

pub mod client;

#[cfg(feature = "dist-tests")]
pub mod dist;

pub const TC_CACHE_SIZE: u64 = 2 * 1024 * 1024 * 1024; // 2GiB

pub fn check_output<O: Into<ProcessOutput>>(output: O) -> Result<()> {
    let output = output.into();
    if !output.success() {
        eprintln!("{}", output.desc());
        if !output.stdout.is_empty() {
            eprintln!(
                "\n\n[BEGIN STDOUT]\n===========\n{}\n===========\n[FIN STDOUT]",
                String::from_utf8_lossy(&output.stdout)
            );
        }
        if !output.stderr.is_empty() {
            eprintln!(
                "\n\n[BEGIN STDERR]\n===========\n{}\n===========\n[FIN STDERR]",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Err(anyhow!("{}", output.desc()))
    } else {
        Ok(())
    }
}

pub fn init_cargo(path: &Path, cargo_name: &str) -> PathBuf {
    let cargo_path = path.join(cargo_name);
    let source_path = "src";
    fs::create_dir_all(cargo_path.join(source_path)).unwrap();
    cargo_path
}

// Prune any environment variables that could adversely affect test execution.
pub fn prune_command(mut cmd: Command) -> Command {
    use sccache::util::OsStrExt;

    for (var, _) in env::vars_os() {
        if var.starts_with("SCCACHE_") {
            cmd.env_remove(var);
        }
    }
    cmd
}

macro_rules! vec_from {
    ( $t:ty, $( $x:expr ),* ) => {
        vec![$( Into::<$t>::into(&$x), )*]
    };
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

pub fn write_json_cfg<T: Serialize>(path: &Path, filename: &str, contents: &T) {
    let p = path.join(filename);
    let mut f = fs::File::create(p).unwrap();
    f.write_all(&serde_json::to_vec(contents).unwrap()).unwrap();
}

pub fn write_source(path: &Path, filename: &str, contents: &str) {
    let p = path.join(filename);
    let mut f = fs::File::create(p).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
}

#[derive(Clone)]
pub struct Compiler {
    pub name: &'static str,
    pub exe: OsString,
    pub env_vars: Vec<(OsString, OsString)>,
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

#[derive(Clone, Debug, Default)]
pub struct AdditionalStats {
    pub cache_writes: Option<u64>,
    pub hashed: Option<u64>,
    pub preprocessed: Option<u64>,
    pub compilations: Option<u64>,
    pub compile_requests: Option<u64>,
    pub non_cacheable_compilations: Option<u64>,
    pub requests_executed: Option<u64>,
    pub cache_hits: Option<Vec<(CCompilerKind, Language, u64)>>,
    pub cache_misses: Option<Vec<(CCompilerKind, Language, u64)>>,
}

impl std::ops::Add<AdditionalStats> for ServerStats {
    type Output = ServerStats;
    fn add(mut self, rhs: AdditionalStats) -> Self::Output {
        self += rhs;
        self
    }
}

impl std::ops::AddAssign<AdditionalStats> for ServerStats {
    fn add_assign(&mut self, rhs: AdditionalStats) {
        self.cache_writes += rhs.cache_writes.unwrap_or(0);
        self.hashed += rhs.hashed.unwrap_or(0);
        self.preprocessed += rhs.preprocessed.unwrap_or(0);
        self.compilations += rhs.compilations.unwrap_or(0);
        self.compile_requests += rhs.compile_requests.unwrap_or(0);
        self.requests_executed += rhs.requests_executed.unwrap_or(0);
        self.non_cacheable_compilations += rhs.non_cacheable_compilations.unwrap_or(0);

        for (kind, lang, count) in rhs.cache_hits.unwrap_or_default() {
            let kind = CompilerKind::C(kind);
            for _ in 0..count {
                self.cache_hits.increment(&kind, &lang);
            }
        }

        for (kind, lang, count) in rhs.cache_misses.unwrap_or_default() {
            let kind = CompilerKind::C(kind);
            for _ in 0..count {
                self.cache_misses.increment(&kind, &lang);
            }
        }

        self.cache_write_duration = Default::default();
        self.cache_read_hit_duration = Default::default();
        self.hasher_duration = Default::default();
        self.preprocessor_duration = Default::default();
        self.compiler_write_duration = Default::default();
    }
}
