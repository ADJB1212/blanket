use std::env;
use std::error::Error;
use std::path::PathBuf;

mod build {
    use std::process::{Command, Stdio};

    use super::*;

    const REPO: &str = "https://code.videolan.org/videolan/dav1d.git";
    const TAG: &str = "1.5.5";

    macro_rules! runner {
            ($cmd:expr, $($arg:expr),* $(,)?) => {{
                let status = Command::new($cmd)
                    $(.arg($arg))*
                    .stderr(Stdio::inherit())
                    .stdout(Stdio::inherit())
                    .status()?;

                if !status.success() {
                    return Err(format!(
                        "{} exited with status {}",
                        $cmd, status
                    ).into());
                }
            }};
        }

    pub fn build_from_src(lib: &str, version: &str) -> Result<pkg_config::Library, Box<dyn Error>> {
        let source = PathBuf::from(env::var("OUT_DIR")?).join("dav1d");
        let build_path = source.join("build");
        let release_path = source.join("release");

        if !source.join(".git").exists() {
            runner!("git", "clone", "--depth", "1", "--branch", TAG, REPO, &source);
        } else {
            runner!("git", "-C", &source, "fetch", "--depth", "1", "origin", TAG);
            runner!("git", "-C", &source, "checkout", "FETCH_HEAD");
        }

        runner!(
            "meson",
            "setup",
            "--buildtype=release",
            "-Ddefault_library=static",
            "-Denable_tools=false",
            "-Denable_tests=false",
            "-Denable_docs=false",
            "--prefix",
            &release_path,
            &build_path,
            &source
        );

        runner!("ninja", "-C", &build_path);
        runner!("meson", "install", "-C", &build_path);

        let pkg_dir = release_path.join("lib").join("pkgconfig");

        let mut paths = vec![pkg_dir];
        if let Some(existing) = env::var_os("PKG_CONFIG_PATH") {
            paths.extend(env::split_paths(&existing));
        }

        // SAFETY: The build script does not modify the
        // environment concurrently with other threads.
        unsafe {
            env::set_var("PKG_CONFIG_PATH", env::join_paths(paths)?);
        }

        Ok(pkg_config::Config::new().atleast_version(version).statik(true).probe(lib)?)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if env::var_os("DOCS_RS").is_some() {
        return Ok(());
    }

    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_LIBDIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR");

    const LIB: &str = "dav1d";
    const MIN_VERSION: &str = "1.5.0";

    if pkg_config::Config::new().atleast_version(MIN_VERSION).probe(LIB).is_err() {
        println!("cargo:warning=dav1d >= {MIN_VERSION} not found; building from source");

        build::build_from_src(LIB, MIN_VERSION)?;
    }

    Ok(())
}
