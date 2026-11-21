use std::env;
use std::env::consts;
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    if env::var_os("DOCS_RS").is_some() {
        return Ok(());
    }

    for name in ["DOCS_RS", "LCMS2_INCLUDE_DIR", "LCMS2_LIB_DIR", "LCMS2_STATIC"] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    if let Ok(include_dir) = env::var("LCMS2_INCLUDE_DIR") {
        println!("cargo:include={include_dir}");
    }

    let requires_static_only = cfg!(feature = "static") || env::var("LCMS2_STATIC").is_ok();
    if let Some(lib_dir) = env::var_os("LCMS2_LIB_DIR") {
        let lib_dir = Path::new(&lib_dir);
        if requires_static_only && ["liblcms2.a", "lcms2.lib"].iter().any(|file| lib_dir.join(file).is_file()) {
            link_static(lib_dir)?;
            return Ok(());
        }
        let dylib_name = format!("{}lcms2{}", consts::DLL_PREFIX, consts::DLL_SUFFIX);
        if !requires_static_only && [&dylib_name, "liblcms2.a", "lcms2.so", "lcms2.lib"].iter().any(|file| lib_dir.join(file).exists()) {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=lcms2");
            return Ok(());
        }
        println!("cargo:warning=LCMS2_LIB_DIR path ({}) did not contain {dylib_name}", lib_dir.display());
    }

    if requires_static_only || (!configure_pkg_config() && cfg!(feature = "static-fallback")) {
        compile_static()?;
    }
    Ok(())
}

#[cfg(feature = "dynamic")]
fn configure_pkg_config() -> bool {
    match pkg_config::probe_library("lcms2") {
        Ok(info) => {
            let joined_paths = env::join_paths(info.include_paths).ok().and_then(|p| p.into_string().ok());
            if let Some(joined_paths) = joined_paths {
                println!("cargo:include={joined_paths}");
                true
            } else {
                println!("cargo:warning=got invalid include paths from pkg-config of lcms2");
                false
            }
        }
        Err(err) => {
            println!("cargo:warning=pkg_config failed ({err}). Falling back to static build.");
            false
        }
    }
}

#[cfg(not(feature = "dynamic"))]
fn configure_pkg_config() -> bool {
    false
}

#[cfg(not(any(feature = "static", feature = "static-fallback")))]
fn compile_static() -> Result<(), Box<dyn Error>> {
    println!("cargo:warning='static' feature of lcms2-sys is disabled, so the library won't be built, and probably won't work at all");
    println!("cargo:rustc-link-lib=lcms2");
    Ok(())
}

#[cfg(any(feature = "static", feature = "static-fallback"))]
fn compile_static() -> Result<(), Box<dyn Error>> {
    use std::path::PathBuf;
    use std::process::Command;

    fn run(command: &mut Command) -> Result<(), Box<dyn Error>> {
        let status = command.status().map_err(|err| format!("failed to run {}: {err}", command.get_program().to_string_lossy()))?;
        if !status.success() {
            return Err(format!("{} exited with status {status}", command.get_program().to_string_lossy()).into());
        }
        Ok(())
    }

    let source = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR must be set")?).join("lcms2");
    if !source.join("include/lcms2.h").is_file() {
        if source.exists() {
            std::fs::remove_dir_all(&source)?;
        }
        run(Command::new("git").args(["clone", "--depth", "1", "--branch", "lcms2.20", "https://github.com/mm2/Little-CMS.git"]).arg(&source))?;
    }

    let build_path = source.join("build");
    let release_path = source.join("release");
    let mut setup = Command::new("meson");
    setup.arg("setup");
    if build_path.join("meson-private/coredata.dat").is_file() {
        setup.arg("--reconfigure");
    }
    setup.args([
        "--buildtype=release",
        "--default-library=static",
        "-Db_staticpic=true",
        "--libdir=lib",
        "-Dtests=disabled",
        "-Dutils=false",
        "-Djpeg=disabled",
        "-Dtiff=disabled",
        "-Dthreaded=true",
        "-Dfastfloat=true",
        "-Dwarning_level=0",
    ]);
    if cfg!(feature = "lcms2-strict-cgats") {
        setup.arg("-Dc_args=-DCMS_STRICT_CGATS=1");
    }
    run(setup.arg("--prefix").arg(&release_path).arg(&build_path).arg(&source))?;
    run(Command::new("ninja").arg("-C").arg(&build_path).arg("install"))?;

    link_static(&release_path.join("lib"))
}

fn link_static(lib_dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut paths = vec![lib_dir.join("pkgconfig")];
    if let Some(existing) = env::var_os("PKG_CONFIG_PATH") {
        paths.extend(env::split_paths(&existing));
    }
    // SAFETY: The build script does not modify the environment concurrently.
    unsafe {
        env::set_var("PKG_CONFIG_PATH", env::join_paths(paths)?);
    }
    let library = pkg_config::Config::new().statik(true).probe("lcms2")?;
    println!("cargo:include={}", env::join_paths(library.include_paths)?.to_string_lossy());
    Ok(())
}
