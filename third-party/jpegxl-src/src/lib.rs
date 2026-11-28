/*
 * This file is part of jpegxl-rs.
 *
 * jpegxl-rs is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * jpegxl-rs is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with jpegxl-rs.  If not, see <https://www.gnu.org/licenses/>.
 */

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use core::panic;
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn source_dir() -> PathBuf {
    println!("cargo:rerun-if-env-changed=DEP_JXL_PATH");
    if let Some(path) = env::var_os("DEP_JXL_PATH") {
        return PathBuf::from(path);
    }

    let source = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set")).join("libjxl");
    if !source.exists() {
        let status = Command::new("git")
            .args(["clone", "--depth", "1", "--branch", "v0.13.0", "https://github.com/libjxl/libjxl.git"])
            .arg(&source)
            .status()
            .expect("failed to run git to clone libjxl");
        assert!(status.success(), "failed to clone libjxl v0.13.0");
    }

    let status = Command::new("git")
        .arg("-C")
        .arg(&source)
        .args([
            "submodule",
            "update",
            "--init",
            "--force",
            "--depth",
            "1",
            "--",
            "third_party/brotli",
            "third_party/highway",
            "third_party/lcms",
            "third_party/skcms",
        ])
        .status()
        .expect("failed to run git to initialize libjxl dependencies");
    assert!(status.success(), "failed to initialize libjxl dependencies");
    source
}

#[cfg_attr(coverage_nightly, coverage(off))]
/// Builds the JPEG XL library.
///
/// # Panics
///
/// Panics if fetching the source or dependencies fails, the source directory is
/// missing, or the native build fails.
pub fn build() {
    let source = source_dir();
    assert!(source.exists() && source.is_dir(), "Source directory {} does not exist", source.display());

    let mut config = cmake::Config::new(source);
    config
        .define("BUILD_TESTING", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("JPEGXL_ENABLE_TOOLS", "OFF")
        .define("JPEGXL_ENABLE_DOXYGEN", "OFF")
        .define("JPEGXL_ENABLE_MANPAGES", "OFF")
        .define("JPEGXL_ENABLE_BENCHMARK", "OFF")
        .define("JPEGXL_ENABLE_EXAMPLES", "OFF")
        .define("JPEGXL_ENABLE_JNI", "OFF")
        .define("JPEGXL_ENABLE_SJPEG", "OFF")
        .define("JPEGXL_ENABLE_OPENEXR", "OFF")
        .define("DJPEGXL_ENABLE_VIEWERS", "OFF")
        .define("DJPEGXL_ENABLE_FUZZERS", "OFF")
        .define("DJPEGXL_ENABLE_COVERAGE", "OFF")
        .define("DJPEGXL_TEST_TOOLS", "OFF")
        .define("DJPEGXL_ENABLE_DEVTOOLS", "OFF")
        .define("DJPEGXL_ENABLE_WASM_THREADS", "OFF")
        .define("DJPEGXL_ENABLE_SKCMS", "OFF")
        .define("JPEGXL_BUNDLE_LIBPNG", "OFF");

    if let Ok(p) = std::thread::available_parallelism() {
        config.env("CMAKE_BUILD_PARALLEL_LEVEL", format!("{p}"));
    }

    if cfg!(asan) {
        config
            .env("SANITIZER", "asan")
            .cflag("-g -DADDRESS_SANITIZER -fsanitize=address")
            .cxxflag("-g -DADDRESS_SANITIZER -fsanitize=address")
            .define("JPEGXL_ENABLE_TCMALLOC", "OFF");
    } else if cfg!(tsan) {
        config
            .env("SANITIZER", "tsan")
            .cflag("-g -DTHREAD_SANITIZER -fsanitize=thread")
            .cxxflag("-g -DTHREAD_SANITIZER -fsanitize=thread")
            .define("JPEGXL_ENABLE_TCMALLOC", "OFF");
    }

    #[cfg(windows)]
    const {
        assert!(!cfg!(tsan), "Thread sanitizer is not supported on Windows");
    }
    if cfg!(windows) {
        // For CMake pre-checking
        let mut exeflags = "MSVCRTD.lib".to_string();
        if cfg!(asan) {
            exeflags.push_str(" clang_rt.asan_dynamic-x86_64.lib clang_rt.asan_dynamic_runtime_thunk-x86_64.lib");
        }

        config
            .generator_toolset("ClangCL")
            .define("CMAKE_VS_GLOBALS", "UseMultiToolTask=true;EnforceProcessCountAcrossBuilds=true") // Enable parallel builds
            .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded")
            .define("CMAKE_EXE_LINKER_FLAGS", exeflags)
            .cflag("/Zl");
    }

    let prefix = config.build();

    let lib_dir = {
        let mut lib_dir = prefix.join("lib");
        if lib_dir.exists() {
            lib_dir
        } else {
            lib_dir.pop();
            lib_dir.push("lib64");
            if lib_dir.exists() {
                lib_dir
            } else {
                panic!("Could not find the library directory, please check the files in {prefix:?}");
            }
        }
    };
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    println!("cargo:rustc-link-lib=static=jxl");
    println!("cargo:rustc-link-lib=static=jxl_cms");
    println!("cargo:rustc-link-lib=static=jxl_threads");

    println!("cargo:rustc-link-lib=static=hwy");
    println!("cargo:rustc-link-lib=static=brotlidec");
    println!("cargo:rustc-link-lib=static=brotlienc");
    println!("cargo:rustc-link-lib=static=brotlicommon");

    if cfg!(any(target_vendor = "apple", target_os = "freebsd")) {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
