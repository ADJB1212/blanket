# Rebuilding the binary release

The corresponding-source archive is separate from the PyPI source distribution.
It includes `project/`, `vendor/`, `native/`, `cargo-metadata.json`, and
`project.commit`. The native archives include complete upstream source trees,
submodules, and `SOURCE.txt` files recording their repositories and revisions.
Cargo packages retain their original license files and checksum manifests.
No codec source changes are applied by the release builder.

Use the Rust toolchain pinned in `project/rust-toolchain.toml`, Python 3.12+,
and a C/C++ toolchain with CMake, Ninja, Meson, pkg-config, and NASM on x86.
The release workflow and `scripts/setup_manylinux.sh` record the toolchain
setup, platform settings, and exact native dependency versions. Toolchain and
operating-system components are installed separately.

1. Extract every archive in `native/` into an empty directory and set
   `BLANKET_SOURCE_ROOT` to that directory's absolute path.
2. From `project/`, set `RUNNER_OS` to `Linux` or `macOS`, `RUNNER_ARCH` to
   `X64` or `ARM64`, and `RUNNER_TEMP` to an empty writable build directory.
3. Run `bash scripts/build_release_dependency.sh NAME VERSION` for each native
   dependency, using the versions recorded in the workflow and `SOURCE.txt`.
   Build aom, x265, libde265, and dav1d before libheif, and LittleCMS before
   JPEG XL. The source-root setting
   uses the extracted sources instead of cloning upstream repositories.
4. Set `PKG_CONFIG_PATH`, `CMAKE_PREFIX_PATH`, and `LD_LIBRARY_PATH` (Linux) or
   `DYLD_LIBRARY_PATH` (macOS) as in `.github/workflows/release.yml`, pointing at
   `$HOME/deps/`. Set `LCMS2_LIB_DIR` and `LCMS2_INCLUDE_DIR` as in that workflow.
5. Build using `maturin build --release --locked --features default`. The supplied
   `project/.cargo/config.toml` selects the adjacent vendored Cargo sources;
   no registry dependency source download is needed. The workflow records the
   platform-specific compiler flags and wheel repair commands.

You may modify the supplied native sources and rebuild/relink the extension.
Shared codec libraries are bundled in the wheel rather than supplied by the
operating system. Preserve the original notices and mark your modifications
when redistributing. Blanket's original source remains MIT licensed; the
combined binary distribution uses GPL-3.0-or-later. Full GPL/LGPL terms are in
`project/licenses/libheif-COPYING`, with component notices in
`project/licenses/third-party/`.
