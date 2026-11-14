# `jpegxl-src`

Builds libjxl from a shallow clone of tag `v0.12.0` in `OUT_DIR`, including
shallow checkouts of the required Brotli, Highway, and skcms submodules.
Git, network access, CMake, and a C/C++ toolchain are required for the first build.
Subsequent builds reuse the checkout. Set `DEP_JXL_PATH` to use an existing source
directory with its dependencies already populated.

License: `BSD-3-Clause`
