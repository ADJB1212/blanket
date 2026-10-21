#!/usr/bin/env bash
set -euo pipefail

dnf install -y numactl-devel brotli brotli-devel pkgconf-pkg-config make perl curl tar xz git mold
manylinux-install-clang -v 23.1.3.0
export PATH="/opt/clang/bin:/opt/python/cp314-cp314/bin:$PATH"
clang --version
[[ "$(clang -dumpversion)" == 23.1.3 ]]

python -m pip install 'cmake>=3.28' ninja meson
export CC=/opt/clang/bin/clang
export CXX=/opt/clang/bin/clang++
export LDFLAGS=-fuse-ld=mold
nasm_version=3.02
nasm_source="$RUNNER_TEMP/nasm-$nasm_version"
curl -fsSL --retry 3 "https://www.nasm.us/pub/nasm/releasebuilds/$nasm_version/nasm-$nasm_version.tar.xz" -o "$RUNNER_TEMP/nasm.tar.xz"
tar -xJf "$RUNNER_TEMP/nasm.tar.xz" -C "$RUNNER_TEMP"
(
	cd "$nasm_source"
	./configure --prefix=/usr/local
	make -j2
	make install
)
for tool in clang clang++ ld.lld llvm-ar llvm-ranlib cmake meson nasm ninja pkg-config mold; do
	command -v "$tool"
done
nasm --version
echo "/opt/clang/bin" >>"$GITHUB_PATH"
echo "/opt/python/cp314-cp314/bin" >>"$GITHUB_PATH"
echo "$HOME/.cargo/bin" >>"$GITHUB_PATH"
echo "$HOME/.local/bin" >>"$GITHUB_PATH"
echo "CC=/opt/clang/bin/clang" >>"$GITHUB_ENV"
echo "CXX=/opt/clang/bin/clang++" >>"$GITHUB_ENV"
echo "AR=/opt/clang/bin/llvm-ar" >>"$GITHUB_ENV"
echo "RANLIB=/opt/clang/bin/llvm-ranlib" >>"$GITHUB_ENV"
echo "LDFLAGS=-fuse-ld=mold" >>"$GITHUB_ENV"
echo "UV_PYTHON=/opt/python/cp314-cp314/bin/python" >>"$GITHUB_ENV"
