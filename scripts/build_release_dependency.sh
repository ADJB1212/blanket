#!/usr/bin/env bash
set -euo pipefail

name=$1
version=$2
prefix="$HOME/deps/$name"
source_dir="$RUNNER_TEMP/$name"
shared_ext=so
if [[ "$RUNNER_OS" == macOS ]]; then
	shared_ext=dylib
fi
case "$RUNNER_ARCH" in
X64)
	export CFLAGS="${CFLAGS:-} -O3 -march=x86-64-v2 -mtune=generic"
	export CXXFLAGS="${CXXFLAGS:-} -O3 -march=x86-64-v2 -mtune=generic"
	;;
ARM64)
	if [[ "$RUNNER_OS" == macOS ]]; then
		export CFLAGS="${CFLAGS:-} -O3 -arch arm64 -mcpu=apple-m1"
		export CXXFLAGS="${CXXFLAGS:-} -O3 -arch arm64 -mcpu=apple-m1"
	else
		export CFLAGS="${CFLAGS:-} -O3 -march=armv8-a"
		export CXXFLAGS="${CXXFLAGS:-} -O3 -march=armv8-a"
	fi
	;;
esac

clone_flags=()
case "$name" in
aom)
	repository=https://aomedia.googlesource.com/aom
	clone_flags=(--depth 1)
	;;
x265)
	repository=https://bitbucket.org/multicoreware/x265_git.git
	clone_flags=(--depth 1)
	;;
libde265)
	repository=https://github.com/strukturag/libde265.git
	clone_flags=(--depth 1)
	;;
dav1d)
	repository=https://code.videolan.org/videolan/dav1d.git
	clone_flags=(--depth 1)
	;;
libheif)
	repository=https://github.com/strukturag/libheif.git
	clone_flags=(--depth 1)
	;;
libwebp)
	repository=https://chromium.googlesource.com/webm/libwebp
	clone_flags=(--depth 1)
	;;
libjxl)
	repository=https://github.com/libjxl/libjxl.git
	clone_flags=(--recursive --shallow-submodules)
	;;
*)
	echo "Unknown dependency: $name" >&2
	exit 1
	;;
esac

git clone --branch "$version" "${clone_flags[@]}" "$repository" "$source_dir"

case "$name" in
x265)
	x265_neon_flag=()
	if [[ "$RUNNER_OS" == macOS ]]; then
		x265_neon_flag=(-DHAVE_NEON=1)
	fi
	for depth in 12 10; do
		main12=OFF
		if [[ "$depth" == 12 ]]; then main12=ON; fi
		cmake -S "$source_dir/source" -B "$source_dir/build-$depth" -G Ninja \
			-DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
			-DHIGH_BIT_DEPTH=ON -DENABLE_HDR10_PLUS=ON -DMAIN12="$main12" \
			-DEXPORT_C_API=OFF -DENABLE_SHARED=OFF -DBUILD_SHARED_LIBS=OFF -DENABLE_CLI=OFF \
			"${x265_neon_flag[@]}"
		cmake --build "$source_dir/build-$depth" --parallel
	done
	cmake -S "$source_dir/source" -B "$source_dir/build" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
		-DCMAKE_INSTALL_LIBDIR=lib -DENABLE_SHARED=ON -DENABLE_CLI=OFF \
		-DHIGH_BIT_DEPTH=OFF -DLINKED_10BIT=ON -DLINKED_12BIT=ON \
		-DEXTRA_LIB="$source_dir/build-10/libx265.a;$source_dir/build-12/libx265.a" \
		"${x265_neon_flag[@]}"
	;;
dav1d)
	meson setup "$source_dir/build" "$source_dir" \
		-Denable_tools=false -Denable_tests=false -Denable_docs=false \
		--prefix="$prefix" --libdir=lib --buildtype=release --default-library=shared
	ninja -C "$source_dir/build" install
	exit 0
	;;
aom)
	cmake -S "$source_dir" -B "$source_dir/build" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
		-DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON \
		-DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_DOCS=OFF
	;;
libde265)
	cmake -S "$source_dir" -B "$source_dir/build" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
		-DCMAKE_INSTALL_LIBDIR=lib -DENABLE_DECODER=OFF \
		-DENABLE_SDL=OFF -DENABLE_SHERLOCK265=OFF
	;;
libheif)
	cmake --preset=release-noplugins -S "$source_dir" -B "$source_dir/build" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
		-DCMAKE_INSTALL_LIBDIR=lib -DWITH_DAV1D=ON -DBUILD_SHARED_LIBS=ON \
		-DENABLE_MULTITHREADING_SUPPORT=ON -DENABLE_PARALLEL_TILE_DECODING=ON \
		-DWITH_LIBSHARPYUV=OFF -DWITH_X264=OFF -DBUILD_TESTING=OFF \
		-DWITH_EXAMPLES=OFF -DBUILD_DOCUMENTATION=OFF \
		-DWITH_EXAMPLE_HEIF_THUMB=OFF -DWITH_EXAMPLE_HEIF_VIEW=OFF \
		-DWITH_GDK_PIXBUF=OFF \
		-DLIBDE265_INCLUDE_DIR="$HOME/deps/libde265/include" \
		-DLIBDE265_LIBRARY="$HOME/deps/libde265/lib/libde265.$shared_ext" \
		-DX265_INCLUDE_DIR="$HOME/deps/x265/include" \
		-DX265_LIBRARY="$HOME/deps/x265/lib/libx265.$shared_ext" \
		-DDAV1D_INCLUDE_DIR="$HOME/deps/dav1d/include" \
		-DDAV1D_LIBRARY="$HOME/deps/dav1d/lib/libdav1d.$shared_ext" \
		-DAOM_INCLUDE_DIR="$HOME/deps/aom/include" \
		-DAOM_LIBRARY="$HOME/deps/aom/lib/libaom.$shared_ext"
	;;
libwebp)
	cmake -S "$source_dir" -B "$source_dir/build" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
		-DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON \
		-DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF \
		-DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF \
		-DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF \
		-DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF \
		-DWEBP_BUILD_LIBWEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF \
		-DWEBP_BUILD_WEBP_JS=OFF -DWEBP_BUILD_FUZZTEST=OFF \
		-DWEBP_ENABLE_SIMD=ON -DWEBP_USE_THREAD=ON
	;;
libjxl)
	libjxl_macos_flags=()
	if [[ "$RUNNER_OS" == macOS ]]; then
		libjxl_macos_flags=(-DCMAKE_AR="$(brew --prefix llvm)/bin/llvm-ar" -DCMAKE_RANLIB="$(brew --prefix llvm)/bin/llvm-ranlib")
	fi

	cmake -S "$source_dir" -B "$source_dir/build" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" -DBUILD_TESTING=OFF \
		-DBUILD_SHARED_LIBS=ON -DJPEGXL_ENABLE_TOOLS=OFF \
		-DJPEGXL_ENABLE_DOXYGEN=OFF -DJPEGXL_ENABLE_MANPAGES=OFF \
		-DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF \
		-DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_SJPEG=OFF \
		-DJPEGXL_ENABLE_OPENEXR=OFF -DJPEGXL_BUNDLE_LIBPNG=OFF \
		-DJPEGXL_ENABLE_FUZZERS=OFF -DJPEGXL_ENABLE_DEVTOOLS=OFF \
		-DJPEGXL_ENABLE_VIEWERS=OFF -DJPEGXL_ENABLE_PLUGINS=OFF \
		-DJPEGXL_ENABLE_COVERAGE=OFF -DJPEGXL_TEST_TOOLS=OFF \
		-DJPEGXL_ENABLE_WASM_THREADS=OFF -DJPEGXL_ENABLE_LTO=ON \
		-DJPEGXL_ENABLE_SKCMS=OFF "${libjxl_macos_flags[@]}"
	;;
esac

cmake --build "$source_dir/build" --parallel
cmake --install "$source_dir/build"
