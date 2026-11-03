#!/usr/bin/env bash
set -euo pipefail

# CUDA 13.2.0 redistributions: https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.2.json
case "$(uname -m)" in
x86_64)
	cuda_arch=linux-x86_64
	cuda_hashes=(
		b04e5ca5b41f8b5629ce5f16930920e6af437f32c9777fcb67e0361f732b6f43
		d6be2b3546fa93e5dec9e3d118d31f9540d0b7af5319ee3c6726f202dfc5b5d4
		174ad29ab90c9301bd25f32f935345e65f19fbb0f39c15788daae25b0e924892
		029fc2a43904a2208a83dfd5834c4c343394d1117279f849863ae94208a88b8a
	)
	;;
aarch64)
	cuda_arch=linux-sbsa
	cuda_hashes=(
		0ae0716074fb8ca9a2eaebb0acd1425d8a41904ace1320793bbad270c3ce043f
		30d07b79d832b517cedced78a9e1aa63ebfaa8d71e4df6d1943cdf33fe563400
		c7a8f385c86413bba4d2c522d32ec7750caed027cb82c121574f611abc808ef8
		14eb9d4d180f8466b3eb25e6801f48b7790b614a19528b876ec5d1b98abf05a1
	)
	;;
*)
	echo "CUDA builds require Linux x86_64 or aarch64" >&2
	exit 1
	;;
esac

cuda_root="${RUNNER_TEMP:?RUNNER_TEMP must be set}/cuda-13.2"
cuda_downloads=$(mktemp -d "$RUNNER_TEMP/cuda-downloads.XXXXXX")
trap 'rm -rf -- "$cuda_downloads"' EXIT
mkdir -p "$cuda_root"
cuda_components=(cuda_nvcc cuda_cudart cuda_crt libnvvm)
for index in "${!cuda_components[@]}"; do
	component="${cuda_components[$index]}"
	archive="$component-$cuda_arch-13.2.86-archive.tar.xz"
	curl --fail --location --retry 3 "https://developer.download.nvidia.com/compute/cuda/redist/$component/$cuda_arch/$archive" -o "$cuda_downloads/$archive"
	printf '%s  %s\n' "${cuda_hashes[$index]}" "$cuda_downloads/$archive" | sha256sum --check
	tar -xJf "$cuda_downloads/$archive" --strip-components=1 -C "$cuda_root"
done

"$cuda_root/bin/nvcc" --version
echo "$cuda_root/bin" >>"${GITHUB_PATH:?GITHUB_PATH must be set}"
echo "NVCC=$cuda_root/bin/nvcc" >>"${GITHUB_ENV:?GITHUB_ENV must be set}"
