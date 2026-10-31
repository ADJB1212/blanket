#!/usr/bin/env bash
set -euo pipefail

# CUDA 13.2.0 redistributions: https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.0.json
case "$(uname -m)" in
    x86_64)
        cuda_arch=linux-x86_64
        cuda_hashes=(
            706b996fefc59dc8d64d317fdf48d0aa84c4ae004eff43009dd918f40c5cc66a
            539edc1056e44d319f2112e9971c6415d78d4dde04b3f6ffbd20ec808e718526
            fbc31fed55b7255591f3a19f575ca078827f5e6757d317d009f7ec1e69fcde4b
            e013fce38130d2337ea695aadc5ddd5dcfb78f9107903d72492b9819539749bb
        )
        ;;
    aarch64)
        cuda_arch=linux-sbsa
        cuda_hashes=(
            28d201ffc46627abd919d623d85e721f33c92cf02749e023cd66d6fa0f1206a7
            379363621c59c8d6faeea2f7a527eae91fc7db8cb2121cb42fe7972a48462f67
            04f8fd03bc5a99507f5d531cc65452881c8949311bc5cc87bc1fe05a7607fd9e
            366da4383992b8d13df63ffecd3a8315815955f50da0c0288ca5123212f67e02
        )
        ;;
    *)
        echo "CUDA builds require Linux x86_64 or aarch64" >&2
        exit 1
        ;;
esac

cuda_root="${RUNNER_TEMP:?RUNNER_TEMP must be set}/cuda-13.2"
cuda_downloads=$(mktemp -d "$RUNNER_TEMP/cuda-downloads.XXXXXX")
mkdir -p "$cuda_root"
cuda_components=(cuda_nvcc cuda_cudart cuda_crt libnvvm)
for index in "${!cuda_components[@]}"; do
    component="${cuda_components[$index]}"
    archive="$component-$cuda_arch-13.2.51-archive.tar.xz"
    curl --fail --location --retry 3 "https://developer.download.nvidia.com/compute/cuda/redist/$component/$cuda_arch/$archive" -o "$cuda_downloads/$archive"
    printf '%s  %s\n' "${cuda_hashes[$index]}" "$cuda_downloads/$archive" | sha256sum --check
    tar -xJf "$cuda_downloads/$archive" --strip-components=1 -C "$cuda_root"
done

"$cuda_root/bin/nvcc" --version
echo "$cuda_root/bin" >> "${GITHUB_PATH:?GITHUB_PATH must be set}"
echo "NVCC=$cuda_root/bin/nvcc" >> "${GITHUB_ENV:?GITHUB_ENV must be set}"
