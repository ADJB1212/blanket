FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    CUDA_HOME=/opt/cuda-13.2 \
    NVCC=/opt/cuda-13.2/bin/nvcc
ENV PATH="/opt/cargo/bin:/opt/cuda-13.2/bin:${PATH}"

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    git \
    mold \
    ninja-build \
    nasm \
    pkg-config \
    xz-utils \
    libwebp-dev \
    libdav1d-dev \
    libheif-dev \
    libde265-dev \
    libx265-dev \
    libnuma-dev \
    liblcms2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 --fail --location --retry 5 \
    https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --no-modify-path --default-toolchain none \
    && rm /tmp/rustup-init.sh

COPY scripts/setup_cuda.sh /tmp/setup_cuda.sh
RUN RUNNER_TEMP=/opt GITHUB_PATH=/dev/null GITHUB_ENV=/dev/null \
    bash /tmp/setup_cuda.sh \
    && rm /tmp/setup_cuda.sh
