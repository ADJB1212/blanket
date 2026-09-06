#!/usr/bin/env bash
set -eu
mkdir -p /tmp/fake-libcxx
ln -sf "$(find /opt/homebrew/Cellar/x86_64-unknown-linux-gnu -name 'libstdc++.so*' | head -1)" /tmp/fake-libcxx/libc++.so
RUSTFLAGS="-L /tmp/fake-libcxx" CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc CXX=x86_64-linux-gnu-g++ CC=x86_64-linux-gnu-gcc maturin build -r --target x86_64-unknown-linux-gnu -i python3.14