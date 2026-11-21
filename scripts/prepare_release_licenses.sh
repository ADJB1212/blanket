#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
license_root="$project_root/licenses/third-party"
mkdir -p "$license_root"
packages=$(mktemp "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/blanket-licenses.XXXXXX")
trap 'rm -f "$packages"' EXIT
cargo metadata --locked --format-version 1 | python3 -c '
import json
import sys
from pathlib import Path

for package in json.load(sys.stdin)["packages"]:
    if package["source"] is not None:
        name = package["name"] + "-" + package["version"]
        print(name, Path(package["manifest_path"]).parent, sep="\t")
' >"$packages"
while IFS=$'\t' read -r name source_root; do
	bash scripts/collect_release_licenses.sh "$source_root" "$license_root/rust/$name" --allow-empty
done <"$packages"
bash scripts/collect_release_licenses.sh "$project_root/third-party" "$license_root/rust"
mkdir -p "$license_root/lcms2"
cp "$project_root/third-party/lcms2-sys/LICENSE" "$license_root/lcms2/LICENSE"
for dependency in aom x265 libde265 dav1d libheif libwebp libjxl; do
	cp -R "$HOME/deps/$dependency/share/blanket-licenses/$dependency" "$license_root/"
done
