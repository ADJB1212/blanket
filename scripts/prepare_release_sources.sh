#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
project_git() {
	git -c "safe.directory=$project_root" -C "$project_root" "$@"
}
if ! project_git rev-parse --show-toplevel >/dev/null; then
	echo "Release sources require a Git checkout" >&2
	exit 1
fi
if ! project_git diff --quiet HEAD --; then
	echo "Release sources must be prepared from an unmodified commit" >&2
	exit 1
fi
staging=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/blanket-sources.XXXXXX")
mkdir -p "$staging/project/.cargo" "$staging/native" "$project_root/licenses/third-party"
project_git archive HEAD | tar -x -C "$staging/project"
(
	cd "$staging/project"
	cargo vendor --locked --versioned-dirs ../vendor >.cargo/config.toml
)
bash scripts/collect_release_licenses.sh "$staging/vendor" "$project_root/licenses/third-party/rust"
cargo metadata --locked --offline --format-version 1 >"$staging/cargo-metadata.json"
for dependency in aom x265 libde265 dav1d libheif libwebp libjxl lcms2; do
	dependency_root="$HOME/deps/$dependency/share"
	test -s "$dependency_root/blanket-sources/$dependency.tar.gz"
	test -s "$dependency_root/blanket-sources/$dependency.txt"
	cp "$dependency_root/blanket-sources/$dependency.tar.gz" "$staging/native/"
	cp "$dependency_root/blanket-sources/$dependency.txt" "$staging/native/"
	cp -R "$dependency_root/blanket-licenses/$dependency" "$project_root/licenses/third-party/"
done
printf 'Corresponding source: https://github.com/ADJB1212/blanket/releases/tag/%s\n' \
	"${BLANKET_RELEASE_TAG:?Release tag required}" >"$project_root/licenses/third-party/SOURCE.txt"
cp -R "$project_root/licenses/third-party" "$staging/project/licenses/"
project_git rev-parse HEAD >"$staging/project.commit"
mkdir -p dist
tar -cjf "dist/pyblanket-sources-${RUNNER_OS:?}-${RUNNER_ARCH:?}.tar.bz2" -C "$staging" .
