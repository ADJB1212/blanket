#!/usr/bin/env bash
set -euo pipefail

source_root=${1%/}
destination=${2%/}
mkdir -p "$destination"
count=0
while IFS= read -r -d '' notice; do
	relative=${notice#"$source_root"/}
	mkdir -p "$destination/$(dirname "$relative")"
	cp "$notice" "$destination/$relative"
	count=$((count + 1))
done < <(find "$source_root" -type f \
	\( -iname '*license*' -o -iname '*copying*' -o -iname '*notice*' \
	-o -iname '*copyright*' -o -iname 'authors*' -o -iname 'patents*' \
	-o -iname 'unlicense*' -o -name 'README.ijg' \) -print0)
if [[ "$count" == 0 ]]; then
	echo "No license files found in $source_root" >&2
	exit 1
fi
