#!/usr/bin/env bash

set -euo pipefail

compatibility_index_dir="${COMPATIBILITY_INDEX_DIR:-/tmp}"

old_version_indexes=()
shopt -s nullglob
for index_file in "${compatibility_index_dir}"/v*_*.index; do
    if [[ -f "$index_file" ]]; then
        name=$(basename "$index_file" .index)
        old_version_indexes+=("$name")
    fi
done
shopt -u nullglob

if [[ ${#old_version_indexes[@]} -eq 0 ]]; then
    echo "Error: No compatibility index files (v*_*.index) found in ${compatibility_index_dir}"
    exit 1
fi

IFS=$'\n' old_version_indexes=($(printf '%s\n' "${old_version_indexes[@]}" | sort))
unset IFS

all_success=true

for version in "${old_version_indexes[@]}"; do
    echo "Checking compatibility for: $version"
    missing_file=false
    required_files=(
        "${compatibility_index_dir}/${version}.index"
        "${compatibility_index_dir}/${version}_build.json"
        "${compatibility_index_dir}/${version}_search.json"
        "${compatibility_index_dir}/random_512d_10K.bin"
    )
    for required_file in "${required_files[@]}"; do
        if [[ ! -s "$required_file" ]]; then
            echo "Error: Missing or empty compatibility input: ${required_file}"
            missing_file=true
        fi
    done
    if [[ "$missing_file" = true ]]; then
        all_success=false
        break
    fi

    if ! COMPATIBILITY_INDEX_DIR="$compatibility_index_dir" \
        ./build-release/tools/check_compatibility/check_compatibility "$version"; then
        echo "Error: Compatibility check failed for $version"
        all_success=false
        break
    fi
done

if [ "$all_success" = true ]; then
    echo "All compatibility checks passed"
    exit 0
else
    exit 1
fi
