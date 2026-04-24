#!/usr/bin/env bash

set -euo pipefail

release_repo="${COMPATIBILITY_RELEASE_REPO:-antgroup/vsag}"
release_tag="${COMPATIBILITY_RELEASE_TAG:-compatibility-indexes}"
output_dir="${COMPATIBILITY_INDEX_DIR:-/tmp}"
max_attempts="${COMPATIBILITY_DOWNLOAD_ATTEMPTS:-3}"

mkdir -p "$output_dir"

if ! command -v gh >/dev/null 2>&1; then
    echo "Error: GitHub CLI (gh) is required to download compatibility assets"
    exit 1
fi

for attempt in $(seq 1 "$max_attempts"); do
    echo "Downloading compatibility assets from ${release_repo}@${release_tag} (attempt ${attempt}/${max_attempts})"
    if gh release download "$release_tag" \
        --repo "$release_repo" \
        --dir "$output_dir" \
        --clobber \
        --pattern "*.index" \
        --pattern "*.bin" \
        --pattern "*.json"; then
        break
    fi

    if [[ "$attempt" -eq "$max_attempts" ]]; then
        echo "Error: Failed to download compatibility assets after ${max_attempts} attempts"
        exit 1
    fi

    sleep $((attempt * 5))
done

shopt -s nullglob
index_files=("${output_dir}"/v*_*.index)
shopt -u nullglob

if [[ ${#index_files[@]} -eq 0 ]]; then
    echo "Error: No compatibility index files (v*_*.index) found in ${output_dir}"
    exit 1
fi

if [[ ! -s "${output_dir}/random_512d_10K.bin" ]]; then
    echo "Error: Missing or empty compatibility dataset: ${output_dir}/random_512d_10K.bin"
    exit 1
fi

for index_file in "${index_files[@]}"; do
    if [[ ! -s "$index_file" ]]; then
        echo "Error: Missing or empty compatibility index: ${index_file}"
        exit 1
    fi

    version=$(basename "$index_file" .index)
    for metadata_file in "${output_dir}/${version}_build.json" "${output_dir}/${version}_search.json"; do
        if [[ ! -s "$metadata_file" ]]; then
            echo "Error: Missing or empty compatibility metadata: ${metadata_file}"
            exit 1
        fi
    done
done

echo "Compatibility assets downloaded and verified in ${output_dir}"
