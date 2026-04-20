#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAGE_BASE="${REPO_ROOT}/.tmp_pybuild_stage"
STAGE_DIR="${1:-}"

if [[ -z "${STAGE_DIR}" ]]; then
  mkdir -p "${STAGE_BASE}"
  STAGE_DIR=$(mktemp -d "${STAGE_BASE}/stage.XXXXXX")
else
  rm -rf "${STAGE_DIR}"
  mkdir -p "${STAGE_DIR}"
fi

copy_file_if_present() {
  local path=$1
  [[ -f "${REPO_ROOT}/${path}" ]] || return 0
  mkdir -p "${STAGE_DIR}/$(dirname "${path}")"
  cp -p "${REPO_ROOT}/${path}" "${STAGE_DIR}/${path}"
}

copy_tracked_tree() {
  local path=$1
  while IFS= read -r -d '' file; do
    [[ -e "${REPO_ROOT}/${file}" ]] || continue
    mkdir -p "${STAGE_DIR}/$(dirname "${file}")"
    cp -p "${REPO_ROOT}/${file}" "${STAGE_DIR}/${file}"
  done < <(git -C "${REPO_ROOT}" ls-files -z --cached -- "${path}")
}

copy_file_if_present "CMakeLists.txt"
copy_file_if_present "LICENSE"
copy_file_if_present "README.md"
copy_file_if_present "scripts/deps/install_deps_centos.sh"
copy_file_if_present "python/pyproject.toml"
copy_file_if_present "python/setup.py"
copy_file_if_present "python/MANIFEST.in"
copy_tracked_tree "cmake"
copy_tracked_tree "include"
copy_tracked_tree "src"
copy_tracked_tree "extern"
copy_tracked_tree "python_bindings"
copy_tracked_tree "tests/python"
copy_tracked_tree "python/pyvsag"

mv "${STAGE_DIR}/python/pyproject.toml" "${STAGE_DIR}/pyproject.toml"
mv "${STAGE_DIR}/python/setup.py" "${STAGE_DIR}/setup.py"
mv "${STAGE_DIR}/python/MANIFEST.in" "${STAGE_DIR}/MANIFEST.in"
mv "${STAGE_DIR}/python/pyvsag" "${STAGE_DIR}/pyvsag"
rmdir "${STAGE_DIR}/python"

printf '%s\n' "${STAGE_DIR}"
