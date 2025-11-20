#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project}"
EXAMPLES_DIR="${EXAMPLES_DIR:-${PROJECT_ROOT}/examples/python}"
FILE_PATTERN="${EXAMPLE_FILE_PATTERN:-*.py}"
EXCLUDE_PATTERN="${EXAMPLE_EXCLUDE_PATTERN:-}"

if [[ ! -d "${EXAMPLES_DIR}" ]]; then
  echo "Examples directory not found: ${EXAMPLES_DIR}" >&2
  exit 1
fi

echo "Scanning for example files in ${EXAMPLES_DIR} (pattern: ${FILE_PATTERN})"

mapfile -t example_files < <(
  find "${EXAMPLES_DIR}" \
    -type f \
    -name "${FILE_PATTERN}" \
    -print | sort
)

if [[ -n "${EXCLUDE_PATTERN}" ]]; then
  mapfile -t filtered < <(printf "%s\n" "${example_files[@]}" | grep -vE "${EXCLUDE_PATTERN}" || true)
  example_files=("${filtered[@]}")
fi

if [[ "${#example_files[@]}" -eq 0 ]]; then
  echo "No example files matched pattern ${FILE_PATTERN} under ${EXAMPLES_DIR}" >&2
  exit 1
fi

for example in "${example_files[@]}"; do
  rel_path="${example#${PROJECT_ROOT}/}"
  echo "::group::Running ${rel_path}"
  python "${example}"
  echo "::endgroup::"
done

echo "Executed ${#example_files[@]} example file(s)."

