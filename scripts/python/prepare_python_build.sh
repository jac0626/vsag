#!/bin/bash

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Error: Python version must be provided as the first argument." >&2
  echo "Usage: $0 <python_version>" >&2
  exit 1
fi

PY_VERSION="$1"
SUPPORTED_VERSIONS=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11" "3.12")

if [[ ! " ${SUPPORTED_VERSIONS[*]} " =~ " ${PY_VERSION} " ]]; then
  echo "Error: Unsupported Python version '${PY_VERSION}'." >&2
  echo "Supported versions: ${SUPPORTED_VERSIONS[*]}" >&2
  exit 1
fi

echo "Preparing Python ${PY_VERSION} build environment..."

if [ "${PY_VERSION}" = "3.6" ]; then
  echo "- Using legacy backend pins: setuptools<60, setuptools_scm<7"
elif [ "${PY_VERSION}" = "3.7" ]; then
  echo "- Using transitional backend pins: setuptools>=61,<68, setuptools_scm>=6.2,<8"
else
  echo "- Using modern backend pins: setuptools>=68, setuptools_scm>=8"
fi

echo "- Build dependency pins are resolved by python/pyproject.toml markers."
echo "Preparation complete."
