#!/bin/bash

set -euo pipefail

VENV_DIR=".venv"
SUPPORTED_VERSIONS=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11" "3.12")
HAVE_DOCKER=true

cleanup() {
  rm -f python/pyvsag/_version.py
}

is_supported_version() {
  local version="$1"
  [[ " ${SUPPORTED_VERSIONS[*]} " =~ " ${version} " ]]
}

echo "Checking prerequisites..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: 'python3' command not found."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Falling back to local build mode."
  HAVE_DOCKER=false
fi

echo "Setting up Python virtual environment..."
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "- Creating virtual environment at ./${VENV_DIR}"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
  ARCH="aarch64"
fi

echo "Detected architecture: ${ARCH}"

run_build() {
  local py_version="$1"
  local cibw_build_pattern="cp$(echo "$py_version" | tr -d '.')-*"
  local cibw_version="2.19.1"

  if [ "$py_version" = "3.6" ]; then
    cibw_version="2.11.4"
  fi

  echo "========================================================================"
  echo "Starting wheel build for Python ${py_version}"
  echo "========================================================================"

  python -m pip install --upgrade pip >/dev/null
  if $HAVE_DOCKER; then
    python -m pip install -q "cibuildwheel==${cibw_version}"
  else
    python -m pip install -q build
  fi

  bash ./scripts/python/prepare_python_build.sh "$py_version"

  export PIP_DEFAULT_TIMEOUT="100"
  export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
  export COPYFILE_DISABLE=1
  export COPY_EXTENDED_ATTRIBUTES_DISABLE=1

  if $HAVE_DOCKER; then
    echo "Running cibuildwheel..."
    CIBW_BUILD="${cibw_build_pattern}" \
    CIBW_ARCHS="${ARCH}" \
    CIBW_TEST_COMMAND="pip install numpy && python /project/tests/python/run_test.py" \
    cibuildwheel --platform linux --output-dir wheelhouse python
  else
    echo "Running local build..."
    bash scripts/python/build_cpp_for_cibw.sh false
    python -m build --wheel --outdir wheelhouse python

    local latest_wheel
    latest_wheel="$(ls -t wheelhouse/pyvsag-*.whl | head -n 1)"
    if [ -z "$latest_wheel" ]; then
      echo "Error: Failed to find built wheel."
      exit 1
    fi

    python -m pip install --force-reinstall "$latest_wheel"
    python -m pip install -q numpy
    python tests/python/run_test.py
  fi

  cleanup
  echo "Wheel build completed for Python ${py_version}."
}

mkdir -p wheelhouse
TARGET_VERSION="${1:-}"

if [ -z "$TARGET_VERSION" ]; then
  echo "No specific version provided. Building all supported versions: ${SUPPORTED_VERSIONS[*]}"
  for version in "${SUPPORTED_VERSIONS[@]}"; do
    run_build "$version"
  done
else
  if is_supported_version "$TARGET_VERSION"; then
    run_build "$TARGET_VERSION"
  else
    echo "Error: Invalid version '$TARGET_VERSION'. Supported versions: ${SUPPORTED_VERSIONS[*]}"
    exit 1
  fi
fi

echo "All tasks completed."
ls -l wheelhouse
