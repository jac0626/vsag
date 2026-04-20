#!/bin/bash

set -euo pipefail

VENV_DIR=".venv"
SUPPORTED_VERSIONS=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11" "3.12" "3.13" "3.14")
HAVE_DOCKER=true
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARCH=$(uname -m)

if [[ "$ARCH" == "arm64" ]]; then
  ARCH="aarch64"
fi

echo "🔎 Checking prerequisites..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ 'python3' command not found. Please ensure Python 3 is installed."
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "⚠️ Docker daemon is not running. Build without Docker."
  HAVE_DOCKER=false
fi
echo "✅ Detected local architecture: $ARCH"

echo "🔎 Setting up Python virtual environment..."
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "   - Virtual environment not found, creating at './${VENV_DIR}'..."
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  echo "✅ Virtual environment activated."
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "✅ Already in virtual environment: $VIRTUAL_ENV"
else
  echo "✅ Already in conda environment: $CONDA_PREFIX"
fi

create_stage_dir() {
  local stage_dir
  mkdir -p "${REPO_ROOT}/.tmp_pybuild_stage"
  stage_dir=$(mktemp -d "${REPO_ROOT}/.tmp_pybuild_stage/local.XXXXXX")
  bash "${REPO_ROOT}/scripts/python/stage_release_tree.sh" "$stage_dir" >/dev/null
  printf '%s\n' "$stage_dir"
}

cleanup_stage_dir() {
  local stage_dir=${1:-}
  [[ -n "$stage_dir" ]] && rm -rf "$stage_dir"
}

run_native_build() {
  local stage_dir=$1

  pip install -q build
  python -m build --wheel --outdir "${REPO_ROOT}/wheelhouse" "$stage_dir"

  local latest_wheel
  latest_wheel=$(ls -t "${REPO_ROOT}"/wheelhouse/pyvsag-*.whl | head -n 1)
  if [[ -z "$latest_wheel" ]]; then
    echo "❌ Failed to find the built wheel."
    exit 1
  fi

  echo "✅ Build complete. Starting host-native test run..."
  pip install --force-reinstall "$latest_wheel"
  pip install numpy pytest
  (
    cd "$stage_dir/tests/python"
    python -m pytest . -v --tb=short
  )
}

run_build() {
  local py_version=$1
  local stage_dir
  local cibw_build_pattern="cp$(echo "$py_version" | tr -d '.')-*"
  local cibw_version="3.3.1"
  local use_uvx=true

  if [[ "$py_version" == "3.6" ]]; then
    cibw_version="2.11.4"
    use_uvx=false
  elif [[ "$py_version" == "3.7" ]]; then
    cibw_version="2.23.3"
    use_uvx=false
  elif python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    use_uvx=false
  fi

  stage_dir=$(create_stage_dir)
  trap 'cleanup_stage_dir "$stage_dir"' RETURN

  echo "========================================================================"
  echo "🚀 Starting wheel build for Python ${py_version}"
  echo "========================================================================"

  if $HAVE_DOCKER; then
    if $use_uvx; then
      pip install -q uv 2>/dev/null || true
    else
      pip install -q "cibuildwheel==${cibw_version}"
    fi

    echo "🛠️  Building Linux wheel via cibuildwheel from staged release tree..."
    if $use_uvx; then
      CIBW_BUILD="${cibw_build_pattern}" \
      CIBW_ARCHS="${ARCH}" \
      uvx --python 3.12 --from "cibuildwheel==${cibw_version}" cibuildwheel --platform linux --output-dir "${REPO_ROOT}/wheelhouse" "$stage_dir"
    else
      CIBW_BUILD="${cibw_build_pattern}" \
      CIBW_ARCHS="${ARCH}" \
      cibuildwheel --platform linux --output-dir "${REPO_ROOT}/wheelhouse" "$stage_dir"
    fi
  else
    if [[ "$(uname -s)" == "Darwin" ]]; then
      echo "⚠️ Docker is unavailable, so this run will produce a macOS wheel instead of a Linux wheel."
    fi
    run_native_build "$stage_dir"
  fi

  echo "🎉 Successfully built wheel for Python ${py_version}!"
}

mkdir -p "${REPO_ROOT}/wheelhouse"
TARGET_VERSION=${1:-}

if [[ -z "$TARGET_VERSION" ]]; then
  echo "ℹ️  No specific version provided. Building all supported versions: ${SUPPORTED_VERSIONS[*]}"
  for version in "${SUPPORTED_VERSIONS[@]}"; do
    run_build "$version"
  done
else
  if [[ " ${SUPPORTED_VERSIONS[*]} " =~ " ${TARGET_VERSION} " ]]; then
    run_build "$TARGET_VERSION"
  else
    echo "❌ Invalid argument: '$TARGET_VERSION'"
    exit 1
  fi
fi

echo ""
echo "✅ All tasks completed."
echo "📦 Wheels have been generated in the 'wheelhouse' directory:"
ls -l "${REPO_ROOT}/wheelhouse"
