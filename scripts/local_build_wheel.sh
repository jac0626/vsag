#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_DIR=".venv"
FILES_TO_PATCH=("python/pyproject.toml" "python/setup.py")
SUPPORTED_VERSIONS=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11" "3.12")

# --- Prerequisite Checks ---
echo "🔎 Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 'python3' command not found. Please ensure Python 3 is installed."
    exit 1
fi
if ! docker info > /dev/null 2>&1; then
  echo "❌ Docker daemon is not running. Please start Docker and try again."
  exit 1
fi
echo "✅ Prerequisites met (Python 3, Docker)."

# --- Virtual Environment Setup ---
echo "🔎 Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
  echo "   - Virtual environment not found, creating at './${VENV_DIR}'..."
  python3 -m venv "$VENV_DIR"
else
  echo "   - Existing virtual environment found."
fi

echo "   - Activating virtual environment..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
echo "✅ Virtual environment activated."

# --- Auto-detect Architecture ---
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
  ARCH="aarch64"
elif [[ "$ARCH" != "x86_64" && "$ARCH" != "aarch64" ]]; then
  echo "❌ Unsupported architecture: $ARCH. Only x86_64 and aarch64 are supported."
  exit 1
fi
echo "✅ Detected local architecture: $ARCH"

# --- Cleanup Function ---
cleanup() {
  echo "🧹 Cleaning up and restoring files..."
  for file in "${FILES_TO_PATCH[@]}"; do
    if [ -f "${file}.bak" ]; then
      echo "   - Restoring ${file}"
      mv "${file}.bak" "${file}"
    fi
  done
  # Remove temporary files created for the build
  rm -f python/pyvsag/_version.py python/example.c
  echo "✅ Cleanup complete."
}

# --- Main Build Function ---
run_build() {
  local py_version=$1
  local cibw_build_pattern="cp$(echo "$py_version" | tr -d '.')-*"
  local cibw_version=""
  local setuptools_scm_req=""

  echo "========================================================================"
  echo "🚀 Starting wheel build for Python ${py_version}"
  echo "========================================================================"

  # Set dependency versions based on the Python version
  case $py_version in
    "3.6")
      cibw_version="2.11.4"
      setuptools_scm_req="setuptools_scm<7"
      ;;
    *)
      cibw_version="2.19.1"
      setuptools_scm_req="setuptools_scm>=6.2"
      ;;
  esac

  echo "📦 Installing version-specific build dependencies in the virtual environment..."
  pip install -q "cibuildwheel==${cibw_version}" "${setuptools_scm_req}"
  echo "   - Installed cibuildwheel==${cibw_version}"
  echo "   - Installed ${setuptools_scm_req}"

  # Set a trap to call the cleanup function on exit
  trap cleanup EXIT

  echo "🛡️  Backing up files that might be modified..."
  for file in "${FILES_TO_PATCH[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "${file}.bak"
    fi
  done

  # Apply patches based on the Python version
  case $py_version in
    "3.6")
      echo "⚙️  Applying patches for Python 3.6..."
      sed -i 's/setuptools>=61.0/setuptools<60/g' python/pyproject.toml
      sed -i 's/setuptools_scm\[toml\]>=6.2/setuptools_scm[toml]<7/g' python/pyproject.toml
      sed -i '/\[tool.setuptools_scm\]/,/^$/d' python/pyproject.toml
      python3 -c "
import setuptools_scm
version = setuptools_scm.get_version()
with open('python/pyvsag/_version.py', 'w') as f:
    f.write(f'__version__ = \"{version}\"\\n')
"
      cat > python/setup.py << 'EOF'
from setuptools import setup, Extension
import os
version = "0.0.0"  
version_file = os.path.join(os.path.dirname(__file__), "pyvsag", "_version.py")
if os.path.exists(version_file):
  with open(version_file) as f:
    exec(f.read())
    version = __version__
setup(
  version=version,
  ext_modules=[Extension('example', sources=['example.c'])],
  zip_safe=False,
)
EOF
      touch python/example.c
      ;;
    "3.7")
      echo "⚙️  Applying patches for Python 3.7..."
      sed -i "s/setuptools>=61.0/setuptools>=61.0,<67/g" python/pyproject.toml
      ;;
    *)
      echo "⚙️  No patch needed for Python 3.8+."
      ;;
  esac

  echo "🛠️  Starting cibuildwheel..."
  # Pass PIP_DEFAULT_TIMEOUT and PIP_INDEX_URL for better network reliability
  PIP_DEFAULT_TIMEOUT="100" \
  PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
  CIBW_BUILD="${cibw_build_pattern}" \
  CIBW_ARCHS="${ARCH}" \
  CIBW_TEST_COMMAND="pip install numpy && ls -alF /project/ && python /project/examples/python/example_hnsw.py" \
  cibuildwheel --platform linux --output-dir wheelhouse python

  cleanup
  trap - EXIT

  echo "🎉 Successfully built wheel for Python ${py_version}!"
}

# --- Main Logic ---
mkdir -p wheelhouse
TARGET_VERSION=$1

if [ -z "$TARGET_VERSION" ]; then
  echo "ℹ️  No specific version provided. Building all supported versions: ${SUPPORTED_VERSIONS[*]}"
  for version in "${SUPPORTED_VERSIONS[@]}"; do
    run_build "$version"
  done
else
  if [[ " ${SUPPORTED_VERSIONS[*]} " =~ " ${TARGET_VERSION} " ]]; then
    run_build "$TARGET_VERSION"
  else
    echo "❌ Invalid argument: '$TARGET_VERSION'"
    echo "Usage: $0 [VERSION]"
    echo "Supported versions: ${SUPPORTED_VERSIONS[*]}"
    echo "If no argument is provided, all versions will be built."
    exit 1
  fi
fi

echo ""
echo "✅ All tasks completed."
echo "📦 Wheels have been generated in the 'wheelhouse' directory:"
ls -l wheelhouse
