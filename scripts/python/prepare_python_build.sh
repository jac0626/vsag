#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# This script prepares the project for building a Python wheel.
# It handles version generation and config file patching based on the Python version.

# --- Input Validation ---
if [ -z "$1" ]; then
  echo "❌ Error: Python version must be provided as the first argument."
  echo "Usage: $0 <python_version>"
  exit 1
fi

PY_VERSION=$1
echo "🚀 Preparing build for Python ${PY_VERSION}..."

# --- Main Logic ---

# 1. Statically generate the version file using setuptools_scm
echo "   - Installing setuptools_scm to generate version..."
pip install -q setuptools_scm

echo "   - Generating python/pyvsag/_version.py..."
python3 scripts/python/generate_version_file.py --fallback "0.0.0"

# 2. Patch pyproject.toml to remove dynamic versioning and set constraints
echo "   - Patching python/pyproject.toml..."
# Remove the entire [tool.setuptools_scm] section
sed -i '/\[tool.setuptools_scm\]/,/^$/d' python/pyproject.toml

# Apply version-specific constraints
case $PY_VERSION in
  "3.6")
    sed -i 's/setuptools>=61.0/setuptools<60/g' python/pyproject.toml
    sed -i 's/setuptools_scm\[toml\]>=6.2/setuptools_scm[toml]<7/g' python/pyproject.toml
    ;;
  "3.7")
    sed -i "s/setuptools>=61.0/setuptools>=61.0,<67/g" python/pyproject.toml
    ;;
  *)
    # No additional patch needed for Python 3.8+
    ;;
esac
echo "   - pyproject.toml patched."

# Generate version file if not already present or if version is explicitly set
PYVSAG_VERSION="${PYVSAG_VERSION:-0.17.2}"
if [ -n "$PYVSAG_VERSION" ]; then
    echo "=== Generating version file ==="
    echo "   - Version: $PYVSAG_VERSION"
    python3 scripts/python/generate_version_file.py --version "$PYVSAG_VERSION"
else
    echo "=== Generating version file (auto-detect) ==="
    python3 scripts/python/generate_version_file.py --fallback "0.0.0"
fi

echo "✅ Build preparation complete for Python ${PY_VERSION}."
