#!/bin/bash

set -eo pipefail

# This script is called by cibuildwheel to build the C++ libraries before
# building the Python wheel.

CMAKE_BUILD_DIR="./build-release/"

# 1. Configure the C++ project with CMake
cmake -S. -B$CMAKE_BUILD_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_PYBINDS=ON \
    -DENABLE_TESTS=OFF \
    -DVSAG_VERSION=$(python -c "from setuptools_scm import get_version; print(get_version(root='..'))")

# 2. Build the C++ project
cmake --build $CMAKE_BUILD_DIR --parallel $(nproc || sysctl -n hw.ncpu)

# 3. Copy the compiled libraries into the Python source directory
PYVSAG_CPYTHON_SO=$(find $CMAKE_BUILD_DIR -name _pyvsag.cpython*.so | head -n1)

if [ -z "$PYVSAG_CPYTHON_SO" ]; then
    echo "Error: Could not find the compiled _pyvsag.cpython*.so file."
    exit 1
fi

cp "$PYVSAG_CPYTHON_SO" python/pyvsag/

# 4. Copy dependencies (you might need to adjust this based on the platform)
# On Linux, we use ldd. On macOS, we would use otool.
if [[ "$(uname)" == "Linux" ]]; then
    get_linked_libraries() {
        local lib_file=$1
        ldd "$lib_file" | awk '/=>/ {print $3}'
    }

    for lib in $(get_linked_libraries "$PYVSAG_CPYTHON_SO"); do
        if [[ $lib == *vsag* ]] || [[ $lib == *gfortran* ]] || [[ $lib == *mkl* ]]; then
            echo "Copying $lib to python/pyvsag/"
            cp "$lib" python/pyvsag/
        fi
    done
fi

# On macOS, you would do something like this:
# if [[ "$(uname)" == "Darwin" ]]; then
#     # Use otool to find and copy dependencies
# fi

echo "Successfully built and copied C++ artifacts."
