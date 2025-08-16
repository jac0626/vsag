#!/bin/bash

set -eo pipefail

# This script is called by cibuildwheel to build the C++ libraries before
# building the Python wheel.

CMAKE_BUILD_DIR="./build-release/"


cmake -S. -B$CMAKE_BUILD_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_PYBINDS=ON \
    -DENABLE_TESTS=OFF \
    -DVSAG_VERSION=$(python -c "from setuptools_scm import get_version; print(get_version(root='..'))")


cmake --build $CMAKE_BUILD_DIR --parallel $(nproc || sysctl -n hw.ncpu)


PYVSAG_CPYTHON_SO=$(find $CMAKE_BUILD_DIR -name _pyvsag.cpython*.so | head -n1)

if [ -z "$PYVSAG_CPYTHON_SO" ]; then
    echo "Error: Could not find the compiled _pyvsag.cpython*.so file."
    exit 1
fi

echo "Copying Python extension module: $(basename $PYVSAG_CPYTHON_SO)"
cp "$PYVSAG_CPYTHON_SO" python/pyvsag/


LIBVSAG_SO=$(find $CMAKE_BUILD_DIR -name "libvsag.so*" -type f | head -n1)

if [ -n "$LIBVSAG_SO" ]; then
    echo "Found libvsag.so at: $LIBVSAG_SO"
    
    
    if ldd "$PYVSAG_CPYTHON_SO" | grep -q "libvsag.so.*=> not found"; then
        echo "libvsag.so not in standard path, copying to package"
        cp "$LIBVSAG_SO" python/pyvsag/
    else
        echo "libvsag.so is accessible, letting auditwheel handle it"
    fi
fi




echo "Extension module dependencies:"
ldd "$PYVSAG_CPYTHON_SO" || true

echo "Successfully built and copied Python extension module."
echo "Dependencies will be handled by auditwheel."