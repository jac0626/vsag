#!/bin/bash

set -eo pipefail

# This script is called by cibuildwheel to build the C++ libraries before
# building the Python wheel.

CMAKE_BUILD_DIR="./build-release/"

# 1. Configure with Release mode and optimization flags
cmake -S. -B$CMAKE_BUILD_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -s" \
    -DENABLE_PYBINDS=ON \
    -DENABLE_TESTS=OFF

# 2. Build the C++ project
cmake --build $CMAKE_BUILD_DIR --parallel $(nproc || sysctl -n hw.ncpu)

# 3. Find and process the Python extension module
PYVSAG_CPYTHON_SO=$(find $CMAKE_BUILD_DIR -name _pyvsag.cpython*.so | head -n1)

if [ -z "$PYVSAG_CPYTHON_SO" ]; then
    echo "Error: Could not find the compiled _pyvsag.cpython*.so file."
    exit 1
fi

# Strip the Python extension module
echo "Original size of $(basename $PYVSAG_CPYTHON_SO): $(du -h $PYVSAG_CPYTHON_SO | cut -f1)"
strip --strip-unneeded "$PYVSAG_CPYTHON_SO"
echo "Stripped size of $(basename $PYVSAG_CPYTHON_SO): $(du -h $PYVSAG_CPYTHON_SO | cut -f1)"

echo "Copying Python extension module: $(basename $PYVSAG_CPYTHON_SO)"
cp "$PYVSAG_CPYTHON_SO" python/pyvsag/

# 4. Handle libvsag.so if needed
LIBVSAG_SO=$(find $CMAKE_BUILD_DIR -name "libvsag.so*" -type f | head -n1)

if [ -n "$LIBVSAG_SO" ]; then
    echo "Found libvsag.so at: $LIBVSAG_SO"
    
    # Check if libvsag.so is needed in the package
    if ldd "$PYVSAG_CPYTHON_SO" | grep -q "libvsag.so.*=> not found"; then
        echo "libvsag.so not in standard path, copying to package"
        
        # Strip libvsag.so before copying
        echo "Original size of $(basename $LIBVSAG_SO): $(du -h $LIBVSAG_SO | cut -f1)"
        strip --strip-unneeded "$LIBVSAG_SO"
        echo "Stripped size of $(basename $LIBVSAG_SO): $(du -h $LIBVSAG_SO | cut -f1)"
        
        cp "$LIBVSAG_SO" python/pyvsag/
    else
        # Even if letting auditwheel handle it, strip it in place
        # so auditwheel copies the stripped version
        echo "Stripping libvsag.so in build directory"
        echo "Original size of $(basename $LIBVSAG_SO): $(du -h $LIBVSAG_SO | cut -f1)"
        strip --strip-unneeded "$LIBVSAG_SO"
        echo "Stripped size of $(basename $LIBVSAG_SO): $(du -h $LIBVSAG_SO | cut -f1)"
        
        echo "libvsag.so is accessible, letting auditwheel handle it"
    fi
fi

# 5. Strip all other SO files in the build directory
# This ensures auditwheel will copy already-stripped versions
echo "Stripping all SO files in build directory..."
find $CMAKE_BUILD_DIR -name "*.so*" -type f -exec sh -c '
    for file; do
        echo "Stripping: $(basename $file)"
        strip --strip-unneeded "$file" 2>/dev/null || true
    done
' sh {} +

# 6. Set library path for auditwheel
if [ -d "$CMAKE_BUILD_DIR/lib" ]; then
    export LD_LIBRARY_PATH="$CMAKE_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
    echo "Added $CMAKE_BUILD_DIR/lib to LD_LIBRARY_PATH for auditwheel"
fi

# Display dependency information
echo "Extension module dependencies:"
ldd "$PYVSAG_CPYTHON_SO" || true

# Display final sizes
echo ""
echo "=== Final package content sizes ==="
echo "Python package directory:"
du -sh python/pyvsag/ 2>/dev/null || echo "Package directory not yet complete"
echo "Individual files:"
ls -lh python/pyvsag/*.so 2>/dev/null || echo "No .so files in package yet"

echo ""
echo "Successfully built and copied Python extension module."
echo "Dependencies will be handled by auditwheel."