#!/bin/bash
set -eo pipefail

CMAKE_BUILD_DIR="./build-release/"
PYTHON_EXECUTABLE="${PYTHON:-$(which python)}"

echo "Building with Python: $PYTHON_EXECUTABLE"
$PYTHON_EXECUTABLE --version

PYTHON_VERSION=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_VERSION_NO_DOT=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

echo "Python version: $PYTHON_VERSION (cp${PYTHON_VERSION_NO_DOT})"

rm -rf "$CMAKE_BUILD_DIR"
rm -f python/pyvsag/*.so
rm -f python/pyvsag/_pyvsag*.so
rm -f python/example*.so
rm -f example*.so

mkdir -p python/pyvsag

export CXXFLAGS="-static-libstdc++ -static-libgcc -fvisibility=hidden"
export LDFLAGS="-static-libstdc++ -static-libgcc"

cmake -S. -B$CMAKE_BUILD_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG ${CXXFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS} -Wl,--exclude-libs,ALL" \
    -DENABLE_PYBINDS=ON \
    -DENABLE_TESTS=OFF \
    -DENABLE_EXAMPLES=OFF \
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE"

cmake --build $CMAKE_BUILD_DIR --parallel $(nproc)

EXPECTED_PATTERN="_pyvsag.cpython-${PYTHON_VERSION_NO_DOT}*.so"
PYVSAG_SO=$(find $CMAKE_BUILD_DIR -name "$EXPECTED_PATTERN" -type f | head -n1)

if [ -z "$PYVSAG_SO" ]; then
    PYVSAG_SO=$(find $CMAKE_BUILD_DIR -name "_pyvsag*.so" -type f | head -n1)
fi

if [ -z "$PYVSAG_SO" ]; then
    echo "Error: _pyvsag*.so not found!"
    exit 1
fi

echo "Found: $(basename $PYVSAG_SO)"

strip --strip-unneeded "$PYVSAG_SO"
cp "$PYVSAG_SO" python/pyvsag/

LIBVSAG_SO=$(find $CMAKE_BUILD_DIR -name "libvsag.so*" -type f | head -n1)
if [ -n "$LIBVSAG_SO" ]; then
    if ldd "$PYVSAG_SO" | grep -q "libvsag.so.*=> not found"; then
        strip --strip-unneeded "$LIBVSAG_SO"
        cp "$LIBVSAG_SO" python/pyvsag/
    fi
fi

find python -name "example*.so" -delete 2>/dev/null || true
find . -maxdepth 1 -name "example*.so" -delete 2>/dev/null || true

SO_COUNT=$(find python/pyvsag -name "_pyvsag*.so" | wc -l)
if [ "$SO_COUNT" -ne 1 ]; then
    echo "ERROR: Expected 1 _pyvsag*.so file, found $SO_COUNT"
    find python/pyvsag -name "_pyvsag*.so"
    exit 1
fi

echo "Build completed successfully for Python $PYTHON_VERSION!"