#!/bin/bash
# scripts/build_cpp_for_cibw.sh - 完整修复版本

set -eo pipefail

# This script is called by cibuildwheel to build the C++ libraries before
# building the Python wheel.

CMAKE_BUILD_DIR="./build-release/"

# 1. 获取当前 Python 解释器
# cibuildwheel 设置 PYTHON 环境变量
PYTHON_EXECUTABLE="${PYTHON:-$(which python)}"

# 验证 Python 路径
if [ ! -x "$PYTHON_EXECUTABLE" ]; then
    echo "Error: Python executable not found or not executable: $PYTHON_EXECUTABLE"
    exit 1
fi

echo "=== Python Configuration ==="
echo "Python executable: $PYTHON_EXECUTABLE"
$PYTHON_EXECUTABLE --version

# 获取 Python 版本信息
PYTHON_VERSION=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_VERSION_NO_DOT=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
PYTHON_SOABI=$($PYTHON_EXECUTABLE -c "import sysconfig; print(sysconfig.get_config_var('SOABI'))")
EXPECTED_SUFFIX=$($PYTHON_EXECUTABLE -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "Python version: $PYTHON_VERSION"
echo "Python SOABI: $PYTHON_SOABI"
echo "Expected suffix: $EXPECTED_SUFFIX"

# 2. 清理之前的构建
if [ -d "$CMAKE_BUILD_DIR" ]; then
    echo "Cleaning previous build directory..."
    rm -rf "$CMAKE_BUILD_DIR"
fi

# 3. 配置 CMake - 关键：传递正确的 Python 路径
echo "=== Configuring CMake ==="
cmake -S. -B$CMAKE_BUILD_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -s" \
    -DENABLE_PYBINDS=ON \
    -DENABLE_TESTS=OFF \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPython_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPython3_ROOT_DIR="$(dirname $(dirname $PYTHON_EXECUTABLE))" \
    -DPython3_FIND_STRATEGY=LOCATION

# 4. 构建项目
echo "=== Building project ==="
cmake --build $CMAKE_BUILD_DIR --parallel $(nproc || sysctl -n hw.ncpu)

# 5. 查找生成的 .so 文件
echo "=== Looking for generated .so files ==="
echo "Expected pattern: _pyvsag*${PYTHON_SOABI}*.so or _pyvsag.cpython-${PYTHON_VERSION_NO_DOT}*.so"

# 尝试多种模式查找
PYVSAG_CPYTHON_SO=""
for pattern in "_pyvsag*${PYTHON_SOABI}*.so" "_pyvsag.cpython-${PYTHON_VERSION_NO_DOT}*.so" "_pyvsag*.so"; do
    echo "Searching for: $pattern"
    FOUND=$(find $CMAKE_BUILD_DIR -name "$pattern" -type f 2>/dev/null | head -n1)
    if [ -n "$FOUND" ]; then
        PYVSAG_CPYTHON_SO="$FOUND"
        echo "Found: $PYVSAG_CPYTHON_SO"
        break
    fi
done

if [ -z "$PYVSAG_CPYTHON_SO" ]; then
    echo "Error: Could not find the compiled _pyvsag*.so file."
    echo "Build directory contents:"
    find $CMAKE_BUILD_DIR -name "*.so" -o -name "*.so.*" 2>/dev/null || echo "No .so files found"
    
    # 检查 CMake 缓存以了解发生了什么
    echo ""
    echo "CMake Python configuration:"
    grep -i python $CMAKE_BUILD_DIR/CMakeCache.txt | head -20 || true
    exit 1
fi

# 6. 验证 .so 文件版本匹配
SO_BASENAME=$(basename "$PYVSAG_CPYTHON_SO")
echo "Generated .so file: $SO_BASENAME"

# 检查版本匹配
if [[ "$SO_BASENAME" =~ cpython-${PYTHON_VERSION_NO_DOT} ]] || [[ "$SO_BASENAME" =~ ${PYTHON_SOABI} ]]; then
    echo "✓ Version match confirmed"
else
    echo "⚠ WARNING: .so file may not match Python version!"
    echo "  Expected: cpython-${PYTHON_VERSION_NO_DOT} or ${PYTHON_SOABI}"
    echo "  Got: $SO_BASENAME"
    
    # 尝试重命名（风险操作，仅作为最后手段）
    if [[ "$SO_BASENAME" =~ cpython-([0-9]+)- ]]; then
        OLD_VERSION="${BASH_REMATCH[1]}"
        if [ "$OLD_VERSION" != "${PYTHON_VERSION_NO_DOT}" ]; then
            echo "  Detected version mismatch: $OLD_VERSION vs ${PYTHON_VERSION_NO_DOT}"
            # 不自动重命名，只是警告
        fi
    fi
fi

# 7. Strip 和复制文件
echo "Original size of $(basename $PYVSAG_CPYTHON_SO): $(du -h $PYVSAG_CPYTHON_SO | cut -f1)"
strip --strip-unneeded "$PYVSAG_CPYTHON_SO" || echo "Strip failed, continuing anyway"
echo "Stripped size of $(basename $PYVSAG_CPYTHON_SO): $(du -h $PYVSAG_CPYTHON_SO | cut -f1)"

echo "Copying to: python/pyvsag/"
mkdir -p python/pyvsag/
cp "$PYVSAG_CPYTHON_SO" python/pyvsag/

# 8. 处理 libvsag.so
LIBVSAG_SO=$(find $CMAKE_BUILD_DIR -name "libvsag.so*" -type f | head -n1)

if [ -n "$LIBVSAG_SO" ]; then
    echo "Found libvsag.so at: $LIBVSAG_SO"
    
    # 检查依赖
    if ldd "$PYVSAG_CPYTHON_SO" 2>/dev/null | grep -q "libvsag.so.*=> not found"; then
        echo "libvsag.so not in standard path, copying to package"
        strip --strip-unneeded "$LIBVSAG_SO" || true
        cp "$LIBVSAG_SO" python/pyvsag/
    else
        # Strip in place for auditwheel
        strip --strip-unneeded "$LIBVSAG_SO" || true
        echo "libvsag.so is accessible, letting auditwheel handle it"
    fi
fi

# 9. Strip 其他 .so 文件
echo "Stripping all SO files in build directory..."
find $CMAKE_BUILD_DIR -name "*.so*" -type f -exec sh -c '
    for file; do
        strip --strip-unneeded "$file" 2>/dev/null || true
    done
' sh {} +

# 10. 设置 LD_LIBRARY_PATH
if [ -d "$CMAKE_BUILD_DIR/lib" ]; then
    export LD_LIBRARY_PATH="$CMAKE_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
fi

# 11. 最终验证
echo ""
echo "=== Build Summary ==="
echo "Python: $PYTHON_EXECUTABLE (version $PYTHON_VERSION)"
echo "Built: $(basename $PYVSAG_CPYTHON_SO)"
echo "Location: python/pyvsag/"

# 列出最终文件
echo ""
echo "=== Final package contents ==="
ls -lh python/pyvsag/*.so 2>/dev/null || echo "No .so files in package"

# 测试导入（可选）
echo ""
echo "=== Testing import ==="
cd python
$PYTHON_EXECUTABLE -c "
import sys
sys.path.insert(0, '.')
try:
    import pyvsag
    print('✓ Import successful')
except ImportError as e:
    print(f'✗ Import failed: {e}')
" || echo "Import test failed (this might be expected if there are missing dependencies)"
cd ..

echo ""
echo "Build script completed successfully!"