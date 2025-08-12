#!/bin/bash

CLEAN_BUILD_DIR=true
DEBUG=true
debug_echo() {
    if [ "$DEBUG" = true ]; then
        echo -e "DEBUG::$*"
    fi
}

PYTHON_VERSIONS=("3.6" "3.7" "3.8" "3.9" "3.10" "3.11" "3.12")
AVAILABLE_PYTHON_PATHS=()

check_env_python() {
    for version in "${PYTHON_VERSIONS[@]}"; do
        local path
        path=$(which python$version 2>/dev/null)

        if [[ -n "$path" ]]; then
            debug_echo "check_env_python::python \e[32m$version\e[0m is installed at $path."
            AVAILABLE_PYTHON_PATHS+=("$path")
        else
            debug_echo "check_env_python::python \e[31m$version\e[0m is not installed."
        fi
    done
}

get_linked_libraries() {
    local PYVSAG_CPYTHON_SO=$1

    link_libraries=$(ldd $PYVSAG_CPYTHON_SO | awk '/=>/ {print $3}')
    echo "$link_libraries"
}

build_pyvsag() {
    set -e

    local PYTHON_PATH=$1
    local CMAKE_ARGS=$2
    local CMAKE_BUILD_ARGS=$3
    local CMAKE_BUILD_DIR=$4

    # step 0: clean build dir
    if [ ! -z "${CMAKE_BUILD_DIR}" ]; then
        if [ "$CLEAN_BUILD_DIR" = true ]; then
            rm $CMAKE_BUILD_DIR/* -rf
        fi
    fi

  

    CMAKE_ARGS=($CMAKE_ARGS -DENABLE_INTEL_MKL=OFF -DPython3_EXECUTABLE=$PYTHON_PATH)
    
    debug_echo "build_pyvsag::python path: $PYTHON_PATH"
    debug_echo "build_pyvsag::cmake config args: ${CMAKE_ARGS[@]}"
    debug_echo "build_pyvsag::cmake build args: $CMAKE_BUILD_ARGS"
    cmake "${CMAKE_ARGS[@]}"
    cmake $CMAKE_BUILD_ARGS

    # step 1: collect libraries
    PYVSAG_CPYTHON_SO_COUNT=$(find $CMAKE_BUILD_DIR/ -name _pyvsag.cpython*.so | wc -l)
    if [ $PYVSAG_CPYTHON_SO_COUNT -gt 1 ]; then
        echo ""
        echo "Found too many pyvsag cpython libraries in $CMAKE_BUILD_DIR. Please keep one only."
        echo ""
        exit 1
    fi
    PYVSAG_CPYTHON_SO=$(find $CMAKE_BUILD_DIR -name _pyvsag.cpython*.so | head -n1)
    cp $PYVSAG_CPYTHON_SO python/pyvsag
    
   
    for lib in $(get_linked_libraries $PYVSAG_CPYTHON_SO); do
        if [[ $lib == *vsag* ]]; then
            strip $lib
        fi
        
        if [[ $lib == *vsag* ]] || [[ $lib == *gfortran* ]] || [[ $lib == *openblas* ]] || [[ $lib == *blas* ]] || [[ $lib == *lapack* ]]; then
            cp $lib python/pyvsag/
        fi
    done

    # step 2: patch libraries
    if ! command -v patchelf &> /dev/null
    then
        echo ""
        echo "patchelf is not installed. Please install patchelf to continue."
        echo ""
        exit 1
    fi
    find python/pyvsag -type f -name "*.so*" -exec patchelf --set-rpath '$ORIGIN' {} \;
    find python/pyvsag -type f -name "*.so*" -exec sh -c "readelf -d {} | grep RUNPATH" \;

    # step 3: install setup requirements
    cd python/ && $PYTHON_PATH -m pip install -r setup-requirements.txt && cd ../ || exit 1

    # step 4: build wheel with ARM64 tag
    cd python/ && $PYTHON_PATH setup.py bdist_wheel --plat-name linux_aarch64 && cd ../ || exit 1

    # step 5: clean workdir
    cd python/ && $PYTHON_PATH setup.py clean --all && cd ../
    for lib in $(find ./python/pyvsag -name *.so*); do
        rm $lib
    done
}


CURRENT_ARCH=$(uname -m)
debug_echo "Current architecture: $CURRENT_ARCH"

if [[ "$CURRENT_ARCH" != "aarch64" ]] && [[ "$CURRENT_ARCH" != "arm64" ]]; then
    echo "Warning: Current architecture is $CURRENT_ARCH, not ARM64."
    echo "Setting up for cross-compilation..."
    export ARM64_CROSS_COMPILE=true
else
    debug_echo "Running on native ARM64 architecture"
    export ARM64_CROSS_COMPILE=false
fi

CMAKE_ARGS="$1"
CMAKE_BUILD_ARGS="$2"
CMAKE_BUILD_DIR="$3"
debug_echo ">> $CMAKE_ARGS"
debug_echo ">> $CMAKE_BUILD_ARGS"
debug_echo ">> $CMAKE_BUILD_DIR"

check_env_python
for p in "${AVAILABLE_PYTHON_PATHS[@]}"; do
    echo "compiling pyvsag for ARM64 with $p ..."
    BP_ARGS=("$p" "$CMAKE_ARGS" "$CMAKE_BUILD_ARGS" "${CMAKE_BUILD_DIR}")
    build_pyvsag "${BP_ARGS[@]}"
done