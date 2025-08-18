message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "ENABLE_INTEL_MKL: ${ENABLE_INTEL_MKL}")

if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" AND ENABLE_INTEL_MKL)

    # --- 1. 查找 MKL 根目录 (更稳健的方式) ---
    find_path(MKL_ROOT_DIR NAMES include/mkl.h
        HINTS
            ENV MKLROOT
            /opt/intel/oneapi/mkl/latest
            /opt/intel/mkl
            /usr
    )
    if(NOT MKL_ROOT_DIR)
        message(FATAL_ERROR "Could not find Intel MKL. Set MKLROOT environment variable.")
    endif()

    set(MKL_PATH "${MKL_ROOT_DIR}/lib/intel64") # 假设为64位
    if(NOT EXISTS "${MKL_PATH}")
        set(MKL_PATH "${MKL_ROOT_DIR}/lib") # 备用路径
    endif()
    set(MKL_INCLUDE_PATH "${MKL_ROOT_DIR}/include")


    # --- 2. 判断使用 GNU 还是 Intel OpenMP ---
    # 由于编译器是GCC, 优先并默认使用 GNU OpenMP
    set(USE_GNU_OPENMP ON)
    message(STATUS "Defaulting to GNU OpenMP for GCC compatibility.")

    
    # --- 3. 设置链接和包含目录 (使用现代 CMake 方式会更好，但这里保持原样) ---
    if(EXISTS ${MKL_PATH})
        link_directories(${MKL_PATH})
    else()
        message(FATAL_ERROR "MKL library path not found: ${MKL_PATH}")
    endif()
    
    if(EXISTS ${MKL_INCLUDE_PATH})
        include_directories(${MKL_INCLUDE_PATH})
    else()
        message(WARNING "MKL include path not found: ${MKL_INCLUDE_PATH}")
    endif()

    # --- 4. 设置 BLAS 库 - 优先使用静态库 (已修正) ---
    if(EXISTS "${MKL_PATH}/libmkl_intel_lp64.a" AND 
       EXISTS "${MKL_PATH}/libmkl_core.a")
        
        message(STATUS "Found MKL static libraries, configuring for static linking.")
        
        if(USE_GNU_OPENMP)
            # 使用 GNU OpenMP
            if(EXISTS "${MKL_PATH}/libmkl_gnu_thread.a")
                set(BLAS_LIBRARIES
                    # 使用链接器组解决 MKL 内部循环依赖
                    -Wl,--start-group
                    "${MKL_PATH}/libmkl_intel_lp64.a"
                    "${MKL_PATH}/libmkl_gnu_thread.a"
                    "${MKL_PATH}/libmkl_core.a"
                    -Wl,--end-group
                    # 系统库放在组外
                    gomp
                    pthread
                    m
                    dl
                )
                message(STATUS "Using MKL with static linking and GNU OpenMP.")
            else()
                # 回退到 sequential
                set(BLAS_LIBRARIES
                    -Wl,--start-group
                    "${MKL_PATH}/libmkl_intel_lp64.a"
                    "${MKL_PATH}/libmkl_sequential.a"
                    "${MKL_PATH}/libmkl_core.a"
                    -Wl,--end-group
                    pthread
                    m
                    dl
                )
                message(STATUS "Using MKL with static linking (sequential).")
            endif()
        else()
            # 使用 Intel OpenMP (逻辑保留，但默认不进入)
            if(EXISTS "${MKL_PATH}/libmkl_intel_thread.a")
                set(BLAS_LIBRARIES
                    -Wl,--start-group
                    "${MKL_PATH}/libmkl_intel_lp64.a"
                    "${MKL_PATH}/libmkl_intel_thread.a"
                    "${MKL_PATH}/libmkl_core.a"
                    -Wl,--end-group
                    iomp5 # 确保 iomp5 库能被链接器找到
                    pthread
                    m
                    dl
                )
                message(STATUS "Using MKL with static linking and Intel OpenMP.")
            else()
                # 回退到 sequential
                set(BLAS_LIBRARIES
                    -Wl,--start-group
                    "${MKL_PATH}/libmkl_intel_lp64.a"
                    "${MKL_PATH}/libmkl_sequential.a"
                    "${MKL_PATH}/libmkl_core.a"
                    -Wl,--end-group
                    pthread
                    m
                    dl
                )
                message(STATUS "Using MKL with static linking (sequential).")
            endif()
        endif()
        # 将链接器选项附加到库列表，而不是编译选项
        list(APPEND BLAS_LIBRARIES "-Wl,--no-as-needed")
    else()
        # 动态链接版本（如果没有静态库）
        message(WARNING "MKL static libraries not found, using dynamic linking.")
        set(BLAS_LIBRARIES
            mkl_intel_lp64
            mkl_gnu_thread
            mkl_core
            gomp
            pthread
            m
            dl
        )
    endif()

    # --- 5. 清理编译选项 ---
    add_compile_options(-m64) # 保留 -m64
    # 移除了错误的 -Wl,--no-as-needed 和 -DMKL_ILP64
    
    message(STATUS "MKL_PATH: ${MKL_PATH}")
    message(STATUS "MKL_INCLUDE_PATH: ${MKL_INCLUDE_PATH}")
    message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")

else()
    # 回退到 OpenBLAS (保持不变)
    set(BLAS_LIBRARIES libopenblas.a gfortran)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(PREPEND BLAS_LIBRARIES omp)
    else()
        list(PREPEND BLAS_LIBRARIES gomp)
    endif()
    message(STATUS "Using OpenBLAS as BLAS backend")
endif()
