message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "ENABLE_INTEL_MKL: ${ENABLE_INTEL_MKL}")

if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" AND ENABLE_INTEL_MKL)
    # 搜索 Intel OpenMP 库
    set(POSSIBLE_OMP_PATHS
        "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so"
        "/usr/lib/x86_64-linux-gnu/libiomp5.so"
        "/opt/intel/lib/intel64_lin/libiomp5.so"
        "/opt/intel/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin/libiomp5.so"
    )

    foreach(POSSIBLE_OMP_PATH ${POSSIBLE_OMP_PATHS})
        if (EXISTS ${POSSIBLE_OMP_PATH})
            get_filename_component(OMP_PATH ${POSSIBLE_OMP_PATH} DIRECTORY)
            break()
        endif()
    endforeach()

    # 搜索 MKL 库路径
    set(POSSIBLE_MKL_LIB_PATHS 
        "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so"
        "/usr/lib/x86_64-linux-gnu/libmkl_core.so"
        "/opt/intel/mkl/lib/intel64/libmkl_core.so"
    )

    foreach(POSSIBLE_MKL_LIB_PATH ${POSSIBLE_MKL_LIB_PATHS})
        if (EXISTS ${POSSIBLE_MKL_LIB_PATH})
            get_filename_component(MKL_PATH ${POSSIBLE_MKL_LIB_PATH} DIRECTORY)
            break()
        endif()
    endforeach()

    # 搜索 MKL 头文件路径
    set(POSSIBLE_MKL_INCLUDE_PATHS 
        "/opt/intel/oneapi/mkl/latest/include"
        "/usr/include/mkl"
        "/opt/intel/mkl/include"
        "/usr/include"
    )

    foreach(POSSIBLE_MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATHS})
        if (EXISTS ${POSSIBLE_MKL_INCLUDE_PATH}/mkl.h OR EXISTS ${POSSIBLE_MKL_INCLUDE_PATH}/mkl/mkl.h)
            set(MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATH})
            break()
        endif()
    endforeach()

    # 如果没有找到 Intel OpenMP，尝试使用 GNU OpenMP
    if(NOT OMP_PATH)
        message(WARNING "Could not find Intel OpenMP, trying to use GNU OpenMP instead")
        set(USE_GNU_OPENMP ON)
    endif()

    if(NOT MKL_PATH)
        message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_PATH to specify the install location")
    endif()

    if(NOT MKL_INCLUDE_PATH)
        message(WARNING "Could not find MKL headers, continuing without include path")
    endif()

    # 查找 libmkl_def.so
    if (EXISTS ${MKL_PATH}/libmkl_def.so.2)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so.2)
    elseif(EXISTS ${MKL_PATH}/libmkl_def.so)
        set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so)
    else()
        message(WARNING "libmkl_def.so not found, continuing without it")
    endif()

    # 设置链接目录和包含目录
    link_directories(${MKL_PATH})
    if(OMP_PATH)
        link_directories(${OMP_PATH})
    endif()
    if(MKL_INCLUDE_PATH)
        include_directories(${MKL_INCLUDE_PATH})
    endif()

    # 设置 BLAS 库 - 修复静态链接问题
    if(EXISTS ${MKL_PATH}/libmkl_intel_lp64.a AND 
       EXISTS ${MKL_PATH}/libmkl_core.a)
        # 静态链接版本 - 使用 -Wl,--start-group 和 -Wl,--end-group 来解决循环依赖
        if(USE_GNU_OPENMP)
            # 使用 GNU OpenMP
            if(EXISTS ${MKL_PATH}/libmkl_gnu_thread.a)
                set(BLAS_LIBRARIES
                    -Wl,--start-group
                    ${MKL_PATH}/libmkl_intel_lp64.a
                    ${MKL_PATH}/libmkl_gnu_thread.a
                    ${MKL_PATH}/libmkl_core.a
                    -Wl,--end-group
                    gomp
                    pthread
                    m
                    dl
                )
                message(STATUS "Using MKL with static linking and GNU OpenMP")
            else()
                # 如果没有 gnu_thread 的静态库，使用 sequential
                set(BLAS_LIBRARIES
                    -Wl,--start-group
                    ${MKL_PATH}/libmkl_intel_lp64.a
                    ${MKL_PATH}/libmkl_sequential.a
                    ${MKL_PATH}/libmkl_core.a
                    -Wl,--end-group
                    pthread
                    m
                    dl
                )
                message(STATUS "Using MKL with static linking (sequential)")
            endif()
        else()
            # 使用 Intel OpenMP
            if(EXISTS ${MKL_PATH}/libmkl_intel_thread.a)
                set(BLAS_LIBRARIES
                    -Wl,--start-group
                    ${MKL_PATH}/libmkl_intel_lp64.a
                    ${MKL_PATH}/libmkl_intel_thread.a
                    ${MKL_PATH}/libmkl_core.a
                    -Wl,--end-group
                    iomp5
                    pthread
                    m
                    dl
                )
                message(STATUS "Using MKL with static linking and Intel OpenMP")
            else()
                # 回退到 sequential
                set(BLAS_LIBRARIES
                    -Wl,--start-group
                    ${MKL_PATH}/libmkl_intel_lp64.a
                    ${MKL_PATH}/libmkl_sequential.a
                    ${MKL_PATH}/libmkl_core.a
                    -Wl,--end-group
                    pthread
                    m
                    dl
                )
                message(STATUS "Using MKL with static linking (sequential)")
            endif()
        endif()
        
        # 添加额外的链接标志以确保所有符号都被包含
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
        
    else()
        # 动态链接版本（推荐，避免静态链接问题）
        message(WARNING "MKL static libraries not found or switching to dynamic linking to avoid link errors")
        
        # 检查动态库是否存在
        if(EXISTS ${MKL_PATH}/libmkl_intel_lp64.so AND
           EXISTS ${MKL_PATH}/libmkl_core.so)
            
            if(USE_GNU_OPENMP)
                if(EXISTS ${MKL_PATH}/libmkl_gnu_thread.so)
                    set(BLAS_LIBRARIES
                        mkl_intel_lp64
                        mkl_gnu_thread
                        mkl_core
                        gomp
                        pthread
                        m
                        dl
                    )
                    message(STATUS "Using MKL with dynamic linking and GNU OpenMP")
                else()
                    set(BLAS_LIBRARIES
                        mkl_intel_lp64
                        mkl_sequential
                        mkl_core
                        pthread
                        m
                        dl
                    )
                    message(STATUS "Using MKL with dynamic linking (sequential)")
                endif()
            else()
                if(EXISTS ${MKL_PATH}/libmkl_intel_thread.so)
                    set(BLAS_LIBRARIES
                        mkl_intel_lp64
                        mkl_intel_thread
                        mkl_core
                        iomp5
                        pthread
                        m
                        dl
                    )
                    message(STATUS "Using MKL with dynamic linking and Intel OpenMP")
                else()
                    set(BLAS_LIBRARIES
                        mkl_intel_lp64
                        mkl_sequential
                        mkl_core
                        pthread
                        m
                        dl
                    )
                    message(STATUS "Using MKL with dynamic linking (sequential)")
                endif()
            endif()
        else()
            message(FATAL_ERROR "Neither static nor dynamic MKL libraries found!")
        endif()
    endif()

    # 添加编译选项
    add_compile_options(-m64)
    
    # 根据是否使用 ILP64 接口添加相应的定义
    # 注意：如果你的代码使用的是 LP64 接口（默认），请注释掉下面这行
    # add_definitions(-DMKL_ILP64)
    
    message(STATUS "MKL_PATH: ${MKL_PATH}")
    message(STATUS "MKL_INCLUDE_PATH: ${MKL_INCLUDE_PATH}")
    message(STATUS "OMP_PATH: ${OMP_PATH}")
    message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")

else()
    # 回退到 OpenBLAS
    set(BLAS_LIBRARIES libopenblas.a gfortran)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(PREPEND BLAS_LIBRARIES omp)
    else()
        list(PREPEND BLAS_LIBRARIES gomp)
    endif()
    message(STATUS "Using OpenBLAS as BLAS backend")
endif()
