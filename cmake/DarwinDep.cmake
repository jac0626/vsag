
# Copyright 2025-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package (Threads REQUIRED)
set (VSAG_SYSTEM_LIBS Threads::Threads m)
if (CMAKE_DL_LIBS)
    list (APPEND VSAG_SYSTEM_LIBS ${CMAKE_DL_LIBS})
endif ()

if (APPLE)
    set (ld_flags_workaround "-Wl,-rpath,@loader_path")

    find_program (BREW_EXECUTABLE NAMES brew PATHS /opt/homebrew/bin /usr/local/bin)
    foreach (_formula libomp openblas gcc)
        string (TOUPPER "${_formula}" _formula_upper)
        if (BREW_EXECUTABLE)
            execute_process (
                COMMAND ${BREW_EXECUTABLE} --prefix ${_formula}
                OUTPUT_VARIABLE _formula_prefix
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _formula_status
            )
            if (_formula_status EQUAL 0 AND EXISTS "${_formula_prefix}")
                set (VSAG_HOMEBREW_${_formula_upper}_PREFIX "${_formula_prefix}")
            endif ()
        endif ()
    endforeach ()

    if (DEFINED VSAG_HOMEBREW_LIBOMP_PREFIX AND NOT OpenMP_ROOT)
        set (OpenMP_ROOT "${VSAG_HOMEBREW_LIBOMP_PREFIX}")
    endif ()

    # Find OpenMP from Homebrew libomp on macOS.
    find_package (OpenMP REQUIRED COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
        message (STATUS "Found OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
    endif ()

    # Find LAPACK - will automatically use Accelerate framework on macOS
    find_package (LAPACK)
    if (LAPACK_FOUND)
        message (STATUS "Found LAPACK (using Accelerate framework on macOS)")
    else ()
        message (WARNING "LAPACK not found")
    endif ()

    # Find gfortran and its library path for OpenBLAS.
    set (_gfortran_search_paths "")
    if (DEFINED VSAG_HOMEBREW_GCC_PREFIX)
        list (APPEND _gfortran_search_paths
              "${VSAG_HOMEBREW_GCC_PREFIX}/bin"
              "${VSAG_HOMEBREW_GCC_PREFIX}/lib/gcc/current")
    endif ()
    find_program (GFORTRAN_EXECUTABLE NAMES gfortran PATHS ${_gfortran_search_paths})
    if (GFORTRAN_EXECUTABLE)
        execute_process (
            COMMAND ${GFORTRAN_EXECUTABLE} -print-file-name=libgfortran.dylib
            OUTPUT_VARIABLE GFORTRAN_LIB
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (EXISTS "${GFORTRAN_LIB}" AND NOT IS_DIRECTORY "${GFORTRAN_LIB}")
            get_filename_component (GFORTRAN_LIB_DIR "${GFORTRAN_LIB}" DIRECTORY)
            list (APPEND CMAKE_INSTALL_RPATH "${GFORTRAN_LIB_DIR}")
        else ()
            unset (GFORTRAN_LIB)
            message (WARNING
                     "gfortran found but libgfortran.dylib not found via -print-file-name; "
                     "OpenBLAS link may fail")
        endif ()
    else ()
        message (WARNING "gfortran not found; OpenBLAS/LAPACKE features may not link on macOS")
    endif ()

    # Fixup: some scripts (e.g. OpenBLAS fallback) set BLAS_LIBRARIES/LAPACK_LIBRARIES to include
    # a bare `gfortran` token (expands to -lgfortran). On macOS libgfortran is often not in the
    # default linker search paths, causing: `ld: library 'gfortran' not found`.
    function (vsag_darwin_fixup_blas_lapack_libs)
        if (NOT DEFINED GFORTRAN_LIB OR NOT EXISTS "${GFORTRAN_LIB}")
            return ()
        endif ()

        foreach (_var BLAS_LIBRARIES LAPACK_LIBRARIES)
            if (DEFINED ${_var})
                set (_new_list "")
                foreach (_item IN LISTS ${_var})
                    if (_item STREQUAL "gfortran")
                        list (APPEND _new_list "${GFORTRAN_LIB}")
                    else ()
                        list (APPEND _new_list "${_item}")
                    endif ()
                endforeach ()
                # Keep cache in sync so downstream includes/targets see the rewritten list.
                set (${_var} "${_new_list}" CACHE STRING "" FORCE)
            endif ()
        endforeach ()
    endfunction ()

    # Run after the whole top-level configure has defined BLAS/LAPACK variables (order independent).
    cmake_language (DEFER CALL vsag_darwin_fixup_blas_lapack_libs)
else ()
    set (ld_flags_workaround "-Wl,-rpath=\\$\\$ORIGIN")
    find_package (OpenMP REQUIRED COMPONENTS CXX)
endif ()

if (NOT TARGET vsag_openmp)
    add_library (vsag_openmp INTERFACE)
endif ()
target_link_libraries (vsag_openmp INTERFACE OpenMP::OpenMP_CXX)
