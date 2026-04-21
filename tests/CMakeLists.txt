
# Copyright 2024-present the vsag project
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

# fixtures is already added by top-level CMakeLists.txt before src

set (VSAG_UNITTEST_LIBS
        vsag_test algorithm_test factory_test attr_test
        datacell_test quantizer_test storage_test io_test utils_test impl_test)

set (VSAG_UNITTEST_FORCE_LOAD_LIBS ${VSAG_UNITTEST_LIBS})

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x86_64h)$")
        list (PREPEND VSAG_UNITTEST_LIBS simd_test)
        list (PREPEND VSAG_UNITTEST_FORCE_LOAD_LIBS simd_test)
else ()
        message (STATUS "Skipping simd_test linkage in unittests on ${CMAKE_SYSTEM_PROCESSOR}")
endif ()


#  unittests
add_executable (unittests
        test_main.cpp
        $<TARGET_OBJECTS:fixtures_allocator>
        $<TARGET_OBJECTS:fixtures_core>
        $<TARGET_OBJECTS:fixtures_data>
        $<TARGET_OBJECTS:fixtures_framework>
)

target_link_libraries (unittests
        PRIVATE
        Catch2::Catch2 
        ${VSAG_UNITTEST_LIBS}
        vsag_static
)

if (APPLE)
        # Ensure static test libraries are not dead-stripped on macOS.
        target_link_options (unittests PRIVATE
        "-Wl,-force_load,$<TARGET_FILE:vsag_test>"
        "-Wl,-force_load,$<TARGET_FILE:algorithm_test>"
        "-Wl,-force_load,$<TARGET_FILE:factory_test>"
        "-Wl,-force_load,$<TARGET_FILE:attr_test>"
        "-Wl,-force_load,$<TARGET_FILE:datacell_test>"
        "-Wl,-force_load,$<TARGET_FILE:quantizer_test>"
        "-Wl,-force_load,$<TARGET_FILE:storage_test>"
        "-Wl,-force_load,$<TARGET_FILE:io_test>"
        "-Wl,-force_load,$<TARGET_FILE:utils_test>"
        "-Wl,-force_load,$<TARGET_FILE:impl_test>"
        )
        if (TARGET simd_test AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x86_64h)$")
                target_link_options (unittests PRIVATE "-Wl,-force_load,$<TARGET_FILE:simd_test>")
        endif ()
else()
        target_link_libraries (unittests
                        PRIVATE
                        "-Wl,--whole-archive"
                        ${VSAG_UNITTEST_LIBS}
                        "-Wl,--no-whole-archive"
        )
endif()

add_dependencies (unittests
        Catch2
        vsag_test
        algorithm_test
        factory_test
        attr_test
        datacell_test
        quantizer_test
        storage_test
        io_test
        utils_test
        impl_test
        vsag)

if (TARGET simd_test AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|x86_64h)$")
        add_dependencies (unittests simd_test)
endif ()

# function tests
file (GLOB FUNCTION_TESTS "test_*.cpp")
add_executable (functests
        ${FUNCTION_TESTS}
        $<TARGET_OBJECTS:fixtures_allocator>
        $<TARGET_OBJECTS:fixtures_core>
        $<TARGET_OBJECTS:fixtures_data>
        $<TARGET_OBJECTS:fixtures_framework>
)

target_link_libraries (functests PRIVATE
        Catch2::Catch2 
        vsag 
        simd
)
add_dependencies (functests Catch2)
