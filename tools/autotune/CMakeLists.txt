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

set (autotune_srcs
    autotune.cpp
    autotune_candidate.cpp
    autotune_candidate_generator.cpp
    autotune_common.cpp
    autotune_constraints.cpp
    autotune_environment.cpp
    autotune_evaluation.cpp
    autotune_planner.cpp
    autotune_request.cpp
    autotune_result.cpp)

add_library (autotune_obj OBJECT ${autotune_srcs})
target_include_directories (autotune_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../eval)
target_link_libraries (autotune_obj PRIVATE
    eval_obj
    vsag_eval_common
    vsag_static
    nlohmann_json::nlohmann_json)

add_library (vsag_autotune SHARED
    $<TARGET_OBJECTS:autotune_obj>
    $<TARGET_OBJECTS:eval_obj>)
add_library (vsag::autotune ALIAS vsag_autotune)
set_target_properties (vsag_autotune PROPERTIES
    C_VISIBILITY_PRESET default
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF)
if (ENABLE_SOVERSION)
    set_target_properties (vsag_autotune PROPERTIES
        VERSION "${VSAG_VERSION_TRIPLE}"
        SOVERSION "${VSAG_VERSION_MAJOR}")
endif ()
target_include_directories (vsag_autotune PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries (vsag_autotune PRIVATE
    simd
    vsag
    vsag_eval_common
    nlohmann_json::nlohmann_json
    ${HDF5_CPP_STATIC_LIBRARY}
    ${HDF5_C_STATIC_LIBRARY}
    z
    ${CMAKE_DL_LIBS})
add_dependencies (vsag_autotune hdf5)

add_executable (autotune main.cpp)
target_include_directories (autotune PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../eval)
target_link_libraries (autotune PRIVATE
    vsag_autotune
    vsag
    nlohmann_json::nlohmann_json)

if (ENABLE_TESTS)
    add_executable (autotune_test
        autotune_test.cpp)
    target_include_directories (autotune_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../eval)
    target_link_libraries (autotune_test PRIVATE
        Catch2::Catch2WithMain
        vsag_autotune
        vsag
        vsag_eval_common
        nlohmann_json::nlohmann_json
        ${HDF5_CPP_STATIC_LIBRARY}
        ${HDF5_C_STATIC_LIBRARY}
        z
        ${CMAKE_DL_LIBS})
    add_dependencies (autotune_test hdf5 Catch2)

    add_executable (autotune_public_api_test
        autotune_public_api_test.cpp)
    target_link_libraries (autotune_public_api_test PRIVATE
        Catch2::Catch2WithMain
        vsag::autotune)
    add_dependencies (autotune_public_api_test Catch2)
endif ()
