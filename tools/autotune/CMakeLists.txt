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
    autotune_common.cpp
    autotune_evaluation.cpp
    autotune_index_policy.cpp
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
    vsag
    nlohmann_json::nlohmann_json)

add_executable (autotune main.cpp)
target_include_directories (autotune PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../eval)
target_link_libraries (autotune PRIVATE
    autotune_obj
    eval_obj
    simd
    vsag
    vsag_eval_common
    nlohmann_json::nlohmann_json
    ${HDF5_CPP_STATIC_LIBRARY}
    ${HDF5_C_STATIC_LIBRARY}
    z dl)
add_dependencies (autotune hdf5)

if (ENABLE_TESTS)
    add_executable (autotune_test
        autotune_test.cpp)
    target_include_directories (autotune_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../eval)
    target_link_libraries (autotune_test PRIVATE
        Catch2::Catch2WithMain
        autotune_obj
        eval_obj
        simd
        vsag
        vsag_eval_common
        nlohmann_json::nlohmann_json
        ${HDF5_CPP_STATIC_LIBRARY}
        ${HDF5_C_STATIC_LIBRARY}
        z dl)
    add_dependencies (autotune_test hdf5 Catch2)
endif ()
