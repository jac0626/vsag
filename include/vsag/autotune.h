// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>

namespace vsag::autotune {

/**
 * @brief Run an AutoTune V1 request.
 *
 * The request and returned compact full report are UTF-8 JSON documents. This API is available
 * from the optional vsag::autotune CMake target when VSAG is built with ENABLE_TOOLS=ON and
 * ENABLE_CXX11_ABI=ON. V1 exposes the target only in the build tree, not in the installed package.
 *
 * @param request_json AutoTune V1 request encoded as JSON.
 * @return Compact full report encoded as JSON. Validation failures are represented by a report
 *         whose status is "failed".
 */
std::string
RunAutoTune(const std::string& request_json);

}  // namespace vsag::autotune
