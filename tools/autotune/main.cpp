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

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "autotune.h"

namespace {

vsag::autotune::JsonType
LoadJsonFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("failed to open request file: " + path);
    }
    vsag::autotune::JsonType request;
    in >> request;
    return request;
}

void
PrintUsage(const char* program) {
    std::cerr << "Usage: " << program << " <request.json>" << std::endl;
}

vsag::autotune::JsonType
MakeCliFailureResult(const std::string& message) {
    vsag::autotune::JsonType result;
    result["version"] = 1;
    result["status"] = "failed";
    result["elapsed_seconds"] = 0.0;
    result["elapsed_breakdown_seconds"] = vsag::autotune::JsonType::object();
    result["recommendation"] = nullptr;
    result["best_effort"] = nullptr;
    result["trial_count"] = 0;
    result["failure"] = {{"message", message}};
    return result;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 2) {
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        auto request = LoadJsonFile(argv[1]);
        auto result = vsag::autotune::RunAutoTune(request);
        std::cout << result.dump(2) << std::endl;
        if (result.contains("status") && result["status"] == "failed") {
            return 1;
        }
    } catch (const std::exception& e) {
        auto result = MakeCliFailureResult(e.what());
        std::cout << result.dump(2) << std::endl;
        return 1;
    }
    return 0;
}
