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

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "autotune.h"
#include "autotune_internal.h"
#include "vsag/options.h"

namespace {

vsag::autotune::JsonType
load_json_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("failed to open request file: " + path);
    }
    vsag::autotune::JsonType request;
    in >> request;
    return request;
}

void
print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <request.json>" << std::endl;
}

vsag::autotune::JsonType
make_cli_failure_result(const std::string& message) {
    return vsag::autotune::JsonType{
        {"version", 1},
        {"status", "failed"},
        {"elapsed_seconds", 0.0},
        {"failure",
         vsag::autotune::JsonType{
             {"stage", "cli"}, {"code", "request_file_error"}, {"message", message}}}};
}

void
print_cli_failure(const std::string& message) noexcept {
    try {
        std::cout << vsag::autotune::internal::FormatResultSummaryForCli(
                         make_cli_failure_result(message))
                  << std::endl;
    } catch (...) {
        std::fputs(
            "{\"failure\":{\"stage\":\"cli\","
            "\"code\":\"request_file_error\",\"message\":\"failed to encode CLI error\"},"
            "\"status\":\"failed\",\"elapsed_seconds\":0.0,\"version\":1}\n",
            stdout);
    }
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
        if (argc != 2) {
            print_usage(argv[0]);
            print_cli_failure("expected exactly one request.json argument");
            return 1;
        }

        auto request = load_json_file(argv[1]);
        if (request.contains("output") && request["output"].is_object() &&
            request["output"].contains("result_path") &&
            request["output"]["result_path"].is_string()) {
            const auto result_path = request["output"]["result_path"].get<std::string>();
            if (!result_path.empty() &&
                vsag::autotune::internal::PathsAlias(argv[1], result_path)) {
                throw std::runtime_error("output.result_path must not alias the request file");
            }
        }
        auto result = vsag::autotune::RunAutoTune(request);
        std::cout << vsag::autotune::internal::FormatResultSummaryForCli(result) << std::endl;
        if (result.contains("status") && result["status"] == "failed") {
            return 1;
        }
    } catch (const std::exception& error) {
        print_cli_failure(error.what());
        return 1;
    } catch (...) {
        print_cli_failure("unknown CLI failure");
        return 1;
    }
    return 0;
}
