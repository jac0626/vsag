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

#include <filesystem>
#include <set>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

bool
IsSupportedIndex(const std::string& index_name) {
    return index_name == kIndexHGraph || index_name == kIndexIvf;
}

bool
IsSupportedConstraint(const std::string& constraint_name) {
    static const std::set<std::string> supported_constraints = {"recall_at_k",
                                                                "latency_avg_ms",
                                                                "latency_p99_ms",
                                                                "qps",
                                                                "memory_peak_mb",
                                                                "build_seconds",
                                                                "index_size_mb"};
    return supported_constraints.find(constraint_name) != supported_constraints.end();
}

ExecutionOptions
ParseExecutionOptions(const JsonType& request) {
    ExecutionOptions options;
    const JsonType execution =
        request.contains("execution") ? request["execution"] : JsonType::object();
    Require(execution.is_object(), "execution must be an object");

    options.top_k = GetInt(execution, "top_k", options.top_k);
    options.search_mode = GetString(execution, "search_mode", options.search_mode);
    options.search_query_count =
        GetUInt64(execution, "search_query_count", options.search_query_count);
    options.num_threads_building =
        GetInt(execution, "num_threads_building", options.num_threads_building);
    options.num_threads_searching =
        GetInt(execution, "num_threads_searching", options.num_threads_searching);
    options.workspace_path = GetString(execution, "workspace_path", options.workspace_path);
    options.keep_intermediate = GetBool(execution, "keep_intermediate", options.keep_intermediate);
    options.max_trials = GetUInt64(execution, "max_trials", options.max_trials);

    Require(options.top_k > 0, "execution.top_k must be positive");
    Require(options.num_threads_building > 0, "execution.num_threads_building must be positive");
    Require(options.num_threads_searching > 0, "execution.num_threads_searching must be positive");
    Require(options.search_mode == "knn",
            "execution.search_mode is unsupported in AutoTune P0: " + options.search_mode);

    const JsonType output = request.contains("output") ? request["output"] : JsonType::object();
    Require(output.is_object(), "output must be an object");
    options.include_trials = GetBool(output, "include_trials", options.include_trials);
    options.result_path = GetString(output, "result_path", options.result_path);
    return options;
}

void
ValidateRequest(const JsonType& request) {
    Require(request.is_object(), "request must be an object");
    Require(request.contains("version"), "version is required");
    Require(request["version"].is_number_integer(), "version must be an integer");
    Require(request["version"].get<int>() == 1, "only version 1 is supported");

    Require(request.contains("data_path"), "data_path is required");
    Require(request["data_path"].is_string(), "data_path must be a string");
    Require(std::filesystem::exists(request["data_path"].get<std::string>()),
            "data_path does not exist: " + request["data_path"].get<std::string>());

    Require(request.contains("indexes"), "indexes is required");
    Require(request["indexes"].is_array() && !request["indexes"].empty(),
            "indexes must be a non-empty array");
    for (uint64_t i = 0; i < request["indexes"].size(); ++i) {
        const auto& index_spec = request["indexes"][i];
        Require(index_spec.is_object(), "indexes[] must be objects");
        const auto index_name = GetString(index_spec, "name", "");
        Require(IsSupportedIndex(index_name), "unsupported index: " + index_name);
    }

    Require(request.contains("constraints"), "constraints is required");
    Require(request["constraints"].is_object() && !request["constraints"].empty(),
            "constraints must be a non-empty object");
    for (const auto& item : request["constraints"].items()) {
        Require(IsSupportedConstraint(item.key()), "unsupported constraint: " + item.key());
        Require(item.value().is_number(), "constraint " + item.key() + " must be a number");
    }
}

}  // namespace vsag::autotune::internal
