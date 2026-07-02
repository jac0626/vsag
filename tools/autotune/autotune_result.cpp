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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <tuple>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

double
MetricOrDefault(const JsonType& trial, const std::string& key, double default_value) {
    if (!trial.contains("metrics") || !trial["metrics"].contains(key) ||
        !trial["metrics"][key].is_number()) {
        return default_value;
    }
    return trial["metrics"][key].get<double>();
}

bool
IsSuccessfulTrial(const JsonType& trial) {
    return trial.contains("status") && trial["status"] == "success";
}

bool
IsSatisfiedTrial(const JsonType& trial) {
    return IsSuccessfulTrial(trial) && trial.contains("satisfied_constraints") &&
           trial["satisfied_constraints"].is_boolean() &&
           trial["satisfied_constraints"].get<bool>();
}

bool
RecommendationLess(const JsonType& left, const JsonType& right) {
    const double inf = std::numeric_limits<double>::infinity();
    const auto left_tuple = std::make_tuple(MetricOrDefault(left, "latency_avg_ms", inf),
                                            MetricOrDefault(left, "memory_peak_mb", inf),
                                            MetricOrDefault(left, "build_seconds", inf),
                                            left["trial_id"].get<std::string>());
    const auto right_tuple = std::make_tuple(MetricOrDefault(right, "latency_avg_ms", inf),
                                             MetricOrDefault(right, "memory_peak_mb", inf),
                                             MetricOrDefault(right, "build_seconds", inf),
                                             right["trial_id"].get<std::string>());
    return left_tuple < right_tuple;
}

bool
BestEffortLess(const JsonType& left, const JsonType& right) {
    const double inf = std::numeric_limits<double>::infinity();
    const auto left_tuple = std::make_tuple(-MetricOrDefault(left, "recall_at_k", -inf),
                                            MetricOrDefault(left, "latency_avg_ms", inf),
                                            left["trial_id"].get<std::string>());
    const auto right_tuple = std::make_tuple(-MetricOrDefault(right, "recall_at_k", -inf),
                                             MetricOrDefault(right, "latency_avg_ms", inf),
                                             right["trial_id"].get<std::string>());
    return left_tuple < right_tuple;
}

JsonType
MakeRecommendation(const JsonType& trial, const std::string& reason) {
    JsonType recommendation = JsonType{{"trial_id", trial["trial_id"]},
                                       {"index_name", trial["index_name"]},
                                       {"create_params", trial["create_params"]},
                                       {"search_params", trial["search_params"]},
                                       {"metrics", trial["metrics"]},
                                       {"selection_reason", reason}};
    if (trial.contains("evaluation_stage")) {
        recommendation["evaluation_stage"] = trial["evaluation_stage"];
    }
    return recommendation;
}

}  // namespace

JsonType
SelectResult(const std::vector<JsonType>& trials) {
    std::vector<JsonType> satisfied_trials;
    std::vector<JsonType> successful_trials;
    for (const auto& trial : trials) {
        if (IsSuccessfulTrial(trial)) {
            successful_trials.emplace_back(trial);
        }
        if (IsSatisfiedTrial(trial)) {
            satisfied_trials.emplace_back(trial);
        }
    }

    JsonType selection;
    if (!satisfied_trials.empty()) {
        auto best =
            std::min_element(satisfied_trials.begin(), satisfied_trials.end(), RecommendationLess);
        selection["status"] = "success";
        selection["recommendation"] =
            MakeRecommendation(*best, "satisfied constraints and had the lowest latency_avg_ms");
        selection["best_effort"] = nullptr;
        selection["failure"] = nullptr;
        return selection;
    }

    if (!successful_trials.empty()) {
        auto best_effort =
            std::min_element(successful_trials.begin(), successful_trials.end(), BestEffortLess);
        selection["status"] = "no_candidate_satisfied";
        selection["recommendation"] = nullptr;
        selection["best_effort"] =
            MakeRecommendation(*best_effort, "no candidate satisfied all constraints");
        selection["failure"] = nullptr;
        return selection;
    }

    selection["status"] = "failed";
    selection["recommendation"] = nullptr;
    selection["best_effort"] = nullptr;
    selection["failure"] = JsonType{{"message", "all trials failed"}};
    return selection;
}

void
WriteJsonFile(const std::string& path, const JsonType& json) {
    if (path.empty()) {
        return;
    }
    auto output_path = std::filesystem::path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream out(path);
    Require(out.good(), "failed to open result_path: " + path);
    out << json.dump(2) << std::endl;
}

std::string
GetOptionalResultPath(const JsonType& request) {
    if (!request.is_object() || !request.contains("output") || !request["output"].is_object() ||
        !request["output"].contains("result_path") ||
        !request["output"]["result_path"].is_string()) {
        return "";
    }
    return request["output"]["result_path"].get<std::string>();
}

JsonType
MakeFailedResult(const JsonType& request,
                 const std::string& message,
                 const Clock::time_point& total_start) {
    JsonType result;
    result["version"] = 1;
    if (request.is_object() && request.contains("version") &&
        request["version"].is_number_integer()) {
        result["version"] = request["version"].get<int>();
    }
    result["status"] = "failed";
    result["elapsed_seconds"] = ElapsedSeconds(total_start);
    result["elapsed_breakdown_seconds"] = JsonType::object();
    result["recommendation"] = nullptr;
    result["best_effort"] = nullptr;
    result["trial_count"] = 0;
    result["build_count"] = 0;
    result["build_group_count"] = 0;
    result["failure"] = JsonType{{"message", message}};
    return result;
}

}  // namespace vsag::autotune::internal
