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
#include <exception>
#include <filesystem>
#include <optional>
#include <sstream>

#include "autotune_internal.h"
#include "case/eval_case.h"
#include "eval_config.h"
#include "vsag/options.h"

namespace vsag::autotune::internal {

namespace {

std::optional<double>
GetJsonDouble(const JsonType& object, const std::string& key) {
    if (!object.is_object() || !object.contains(key) || !object[key].is_number()) {
        return std::nullopt;
    }
    return object[key].get<double>();
}

std::optional<double>
GetNestedJsonDouble(const JsonType& object,
                    const std::string& first_key,
                    const std::string& second_key) {
    if (!object.is_object() || !object.contains(first_key) || !object[first_key].is_object()) {
        return std::nullopt;
    }
    return GetJsonDouble(object[first_key], second_key);
}

std::optional<double>
ParseMemoryMb(const JsonType& value) {
    if (!value.is_string()) {
        return std::nullopt;
    }
    std::istringstream iss(value.get<std::string>());
    double number = 0.0;
    std::string unit;
    iss >> number >> unit;
    if (unit == "B") {
        return number / 1024.0 / 1024.0;
    }
    if (unit == "KB") {
        return number / 1024.0;
    }
    if (unit == "MB") {
        return number;
    }
    if (unit == "GB") {
        return number * 1024.0;
    }
    if (unit == "TB") {
        return number * 1024.0 * 1024.0;
    }
    return std::nullopt;
}

void
SetMetricIfPresent(JsonType& metrics, const std::string& key, const std::optional<double>& value) {
    if (value.has_value()) {
        metrics[key] = value.value();
    }
}

JsonType
ExtractMetrics(const JsonType& raw_eval_result, const TrialSpec& trial) {
    JsonType metrics = JsonType::object();
    SetMetricIfPresent(metrics, "recall_at_k", GetJsonDouble(raw_eval_result, "recall_avg"));
    SetMetricIfPresent(
        metrics, "latency_avg_ms", GetJsonDouble(raw_eval_result, "latency_avg(ms)"));
    SetMetricIfPresent(metrics,
                       "latency_p99_ms",
                       GetNestedJsonDouble(raw_eval_result, "latency_detail(ms)", "p99"));
    SetMetricIfPresent(metrics, "qps", GetJsonDouble(raw_eval_result, "qps"));
    SetMetricIfPresent(metrics, "build_seconds", GetJsonDouble(raw_eval_result, "duration(s)"));

    std::optional<double> memory_peak_mb;
    if (raw_eval_result.contains("memory_peak(build)")) {
        memory_peak_mb = ParseMemoryMb(raw_eval_result["memory_peak(build)"]);
    }
    if (raw_eval_result.contains("memory_peak(search)")) {
        auto search_memory = ParseMemoryMb(raw_eval_result["memory_peak(search)"]);
        if (search_memory.has_value()) {
            memory_peak_mb = memory_peak_mb.has_value()
                                 ? std::max(memory_peak_mb.value(), search_memory.value())
                                 : search_memory;
        }
    }
    SetMetricIfPresent(metrics, "memory_peak_mb", memory_peak_mb);

    if (!trial.index_path.empty() && std::filesystem::exists(trial.index_path)) {
        const double index_size_mb =
            static_cast<double>(std::filesystem::file_size(trial.index_path)) / 1024.0 / 1024.0;
        metrics["index_size_mb"] = index_size_mb;
    }
    return metrics;
}

JsonType
EvaluateConstraints(const JsonType& constraints, const JsonType& metrics) {
    JsonType result;
    result["satisfied_constraints"] = true;
    result["violated_constraints"] = JsonType::array();

    for (const auto& constraint : constraints.items()) {
        const auto& key = constraint.key();
        const double threshold = constraint.value().get<double>();
        if (!metrics.contains(key) || !metrics[key].is_number()) {
            result["satisfied_constraints"] = false;
            result["violated_constraints"].push_back(
                JsonType{{"name", key}, {"reason", "missing_metric"}});
            continue;
        }

        const double actual = metrics[key].get<double>();
        bool satisfied = false;
        std::string direction;
        if (key == "recall_at_k" || key == "qps") {
            satisfied = actual >= threshold;
            direction = "min";
        } else {
            satisfied = actual <= threshold;
            direction = "max";
        }
        if (!satisfied) {
            result["satisfied_constraints"] = false;
            result["violated_constraints"].push_back(JsonType{{"name", key},
                                                              {"direction", direction},
                                                              {"expected", threshold},
                                                              {"actual", actual}});
        }
    }
    return result;
}

}  // namespace

JsonType
RunTrial(const TrialSpec& trial,
         const JsonType& request,
         const JsonType& constraints,
         const ExecutionOptions& options) {
    JsonType trial_result;
    trial_result["trial_id"] = trial.trial_id;
    trial_result["index_name"] = trial.index_name;
    trial_result["eval_type"] = trial.eval_type;
    trial_result["create_params"] = trial.create_params;
    trial_result["search_params"] = trial.search_params;
    trial_result["artifacts"] = JsonType{{"index_path", trial.index_path}};

    const auto start = Clock::now();
    try {
        eval::EvalConfig config;
        config.dataset_path = request["data_path"].get<std::string>();
        config.action_type = trial.eval_type;
        config.index_name = trial.index_name;
        config.build_param = trial.create_params.dump();
        config.index_path = trial.index_path;
        config.search_param = trial.search_params.dump();
        config.search_mode = options.search_mode;
        config.top_k = options.top_k;
        config.search_query_count = options.search_query_count;
        config.num_threads_building = options.num_threads_building;
        config.num_threads_searching = options.num_threads_searching;
        config.delete_index_after_search = false;

        vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
        vsag::Options::Instance().set_num_threads_building(config.num_threads_building);
        auto eval_case = eval::EvalCase::MakeInstance(config);
        Require(eval_case != nullptr, "failed to create eval case");
        auto raw_eval_result = eval_case->Run();
        auto metrics = ExtractMetrics(raw_eval_result, trial);
        auto constraint_result = EvaluateConstraints(constraints, metrics);

        trial_result["status"] = "success";
        trial_result["metrics"] = metrics;
        trial_result["raw_eval_result"] = raw_eval_result;
        trial_result["satisfied_constraints"] = constraint_result["satisfied_constraints"];
        trial_result["violated_constraints"] = constraint_result["violated_constraints"];
        trial_result["failure"] = nullptr;
    } catch (const std::exception& e) {
        trial_result["status"] = "failed";
        trial_result["metrics"] = JsonType::object();
        trial_result["raw_eval_result"] = nullptr;
        trial_result["satisfied_constraints"] = false;
        trial_result["violated_constraints"] = JsonType::array();
        trial_result["failure"] = e.what();
    }
    trial_result["elapsed_seconds"] = ElapsedSeconds(start);

    if (!options.keep_intermediate && trial.eval_type != "search" &&
        std::filesystem::exists(trial.index_path)) {
        std::error_code error;
        std::filesystem::remove(trial.index_path, error);
    }
    return trial_result;
}

}  // namespace vsag::autotune::internal
