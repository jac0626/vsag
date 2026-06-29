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

void
SetIndexSizeIfPresent(JsonType& metrics, const std::string& index_path) {
    if (!index_path.empty() && std::filesystem::exists(index_path)) {
        const double index_size_mb =
            static_cast<double>(std::filesystem::file_size(index_path)) / 1024.0 / 1024.0;
        metrics["index_size_mb"] = index_size_mb;
    }
}

JsonType
ExtractBuildMetrics(const JsonType& raw_eval_result, const std::string& index_path) {
    JsonType metrics = JsonType::object();
    SetMetricIfPresent(metrics, "build_seconds", GetJsonDouble(raw_eval_result, "duration(s)"));
    if (raw_eval_result.contains("memory_peak(build)")) {
        SetMetricIfPresent(
            metrics, "memory_peak_mb", ParseMemoryMb(raw_eval_result["memory_peak(build)"]));
    }
    SetIndexSizeIfPresent(metrics, index_path);
    return metrics;
}

JsonType
ExtractSearchMetrics(const JsonType& raw_eval_result) {
    JsonType metrics = JsonType::object();
    SetMetricIfPresent(metrics, "recall_at_k", GetJsonDouble(raw_eval_result, "recall_avg"));
    SetMetricIfPresent(
        metrics, "latency_avg_ms", GetJsonDouble(raw_eval_result, "latency_avg(ms)"));
    SetMetricIfPresent(metrics,
                       "latency_p99_ms",
                       GetNestedJsonDouble(raw_eval_result, "latency_detail(ms)", "p99"));
    SetMetricIfPresent(metrics, "qps", GetJsonDouble(raw_eval_result, "qps"));
    SetMetricIfPresent(metrics, "search_seconds", GetJsonDouble(raw_eval_result, "duration(s)"));
    if (raw_eval_result.contains("memory_peak(search)")) {
        SetMetricIfPresent(
            metrics, "memory_peak_mb", ParseMemoryMb(raw_eval_result["memory_peak(search)"]));
    }
    return metrics;
}

JsonType
MergeMetrics(const JsonType& build_metrics, const JsonType& search_metrics) {
    JsonType metrics = build_metrics.is_object() ? build_metrics : JsonType::object();
    for (const auto& item : search_metrics.items()) {
        if (item.key() == "memory_peak_mb" && metrics.contains("memory_peak_mb") &&
            metrics["memory_peak_mb"].is_number() && item.value().is_number()) {
            metrics["memory_peak_mb"] =
                std::max(metrics["memory_peak_mb"].get<double>(), item.value().get<double>());
            continue;
        }
        metrics[item.key()] = item.value();
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

eval::EvalConfig
MakeEvalConfig(const std::string& action_type,
               const std::string& index_name,
               const JsonType& create_params,
               const std::string& index_path,
               const JsonType& search_params,
               const JsonType& request,
               const ExecutionOptions& options) {
    eval::EvalConfig config;
    config.dataset_path = request["data_path"].get<std::string>();
    config.action_type = action_type;
    config.index_name = index_name;
    config.build_param = create_params.dump();
    config.index_path = index_path;
    config.search_param = search_params.dump();
    config.search_mode = options.search_mode;
    config.top_k = options.top_k;
    config.search_query_count = options.search_query_count;
    config.num_threads_building = options.num_threads_building;
    config.num_threads_searching = options.num_threads_searching;
    config.delete_index_after_search = false;
    return config;
}

JsonType
BuildMetricsFromResult(const JsonType& build_result) {
    if (!build_result.is_object() || !build_result.contains("metrics") ||
        !build_result["metrics"].is_object()) {
        return JsonType::object();
    }
    return build_result["metrics"];
}

bool
IsSuccessfulBuild(const JsonType& build_result) {
    return build_result.is_object() && build_result.contains("status") &&
           build_result["status"] == "success";
}

}  // namespace

JsonType
RunBuild(const BuildSpec& build, const JsonType& request, const ExecutionOptions& options) {
    JsonType build_result;
    build_result["build_id"] = build.build_id;
    build_result["index_name"] = build.index_name;
    build_result["eval_type"] = build.use_existing_index ? "existing_index" : "build";
    build_result["create_params"] = build.create_params;
    build_result["artifacts"] =
        JsonType{{"index_path", build.index_path},
                 {"use_existing_index", build.use_existing_index},
                 {"cleanup_index_after_build_group", build.cleanup_index_after_build_group}};

    const auto start = Clock::now();
    try {
        if (build.use_existing_index) {
            JsonType metrics = JsonType::object();
            SetIndexSizeIfPresent(metrics, build.index_path);
            build_result["status"] = "success";
            build_result["metrics"] = metrics;
            build_result["raw_eval_result"] = nullptr;
            build_result["failure"] = nullptr;
        } else {
            auto config = MakeEvalConfig("build",
                                         build.index_name,
                                         build.create_params,
                                         build.index_path,
                                         JsonType::object(),
                                         request,
                                         options);

            vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
            vsag::Options::Instance().set_num_threads_building(config.num_threads_building);
            auto eval_case = eval::EvalCase::MakeInstance(config);
            Require(eval_case != nullptr, "failed to create build eval case");
            auto raw_eval_result = eval_case->Run();

            build_result["status"] = "success";
            build_result["metrics"] = ExtractBuildMetrics(raw_eval_result, build.index_path);
            build_result["raw_eval_result"] = raw_eval_result;
            build_result["failure"] = nullptr;
        }
    } catch (const std::exception& e) {
        build_result["status"] = "failed";
        build_result["metrics"] = JsonType::object();
        build_result["raw_eval_result"] = nullptr;
        build_result["failure"] = e.what();
    }
    build_result["elapsed_seconds"] = ElapsedSeconds(start);
    return build_result;
}

JsonType
RunSearchTrial(const TrialSpec& trial,
               const JsonType& build_result,
               const JsonType& request,
               const JsonType& constraints,
               const ExecutionOptions& options) {
    JsonType trial_result;
    trial_result["trial_id"] = trial.trial_id;
    trial_result["build_id"] = trial.build_id;
    trial_result["index_name"] = trial.index_name;
    trial_result["eval_type"] = trial.eval_type;
    trial_result["create_params"] = trial.create_params;
    trial_result["search_params"] = trial.search_params;
    trial_result["artifacts"] = JsonType{{"index_path", trial.index_path}};

    const auto start = Clock::now();
    try {
        const auto build_metrics = BuildMetricsFromResult(build_result);
        if (!IsSuccessfulBuild(build_result)) {
            trial_result["status"] = "failed";
            trial_result["metrics"] = build_metrics;
            trial_result["raw_eval_result"] =
                JsonType{{"build", build_result.value("raw_eval_result", JsonType(nullptr))},
                         {"search", nullptr}};
            trial_result["satisfied_constraints"] = false;
            trial_result["violated_constraints"] = JsonType::array();
            trial_result["failure"] =
                "build failed: " + build_result.value("failure", std::string("unknown error"));
            trial_result["elapsed_seconds"] = ElapsedSeconds(start);
            return trial_result;
        }

        auto config = MakeEvalConfig("search",
                                     trial.index_name,
                                     trial.create_params,
                                     trial.index_path,
                                     trial.search_params,
                                     request,
                                     options);
        vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
        vsag::Options::Instance().set_num_threads_building(config.num_threads_building);
        auto eval_case = eval::EvalCase::MakeInstance(config);
        Require(eval_case != nullptr, "failed to create search eval case");
        auto raw_eval_result = eval_case->Run();
        auto metrics = MergeMetrics(build_metrics, ExtractSearchMetrics(raw_eval_result));
        auto constraint_result = EvaluateConstraints(constraints, metrics);

        trial_result["status"] = "success";
        trial_result["metrics"] = metrics;
        trial_result["raw_eval_result"] =
            JsonType{{"build", build_result.value("raw_eval_result", JsonType(nullptr))},
                     {"search", raw_eval_result}};
        trial_result["satisfied_constraints"] = constraint_result["satisfied_constraints"];
        trial_result["violated_constraints"] = constraint_result["violated_constraints"];
        trial_result["failure"] = nullptr;
    } catch (const std::exception& e) {
        trial_result["status"] = "failed";
        trial_result["metrics"] = BuildMetricsFromResult(build_result);
        trial_result["raw_eval_result"] =
            JsonType{{"build", build_result.value("raw_eval_result", JsonType(nullptr))},
                     {"search", nullptr}};
        trial_result["satisfied_constraints"] = false;
        trial_result["violated_constraints"] = JsonType::array();
        trial_result["failure"] = e.what();
    }
    trial_result["elapsed_seconds"] = ElapsedSeconds(start);
    return trial_result;
}

void
CleanupBuildArtifact(const BuildSpec& build, const ExecutionOptions& options) {
    if (!options.keep_intermediate && build.cleanup_index_after_build_group &&
        std::filesystem::exists(build.index_path)) {
        std::error_code error;
        std::filesystem::remove(build.index_path, error);
    }
}

}  // namespace vsag::autotune::internal
