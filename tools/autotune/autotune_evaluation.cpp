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
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>

#include "autotune_constraints.h"
#include "autotune_internal.h"
#include "case/eval_case.h"
#include "eval_config.h"

namespace vsag::autotune::internal {

namespace {

constexpr double BYTES_PER_MEBIBYTE = 1024.0 * 1024.0;

std::mutex&
evaluation_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::optional<double>
get_json_double(const JsonType& object, const std::string& key) {
    if (!object.is_object() || !object.contains(key) || !object[key].is_number()) {
        return std::nullopt;
    }
    return object[key].get<double>();
}

std::optional<double>
get_nested_json_double(const JsonType& object,
                       const std::string& first_key,
                       const std::string& second_key) {
    if (!object.is_object() || !object.contains(first_key) || !object[first_key].is_object()) {
        return std::nullopt;
    }
    return get_json_double(object[first_key], second_key);
}

void
set_metric(MetricMap& metrics, const std::string& name, const std::optional<double>& value) {
    if (value.has_value()) {
        metrics[name] = *value;
    }
}

void
set_index_size(MetricMap& metrics, const std::string& index_path) {
    std::error_code error;
    const auto bytes = std::filesystem::file_size(index_path, error);
    if (!error) {
        metrics["index_size_mb"] = static_cast<double>(bytes) / BYTES_PER_MEBIBYTE;
    }
}

MetricMap
extract_build_metrics(const JsonType& raw_result, const std::string& index_path) {
    MetricMap metrics;
    set_metric(metrics, "build_seconds", get_json_double(raw_result, "duration(s)"));
    const auto memory_bytes = get_json_double(raw_result, "index_memory(B)");
    if (memory_bytes.has_value()) {
        metrics["index_memory_mb"] = *memory_bytes / BYTES_PER_MEBIBYTE;
    }
    set_index_size(metrics, index_path);
    return metrics;
}

MetricMap
extract_search_metrics(const JsonType& raw_result, double measured_search_seconds) {
    MetricMap metrics;
    set_metric(metrics, "recall_at_k", get_json_double(raw_result, "recall_avg"));
    set_metric(metrics, "latency_avg_ms", get_json_double(raw_result, "latency_avg(ms)"));
    set_metric(
        metrics, "latency_p99_ms", get_nested_json_double(raw_result, "latency_detail(ms)", "p99"));
    set_metric(metrics, "qps", get_json_double(raw_result, "qps"));
    auto search_seconds = get_json_double(raw_result, "duration(s)");
    metrics["search_seconds"] = search_seconds.value_or(measured_search_seconds);
    const auto memory_bytes = get_json_double(raw_result, "index_memory(B)");
    if (memory_bytes.has_value()) {
        metrics["index_memory_mb"] = *memory_bytes / BYTES_PER_MEBIBYTE;
    }
    return metrics;
}

MetricMap
merge_metrics(const MetricMap& shared_metrics, const MetricMap& workload_metrics) {
    MetricMap result = shared_metrics;
    for (const auto& [name, value] : workload_metrics) {
        result[name] = value;
    }
    return result;
}

ConstraintMap
shared_constraints(const ConstraintMap& constraints) {
    ConstraintMap result;
    for (const auto& [name, threshold] : constraints) {
        if (GetMetricDefinition(name).scope == MetricScope::Shared) {
            result.emplace(name, threshold);
        }
    }
    return result;
}

JsonType
metrics_to_json(const MetricMap& metrics) {
    JsonType value = JsonType::object();
    for (const auto& [name, metric_value] : metrics) {
        value[name] = metric_value;
    }
    return value;
}

JsonType
make_failure(const std::string& stage, const std::string& code, const std::string& message) {
    return JsonType{{"stage", stage}, {"code", code}, {"message", message}};
}

eval::EvalConfig
make_build_config(const AutoTuneRequest& request, const BuildSpec& build) {
    eval::EvalConfig config;
    config.dataset_path = request.data_path;
    config.index_name = build.index_name;
    config.build_param = build.create_params.dump();
    config.index_path = build.index_path;
    config.enable_memory = false;
    return config;
}

eval::EvalConfig
make_search_config(const AutoTuneRequest& request, const TrialSpec& trial, const BuildSpec& build) {
    eval::EvalConfig config;
    config.dataset_path = request.data_path;
    config.index_name = build.index_name;
    config.build_param = build.create_params.dump();
    config.index_path = build.index_path;
    config.search_param = trial.search_params.dump();
    config.search_mode = "knn";
    config.top_k = static_cast<int>(request.workload.top_k);
    config.search_query_count = request.dataset.query_count;
    config.num_threads_searching = static_cast<int32_t>(request.workload.concurrency);
    config.delete_index_after_search = false;
    config.enable_memory = false;
    return config;
}

JsonType
make_build_shell(const BuildSpec& build) {
    JsonType result{{"build_id", build.build_id},
                    {"status", "failed"},
                    {"index_name", build.index_name},
                    {"eval_type", build.use_existing_index ? "existing_index" : "build"},
                    {"create_params", build.create_params},
                    {"metrics", JsonType::object()},
                    {"build_invoked", false},
                    {"artifacts",
                     JsonType{{"index_path", build.index_path},
                              {"use_existing_index", build.use_existing_index},
                              {"cleanup_planned", build.cleanup_index_after_build_group},
                              {"expected_to_exist_after_response", false}}},
                    {"failure", nullptr}};
    return result;
}

JsonType
run_build(const AutoTuneRequest& request, const BuildSpec& build) {
    auto result = make_build_shell(build);
    const auto start = Clock::now();
    try {
        MetricMap metrics;
        if (build.use_existing_index) {
            set_index_size(metrics, build.index_path);
        } else {
            auto config = make_build_config(request, build);
            auto eval_case = eval::EvalCase::MakeInstance(config, "build");
            Require(eval_case != nullptr, "failed to create build eval case");
            result["build_invoked"] = true;
            auto raw_result = eval_case->Run();
            if (request.output.include_raw_eval) {
                result["raw_eval_result"] = raw_result;
            }
            metrics = extract_build_metrics(raw_result, build.index_path);
        }
        std::error_code artifact_error;
        const bool artifact_is_regular =
            std::filesystem::is_regular_file(build.index_path, artifact_error);
        Require(!artifact_error && artifact_is_regular,
                "build did not produce a complete index artifact: " + build.index_path);
        result["metrics"] = metrics_to_json(metrics);
        result["artifacts"]["expected_to_exist_after_response"] =
            build.use_existing_index || !build.cleanup_index_after_build_group;
        result["status"] = "success";
    } catch (const std::exception& error) {
        result["artifacts"]["expected_to_exist_after_response"] = false;
        if (!build.use_existing_index) {
            std::error_code cleanup_error;
            std::filesystem::remove(build.index_path, cleanup_error);
            if (cleanup_error) {
                result["artifacts"]["incomplete_artifact_cleanup_error"] = cleanup_error.message();
            }
        }
        result["failure"] = make_failure("build", "build_evaluation_failed", error.what());
    }
    result["elapsed_seconds"] = ElapsedSeconds(start);
    return result;
}

JsonType
make_trial_shell(const TrialSpec& trial) {
    JsonType result{
        {"trial_id", trial.trial_id},
        {"build_id", trial.build_id},
        {"status", "failed"},
        {"query_count", nullptr},
        {"search_params", trial.search_params},
        {"metrics", JsonType::object()},
        {"artifacts",
         JsonType{{"index_instance_reuse", false}, {"load_policy", "fresh_deserialize_per_trial"}}},
        {"failure", nullptr}};
    return result;
}

JsonType
run_trial(const AutoTuneRequest& request,
          const TrialSpec& trial,
          const BuildSpec& build,
          const JsonType& build_result) {
    auto result = make_trial_shell(trial);
    const auto start = Clock::now();
    if (build_result.value("status", std::string()) != "success") {
        result["artifacts"]["reload_succeeded"] = false;
        result["failure"] =
            make_failure("search", "build_failed", "search trial skipped because its build failed");
        result["elapsed_seconds"] = ElapsedSeconds(start);
        return result;
    }

    result["artifacts"]["reload_succeeded"] = false;
    try {
        auto config = make_search_config(request, trial, build);
        auto eval_case = eval::EvalCase::MakeInstance(config, "search");
        Require(eval_case != nullptr, "failed to create search eval case");
        const auto search_start = Clock::now();
        auto raw_result = eval_case->Run();
        const double measured_search_seconds = ElapsedSeconds(search_start);
        result["artifacts"]["reload_succeeded"] = true;
        if (request.output.include_raw_eval) {
            result["raw_eval_result"] = raw_result;
        }
        result["query_count"] = raw_result.value("statistics_query_count", uint64_t{0});
        Require(result["query_count"] == request.dataset.query_count,
                "eval did not execute the complete query set");

        auto shared_metrics = MetricsFromJson(build_result["metrics"]);
        auto workload_metrics = extract_search_metrics(raw_result, measured_search_seconds);
        if (!build.use_existing_index) {
            const auto build_seconds = shared_metrics.find("build_seconds");
            const auto search_seconds = workload_metrics.find("search_seconds");
            if (build_seconds != shared_metrics.end() && search_seconds != workload_metrics.end()) {
                workload_metrics["build_and_search_seconds"] =
                    build_seconds->second + search_seconds->second;
            }
        }
        result["metrics"] = metrics_to_json(merge_metrics(shared_metrics, workload_metrics));
        result["artifacts"]["index_deserialize_count"] = 1;
        result["status"] = "success";
    } catch (const std::exception& error) {
        result["status"] = "failed";
        const auto shared_metrics = MetricsFromJson(build_result["metrics"]);
        result["metrics"] = metrics_to_json(shared_metrics);
        result["failure"] = make_failure("search", "search_evaluation_failed", error.what());
    }
    result["elapsed_seconds"] = ElapsedSeconds(start);
    return result;
}

void
update_build_metrics_from_trials(JsonType& build_result,
                                 const ConstraintMap& constraints,
                                 const std::vector<JsonType>& trials) {
    auto metrics = MetricsFromJson(build_result["metrics"]);
    for (const auto& trial : trials) {
        if (trial.value("status", std::string()) != "success") {
            continue;
        }
        const auto trial_metrics = MetricsFromJson(trial["metrics"]);
        const auto memory = trial_metrics.find("index_memory_mb");
        if (memory != trial_metrics.end()) {
            metrics[memory->first] = memory->second;
            break;
        }
    }
    build_result["metrics"] = metrics_to_json(metrics);
    build_result["constraint_evaluation"] =
        ConstraintEvaluationToJson(EvaluateConstraints(constraints, metrics));
}

void
normalize_trial_shared_metrics(const ConstraintMap& constraints,
                               const JsonType& build_result,
                               std::vector<JsonType>& trials) {
    const auto canonical_metrics = MetricsFromJson(build_result["metrics"]);
    for (auto& trial : trials) {
        auto metrics = MetricsFromJson(trial["metrics"]);
        for (auto metric = metrics.begin(); metric != metrics.end();) {
            if (GetMetricDefinition(metric->first).scope == MetricScope::Shared) {
                metric = metrics.erase(metric);
            } else {
                ++metric;
            }
        }
        for (const auto& [name, value] : canonical_metrics) {
            if (GetMetricDefinition(name).scope == MetricScope::Shared) {
                metrics[name] = value;
            }
        }
        trial["metrics"] = metrics_to_json(metrics);
        const auto constraint_evaluation = EvaluateConstraints(constraints, metrics);
        trial["constraint_evaluation"] = ConstraintEvaluationToJson(constraint_evaluation);
    }
}

void
cleanup_artifact(const BuildSpec& build, JsonType& build_result) {
    if (!build.cleanup_index_after_build_group || build.use_existing_index) {
        build_result["artifacts"]["cleaned"] = false;
        return;
    }
    std::error_code error;
    const bool removed = std::filesystem::remove(build.index_path, error);
    build_result["artifacts"]["cleaned"] = removed;
    if (error) {
        build_result["artifacts"]["cleanup_error"] = error.message();
    }
}

}  // namespace

EvaluationResult
EvaluatePlan(const AutoTuneRequest& request, const AutoTunePlan& plan) {
    std::lock_guard<std::mutex> lock(evaluation_mutex());
    const auto build_constraints = shared_constraints(request.constraints);
    EvaluationResult evaluation;
    evaluation.build_results.reserve(plan.builds.size());
    evaluation.trial_results.reserve(plan.trials.size());

    std::map<std::string, std::vector<const TrialSpec*>> trials_by_build;
    for (const auto& trial : plan.trials) {
        trials_by_build[trial.build_id].push_back(&trial);
    }

    for (const auto& build : plan.builds) {
        auto build_result = run_build(request, build);
        if (build_result.value("build_invoked", false)) {
            ++evaluation.executed_build_count;
        }

        std::vector<JsonType> group_trials;
        const auto group = trials_by_build.find(build.build_id);
        Require(group != trials_by_build.end(), "build group has no search trials");
        group_trials.reserve(group->second.size());
        for (const auto* trial : group->second) {
            auto trial_result = run_trial(request, *trial, build, build_result);
            group_trials.emplace_back(std::move(trial_result));
        }
        update_build_metrics_from_trials(build_result, build_constraints, group_trials);
        normalize_trial_shared_metrics(request.constraints, build_result, group_trials);
        for (auto& trial_result : group_trials) {
            evaluation.trial_results.emplace_back(std::move(trial_result));
        }
        cleanup_artifact(build, build_result);
        evaluation.build_results.emplace_back(std::move(build_result));
    }
    return evaluation;
}

}  // namespace vsag::autotune::internal
