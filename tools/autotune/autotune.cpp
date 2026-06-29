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

#include "autotune.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "case/eval_case.h"
#include "eval_config.h"
#include "vsag/options.h"

namespace vsag::autotune {
namespace {

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

constexpr const char* kIndexHGraph = "hgraph";
constexpr const char* kIndexIvf = "ivf";

struct ExecutionOptions {
    int top_k{10};
    std::string search_mode{"knn"};
    uint64_t search_query_count{0};
    int32_t num_threads_building{1};
    int32_t num_threads_searching{1};
    std::string workspace_path{"/tmp/vsag_autotune"};
    bool keep_intermediate{false};
    uint64_t max_trials{0};
    bool include_trials{true};
    std::string result_path;
};

struct CandidateSpec {
    std::string index_name;
    JsonType create_params;
    JsonType search_params;
};

struct TrialSpec {
    std::string trial_id;
    std::string index_name;
    std::string eval_type;
    std::string index_path;
    JsonType create_params;
    JsonType search_params;
};

double
ElapsedSeconds(const Clock::time_point& start) {
    return std::chrono::duration_cast<Seconds>(Clock::now() - start).count();
}

void
Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string
GetString(const JsonType& object, const std::string& key, const std::string& default_value) {
    if (!object.is_object() || !object.contains(key)) {
        return default_value;
    }
    Require(object[key].is_string(), key + " must be a string");
    return object[key].get<std::string>();
}

int
GetInt(const JsonType& object, const std::string& key, int default_value) {
    if (!object.is_object() || !object.contains(key)) {
        return default_value;
    }
    Require(object[key].is_number_integer(), key + " must be an integer");
    return object[key].get<int>();
}

uint64_t
GetUInt64(const JsonType& object, const std::string& key, uint64_t default_value) {
    if (!object.is_object() || !object.contains(key)) {
        return default_value;
    }
    Require(object[key].is_number_unsigned() || object[key].is_number_integer(),
            key + " must be an unsigned integer");
    auto value = object[key].get<int64_t>();
    Require(value >= 0, key + " must be an unsigned integer");
    return static_cast<uint64_t>(value);
}

bool
GetBool(const JsonType& object, const std::string& key, bool default_value) {
    if (!object.is_object() || !object.contains(key)) {
        return default_value;
    }
    Require(object[key].is_boolean(), key + " must be a boolean");
    return object[key].get<bool>();
}

JsonType&
EnsureObject(JsonType& object, const std::string& key) {
    if (!object.contains(key) || object[key].is_null()) {
        object[key] = JsonType::object();
    }
    Require(object[key].is_object(), key + " must be an object");
    return object[key];
}

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

std::vector<JsonType>
ExpandRange(const JsonType& range) {
    Require(range.is_object(), "$range must be an object");
    Require(range.contains("start") && range.contains("stop") && range.contains("step"),
            "$range requires start, stop and step");
    Require(range["start"].is_number() && range["stop"].is_number() && range["step"].is_number(),
            "$range start, stop and step must be numbers");

    const double start = range["start"].get<double>();
    const double stop = range["stop"].get<double>();
    const double step = range["step"].get<double>();
    Require(step != 0.0, "$range step must not be zero");
    Require((start <= stop && step > 0.0) || (start >= stop && step < 0.0),
            "$range step direction does not reach stop");

    const bool integer_values = range["start"].is_number_integer() &&
                                range["stop"].is_number_integer() &&
                                range["step"].is_number_integer();
    std::vector<JsonType> values;
    double current = start;
    uint64_t guard = 0;
    while ((step > 0.0 && current <= stop + 1e-9) || (step < 0.0 && current >= stop - 1e-9)) {
        if (integer_values) {
            values.emplace_back(static_cast<int64_t>(std::llround(current)));
        } else {
            values.emplace_back(current);
        }
        current += step;
        ++guard;
        Require(guard <= 1000000, "$range generated too many values");
    }
    Require(!values.empty(), "$range generated no value");
    return values;
}

std::vector<JsonType>
ExpandJson(const JsonType& value) {
    if (value.is_object()) {
        if (value.contains("$value")) {
            Require(value.size() == 1, "$value cannot be mixed with other keys");
            return {value["$value"]};
        }
        if (value.contains("$range")) {
            Require(value.size() == 1, "$range cannot be mixed with other keys");
            return ExpandRange(value["$range"]);
        }

        std::vector<JsonType> partials{JsonType::object()};
        for (auto it = value.begin(); it != value.end(); ++it) {
            auto expanded_values = ExpandJson(it.value());
            std::vector<JsonType> next_partials;
            for (const auto& partial : partials) {
                for (const auto& expanded_value : expanded_values) {
                    JsonType next = partial;
                    next[it.key()] = expanded_value;
                    next_partials.emplace_back(std::move(next));
                }
            }
            partials = std::move(next_partials);
        }
        return partials;
    }

    if (value.is_array()) {
        Require(!value.empty(), "candidate array must not be empty");
        std::vector<JsonType> values;
        values.reserve(value.size());
        for (const auto& item : value) {
            values.emplace_back(item);
        }
        return values;
    }

    return {value};
}

void
FillHGraphDefaults(JsonType& index_spec) {
    auto& create_params = EnsureObject(index_spec, "create_params");
    auto& index_param = EnsureObject(create_params, "index_param");
    if (!index_param.contains("base_quantization_type")) {
        index_param["base_quantization_type"] = JsonType::array({"fp32", "sq8_uniform"});
    }
    if (!index_param.contains("max_degree")) {
        index_param["max_degree"] = JsonType::array({16, 32});
    }
    if (!index_param.contains("ef_construction")) {
        index_param["ef_construction"] = JsonType::array({100, 200});
    }

    auto& search_params = EnsureObject(index_spec, "search_params");
    auto& hgraph_params = EnsureObject(search_params, kIndexHGraph);
    if (!hgraph_params.contains("ef_search")) {
        hgraph_params["ef_search"] = JsonType::array({40, 80, 120});
    }
}

void
FillIvfDefaults(JsonType& index_spec) {
    auto& create_params = EnsureObject(index_spec, "create_params");
    auto& index_param = EnsureObject(create_params, "index_param");
    if (!index_param.contains("partition_strategy_type")) {
        index_param["partition_strategy_type"] = "ivf";
    }
    if (!index_param.contains("base_quantization_type")) {
        index_param["base_quantization_type"] = JsonType::array({"fp32", "sq8_uniform"});
    }
    if (!index_param.contains("buckets_count")) {
        index_param["buckets_count"] = JsonType::array({1024, 2048});
    }
    if (!index_param.contains("ivf_train_type")) {
        index_param["ivf_train_type"] = "kmeans";
    }

    auto& search_params = EnsureObject(index_spec, "search_params");
    auto& ivf_params = EnsureObject(search_params, kIndexIvf);
    if (!ivf_params.contains("scan_buckets_count")) {
        ivf_params["scan_buckets_count"] = JsonType::array({16, 32, 64});
    }
}

void
FillIndexDefaults(JsonType& index_spec) {
    const auto index_name = GetString(index_spec, "name", "");
    Require(IsSupportedIndex(index_name), "unsupported index: " + index_name);
    if (index_name == kIndexHGraph) {
        FillHGraphDefaults(index_spec);
    } else if (index_name == kIndexIvf) {
        FillIvfDefaults(index_spec);
    }

    const auto& create_params = index_spec["create_params"];
    Require(create_params.contains("dim"), index_name + " create_params.dim is required");
    Require(create_params.contains("dtype"), index_name + " create_params.dtype is required");
    Require(create_params.contains("metric_type"),
            index_name + " create_params.metric_type is required");
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

std::vector<CandidateSpec>
GenerateCandidates(const JsonType& request) {
    std::vector<CandidateSpec> candidates;
    for (const auto& raw_index_spec : request["indexes"]) {
        JsonType index_spec = raw_index_spec;
        FillIndexDefaults(index_spec);

        const auto index_name = index_spec["name"].get<std::string>();
        auto create_candidates = ExpandJson(index_spec["create_params"]);
        auto search_candidates = ExpandJson(index_spec["search_params"]);

        for (const auto& create_params : create_candidates) {
            for (const auto& search_params : search_candidates) {
                candidates.emplace_back(CandidateSpec{index_name, create_params, search_params});
            }
        }
    }
    return candidates;
}

std::string
MakeTrialId(const std::string& index_name, uint64_t ordinal) {
    std::ostringstream oss;
    oss << index_name << "-" << std::setw(6) << std::setfill('0') << ordinal;
    return oss.str();
}

std::vector<TrialSpec>
PlanTrials(const JsonType& request,
           const std::vector<CandidateSpec>& candidates,
           const ExecutionOptions& options) {
    Require(!candidates.empty(), "no candidates generated");
    if (options.max_trials > 0) {
        Require(candidates.size() <= options.max_trials,
                "candidate count exceeds execution.max_trials");
    }

    const std::string existing_index_path = GetString(request, "index_path", "");
    std::set<std::string> index_names;
    std::set<std::string> create_param_dumps;
    for (const auto& candidate : candidates) {
        index_names.emplace(candidate.index_name);
        create_param_dumps.emplace(candidate.create_params.dump());
    }

    const bool search_only =
        !existing_index_path.empty() && index_names.size() == 1 && create_param_dumps.size() == 1;
    if (!existing_index_path.empty()) {
        Require(std::filesystem::exists(existing_index_path),
                "index_path does not exist: " + existing_index_path);
    }

    std::vector<TrialSpec> trials;
    const auto trial_dir = std::filesystem::path(options.workspace_path) / "trials";
    uint64_t ordinal = 0;
    for (const auto& candidate : candidates) {
        ++ordinal;
        auto trial_id = MakeTrialId(candidate.index_name, ordinal);
        std::string index_path = existing_index_path;
        std::string eval_type = "search";
        if (!search_only) {
            eval_type = "build,search";
            index_path = (trial_dir / (trial_id + ".index")).string();
        }
        trials.emplace_back(TrialSpec{trial_id,
                                      candidate.index_name,
                                      eval_type,
                                      index_path,
                                      candidate.create_params,
                                      candidate.search_params});
    }
    return trials;
}

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
    return JsonType{{"trial_id", trial["trial_id"]},
                    {"index_name", trial["index_name"]},
                    {"create_params", trial["create_params"]},
                    {"search_params", trial["search_params"]},
                    {"metrics", trial["metrics"]},
                    {"selection_reason", reason}};
}

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
    result["failure"] = JsonType{{"message", message}};
    return result;
}

}  // namespace

JsonType
RunAutoTune(const JsonType& request) {
    const auto total_start = Clock::now();
    try {
        JsonType elapsed_breakdown;

        const auto validation_start = Clock::now();
        ValidateRequest(request);
        auto options = ParseExecutionOptions(request);
        elapsed_breakdown["validation"] = ElapsedSeconds(validation_start);

        const auto candidate_start = Clock::now();
        auto candidates = GenerateCandidates(request);
        auto trials = PlanTrials(request, candidates, options);
        elapsed_breakdown["candidate_generation"] = ElapsedSeconds(candidate_start);

        const auto evaluation_start = Clock::now();
        std::vector<JsonType> trial_results;
        trial_results.reserve(trials.size());
        uint64_t trial_ordinal = 0;
        for (const auto& trial : trials) {
            ++trial_ordinal;
            std::cerr << "[AutoTune] running trial " << trial_ordinal << "/" << trials.size() << " "
                      << trial.trial_id << " index=" << trial.index_name
                      << " eval_type=" << trial.eval_type << std::endl;
            auto trial_result = RunTrial(trial, request, request["constraints"], options);
            std::cerr << "[AutoTune] finished trial " << trial.trial_id
                      << " status=" << trial_result["status"].get<std::string>()
                      << " elapsed_seconds=" << trial_result["elapsed_seconds"].get<double>()
                      << std::endl;
            trial_results.emplace_back(std::move(trial_result));
        }
        elapsed_breakdown["evaluation"] = ElapsedSeconds(evaluation_start);

        const auto selection_start = Clock::now();
        auto selection = SelectResult(trial_results);
        elapsed_breakdown["selection"] = ElapsedSeconds(selection_start);

        JsonType result;
        result["version"] = 1;
        result["status"] = selection["status"];
        result["elapsed_seconds"] = ElapsedSeconds(total_start);
        result["elapsed_breakdown_seconds"] = elapsed_breakdown;
        result["recommendation"] = selection["recommendation"];
        result["best_effort"] = selection["best_effort"];
        result["trial_count"] = trial_results.size();
        result["failure"] = selection["failure"];
        if (options.include_trials) {
            result["trials"] = trial_results;
        }

        WriteJsonFile(options.result_path, result);
        return result;
    } catch (const std::exception& e) {
        auto result = MakeFailedResult(request, e.what(), total_start);
        const auto result_path = GetOptionalResultPath(request);
        if (!result_path.empty()) {
            try {
                WriteJsonFile(result_path, result);
            } catch (const std::exception& write_error) {
                result["failure"]["result_write_error"] = write_error.what();
            }
        }
        return result;
    }
}

std::vector<JsonType>
ExpandJsonForTest(const JsonType& value) {
    return ExpandJson(value);
}

JsonType
GenerateCandidatesForTest(const JsonType& request) {
    ValidateRequest(request);
    auto options = ParseExecutionOptions(request);
    auto candidates = GenerateCandidates(request);
    auto trials = PlanTrials(request, candidates, options);
    JsonType result;
    result["candidate_count"] = candidates.size();
    result["trial_count"] = trials.size();
    result["trials"] = JsonType::array();
    for (const auto& trial : trials) {
        result["trials"].push_back(JsonType{{"trial_id", trial.trial_id},
                                            {"index_name", trial.index_name},
                                            {"eval_type", trial.eval_type},
                                            {"index_path", trial.index_path},
                                            {"create_params", trial.create_params},
                                            {"search_params", trial.search_params}});
    }
    return result;
}

}  // namespace vsag::autotune
