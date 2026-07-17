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
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <tuple>

#include "autotune_constraints.h"
#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

struct violation_score {
    uint64_t missing_metrics{0};
    uint64_t violated_constraints{0};
    double normalized_violation{0.0};
};

struct selection_candidate {
    JsonType value;
    violation_score score;
    double objective_value{0.0};
    std::string trial_id;
};

std::optional<double>
metric_value(const MetricMap& metrics, const std::string& name) {
    const auto found = metrics.find(name);
    if (found == metrics.end() || !std::isfinite(found->second)) {
        return std::nullopt;
    }
    return found->second;
}

bool
is_successful(const JsonType& record) {
    return record.value("status", std::string()) == "success";
}

violation_score
score_constraints(const ConstraintEvaluation& evaluation) {
    violation_score score;
    for (const auto& violation : evaluation.violations) {
        if (!violation.actual.has_value()) {
            ++score.missing_metrics;
            continue;
        }
        ++score.violated_constraints;
        score.normalized_violation += std::abs(*violation.actual - violation.expected) /
                                      std::max(std::abs(violation.expected), 1e-12);
    }
    return score;
}

bool
objective_less(double left, double right, ObjectiveDirection direction) {
    return direction == ObjectiveDirection::Maximize ? left > right : left < right;
}

selection_candidate
make_selection_candidate(const AutoTuneRequest& request,
                         const JsonType& trial,
                         const JsonType& build,
                         const MetricMap& metrics,
                         double objective_value) {
    const auto constraint_evaluation = EvaluateConstraints(request.constraints, metrics);
    const auto trial_id = trial.value("trial_id", std::string());
    const auto build_id = trial.value("build_id", std::string());
    JsonType value{{"index_name", build["index_name"]},
                   {"create_params", build["create_params"]},
                   {"search_params", trial["search_params"]},
                   {"workload",
                    JsonType{{"top_k", request.workload.top_k},
                             {"concurrency", request.workload.concurrency}}},
                   {"metrics", trial["metrics"]},
                   {"constraint_evaluation", ConstraintEvaluationToJson(constraint_evaluation)},
                   {"evidence", JsonType{{"build_id", build_id}, {"trial_id", trial_id}}},
                   {"artifacts", build["artifacts"]}};
    return selection_candidate{
        std::move(value), score_constraints(constraint_evaluation), objective_value, trial_id};
}

bool
better_feasible_candidate(const selection_candidate& left,
                          const selection_candidate& right,
                          ObjectiveDirection direction) {
    if (left.objective_value != right.objective_value) {
        return objective_less(left.objective_value, right.objective_value, direction);
    }
    return left.trial_id < right.trial_id;
}

bool
better_best_effort_candidate(const selection_candidate& left,
                             const selection_candidate& right,
                             ObjectiveDirection direction) {
    const auto left_score = std::tie(left.score.missing_metrics,
                                     left.score.violated_constraints,
                                     left.score.normalized_violation);
    const auto right_score = std::tie(right.score.missing_metrics,
                                      right.score.violated_constraints,
                                      right.score.normalized_violation);
    if (left_score != right_score) {
        return left_score < right_score;
    }
    return better_feasible_candidate(left, right, direction);
}

JsonType
violation_score_to_json(const violation_score& score) {
    return JsonType{{"missing_metric_count", score.missing_metrics},
                    {"violated_constraint_count", score.violated_constraints},
                    {"normalized_violation", score.normalized_violation}};
}

JsonType
value_or_null(const JsonType& object, const std::string& key) {
    if (!object.is_object() || !object.contains(key)) {
        return nullptr;
    }
    return object[key];
}

JsonType
make_build_report_record(const JsonType& build, bool include_raw_eval) {
    JsonType result{{"build_id", value_or_null(build, "build_id")},
                    {"status", value_or_null(build, "status")},
                    {"index_name", value_or_null(build, "index_name")},
                    {"eval_type", value_or_null(build, "eval_type")},
                    {"create_params", value_or_null(build, "create_params")},
                    {"metrics", value_or_null(build, "metrics")},
                    {"constraint_evaluation", value_or_null(build, "constraint_evaluation")},
                    {"artifacts", value_or_null(build, "artifacts")},
                    {"elapsed_seconds", value_or_null(build, "elapsed_seconds")},
                    {"failure", value_or_null(build, "failure")}};
    if (include_raw_eval && build.contains("raw_eval_result") &&
        !build["raw_eval_result"].is_null()) {
        result["raw_eval_result"] = build["raw_eval_result"];
    }
    return result;
}

JsonType
make_trial_execution_evidence(const JsonType& trial, const WorkloadSpec& workload) {
    const auto artifacts = trial.value("artifacts", JsonType::object());
    return JsonType{
        {"query_count", value_or_null(trial, "query_count")},
        {"requested_concurrency", workload.concurrency},
        {"index_instance_reuse", value_or_null(artifacts, "index_instance_reuse")},
        {"load_policy", value_or_null(artifacts, "load_policy")},
        {"reload_succeeded", value_or_null(artifacts, "reload_succeeded")},
        {"index_deserialize_count", value_or_null(artifacts, "index_deserialize_count")}};
}

JsonType
make_trial_report_record(const JsonType& trial,
                         const WorkloadSpec& workload,
                         bool include_raw_eval) {
    JsonType result{{"trial_id", value_or_null(trial, "trial_id")},
                    {"build_id", value_or_null(trial, "build_id")},
                    {"status", value_or_null(trial, "status")},
                    {"top_k", workload.top_k},
                    {"search_params", value_or_null(trial, "search_params")},
                    {"metrics", value_or_null(trial, "metrics")},
                    {"constraint_evaluation", value_or_null(trial, "constraint_evaluation")},
                    {"execution", make_trial_execution_evidence(trial, workload)},
                    {"elapsed_seconds", value_or_null(trial, "elapsed_seconds")},
                    {"failure", value_or_null(trial, "failure")}};
    if (include_raw_eval && trial.contains("raw_eval_result") &&
        !trial["raw_eval_result"].is_null()) {
        result["raw_eval_result"] = trial["raw_eval_result"];
    }
    return result;
}

using OrderedJsonType = nlohmann::ordered_json;

OrderedJsonType
ordered_copy(const JsonType& value) {
    return OrderedJsonType(value);
}

OrderedJsonType
order_object(const JsonType& source, std::initializer_list<const char*> keys) {
    if (!source.is_object()) {
        return ordered_copy(source);
    }
    OrderedJsonType result = OrderedJsonType::object();
    for (const auto* key : keys) {
        if (source.contains(key)) {
            result[key] = ordered_copy(source[key]);
        }
    }
    for (const auto& item : source.items()) {
        if (!result.contains(item.key())) {
            result[item.key()] = ordered_copy(item.value());
        }
    }
    return result;
}

OrderedJsonType
order_recommendation(const JsonType& recommendation) {
    return order_object(recommendation,
                        {"index_name",
                         "create_params",
                         "search_params",
                         "workload",
                         "metrics",
                         "constraint_evaluation",
                         "violation_summary",
                         "artifacts",
                         "evidence"});
}

OrderedJsonType
order_failure(const JsonType& failure) {
    return order_object(failure, {"stage", "code", "message"});
}

}  // namespace

JsonType
SelectResult(const AutoTuneRequest& request, const EvaluationResult& evaluation) {
    JsonType selection{{"recommendation", nullptr}, {"best_effort", nullptr}, {"failure", nullptr}};
    const auto objective_direction = GetMetricDefinition(request.objective.metric).direction;
    std::map<std::string, const JsonType*> builds;
    for (const auto& build : evaluation.build_results) {
        builds.emplace(build.value("build_id", std::string()), &build);
    }

    std::optional<selection_candidate> best_feasible;
    std::optional<selection_candidate> best_effort;
    for (const auto& trial : evaluation.trial_results) {
        if (!is_successful(trial)) {
            continue;
        }
        const auto metrics = MetricsFromJson(trial.value("metrics", JsonType::object()));
        const auto objective_value = metric_value(metrics, request.objective.metric);
        if (!objective_value.has_value()) {
            selection["status"] = "failed";
            selection["failure"] =
                JsonType{{"stage", "selection"},
                         {"code", "objective_metric_missing"},
                         {"message", "successful trial is missing the objective metric"},
                         {"metric", request.objective.metric},
                         {"trial_id", trial.value("trial_id", std::string())}};
            return selection;
        }
        const auto build = builds.find(trial.value("build_id", std::string()));
        Require(build != builds.end(), "successful trial references an unknown build");
        Require(is_successful(*build->second), "failed to assemble a successful trial result");
        auto candidate =
            make_selection_candidate(request, trial, *build->second, metrics, *objective_value);
        if (candidate.score.missing_metrics == 0 && candidate.score.violated_constraints == 0) {
            if (!best_feasible.has_value() ||
                better_feasible_candidate(candidate, *best_feasible, objective_direction)) {
                best_feasible = std::move(candidate);
            }
        } else if (!best_effort.has_value() ||
                   better_best_effort_candidate(candidate, *best_effort, objective_direction)) {
            best_effort = std::move(candidate);
        }
    }

    if (best_feasible.has_value()) {
        best_feasible->value.erase("constraint_evaluation");
        selection["status"] = "success";
        selection["recommendation"] = std::move(best_feasible->value);
        return selection;
    }

    if (best_effort.has_value()) {
        best_effort->value["violation_summary"] = violation_score_to_json(best_effort->score);
        selection["status"] = "no_candidate_satisfied";
        selection["best_effort"] = std::move(best_effort->value);
        return selection;
    }

    selection["status"] = "failed";
    selection["failure"] = JsonType{{"stage", "selection"},
                                    {"code", "no_successful_trial"},
                                    {"message", "no search trial completed successfully"}};
    return selection;
}

JsonType
MakeBuildReport(const std::vector<JsonType>& builds, bool include_raw_eval) {
    JsonType result = JsonType::array();
    for (const auto& build : builds) {
        result.push_back(make_build_report_record(build, include_raw_eval));
    }
    return result;
}

JsonType
MakeTrialReport(const std::vector<JsonType>& trials,
                const WorkloadSpec& workload,
                bool include_raw_eval) {
    JsonType result = JsonType::array();
    for (const auto& trial : trials) {
        result.push_back(make_trial_report_record(trial, workload, include_raw_eval));
    }
    return result;
}

JsonType
MakeResultSummary(const JsonType& report) {
    JsonType result{{"version", value_or_null(report, "version")},
                    {"status", value_or_null(report, "status")},
                    {"elapsed_seconds", value_or_null(report, "elapsed_seconds")}};
    const auto status = report.value("status", std::string());
    const auto* result_key = status == "success"                  ? "recommendation"
                             : status == "no_candidate_satisfied" ? "best_effort"
                                                                  : "failure";
    const auto result_value = value_or_null(report, result_key);
    if (!result_value.is_null()) {
        result[result_key] = result_value;
    }
    const auto report_path = value_or_null(report, "report_path");
    if (!report_path.is_null()) {
        result["report_path"] = report_path;
    }
    return result;
}

std::string
FormatResultSummaryForCli(const JsonType& report) {
    const auto summary = MakeResultSummary(report);
    OrderedJsonType result = OrderedJsonType::object();
    const auto status = summary["status"].is_string() ? summary["status"].get<std::string>() : "";
    if (status == "success" && summary.contains("recommendation")) {
        result["recommendation"] = order_recommendation(summary["recommendation"]);
    } else if (status == "no_candidate_satisfied" && summary.contains("best_effort")) {
        result["best_effort"] = order_recommendation(summary["best_effort"]);
    } else if (summary.contains("failure")) {
        result["failure"] = order_failure(summary["failure"]);
    }
    result["status"] = ordered_copy(summary["status"]);
    result["elapsed_seconds"] = ordered_copy(summary["elapsed_seconds"]);
    if (summary.contains("report_path")) {
        result["report_path"] = ordered_copy(summary["report_path"]);
    }
    result["version"] = ordered_copy(summary["version"]);
    return result.dump(2);
}

void
WriteJsonFile(const std::string& path, const JsonType& json) {
    if (path.empty()) {
        return;
    }
    const auto output_path = std::filesystem::path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(path);
    Require(output.good(), "failed to open result_path: " + path);
    output << json.dump(2) << std::endl;
    Require(output.good(), "failed to write result_path: " + path);
}

JsonType
MakeFailedResult(const JsonType& request,
                 const std::string& stage,
                 const std::string& code,
                 const std::string& message,
                 const Clock::time_point& total_start) {
    JsonType result{{"version", 1},
                    {"run_id", nullptr},
                    {"run_workspace_path", nullptr},
                    {"report_path", nullptr},
                    {"status", "failed"},
                    {"input_request", request},
                    {"effective_request", nullptr},
                    {"environment", JsonType::object()},
                    {"evaluation_strategy", nullptr},
                    {"objective", nullptr},
                    {"elapsed_seconds", ElapsedSeconds(total_start)},
                    {"elapsed_breakdown_seconds", JsonType::object()},
                    {"recommendation", nullptr},
                    {"best_effort", nullptr},
                    {"trial_count", 0},
                    {"build_count", 0},
                    {"build_group_count", 0},
                    {"builds", JsonType::array()},
                    {"trials", JsonType::array()},
                    {"failure", JsonType{{"stage", stage}, {"code", code}, {"message", message}}}};
    return result;
}

}  // namespace vsag::autotune::internal
