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

#include "tuning/auto_tuning_api.h"

#include <exception>
#include <limits>
#include <nlohmann/json.hpp>

namespace vsag {
namespace {

using JsonType = nlohmann::json;

AutoTuningApiParseResult
MakeError(AutoTuningApiStatus status, const std::string& code, const std::string& message) {
    AutoTuningApiParseResult result;
    result.status = status;
    result.error_code = code;
    result.error_message = message;
    return result;
}

AutoTuningApiParseResult
InvalidArgument(const std::string& code, const std::string& message) {
    return MakeError(AutoTuningApiStatus::INVALID_ARGUMENT, code, message);
}

AutoTuningApiParseResult
Unsupported(const std::string& code, const std::string& message) {
    return MakeError(AutoTuningApiStatus::UNSUPPORTED, code, message);
}

AutoTuningApiParseResult
ReadRequiredObject(const JsonType& parent,
                   const std::string& key,
                   const std::string& path,
                   const JsonType*& value) {
    if (not parent.contains(key)) {
        return InvalidArgument("missing_field", path + "." + key + " is required");
    }
    if (not parent.at(key).is_object()) {
        return InvalidArgument("invalid_field", path + "." + key + " must be an object");
    }
    value = &parent.at(key);
    return {};
}

AutoTuningApiParseResult
ReadRequiredString(const JsonType& parent,
                   const std::string& key,
                   const std::string& path,
                   std::string& value) {
    if (not parent.contains(key)) {
        return InvalidArgument("missing_field", path + "." + key + " is required");
    }
    if (not parent.at(key).is_string()) {
        return InvalidArgument("invalid_field", path + "." + key + " must be a string");
    }
    value = parent.at(key).get<std::string>();
    return {};
}

bool
ReadUint64Value(const JsonType& value, uint64_t& output) {
    if (value.is_number_unsigned()) {
        output = value.get<uint64_t>();
        return true;
    }
    if (value.is_number_integer()) {
        const auto signed_value = value.get<int64_t>();
        if (signed_value < 0) {
            return false;
        }
        output = static_cast<uint64_t>(signed_value);
        return true;
    }
    return false;
}

AutoTuningApiParseResult
ReadRequiredUint64(const JsonType& parent,
                   const std::string& key,
                   const std::string& path,
                   uint64_t& value) {
    if (not parent.contains(key)) {
        return InvalidArgument("missing_field", path + "." + key + " is required");
    }
    if (not ReadUint64Value(parent.at(key), value)) {
        return InvalidArgument("invalid_field", path + "." + key + " must be a uint64");
    }
    return {};
}

AutoTuningApiParseResult
ReadRequiredDouble(const JsonType& parent,
                   const std::string& key,
                   const std::string& path,
                   double& value) {
    if (not parent.contains(key)) {
        return InvalidArgument("missing_field", path + "." + key + " is required");
    }
    if (not parent.at(key).is_number()) {
        return InvalidArgument("invalid_field", path + "." + key + " must be a number");
    }
    value = parent.at(key).get<double>();
    return {};
}

AutoTuningApiParseResult
ValidateUnsupportedObject(const JsonType& parent,
                          const std::string& key,
                          const std::string& code,
                          const std::string& message) {
    if (parent.contains(key) && not parent.at(key).is_null()) {
        if (not parent.at(key).is_object() || not parent.at(key).empty()) {
            return Unsupported(code, message);
        }
    }
    return {};
}

AutoTuningApiParseResult
RejectPresentField(const JsonType& parent,
                   const std::string& key,
                   const std::string& code,
                   const std::string& message) {
    if (parent.contains(key) && not parent.at(key).is_null()) {
        return Unsupported(code, message);
    }
    return {};
}

const char*
ApiStatusName(AutoTuningApiStatus status) {
    switch (status) {
        case AutoTuningApiStatus::SUCCESS:
            return "success";
        case AutoTuningApiStatus::INVALID_ARGUMENT:
            return "invalid_argument";
        case AutoTuningApiStatus::UNSUPPORTED:
            return "unsupported";
    }
    return "unknown";
}

const char*
StageName(TuningStage stage) {
    switch (stage) {
        case TuningStage::WORKLOAD_VALIDATION:
            return "workload_validation";
        case TuningStage::SEARCH_SPACE_CONSTRUCTION:
            return "search_space_construction";
        case TuningStage::BUILD_PARAMETER_TUNING:
            return "build_parameter_tuning";
        case TuningStage::QUANTIZER_TUNING:
            return "quantizer_tuning";
        case TuningStage::CANDIDATE_GENERATION:
            return "candidate_generation";
        case TuningStage::CANDIDATE_PRUNING:
            return "candidate_pruning";
        case TuningStage::TRIAL_PLANNING:
            return "trial_planning";
        case TuningStage::TRIAL_EXECUTION:
            return "trial_execution";
        case TuningStage::SELECTION:
            return "selection";
    }
    return "unknown";
}

const char*
StageStatusName(TuningStageStatus status) {
    switch (status) {
        case TuningStageStatus::COMPLETED:
            return "completed";
        case TuningStageStatus::SKIPPED:
            return "skipped";
        case TuningStageStatus::FAILED:
            return "failed";
    }
    return "unknown";
}

const char*
TrialStatusName(TuningTrialStatus status) {
    switch (status) {
        case TuningTrialStatus::COMPLETED:
            return "completed";
        case TuningTrialStatus::SKIPPED:
            return "skipped";
        case TuningTrialStatus::FAILED:
            return "failed";
    }
    return "unknown";
}

const char*
EvaluationStatusName(EvaluationStatus status) {
    switch (status) {
        case EvaluationStatus::SUCCESS:
            return "success";
        case EvaluationStatus::INVALID_ARGUMENT:
            return "invalid_argument";
        case EvaluationStatus::SEARCH_ERROR:
            return "search_error";
    }
    return "unknown";
}

JsonType
EfSearchPatch(uint64_t ef_search) {
    return JsonType{{"hgraph", JsonType{{"ef_search", ef_search}}}};
}

JsonType
CandidateToJson(const EfSearchCandidate& candidate) {
    return JsonType{{"hgraph.ef_search", candidate.ef_search}};
}

JsonType
EvaluationToJson(const EvaluationResult& evaluation) {
    return JsonType{{"status", EvaluationStatusName(evaluation.status)},
                    {"error_message", evaluation.error_message},
                    {"query_count", evaluation.query_count},
                    {"recall",
                     JsonType{{"average", evaluation.recall.average},
                              {"p0", evaluation.recall.p0},
                              {"p10", evaluation.recall.p10},
                              {"p30", evaluation.recall.p30},
                              {"p50", evaluation.recall.p50},
                              {"p70", evaluation.recall.p70},
                              {"p90", evaluation.recall.p90}}},
                    {"latency",
                     JsonType{{"average_ms", evaluation.latency.average_ms},
                              {"p50_ms", evaluation.latency.p50_ms},
                              {"p90_ms", evaluation.latency.p90_ms},
                              {"p95_ms", evaluation.latency.p95_ms},
                              {"p99_ms", evaluation.latency.p99_ms}}},
                    {"qps", evaluation.qps},
                    {"memory_bytes", evaluation.memory_bytes}};
}

JsonType
TrialToJson(const EfSearchTrialResult& trial) {
    return JsonType{{"trial_id", trial.trial_id},
                    {"candidate", CandidateToJson(trial.candidate)},
                    {"search_parameters_patch", EfSearchPatch(trial.candidate.ef_search)},
                    {"status", TrialStatusName(trial.status)},
                    {"message", trial.message},
                    {"evaluation", EvaluationToJson(trial.evaluation)}};
}

AutoTuningApiParseResult
ParseVersion(const JsonType& root) {
    if (not root.contains("version")) {
        return {};
    }

    uint64_t version = 0;
    if (not ReadUint64Value(root.at("version"), version)) {
        return InvalidArgument("invalid_field", "$.version must be a uint64");
    }
    if (version != 1) {
        return Unsupported("unsupported_version",
                           "only auto tuning request version 1 is supported");
    }
    return {};
}

AutoTuningApiParseResult
ParseUnsupportedRootControls(const JsonType& root) {
    auto result = RejectPresentField(
        root, "budget", "unsupported_budget", "P0 auto tuning does not support budget controls");
    if (not result.Succeeded()) {
        return result;
    }
    return RejectPresentField(
        root, "output", "unsupported_output", "P0 auto tuning does not support output controls");
}

AutoTuningApiParseResult
ParseSource(const JsonType& root) {
    const JsonType* source = nullptr;
    auto result = ReadRequiredObject(root, "source", "$", source);
    if (not result.Succeeded()) {
        return result;
    }

    std::string source_type;
    result = ReadRequiredString(*source, "type", "$.source", source_type);
    if (not result.Succeeded()) {
        return result;
    }
    if (source_type != "existing_index") {
        return Unsupported("unsupported_source_type",
                           "P0 auto tuning only supports source.type = existing_index");
    }
    return {};
}

AutoTuningApiParseResult
ParseIndexType(const JsonType& root, AutoTuningRequest& request) {
    std::string index_type;
    auto result = ReadRequiredString(root, "index_type", "$", index_type);
    if (not result.Succeeded()) {
        return result;
    }
    if (index_type != "hgraph") {
        return Unsupported("unsupported_index_type",
                           "P0 auto tuning only supports index_type = hgraph");
    }
    request.index_name = index_type;
    return {};
}

AutoTuningApiParseResult
ParseWorkload(const JsonType& root, AutoTuningRequest& request) {
    const JsonType* workload = nullptr;
    auto result = ReadRequiredObject(root, "workload", "$", workload);
    if (not result.Succeeded()) {
        return result;
    }

    result = ReadRequiredUint64(*workload, "topk", "$.workload", request.topk);
    if (not result.Succeeded()) {
        return result;
    }
    if (request.topk == 0) {
        return InvalidArgument("invalid_topk", "$.workload.topk must be greater than 0");
    }
    if (request.topk > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return InvalidArgument("invalid_topk", "$.workload.topk is too large");
    }
    return {};
}

AutoTuningApiParseResult
ParseConfig(const JsonType& root, AutoTuningRequest& request) {
    const JsonType* config = nullptr;
    auto result = ReadRequiredObject(root, "config", "$", config);
    if (not result.Succeeded()) {
        return result;
    }

    result = ValidateUnsupportedObject(*config,
                                       "build_parameters",
                                       "unsupported_config",
                                       "P0 auto tuning does not consume config.build_parameters");
    if (not result.Succeeded()) {
        return result;
    }

    const JsonType* search_parameters = nullptr;
    result = ReadRequiredObject(*config, "search_parameters", "$.config", search_parameters);
    if (not result.Succeeded()) {
        return result;
    }
    request.base_search_parameters = search_parameters->dump();
    return {};
}

AutoTuningApiParseResult
ParseObjective(const JsonType& root, AutoTuningRequest& request) {
    const JsonType* objective = nullptr;
    auto result = ReadRequiredObject(root, "objective", "$", objective);
    if (not result.Succeeded()) {
        return result;
    }
    if (objective->contains("primary") && not objective->at("primary").is_string()) {
        return InvalidArgument("invalid_field", "$.objective.primary must be a string");
    }
    if (objective->contains("primary") &&
        objective->at("primary").get<std::string>() != "latency") {
        return Unsupported("unsupported_objective",
                           "P0 auto tuning only supports objective.primary = latency");
    }

    const JsonType* recall = nullptr;
    result = ReadRequiredObject(*objective, "recall_at_k", "$.objective", recall);
    if (not result.Succeeded()) {
        return result;
    }
    result = ReadRequiredDouble(*recall, "min", "$.objective.recall_at_k", request.target_recall);
    if (not result.Succeeded()) {
        return result;
    }
    if (request.target_recall < 0.0 || request.target_recall > 1.0) {
        return InvalidArgument("invalid_objective",
                               "$.objective.recall_at_k.min must be within [0, 1]");
    }
    if (objective->contains("constraints") && not objective->at("constraints").is_null()) {
        return Unsupported("unsupported_objective",
                           "P0 auto tuning does not support objective.constraints");
    }
    return {};
}

AutoTuningApiParseResult
ParseSearchSpace(const JsonType& root, AutoTuningRequest& request) {
    const JsonType* search_space = nullptr;
    auto result = ReadRequiredObject(root, "search_space", "$", search_space);
    if (not result.Succeeded()) {
        return result;
    }

    result = ValidateUnsupportedObject(*search_space,
                                       "build",
                                       "unsupported_search_space",
                                       "P0 auto tuning does not support build search space");
    if (not result.Succeeded()) {
        return result;
    }
    result = ValidateUnsupportedObject(*search_space,
                                       "quantizer",
                                       "unsupported_search_space",
                                       "P0 auto tuning does not support quantizer search space");
    if (not result.Succeeded()) {
        return result;
    }

    const JsonType* search = nullptr;
    result = ReadRequiredObject(*search_space, "search", "$.search_space", search);
    if (not result.Succeeded()) {
        return result;
    }
    if (search->size() != 1 || not search->contains("hgraph.ef_search")) {
        return Unsupported("unsupported_parameter",
                           "P0 auto tuning only supports search_space.search.hgraph.ef_search");
    }

    const auto& ef_search = search->at("hgraph.ef_search");
    if (not ef_search.is_object()) {
        return InvalidArgument("invalid_field",
                               "$.search_space.search.hgraph.ef_search must be an object");
    }
    if (not ef_search.contains("values") || not ef_search.at("values").is_array()) {
        return InvalidArgument("invalid_field",
                               "$.search_space.search.hgraph.ef_search.values must be an array");
    }
    if (ef_search.at("values").empty()) {
        return InvalidArgument("invalid_search_space",
                               "$.search_space.search.hgraph.ef_search.values must not be empty");
    }

    for (const auto& value : ef_search.at("values")) {
        uint64_t candidate = 0;
        if (not ReadUint64Value(value, candidate)) {
            return InvalidArgument(
                "invalid_field",
                "$.search_space.search.hgraph.ef_search.values must contain uint64 values");
        }
        request.ef_search_candidates.push_back(candidate);
    }
    return {};
}

AutoTuningApiParseResult
ParseEvaluation(const JsonType& root, AutoTuningRequest& request) {
    const JsonType* evaluation = nullptr;
    auto result = ReadRequiredObject(root, "evaluation", "$", evaluation);
    if (not result.Succeeded()) {
        return result;
    }

    if (evaluation->contains("query_count")) {
        result =
            ReadRequiredUint64(*evaluation, "query_count", "$.evaluation", request.query_count);
        if (not result.Succeeded()) {
            return result;
        }
    }
    result = RejectPresentField(*evaluation,
                                "warmup_query_count",
                                "unsupported_evaluation_option",
                                "P0 auto tuning does not support warmup_query_count");
    if (not result.Succeeded()) {
        return result;
    }
    if (evaluation->contains("successive_halving") &&
        not evaluation->at("successive_halving").is_null()) {
        if (not evaluation->at("successive_halving").is_object()) {
            return InvalidArgument("invalid_field",
                                   "$.evaluation.successive_halving must be an object");
        }
        const auto& successive_halving = evaluation->at("successive_halving");
        if (successive_halving.contains("enabled")) {
            if (not successive_halving.at("enabled").is_boolean()) {
                return InvalidArgument("invalid_field",
                                       "$.evaluation.successive_halving.enabled must be a boolean");
            }
            if (successive_halving.at("enabled").get<bool>()) {
                return Unsupported("unsupported_evaluation_strategy",
                                   "P0 auto tuning does not support successive halving");
            }
        }
    }
    return {};
}

}  // namespace

AutoTuningApiParseResult
ParseAutoTuningRequestJson(const std::string& request_json, const AutoTuningApiContext& context) {
    JsonType root;
    try {
        root = JsonType::parse(request_json);
    } catch (const std::exception& e) {
        return InvalidArgument("invalid_json",
                               std::string("failed to parse request json: ") + e.what());
    }

    if (not root.is_object()) {
        return InvalidArgument("invalid_request", "auto tuning request must be a json object");
    }

    AutoTuningApiParseResult result;
    result.request.index = context.index;
    result.request.queries = context.queries;
    result.request.ground_truth = context.ground_truth;

    auto parse_result = ParseVersion(root);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    parse_result = ParseUnsupportedRootControls(root);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    parse_result = ParseIndexType(root, result.request);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    parse_result = ParseSource(root);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    parse_result = ParseWorkload(root, result.request);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    parse_result = ParseConfig(root, result.request);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    parse_result = ParseObjective(root, result.request);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    parse_result = ParseSearchSpace(root, result.request);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    parse_result = ParseEvaluation(root, result.request);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }

    return result;
}

std::string
SerializeAutoTuningReportJson(const AutoTuningReport& report) {
    JsonType stages = JsonType::array();
    for (const auto& stage : report.stages) {
        stages.push_back(JsonType{{"stage", StageName(stage.stage)},
                                  {"status", StageStatusName(stage.status)},
                                  {"message", stage.message},
                                  {"input_count", stage.input_count},
                                  {"output_count", stage.output_count}});
    }

    JsonType trials = JsonType::array();
    for (const auto& trial : report.ef_search.trials) {
        trials.push_back(TrialToJson(trial));
    }

    JsonType root{{"version", 1},
                  {"succeeded", report.Succeeded()},
                  {"status", report.Succeeded() ? "succeeded" : "failed"},
                  {"elapsed_ms", report.elapsed_ms},
                  {"stages", stages},
                  {"trials", trials}};

    root["recommendation"] =
        report.recommendation.has_value() ? TrialToJson(report.recommendation.value()) : JsonType();
    root["best_effort"] =
        report.best_effort.has_value() ? TrialToJson(report.best_effort.value()) : JsonType();
    return root.dump(2);
}

std::string
SerializeAutoTuningErrorJson(const AutoTuningApiParseResult& result) {
    JsonType root{{"version", 1},
                  {"succeeded", false},
                  {"status", "failed"},
                  {"error",
                   JsonType{{"type", ApiStatusName(result.status)},
                            {"code", result.error_code},
                            {"message", result.error_message}}}};
    return root.dump(2);
}

}  // namespace vsag
