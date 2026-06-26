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
#include <utility>

#include "vsag/factory.h"

namespace vsag {
namespace {

using JsonType = nlohmann::json;

enum class SourceType {
    EXISTING_INDEX,
    RAW_DATASET,
};

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
SourceTypeName(SourceType source_type) {
    switch (source_type) {
        case SourceType::EXISTING_INDEX:
            return "existing_index";
        case SourceType::RAW_DATASET:
            return "raw_dataset";
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
        case TuningStage::CANDIDATE_VALIDATION:
            return "candidate_validation";
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
ParseJsonForReport(const std::string& value) {
    if (value.empty()) {
        return JsonType::object();
    }
    try {
        return JsonType::parse(value);
    } catch (const std::exception&) {
        return value;
    }
}

JsonType
CandidateToJson(const TuningCandidateReport& candidate) {
    JsonType result = JsonType::object();
    for (const auto& patch : candidate.patches) {
        result[patch.path] = ParseJsonForReport(patch.value);
    }
    if (not result.contains("hgraph.ef_search") && candidate.ef_search > 0) {
        result["hgraph.ef_search"] = candidate.ef_search;
    }
    return result;
}

JsonType
SearchPatchToJson(const TuningCandidateReport& candidate) {
    JsonType result = JsonType::object();
    for (const auto& patch : candidate.patches) {
        if (patch.path == "hgraph.ef_search") {
            result["hgraph"]["ef_search"] = ParseJsonForReport(patch.value);
        }
    }
    if (not result.contains("hgraph") && candidate.ef_search > 0) {
        result["hgraph"]["ef_search"] = candidate.ef_search;
    }
    return result;
}

JsonType
ParameterSpacesToJson(const std::vector<TuningParameterSpace>& parameter_spaces) {
    JsonType result = JsonType::object();
    for (const auto& parameter_space : parameter_spaces) {
        JsonType values = JsonType::array();
        for (const auto& value : parameter_space.values) {
            values.push_back(ParseJsonForReport(value));
        }
        result[parameter_space.path] = JsonType{{"values", values}};
    }
    return result;
}

JsonType
SourceCapabilitiesToJson(const TuningSourceCapabilities& capabilities) {
    return JsonType{{"has_index", capabilities.has_index},
                    {"has_base", capabilities.has_base},
                    {"has_build_parameters", capabilities.has_build_parameters},
                    {"supports_search_tuning", capabilities.supports_search_tuning},
                    {"supports_rebuild_tuning", capabilities.supports_rebuild_tuning}};
}

JsonType
RequestSummaryToJson(const AutoTuningRequestSummary& request) {
    JsonType search_space = JsonType::object();
    if (not request.build_parameter_spaces.empty()) {
        search_space["build"] = ParameterSpacesToJson(request.build_parameter_spaces);
    }
    if (not request.quantizer_parameter_spaces.empty()) {
        search_space["quantizer"] = ParameterSpacesToJson(request.quantizer_parameter_spaces);
    }
    if (not request.search_parameter_spaces.empty()) {
        search_space["search"] = ParameterSpacesToJson(request.search_parameter_spaces);
    } else {
        search_space["search"] =
            JsonType{{"hgraph.ef_search", JsonType{{"values", request.ef_search_candidates}}}};
    }

    return JsonType{
        {"index_type", request.index_name},
        {"source",
         JsonType{{"type", request.source_type},
                  {"capabilities", SourceCapabilitiesToJson(request.source_capabilities)}}},
        {"workload", JsonType{{"topk", request.topk}}},
        {"config",
         JsonType{{"build_parameters",
                   request.build_parameters.empty() ? JsonType()
                                                    : ParseJsonForReport(request.build_parameters)},
                  {"search_parameters", ParseJsonForReport(request.base_search_parameters)}}},
        {"objective",
         JsonType{{"primary", "latency"},
                  {"recall_at_k", JsonType{{"min", request.target_recall}}}}},
        {"search_space", search_space},
        {"evaluation",
         JsonType{
             {"query_count", request.requested_query_count},
             {"effective_query_count", request.effective_query_count},
             {"successive_halving", JsonType{{"enabled", request.enable_successive_halving}}}}},
        {"budget", JsonType{{"max_trials", request.max_trials}}}};
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
TrialToJson(const TuningTrialResult& trial) {
    return JsonType{{"trial_id", trial.trial_id},
                    {"candidate", CandidateToJson(trial.candidate)},
                    {"parameters_patch", CandidateToJson(trial.candidate)},
                    {"search_parameters_patch", SearchPatchToJson(trial.candidate)},
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
    return RejectPresentField(
        root, "output", "unsupported_output", "P0 auto tuning does not support output controls");
}

AutoTuningApiParseResult
ParseSource(const JsonType& root, SourceType& source_type) {
    const JsonType* source = nullptr;
    auto result = ReadRequiredObject(root, "source", "$", source);
    if (not result.Succeeded()) {
        return result;
    }

    std::string source_type_name;
    result = ReadRequiredString(*source, "type", "$.source", source_type_name);
    if (not result.Succeeded()) {
        return result;
    }
    if (source_type_name == "existing_index") {
        source_type = SourceType::EXISTING_INDEX;
        return {};
    }
    if (source_type_name == "raw_dataset") {
        source_type = SourceType::RAW_DATASET;
        return {};
    }
    return Unsupported("unsupported_source_type",
                       "P0 auto tuning only supports existing_index or raw_dataset source");
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
ParseConfig(const JsonType& root, SourceType source_type, AutoTuningRequest& request) {
    const JsonType* config = nullptr;
    auto result = ReadRequiredObject(root, "config", "$", config);
    if (not result.Succeeded()) {
        return result;
    }

    if (source_type == SourceType::EXISTING_INDEX) {
        if (config->contains("build_parameters") && not config->at("build_parameters").is_null()) {
            if (not config->at("build_parameters").is_object()) {
                return InvalidArgument("invalid_field",
                                       "$.config.build_parameters must be an object");
            }
            request.build_parameters = config->at("build_parameters").dump();
        }
    } else {
        const JsonType* build_parameters = nullptr;
        result = ReadRequiredObject(*config, "build_parameters", "$.config", build_parameters);
        if (not result.Succeeded()) {
            return result;
        }
        if (build_parameters->empty()) {
            return InvalidArgument("invalid_field",
                                   "$.config.build_parameters must not be empty for raw_dataset");
        }
        request.build_parameters = build_parameters->dump();
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
ValidateTuningSearchSpaceGroup(const JsonType& search_space,
                               const std::string& group_name,
                               const std::string& index_name,
                               std::vector<TuningParameterSpace>& parameter_spaces,
                               bool& enabled) {
    enabled = false;
    if (not search_space.contains(group_name) || search_space.at(group_name).is_null()) {
        return {};
    }
    if (not search_space.at(group_name).is_object()) {
        return InvalidArgument("invalid_field",
                               "$.search_space." + group_name + " must be an object");
    }

    const auto prefix = index_name + ".";
    for (const auto& item : search_space.at(group_name).items()) {
        if (item.value().is_null()) {
            continue;
        }
        if (item.key().rfind(prefix, 0) != 0) {
            return Unsupported("unsupported_parameter",
                               "P0 auto tuning only supports " + index_name + " parameter paths");
        }
        if (not item.value().is_object()) {
            return InvalidArgument(
                "invalid_field",
                "$.search_space." + group_name + "." + item.key() + " must be an object");
        }
        if (group_name == "build" && item.key() != index_name + ".max_degree" &&
            item.key() != index_name + ".ef_construction") {
            return Unsupported("unsupported_parameter",
                               "P0 build tuning only supports " + index_name + ".max_degree and " +
                                   index_name + ".ef_construction");
        }
        if (group_name == "quantizer" && item.key() != index_name + ".base_quantization_type") {
            return Unsupported(
                "unsupported_parameter",
                "P0 quantizer tuning only supports " + index_name + ".base_quantization_type");
        }
        if (not item.value().contains("values") || not item.value().at("values").is_array()) {
            return InvalidArgument(
                "invalid_field",
                "$.search_space." + group_name + "." + item.key() + ".values must be an array");
        }
        if (item.value().at("values").empty()) {
            return InvalidArgument(
                "invalid_search_space",
                "$.search_space." + group_name + "." + item.key() + ".values must not be empty");
        }

        TuningParameterSpace parameter_space;
        parameter_space.path = item.key();
        for (const auto& value : item.value().at("values")) {
            if (group_name == "build") {
                uint64_t parsed = 0;
                if (not ReadUint64Value(value, parsed)) {
                    return InvalidArgument("invalid_field",
                                           "$.search_space." + group_name + "." + item.key() +
                                               ".values must contain uint64 values");
                }
            }
            if (group_name == "quantizer" && not value.is_string()) {
                return InvalidArgument("invalid_field",
                                       "$.search_space." + group_name + "." + item.key() +
                                           ".values must contain string values");
            }
            parameter_space.values.push_back(value.dump());
        }
        parameter_spaces.push_back(std::move(parameter_space));
        enabled = true;
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

    bool has_build_space = false;
    result = ValidateTuningSearchSpaceGroup(*search_space,
                                            "build",
                                            request.index_name,
                                            request.build_parameter_spaces,
                                            has_build_space);
    if (not result.Succeeded()) {
        return result;
    }
    request.enable_build_parameter_tuning = has_build_space;

    bool has_quantizer_space = false;
    result = ValidateTuningSearchSpaceGroup(*search_space,
                                            "quantizer",
                                            request.index_name,
                                            request.quantizer_parameter_spaces,
                                            has_quantizer_space);
    if (not result.Succeeded()) {
        return result;
    }
    request.enable_quantizer_tuning = has_quantizer_space;

    const JsonType* search = nullptr;
    const auto has_search_space =
        search_space->contains("search") && not search_space->at("search").is_null();
    if (not has_search_space) {
        if (has_build_space || has_quantizer_space) {
            return {};
        }
        return InvalidArgument("missing_field",
                               "$.search_space.search is required unless build or quantizer "
                               "search space is provided");
    }
    if (not search_space->at("search").is_object()) {
        return InvalidArgument("invalid_field", "$.search_space.search must be an object");
    }
    search = &search_space->at("search");
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

    TuningParameterSpace search_parameter_space;
    search_parameter_space.path = "hgraph.ef_search";
    for (const auto& value : ef_search.at("values")) {
        uint64_t candidate = 0;
        if (not ReadUint64Value(value, candidate)) {
            return InvalidArgument(
                "invalid_field",
                "$.search_space.search.hgraph.ef_search.values must contain uint64 values");
        }
        request.ef_search_candidates.push_back(candidate);
        search_parameter_space.values.push_back(value.dump());
    }
    request.search_parameter_spaces.push_back(std::move(search_parameter_space));
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
                request.enable_successive_halving = true;
            }
        }
    }
    return {};
}

AutoTuningApiParseResult
ParseBudget(const JsonType& root, AutoTuningRequest& request) {
    if (not root.contains("budget") || root.at("budget").is_null()) {
        return {};
    }
    if (not root.at("budget").is_object()) {
        return InvalidArgument("invalid_field", "$.budget must be an object");
    }

    const auto& budget = root.at("budget");
    if (budget.contains("max_trials")) {
        auto result = ReadRequiredUint64(budget, "max_trials", "$.budget", request.max_trials);
        if (not result.Succeeded()) {
            return result;
        }
    }

    for (const auto& item : budget.items()) {
        if (item.key() == "max_trials" || item.value().is_null()) {
            continue;
        }
        return Unsupported("unsupported_budget", "P0 auto tuning only supports budget.max_trials");
    }
    return {};
}

AutoTuningApiParseResult
PrepareSource(const AutoTuningApiContext& context, AutoTuningRequest& request) {
    if (context.queries == nullptr) {
        return InvalidArgument("invalid_context", "queries are required in AutoTuningApiContext");
    }
    if (context.ground_truth == nullptr) {
        return InvalidArgument("invalid_context",
                               "ground_truth is required in AutoTuningApiContext");
    }

    request.queries = context.queries;
    request.ground_truth = context.ground_truth;
    request.base = context.base;

    if (request.source_type == "existing_index") {
        if (context.index == nullptr) {
            return InvalidArgument("invalid_context",
                                   "index is required for source.type = existing_index");
        }
        request.index = context.index;
        return {};
    }

    if (request.source_type != "raw_dataset") {
        return InvalidArgument("invalid_source_type", "prepared request source_type is invalid");
    }
    if (context.base == nullptr) {
        return InvalidArgument("invalid_context", "base is required for source.type = raw_dataset");
    }

    auto index = Factory::CreateIndex(request.index_name, request.build_parameters);
    if (not index.has_value()) {
        return InvalidArgument("build_index_error",
                               "failed to create baseline index: " + index.error().message);
    }

    auto build_result = index.value()->Build(context.base);
    if (not build_result.has_value()) {
        return InvalidArgument("build_index_error",
                               "failed to build baseline index: " + build_result.error().message);
    }

    request.index = index.value();
    return {};
}

}  // namespace

AutoTuningApiParseResult
ParseAutoTuningRequestJson(const std::string& request_json) {
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
    SourceType source_type = SourceType::EXISTING_INDEX;

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
    parse_result = ParseSource(root, source_type);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    result.request.source_type = SourceTypeName(source_type);
    parse_result = ParseWorkload(root, result.request);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    parse_result = ParseConfig(root, source_type, result.request);
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
    parse_result = ParseBudget(root, result.request);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }

    return result;
}

AutoTuningApiParseResult
PrepareAutoTuningRequest(const AutoTuningRequest& parsed_request,
                         const AutoTuningApiContext& context) {
    AutoTuningApiParseResult result;
    result.request = parsed_request;

    auto parse_result = PrepareSource(context, result.request);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }

    return result;
}

AutoTuningApiParseResult
PrepareAutoTuningRequestJson(const std::string& request_json, const AutoTuningApiContext& context) {
    auto parse_result = ParseAutoTuningRequestJson(request_json);
    if (not parse_result.Succeeded()) {
        return parse_result;
    }
    return PrepareAutoTuningRequest(parse_result.request, context);
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
    for (const auto& trial : report.trial_report.trials) {
        trials.push_back(TrialToJson(trial));
    }

    JsonType root{{"version", 1},
                  {"succeeded", report.Succeeded()},
                  {"status", report.Succeeded() ? "succeeded" : "failed"},
                  {"request", RequestSummaryToJson(report.request)},
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
