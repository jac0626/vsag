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

#include "tuning/tuning_pipeline.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <utility>

#include "common.h"
#include "vsag/factory.h"

namespace vsag {
namespace {

TuningStageResult
MakeStage(TuningStage stage,
          TuningStageStatus status,
          const std::string& message,
          uint64_t input_count = 0,
          uint64_t output_count = 0) {
    TuningStageResult result;
    result.stage = stage;
    result.status = status;
    result.message = message;
    result.input_count = input_count;
    result.output_count = output_count;
    return result;
}

bool
HasFailedStage(const std::vector<TuningStageResult>& stages) {
    return std::any_of(stages.begin(), stages.end(), [](const TuningStageResult& stage) {
        return stage.status == TuningStageStatus::FAILED;
    });
}

uint64_t
CountTrials(const TuningTrialReport& report, TuningTrialStatus status) {
    return static_cast<uint64_t>(
        std::count_if(report.trials.begin(), report.trials.end(), [status](const auto& trial) {
            return trial.status == status;
        }));
}

TuningSourceCapabilities
MakeSourceCapabilities(const AutoTuningRequest& request) {
    TuningSourceCapabilities capabilities;
    capabilities.has_index = request.index != nullptr;
    capabilities.has_base = request.base != nullptr;
    capabilities.has_build_parameters = not request.build_parameters.empty();
    capabilities.supports_search_tuning = capabilities.has_index;
    capabilities.supports_rebuild_tuning =
        capabilities.has_base && capabilities.has_build_parameters;
    return capabilities;
}

std::vector<uint64_t>
NormalizeEfSearchCandidates(std::vector<uint64_t> candidates) {
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

uint64_t
MaxEfSearch(uint64_t topk) {
    if (topk > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(AMPLIFICATION_FACTOR)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return std::max<uint64_t>(static_cast<uint64_t>(AMPLIFICATION_FACTOR) * topk, 1000);
}

bool
ReadUint64CandidateValue(const std::string& value, uint64_t& output) {
    try {
        const auto json_value = nlohmann::json::parse(value);
        if (json_value.is_number_unsigned()) {
            output = json_value.get<uint64_t>();
            return true;
        }
        if (json_value.is_number_integer()) {
            const auto signed_value = json_value.get<int64_t>();
            if (signed_value < 0) {
                return false;
            }
            output = static_cast<uint64_t>(signed_value);
            return true;
        }
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

bool
ReadStringCandidateValue(const std::string& value, std::string& output) {
    try {
        const auto json_value = nlohmann::json::parse(value);
        if (json_value.is_string()) {
            output = json_value.get<std::string>();
            return true;
        }
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

bool
ReadEfSearchPatch(const std::vector<TuningParameterPatch>& patches,
                  bool& has_ef_search,
                  uint64_t& ef_search,
                  std::string& error_message) {
    has_ef_search = false;
    ef_search = 0;
    for (const auto& patch : patches) {
        if (patch.path != "hgraph.ef_search") {
            continue;
        }
        has_ef_search = true;
        if (not ReadUint64CandidateValue(patch.value, ef_search)) {
            error_message = "hgraph.ef_search candidate value must be a uint64";
            return false;
        }
    }
    return true;
}

bool
MakeSearchParameters(const std::string& base_search_parameters,
                     const std::string& index_name,
                     uint64_t ef_search,
                     std::string& search_parameters,
                     std::string& error_message) {
    try {
        auto parameters = base_search_parameters.empty()
                              ? nlohmann::json::object()
                              : nlohmann::json::parse(base_search_parameters);
        if (not parameters.is_object()) {
            error_message = "base search parameters must be a json object";
            return false;
        }
        if (not parameters.contains(index_name) || not parameters[index_name].is_object()) {
            parameters[index_name] = nlohmann::json::object();
        }
        parameters[index_name]["ef_search"] = ef_search;
        search_parameters = parameters.dump();
        return true;
    } catch (const std::exception& e) {
        error_message = fmt::format("failed to parse base search parameters: {}", e.what());
        return false;
    }
}

bool
IsLowerCostTrial(const TuningTrialResult& lhs, const TuningTrialResult& rhs) {
    if (lhs.evaluation.latency.average_ms != rhs.evaluation.latency.average_ms) {
        return lhs.evaluation.latency.average_ms < rhs.evaluation.latency.average_ms;
    }
    if (lhs.candidate.ef_search != rhs.candidate.ef_search) {
        return lhs.candidate.ef_search < rhs.candidate.ef_search;
    }
    return lhs.trial_id < rhs.trial_id;
}

bool
IsBetterBestEffort(const TuningTrialResult& lhs, const TuningTrialResult& rhs) {
    if (lhs.evaluation.recall.average != rhs.evaluation.recall.average) {
        return lhs.evaluation.recall.average > rhs.evaluation.recall.average;
    }
    return IsLowerCostTrial(lhs, rhs);
}

double
ElapsedMs(std::chrono::steady_clock::time_point started_at) {
    const auto elapsed = std::chrono::steady_clock::now() - started_at;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

EvaluationResult
ValidateWorkload(const AutoTuningRequest& request) {
    EvaluationResult result;
    if (request.index == nullptr) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "index is required";
        return result;
    }
    if (request.index_name != "hgraph") {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "P0 auto tuning only supports hgraph index_name";
        return result;
    }
    try {
        if (request.index->GetIndexType() != IndexType::HGRAPH) {
            result.status = EvaluationStatus::INVALID_ARGUMENT;
            result.error_message = "P0 auto tuning only supports HGraph index";
            return result;
        }
    } catch (const std::exception& e) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = std::string("failed to read index type: ") + e.what();
        return result;
    }
    if (request.queries == nullptr) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "queries are required";
        return result;
    }
    if (request.ground_truth == nullptr) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "ground truth is required";
        return result;
    }
    if (request.topk == 0) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "topk must be greater than 0";
        return result;
    }
    if (request.topk > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "topk is too large";
        return result;
    }
    if (request.queries->GetNumElements() <= 0) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "queries must not be empty";
        return result;
    }
    if (request.queries->GetFloat32Vectors() == nullptr) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "float32 query vectors are required";
        return result;
    }
    if (request.ground_truth->GetIds() == nullptr) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "ground truth ids are required";
        return result;
    }
    if (request.ground_truth->GetDim() < static_cast<int64_t>(request.topk)) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "ground truth dim must cover topk";
        return result;
    }

    const auto total_queries = static_cast<uint64_t>(request.queries->GetNumElements());
    const auto query_count =
        request.query_count == 0 ? total_queries : std::min(request.query_count, total_queries);
    if (request.ground_truth->GetNumElements() < static_cast<int64_t>(query_count)) {
        result.status = EvaluationStatus::INVALID_ARGUMENT;
        result.error_message = "ground truth query count is too small";
        return result;
    }

    result.query_count = query_count;
    return result;
}

AutoTuningRequestSummary
MakeRequestSummary(const AutoTuningRequest& request) {
    AutoTuningRequestSummary summary;
    summary.index_name = request.index_name;
    summary.source_type = request.source_type;
    summary.source_capabilities = MakeSourceCapabilities(request);
    summary.topk = request.topk;
    summary.requested_query_count = request.query_count;
    summary.target_recall = request.target_recall;
    summary.build_parameters = request.build_parameters;
    summary.base_search_parameters = request.base_search_parameters;
    summary.ef_search_candidates = request.ef_search_candidates;
    summary.build_parameter_spaces = request.build_parameter_spaces;
    summary.quantizer_parameter_spaces = request.quantizer_parameter_spaces;
    summary.search_parameter_spaces = request.search_parameter_spaces;
    summary.max_trials = request.max_trials;
    summary.enable_build_parameter_tuning = request.enable_build_parameter_tuning;
    summary.enable_quantizer_tuning = request.enable_quantizer_tuning;
    summary.enable_successive_halving = request.enable_successive_halving;
    return summary;
}

std::vector<TuningParameterSpace>
MakeEfSearchParameterSpaces(const AutoTuningRequest& request) {
    if (not request.search_parameter_spaces.empty()) {
        return request.search_parameter_spaces;
    }

    TuningParameterSpace parameter_space;
    parameter_space.path = "hgraph.ef_search";
    for (const auto ef_search : request.ef_search_candidates) {
        parameter_space.values.push_back(std::to_string(ef_search));
    }
    if (parameter_space.values.empty()) {
        return {};
    }
    return {parameter_space};
}

void
AssignCandidateIds(std::vector<TuningCandidate>& candidates) {
    uint64_t candidate_id = 0;
    for (auto& candidate : candidates) {
        candidate.id = candidate_id++;
    }
}

void
ExpandCandidates(std::vector<TuningCandidate>& candidates,
                 const std::vector<TuningParameterSpace>& parameter_spaces) {
    for (const auto& parameter_space : parameter_spaces) {
        std::vector<TuningCandidate> expanded_candidates;
        for (const auto& candidate : candidates) {
            for (const auto& value : parameter_space.values) {
                auto expanded = candidate;
                expanded.patches.push_back({parameter_space.path, value});
                expanded_candidates.push_back(std::move(expanded));
            }
        }
        candidates = std::move(expanded_candidates);
    }
    AssignCandidateIds(candidates);
}

uint64_t
CandidateCount(const TuningState& state) {
    return static_cast<uint64_t>(state.candidates.size());
}

std::vector<uint64_t>
ExtractEfSearchCandidates(const std::vector<TuningCandidate>& candidates) {
    std::vector<uint64_t> ef_search_candidates;
    for (const auto& candidate : candidates) {
        for (const auto& patch : candidate.patches) {
            if (patch.path != "hgraph.ef_search") {
                continue;
            }
            uint64_t ef_search = 0;
            if (ReadUint64CandidateValue(patch.value, ef_search)) {
                ef_search_candidates.push_back(ef_search);
            }
        }
    }
    return NormalizeEfSearchCandidates(std::move(ef_search_candidates));
}

bool
ApplyHGraphBuildPatch(nlohmann::json& parameters,
                      const TuningParameterPatch& patch,
                      std::string& error_message) {
    if (patch.path == "hgraph.ef_search") {
        return true;
    }
    if (not parameters.contains("index_param") || not parameters["index_param"].is_object()) {
        parameters["index_param"] = nlohmann::json::object();
    }

    if (patch.path == "hgraph.max_degree" || patch.path == "hgraph.ef_construction") {
        uint64_t value = 0;
        if (not ReadUint64CandidateValue(patch.value, value)) {
            error_message = fmt::format("{} candidate value must be a uint64", patch.path);
            return false;
        }
        const auto key = patch.path == "hgraph.max_degree" ? "max_degree" : "ef_construction";
        parameters["index_param"][key] = value;
        return true;
    }

    if (patch.path == "hgraph.base_quantization_type") {
        std::string value;
        if (not ReadStringCandidateValue(patch.value, value)) {
            error_message = "hgraph.base_quantization_type candidate value must be a string";
            return false;
        }
        parameters["index_param"]["base_quantization_type"] = value;
        return true;
    }

    error_message = fmt::format("unsupported HGraph tuning parameter path: {}", patch.path);
    return false;
}

bool
MakeCandidateBuildParameters(const std::string& base_build_parameters,
                             const std::vector<TuningParameterPatch>& patches,
                             std::string& build_parameters,
                             std::string& error_message) {
    if (base_build_parameters.empty()) {
        error_message = "build parameters are required for build or quantizer candidate execution";
        return false;
    }

    try {
        auto parameters = nlohmann::json::parse(base_build_parameters);
        if (not parameters.is_object()) {
            error_message = "build parameters must be a json object";
            return false;
        }
        for (const auto& patch : patches) {
            if (not ApplyHGraphBuildPatch(parameters, patch, error_message)) {
                return false;
            }
        }
        build_parameters = parameters.dump();
        return true;
    } catch (const std::exception& e) {
        error_message = fmt::format("failed to parse build parameters: {}", e.what());
        return false;
    }
}

bool
MakeCandidateSearchParameters(const std::string& base_search_parameters,
                              const std::string& index_name,
                              const std::vector<TuningParameterPatch>& patches,
                              std::string& search_parameters,
                              std::string& error_message) {
    try {
        auto parameters = base_search_parameters.empty()
                              ? nlohmann::json::object()
                              : nlohmann::json::parse(base_search_parameters);
        if (not parameters.is_object()) {
            error_message = "base search parameters must be a json object";
            return false;
        }
        if (not parameters.contains(index_name) || not parameters[index_name].is_object()) {
            parameters[index_name] = nlohmann::json::object();
        }
        for (const auto& patch : patches) {
            if (patch.path != "hgraph.ef_search") {
                continue;
            }
            uint64_t value = 0;
            if (not ReadUint64CandidateValue(patch.value, value)) {
                error_message = "hgraph.ef_search candidate value must be a uint64";
                return false;
            }
            parameters[index_name]["ef_search"] = value;
        }
        search_parameters = parameters.dump();
        return true;
    } catch (const std::exception& e) {
        error_message = fmt::format("failed to parse base search parameters: {}", e.what());
        return false;
    }
}

bool
ValidateCandidatePatch(const TuningParameterPatch& patch, std::string& error_message) {
    if (patch.path == "hgraph.ef_search" || patch.path == "hgraph.max_degree" ||
        patch.path == "hgraph.ef_construction") {
        uint64_t value = 0;
        if (ReadUint64CandidateValue(patch.value, value)) {
            return true;
        }
        error_message = fmt::format("{} candidate value must be a uint64", patch.path);
        return false;
    }

    if (patch.path == "hgraph.base_quantization_type") {
        std::string value;
        if (ReadStringCandidateValue(patch.value, value)) {
            return true;
        }
        error_message = "hgraph.base_quantization_type candidate value must be a string";
        return false;
    }

    error_message = fmt::format("unsupported HGraph tuning parameter path: {}", patch.path);
    return false;
}

bool
CandidateHasPatch(const TuningCandidate& candidate, const std::string& path) {
    return std::any_of(candidate.patches.begin(),
                       candidate.patches.end(),
                       [&path](const auto& patch) { return patch.path == path; });
}

bool
ValidateSearchOnlyCandidate(const AutoTuningRequest& request,
                            const TuningCandidate& candidate,
                            std::string& error_message) {
    if (not CandidateHasPatch(candidate, "hgraph.ef_search")) {
        error_message = "search-only tuning requires hgraph.ef_search candidates";
        return false;
    }

    bool has_ef_search = false;
    uint64_t ef_search = 0;
    if (not ReadEfSearchPatch(candidate.patches, has_ef_search, ef_search, error_message)) {
        return false;
    }

    std::string search_parameters;
    return MakeSearchParameters(request.base_search_parameters,
                                request.index_name,
                                ef_search,
                                search_parameters,
                                error_message);
}

bool
ValidateRebuildCandidate(const AutoTuningRequest& request,
                         const TuningCandidate& candidate,
                         std::string& error_message) {
    const auto capabilities = MakeSourceCapabilities(request);
    if (not capabilities.supports_rebuild_tuning) {
        if (not capabilities.has_base && not capabilities.has_build_parameters) {
            error_message =
                "source capability does not support rebuild tuning: base dataset and build "
                "parameters are required";
            return false;
        }
        if (not capabilities.has_base) {
            error_message =
                "source capability does not support rebuild tuning: base dataset is required";
            return false;
        }
        error_message =
            "source capability does not support rebuild tuning: build parameters are required";
        return false;
    }

    std::string build_parameters;
    if (not MakeCandidateBuildParameters(
            request.build_parameters, candidate.patches, build_parameters, error_message)) {
        return false;
    }

    std::string search_parameters;
    return MakeCandidateSearchParameters(request.base_search_parameters,
                                         request.index_name,
                                         candidate.patches,
                                         search_parameters,
                                         error_message);
}

bool
ValidateCandidate(const AutoTuningRequest& request,
                  const TuningCandidate& candidate,
                  std::string& error_message) {
    for (const auto& patch : candidate.patches) {
        if (not ValidateCandidatePatch(patch, error_message)) {
            return false;
        }
    }

    if (request.enable_build_parameter_tuning || request.enable_quantizer_tuning) {
        return ValidateRebuildCandidate(request, candidate, error_message);
    }
    return ValidateSearchOnlyCandidate(request, candidate, error_message);
}

IndexPtr
BuildCandidateIndex(const std::string& index_name,
                    const std::string& build_parameters,
                    const DatasetPtr& base,
                    std::string& error_message) {
    auto index = Factory::CreateIndex(index_name, build_parameters);
    if (not index.has_value()) {
        error_message = "failed to create candidate index: " + index.error().message;
        return nullptr;
    }

    auto build_result = index.value()->Build(base);
    if (not build_result.has_value()) {
        error_message = "failed to build candidate index: " + build_result.error().message;
        return nullptr;
    }
    return index.value();
}

TuningTrialReport
PlanEfSearchTrials(const AutoTuningRequest& request,
                   const std::vector<TuningCandidate>& candidates) {
    TuningTrialReport report;
    const auto ef_search_candidates = ExtractEfSearchCandidates(candidates);
    const auto max_ef_search = MaxEfSearch(request.topk);
    uint64_t trial_id = 0;
    uint64_t attempted_trials = 0;

    for (const auto ef_search : ef_search_candidates) {
        TuningTrialResult trial;
        trial.trial_id = trial_id++;
        trial.candidate.ef_search = ef_search;
        trial.candidate.patches.push_back({"hgraph.ef_search", std::to_string(ef_search)});

        if (ef_search == 0) {
            trial.status = TuningTrialStatus::SKIPPED;
            trial.message = "ef_search must be greater than 0";
            report.trials.push_back(trial);
            continue;
        }
        if (request.topk == 0) {
            trial.status = TuningTrialStatus::SKIPPED;
            trial.message = "topk must be greater than 0";
            report.trials.push_back(trial);
            continue;
        }
        if (ef_search > max_ef_search) {
            trial.status = TuningTrialStatus::SKIPPED;
            trial.message = fmt::format("ef_search must be no greater than {}", max_ef_search);
            report.trials.push_back(trial);
            continue;
        }
        if (request.max_trials > 0 && attempted_trials >= request.max_trials) {
            trial.status = TuningTrialStatus::SKIPPED;
            trial.message = fmt::format("budget exceeded: max_trials = {}", request.max_trials);
            report.trials.push_back(trial);
            continue;
        }

        ++attempted_trials;
        report.trials.push_back(trial);
    }
    return report;
}

TuningTrialReport
PlanFullCandidateTrials(const AutoTuningRequest& request,
                        const std::vector<TuningCandidate>& candidates) {
    TuningTrialReport report;
    const auto max_ef_search = MaxEfSearch(request.topk);
    uint64_t trial_id = 0;
    uint64_t attempted_trials = 0;

    for (const auto& candidate : candidates) {
        TuningTrialResult trial;
        trial.trial_id = trial_id++;
        trial.candidate.patches = candidate.patches;

        bool has_ef_search = false;
        uint64_t ef_search = 0;
        if (not ReadEfSearchPatch(candidate.patches, has_ef_search, ef_search, trial.message)) {
            trial.status = TuningTrialStatus::FAILED;
            report.trials.push_back(trial);
            continue;
        }
        if (has_ef_search) {
            trial.candidate.ef_search = ef_search;
            if (ef_search == 0) {
                trial.status = TuningTrialStatus::SKIPPED;
                trial.message = "ef_search must be greater than 0";
                report.trials.push_back(trial);
                continue;
            }
            if (request.topk == 0) {
                trial.status = TuningTrialStatus::SKIPPED;
                trial.message = "topk must be greater than 0";
                report.trials.push_back(trial);
                continue;
            }
            if (ef_search > max_ef_search) {
                trial.status = TuningTrialStatus::SKIPPED;
                trial.message = fmt::format("ef_search must be no greater than {}", max_ef_search);
                report.trials.push_back(trial);
                continue;
            }
        }
        if (request.max_trials > 0 && attempted_trials >= request.max_trials) {
            trial.status = TuningTrialStatus::SKIPPED;
            trial.message = fmt::format("budget exceeded: max_trials = {}", request.max_trials);
            report.trials.push_back(trial);
            continue;
        }

        ++attempted_trials;
        report.trials.push_back(trial);
    }
    return report;
}

class WorkloadValidationStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::WORKLOAD_VALIDATION;
    }

    void
    Run(TuningState& state, const TuningStageRuntime&) const override {
        auto validation = ValidateWorkload(state.request);
        if (not validation.Succeeded()) {
            state.report.stages.push_back(
                MakeStage(Stage(), TuningStageStatus::FAILED, validation.error_message));
            state.should_stop = true;
            return;
        }
        state.report.request.effective_query_count = validation.query_count;
        state.report.stages.push_back(MakeStage(Stage(),
                                                TuningStageStatus::COMPLETED,
                                                "workload is valid",
                                                validation.query_count,
                                                validation.query_count));
    }
};

class SearchSpaceConstructionStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::SEARCH_SPACE_CONSTRUCTION;
    }

    void
    Run(TuningState& state, const TuningStageRuntime&) const override {
        TuningCandidate candidate;
        candidate.id = 0;
        candidate.index_name = state.request.index_name;
        candidate.source_type = state.request.source_type;
        candidate.build_parameters = state.request.build_parameters;
        candidate.search_parameters = state.request.base_search_parameters;
        state.candidates = {candidate};

        state.report.stages.push_back(
            MakeStage(Stage(),
                      TuningStageStatus::COMPLETED,
                      "using fixed build representation and ef_search space",
                      1,
                      1));
    }
};

class BuildParameterTuningStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::BUILD_PARAMETER_TUNING;
    }

    void
    Run(TuningState& state, const TuningStageRuntime&) const override {
        const auto enabled = state.request.enable_build_parameter_tuning;
        if (not enabled) {
            state.report.stages.push_back(
                MakeStage(Stage(),
                          TuningStageStatus::SKIPPED,
                          "build parameter tuning is disabled in this stage"));
            return;
        }

        const auto input_count = CandidateCount(state);
        ExpandCandidates(state.candidates, state.request.build_parameter_spaces);
        state.report.stages.push_back(MakeStage(Stage(),
                                                TuningStageStatus::COMPLETED,
                                                "enumerated build parameter candidates",
                                                input_count,
                                                CandidateCount(state)));
    }
};

class QuantizerTuningStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::QUANTIZER_TUNING;
    }

    void
    Run(TuningState& state, const TuningStageRuntime&) const override {
        const auto enabled = state.request.enable_quantizer_tuning;
        if (not enabled) {
            state.report.stages.push_back(MakeStage(
                Stage(), TuningStageStatus::SKIPPED, "quantizer tuning is disabled in this stage"));
            return;
        }

        const auto input_count = CandidateCount(state);
        ExpandCandidates(state.candidates, state.request.quantizer_parameter_spaces);
        state.report.stages.push_back(MakeStage(Stage(),
                                                TuningStageStatus::COMPLETED,
                                                "enumerated quantizer candidates",
                                                input_count,
                                                CandidateCount(state)));
    }
};

class CandidateGenerationStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::CANDIDATE_GENERATION;
    }

    void
    Run(TuningState& state, const TuningStageRuntime&) const override {
        const auto input_count = CandidateCount(state);
        ExpandCandidates(state.candidates, MakeEfSearchParameterSpaces(state.request));

        state.report.stages.push_back(MakeStage(Stage(),
                                                TuningStageStatus::COMPLETED,
                                                "generated tuning candidates",
                                                input_count,
                                                CandidateCount(state)));
    }
};

class CandidateValidationStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::CANDIDATE_VALIDATION;
    }

    void
    Run(TuningState& state, const TuningStageRuntime&) const override {
        if (state.candidates.empty()) {
            state.report.stages.push_back(MakeStage(
                Stage(), TuningStageStatus::FAILED, "no tuning candidates were generated"));
            state.should_stop = true;
            return;
        }

        for (const auto& candidate : state.candidates) {
            std::string error_message;
            if (ValidateCandidate(state.request, candidate, error_message)) {
                continue;
            }
            state.report.stages.push_back(MakeStage(
                Stage(), TuningStageStatus::FAILED, error_message, CandidateCount(state), 0));
            state.should_stop = true;
            return;
        }

        state.report.stages.push_back(MakeStage(Stage(),
                                                TuningStageStatus::COMPLETED,
                                                "validated tuning candidates",
                                                CandidateCount(state),
                                                CandidateCount(state)));
    }
};

class CandidatePruningStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::CANDIDATE_PRUNING;
    }

    void
    Run(TuningState& state, const TuningStageRuntime& runtime) const override {
        if (state.request.enable_successive_halving) {
            return;
        }
        if (state.request.enable_build_parameter_tuning || state.request.enable_quantizer_tuning) {
            state.trial_report = PlanFullCandidateTrials(state.request, state.candidates);
            state.report.trial_report = state.trial_report;
            const auto skipped_trials = CountTrials(state.trial_report, TuningTrialStatus::SKIPPED);
            const auto failed_trials = CountTrials(state.trial_report, TuningTrialStatus::FAILED);
            const auto runnable_trials = static_cast<uint64_t>(state.trial_report.trials.size()) -
                                         skipped_trials - failed_trials;
            state.report.stages.push_back(
                MakeStage(Stage(),
                          TuningStageStatus::COMPLETED,
                          "using exhaustive candidates without optimized pruning",
                          CandidateCount(state),
                          runnable_trials));
            return;
        }
        state.trial_report = PlanEfSearchTrials(state.request, state.candidates);
        state.report.trial_report = state.trial_report;

        const auto skipped_trials = CountTrials(state.trial_report, TuningTrialStatus::SKIPPED);
        const auto runnable_trials =
            static_cast<uint64_t>(state.trial_report.trials.size()) - skipped_trials;
        state.report.stages.push_back(
            MakeStage(Stage(),
                      TuningStageStatus::COMPLETED,
                      "skipped invalid or budgeted ef_search candidates",
                      static_cast<uint64_t>(state.trial_report.trials.size()),
                      runnable_trials));
    }
};

class TrialPlanningStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::TRIAL_PLANNING;
    }

    void
    Run(TuningState& state, const TuningStageRuntime&) const override {
        if (state.request.enable_successive_halving) {
            state.report.stages.push_back(MakeStage(
                Stage(), TuningStageStatus::FAILED, "successive halving is not implemented yet"));
            state.should_stop = true;
            return;
        }

        if (state.request.enable_build_parameter_tuning || state.request.enable_quantizer_tuning) {
            const auto skipped_trials = CountTrials(state.trial_report, TuningTrialStatus::SKIPPED);
            const auto failed_trials = CountTrials(state.trial_report, TuningTrialStatus::FAILED);
            const auto runnable_trials = static_cast<uint64_t>(state.trial_report.trials.size()) -
                                         skipped_trials - failed_trials;
            state.report.stages.push_back(MakeStage(Stage(),
                                                    TuningStageStatus::COMPLETED,
                                                    "using single-round exhaustive evaluation",
                                                    runnable_trials,
                                                    runnable_trials));
            return;
        }

        const auto skipped_trials = CountTrials(state.trial_report, TuningTrialStatus::SKIPPED);
        const auto runnable_trials =
            static_cast<uint64_t>(state.trial_report.trials.size()) - skipped_trials;
        state.report.stages.push_back(MakeStage(Stage(),
                                                TuningStageStatus::COMPLETED,
                                                "using single-round full evaluation",
                                                runnable_trials,
                                                runnable_trials));
    }
};

class TrialExecutionStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::TRIAL_EXECUTION;
    }

    void
    Run(TuningState& state, const TuningStageRuntime& runtime) const override {
        if (state.request.enable_build_parameter_tuning || state.request.enable_quantizer_tuning) {
            RunRebuildTrials(state, runtime);
            return;
        }

        if (runtime.ef_search_tuner == nullptr) {
            state.report.stages.push_back(
                MakeStage(Stage(), TuningStageStatus::FAILED, "ef_search tuner is required"));
            state.should_stop = true;
            return;
        }

        for (auto& trial : state.trial_report.trials) {
            if (trial.status == TuningTrialStatus::SKIPPED) {
                continue;
            }

            EvaluationRequest evaluation_request;
            evaluation_request.index = state.request.index;
            evaluation_request.queries = state.request.queries;
            evaluation_request.ground_truth = state.request.ground_truth;
            evaluation_request.topk = state.request.topk;
            evaluation_request.query_count = state.request.query_count;
            if (not MakeSearchParameters(state.request.base_search_parameters,
                                         state.request.index_name,
                                         trial.candidate.ef_search,
                                         evaluation_request.search_parameters,
                                         trial.message)) {
                trial.status = TuningTrialStatus::FAILED;
                continue;
            }

            trial.evaluation = runtime.ef_search_tuner->Evaluate(evaluation_request);
            if (trial.evaluation.Succeeded()) {
                trial.status = TuningTrialStatus::COMPLETED;
            } else {
                trial.status = TuningTrialStatus::FAILED;
                trial.message = trial.evaluation.error_message;
            }
        }
        state.report.trial_report = state.trial_report;

        const auto skipped_trials = CountTrials(state.trial_report, TuningTrialStatus::SKIPPED);
        const auto runnable_trials =
            static_cast<uint64_t>(state.trial_report.trials.size()) - skipped_trials;
        const auto completed_trials = CountTrials(state.trial_report, TuningTrialStatus::COMPLETED);
        const auto failed_trials = CountTrials(state.trial_report, TuningTrialStatus::FAILED);
        state.report.stages.push_back(
            MakeStage(Stage(),
                      failed_trials > 0 ? TuningStageStatus::FAILED : TuningStageStatus::COMPLETED,
                      failed_trials > 0 ? "one or more trials failed" : "trials completed",
                      runnable_trials,
                      completed_trials));
        if (failed_trials > 0) {
            state.should_stop = true;
        }
    }

private:
    void
    RunRebuildTrials(TuningState& state, const TuningStageRuntime& runtime) const {
        if (runtime.ef_search_tuner == nullptr) {
            state.report.stages.push_back(
                MakeStage(Stage(), TuningStageStatus::FAILED, "ef_search tuner is required"));
            state.should_stop = true;
            return;
        }
        if (state.request.base == nullptr) {
            state.report.stages.push_back(
                MakeStage(Stage(),
                          TuningStageStatus::FAILED,
                          "base dataset is required for build or quantizer candidate execution",
                          CandidateCount(state),
                          0));
            state.should_stop = true;
            return;
        }

        for (auto& trial : state.trial_report.trials) {
            if (trial.status == TuningTrialStatus::SKIPPED ||
                trial.status == TuningTrialStatus::FAILED) {
                continue;
            }

            std::string build_parameters;
            if (not MakeCandidateBuildParameters(state.request.build_parameters,
                                                 trial.candidate.patches,
                                                 build_parameters,
                                                 trial.message)) {
                trial.status = TuningTrialStatus::FAILED;
                continue;
            }

            auto index = BuildCandidateIndex(
                state.request.index_name, build_parameters, state.request.base, trial.message);
            if (index == nullptr) {
                trial.status = TuningTrialStatus::FAILED;
                continue;
            }

            std::string search_parameters;
            if (not MakeCandidateSearchParameters(state.request.base_search_parameters,
                                                  state.request.index_name,
                                                  trial.candidate.patches,
                                                  search_parameters,
                                                  trial.message)) {
                trial.status = TuningTrialStatus::FAILED;
                continue;
            }

            EvaluationRequest evaluation_request;
            evaluation_request.index = index;
            evaluation_request.queries = state.request.queries;
            evaluation_request.ground_truth = state.request.ground_truth;
            evaluation_request.topk = state.request.topk;
            evaluation_request.query_count = state.request.query_count;
            evaluation_request.search_parameters = std::move(search_parameters);

            trial.evaluation = runtime.ef_search_tuner->Evaluate(evaluation_request);
            if (trial.evaluation.Succeeded()) {
                trial.status = TuningTrialStatus::COMPLETED;
            } else {
                trial.status = TuningTrialStatus::FAILED;
                trial.message = trial.evaluation.error_message;
            }
        }
        state.report.trial_report = state.trial_report;

        const auto skipped_trials = CountTrials(state.trial_report, TuningTrialStatus::SKIPPED);
        const auto runnable_trials =
            static_cast<uint64_t>(state.trial_report.trials.size()) - skipped_trials;
        const auto completed_trials = CountTrials(state.trial_report, TuningTrialStatus::COMPLETED);
        const auto failed_trials = CountTrials(state.trial_report, TuningTrialStatus::FAILED);
        state.report.stages.push_back(
            MakeStage(Stage(),
                      failed_trials > 0 ? TuningStageStatus::FAILED : TuningStageStatus::COMPLETED,
                      failed_trials > 0 ? "one or more trials failed" : "trials completed",
                      runnable_trials,
                      completed_trials));
        if (failed_trials > 0) {
            state.should_stop = true;
        }
    }
};

class SelectionStage final : public TuningStageExecutor {
public:
    [[nodiscard]] TuningStage
    Stage() const override {
        return TuningStage::SELECTION;
    }

    void
    Run(TuningState& state, const TuningStageRuntime&) const override {
        state.trial_report.recommendation.reset();
        state.trial_report.best_effort.reset();
        for (const auto& trial : state.trial_report.trials) {
            if (trial.status != TuningTrialStatus::COMPLETED) {
                continue;
            }
            if (not state.trial_report.best_effort.has_value() ||
                IsBetterBestEffort(trial, state.trial_report.best_effort.value())) {
                state.trial_report.best_effort = trial;
            }
            if (trial.evaluation.recall.average >= state.request.target_recall &&
                (not state.trial_report.recommendation.has_value() ||
                 IsLowerCostTrial(trial, state.trial_report.recommendation.value()))) {
                state.trial_report.recommendation = trial;
            }
        }

        state.report.trial_report = state.trial_report;
        state.report.recommendation = state.trial_report.recommendation;
        state.report.best_effort = state.trial_report.best_effort;
        state.report.stages.push_back(MakeStage(
            Stage(),
            state.report.recommendation.has_value() ? TuningStageStatus::COMPLETED
                                                    : TuningStageStatus::SKIPPED,
            state.report.recommendation.has_value() ? "selected recommended candidate"
                                                    : "no candidate met the target recall"));
    }
};

}  // namespace

bool
AutoTuningReport::Succeeded() const {
    return not HasFailedStage(stages) && recommendation.has_value();
}

AutoTuningPipeline::AutoTuningPipeline() : AutoTuningPipeline(EfSearchTuner()) {
}

AutoTuningPipeline::AutoTuningPipeline(EfSearchTuner ef_search_tuner)
    : ef_search_tuner_(std::move(ef_search_tuner)) {
}

TuningPlan
AutoTuningPlanner::Plan(const AutoTuningRequest&) const {
    TuningPlan plan;
    plan.stages.push_back(std::make_unique<WorkloadValidationStage>());
    plan.stages.push_back(std::make_unique<SearchSpaceConstructionStage>());
    plan.stages.push_back(std::make_unique<BuildParameterTuningStage>());
    plan.stages.push_back(std::make_unique<QuantizerTuningStage>());
    plan.stages.push_back(std::make_unique<CandidateGenerationStage>());
    plan.stages.push_back(std::make_unique<CandidateValidationStage>());
    plan.stages.push_back(std::make_unique<CandidatePruningStage>());
    plan.stages.push_back(std::make_unique<TrialPlanningStage>());
    plan.stages.push_back(std::make_unique<TrialExecutionStage>());
    plan.stages.push_back(std::make_unique<SelectionStage>());
    return plan;
}

AutoTuningReport
AutoTuningPipeline::Tune(const AutoTuningRequest& request) const {
    const auto started_at = std::chrono::steady_clock::now();
    TuningState state;
    state.request = request;
    state.report.request = MakeRequestSummary(request);

    AutoTuningPlanner planner;
    auto plan = planner.Plan(request);
    TuningStageRuntime runtime;
    runtime.ef_search_tuner = &ef_search_tuner_;

    for (const auto& stage : plan.stages) {
        if (state.should_stop) {
            break;
        }
        stage->Run(state, runtime);
    }

    state.report.trial_report = state.trial_report;
    state.report.ef_search = state.trial_report;
    state.report.recommendation = state.trial_report.recommendation;
    state.report.best_effort = state.trial_report.best_effort;
    state.report.elapsed_ms = ElapsedMs(started_at);
    return state.report;
}

}  // namespace vsag
