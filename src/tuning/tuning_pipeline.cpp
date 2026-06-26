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

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <memory>
#include <utility>

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
CountTrials(const EfSearchTuningReport& report, TuningTrialStatus status) {
    return static_cast<uint64_t>(
        std::count_if(report.trials.begin(), report.trials.end(), [status](const auto& trial) {
            return trial.status == status;
        }));
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

EfSearchTuningRequest
MakeEfSearchTuningRequest(const AutoTuningRequest& request) {
    EfSearchTuningRequest ef_request;
    ef_request.index = request.index;
    ef_request.queries = request.queries;
    ef_request.ground_truth = request.ground_truth;
    ef_request.topk = request.topk;
    ef_request.query_count = request.query_count;
    ef_request.target_recall = request.target_recall;
    ef_request.index_name = request.index_name;
    ef_request.base_search_parameters = request.base_search_parameters;
    ef_request.ef_search_candidates = request.ef_search_candidates;
    ef_request.max_trials = request.max_trials;
    return ef_request;
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
            state.report.stages.push_back(
                MakeStage(Stage(),
                          TuningStageStatus::COMPLETED,
                          "using exhaustive candidates without optimized pruning",
                          CandidateCount(state),
                          CandidateCount(state)));
            return;
        }
        if (runtime.ef_search_tuner == nullptr) {
            state.report.stages.push_back(
                MakeStage(Stage(), TuningStageStatus::FAILED, "ef_search tuner is required"));
            state.should_stop = true;
            return;
        }

        state.ef_search = runtime.ef_search_tuner->Tune(MakeEfSearchTuningRequest(state.request));
        state.report.ef_search = state.ef_search;

        const auto skipped_trials = CountTrials(state.ef_search, TuningTrialStatus::SKIPPED);
        const auto runnable_trials =
            static_cast<uint64_t>(state.ef_search.trials.size()) - skipped_trials;
        state.report.stages.push_back(
            MakeStage(Stage(),
                      TuningStageStatus::COMPLETED,
                      "skipped invalid or budgeted ef_search candidates",
                      static_cast<uint64_t>(state.ef_search.trials.size()),
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
            state.report.stages.push_back(MakeStage(Stage(),
                                                    TuningStageStatus::COMPLETED,
                                                    "using single-round exhaustive evaluation",
                                                    CandidateCount(state),
                                                    CandidateCount(state)));
            return;
        }

        const auto skipped_trials = CountTrials(state.ef_search, TuningTrialStatus::SKIPPED);
        const auto runnable_trials =
            static_cast<uint64_t>(state.ef_search.trials.size()) - skipped_trials;
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
    Run(TuningState& state, const TuningStageRuntime&) const override {
        if (state.request.enable_build_parameter_tuning || state.request.enable_quantizer_tuning) {
            state.report.stages.push_back(
                MakeStage(Stage(),
                          TuningStageStatus::FAILED,
                          "build or quantizer candidate execution is not implemented yet",
                          CandidateCount(state),
                          0));
            state.should_stop = true;
            return;
        }

        const auto skipped_trials = CountTrials(state.ef_search, TuningTrialStatus::SKIPPED);
        const auto runnable_trials =
            static_cast<uint64_t>(state.ef_search.trials.size()) - skipped_trials;
        const auto completed_trials = CountTrials(state.ef_search, TuningTrialStatus::COMPLETED);
        const auto failed_trials = CountTrials(state.ef_search, TuningTrialStatus::FAILED);
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
        state.report.recommendation = state.ef_search.recommendation;
        state.report.best_effort = state.ef_search.best_effort;
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

    state.report.elapsed_ms = ElapsedMs(started_at);
    return state.report;
}

}  // namespace vsag
