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
    summary.max_trials = request.max_trials;
    summary.enable_build_parameter_tuning = request.enable_build_parameter_tuning;
    summary.enable_quantizer_tuning = request.enable_quantizer_tuning;
    summary.enable_successive_halving = request.enable_successive_halving;
    return summary;
}

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

AutoTuningReport
AutoTuningPipeline::Tune(const AutoTuningRequest& request) const {
    const auto started_at = std::chrono::steady_clock::now();
    AutoTuningReport report;
    report.request = MakeRequestSummary(request);
    auto finish = [started_at](AutoTuningReport& tuning_report) {
        tuning_report.elapsed_ms = ElapsedMs(started_at);
        return tuning_report;
    };

    auto validation = ValidateWorkload(request);
    if (not validation.Succeeded()) {
        report.stages.push_back(MakeStage(
            TuningStage::WORKLOAD_VALIDATION, TuningStageStatus::FAILED, validation.error_message));
        return finish(report);
    }
    report.request.effective_query_count = validation.query_count;
    report.stages.push_back(MakeStage(TuningStage::WORKLOAD_VALIDATION,
                                      TuningStageStatus::COMPLETED,
                                      "workload is valid",
                                      validation.query_count,
                                      validation.query_count));

    report.stages.push_back(MakeStage(TuningStage::SEARCH_SPACE_CONSTRUCTION,
                                      TuningStageStatus::COMPLETED,
                                      "using fixed build representation and ef_search space",
                                      1,
                                      1));

    report.stages.push_back(MakeStage(TuningStage::BUILD_PARAMETER_TUNING,
                                      request.enable_build_parameter_tuning
                                          ? TuningStageStatus::FAILED
                                          : TuningStageStatus::SKIPPED,
                                      request.enable_build_parameter_tuning
                                          ? "build parameter tuning is not implemented yet"
                                          : "build parameter tuning is disabled in this stage"));
    if (request.enable_build_parameter_tuning) {
        return finish(report);
    }

    report.stages.push_back(MakeStage(
        TuningStage::QUANTIZER_TUNING,
        request.enable_quantizer_tuning ? TuningStageStatus::FAILED : TuningStageStatus::SKIPPED,
        request.enable_quantizer_tuning ? "quantizer tuning is not implemented yet"
                                        : "quantizer tuning is disabled in this stage"));
    if (request.enable_quantizer_tuning) {
        return finish(report);
    }

    report.stages.push_back(MakeStage(TuningStage::CANDIDATE_GENERATION,
                                      TuningStageStatus::COMPLETED,
                                      "generated ef_search candidates",
                                      0,
                                      static_cast<uint64_t>(request.ef_search_candidates.size())));

    if (request.enable_successive_halving) {
        report.stages.push_back(MakeStage(TuningStage::TRIAL_PLANNING,
                                          TuningStageStatus::FAILED,
                                          "successive halving is not implemented yet"));
        return finish(report);
    }

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

    report.ef_search = ef_search_tuner_.Tune(ef_request);

    const auto skipped_trials = CountTrials(report.ef_search, TuningTrialStatus::SKIPPED);
    const auto runnable_trials =
        static_cast<uint64_t>(report.ef_search.trials.size()) - skipped_trials;
    report.stages.push_back(MakeStage(TuningStage::CANDIDATE_PRUNING,
                                      TuningStageStatus::COMPLETED,
                                      "skipped invalid or budgeted ef_search candidates",
                                      static_cast<uint64_t>(report.ef_search.trials.size()),
                                      runnable_trials));

    report.stages.push_back(MakeStage(TuningStage::TRIAL_PLANNING,
                                      TuningStageStatus::COMPLETED,
                                      "using single-round full evaluation",
                                      runnable_trials,
                                      runnable_trials));

    const auto completed_trials = CountTrials(report.ef_search, TuningTrialStatus::COMPLETED);
    const auto failed_trials = CountTrials(report.ef_search, TuningTrialStatus::FAILED);
    report.stages.push_back(
        MakeStage(TuningStage::TRIAL_EXECUTION,
                  failed_trials > 0 ? TuningStageStatus::FAILED : TuningStageStatus::COMPLETED,
                  failed_trials > 0 ? "one or more trials failed" : "trials completed",
                  runnable_trials,
                  completed_trials));
    if (failed_trials > 0) {
        return finish(report);
    }

    report.recommendation = report.ef_search.recommendation;
    report.best_effort = report.ef_search.best_effort;
    report.stages.push_back(
        MakeStage(TuningStage::SELECTION,
                  report.recommendation.has_value() ? TuningStageStatus::COMPLETED
                                                    : TuningStageStatus::SKIPPED,
                  report.recommendation.has_value() ? "selected recommended candidate"
                                                    : "no candidate met the target recall"));

    return finish(report);
}

}  // namespace vsag
