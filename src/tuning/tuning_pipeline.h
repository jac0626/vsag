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

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tuning/ef_search_tuner.h"

namespace vsag {

enum class TuningStage {
    WORKLOAD_VALIDATION,
    SEARCH_SPACE_CONSTRUCTION,
    BUILD_PARAMETER_TUNING,
    QUANTIZER_TUNING,
    CANDIDATE_GENERATION,
    CANDIDATE_PRUNING,
    TRIAL_PLANNING,
    TRIAL_EXECUTION,
    SELECTION,
};

enum class TuningStageStatus {
    COMPLETED,
    SKIPPED,
    FAILED,
};

struct TuningStageResult {
    TuningStage stage = TuningStage::WORKLOAD_VALIDATION;
    TuningStageStatus status = TuningStageStatus::COMPLETED;
    std::string message;
    uint64_t input_count = 0;
    uint64_t output_count = 0;
};

struct TuningParameterSpace {
    std::string path;
    std::vector<std::string> values;
};

// Internal request for HGraph V1 skeleton tuning.
// The caller owns input object lifetime and ground-truth preparation.
struct AutoTuningRequest {
    IndexPtr index = nullptr;
    DatasetPtr base = nullptr;
    DatasetPtr queries = nullptr;
    DatasetPtr ground_truth = nullptr;
    std::string source_type = "existing_index";
    uint64_t topk = 0;
    uint64_t query_count = 0;
    double target_recall = 0.0;
    std::string index_name = "hgraph";
    std::string build_parameters;
    std::string base_search_parameters;
    std::vector<uint64_t> ef_search_candidates;
    std::vector<TuningParameterSpace> build_parameter_spaces;
    std::vector<TuningParameterSpace> quantizer_parameter_spaces;
    std::vector<TuningParameterSpace> search_parameter_spaces;
    uint64_t max_trials = 0;
    bool enable_build_parameter_tuning = false;
    bool enable_quantizer_tuning = false;
    bool enable_successive_halving = false;
};

struct AutoTuningRequestSummary {
    std::string index_name = "hgraph";
    std::string source_type = "existing_index";
    uint64_t topk = 0;
    uint64_t requested_query_count = 0;
    uint64_t effective_query_count = 0;
    double target_recall = 0.0;
    std::string build_parameters;
    std::string base_search_parameters;
    std::vector<uint64_t> ef_search_candidates;
    std::vector<TuningParameterSpace> build_parameter_spaces;
    std::vector<TuningParameterSpace> quantizer_parameter_spaces;
    std::vector<TuningParameterSpace> search_parameter_spaces;
    uint64_t max_trials = 0;
    bool enable_build_parameter_tuning = false;
    bool enable_quantizer_tuning = false;
    bool enable_successive_halving = false;
};

// Internal report. elapsed_ms covers AutoTuningPipeline::Tune() only.
struct AutoTuningReport {
    AutoTuningRequestSummary request;
    std::vector<TuningStageResult> stages;
    TuningTrialReport trial_report;
    // Compatibility mirror for the original search-only report field.
    EfSearchTuningReport ef_search;
    std::optional<TuningTrialResult> recommendation;
    std::optional<TuningTrialResult> best_effort;
    double elapsed_ms = 0.0;

    [[nodiscard]] bool
    Succeeded() const;
};

struct TuningCandidate {
    uint64_t id = 0;
    std::string index_name = "hgraph";
    std::string source_type = "existing_index";
    std::string build_parameters;
    std::string search_parameters;
    std::vector<TuningParameterPatch> patches;
};

struct TuningState {
    AutoTuningRequest request;
    AutoTuningReport report;
    std::vector<TuningCandidate> candidates;
    TuningTrialReport trial_report;
    bool should_stop = false;
};

struct TuningStageRuntime {
    const EfSearchTuner* ef_search_tuner = nullptr;
};

class TuningStageExecutor {
public:
    virtual ~TuningStageExecutor() = default;

    [[nodiscard]] virtual TuningStage
    Stage() const = 0;

    virtual void
    Run(TuningState& state, const TuningStageRuntime& runtime) const = 0;
};

struct TuningPlan {
    std::vector<std::unique_ptr<TuningStageExecutor>> stages;
};

class AutoTuningPlanner {
public:
    [[nodiscard]] TuningPlan
    Plan(const AutoTuningRequest& request) const;
};

class AutoTuningPipeline {
public:
    AutoTuningPipeline();
    explicit AutoTuningPipeline(EfSearchTuner ef_search_tuner);

    [[nodiscard]] AutoTuningReport
    Tune(const AutoTuningRequest& request) const;

private:
    EfSearchTuner ef_search_tuner_;
};

}  // namespace vsag
