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

// P0 internal request for HGraph existing-index ef_search tuning.
// The caller owns index construction and ground-truth preparation.
struct AutoTuningRequest {
    IndexPtr index = nullptr;
    DatasetPtr queries = nullptr;
    DatasetPtr ground_truth = nullptr;
    uint64_t topk = 0;
    uint64_t query_count = 0;
    double target_recall = 0.0;
    std::string index_name = "hgraph";
    std::string base_search_parameters;
    std::vector<uint64_t> ef_search_candidates;
    uint64_t max_trials = 0;
    bool enable_build_parameter_tuning = false;
    bool enable_quantizer_tuning = false;
    bool enable_successive_halving = false;
};

// P0 internal report. elapsed_ms covers AutoTuningPipeline::Tune() only.
struct AutoTuningReport {
    std::vector<TuningStageResult> stages;
    EfSearchTuningReport ef_search;
    std::optional<EfSearchTrialResult> recommendation;
    std::optional<EfSearchTrialResult> best_effort;
    double elapsed_ms = 0.0;

    [[nodiscard]] bool
    Succeeded() const;
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
