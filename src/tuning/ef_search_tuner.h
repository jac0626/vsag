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
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "tuning/evaluation_result.h"
#include "tuning/evaluation_runner.h"

namespace vsag {

enum class TuningTrialStatus {
    COMPLETED,
    SKIPPED,
    FAILED,
};

struct TuningParameterPatch {
    std::string path;
    std::string value;
};

struct TuningCandidateReport {
    uint64_t ef_search = 0;
    std::vector<TuningParameterPatch> patches;
};

struct TuningTrialResult {
    uint64_t trial_id = 0;
    TuningCandidateReport candidate;
    TuningTrialStatus status = TuningTrialStatus::COMPLETED;
    std::string message;
    EvaluationResult evaluation;
};

struct TuningTrialReport {
    std::vector<TuningTrialResult> trials;
    std::optional<TuningTrialResult> recommendation;
    std::optional<TuningTrialResult> best_effort;
};

using EfSearchCandidate = TuningCandidateReport;
using EfSearchTrialResult = TuningTrialResult;
using EfSearchTuningReport = TuningTrialReport;

struct EfSearchTuningRequest {
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
};

class EfSearchTuner {
public:
    using EvaluationFunction = std::function<EvaluationResult(const EvaluationRequest&)>;

    EfSearchTuner();
    explicit EfSearchTuner(InMemoryEvaluationRunner runner);
    explicit EfSearchTuner(EvaluationFunction evaluator);

    [[nodiscard]] EfSearchTuningReport
    Tune(const EfSearchTuningRequest& request) const;

    [[nodiscard]] EvaluationResult
    Evaluate(const EvaluationRequest& request) const;

private:
    EvaluationFunction evaluator_;
};

}  // namespace vsag
