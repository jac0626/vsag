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

#include "tuning/evaluation_result.h"
#include "tuning/evaluation_runner.h"

namespace vsag {

enum class TuningTrialStatus {
    COMPLETED,
    SKIPPED,
    FAILED,
};

struct EfSearchCandidate {
    uint64_t ef_search = 0;
};

struct EfSearchTrialResult {
    uint64_t trial_id = 0;
    EfSearchCandidate candidate;
    TuningTrialStatus status = TuningTrialStatus::COMPLETED;
    std::string message;
    EvaluationResult evaluation;
};

struct EfSearchTuningRequest {
    IndexPtr index = nullptr;
    DatasetPtr queries = nullptr;
    DatasetPtr ground_truth = nullptr;
    uint64_t topk = 0;
    uint64_t query_count = 0;
    double target_recall = 0.0;
    std::string index_name = "hgraph";
    std::vector<uint64_t> ef_search_candidates;
};

struct EfSearchTuningReport {
    std::vector<EfSearchTrialResult> trials;
    std::optional<EfSearchTrialResult> recommendation;
    std::optional<EfSearchTrialResult> best_effort;
};

class EfSearchTuner {
public:
    explicit EfSearchTuner(InMemoryEvaluationRunner runner = InMemoryEvaluationRunner());

    [[nodiscard]] EfSearchTuningReport
    Tune(const EfSearchTuningRequest& request) const;

private:
    InMemoryEvaluationRunner runner_;
};

}  // namespace vsag
