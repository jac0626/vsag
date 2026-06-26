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
#include <string>

#include "tuning/evaluation_result.h"
#include "vsag/dataset.h"
#include "vsag/index.h"

namespace vsag {

struct EvaluationRequest {
    IndexPtr index = nullptr;
    DatasetPtr queries = nullptr;
    DatasetPtr ground_truth = nullptr;
    std::string search_parameters;
    uint64_t topk = 0;
    uint64_t query_count = 0;
    bool collect_memory = true;
};

class InMemoryEvaluationRunner {
public:
    [[nodiscard]] EvaluationResult
    Run(const EvaluationRequest& request) const;
};

}  // namespace vsag
