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

namespace vsag {

enum class EvaluationStatus {
    SUCCESS,
    INVALID_ARGUMENT,
    SEARCH_ERROR,
};

struct RecallMetrics {
    double average = 0.0;
    double p0 = 0.0;
    double p10 = 0.0;
    double p30 = 0.0;
    double p50 = 0.0;
    double p70 = 0.0;
    double p90 = 0.0;
};

struct LatencyMetrics {
    double average_ms = 0.0;
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
};

struct EvaluationResult {
    EvaluationStatus status = EvaluationStatus::SUCCESS;
    std::string error_message;
    uint64_t query_count = 0;
    RecallMetrics recall;
    LatencyMetrics latency;
    double qps = 0.0;
    uint64_t memory_bytes = 0;

    [[nodiscard]] bool
    Succeeded() const {
        return status == EvaluationStatus::SUCCESS;
    }
};

}  // namespace vsag
