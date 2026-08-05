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

/**
 * @brief One calibrated operating point for target-recall KNN search.
 *
 * The entry applies to one top-k and, for hierarchical indexes, one path. The
 * search parameters were measured to achieve validated_recall for a workload
 * whose requested recall was target_recall.
 */
struct RecallSearchProfileEntry {
    /// Exact top-k used while measuring this operating point.
    int64_t top_k{0};
    /// Recall constraint requested by the calibration workload.
    double target_recall{0.0};
    /// Recall measured over the complete calibration workload.
    double validated_recall{0.0};
    /// Pyramid path; empty for HGraph, IVF, and Pyramid root search.
    std::string path{};
    /// Complete index-specific search parameters as a JSON object.
    std::string search_parameters{};
};

}  // namespace vsag
