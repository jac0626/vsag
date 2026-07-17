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
#include <map>
#include <string>
#include <vector>

#include "autotune.h"

namespace vsag::autotune::internal {

constexpr uint64_t DEFAULT_MAX_TRIALS = 1000;
constexpr uint64_t MAX_V1_TRIALS = 100000;
constexpr uint64_t MAX_V1_TOP_K = 1000000;
constexpr uint64_t MAX_V1_CONCURRENT_NEIGHBOR_IDS = 1000000;
constexpr uint64_t MAX_V1_EXPANDED_VALUES = 100000;
constexpr uint64_t MAX_V1_EXPANDED_COMBINATIONS = 1000000;
constexpr uint64_t MAX_V1_EXPANSION_DEPTH = 128;

enum class ObjectiveDirection { Minimize, Maximize };

using ConstraintMap = std::map<std::string, double>;
using MetricMap = std::map<std::string, double>;

struct ObjectiveSpec {
    std::string metric;
};

struct DatasetDescription {
    uint64_t dim{0};
    std::string dtype;
    std::string metric_type;
    uint64_t base_count{0};
    uint64_t query_count{0};
    uint64_t ground_truth_k{0};
    std::string vector_type;
};

struct IndexSpec {
    std::string name;
    JsonType create_params = JsonType::object();
    JsonType search_params = JsonType::object();
};

struct WorkloadSpec {
    uint64_t top_k{0};
    uint64_t concurrency{1};
};

struct TuningConfig {
    std::string workspace_path{"/tmp/vsag_autotune"};
    bool keep_intermediate{false};
    uint64_t max_trials{DEFAULT_MAX_TRIALS};
};

struct OutputConfig {
    std::string result_path;
    bool include_raw_eval{false};
};

struct AutoTuneRequest {
    JsonType effective_request = JsonType::object();
    uint64_t version{1};
    std::string data_path;
    std::string index_path;
    std::vector<IndexSpec> indexes;
    WorkloadSpec workload;
    ConstraintMap constraints;
    ObjectiveSpec objective;
    TuningConfig tuning_config;
    OutputConfig output;
    DatasetDescription dataset;
    bool dataset_resolved{false};
};

struct CandidateSpec {
    std::string index_name;
    JsonType create_params;
    JsonType search_params;
};

struct BuildSpec {
    std::string build_id;
    std::string index_name;
    std::string index_path;
    JsonType create_params;
    bool use_existing_index{false};
    bool cleanup_index_after_build_group{false};
};

struct TrialSpec {
    std::string trial_id;
    std::string build_id;
    JsonType search_params;
};

struct AutoTunePlan {
    std::vector<BuildSpec> builds;
    std::vector<TrialSpec> trials;
};

}  // namespace vsag::autotune::internal
