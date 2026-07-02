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

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "autotune.h"

namespace vsag::autotune::internal {

using Clock = std::chrono::steady_clock;

struct ExecutionOptions {
    int top_k{10};
    std::string search_mode{"knn"};
    uint64_t search_query_count{0};
    uint64_t query_limit_count{0};
    int32_t num_threads_building{1};
    int32_t num_threads_searching{1};
    std::string workspace_path{"/tmp/vsag_autotune"};
    bool keep_intermediate{false};
    uint64_t max_trials{0};
    std::string evaluation_strategy{"full_grid"};
    uint64_t sample_query_count{0};
    uint64_t finalist_count{1};
    bool include_trials{true};
    std::string result_path;
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
    std::string index_name;
    std::string eval_type;
    std::string index_path;
    JsonType create_params;
    JsonType search_params;
};

struct AutoTunePlan {
    std::vector<BuildSpec> builds;
    std::vector<TrialSpec> trials;
};

struct EvaluationResult {
    std::vector<JsonType> build_results;
    std::vector<JsonType> trial_results;
    std::vector<JsonType> selection_trial_results;
    uint64_t executed_build_count{0};
    JsonType strategy_report{JsonType::object()};
};

class EvaluationStrategy {
public:
    virtual ~EvaluationStrategy() = default;

    virtual EvaluationResult
    Run(const AutoTunePlan& plan,
        const JsonType& request,
        const ExecutionOptions& options) const = 0;
};

double
ElapsedSeconds(const Clock::time_point& start);

void
Require(bool condition, const std::string& message);

std::string
GetString(const JsonType& object, const std::string& key, const std::string& default_value);

int
GetInt(const JsonType& object, const std::string& key, int default_value);

uint64_t
GetUInt64(const JsonType& object, const std::string& key, uint64_t default_value);

bool
GetBool(const JsonType& object, const std::string& key, bool default_value);

JsonType&
EnsureObject(JsonType& object, const std::string& key);

bool
IsSupportedConstraint(const std::string& constraint_name);

void
ValidateRequest(const JsonType& request);

ExecutionOptions
ParseExecutionOptions(const JsonType& request);

std::vector<JsonType>
ExpandJson(const JsonType& value);

std::vector<CandidateSpec>
GenerateCandidates(const JsonType& request);

AutoTunePlan
PlanTrials(const JsonType& request,
           const std::vector<CandidateSpec>& candidates,
           const ExecutionOptions& options);

JsonType
RunBuild(const BuildSpec& build, const JsonType& request, const ExecutionOptions& options);

JsonType
RunSearchTrial(const TrialSpec& trial,
               const JsonType& build_result,
               const JsonType& request,
               const JsonType& constraints,
               const ExecutionOptions& options);

void
CleanupBuildArtifact(const BuildSpec& build, const ExecutionOptions& options);

const EvaluationStrategy&
FullGridEvaluationStrategy();

const EvaluationStrategy&
QuerySamplingEvaluationStrategy();

const EvaluationStrategy&
GetEvaluationStrategy(const ExecutionOptions& options);

JsonType
SelectResult(const std::vector<JsonType>& trials);

void
WriteJsonFile(const std::string& path, const JsonType& json);

std::string
GetOptionalResultPath(const JsonType& request);

JsonType
MakeFailedResult(const JsonType& request,
                 const std::string& message,
                 const Clock::time_point& total_start);

}  // namespace vsag::autotune::internal
