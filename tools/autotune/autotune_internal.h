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

#include "autotune_types.h"

namespace vsag::autotune::internal {

using Clock = std::chrono::steady_clock;

struct EvaluationResult {
    std::vector<JsonType> build_results;
    std::vector<JsonType> trial_results;
    uint64_t executed_build_count{0};
};

double
ElapsedSeconds(const Clock::time_point& start);

void
Require(bool condition, const std::string& message);

bool
PathsAlias(const std::string& left, const std::string& right);

MetricMap
MetricsFromJson(const JsonType& value);

EvaluationResult
EvaluatePlan(const AutoTuneRequest& request, const AutoTunePlan& plan);

JsonType
SelectResult(const AutoTuneRequest& request, const EvaluationResult& evaluation);

JsonType
MakeBuildReport(const std::vector<JsonType>& builds, bool include_raw_eval);

JsonType
MakeTrialReport(const std::vector<JsonType>& trials,
                const WorkloadSpec& workload,
                bool include_raw_eval);

JsonType
MakeResultSummary(const JsonType& report);

std::string
FormatResultSummaryForCli(const JsonType& report);

void
WriteJsonFile(const std::string& path, const JsonType& json);

JsonType
MakeFailedResult(const JsonType& request,
                 const std::string& stage,
                 const std::string& code,
                 const std::string& message,
                 const Clock::time_point& total_start);

}  // namespace vsag::autotune::internal
