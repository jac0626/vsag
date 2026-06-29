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

#include "autotune.h"

#include <exception>
#include <iostream>
#include <utility>
#include <vector>

#include "autotune_internal.h"

namespace vsag::autotune {

JsonType
RunAutoTune(const JsonType& request) {
    const auto total_start = internal::Clock::now();
    try {
        JsonType elapsed_breakdown;

        const auto validation_start = internal::Clock::now();
        internal::ValidateRequest(request);
        auto options = internal::ParseExecutionOptions(request);
        elapsed_breakdown["validation"] = internal::ElapsedSeconds(validation_start);

        const auto candidate_start = internal::Clock::now();
        auto candidates = internal::GenerateCandidates(request);
        auto trials = internal::PlanTrials(request, candidates, options);
        elapsed_breakdown["candidate_generation"] = internal::ElapsedSeconds(candidate_start);

        const auto evaluation_start = internal::Clock::now();
        std::vector<JsonType> trial_results;
        trial_results.reserve(trials.size());
        uint64_t trial_ordinal = 0;
        for (const auto& trial : trials) {
            ++trial_ordinal;
            std::cerr << "[AutoTune] running trial " << trial_ordinal << "/" << trials.size() << " "
                      << trial.trial_id << " index=" << trial.index_name
                      << " eval_type=" << trial.eval_type << std::endl;
            auto trial_result = internal::RunTrial(trial, request, request["constraints"], options);
            std::cerr << "[AutoTune] finished trial " << trial.trial_id
                      << " status=" << trial_result["status"].get<std::string>()
                      << " elapsed_seconds=" << trial_result["elapsed_seconds"].get<double>()
                      << std::endl;
            trial_results.emplace_back(std::move(trial_result));
        }
        elapsed_breakdown["evaluation"] = internal::ElapsedSeconds(evaluation_start);

        const auto selection_start = internal::Clock::now();
        auto selection = internal::SelectResult(trial_results);
        elapsed_breakdown["selection"] = internal::ElapsedSeconds(selection_start);

        JsonType result;
        result["version"] = 1;
        result["status"] = selection["status"];
        result["elapsed_seconds"] = internal::ElapsedSeconds(total_start);
        result["elapsed_breakdown_seconds"] = elapsed_breakdown;
        result["recommendation"] = selection["recommendation"];
        result["best_effort"] = selection["best_effort"];
        result["trial_count"] = trial_results.size();
        result["failure"] = selection["failure"];
        if (options.include_trials) {
            result["trials"] = trial_results;
        }

        internal::WriteJsonFile(options.result_path, result);
        return result;
    } catch (const std::exception& e) {
        auto result = internal::MakeFailedResult(request, e.what(), total_start);
        const auto result_path = internal::GetOptionalResultPath(request);
        if (!result_path.empty()) {
            try {
                internal::WriteJsonFile(result_path, result);
            } catch (const std::exception& write_error) {
                result["failure"]["result_write_error"] = write_error.what();
            }
        }
        return result;
    }
}

std::vector<JsonType>
ExpandJsonForTest(const JsonType& value) {
    return internal::ExpandJson(value);
}

JsonType
GenerateCandidatesForTest(const JsonType& request) {
    internal::ValidateRequest(request);
    auto options = internal::ParseExecutionOptions(request);
    auto candidates = internal::GenerateCandidates(request);
    auto trials = internal::PlanTrials(request, candidates, options);
    JsonType result;
    result["candidate_count"] = candidates.size();
    result["trial_count"] = trials.size();
    result["trials"] = JsonType::array();
    for (const auto& trial : trials) {
        result["trials"].push_back(
            JsonType{{"trial_id", trial.trial_id},
                     {"build_id", trial.build_id},
                     {"index_name", trial.index_name},
                     {"eval_type", trial.eval_type},
                     {"index_path", trial.index_path},
                     {"create_params", trial.create_params},
                     {"search_params", trial.search_params},
                     {"cleanup_index_after_trial", trial.cleanup_index_after_trial}});
    }
    return result;
}

}  // namespace vsag::autotune
