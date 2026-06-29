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
#include <map>
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
        auto plan = internal::PlanTrials(request, candidates, options);
        elapsed_breakdown["candidate_generation"] = internal::ElapsedSeconds(candidate_start);

        const auto evaluation_start = internal::Clock::now();
        std::vector<JsonType> trial_results;
        trial_results.reserve(plan.trials.size());
        std::vector<JsonType> build_results;
        build_results.reserve(plan.builds.size());
        std::map<std::string, std::vector<internal::TrialSpec>> trials_by_build_id;
        for (const auto& trial : plan.trials) {
            trials_by_build_id[trial.build_id].emplace_back(trial);
        }

        uint64_t trial_ordinal = 0;
        uint64_t build_ordinal = 0;
        uint64_t executed_build_count = 0;
        for (const auto& build : plan.builds) {
            ++build_ordinal;
            if (!build.use_existing_index) {
                ++executed_build_count;
            }
            std::cerr << "[AutoTune] running build group " << build_ordinal << "/"
                      << plan.builds.size() << " " << build.build_id
                      << " index=" << build.index_name
                      << " use_existing_index=" << build.use_existing_index << std::endl;
            auto build_result = internal::RunBuild(build, request, options);
            std::cerr << "[AutoTune] finished build group " << build.build_id
                      << " status=" << build_result["status"].get<std::string>()
                      << " elapsed_seconds=" << build_result["elapsed_seconds"].get<double>()
                      << std::endl;

            const auto& group_trials = trials_by_build_id[build.build_id];
            for (const auto& trial : group_trials) {
                ++trial_ordinal;
                std::cerr << "[AutoTune] running trial " << trial_ordinal << "/"
                          << plan.trials.size() << " " << trial.trial_id
                          << " index=" << trial.index_name << " eval_type=" << trial.eval_type
                          << " build_id=" << trial.build_id << std::endl;
                auto trial_result = internal::RunSearchTrial(
                    trial, build_result, request, request["constraints"], options);
                std::cerr << "[AutoTune] finished trial " << trial.trial_id
                          << " status=" << trial_result["status"].get<std::string>()
                          << " elapsed_seconds=" << trial_result["elapsed_seconds"].get<double>()
                          << std::endl;
                trial_results.emplace_back(std::move(trial_result));
            }

            internal::CleanupBuildArtifact(build, options);
            build_results.emplace_back(std::move(build_result));
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
        result["build_count"] = executed_build_count;
        result["build_group_count"] = build_results.size();
        result["failure"] = selection["failure"];
        if (options.include_trials) {
            result["builds"] = build_results;
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
    auto plan = internal::PlanTrials(request, candidates, options);
    JsonType result;
    result["candidate_count"] = candidates.size();
    result["build_group_count"] = plan.builds.size();
    uint64_t executed_build_count = 0;
    result["builds"] = JsonType::array();
    for (const auto& build : plan.builds) {
        if (!build.use_existing_index) {
            ++executed_build_count;
        }
        result["builds"].push_back(
            JsonType{{"build_id", build.build_id},
                     {"index_name", build.index_name},
                     {"index_path", build.index_path},
                     {"create_params", build.create_params},
                     {"use_existing_index", build.use_existing_index},
                     {"cleanup_index_after_build_group", build.cleanup_index_after_build_group}});
    }
    result["build_count"] = executed_build_count;
    result["trial_count"] = plan.trials.size();
    result["trials"] = JsonType::array();
    for (const auto& trial : plan.trials) {
        result["trials"].push_back(JsonType{{"trial_id", trial.trial_id},
                                            {"build_id", trial.build_id},
                                            {"index_name", trial.index_name},
                                            {"eval_type", trial.eval_type},
                                            {"index_path", trial.index_path},
                                            {"create_params", trial.create_params},
                                            {"search_params", trial.search_params}});
    }
    return result;
}

}  // namespace vsag::autotune
