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

#include <iostream>
#include <map>
#include <utility>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

class FullGridEvaluationStrategyImpl final : public EvaluationStrategy {
public:
    EvaluationResult
    Run(const AutoTunePlan& plan,
        const JsonType& request,
        const ExecutionOptions& options) const override {
        EvaluationResult result;
        result.trial_results.reserve(plan.trials.size());
        result.build_results.reserve(plan.builds.size());

        std::map<std::string, std::vector<TrialSpec>> trials_by_build_id;
        for (const auto& trial : plan.trials) {
            trials_by_build_id[trial.build_id].emplace_back(trial);
        }

        uint64_t trial_ordinal = 0;
        uint64_t build_ordinal = 0;
        for (const auto& build : plan.builds) {
            ++build_ordinal;
            if (!build.use_existing_index) {
                ++result.executed_build_count;
            }

            std::cerr << "[AutoTune] running build group " << build_ordinal << "/"
                      << plan.builds.size() << " " << build.build_id
                      << " index=" << build.index_name
                      << " use_existing_index=" << build.use_existing_index << std::endl;
            auto build_result = RunBuild(build, request, options);
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
                auto trial_result =
                    RunSearchTrial(trial, build_result, request, request["constraints"], options);
                std::cerr << "[AutoTune] finished trial " << trial.trial_id
                          << " status=" << trial_result["status"].get<std::string>()
                          << " elapsed_seconds=" << trial_result["elapsed_seconds"].get<double>()
                          << std::endl;
                result.trial_results.emplace_back(std::move(trial_result));
            }

            CleanupBuildArtifact(build, options);
            result.build_results.emplace_back(std::move(build_result));
        }

        return result;
    }
};

}  // namespace

const EvaluationStrategy&
FullGridEvaluationStrategy() {
    static const FullGridEvaluationStrategyImpl strategy;
    return strategy;
}

}  // namespace vsag::autotune::internal
