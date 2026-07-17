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

#include "autotune_planner.h"

#include <filesystem>
#include <iomanip>
#include <map>
#include <sstream>
#include <utility>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

std::string
make_ordinal_id(const std::string& index_name, const std::string& suffix, uint64_t ordinal) {
    std::ostringstream stream;
    stream << index_name << suffix << "-" << std::setw(6) << std::setfill('0') << ordinal;
    return stream.str();
}

std::string
make_build_key(const CandidateSpec& candidate) {
    return JsonType{{"index_name", candidate.index_name},
                    {"create_params", candidate.create_params}}
        .dump();
}

}  // namespace

AutoTunePlan
PlanTrials(const AutoTuneRequest& request,
           const std::vector<CandidateSpec>& candidates,
           const std::string& workspace_path) {
    Require(request.dataset_resolved, "dataset metadata must be resolved before planning trials");
    Require(!candidates.empty(), "no candidates generated");
    Require(candidates.size() <= request.tuning_config.max_trials,
            "trial count exceeds tuning_config.max_trials");
    Require(!workspace_path.empty(), "workspace path must not be empty");

    const bool use_existing_index = !request.index_path.empty();
    AutoTunePlan plan;
    std::map<std::string, uint64_t> build_indexes;
    for (uint64_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        const auto build_key = make_build_key(candidate);
        const auto [position, inserted] = build_indexes.emplace(build_key, plan.builds.size());
        if (inserted) {
            Require(!use_existing_index || plan.builds.empty(),
                    "index_path can only be used with one unique index_name + create_params");
            BuildSpec build;
            build.build_id =
                make_ordinal_id(candidate.index_name, "-build", plan.builds.size() + 1);
            build.index_name = candidate.index_name;
            build.index_path = use_existing_index ? request.index_path
                                                  : (std::filesystem::path(workspace_path) /
                                                     "artifacts" / (build.build_id + ".index"))
                                                        .string();
            build.create_params = candidate.create_params;
            build.use_existing_index = use_existing_index;
            build.cleanup_index_after_build_group =
                !use_existing_index && !request.tuning_config.keep_intermediate;
            plan.builds.emplace_back(std::move(build));
        }

        const auto& build = plan.builds[position->second];
        TrialSpec trial;
        trial.trial_id = make_ordinal_id(candidate.index_name, "-trial", i + 1);
        trial.build_id = build.build_id;
        trial.search_params = candidate.search_params;
        plan.trials.emplace_back(std::move(trial));
    }
    return plan;
}

}  // namespace vsag::autotune::internal
