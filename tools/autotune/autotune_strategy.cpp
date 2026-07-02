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

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

double
MetricOrDefault(const JsonType& trial, const std::string& key, double default_value) {
    if (!trial.contains("metrics") || !trial["metrics"].is_object() ||
        !trial["metrics"].contains(key) || !trial["metrics"][key].is_number()) {
        return default_value;
    }
    return trial["metrics"][key].get<double>();
}

bool
IsSuccessfulTrial(const JsonType& trial) {
    return trial.contains("status") && trial["status"] == "success";
}

bool
IsSatisfiedTrial(const JsonType& trial) {
    return IsSuccessfulTrial(trial) && trial.contains("satisfied_constraints") &&
           trial["satisfied_constraints"].is_boolean() &&
           trial["satisfied_constraints"].get<bool>();
}

bool
SampledTrialLess(const JsonType& left, const JsonType& right) {
    const double inf = std::numeric_limits<double>::infinity();
    const bool left_satisfied = IsSatisfiedTrial(left);
    const bool right_satisfied = IsSatisfiedTrial(right);
    if (left_satisfied != right_satisfied) {
        return left_satisfied;
    }

    if (left_satisfied) {
        const auto left_tuple = std::make_tuple(MetricOrDefault(left, "latency_avg_ms", inf),
                                                MetricOrDefault(left, "memory_peak_mb", inf),
                                                MetricOrDefault(left, "build_seconds", inf),
                                                left["trial_id"].get<std::string>());
        const auto right_tuple = std::make_tuple(MetricOrDefault(right, "latency_avg_ms", inf),
                                                 MetricOrDefault(right, "memory_peak_mb", inf),
                                                 MetricOrDefault(right, "build_seconds", inf),
                                                 right["trial_id"].get<std::string>());
        return left_tuple < right_tuple;
    }

    const auto left_tuple = std::make_tuple(-MetricOrDefault(left, "recall_at_k", -inf),
                                            MetricOrDefault(left, "latency_avg_ms", inf),
                                            left["trial_id"].get<std::string>());
    const auto right_tuple = std::make_tuple(-MetricOrDefault(right, "recall_at_k", -inf),
                                             MetricOrDefault(right, "latency_avg_ms", inf),
                                             right["trial_id"].get<std::string>());
    return left_tuple < right_tuple;
}

std::vector<JsonType>
SelectSampledFinalists(std::vector<JsonType> sampled_results, uint64_t finalist_count) {
    std::vector<JsonType> successful_results;
    for (auto& trial : sampled_results) {
        if (IsSuccessfulTrial(trial)) {
            successful_results.emplace_back(std::move(trial));
        }
    }
    std::sort(successful_results.begin(), successful_results.end(), SampledTrialLess);
    if (successful_results.size() > finalist_count) {
        successful_results.resize(finalist_count);
    }
    return successful_results;
}

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

        result.selection_trial_results = result.trial_results;
        result.strategy_report = JsonType{{"name", "full_grid"}};
        return result;
    }
};

class QuerySamplingEvaluationStrategyImpl final : public EvaluationStrategy {
public:
    EvaluationResult
    Run(const AutoTunePlan& plan,
        const JsonType& request,
        const ExecutionOptions& options) const override {
        EvaluationResult result;
        result.trial_results.reserve(plan.trials.size() + options.finalist_count);
        result.build_results.reserve(plan.builds.size());

        std::map<std::string, std::vector<TrialSpec>> trials_by_build_id;
        std::map<std::string, TrialSpec> trials_by_trial_id;
        for (const auto& trial : plan.trials) {
            trials_by_build_id[trial.build_id].emplace_back(trial);
            trials_by_trial_id.emplace(trial.trial_id, trial);
        }

        std::map<std::string, JsonType> build_results_by_id;
        std::vector<JsonType> sampled_results;
        uint64_t trial_ordinal = 0;
        uint64_t build_ordinal = 0;
        for (const auto& build : plan.builds) {
            ++build_ordinal;
            if (!build.use_existing_index) {
                ++result.executed_build_count;
            }

            std::cerr << "[AutoTune] running sampled build group " << build_ordinal << "/"
                      << plan.builds.size() << " " << build.build_id
                      << " index=" << build.index_name
                      << " use_existing_index=" << build.use_existing_index << std::endl;
            auto build_result = RunBuild(build, request, options);
            std::cerr << "[AutoTune] finished sampled build group " << build.build_id
                      << " status=" << build_result["status"].get<std::string>()
                      << " elapsed_seconds=" << build_result["elapsed_seconds"].get<double>()
                      << std::endl;

            const auto& group_trials = trials_by_build_id[build.build_id];
            ExecutionOptions sampled_options = options;
            sampled_options.query_limit_count = options.sample_query_count;
            for (const auto& trial : group_trials) {
                ++trial_ordinal;
                std::cerr << "[AutoTune] running sampled trial " << trial_ordinal << "/"
                          << plan.trials.size() << " " << trial.trial_id
                          << " index=" << trial.index_name << " build_id=" << trial.build_id
                          << " sample_query_count=" << options.sample_query_count << std::endl;
                auto trial_result = RunSearchTrial(
                    trial, build_result, request, request["constraints"], sampled_options);
                trial_result["candidate_trial_id"] = trial.trial_id;
                trial_result["trial_id"] = trial.trial_id + "-sampled";
                trial_result["evaluation_stage"] = "sampled";
                trial_result["query_limit_count"] = options.sample_query_count;
                std::cerr << "[AutoTune] finished sampled trial "
                          << trial_result["trial_id"].get<std::string>()
                          << " status=" << trial_result["status"].get<std::string>()
                          << " elapsed_seconds=" << trial_result["elapsed_seconds"].get<double>()
                          << std::endl;

                sampled_results.emplace_back(trial_result);
                result.trial_results.emplace_back(std::move(trial_result));
            }

            build_results_by_id.emplace(build.build_id, build_result);
            result.build_results.emplace_back(std::move(build_result));
        }

        auto finalists = SelectSampledFinalists(sampled_results, options.finalist_count);
        std::set<std::string> validated_trial_ids;
        uint64_t finalist_ordinal = 0;
        for (const auto& sampled_finalist : finalists) {
            const auto candidate_trial_id =
                sampled_finalist["candidate_trial_id"].get<std::string>();
            if (validated_trial_ids.find(candidate_trial_id) != validated_trial_ids.end()) {
                continue;
            }
            validated_trial_ids.emplace(candidate_trial_id);

            const auto& trial = trials_by_trial_id.at(candidate_trial_id);
            const auto& build_result = build_results_by_id.at(trial.build_id);
            ++finalist_ordinal;
            std::cerr << "[AutoTune] running full validation trial " << finalist_ordinal << "/"
                      << finalists.size() << " " << trial.trial_id << " index=" << trial.index_name
                      << " build_id=" << trial.build_id << std::endl;
            ExecutionOptions full_options = options;
            full_options.query_limit_count = 0;
            auto trial_result =
                RunSearchTrial(trial, build_result, request, request["constraints"], full_options);
            trial_result["evaluation_stage"] = "full_validation";
            trial_result["selected_by_sampled_trial_id"] = sampled_finalist["trial_id"];
            std::cerr << "[AutoTune] finished full validation trial " << trial.trial_id
                      << " status=" << trial_result["status"].get<std::string>()
                      << " elapsed_seconds=" << trial_result["elapsed_seconds"].get<double>()
                      << std::endl;

            result.selection_trial_results.emplace_back(trial_result);
            result.trial_results.emplace_back(std::move(trial_result));
        }

        for (const auto& build : plan.builds) {
            CleanupBuildArtifact(build, options);
        }

        result.strategy_report =
            JsonType{{"name", "query_sampling"},
                     {"sample_query_count", options.sample_query_count},
                     {"finalist_count", options.finalist_count},
                     {"sampled_trial_count", sampled_results.size()},
                     {"full_validation_trial_count", result.selection_trial_results.size()}};
        return result;
    }
};

}  // namespace

const EvaluationStrategy&
FullGridEvaluationStrategy() {
    static const FullGridEvaluationStrategyImpl strategy;
    return strategy;
}

const EvaluationStrategy&
QuerySamplingEvaluationStrategy() {
    static const QuerySamplingEvaluationStrategyImpl strategy;
    return strategy;
}

const EvaluationStrategy&
GetEvaluationStrategy(const ExecutionOptions& options) {
    if (options.evaluation_strategy == "query_sampling") {
        return QuerySamplingEvaluationStrategy();
    }
    return FullGridEvaluationStrategy();
}

}  // namespace vsag::autotune::internal
