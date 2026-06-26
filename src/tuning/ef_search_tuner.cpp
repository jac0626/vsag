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

#include "tuning/ef_search_tuner.h"

#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <utility>

#include "common.h"

namespace vsag {
namespace {

std::vector<uint64_t>
NormalizeCandidates(std::vector<uint64_t> candidates) {
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

uint64_t
MaxEfSearch(uint64_t topk) {
    if (topk > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(AMPLIFICATION_FACTOR)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return std::max<uint64_t>(static_cast<uint64_t>(AMPLIFICATION_FACTOR) * topk, 1000);
}

bool
MakeSearchParameters(const std::string& base_search_parameters,
                     const std::string& index_name,
                     uint64_t ef_search,
                     std::string& search_parameters,
                     std::string& error_message) {
    try {
        auto parameters = base_search_parameters.empty()
                              ? nlohmann::json::object()
                              : nlohmann::json::parse(base_search_parameters);
        if (not parameters.is_object()) {
            error_message = "base search parameters must be a json object";
            return false;
        }
        if (not parameters.contains(index_name) || not parameters[index_name].is_object()) {
            parameters[index_name] = nlohmann::json::object();
        }
        parameters[index_name]["ef_search"] = ef_search;
        search_parameters = parameters.dump();
        return true;
    } catch (const std::exception& e) {
        error_message = fmt::format("failed to parse base search parameters: {}", e.what());
        return false;
    }
}

bool
IsBetterBestEffort(const EfSearchTrialResult& lhs, const EfSearchTrialResult& rhs) {
    if (lhs.evaluation.recall.average != rhs.evaluation.recall.average) {
        return lhs.evaluation.recall.average > rhs.evaluation.recall.average;
    }
    return lhs.candidate.ef_search < rhs.candidate.ef_search;
}

}  // namespace

EfSearchTuner::EfSearchTuner() : EfSearchTuner(InMemoryEvaluationRunner()) {
}

EfSearchTuner::EfSearchTuner(InMemoryEvaluationRunner runner)
    : evaluator_([runner](const EvaluationRequest& request) { return runner.Run(request); }) {
}

EfSearchTuner::EfSearchTuner(EvaluationFunction evaluator) : evaluator_(std::move(evaluator)) {
}

EfSearchTuningReport
EfSearchTuner::Tune(const EfSearchTuningRequest& request) const {
    EfSearchTuningReport report;
    const auto candidates = NormalizeCandidates(request.ef_search_candidates);
    const auto max_ef_search = MaxEfSearch(request.topk);
    uint64_t trial_id = 0;
    uint64_t attempted_trials = 0;

    for (const auto ef_search : candidates) {
        EfSearchTrialResult trial;
        trial.trial_id = trial_id++;
        trial.candidate.ef_search = ef_search;

        if (ef_search == 0) {
            trial.status = TuningTrialStatus::SKIPPED;
            trial.message = "ef_search must be greater than 0";
            report.trials.push_back(trial);
            continue;
        }
        if (request.topk == 0) {
            trial.status = TuningTrialStatus::SKIPPED;
            trial.message = "topk must be greater than 0";
            report.trials.push_back(trial);
            continue;
        }
        if (ef_search > max_ef_search) {
            trial.status = TuningTrialStatus::SKIPPED;
            trial.message = fmt::format("ef_search must be no greater than {}", max_ef_search);
            report.trials.push_back(trial);
            continue;
        }
        if (request.max_trials > 0 && attempted_trials >= request.max_trials) {
            trial.status = TuningTrialStatus::SKIPPED;
            trial.message = fmt::format("budget exceeded: max_trials = {}", request.max_trials);
            report.trials.push_back(trial);
            continue;
        }
        ++attempted_trials;

        EvaluationRequest evaluation_request;
        evaluation_request.index = request.index;
        evaluation_request.queries = request.queries;
        evaluation_request.ground_truth = request.ground_truth;
        evaluation_request.topk = request.topk;
        evaluation_request.query_count = request.query_count;
        if (not MakeSearchParameters(request.base_search_parameters,
                                     request.index_name,
                                     ef_search,
                                     evaluation_request.search_parameters,
                                     trial.message)) {
            trial.status = TuningTrialStatus::FAILED;
            report.trials.push_back(trial);
            continue;
        }

        trial.evaluation = evaluator_(evaluation_request);
        if (trial.evaluation.Succeeded()) {
            trial.status = TuningTrialStatus::COMPLETED;
            if (not report.best_effort.has_value() ||
                IsBetterBestEffort(trial, report.best_effort.value())) {
                report.best_effort = trial;
            }
            if (not report.recommendation.has_value() &&
                trial.evaluation.recall.average >= request.target_recall) {
                report.recommendation = trial;
            }
        } else {
            trial.status = TuningTrialStatus::FAILED;
            trial.message = trial.evaluation.error_message;
        }
        report.trials.push_back(trial);
    }

    return report;
}

}  // namespace vsag
