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

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <set>
#include <sstream>
#include <utility>

#include "autotune_index_policy.h"
#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

std::vector<JsonType>
ExpandRange(const JsonType& range) {
    Require(range.is_object(), "$range must be an object");
    Require(range.contains("start") && range.contains("stop") && range.contains("step"),
            "$range requires start, stop and step");
    Require(range["start"].is_number() && range["stop"].is_number() && range["step"].is_number(),
            "$range start, stop and step must be numbers");

    const double start = range["start"].get<double>();
    const double stop = range["stop"].get<double>();
    const double step = range["step"].get<double>();
    Require(step != 0.0, "$range step must not be zero");
    Require((start <= stop && step > 0.0) || (start >= stop && step < 0.0),
            "$range step direction does not reach stop");

    const bool integer_values = range["start"].is_number_integer() &&
                                range["stop"].is_number_integer() &&
                                range["step"].is_number_integer();
    std::vector<JsonType> values;
    double current = start;
    uint64_t guard = 0;
    while ((step > 0.0 && current <= stop + 1e-9) || (step < 0.0 && current >= stop - 1e-9)) {
        if (integer_values) {
            values.emplace_back(static_cast<int64_t>(std::llround(current)));
        } else {
            values.emplace_back(current);
        }
        current += step;
        ++guard;
        Require(guard <= 1000000, "$range generated too many values");
    }
    Require(!values.empty(), "$range generated no value");
    return values;
}

std::string
MakeTrialId(const std::string& index_name, uint64_t ordinal) {
    std::ostringstream oss;
    oss << index_name << "-" << std::setw(6) << std::setfill('0') << ordinal;
    return oss.str();
}

}  // namespace

std::vector<JsonType>
ExpandJson(const JsonType& value) {
    if (value.is_object()) {
        if (value.contains("$value")) {
            Require(value.size() == 1, "$value cannot be mixed with other keys");
            return {value["$value"]};
        }
        if (value.contains("$range")) {
            Require(value.size() == 1, "$range cannot be mixed with other keys");
            return ExpandRange(value["$range"]);
        }

        std::vector<JsonType> partials{JsonType::object()};
        for (auto it = value.begin(); it != value.end(); ++it) {
            auto expanded_values = ExpandJson(it.value());
            std::vector<JsonType> next_partials;
            for (const auto& partial : partials) {
                for (const auto& expanded_value : expanded_values) {
                    JsonType next = partial;
                    next[it.key()] = expanded_value;
                    next_partials.emplace_back(std::move(next));
                }
            }
            partials = std::move(next_partials);
        }
        return partials;
    }

    if (value.is_array()) {
        Require(!value.empty(), "candidate array must not be empty");
        std::vector<JsonType> values;
        values.reserve(value.size());
        for (const auto& item : value) {
            values.emplace_back(item);
        }
        return values;
    }

    return {value};
}

std::vector<CandidateSpec>
GenerateCandidates(const JsonType& request) {
    std::vector<CandidateSpec> candidates;
    for (const auto& raw_index_spec : request["indexes"]) {
        JsonType index_spec = raw_index_spec;
        ApplyIndexDefaults(index_spec);
        ValidateIndexSpec(index_spec);

        const auto index_name = index_spec["name"].get<std::string>();
        auto create_candidates = ExpandJson(index_spec["create_params"]);
        auto search_candidates = ExpandJson(index_spec["search_params"]);

        for (const auto& create_params : create_candidates) {
            for (const auto& search_params : search_candidates) {
                candidates.emplace_back(CandidateSpec{index_name, create_params, search_params});
            }
        }
    }
    return candidates;
}

std::vector<TrialSpec>
PlanTrials(const JsonType& request,
           const std::vector<CandidateSpec>& candidates,
           const ExecutionOptions& options) {
    Require(!candidates.empty(), "no candidates generated");
    if (options.max_trials > 0) {
        Require(candidates.size() <= options.max_trials,
                "candidate count exceeds execution.max_trials");
    }

    const std::string existing_index_path = GetString(request, "index_path", "");
    std::set<std::string> index_names;
    std::set<std::string> create_param_dumps;
    for (const auto& candidate : candidates) {
        index_names.emplace(candidate.index_name);
        create_param_dumps.emplace(candidate.create_params.dump());
    }

    const bool search_only =
        !existing_index_path.empty() && index_names.size() == 1 && create_param_dumps.size() == 1;
    if (!existing_index_path.empty()) {
        Require(std::filesystem::exists(existing_index_path),
                "index_path does not exist: " + existing_index_path);
    }

    std::vector<TrialSpec> trials;
    const auto trial_dir = std::filesystem::path(options.workspace_path) / "trials";
    uint64_t ordinal = 0;
    for (const auto& candidate : candidates) {
        ++ordinal;
        auto trial_id = MakeTrialId(candidate.index_name, ordinal);
        std::string index_path = existing_index_path;
        std::string eval_type = "search";
        if (!search_only) {
            eval_type = "build,search";
            index_path = (trial_dir / (trial_id + ".index")).string();
        }
        trials.emplace_back(TrialSpec{trial_id,
                                      candidate.index_name,
                                      eval_type,
                                      index_path,
                                      candidate.create_params,
                                      candidate.search_params});
    }
    return trials;
}

}  // namespace vsag::autotune::internal
