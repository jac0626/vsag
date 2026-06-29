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
#include <map>
#include <sstream>
#include <utility>

#include "autotune_index_policy.h"
#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

struct BuildGroupPlan {
    std::string build_id;
    std::string index_path;
    std::string index_name;
    JsonType create_params;
};

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
MakeOrdinalId(const std::string& index_name, const std::string& suffix, uint64_t ordinal) {
    std::ostringstream oss;
    oss << index_name << suffix << "-" << std::setw(6) << std::setfill('0') << ordinal;
    return oss.str();
}

std::string
MakeTrialId(const std::string& index_name, uint64_t ordinal) {
    return MakeOrdinalId(index_name, "", ordinal);
}

std::string
MakeBuildId(const std::string& index_name, uint64_t ordinal) {
    return MakeOrdinalId(index_name, "-build", ordinal);
}

std::string
FormatPath(const std::vector<std::string>& path) {
    std::string formatted;
    for (const auto& segment : path) {
        formatted += "/";
        formatted += segment;
    }
    return formatted;
}

void
ValidateDefaultCandidateScopes(const std::string& index_name) {
    for (const auto& param : GetIndexDefaultCandidateParams(index_name)) {
        Require(!param.path.empty(),
                index_name + " default candidate param path must not be empty");
        if (param.scope == CandidateParamScope::Build) {
            Require(param.path[0] == "create_params",
                    index_name +
                        " build-scoped default candidate param must be under create_params: " +
                        FormatPath(param.path));
        } else {
            Require(param.path[0] == "search_params",
                    index_name +
                        " search-scoped default candidate param must be under search_params: " +
                        FormatPath(param.path));
        }
    }
}

std::string
MakeBuildKey(const CandidateSpec& candidate) {
    ValidateDefaultCandidateScopes(candidate.index_name);
    const JsonType build_identity =
        JsonType{{"index_name", candidate.index_name}, {"create_params", candidate.create_params}};
    return build_identity.dump();
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

AutoTunePlan
PlanTrials(const JsonType& request,
           const std::vector<CandidateSpec>& candidates,
           const ExecutionOptions& options) {
    Require(!candidates.empty(), "no candidates generated");
    if (options.max_trials > 0) {
        Require(candidates.size() <= options.max_trials,
                "candidate count exceeds execution.max_trials");
    }

    const std::string existing_index_path = GetString(request, "index_path", "");
    if (!existing_index_path.empty()) {
        Require(std::filesystem::exists(existing_index_path),
                "index_path does not exist: " + existing_index_path);
    }

    std::map<std::string, BuildGroupPlan> build_groups;
    std::vector<std::string> build_keys;
    build_keys.reserve(candidates.size());
    uint64_t build_ordinal = 0;
    for (const auto& candidate : candidates) {
        auto build_key = MakeBuildKey(candidate);
        build_keys.emplace_back(build_key);
        auto& group = build_groups[build_key];
        if (group.build_id.empty()) {
            ++build_ordinal;
            group.build_id = MakeBuildId(candidate.index_name, build_ordinal);
            group.index_name = candidate.index_name;
            group.create_params = candidate.create_params;
            group.index_path = (std::filesystem::path(options.workspace_path) / "trials" /
                                (group.build_id + ".index"))
                                   .string();
        }
    }

    Require(existing_index_path.empty() || build_groups.size() == 1,
            "index_path can only be used when indexes expand to one build candidate");

    const bool use_existing_index = !existing_index_path.empty();
    if (use_existing_index) {
        for (auto& build_group : build_groups) {
            build_group.second.index_path = existing_index_path;
        }
    }

    AutoTunePlan plan;
    std::map<std::string, bool> emitted_builds;
    for (const auto& build_key : build_keys) {
        if (emitted_builds[build_key]) {
            continue;
        }
        emitted_builds[build_key] = true;
        const auto& build_group = build_groups[build_key];
        plan.builds.emplace_back(BuildSpec{build_group.build_id,
                                           build_group.index_name,
                                           build_group.index_path,
                                           build_group.create_params,
                                           use_existing_index,
                                           !use_existing_index});
    }

    std::vector<TrialSpec> trials;
    for (uint64_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        auto& build_group = build_groups[build_keys[i]];

        const auto trial_id = MakeTrialId(candidate.index_name, i + 1);
        trials.emplace_back(TrialSpec{trial_id,
                                      build_group.build_id,
                                      candidate.index_name,
                                      "search",
                                      build_group.index_path,
                                      candidate.create_params,
                                      candidate.search_params});
    }
    plan.trials = std::move(trials);
    return plan;
}

}  // namespace vsag::autotune::internal
