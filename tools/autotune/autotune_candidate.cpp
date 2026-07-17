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

#include "autotune_candidate.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <utility>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

constexpr const char* TRIAL_LIMIT_ERROR = "trial count exceeds tuning_config.max_trials";
constexpr const char* EXPANSION_LIMIT_ERROR =
    "candidate value expansion exceeds the V1 safety limit";
constexpr const char* COMBINATION_LIMIT_ERROR =
    "candidate combination expansion exceeds the V1 safety limit";
constexpr const char* DEPTH_LIMIT_ERROR =
    "candidate expression nesting or object field count exceeds the V1 safety limit";

struct expansion_budget {
    uint64_t remaining;
    const char* limit_error;
};

void
consume_expansion_budget(expansion_budget* budget, uint64_t count) {
    if (budget == nullptr) {
        return;
    }
    Require(count <= budget->remaining, budget->limit_error);
    budget->remaining -= count;
}

using ExpansionCallback = std::function<void(const JsonType&)>;

int64_t
get_range_int64(const JsonType& range, const std::string& key) {
    const auto& value = range[key];
    Require(value.is_number_integer(), "$range " + key + " must be an integer");
    if (value.is_number_unsigned()) {
        const auto unsigned_value = value.get<uint64_t>();
        Require(unsigned_value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "$range " + key + " exceeds int64 range");
        return static_cast<int64_t>(unsigned_value);
    }
    return value.get<int64_t>();
}

void
visit_expanded_json(const JsonType& value,
                    uint64_t max_values,
                    expansion_budget* budget,
                    uint64_t depth,
                    const ExpansionCallback& callback);

void
visit_range(const JsonType& range,
            uint64_t max_values,
            expansion_budget* budget,
            const ExpansionCallback& callback) {
    Require(range.is_object(), "$range must be an object");
    Require(range.size() == 3 && range.contains("start") && range.contains("stop") &&
                range.contains("step"),
            "$range requires start, stop and step");
    Require(range["start"].is_number() && range["stop"].is_number() && range["step"].is_number(),
            "$range start, stop and step must be numbers");

    const bool integer_values = range["start"].is_number_integer() &&
                                range["stop"].is_number_integer() &&
                                range["step"].is_number_integer();
    if (integer_values) {
        const auto start = get_range_int64(range, "start");
        const auto stop = get_range_int64(range, "stop");
        const auto step = get_range_int64(range, "step");
        Require(step != 0, "$range step must not be zero");
        Require((start <= stop && step > 0) || (start >= stop && step < 0),
                "$range step direction does not reach stop");

        int64_t current = start;
        while ((step > 0 && current <= stop) || (step < 0 && current >= stop)) {
            consume_expansion_budget(budget, 1);
            callback(JsonType(current));
            if (step > 0) {
                const auto distance = static_cast<uint64_t>(stop) - static_cast<uint64_t>(current);
                if (distance < static_cast<uint64_t>(step)) {
                    break;
                }
            } else {
                const auto distance = static_cast<uint64_t>(current) - static_cast<uint64_t>(stop);
                const auto magnitude = uint64_t{0} - static_cast<uint64_t>(step);
                if (distance < magnitude) {
                    break;
                }
            }
            current += step;
        }
        return;
    }

    const double start = range["start"].get<double>();
    const double stop = range["stop"].get<double>();
    const double step = range["step"].get<double>();
    Require(std::isfinite(start) && std::isfinite(stop) && std::isfinite(step),
            "$range start, stop and step must be finite");
    Require(step != 0.0, "$range step must not be zero");
    Require((start <= stop && step > 0.0) || (start >= stop && step < 0.0),
            "$range step direction does not reach stop");

    const long double raw_intervals =
        (static_cast<long double>(stop) - static_cast<long double>(start)) /
        static_cast<long double>(step);
    const long double nearest_intervals = std::round(raw_intervals);
    const long double interval_tolerance =
        static_cast<long double>(std::numeric_limits<double>::epsilon()) *
        std::max(std::abs(raw_intervals), 1.0L) * 8.0L;
    const long double interval_count =
        std::abs(raw_intervals - nearest_intervals) <= interval_tolerance
            ? nearest_intervals
            : std::floor(raw_intervals);
    Require(interval_count >= 0.0L && std::isfinite(interval_count),
            "$range generated an invalid number of values");

    const auto emission_limit =
        max_values == std::numeric_limits<uint64_t>::max() ? max_values : max_values + 1;
    if (max_values == std::numeric_limits<uint64_t>::max()) {
        Require(interval_count < static_cast<long double>(max_values),
                "candidate expansion is too large");
    }

    std::optional<double> previous;
    for (uint64_t i = 0; i < emission_limit && static_cast<long double>(i) <= interval_count; ++i) {
        const auto generated = static_cast<long double>(start) +
                               static_cast<long double>(i) * static_cast<long double>(step);
        const auto output = static_cast<double>(generated);
        Require(std::isfinite(output), "$range generated a non-finite value");
        Require(!previous.has_value() || output != *previous,
                "$range step is too small to advance");
        previous = output;
        consume_expansion_budget(budget, 1);
        callback(JsonType(output));
    }
}

void
visit_object_fields(const JsonType& object,
                    const JsonType::const_iterator& field,
                    uint64_t max_values,
                    expansion_budget* budget,
                    uint64_t depth,
                    JsonType& partial,
                    const ExpansionCallback& callback) {
    Require(depth <= MAX_V1_EXPANSION_DEPTH, DEPTH_LIMIT_ERROR);
    if (field == object.end()) {
        consume_expansion_budget(budget, 1);
        callback(partial);
        return;
    }

    const auto& key = field.key();
    const auto next = std::next(field);
    visit_expanded_json(
        field.value(), max_values, budget, depth + 1, [&](const JsonType& expanded_value) {
            partial[key] = expanded_value;
            visit_object_fields(object, next, max_values, budget, depth + 1, partial, callback);
        });
}

void
visit_expanded_json(const JsonType& value,
                    uint64_t max_values,
                    expansion_budget* budget,
                    uint64_t depth,
                    const ExpansionCallback& callback) {
    Require(depth <= MAX_V1_EXPANSION_DEPTH, DEPTH_LIMIT_ERROR);
    if (value.is_object()) {
        if (value.contains("$range")) {
            Require(value.size() == 1, "$range cannot be mixed with other keys");
            visit_range(value["$range"], max_values, budget, callback);
            return;
        }

        JsonType partial = JsonType::object();
        visit_object_fields(value, value.begin(), max_values, budget, depth + 1, partial, callback);
        return;
    }

    if (value.is_array()) {
        Require(!value.empty(), "candidate array must not be empty");
        std::set<std::string> seen;
        for (const auto& item : value) {
            visit_expanded_json(
                item, max_values, budget, depth + 1, [&](const JsonType& expanded_item) {
                    if (seen.emplace(expanded_item.dump()).second) {
                        callback(expanded_item);
                    }
                });
        }
        return;
    }

    consume_expansion_budget(budget, 1);
    callback(value);
}

void
visit_json_with_limit(const JsonType& value,
                      uint64_t max_values,
                      const std::string& limit_error,
                      expansion_budget* budget,
                      const ExpansionCallback& callback) {
    Require(max_values > 0, limit_error);
    uint64_t emitted = 0;
    visit_expanded_json(value, max_values, budget, 0, [&](const JsonType& expanded_value) {
        Require(emitted < max_values, limit_error);
        ++emitted;
        callback(expanded_value);
    });
    Require(emitted > 0, "candidate expansion generated no value");
}

}  // namespace

std::vector<JsonType>
ExpandJson(const JsonType& value) {
    std::vector<JsonType> result;
    visit_json_with_limit(
        value,
        std::numeric_limits<uint64_t>::max(),
        "candidate expansion is too large",
        nullptr,
        [&](const JsonType& expanded_value) { result.emplace_back(expanded_value); });
    return result;
}

std::vector<CandidateSpec>
GenerateCandidates(const AutoTuneRequest& request, const CandidateGenerator& generator) {
    Require(request.dataset_resolved,
            "dataset metadata must be resolved before generating candidates");
    Require(request.workload.top_k > 0, "workload top_k must be positive");
    Require(request.workload.concurrency > 0, "workload concurrency must be positive");
    Require(request.tuning_config.max_trials > 0, "tuning_config.max_trials must be positive");

    const auto trial_limit = request.tuning_config.max_trials;
    std::vector<CandidateSpec> candidates;
    std::set<std::string> seen_candidates;
    expansion_budget budget{MAX_V1_EXPANDED_COMBINATIONS, COMBINATION_LIMIT_ERROR};

    for (const auto& index : request.indexes) {
        std::set<std::string> seen_create_candidates;
        const auto process_create_candidate = [&](const JsonType& create_params) {
            visit_json_with_limit(
                index.search_params,
                MAX_V1_EXPANDED_VALUES,
                EXPANSION_LIMIT_ERROR,
                &budget,
                [&](const JsonType& user_search_params) {
                    const SearchCandidateContext generator_context{
                        index, create_params, request.workload, user_search_params};
                    auto search_patches = generator.GenerateSearchPatches(generator_context);
                    if (search_patches.empty()) {
                        search_patches.emplace_back(JsonType::object());
                    }
                    for (const auto& search_patch : search_patches) {
                        JsonType search_spec = user_search_params;
                        MergeCandidatePatch(
                            search_spec, search_patch, index.name + " search_params");
                        visit_json_with_limit(
                            search_spec,
                            MAX_V1_EXPANDED_VALUES,
                            EXPANSION_LIMIT_ERROR,
                            &budget,
                            [&](const JsonType& search_params) {
                                const auto candidate_key = JsonType{
                                    {"index_name", index.name},
                                    {"create_params", create_params},
                                    {"search_params",
                                     search_params}}.dump();
                                if (!seen_candidates.emplace(candidate_key).second) {
                                    return;
                                }
                                Require(candidates.size() < trial_limit, TRIAL_LIMIT_ERROR);

                                CandidateSpec candidate;
                                candidate.index_name = index.name;
                                candidate.create_params = create_params;
                                candidate.search_params = search_params;
                                candidates.emplace_back(std::move(candidate));
                            });
                    }
                });
        };

        std::optional<JsonType> existing_create_candidate;
        visit_json_with_limit(
            index.create_params,
            MAX_V1_EXPANDED_VALUES,
            EXPANSION_LIMIT_ERROR,
            &budget,
            [&](const JsonType& user_create_params) {
                const BuildCandidateContext generator_context{request, index, user_create_params};
                auto build_patches = generator.GenerateBuildPatches(generator_context);
                if (build_patches.empty()) {
                    build_patches.emplace_back(JsonType::object());
                }
                for (const auto& build_patch : build_patches) {
                    JsonType create_spec = user_create_params;
                    MergeCandidatePatch(create_spec, build_patch, index.name + " create_params");
                    visit_json_with_limit(
                        create_spec,
                        MAX_V1_EXPANDED_VALUES,
                        EXPANSION_LIMIT_ERROR,
                        &budget,
                        [&](const JsonType& create_params) {
                            if (!seen_create_candidates.emplace(create_params.dump()).second) {
                                return;
                            }
                            if (!request.index_path.empty()) {
                                Require(!existing_create_candidate.has_value(),
                                        "index_path requires exactly one concrete create_params "
                                        "candidate");
                                existing_create_candidate = create_params;
                                return;
                            }
                            process_create_candidate(create_params);
                        });
                }
            });
        if (!request.index_path.empty()) {
            Require(existing_create_candidate.has_value(),
                    "index_path requires exactly one concrete create_params candidate");
            process_create_candidate(*existing_create_candidate);
        }
    }

    Require(!candidates.empty(), "no candidates generated");
    return candidates;
}

std::vector<CandidateSpec>
GenerateCandidates(const AutoTuneRequest& request) {
    return GenerateCandidates(request, GetDefaultCandidateGenerator());
}

}  // namespace vsag::autotune::internal
