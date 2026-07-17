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

#include "autotune_constraints.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace vsag::autotune::internal {

namespace {

constexpr std::array<MetricDefinition, 9> METRICS = {
    MetricDefinition{"build_seconds", MetricScope::Shared, ObjectiveDirection::Minimize, false},
    MetricDefinition{"index_size_mb", MetricScope::Shared, ObjectiveDirection::Minimize, true},
    MetricDefinition{"index_memory_mb", MetricScope::Shared, ObjectiveDirection::Minimize, true},
    MetricDefinition{"recall_at_k", MetricScope::Workload, ObjectiveDirection::Maximize, true},
    MetricDefinition{"latency_avg_ms", MetricScope::Workload, ObjectiveDirection::Minimize, true},
    MetricDefinition{"latency_p99_ms", MetricScope::Workload, ObjectiveDirection::Minimize, true},
    MetricDefinition{"qps", MetricScope::Workload, ObjectiveDirection::Maximize, true},
    MetricDefinition{"search_seconds", MetricScope::Workload, ObjectiveDirection::Minimize, true},
    MetricDefinition{
        "build_and_search_seconds", MetricScope::Workload, ObjectiveDirection::Minimize, false},
};

const MetricDefinition*
find_metric(const std::string& name) {
    for (const auto& metric : METRICS) {
        if (name == metric.name) {
            return &metric;
        }
    }
    return nullptr;
}

}  // namespace

const MetricDefinition&
GetMetricDefinition(const std::string& name) {
    const auto* metric = find_metric(name);
    if (metric == nullptr) {
        throw std::invalid_argument("unsupported metric: " + name);
    }
    return *metric;
}

void
ValidateConstraints(const ConstraintMap& constraints,
                    bool use_existing_index,
                    const std::string& field_path) {
    if (constraints.empty()) {
        throw std::invalid_argument(field_path + " must not be empty");
    }
    for (const auto& [name, threshold] : constraints) {
        const auto& metric = GetMetricDefinition(name);
        if (use_existing_index && not metric.available_for_existing_index) {
            auto message = field_path;
            message += ".";
            message += name;
            message += " is unavailable for an existing index";
            throw std::invalid_argument(message);
        }
        if (not std::isfinite(threshold) || threshold < 0.0) {
            auto message = field_path;
            message += ".";
            message += name;
            message += " must be a finite non-negative number";
            throw std::invalid_argument(message);
        }
        if (name == "recall_at_k" && threshold > 1.0) {
            throw std::invalid_argument(field_path + ".recall_at_k must be in [0, 1]");
        }
    }
}

ObjectiveSpec
ResolveObjective(const std::string& metric,
                 bool use_existing_index,
                 const std::string& field_path) {
    const auto& definition = GetMetricDefinition(metric);
    if (use_existing_index && not definition.available_for_existing_index) {
        throw std::invalid_argument(field_path +
                                    ".metric is unavailable for an existing index: " + metric);
    }
    return ObjectiveSpec{metric};
}

ConstraintEvaluation
EvaluateConstraints(const ConstraintMap& constraints, const MetricMap& metrics) {
    ConstraintEvaluation evaluation;
    for (const auto& [name, expected] : constraints) {
        const auto& definition = GetMetricDefinition(name);
        const auto actual = metrics.find(name);
        if (actual == metrics.end() || not std::isfinite(actual->second)) {
            evaluation.violations.emplace_back(ConstraintViolation{name, expected, std::nullopt});
            continue;
        }
        const bool satisfied = definition.direction == ObjectiveDirection::Maximize
                                   ? actual->second >= expected
                                   : actual->second <= expected;
        if (not satisfied) {
            evaluation.violations.emplace_back(ConstraintViolation{name, expected, actual->second});
        }
    }
    return evaluation;
}

JsonType
ConstraintEvaluationToJson(const ConstraintEvaluation& evaluation) {
    JsonType result;
    result["satisfied_constraints"] = evaluation.violations.empty();
    result["violated_constraints"] = JsonType::array();
    for (const auto& violation : evaluation.violations) {
        const auto direction = GetMetricDefinition(violation.name).direction;
        JsonType item{
            {"name", violation.name},
            {"comparison", direction == ObjectiveDirection::Maximize ? "at_least" : "at_most"},
            {"expected", violation.expected},
            {"reason", violation.actual.has_value() ? "threshold_not_met" : "missing_metric"}};
        if (violation.actual.has_value()) {
            item["actual"] = violation.actual.value();
        }
        result["violated_constraints"].push_back(std::move(item));
    }
    return result;
}

}  // namespace vsag::autotune::internal
