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

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "autotune_types.h"

namespace vsag::autotune::internal {

enum class MetricScope { Shared, Workload };

struct MetricDefinition {
    const char* name;
    MetricScope scope;
    ObjectiveDirection direction;
    bool available_for_existing_index;
};

struct ConstraintViolation {
    std::string name;
    double expected{0.0};
    std::optional<double> actual;
};

struct ConstraintEvaluation {
    std::vector<ConstraintViolation> violations;
};

const MetricDefinition&
GetMetricDefinition(const std::string& name);

void
ValidateConstraints(const ConstraintMap& constraints,
                    bool use_existing_index,
                    const std::string& field_path);

ObjectiveSpec
ResolveObjective(const std::string& metric, bool use_existing_index, const std::string& field_path);

ConstraintEvaluation
EvaluateConstraints(const ConstraintMap& constraints, const MetricMap& metrics);

JsonType
ConstraintEvaluationToJson(const ConstraintEvaluation& evaluation);

}  // namespace vsag::autotune::internal
