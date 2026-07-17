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

#include <filesystem>
#include <stdexcept>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

using Seconds = std::chrono::duration<double>;

}  // namespace

double
ElapsedSeconds(const Clock::time_point& start) {
    return std::chrono::duration_cast<Seconds>(Clock::now() - start).count();
}

void
Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool
PathsAlias(const std::string& left, const std::string& right) {
    if (left.empty() || right.empty()) {
        return false;
    }
    std::error_code error;
    if (std::filesystem::exists(left, error) && !error && std::filesystem::exists(right, error) &&
        !error) {
        error.clear();
        const bool equivalent = std::filesystem::equivalent(left, right, error);
        if (!error) {
            return equivalent;
        }
    }
    error.clear();
    const auto canonical_left = std::filesystem::weakly_canonical(left, error);
    if (error) {
        return std::filesystem::absolute(left).lexically_normal() ==
               std::filesystem::absolute(right).lexically_normal();
    }
    error.clear();
    const auto canonical_right = std::filesystem::weakly_canonical(right, error);
    if (error) {
        return canonical_left == std::filesystem::absolute(right).lexically_normal();
    }
    return canonical_left == canonical_right;
}

MetricMap
MetricsFromJson(const JsonType& value) {
    MetricMap metrics;
    if (!value.is_object()) {
        return metrics;
    }
    for (const auto& item : value.items()) {
        if (item.value().is_number()) {
            metrics[item.key()] = item.value().get<double>();
        }
    }
    return metrics;
}

}  // namespace vsag::autotune::internal
