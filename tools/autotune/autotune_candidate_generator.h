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

#include <string>
#include <vector>

#include "autotune_types.h"

namespace vsag::autotune::internal {

struct BuildCandidateContext {
    const AutoTuneRequest& request;
    const IndexSpec& index;
    const JsonType& user_create_params;
};

struct SearchCandidateContext {
    const IndexSpec& index;
    const JsonType& create_params;
    const WorkloadSpec& workload;
    const JsonType& user_search_params;
};

class CandidateGenerator {
public:
    virtual ~CandidateGenerator() = default;

    [[nodiscard]] virtual std::vector<JsonType>
    GenerateBuildPatches(const BuildCandidateContext& context) const = 0;

    [[nodiscard]] virtual std::vector<JsonType>
    GenerateSearchPatches(const SearchCandidateContext& context) const = 0;
};

const CandidateGenerator&
GetDefaultCandidateGenerator();

std::string
NormalizeIndexName(std::string index_name);

bool
SupportsDefaultCandidateGeneration(const std::string& index_name);

void
MergeCandidatePatch(JsonType& target, const JsonType& patch, const std::string& path);

}  // namespace vsag::autotune::internal
