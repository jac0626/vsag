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

#include "autotune_index_policy.h"

namespace vsag::autotune::internal {

namespace {

constexpr const char* kIndexHGraph = "hgraph";

const std::vector<IndexDefaultCandidateParam>&
HGraphDefaultCandidateParams() {
    static const std::vector<IndexDefaultCandidateParam> params = {
        IndexDefaultCandidateParam{{"create_params", "index_param", "base_quantization_type"},
                                   CandidateParamScope::Build,
                                   JsonType::array({"fp32", "sq8_uniform"})},
        IndexDefaultCandidateParam{{"create_params", "index_param", "max_degree"},
                                   CandidateParamScope::Build,
                                   JsonType::array({16, 32})},
        IndexDefaultCandidateParam{{"create_params", "index_param", "ef_construction"},
                                   CandidateParamScope::Build,
                                   JsonType::array({100, 200})},
        IndexDefaultCandidateParam{{"search_params", kIndexHGraph, "ef_search"},
                                   CandidateParamScope::Search,
                                   JsonType::array({40, 80, 120})},
    };
    return params;
}

const std::vector<IndexDefaultParam>&
HGraphFixedDefaults() {
    static const std::vector<IndexDefaultParam> defaults;
    return defaults;
}

void
ValidateHGraphSpec(const JsonType& index_spec) {
    ValidateCommonCreateParams(index_spec);
}

}  // namespace

const IndexTunePolicy&
HGraphTunePolicy() {
    static const IndexTunePolicy policy{
        kIndexHGraph, &HGraphDefaultCandidateParams, &HGraphFixedDefaults, &ValidateHGraphSpec};
    return policy;
}

}  // namespace vsag::autotune::internal
