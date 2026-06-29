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

constexpr const char* kIndexIvf = "ivf";

const std::vector<IndexDefaultCandidateParam>&
IvfDefaultCandidateParams() {
    static const std::vector<IndexDefaultCandidateParam> params = {
        IndexDefaultCandidateParam{{"create_params", "index_param", "base_quantization_type"},
                                   CandidateParamScope::Build,
                                   JsonType::array({"fp32", "sq8_uniform"})},
        IndexDefaultCandidateParam{{"create_params", "index_param", "buckets_count"},
                                   CandidateParamScope::Build,
                                   JsonType::array({1024, 2048})},
        IndexDefaultCandidateParam{{"search_params", kIndexIvf, "scan_buckets_count"},
                                   CandidateParamScope::Search,
                                   JsonType::array({16, 32, 64})},
    };
    return params;
}

const std::vector<IndexDefaultParam>&
IvfFixedDefaults() {
    static const std::vector<IndexDefaultParam> defaults = {
        IndexDefaultParam{{"create_params", "index_param", "partition_strategy_type"}, "ivf"},
        IndexDefaultParam{{"create_params", "index_param", "ivf_train_type"}, "kmeans"},
    };
    return defaults;
}

void
ValidateIvfSpec(const JsonType& index_spec) {
    ValidateCommonCreateParams(index_spec);
}

}  // namespace

const IndexTunePolicy&
IvfTunePolicy() {
    static const IndexTunePolicy policy{
        kIndexIvf, &IvfDefaultCandidateParams, &IvfFixedDefaults, &ValidateIvfSpec};
    return policy;
}

}  // namespace vsag::autotune::internal
