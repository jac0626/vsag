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

#include <array>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

constexpr const char* kIndexHGraph = "hgraph";
constexpr const char* kIndexIvf = "ivf";

void
ValidateCommonCreateParams(const JsonType& index_spec) {
    const auto index_name = GetString(index_spec, "name", "");
    Require(index_spec.contains("create_params"), index_name + " create_params is required");
    const auto& create_params = index_spec["create_params"];
    Require(create_params.is_object(), index_name + " create_params must be an object");
    Require(create_params.contains("dim"), index_name + " create_params.dim is required");
    Require(create_params.contains("dtype"), index_name + " create_params.dtype is required");
    Require(create_params.contains("metric_type"),
            index_name + " create_params.metric_type is required");
}

void
FillHGraphDefaults(JsonType& index_spec) {
    auto& create_params = EnsureObject(index_spec, "create_params");
    auto& index_param = EnsureObject(create_params, "index_param");
    if (!index_param.contains("base_quantization_type")) {
        index_param["base_quantization_type"] = JsonType::array({"fp32", "sq8_uniform"});
    }
    if (!index_param.contains("max_degree")) {
        index_param["max_degree"] = JsonType::array({16, 32});
    }
    if (!index_param.contains("ef_construction")) {
        index_param["ef_construction"] = JsonType::array({100, 200});
    }

    auto& search_params = EnsureObject(index_spec, "search_params");
    auto& hgraph_params = EnsureObject(search_params, kIndexHGraph);
    if (!hgraph_params.contains("ef_search")) {
        hgraph_params["ef_search"] = JsonType::array({40, 80, 120});
    }
}

void
ValidateHGraphSpec(const JsonType& index_spec) {
    ValidateCommonCreateParams(index_spec);
}

void
FillIvfDefaults(JsonType& index_spec) {
    auto& create_params = EnsureObject(index_spec, "create_params");
    auto& index_param = EnsureObject(create_params, "index_param");
    if (!index_param.contains("partition_strategy_type")) {
        index_param["partition_strategy_type"] = "ivf";
    }
    if (!index_param.contains("base_quantization_type")) {
        index_param["base_quantization_type"] = JsonType::array({"fp32", "sq8_uniform"});
    }
    if (!index_param.contains("buckets_count")) {
        index_param["buckets_count"] = JsonType::array({1024, 2048});
    }
    if (!index_param.contains("ivf_train_type")) {
        index_param["ivf_train_type"] = "kmeans";
    }

    auto& search_params = EnsureObject(index_spec, "search_params");
    auto& ivf_params = EnsureObject(search_params, kIndexIvf);
    if (!ivf_params.contains("scan_buckets_count")) {
        ivf_params["scan_buckets_count"] = JsonType::array({16, 32, 64});
    }
}

void
ValidateIvfSpec(const JsonType& index_spec) {
    ValidateCommonCreateParams(index_spec);
}

const std::array<IndexTunePolicy, 2>&
Policies() {
    static const std::array<IndexTunePolicy, 2> policies = {
        IndexTunePolicy{kIndexHGraph, &FillHGraphDefaults, &ValidateHGraphSpec},
        IndexTunePolicy{kIndexIvf, &FillIvfDefaults, &ValidateIvfSpec},
    };
    return policies;
}

const IndexTunePolicy*
FindPolicy(const std::string& index_name) {
    for (const auto& policy : Policies()) {
        if (index_name == policy.name) {
            return &policy;
        }
    }
    return nullptr;
}

}  // namespace

bool
HasIndexTunePolicy(const std::string& index_name) {
    return FindPolicy(index_name) != nullptr;
}

void
ApplyIndexDefaults(JsonType& index_spec) {
    const auto index_name = GetString(index_spec, "name", "");
    const auto* policy = FindPolicy(index_name);
    Require(policy != nullptr, "unsupported index: " + index_name);
    policy->fill_defaults(index_spec);
}

void
ValidateIndexSpec(const JsonType& index_spec) {
    const auto index_name = GetString(index_spec, "name", "");
    const auto* policy = FindPolicy(index_name);
    Require(policy != nullptr, "unsupported index: " + index_name);
    policy->validate(index_spec);
}

}  // namespace vsag::autotune::internal
