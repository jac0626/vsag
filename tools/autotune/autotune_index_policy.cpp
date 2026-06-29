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

const char*
ScopeName(CandidateParamScope scope) {
    switch (scope) {
        case CandidateParamScope::Build:
            return "build";
        case CandidateParamScope::Search:
            return "search";
    }
    return "unknown";
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
ApplyDefault(JsonType& index_spec, const std::vector<std::string>& path, const JsonType& value) {
    Require(!path.empty(), "index policy default path must not be empty");

    JsonType* current = &index_spec;
    for (uint64_t i = 0; i + 1 < path.size(); ++i) {
        current = &EnsureObject(*current, path[i]);
    }

    const auto& key = path.back();
    if (!current->contains(key)) {
        (*current)[key] = value;
    }
}

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

const std::array<IndexTunePolicy, 2>&
Policies() {
    static const std::array<IndexTunePolicy, 2> policies = {
        IndexTunePolicy{
            kIndexHGraph, &HGraphDefaultCandidateParams, &HGraphFixedDefaults, &ValidateHGraphSpec},
        IndexTunePolicy{kIndexIvf, &IvfDefaultCandidateParams, &IvfFixedDefaults, &ValidateIvfSpec},
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

JsonType
DescribeParam(const IndexDefaultCandidateParam& param) {
    return JsonType{{"name", param.path.back()},
                    {"path", FormatPath(param.path)},
                    {"scope", ScopeName(param.scope)},
                    {"default_candidates", param.default_candidates}};
}

JsonType
DescribeDefault(const IndexDefaultParam& param) {
    return JsonType{
        {"name", param.path.back()}, {"path", FormatPath(param.path)}, {"value", param.value}};
}

}  // namespace

bool
HasIndexTunePolicy(const std::string& index_name) {
    return FindPolicy(index_name) != nullptr;
}

const std::vector<IndexDefaultCandidateParam>&
GetIndexDefaultCandidateParams(const std::string& index_name) {
    const auto* policy = FindPolicy(index_name);
    Require(policy != nullptr, "unsupported index: " + index_name);
    return policy->default_candidate_params();
}

JsonType
DescribeIndexTunePolicy(const std::string& index_name) {
    const auto* policy = FindPolicy(index_name);
    Require(policy != nullptr, "unsupported index: " + index_name);

    JsonType description;
    description["name"] = policy->name;
    description["default_candidate_params"] = JsonType::array();
    for (const auto& param : policy->default_candidate_params()) {
        description["default_candidate_params"].push_back(DescribeParam(param));
    }
    description["fixed_defaults"] = JsonType::array();
    for (const auto& param : policy->fixed_defaults()) {
        description["fixed_defaults"].push_back(DescribeDefault(param));
    }
    return description;
}

void
ApplyIndexDefaults(JsonType& index_spec) {
    const auto index_name = GetString(index_spec, "name", "");
    const auto* policy = FindPolicy(index_name);
    Require(policy != nullptr, "unsupported index: " + index_name);
    for (const auto& param : policy->fixed_defaults()) {
        ApplyDefault(index_spec, param.path, param.value);
    }
    for (const auto& param : policy->default_candidate_params()) {
        ApplyDefault(index_spec, param.path, param.default_candidates);
    }
}

void
ValidateIndexSpec(const JsonType& index_spec) {
    const auto index_name = GetString(index_spec, "name", "");
    const auto* policy = FindPolicy(index_name);
    Require(policy != nullptr, "unsupported index: " + index_name);
    policy->validate(index_spec);
}

}  // namespace vsag::autotune::internal
