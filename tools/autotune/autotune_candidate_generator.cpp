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

#include "autotune_candidate_generator.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

constexpr const char* HGRAPH_NAME = "hgraph";
constexpr const char* IVF_NAME = "ivf";

std::string
normalize_index_name_impl(std::string index_name) {
    std::transform(
        index_name.begin(), index_name.end(), index_name.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return index_name;
}

int64_t
saturating_multiply(int64_t value, int64_t factor) {
    if (value > std::numeric_limits<int64_t>::max() / factor) {
        return std::numeric_limits<int64_t>::max();
    }
    return value * factor;
}

template <typename T>
void
append_unique(std::vector<T>& values, std::set<T>& seen, T value) {
    if (seen.emplace(value).second) {
        values.emplace_back(value);
    }
}

uint64_t
positive_uint64_or_zero(const JsonType& value) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        return 0;
    }
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    }
    const auto signed_value = value.get<int64_t>();
    return signed_value > 0 ? static_cast<uint64_t>(signed_value) : 0;
}

const JsonType&
index_params(const JsonType& create_params, const std::string& index_name) {
    static const JsonType empty = JsonType::object();
    if (!create_params.contains("index_param")) {
        return empty;
    }
    Require(create_params["index_param"].is_object(),
            index_name + " create_params.index_param must be an object");
    return create_params["index_param"];
}

std::vector<JsonType>
propose_hgraph_build_candidates(const BuildCandidateContext& context) {
    const auto& user_params = index_params(context.user_create_params, HGRAPH_NAME);
    JsonType patch{{"index_param", JsonType::object()}};
    auto& generated = patch["index_param"];

    if (!user_params.contains("base_quantization_type")) {
        generated["base_quantization_type"] = JsonType::array({"fp32", "sq8_uniform"});
    }
    if (!user_params.contains("max_degree")) {
        generated["max_degree"] = JsonType::array({16, 32});
    }
    if (user_params.contains("ef_construction")) {
        return {std::move(patch)};
    }

    int64_t max_degree = 0;
    if (user_params.contains("max_degree")) {
        const auto& value = user_params["max_degree"];
        if (value.is_number_unsigned()) {
            const auto unsigned_degree = value.get<uint64_t>();
            if (unsigned_degree <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                max_degree = static_cast<int64_t>(unsigned_degree);
            }
        } else if (value.is_number_integer()) {
            max_degree = value.get<int64_t>();
        }
    }

    std::vector<int64_t> candidates;
    std::set<int64_t> seen;
    if (max_degree > 0) {
        append_unique(candidates, seen, std::max<int64_t>(100, max_degree));
        append_unique(candidates, seen, std::max<int64_t>(200, saturating_multiply(max_degree, 2)));
    } else {
        append_unique(candidates, seen, int64_t{100});
        append_unique(candidates, seen, int64_t{200});
    }
    generated["ef_construction"] = candidates;
    return {std::move(patch)};
}

std::vector<JsonType>
propose_ivf_build_candidates(const BuildCandidateContext& context) {
    const auto& user_params = index_params(context.user_create_params, IVF_NAME);
    JsonType patch{{"index_param", JsonType::object()}};
    auto& generated = patch["index_param"];

    if (!user_params.contains("base_quantization_type")) {
        generated["base_quantization_type"] = JsonType::array({"fp32", "sq8_uniform"});
    }
    if (user_params.contains("buckets_count")) {
        return {std::move(patch)};
    }

    const auto native_limit = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    const auto max_buckets = std::min(context.request.dataset.base_count, native_limit);
    std::vector<uint64_t> candidates;
    std::set<uint64_t> seen;
    append_unique(candidates, seen, std::min<uint64_t>(1024, max_buckets));
    append_unique(candidates, seen, std::min<uint64_t>(2048, max_buckets));
    generated["buckets_count"] = candidates;
    return {std::move(patch)};
}

std::vector<JsonType>
propose_hgraph_search_candidates(const SearchCandidateContext& context) {
    JsonType patch{{HGRAPH_NAME, JsonType::object()}};
    auto& generated = patch[HGRAPH_NAME];
    if (context.user_search_params.contains(HGRAPH_NAME)) {
        Require(context.user_search_params[HGRAPH_NAME].is_object(),
                "hgraph search_params.hgraph must be an object");
        if (context.user_search_params[HGRAPH_NAME].contains("ef_search")) {
            return {std::move(patch)};
        }
    }

    const auto top_k = static_cast<int64_t>(context.workload.top_k);
    std::vector<int64_t> candidates;
    std::set<int64_t> seen;
    append_unique(candidates, seen, std::max<int64_t>(40, top_k));
    append_unique(candidates, seen, std::max<int64_t>(80, saturating_multiply(top_k, 2)));
    append_unique(candidates, seen, std::max<int64_t>(120, saturating_multiply(top_k, 4)));
    generated["ef_search"] = candidates;
    return {std::move(patch)};
}

std::vector<JsonType>
propose_ivf_search_candidates(const SearchCandidateContext& context) {
    JsonType patch{{IVF_NAME, JsonType::object()}};
    auto& generated = patch[IVF_NAME];
    if (context.user_search_params.contains(IVF_NAME)) {
        Require(context.user_search_params[IVF_NAME].is_object(),
                "ivf search_params.ivf must be an object");
        if (context.user_search_params[IVF_NAME].contains("scan_buckets_count")) {
            return {std::move(patch)};
        }
    }

    const auto& create_index_params = index_params(context.create_params, IVF_NAME);
    if (!create_index_params.contains("buckets_count")) {
        return {std::move(patch)};
    }
    const auto buckets_count = positive_uint64_or_zero(create_index_params["buckets_count"]);
    if (buckets_count == 0) {
        return {std::move(patch)};
    }

    const auto native_limit = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    const auto max_scan = std::min(buckets_count, native_limit);
    std::vector<uint64_t> candidates;
    std::set<uint64_t> seen;
    if (max_scan <= 16) {
        append_unique(candidates, seen, uint64_t{1});
        append_unique(candidates, seen, std::max<uint64_t>(1, max_scan / 4));
        append_unique(candidates, seen, std::max<uint64_t>(1, max_scan / 2));
        append_unique(candidates, seen, max_scan);
    } else {
        append_unique(candidates, seen, std::min<uint64_t>(16, max_scan));
        append_unique(candidates, seen, std::min<uint64_t>(32, max_scan));
        append_unique(candidates, seen, std::min<uint64_t>(64, max_scan));
    }
    generated["scan_buckets_count"] = candidates;
    return {std::move(patch)};
}

using BuildProposal = std::vector<JsonType> (*)(const BuildCandidateContext&);
using SearchProposal = std::vector<JsonType> (*)(const SearchCandidateContext&);

struct index_tuning_descriptor {
    const char* name;
    BuildProposal propose_build_candidates;
    SearchProposal propose_search_candidates;
};

const std::array<index_tuning_descriptor, 2>&
index_tuning_descriptors() {
    static const std::array<index_tuning_descriptor, 2> descriptors{{
        {HGRAPH_NAME, &propose_hgraph_build_candidates, &propose_hgraph_search_candidates},
        {IVF_NAME, &propose_ivf_build_candidates, &propose_ivf_search_candidates},
    }};
    return descriptors;
}

const index_tuning_descriptor*
find_index_tuning_descriptor(const std::string& index_name) {
    const auto normalized = normalize_index_name_impl(index_name);
    const auto& descriptors = index_tuning_descriptors();
    const auto* const descriptor =
        std::find_if(descriptors.begin(), descriptors.end(), [&](const auto& candidate) {
            return normalized == candidate.name;
        });
    return descriptor == descriptors.end() ? nullptr : &*descriptor;
}

class DefaultCandidateGenerator final : public CandidateGenerator {
public:
    [[nodiscard]] std::vector<JsonType>
    GenerateBuildPatches(const BuildCandidateContext& context) const override {
        const auto* descriptor = find_index_tuning_descriptor(context.index.name);
        Require(descriptor != nullptr, "unsupported index: " + context.index.name);
        return descriptor->propose_build_candidates(context);
    }

    [[nodiscard]] std::vector<JsonType>
    GenerateSearchPatches(const SearchCandidateContext& context) const override {
        const auto* descriptor = find_index_tuning_descriptor(context.index.name);
        Require(descriptor != nullptr, "unsupported index: " + context.index.name);
        return descriptor->propose_search_candidates(context);
    }
};

}  // namespace

const CandidateGenerator&
GetDefaultCandidateGenerator() {
    static const DefaultCandidateGenerator generator;
    return generator;
}

std::string
NormalizeIndexName(std::string index_name) {
    return normalize_index_name_impl(std::move(index_name));
}

bool
SupportsDefaultCandidateGeneration(const std::string& index_name) {
    return find_index_tuning_descriptor(index_name) != nullptr;
}

void
MergeCandidatePatch(JsonType& target, const JsonType& patch, const std::string& path) {
    Require(target.is_object(), path + " must be an object");
    Require(patch.is_object(), "candidate generator patch for " + path + " must be an object");
    for (const auto& item : patch.items()) {
        const auto child_path = path + "." + item.key();
        if (!target.contains(item.key())) {
            target[item.key()] = item.value();
            continue;
        }
        Require(target[item.key()].is_object() && item.value().is_object(),
                "candidate generator must not overwrite user parameter " + child_path);
        MergeCandidatePatch(target[item.key()], item.value(), child_path);
    }
}

}  // namespace vsag::autotune::internal
