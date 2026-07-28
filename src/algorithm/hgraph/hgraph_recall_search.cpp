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

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hgraph.h"
#include "impl/logger/logger.h"

namespace vsag {

namespace {

constexpr float RECALL_EPSILON = 1e-6F;
constexpr double REFERENCE_STABILITY_THRESHOLD = 0.995;
constexpr int64_t MIN_REFERENCE_EF_SEARCH = 256;
constexpr int64_t DEFAULT_REFERENCE_EF_SEARCH_CAP = 4096;

using SearchResults = std::vector<std::vector<int64_t>>;

bool
is_valid_recall(float recall) {
    return std::isfinite(recall) and recall > 0.0F and recall <= 1.0F;
}

std::vector<int64_t>
take_search_ids(const DatasetPtr& result, int64_t top_k) {
    CHECK_ARGUMENT(result->GetDim() >= top_k,
                   "not enough vectors to calculate recall-search accuracy");
    return {result->GetIds(), result->GetIds() + top_k};
}

std::vector<int64_t>
run_hgraph_search(const HGraph& index, const DatasetPtr& query, int64_t top_k, int64_t ef_search) {
    SearchRequest request;
    request.query_ = query;
    request.topk_ = top_k;
    request.params_str_ =
        fmt::format(R"({{"hgraph":{{"ef_search":{},"parallelism":1}}}})", ef_search);

    return take_search_ids(index.SearchWithRequest(request), top_k);
}

SearchResults
run_hgraph_searches(const HGraph& index,
                    const std::vector<DatasetPtr>& queries,
                    int64_t top_k,
                    int64_t ef_search) {
    SearchResults results;
    results.reserve(queries.size());
    for (const auto& query : queries) {
        results.push_back(run_hgraph_search(index, query, top_k, ef_search));
    }
    return results;
}

double
recall_at_k(const std::vector<int64_t>& result,
            const std::vector<int64_t>& reference,
            int64_t top_k) {
    CHECK_ARGUMENT(result.size() >= static_cast<uint64_t>(top_k),
                   "not enough search results to calculate recall-search accuracy");
    CHECK_ARGUMENT(reference.size() >= static_cast<uint64_t>(top_k),
                   "not enough reference results to calculate recall-search accuracy");
    std::unordered_multiset<int64_t> expected(reference.begin(), reference.begin() + top_k);
    uint64_t recalled = 0;
    for (int64_t i = 0; i < top_k; ++i) {
        const auto match = expected.find(result[i]);
        if (match != expected.end()) {
            ++recalled;
            expected.erase(match);
        }
    }
    return static_cast<double>(recalled) / static_cast<double>(top_k);
}

double
average_recall(const SearchResults& results, const SearchResults& references, int64_t top_k) {
    CHECK_ARGUMENT(results.size() == references.size(),
                   "recall-search result count does not match reference count");
    double recall_sum = 0.0;
    for (uint64_t i = 0; i < results.size(); ++i) {
        recall_sum += recall_at_k(results[i], references[i], top_k);
    }
    return recall_sum / static_cast<double>(results.size());
}

double
average_recall(const HGraph& index,
               const std::vector<DatasetPtr>& queries,
               const SearchResults& references,
               int64_t top_k,
               int64_t ef_search) {
    return average_recall(run_hgraph_searches(index, queries, top_k, ef_search), references, top_k);
}

struct reference_results {
    SearchResults results;
    int64_t ef_search{0};
};

reference_results
find_reference_results(const HGraph& index,
                       const std::vector<DatasetPtr>& queries,
                       int64_t top_k,
                       int64_t index_size) {
    const auto doubled_top_k = top_k > index_size / 2 ? index_size : top_k * 2;
    auto reference_ef =
        std::min(index_size, std::max<int64_t>(MIN_REFERENCE_EF_SEARCH, doubled_top_k));
    auto reference = run_hgraph_searches(index, queries, top_k, reference_ef);
    if (reference_ef == index_size) {
        return {std::move(reference), reference_ef};
    }

    const auto doubled_start = reference_ef > index_size / 2 ? index_size : reference_ef * 2;
    const auto max_reference_ef =
        std::min(index_size, std::max<int64_t>(DEFAULT_REFERENCE_EF_SEARCH_CAP, doubled_start));
    while (reference_ef < max_reference_ef) {
        const auto next_ef =
            reference_ef > max_reference_ef / 2 ? max_reference_ef : reference_ef * 2;
        auto next_reference = run_hgraph_searches(index, queries, top_k, next_ef);
        const auto stability = average_recall(next_reference, reference, top_k);
        reference = std::move(next_reference);
        reference_ef = next_ef;
        if (stability + RECALL_EPSILON >= REFERENCE_STABILITY_THRESHOLD) {
            logger::info(
                "HGraph recall search selected reference ef_search={} for top_k={}, "
                "stability={}",
                reference_ef,
                top_k,
                stability);
            return {std::move(reference), reference_ef};
        }
    }

    CHECK_ARGUMENT(
        reference_ef == index_size,
        fmt::format("recall-search reference did not stabilize by ef_search={} for top_k={}",
                    reference_ef,
                    top_k));
    return {std::move(reference), reference_ef};
}

std::vector<int64_t>
calibrate_ef_search(const HGraph& index,
                    const std::vector<DatasetPtr>& queries,
                    const std::vector<RecallTarget>& targets,
                    int64_t max_ef_search) {
    CHECK_ARGUMENT(not queries.empty(), "recall search has no calibration queries");
    CHECK_ARGUMENT(not targets.empty(), "recall search has no recall targets");

    std::vector<int64_t> selected_efs(targets.size());
    std::vector<uint64_t> target_order;
    target_order.reserve(targets.size());
    for (uint64_t i = 0; i < targets.size(); ++i) {
        target_order.push_back(i);
    }
    std::sort(target_order.begin(), target_order.end(), [&targets](uint64_t lhs, uint64_t rhs) {
        if (targets[lhs].top_k != targets[rhs].top_k) {
            return targets[lhs].top_k < targets[rhs].top_k;
        }
        return targets[lhs].recall < targets[rhs].recall;
    });

    for (uint64_t group_begin = 0; group_begin < target_order.size();) {
        const auto top_k = targets[target_order[group_begin]].top_k;
        auto group_end = group_begin + 1;
        while (group_end < target_order.size() and
               targets[target_order[group_end]].top_k == top_k) {
            ++group_end;
        }

        auto reference = find_reference_results(index, queries, top_k, max_ef_search);
        std::unordered_map<int64_t, double> recall_cache;
        recall_cache.emplace(reference.ef_search, 1.0);
        auto get_recall = [&](int64_t ef_search) {
            const auto cached = recall_cache.find(ef_search);
            if (cached != recall_cache.end()) {
                return cached->second;
            }
            const auto recall = average_recall(index, queries, reference.results, top_k, ef_search);
            recall_cache.emplace(ef_search, recall);
            return recall;
        };

        int64_t lower_bound = top_k;
        for (auto position = group_begin; position < group_end; ++position) {
            const auto target_index = target_order[position];
            const auto& target = targets[target_index];
            auto meets_recall = [&](int64_t ef_search) {
                return get_recall(ef_search) + RECALL_EPSILON >= target.recall;
            };

            if (meets_recall(lower_bound)) {
                selected_efs[target_index] = lower_bound;
                continue;
            }

            int64_t low = lower_bound + 1;
            int64_t high = lower_bound;
            while (high < reference.ef_search) {
                high = high > reference.ef_search / 2 ? reference.ef_search : high * 2;
                if (meets_recall(high)) {
                    break;
                }
                low = high + 1;
            }
            CHECK_ARGUMENT(
                meets_recall(high),
                fmt::format(
                    "target recall {} at top_k {} cannot be reached", target.recall, top_k));

            while (low < high) {
                const auto middle = low + (high - low) / 2;
                if (meets_recall(middle)) {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            selected_efs[target_index] = low;
            lower_bound = low;
        }
        group_begin = group_end;
    }
    return selected_efs;
}

}  // namespace

DatasetPtr
HGraph::make_calibration_query(const DatasetPtr& queries, uint64_t index) const {
    auto query = Dataset::Make()->NumElements(1)->Dim(this->dim_);
    const auto offset = static_cast<int64_t>(index) * this->dim_;

    if (this->data_type_ == DataTypes::DATA_TYPE_FLOAT) {
        const auto* vectors = queries->GetFloat32Vectors();
        CHECK_ARGUMENT(vectors != nullptr, "recall search query must contain float32 vectors");
        query->Float32Vectors(vectors + offset);
    } else if (this->data_type_ == DataTypes::DATA_TYPE_INT8) {
        const auto* vectors = queries->GetInt8Vectors();
        CHECK_ARGUMENT(vectors != nullptr, "recall search query must contain int8 vectors");
        query->Int8Vectors(vectors + offset);
    } else if (this->data_type_ == DataTypes::DATA_TYPE_FP16 or
               this->data_type_ == DataTypes::DATA_TYPE_BF16) {
        const auto* vectors = queries->GetFloat16Vectors();
        CHECK_ARGUMENT(vectors != nullptr, "recall search query must contain float16 vectors");
        query->Float16Vectors(vectors + offset);
    } else if (this->data_type_ == DataTypes::DATA_TYPE_SPARSE) {
        const auto* vectors = queries->GetSparseVectors();
        CHECK_ARGUMENT(vectors != nullptr, "recall search query must contain sparse vectors");
        query->SparseVectors(vectors + index);
    } else {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "recall search does not support this query data type");
    }
    return query->Owner(false);
}

std::vector<RecallSearchResult>
HGraph::CalibrateRecallSearch(const std::vector<RecallTarget>& targets,
                              const DatasetPtr& calibration_queries) {
    std::lock_guard calibration_lock(this->recall_search_calibration_mutex_);
    CHECK_ARGUMENT(this->immutable_.load(std::memory_order_acquire),
                   "recall search calibration requires an immutable index");
    CHECK_ARGUMENT(this->GetNumElements() > 0, "cannot calibrate recall search on an empty index");
    CHECK_ARGUMENT(not targets.empty(), "recall search targets cannot be empty");
    CHECK_ARGUMENT(calibration_queries != nullptr,
                   "recall search calibration_queries are required");
    CHECK_ARGUMENT(calibration_queries->GetNumElements() > 0,
                   "recall search calibration_queries cannot be empty");
    CHECK_ARGUMENT(calibration_queries->GetDim() == this->dim_,
                   "recall search calibration query dimension does not match the index");

    std::vector<RecallTarget> unique_targets;
    unique_targets.reserve(targets.size());
    for (const auto& target : targets) {
        CHECK_ARGUMENT(target.top_k > 0, "recall search top_k must be positive");
        CHECK_ARGUMENT(target.top_k <= this->GetNumElements(),
                       "recall search top_k cannot exceed the index size");
        CHECK_ARGUMENT(is_valid_recall(target.recall),
                       "recall search target must be finite and in (0, 1]");
        const auto duplicate = std::find_if(
            unique_targets.begin(), unique_targets.end(), [&target](const auto& existing) {
                return existing.top_k == target.top_k and
                       std::fabs(existing.recall - target.recall) <= RECALL_EPSILON;
            });
        if (duplicate == unique_targets.end()) {
            unique_targets.push_back(target);
        }
    }

    const auto query_count = static_cast<uint64_t>(calibration_queries->GetNumElements());
    std::vector<DatasetPtr> queries;
    queries.reserve(query_count);
    for (uint64_t i = 0; i < query_count; ++i) {
        queries.push_back(this->make_calibration_query(calibration_queries, i));
    }

    const auto selected_efs =
        calibrate_ef_search(*this, queries, unique_targets, this->GetNumElements());
    std::vector<RecallSearchResult> results;
    results.reserve(unique_targets.size());
    for (uint64_t i = 0; i < unique_targets.size(); ++i) {
        const auto& target = unique_targets[i];
        const auto parameters = fmt::format(R"({{"hgraph":{{"ef_search":{}}}}})", selected_efs[i]);
        results.push_back({target.top_k, target.recall, parameters});
        logger::info("HGraph recall search selected ef_search={} for top_k={}, recall={}",
                     selected_efs[i],
                     target.top_k,
                     target.recall);
    }

    auto profile = std::make_shared<const std::vector<RecallSearchResult>>(results);
    std::atomic_store_explicit(&this->recall_search_profile_, profile, std::memory_order_release);
    return results;
}

std::string
HGraph::resolve_recall_search_parameters(int64_t top_k, float recall) const {
    const auto profile =
        std::atomic_load_explicit(&this->recall_search_profile_, std::memory_order_acquire);
    CHECK_ARGUMENT(profile != nullptr, "recall search has not been calibrated");
    const auto target = std::find_if(profile->begin(), profile->end(), [&](const auto& candidate) {
        return candidate.top_k == top_k and std::fabs(candidate.recall - recall) <= RECALL_EPSILON;
    });
    CHECK_ARGUMENT(target != profile->end(),
                   "the requested (top_k, recall) pair has not been calibrated");
    return target->search_parameters;
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query, int64_t k, float target_recall) const {
    CHECK_ARGUMENT(is_valid_recall(target_recall), "target_recall must be finite and in (0, 1]");
    this->validate_knn_args(query, k);

    SearchRequest request;
    request.query_ = query;
    request.topk_ = k;
    request.params_str_ = this->resolve_recall_search_parameters(k, target_recall);
    return this->SearchWithRequest(request);
}

}  // namespace vsag
