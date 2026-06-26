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

#include "tuning/evaluation_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace vsag {
namespace {

constexpr double MS_PER_SECOND = 1000.0;

EvaluationResult
Fail(EvaluationStatus status, const std::string& message) {
    EvaluationResult result;
    result.status = status;
    result.error_message = message;
    return result;
}

double
Average(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double
Percentile(std::vector<double> values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = percentile * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<uint64_t>(std::floor(position));
    const auto upper = static_cast<uint64_t>(std::ceil(position));
    if (lower == upper) {
        return values[lower];
    }
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

bool
ContainsId(const int64_t* ids, uint64_t count, int64_t target) {
    for (uint64_t i = 0; i < count; ++i) {
        if (ids[i] == target) {
            return true;
        }
    }
    return false;
}

double
CalculateRecall(const DatasetPtr& result,
                const DatasetPtr& ground_truth,
                uint64_t query_offset,
                uint64_t topk) {
    const auto result_dim = static_cast<uint64_t>(std::max<int64_t>(result->GetDim(), 0));
    const auto result_count = std::min(result_dim, topk);
    const auto gt_dim = static_cast<uint64_t>(ground_truth->GetDim());
    const auto* result_ids = result->GetIds();
    const auto* gt_ids = ground_truth->GetIds() + query_offset * gt_dim;

    uint64_t hit_count = 0;
    for (uint64_t i = 0; i < result_count; ++i) {
        if (ContainsId(gt_ids, topk, result_ids[i])) {
            ++hit_count;
        }
    }
    return static_cast<double>(hit_count) / static_cast<double>(topk);
}

EvaluationResult
ValidateRequest(const EvaluationRequest& request) {
    if (request.index == nullptr) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "index is required");
    }
    if (request.queries == nullptr) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "queries are required");
    }
    if (request.ground_truth == nullptr) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "ground truth is required");
    }
    if (request.topk == 0) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "topk must be greater than 0");
    }
    if (request.topk > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "topk is too large");
    }
    if (request.queries->GetNumElements() <= 0) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "queries must not be empty");
    }
    if (request.queries->GetDim() <= 0) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "query dimension must be positive");
    }
    if (request.queries->GetFloat32Vectors() == nullptr) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "float32 query vectors are required");
    }
    if (request.ground_truth->GetIds() == nullptr) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "ground truth ids are required");
    }
    if (request.ground_truth->GetDim() < static_cast<int64_t>(request.topk)) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "ground truth dim must cover topk");
    }

    const auto total_queries = static_cast<uint64_t>(request.queries->GetNumElements());
    const auto query_count =
        request.query_count == 0 ? total_queries : std::min(request.query_count, total_queries);
    if (request.ground_truth->GetNumElements() < static_cast<int64_t>(query_count)) {
        return Fail(EvaluationStatus::INVALID_ARGUMENT, "ground truth query count is too small");
    }
    return {};
}

}  // namespace

EvaluationResult
InMemoryEvaluationRunner::Run(const EvaluationRequest& request) const {
    if (auto invalid = ValidateRequest(request); not invalid.Succeeded()) {
        return invalid;
    }

    const auto dim = request.queries->GetDim();
    const auto total_queries = static_cast<uint64_t>(request.queries->GetNumElements());
    const auto query_count =
        request.query_count == 0 ? total_queries : std::min(request.query_count, total_queries);
    const auto topk = request.topk;
    const auto* query_vectors = request.queries->GetFloat32Vectors();

    std::vector<double> recall_values;
    std::vector<double> latency_values;
    recall_values.reserve(static_cast<size_t>(query_count));
    latency_values.reserve(static_cast<size_t>(query_count));

    for (uint64_t i = 0; i < query_count; ++i) {
        auto single_query = Dataset::Make();
        single_query->Dim(dim)
            ->NumElements(1)
            ->Float32Vectors(query_vectors + i * static_cast<uint64_t>(dim))
            ->Owner(false);

        const auto start = std::chrono::steady_clock::now();
        auto search_result = request.index->KnnSearch(
            single_query, static_cast<int64_t>(topk), request.search_parameters);
        const auto finish = std::chrono::steady_clock::now();
        const auto latency_ms = std::chrono::duration<double, std::milli>(finish - start).count();

        if (not search_result.has_value()) {
            return Fail(EvaluationStatus::SEARCH_ERROR, search_result.error().message);
        }
        if (search_result.value() == nullptr || search_result.value()->GetIds() == nullptr) {
            return Fail(EvaluationStatus::SEARCH_ERROR, "search result ids are required");
        }

        latency_values.push_back(latency_ms);
        recall_values.push_back(
            CalculateRecall(search_result.value(), request.ground_truth, i, topk));
    }

    EvaluationResult result;
    result.query_count = query_count;
    result.recall.average = Average(recall_values);
    result.recall.p0 = Percentile(recall_values, 0.0);
    result.recall.p10 = Percentile(recall_values, 0.1);
    result.recall.p30 = Percentile(recall_values, 0.3);
    result.recall.p50 = Percentile(recall_values, 0.5);
    result.recall.p70 = Percentile(recall_values, 0.7);
    result.recall.p90 = Percentile(recall_values, 0.9);
    result.latency.average_ms = Average(latency_values);
    result.latency.p50_ms = Percentile(latency_values, 0.5);
    result.latency.p90_ms = Percentile(latency_values, 0.9);
    result.latency.p95_ms = Percentile(latency_values, 0.95);
    result.latency.p99_ms = Percentile(latency_values, 0.99);

    const auto total_ms = std::accumulate(latency_values.begin(), latency_values.end(), 0.0);
    if (total_ms > 0.0) {
        result.qps = static_cast<double>(query_count) * MS_PER_SECOND / total_ms;
    }

    if (request.collect_memory) {
        const auto memory_usage = request.index->GetMemoryUsage();
        result.memory_bytes = memory_usage > 0 ? static_cast<uint64_t>(memory_usage) : 0;
    }
    return result;
}

}  // namespace vsag
