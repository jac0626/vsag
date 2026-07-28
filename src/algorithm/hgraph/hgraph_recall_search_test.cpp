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

#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "unittest.h"

namespace {

constexpr int64_t DIM = 4;
constexpr int64_t COUNT = 320;
constexpr auto DEFAULT_INDEX_PARAMS = R"({
    "base_quantization_type": "fp32",
    "max_degree": 8,
    "ef_construction": 32,
    "build_thread_count": 1
})";

std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>
make_index(const std::string& params_json = DEFAULT_INDEX_PARAMS) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = DIM;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto params = vsag::JsonType::Parse(params_json);
    return std::make_shared<vsag::IndexImpl<vsag::HGraph>>(params, common_param);
}

void
build_index(const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& index) {
    std::vector<float> vectors(COUNT * DIM);
    std::vector<int64_t> ids(COUNT);
    for (int64_t i = 0; i < COUNT; ++i) {
        ids[i] = 100 + i;
        for (int64_t j = 0; j < DIM; ++j) {
            vectors[i * DIM + j] = static_cast<float>(i * DIM + j);
        }
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(COUNT)
                    ->Dim(DIM)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
}

vsag::DatasetPtr
make_queries(std::vector<float>& vectors, int64_t count) {
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(DIM)
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

double
average_overlap(const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& index,
                std::vector<float>& query_vectors,
                int64_t query_count,
                int64_t top_k,
                const std::string& parameters) {
    const auto reference_parameters =
        std::string(R"({"hgraph":{"ef_search":)") + std::to_string(COUNT) + "}}";
    double overlap_sum = 0.0;
    for (int64_t query_index = 0; query_index < query_count; ++query_index) {
        auto query = vsag::Dataset::Make()
                         ->NumElements(1)
                         ->Dim(DIM)
                         ->Float32Vectors(query_vectors.data() + query_index * DIM)
                         ->Owner(false);
        const auto result = index->KnnSearch(query, top_k, parameters).value();
        const auto reference = index->KnnSearch(query, top_k, reference_parameters).value();
        std::unordered_multiset<int64_t> expected(reference->GetIds(), reference->GetIds() + top_k);
        int64_t overlap = 0;
        for (int64_t i = 0; i < top_k; ++i) {
            const auto found = expected.find(result->GetIds()[i]);
            if (found != expected.end()) {
                ++overlap;
                expected.erase(found);
            }
        }
        overlap_sum += static_cast<double>(overlap) / static_cast<double>(top_k);
    }
    return overlap_sum / static_cast<double>(query_count);
}

}  // namespace

TEST_CASE("HGraph recall search validates calibration input", "[ut][hgraph][recall_search]") {
    auto index = make_index();
    build_index(index);
    std::vector<float> calibration_vectors = {1.0F, 2.0F, 3.0F, 4.0F};
    auto calibration_queries = make_queries(calibration_vectors, 1);
    const std::vector<vsag::RecallTarget> valid{{3, 0.5F}};

    auto result = index->CalibrateRecallSearch(valid, calibration_queries);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(index->SetImmutable().has_value());

    SECTION("empty or invalid targets") {
        REQUIRE_FALSE(index->CalibrateRecallSearch({}, calibration_queries).has_value());
        REQUIRE_FALSE(index->CalibrateRecallSearch({{0, 0.5F}}, calibration_queries).has_value());
        REQUIRE_FALSE(index->CalibrateRecallSearch({{3, 1.1F}}, calibration_queries).has_value());
        REQUIRE_FALSE(index
                          ->CalibrateRecallSearch({{3, std::numeric_limits<float>::quiet_NaN()}},
                                                  calibration_queries)
                          .has_value());
        REQUIRE_FALSE(
            index->CalibrateRecallSearch({{COUNT + 1, 0.5F}}, calibration_queries).has_value());
    }

    SECTION("duplicate targets are coalesced") {
        result = index->CalibrateRecallSearch({{3, 0.5F}, {3, 0.5F}}, calibration_queries);
        REQUIRE(result.has_value());
        REQUIRE(result->size() == 1);
    }

    SECTION("invalid calibration queries") {
        REQUIRE_FALSE(index->CalibrateRecallSearch(valid, nullptr).has_value());
        auto empty_queries = vsag::Dataset::Make()->NumElements(0)->Dim(DIM);
        REQUIRE_FALSE(index->CalibrateRecallSearch(valid, empty_queries).has_value());

        std::vector<int8_t> int8_vectors(DIM);
        auto wrong_type_queries = vsag::Dataset::Make()
                                      ->NumElements(1)
                                      ->Dim(DIM)
                                      ->Int8Vectors(int8_vectors.data())
                                      ->Owner(false);
        REQUIRE_FALSE(index->CalibrateRecallSearch(valid, wrong_type_queries).has_value());

        std::vector<float> vectors(DIM + 1);
        auto wrong_dimension_queries = vsag::Dataset::Make()
                                           ->NumElements(1)
                                           ->Dim(DIM + 1)
                                           ->Float32Vectors(vectors.data())
                                           ->Owner(false);
        REQUIRE_FALSE(index->CalibrateRecallSearch(valid, wrong_dimension_queries).has_value());
    }

    SECTION("quantized and duplicate-aware indexes") {
        auto duplicate_index = make_index(R"({
            "base_quantization_type": "fp32",
            "max_degree": 8,
            "ef_construction": 32,
            "build_thread_count": 1,
            "support_duplicate": true
        })");
        build_index(duplicate_index);
        REQUIRE(duplicate_index->SetImmutable().has_value());
        REQUIRE(duplicate_index->CalibrateRecallSearch(valid, calibration_queries).has_value());

        auto sq8_index = make_index(R"({
            "base_quantization_type": "sq8",
            "max_degree": 8,
            "ef_construction": 32,
            "build_thread_count": 1
        })");
        build_index(sq8_index);
        REQUIRE(sq8_index->SetImmutable().has_value());
        REQUIRE(sq8_index->CalibrateRecallSearch(valid, calibration_queries).has_value());
    }
}

TEST_CASE("HGraph recall search publishes and replaces an immutable profile",
          "[ut][hgraph][recall_search]") {
    auto index = make_index();
    build_index(index);
    REQUIRE(index->SetImmutable().has_value());

    std::vector<float> calibration_vectors = {1.5F, 2.5F, 3.5F, 4.5F, 40.5F, 41.5F, 42.5F, 43.5F};
    auto calibration_queries = make_queries(calibration_vectors, 2);
    auto query = make_queries(calibration_vectors, 1);

    REQUIRE_FALSE(index->KnnSearch(query, 3, 1.0F).has_value());
    const std::vector<vsag::RecallTarget> targets{{3, 0.5F}, {3, 1.0F}, {5, 0.5F}};
    const auto calibration = index->CalibrateRecallSearch(targets, calibration_queries);
    REQUIRE(calibration.has_value());
    REQUIRE(calibration->size() == targets.size());

    for (const auto& result : calibration.value()) {
        REQUIRE(result.top_k > 0);
        REQUIRE(result.recall > 0.0F);
        REQUIRE(result.search_parameters.find("ef_search") != std::string::npos);
        REQUIRE(
            average_overlap(index, calibration_vectors, 2, result.top_k, result.search_parameters) +
                1e-6 >=
            result.recall);

        const auto recall_result = index->KnnSearch(query, result.top_k, result.recall);
        const auto explicit_result =
            index->KnnSearch(query, result.top_k, result.search_parameters);
        REQUIRE(recall_result.has_value());
        REQUIRE(explicit_result.has_value());
        for (int64_t i = 0; i < result.top_k; ++i) {
            REQUIRE(recall_result.value()->GetIds()[i] == explicit_result.value()->GetIds()[i]);
        }
    }

    const auto replacement = index->CalibrateRecallSearch({{4, 0.5F}}, calibration_queries);
    REQUIRE(replacement.has_value());
    REQUIRE(replacement->size() == 1);
    REQUIRE_FALSE(index->KnnSearch(query, 3, 0.5F).has_value());
    REQUIRE(index->KnnSearch(query, 4, 0.5F).has_value());

    REQUIRE_FALSE(index->CalibrateRecallSearch({}, calibration_queries).has_value());
    REQUIRE(index->KnnSearch(query, 4, 0.5F).has_value());
    const auto empty_query = vsag::Dataset::Make()->NumElements(0)->Dim(DIM);
    const auto empty_result = index->KnnSearch(empty_query, 4, 0.5F);
    REQUIRE(empty_result.has_value());
    REQUIRE(empty_result.value()->GetDim() == 0);
    REQUIRE_FALSE(index->KnnSearch(query, 4, -0.1F).has_value());
    REQUIRE_FALSE(index->KnnSearch(query, 4, std::numeric_limits<float>::quiet_NaN()).has_value());
}
