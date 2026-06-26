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

#include <nlohmann/json.hpp>

#include "framework/test_dataset_pool.h"
#include "unittest.h"
#include "vsag/factory.h"

namespace {

std::string
MakeHGraphBuildParameters(uint64_t dim) {
    nlohmann::json index_param{{"base_quantization_type", "fp32"},
                               {"max_degree", 16},
                               {"ef_construction", 100},
                               {"build_thread_count", 1}};
    nlohmann::json parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"index_param", index_param}};
    return parameters.dump();
}

std::string
MakeHGraphSearchParameters(uint64_t ef_search) {
    return nlohmann::json{{"hgraph", {{"ef_search", ef_search}}}}.dump();
}

vsag::IndexPtr
BuildHGraphIndex(const fixtures::TestDatasetPtr& dataset) {
    auto index = vsag::Factory::CreateIndex("hgraph", MakeHGraphBuildParameters(dataset->dim_));
    REQUIRE(index.has_value());
    auto build_result = index.value()->Build(dataset->base_);
    REQUIRE(build_result.has_value());
    return index.value();
}

}  // namespace

TEST_CASE("in-memory evaluation runner evaluates hgraph knn search", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::EvaluationRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.search_parameters = MakeHGraphSearchParameters(80);

    vsag::InMemoryEvaluationRunner runner;
    const auto result = runner.Run(request);

    REQUIRE(result.Succeeded());
    REQUIRE(result.query_count == request.query_count);
    REQUIRE(result.recall.average >= 0.0);
    REQUIRE(result.recall.average <= 1.0);
    REQUIRE(result.latency.average_ms >= 0.0);
    REQUIRE(result.qps > 0.0);
    REQUIRE(result.memory_bytes > 0);
}

TEST_CASE("in-memory evaluation runner rejects invalid requests", "[ut][tuning]") {
    vsag::EvaluationRequest request;
    request.topk = 10;

    vsag::InMemoryEvaluationRunner runner;
    const auto result = runner.Run(request);

    REQUIRE_FALSE(result.Succeeded());
    REQUIRE(result.status == vsag::EvaluationStatus::INVALID_ARGUMENT);
}
