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

#include "tuning/ef_search_tuner.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

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

vsag::IndexPtr
BuildHGraphIndex(const fixtures::TestDatasetPtr& dataset) {
    auto index = vsag::Factory::CreateIndex("hgraph", MakeHGraphBuildParameters(dataset->dim_));
    REQUIRE(index.has_value());
    auto build_result = index.value()->Build(dataset->base_);
    REQUIRE(build_result.has_value());
    return index.value();
}

}  // namespace

TEST_CASE("ef search tuner prunes invalid candidates and recommends the smallest valid trial",
          "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::EfSearchTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.target_recall = 0.0;
    request.ef_search_candidates = {1201, 80, 10, 0, 80};

    vsag::EfSearchTuner tuner;
    const auto report = tuner.Tune(request);

    REQUIRE(report.trials.size() == 4);
    REQUIRE(report.trials[0].candidate.ef_search == 0);
    REQUIRE(report.trials[0].status == vsag::TuningTrialStatus::SKIPPED);
    REQUIRE(report.trials[1].candidate.ef_search == 10);
    REQUIRE(report.trials[1].status == vsag::TuningTrialStatus::COMPLETED);
    REQUIRE(report.trials[2].candidate.ef_search == 80);
    REQUIRE(report.trials[2].status == vsag::TuningTrialStatus::COMPLETED);
    REQUIRE(report.trials[3].candidate.ef_search == 1201);
    REQUIRE(report.trials[3].status == vsag::TuningTrialStatus::SKIPPED);
    REQUIRE(report.recommendation.has_value());
    REQUIRE(report.recommendation->candidate.ef_search == 10);
    REQUIRE(report.best_effort.has_value());
}

TEST_CASE("ef search tuner keeps best effort when target recall is unreachable", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::EfSearchTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.target_recall = 1.1;
    request.ef_search_candidates = {10, 80};

    vsag::EfSearchTuner tuner;
    const auto report = tuner.Tune(request);

    REQUIRE_FALSE(report.recommendation.has_value());
    REQUIRE(report.best_effort.has_value());
    REQUIRE(report.best_effort->status == vsag::TuningTrialStatus::COMPLETED);
}

TEST_CASE("ef search tuner preserves base search parameters and applies recall threshold",
          "[ut][tuning]") {
    std::vector<std::string> observed_parameters;
    vsag::EfSearchTuner tuner([&observed_parameters](const vsag::EvaluationRequest& request) {
        observed_parameters.push_back(request.search_parameters);
        auto parameters = nlohmann::json::parse(request.search_parameters);
        const auto ef_search = parameters["hgraph"]["ef_search"].get<uint64_t>();

        vsag::EvaluationResult result;
        result.query_count = request.query_count;
        result.recall.average = ef_search >= 80 ? 0.95 : 0.50;
        result.qps = ef_search >= 80 ? 100.0 : 200.0;
        return result;
    });

    vsag::EfSearchTuningRequest request;
    request.topk = 10;
    request.query_count = 8;
    request.target_recall = 0.90;
    request.base_search_parameters = R"({"hgraph":{"factor":2,"ef_search":1},"other":true})";
    request.ef_search_candidates = {10, 80};

    const auto report = tuner.Tune(request);

    REQUIRE(report.recommendation.has_value());
    REQUIRE(report.recommendation->candidate.ef_search == 80);
    REQUIRE(report.best_effort.has_value());
    REQUIRE(report.best_effort->candidate.ef_search == 80);
    REQUIRE(observed_parameters.size() == 2);
    for (const auto& parameter : observed_parameters) {
        auto parsed = nlohmann::json::parse(parameter);
        REQUIRE(parsed["hgraph"]["factor"].get<uint64_t>() == 2);
        REQUIRE(parsed["other"].get<bool>());
    }
}

TEST_CASE("ef search tuner fails trials with invalid base search parameters", "[ut][tuning]") {
    vsag::EfSearchTuner tuner([](const vsag::EvaluationRequest&) {
        vsag::EvaluationResult result;
        result.recall.average = 1.0;
        return result;
    });

    vsag::EfSearchTuningRequest request;
    request.topk = 10;
    request.target_recall = 0.90;
    request.base_search_parameters = "{invalid";
    request.ef_search_candidates = {10};

    const auto report = tuner.Tune(request);

    REQUIRE(report.trials.size() == 1);
    REQUIRE(report.trials[0].status == vsag::TuningTrialStatus::FAILED);
    REQUIRE_FALSE(report.recommendation.has_value());
    REQUIRE_FALSE(report.best_effort.has_value());
}
