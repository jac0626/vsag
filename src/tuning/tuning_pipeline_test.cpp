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

#include "tuning/tuning_pipeline.h"

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
MakeBruteForceBuildParameters(uint64_t dim) {
    nlohmann::json parameters{{"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}};
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

vsag::IndexPtr
BuildBruteForceIndex(const fixtures::TestDatasetPtr& dataset) {
    auto index =
        vsag::Factory::CreateIndex("brute_force", MakeBruteForceBuildParameters(dataset->dim_));
    REQUIRE(index.has_value());
    auto build_result = index.value()->Build(dataset->base_);
    REQUIRE(build_result.has_value());
    return index.value();
}

}  // namespace

TEST_CASE("auto tuning pipeline runs all P0 stages with explicit skipped stages", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::AutoTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.target_recall = 0.0;
    request.base_search_parameters = R"({"hgraph":{"factor":2}})";
    request.ef_search_candidates = {0, 10, 80, 1201};

    vsag::AutoTuningPipeline pipeline;
    const auto report = pipeline.Tune(request);

    REQUIRE(report.Succeeded());
    REQUIRE(report.elapsed_ms >= 0.0);
    REQUIRE(report.stages.size() == 9);
    REQUIRE(report.stages[0].stage == vsag::TuningStage::WORKLOAD_VALIDATION);
    REQUIRE(report.stages[0].status == vsag::TuningStageStatus::COMPLETED);
    REQUIRE(report.stages[2].stage == vsag::TuningStage::BUILD_PARAMETER_TUNING);
    REQUIRE(report.stages[2].status == vsag::TuningStageStatus::SKIPPED);
    REQUIRE(report.stages[3].stage == vsag::TuningStage::QUANTIZER_TUNING);
    REQUIRE(report.stages[3].status == vsag::TuningStageStatus::SKIPPED);
    REQUIRE(report.stages[5].stage == vsag::TuningStage::CANDIDATE_PRUNING);
    REQUIRE(report.stages[5].input_count == 4);
    REQUIRE(report.stages[5].output_count == 2);
    REQUIRE(report.stages[7].stage == vsag::TuningStage::TRIAL_EXECUTION);
    REQUIRE(report.stages[7].status == vsag::TuningStageStatus::COMPLETED);
    REQUIRE(report.recommendation.has_value());
    REQUIRE(report.best_effort.has_value());
}

TEST_CASE("auto tuning pipeline fails early for invalid workloads", "[ut][tuning]") {
    vsag::AutoTuningRequest request;
    request.topk = 10;

    vsag::AutoTuningPipeline pipeline;
    const auto report = pipeline.Tune(request);

    REQUIRE_FALSE(report.Succeeded());
    REQUIRE(report.elapsed_ms >= 0.0);
    REQUIRE(report.stages.size() == 1);
    REQUIRE(report.stages[0].stage == vsag::TuningStage::WORKLOAD_VALIDATION);
    REQUIRE(report.stages[0].status == vsag::TuningStageStatus::FAILED);
}

TEST_CASE("auto tuning pipeline rejects non-HGraph P0 inputs", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildBruteForceIndex(dataset);

    vsag::AutoTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.ef_search_candidates = {10};

    vsag::AutoTuningPipeline pipeline;
    const auto report = pipeline.Tune(request);

    REQUIRE_FALSE(report.Succeeded());
    REQUIRE(report.stages.size() == 1);
    REQUIRE(report.stages[0].stage == vsag::TuningStage::WORKLOAD_VALIDATION);
    REQUIRE(report.stages[0].status == vsag::TuningStageStatus::FAILED);
}

TEST_CASE("auto tuning pipeline rejects non-HGraph P0 parameter paths", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::AutoTuningRequest request;
    request.index = index;
    request.index_name = "ivf";
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.ef_search_candidates = {10};

    vsag::AutoTuningPipeline pipeline;
    const auto report = pipeline.Tune(request);

    REQUIRE_FALSE(report.Succeeded());
    REQUIRE(report.stages.size() == 1);
    REQUIRE(report.stages[0].stage == vsag::TuningStage::WORKLOAD_VALIDATION);
    REQUIRE(report.stages[0].status == vsag::TuningStageStatus::FAILED);
}

TEST_CASE("auto tuning pipeline reports not implemented stages explicitly", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::AutoTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.enable_build_parameter_tuning = true;
    request.ef_search_candidates = {10};

    vsag::AutoTuningPipeline pipeline;
    const auto report = pipeline.Tune(request);

    REQUIRE_FALSE(report.Succeeded());
    REQUIRE(report.stages.size() == 3);
    REQUIRE(report.stages[2].stage == vsag::TuningStage::BUILD_PARAMETER_TUNING);
    REQUIRE(report.stages[2].status == vsag::TuningStageStatus::FAILED);
}

TEST_CASE("auto tuning pipeline reports quantizer tuning as not implemented", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::AutoTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.enable_quantizer_tuning = true;
    request.ef_search_candidates = {10};

    vsag::AutoTuningPipeline pipeline;
    const auto report = pipeline.Tune(request);

    REQUIRE_FALSE(report.Succeeded());
    REQUIRE(report.stages.size() == 4);
    REQUIRE(report.stages[3].stage == vsag::TuningStage::QUANTIZER_TUNING);
    REQUIRE(report.stages[3].status == vsag::TuningStageStatus::FAILED);
}

TEST_CASE("auto tuning pipeline reports successive halving as not implemented", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::AutoTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.enable_successive_halving = true;
    request.ef_search_candidates = {10};

    vsag::AutoTuningPipeline pipeline;
    const auto report = pipeline.Tune(request);

    REQUIRE_FALSE(report.Succeeded());
    REQUIRE(report.stages.size() == 6);
    REQUIRE(report.stages[5].stage == vsag::TuningStage::TRIAL_PLANNING);
    REQUIRE(report.stages[5].status == vsag::TuningStageStatus::FAILED);
}

TEST_CASE("auto tuning pipeline marks trial execution failure", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::AutoTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.base_search_parameters = "{invalid";
    request.ef_search_candidates = {10};

    vsag::AutoTuningPipeline pipeline;
    const auto report = pipeline.Tune(request);

    REQUIRE_FALSE(report.Succeeded());
    REQUIRE(report.stages.size() == 8);
    REQUIRE(report.stages[7].stage == vsag::TuningStage::TRIAL_EXECUTION);
    REQUIRE(report.stages[7].status == vsag::TuningStageStatus::FAILED);
    REQUIRE(report.ef_search.trials.size() == 1);
    REQUIRE(report.ef_search.trials[0].status == vsag::TuningTrialStatus::FAILED);
}

TEST_CASE("auto tuning pipeline propagates max trial budget", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    uint64_t evaluation_count = 0;
    vsag::EfSearchTuner ef_tuner([&evaluation_count](const vsag::EvaluationRequest&) {
        ++evaluation_count;

        vsag::EvaluationResult result;
        result.recall.average = 1.0;
        return result;
    });

    vsag::AutoTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.max_trials = 1;
    request.ef_search_candidates = {10, 80};

    vsag::AutoTuningPipeline pipeline(ef_tuner);
    const auto report = pipeline.Tune(request);

    REQUIRE(report.Succeeded());
    REQUIRE(evaluation_count == 1);
    REQUIRE(report.ef_search.trials.size() == 2);
    REQUIRE(report.ef_search.trials[0].status == vsag::TuningTrialStatus::COMPLETED);
    REQUIRE(report.ef_search.trials[1].status == vsag::TuningTrialStatus::SKIPPED);
    REQUIRE(report.ef_search.trials[1].message == "budget exceeded: max_trials = 1");
    REQUIRE(report.stages[5].stage == vsag::TuningStage::CANDIDATE_PRUNING);
    REQUIRE(report.stages[5].output_count == 1);
    REQUIRE(report.stages[6].stage == vsag::TuningStage::TRIAL_PLANNING);
    REQUIRE(report.stages[6].input_count == 1);
}

TEST_CASE("auto tuning pipeline keeps best effort when no candidate meets recall target",
          "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    vsag::EfSearchTuner ef_tuner([](const vsag::EvaluationRequest& request) {
        auto parameters = nlohmann::json::parse(request.search_parameters);
        const auto ef_search = parameters["hgraph"]["ef_search"].get<uint64_t>();

        vsag::EvaluationResult result;
        result.query_count = request.query_count;
        result.recall.average = ef_search >= 80 ? 0.80 : 0.40;
        return result;
    });

    vsag::AutoTuningRequest request;
    request.index = index;
    request.queries = dataset->query_;
    request.ground_truth = dataset->ground_truth_;
    request.topk = static_cast<uint64_t>(dataset->top_k);
    request.query_count = 8;
    request.target_recall = 0.95;
    request.ef_search_candidates = {10, 80};

    vsag::AutoTuningPipeline pipeline(ef_tuner);
    const auto report = pipeline.Tune(request);

    REQUIRE_FALSE(report.Succeeded());
    REQUIRE_FALSE(report.recommendation.has_value());
    REQUIRE(report.best_effort.has_value());
    REQUIRE(report.best_effort->candidate.ef_search == 80);
    REQUIRE(report.stages.size() == 9);
    REQUIRE(report.stages[8].stage == vsag::TuningStage::SELECTION);
    REQUIRE(report.stages[8].status == vsag::TuningStageStatus::SKIPPED);
}
