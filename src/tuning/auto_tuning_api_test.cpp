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

#include "tuning/auto_tuning_api.h"

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

nlohmann::json
MakeValidRequestJson(uint64_t topk, uint64_t query_count, double target_recall) {
    return nlohmann::json{
        {"version", 1},
        {"index_type", "hgraph"},
        {"source", {{"type", "existing_index"}}},
        {"workload", {{"topk", topk}}},
        {"config", {{"search_parameters", {{"hgraph", {{"factor", 2}}}}}}},
        {"objective", {{"recall_at_k", {{"min", target_recall}}}}},
        {"search_space", {{"search", {{"hgraph.ef_search", {{"values", {0, 10, 20, 1201}}}}}}}},
        {"evaluation",
         {{"query_count", query_count}, {"successive_halving", {{"enabled", false}}}}}};
}

nlohmann::json
MakeRawDatasetRequestJson(uint64_t topk, uint64_t query_count, double target_recall, uint64_t dim) {
    auto request = MakeValidRequestJson(topk, query_count, target_recall);
    request["source"]["type"] = "raw_dataset";
    request["config"]["build_parameters"] = nlohmann::json::parse(MakeHGraphBuildParameters(dim));
    return request;
}

vsag::AutoTuningApiContext
MakeContext(const vsag::IndexPtr& index, const fixtures::TestDatasetPtr& dataset) {
    vsag::AutoTuningApiContext context;
    context.index = index;
    context.base = dataset->base_;
    context.queries = dataset->query_;
    context.ground_truth = dataset->ground_truth_;
    return context;
}

}  // namespace

TEST_CASE("auto tuning api parses P0 json and serializes report", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    auto request = MakeValidRequestJson(static_cast<uint64_t>(dataset->top_k), 8, 0.95);
    request["budget"] = {{"max_trials", 2}};
    auto parse_result = vsag::ParseAutoTuningRequestJson(request.dump());

    REQUIRE(parse_result.Succeeded());
    REQUIRE(parse_result.request.index == nullptr);
    REQUIRE(parse_result.request.queries == nullptr);
    REQUIRE(parse_result.request.ground_truth == nullptr);
    REQUIRE(parse_result.request.topk == static_cast<uint64_t>(dataset->top_k));
    REQUIRE(parse_result.request.query_count == 8);
    REQUIRE(parse_result.request.target_recall == 0.95);
    REQUIRE(parse_result.request.index_name == "hgraph");
    REQUIRE(parse_result.request.ef_search_candidates == std::vector<uint64_t>{0, 10, 20, 1201});
    REQUIRE(parse_result.request.search_parameter_spaces.size() == 1);
    REQUIRE(parse_result.request.search_parameter_spaces[0].path == "hgraph.ef_search");
    REQUIRE(parse_result.request.search_parameter_spaces[0].values ==
            std::vector<std::string>{"0", "10", "20", "1201"});
    REQUIRE(parse_result.request.max_trials == 2);

    auto prepare_result =
        vsag::PrepareAutoTuningRequest(parse_result.request, MakeContext(index, dataset));
    REQUIRE(prepare_result.Succeeded());
    REQUIRE(prepare_result.request.index == index);
    REQUIRE(prepare_result.request.queries == dataset->query_);
    REQUIRE(prepare_result.request.ground_truth == dataset->ground_truth_);

    vsag::EfSearchTuner tuner([](const vsag::EvaluationRequest& request) {
        auto parameters = nlohmann::json::parse(request.search_parameters);
        REQUIRE(parameters["hgraph"]["factor"].get<uint64_t>() == 2);
        const auto ef_search = parameters["hgraph"]["ef_search"].get<uint64_t>();

        vsag::EvaluationResult result;
        result.query_count = request.query_count;
        result.recall.average = ef_search >= 20 ? 0.96 : 0.80;
        result.latency.average_ms = static_cast<double>(ef_search) / 100.0;
        result.qps = 1000.0;
        return result;
    });

    vsag::AutoTuningPipeline pipeline(tuner);
    const auto report = pipeline.Tune(prepare_result.request);
    REQUIRE(report.Succeeded());
    REQUIRE(report.recommendation.has_value());
    REQUIRE(report.recommendation->candidate.ef_search == 20);

    const auto report_json = nlohmann::json::parse(vsag::SerializeAutoTuningReportJson(report));
    REQUIRE(report_json["version"].get<uint64_t>() == 1);
    REQUIRE(report_json["succeeded"].get<bool>());
    REQUIRE(report_json["status"].get<std::string>() == "succeeded");
    REQUIRE(report_json["request"]["index_type"].get<std::string>() == "hgraph");
    REQUIRE(report_json["request"]["source"]["type"].get<std::string>() == "existing_index");
    REQUIRE(report_json["request"]["workload"]["topk"].get<uint64_t>() ==
            static_cast<uint64_t>(dataset->top_k));
    REQUIRE(report_json["request"]["config"]["build_parameters"].is_null());
    REQUIRE(
        report_json["request"]["config"]["search_parameters"]["hgraph"]["factor"].get<uint64_t>() ==
        2);
    REQUIRE(report_json["request"]["objective"]["primary"].get<std::string>() == "latency");
    REQUIRE(report_json["request"]["objective"]["recall_at_k"]["min"].get<double>() == 0.95);
    REQUIRE(report_json["request"]["search_space"]["search"]["hgraph.ef_search"]["values"] ==
            nlohmann::json::array({0, 10, 20, 1201}));
    REQUIRE(report_json["request"]["evaluation"]["query_count"].get<uint64_t>() == 8);
    REQUIRE(report_json["request"]["evaluation"]["effective_query_count"].get<uint64_t>() == 8);
    REQUIRE_FALSE(
        report_json["request"]["evaluation"]["successive_halving"]["enabled"].get<bool>());
    REQUIRE(report_json["request"]["budget"]["max_trials"].get<uint64_t>() == 2);
    REQUIRE(report_json["elapsed_ms"].get<double>() >= 0.0);
    REQUIRE(report_json["stages"].size() == 9);
    REQUIRE(report_json["trials"].size() == 4);
    REQUIRE(report_json["recommendation"]["candidate"]["hgraph.ef_search"].get<uint64_t>() == 20);
    REQUIRE(report_json["recommendation"]["search_parameters_patch"]["hgraph"]["ef_search"]
                .get<uint64_t>() == 20);
}

TEST_CASE("auto tuning api builds baseline hgraph for raw dataset P0 request", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");

    const auto request_json =
        MakeRawDatasetRequestJson(static_cast<uint64_t>(dataset->top_k), 8, 0.95, dataset->dim_)
            .dump();
    vsag::AutoTuningApiContext context;
    context.base = dataset->base_;
    context.queries = dataset->query_;
    context.ground_truth = dataset->ground_truth_;

    auto parse_result = vsag::ParseAutoTuningRequestJson(request_json);

    REQUIRE(parse_result.Succeeded());
    REQUIRE(parse_result.request.index == nullptr);
    REQUIRE(parse_result.request.source_type == "raw_dataset");
    REQUIRE(nlohmann::json::parse(parse_result.request.build_parameters)["dim"].get<uint64_t>() ==
            dataset->dim_);
    REQUIRE(parse_result.request.ef_search_candidates == std::vector<uint64_t>{0, 10, 20, 1201});

    auto prepare_result = vsag::PrepareAutoTuningRequest(parse_result.request, context);
    REQUIRE(prepare_result.Succeeded());
    REQUIRE(prepare_result.request.index != nullptr);
    REQUIRE(prepare_result.request.index->GetIndexType() == vsag::IndexType::HGRAPH);
    REQUIRE(prepare_result.request.queries == dataset->query_);
    REQUIRE(prepare_result.request.ground_truth == dataset->ground_truth_);

    vsag::EfSearchTuner tuner([](const vsag::EvaluationRequest& request) {
        auto parameters = nlohmann::json::parse(request.search_parameters);
        const auto ef_search = parameters["hgraph"]["ef_search"].get<uint64_t>();

        vsag::EvaluationResult result;
        result.query_count = request.query_count;
        result.recall.average = ef_search >= 20 ? 0.96 : 0.80;
        result.latency.average_ms = static_cast<double>(ef_search) / 100.0;
        return result;
    });

    vsag::AutoTuningPipeline pipeline(tuner);
    const auto report = pipeline.Tune(prepare_result.request);
    REQUIRE(report.Succeeded());
    REQUIRE(report.recommendation.has_value());
    REQUIRE(report.recommendation->candidate.ef_search == 20);

    const auto report_json = nlohmann::json::parse(vsag::SerializeAutoTuningReportJson(report));
    REQUIRE(report_json["request"]["source"]["type"].get<std::string>() == "raw_dataset");
    REQUIRE(report_json["request"]["config"]["build_parameters"]["index_param"]["max_degree"]
                .get<uint64_t>() == 16);
    REQUIRE(report_json["request"]["evaluation"]["effective_query_count"].get<uint64_t>() == 8);
}

TEST_CASE("auto tuning api validates request fields", "[ut][tuning]") {
    SECTION("raw dataset source without build parameters") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["source"]["type"] = "raw_dataset";

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE_FALSE(result.Succeeded());
        REQUIRE(result.status == vsag::AutoTuningApiStatus::INVALID_ARGUMENT);
        REQUIRE(result.error_code == "missing_field");
    }

    SECTION("raw dataset source without base context") {
        auto request = MakeRawDatasetRequestJson(10, 8, 0.95, 16);
        auto parse_result = vsag::ParseAutoTuningRequestJson(request.dump());
        REQUIRE(parse_result.Succeeded());

        vsag::AutoTuningApiContext context;
        context.queries = vsag::Dataset::Make();
        context.ground_truth = vsag::Dataset::Make();

        const auto result = vsag::PrepareAutoTuningRequest(parse_result.request, context);

        REQUIRE_FALSE(result.Succeeded());
        REQUIRE(result.status == vsag::AutoTuningApiStatus::INVALID_ARGUMENT);
        REQUIRE(result.error_code == "invalid_context");
    }

    SECTION("existing index source accepts build parameters as rebuild metadata") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["config"]["build_parameters"] =
            nlohmann::json::parse(MakeHGraphBuildParameters(16));

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE(result.Succeeded());
        REQUIRE(nlohmann::json::parse(result.request.build_parameters)["dim"].get<uint64_t>() ==
                16);
    }

    SECTION("non hgraph index type") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["index_type"] = "brute_force";

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE_FALSE(result.Succeeded());
        REQUIRE(result.status == vsag::AutoTuningApiStatus::UNSUPPORTED);
        REQUIRE(result.error_code == "unsupported_index_type");
    }

    SECTION("build search space enters V1 skeleton stage") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["search_space"]["build"] = {{"hgraph.max_degree", {{"values", {16, 32}}}}};

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE(result.Succeeded());
        REQUIRE(result.request.enable_build_parameter_tuning);
        REQUIRE(result.request.build_parameter_spaces.size() == 1);
        REQUIRE(result.request.build_parameter_spaces[0].path == "hgraph.max_degree");
        REQUIRE(result.request.build_parameter_spaces[0].values ==
                std::vector<std::string>{"16", "32"});
    }

    SECTION("quantizer search space enters V1 skeleton stage") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["search_space"]["quantizer"] = {
            {"hgraph.base_quantization_type", {{"values", {"fp32", "sq8_uniform"}}}}};

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE(result.Succeeded());
        REQUIRE(result.request.enable_quantizer_tuning);
        REQUIRE(result.request.quantizer_parameter_spaces.size() == 1);
        REQUIRE(result.request.quantizer_parameter_spaces[0].path ==
                "hgraph.base_quantization_type");
        REQUIRE(result.request.quantizer_parameter_spaces[0].values ==
                std::vector<std::string>{"\"fp32\"", "\"sq8_uniform\""});
    }

    SECTION("non hgraph parameter path") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["search_space"]["search"] = {{"ivf.nprobe", {{"values", {10, 20}}}}};

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE_FALSE(result.Succeeded());
        REQUIRE(result.status == vsag::AutoTuningApiStatus::UNSUPPORTED);
        REQUIRE(result.error_code == "unsupported_parameter");
    }

    SECTION("successive halving enters V1 skeleton stage") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["evaluation"]["successive_halving"]["enabled"] = true;

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE(result.Succeeded());
        REQUIRE(result.request.enable_successive_halving);
    }

    SECTION("warmup query count") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["evaluation"]["warmup_query_count"] = 10;

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE_FALSE(result.Succeeded());
        REQUIRE(result.status == vsag::AutoTuningApiStatus::UNSUPPORTED);
        REQUIRE(result.error_code == "unsupported_evaluation_option");
    }

    SECTION("unsupported budget field") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["budget"] = {{"timeout_seconds", 10}};

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE_FALSE(result.Succeeded());
        REQUIRE(result.status == vsag::AutoTuningApiStatus::UNSUPPORTED);
        REQUIRE(result.error_code == "unsupported_budget");
    }

    SECTION("output controls") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["output"] = {{"include_all_trials", false}};

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE_FALSE(result.Succeeded());
        REQUIRE(result.status == vsag::AutoTuningApiStatus::UNSUPPORTED);
        REQUIRE(result.error_code == "unsupported_output");
    }

    SECTION("non latency primary objective") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["objective"]["primary"] = "memory";

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE_FALSE(result.Succeeded());
        REQUIRE(result.status == vsag::AutoTuningApiStatus::UNSUPPORTED);
        REQUIRE(result.error_code == "unsupported_objective");
    }

    SECTION("missing topk") {
        auto request = MakeValidRequestJson(10, 8, 0.95);
        request["workload"].erase("topk");

        const auto result = vsag::ParseAutoTuningRequestJson(request.dump());

        REQUIRE_FALSE(result.Succeeded());
        REQUIRE(result.status == vsag::AutoTuningApiStatus::INVALID_ARGUMENT);
        REQUIRE(result.error_code == "missing_field");
    }
}

TEST_CASE("auto tuning api serializes best effort without recommendation", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    auto request = MakeValidRequestJson(static_cast<uint64_t>(dataset->top_k), 8, 0.95);
    request["search_space"]["search"]["hgraph.ef_search"]["values"] = {10, 80};
    auto parse_result =
        vsag::PrepareAutoTuningRequestJson(request.dump(), MakeContext(index, dataset));
    REQUIRE(parse_result.Succeeded());

    vsag::EfSearchTuner tuner([](const vsag::EvaluationRequest& request) {
        auto parameters = nlohmann::json::parse(request.search_parameters);
        const auto ef_search = parameters["hgraph"]["ef_search"].get<uint64_t>();

        vsag::EvaluationResult result;
        result.query_count = request.query_count;
        result.recall.average = ef_search >= 80 ? 0.80 : 0.40;
        result.latency.average_ms = static_cast<double>(ef_search) / 100.0;
        return result;
    });

    vsag::AutoTuningPipeline pipeline(tuner);
    const auto report = pipeline.Tune(parse_result.request);
    REQUIRE_FALSE(report.Succeeded());
    REQUIRE_FALSE(report.recommendation.has_value());
    REQUIRE(report.best_effort.has_value());

    const auto report_json = nlohmann::json::parse(vsag::SerializeAutoTuningReportJson(report));
    REQUIRE_FALSE(report_json["succeeded"].get<bool>());
    REQUIRE(report_json["status"].get<std::string>() == "failed");
    REQUIRE(report_json["recommendation"].is_null());
    REQUIRE(report_json["best_effort"]["candidate"]["hgraph.ef_search"].get<uint64_t>() == 80);
    REQUIRE(report_json["stages"].back()["stage"].get<std::string>() == "selection");
    REQUIRE(report_json["stages"].back()["status"].get<std::string>() == "skipped");
}
