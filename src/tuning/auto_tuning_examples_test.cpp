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

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "framework/test_dataset_pool.h"
#include "tuning/auto_tuning_api.h"
#include "unittest.h"
#include "vsag/factory.h"

namespace {

void
RequireObjectKeys(const nlohmann::json& object, const std::vector<std::string>& keys) {
    REQUIRE(object.is_object());
    REQUIRE(static_cast<uint64_t>(object.size()) == static_cast<uint64_t>(keys.size()));
    for (const auto& key : keys) {
        REQUIRE(object.contains(key));
    }
}

void
RequireExampleRequestContract(const std::string& request_json) {
    const auto request = nlohmann::json::parse(request_json);
    RequireObjectKeys(request,
                      {"version",
                       "index_type",
                       "source",
                       "workload",
                       "config",
                       "objective",
                       "search_space",
                       "evaluation",
                       "budget"});
    RequireObjectKeys(request["source"], {"type"});
    REQUIRE(request["index_type"].get<std::string>() == "hgraph");
    REQUIRE(request["workload"].contains("topk"));
    REQUIRE(request["config"].contains("search_parameters"));
    REQUIRE(request["objective"].contains("recall_at_k"));
    REQUIRE(request["search_space"].contains("search"));
    REQUIRE(request["evaluation"].contains("query_count"));
    REQUIRE(request["budget"].contains("max_trials"));
}

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

std::string
ReadExample(const std::string& filename) {
    std::ifstream input(std::string(VSAG_TUNING_EXAMPLES_DIR) + "/" + filename);
    REQUIRE(input.is_open());

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

vsag::AutoTuningApiContext
MakeContext(const fixtures::TestDatasetPtr& dataset, const vsag::IndexPtr& index = nullptr) {
    vsag::AutoTuningApiContext context;
    context.index = index;
    context.base = dataset->base_;
    context.queries = dataset->query_;
    context.ground_truth = dataset->ground_truth_;
    return context;
}

}  // namespace

TEST_CASE("auto tuning request examples parse and prepare", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    SECTION("existing index") {
        auto request_json = ReadExample("hgraph_auto_tuning_existing_index_request.json");
        RequireExampleRequestContract(request_json);

        auto parse_result = vsag::ParseAutoTuningRequestJson(request_json);
        REQUIRE(parse_result.Succeeded());
        REQUIRE(parse_result.request.source_type == "existing_index");
        REQUIRE(parse_result.request.index == nullptr);

        auto prepare_result =
            vsag::PrepareAutoTuningRequest(parse_result.request, MakeContext(dataset, index));
        REQUIRE(prepare_result.Succeeded());
        REQUIRE(prepare_result.request.index == index);
    }

    SECTION("raw dataset") {
        auto request_json = ReadExample("hgraph_auto_tuning_raw_dataset_request.json");
        RequireExampleRequestContract(request_json);

        auto parse_result = vsag::ParseAutoTuningRequestJson(request_json);
        REQUIRE(parse_result.Succeeded());
        REQUIRE(parse_result.request.source_type == "raw_dataset");
        REQUIRE(parse_result.request.index == nullptr);

        auto prepare_result =
            vsag::PrepareAutoTuningRequest(parse_result.request, MakeContext(dataset));
        REQUIRE(prepare_result.Succeeded());
        REQUIRE(prepare_result.request.index != nullptr);
        REQUIRE(prepare_result.request.index->GetIndexType() == vsag::IndexType::HGRAPH);
    }

    SECTION("max trials") {
        auto request_json = ReadExample("hgraph_auto_tuning_max_trials_request.json");
        RequireExampleRequestContract(request_json);

        auto parse_result = vsag::ParseAutoTuningRequestJson(request_json);
        REQUIRE(parse_result.Succeeded());
        REQUIRE(parse_result.request.max_trials == 2);
    }

    SECTION("rebuild tuning") {
        auto request_json = ReadExample("hgraph_auto_tuning_rebuild_request.json");
        RequireExampleRequestContract(request_json);

        auto parse_result = vsag::ParseAutoTuningRequestJson(request_json);
        REQUIRE(parse_result.Succeeded());
        REQUIRE(parse_result.request.source_type == "raw_dataset");
        REQUIRE(parse_result.request.enable_build_parameter_tuning);
        REQUIRE(parse_result.request.enable_quantizer_tuning);

        auto prepare_result =
            vsag::PrepareAutoTuningRequest(parse_result.request, MakeContext(dataset));
        REQUIRE(prepare_result.Succeeded());
        REQUIRE(prepare_result.request.base == dataset->base_);
    }
}

TEST_CASE("auto tuning request query_count zero evaluates all context queries", "[ut][tuning]") {
    fixtures::TestDatasetPool pool;
    auto dataset = pool.GetDatasetAndCreate(16, 200, "l2");
    auto index = BuildHGraphIndex(dataset);

    auto request =
        nlohmann::json::parse(ReadExample("hgraph_auto_tuning_existing_index_request.json"));
    request["evaluation"]["query_count"] = 0;

    auto prepare_result =
        vsag::PrepareAutoTuningRequestJson(request.dump(), MakeContext(dataset, index));
    REQUIRE(prepare_result.Succeeded());
    REQUIRE(prepare_result.request.query_count == 0);

    vsag::EfSearchTuner tuner([](const vsag::EvaluationRequest& request) {
        vsag::EvaluationResult result;
        result.query_count = request.query_count;
        result.recall.average = 1.0;
        return result;
    });

    vsag::AutoTuningPipeline pipeline(tuner);
    const auto report = pipeline.Tune(prepare_result.request);

    REQUIRE(report.Succeeded());
    REQUIRE(report.request.requested_query_count == 0);
    REQUIRE(report.request.effective_query_count ==
            static_cast<uint64_t>(dataset->query_->GetNumElements()));
}
