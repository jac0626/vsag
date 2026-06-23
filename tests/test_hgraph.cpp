
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

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
#include <thread>

#include "functest.h"
#include "inner_string_params.h"
#include "test_index.h"
#include "typing.h"
#include "vsag/filter.h"
#include "vsag/options.h"
#include "vsag/search_request.h"

namespace fixtures {

class HGraphTestResource {
public:
    std::vector<int> dims;
    std::vector<std::pair<std::string, float>> test_cases;
    std::vector<std::string> metric_types;
    uint64_t base_count;
};

using HGraphResourcePtr = std::shared_ptr<HGraphTestResource>;
class HGraphTestIndex : public fixtures::TestIndex {
public:
    struct HGraphBuildParam {
        std::string metric_type;
        int64_t dim;
        std::string quantization_str = "sq8";
        uint32_t rabitq_num_bit_base = 1;
        uint32_t rabitq_num_bit_query = 32;
        int thread_count = 5;
        int extra_info_size = 0;
        std::string data_type = "float32";
        std::string graph_type = "nsw";
        std::string graph_storage = "flat";
        bool support_remove = false;
        bool use_attr_filter = false;
        bool store_raw_vector = false;
        bool support_duplicate = false;
        std::string graph_io_type = "block_memory_io";
        std::string graph_file_path = "./graph_storage";
        HGraphBuildParam(const std::string& metric_type,
                         int64_t dim,
                         const std::string& quantization_str)
            : metric_type(metric_type), dim(dim), quantization_str(quantization_str) {
        }
    };

    static std::string
    GenerateHGraphBuildParametersString(const HGraphBuildParam& param);

    static HGraphResourcePtr
    GetResource(bool sample = true);

    static bool
    IsRaBitQ(const std::string& quantization_str);

    static void
    TestGeneral(const IndexPtr& index,
                const TestDatasetPtr& dataset,
                const std::string& search_param,
                float recall,
                bool expect_success = true);

    static void
    TestMemoryUsageDetail(const IndexPtr& index);

    static TestDatasetPool pool;

    static fixtures::TempDir dir;

    static uint64_t base_count;

    static const std::string name;

    static const std::vector<std::pair<std::string, float>> all_test_cases;
};
using HGraphTestIndexPtr = std::shared_ptr<HGraphTestIndex>;

class RejectAllFilter : public vsag::Filter {
public:
    bool
    CheckValid(int64_t) const override {
        return false;
    }

    bool
    CheckValid(const char*) const override {
        return false;
    }

    float
    ValidRatio() const override {
        return 0.0F;
    }
};

TestDatasetPool HGraphTestIndex::pool{};
fixtures::TempDir HGraphTestIndex::dir{"hgraph_test"};
uint64_t HGraphTestIndex::base_count = 600;
const std::string HGraphTestIndex::name = "hgraph";
const std::vector<std::pair<std::string, float>> HGraphTestIndex::all_test_cases = {
    {"fp32", 0.99},
    {"bf16", 0.98},
    {"fp16", 0.98},
    {"sq8", 0.95},
    {"sq8_uniform", 0.95},
    {"rabitq,fp32,block_memory_io,4,1", 0.3},
    {"rabitq,fp32,block_memory_io,32,1", 0.3},
    {"rabitq,fp32,block_memory_io,32,2", 0.3},
    {"rabitq,fp32,block_memory_io,32,4", 0.3},
    {"rabitq,fp32,block_memory_io,32,8", 0.3},
    {"pq,fp32", 0.95},
    {"sq4_uniform,fp32", 0.95},
    {"sq8_uniform,fp32", 0.98},
    {"sq8_uniform,fp16", 0.98},
    {"sq8_uniform,bf16", 0.98},
};

constexpr static const char* search_param_tmp = R"(
        {{
            "hgraph": {{
                "ef_search": {},
                "use_extra_info_filter": {}
            }}
        }})";

HGraphResourcePtr
HGraphTestIndex::GetResource(bool sample) {
    auto resource = std::make_shared<HGraphTestResource>();
    if (sample) {
        resource->dims = fixtures::get_common_used_dims(1, RandomValue(0, 999), 257);
        resource->test_cases = fixtures::RandomSelect(HGraphTestIndex::all_test_cases, 2);
        resource->metric_types = fixtures::RandomSelect<std::string>({"ip", "l2", "cosine"}, 1);
        resource->base_count = HGraphTestIndex::base_count;
    } else {
        resource->dims = fixtures::get_index_test_dims(3, RandomValue(0, 999));
        resource->test_cases = HGraphTestIndex::all_test_cases;
        resource->metric_types = fixtures::RandomSelect<std::string>({"ip", "l2", "cosine"}, 2);
        resource->base_count = HGraphTestIndex::base_count * 3;
    }
    return resource;
}

std::string
HGraphTestIndex::GenerateHGraphBuildParametersString(const HGraphBuildParam& param) {
    std::string build_parameters_str;

    constexpr auto parameter_temp_reorder = R"(
    {{
        "dtype": "{}",
        "metric_type": "{}",
        "dim": {},
        "extra_info_size": {},
        "index_param": {{
            "use_reorder": {},
            "base_quantization_type": "{}",
            "max_degree": 96,
            "ef_construction": 500,
            "build_thread_count": {},
            "base_pq_dim": {},
            "precise_quantization_type": "{}",
            "precise_io_type": "{}",
            "precise_file_path": "{}",
            "graph_type": "{}",
            "graph_storage_type": "{}",
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.3,
            "alpha": 1.2,
            "support_remove": {},
            "use_attribute_filter": {},
            "store_raw_vector": {},
            "support_duplicate": {},
            "graph_io_type": "{}",
            "graph_file_path": "{}",
            "rabitq_bits_per_dim_base": {},
            "rabitq_bits_per_dim_query": {}
        }}
    }}
    )";

    constexpr auto parameter_temp_origin = R"(
    {{
        "dtype": "{}",
        "metric_type": "{}",
        "dim": {},
        "extra_info_size": {},
        "index_param": {{
            "base_quantization_type": "{}",
            "max_degree": 96,
            "base_pq_dim": {},
            "ef_construction": 500,
            "build_thread_count": {},
            "graph_type": "{}",
            "graph_storage_type": "{}",
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.3,
            "alpha": 1.2,
            "support_remove": {},
            "use_attribute_filter": {},
            "store_raw_vector": {},
            "support_duplicate": {},
            "graph_io_type": "{}",
            "graph_file_path": "{}",
            "rabitq_bits_per_dim_base": {},
            "rabitq_bits_per_dim_query": {}
        }}
    }}
    )";

    int pq_dim = param.dim;
    if (pq_dim % 2 == 0) {
        pq_dim /= 2;
    }

    auto strs = fixtures::SplitString(param.quantization_str, ',');
    std::string high_quantizer_str, precise_io_type = "block_memory_io";
    auto& base_quantizer_str = strs[0];
    uint32_t rabitq_num_bit_query = 32, rabitq_num_bit_base = 1;
    if (strs.size() > 1) {
        high_quantizer_str = strs[1];
        if (strs.size() > 2) {
            precise_io_type = strs[2];
        }
        if (strs.size() > 4 and base_quantizer_str == vsag::QUANTIZATION_TYPE_VALUE_RABITQ) {
            rabitq_num_bit_query = std::stoi(strs[3]);
            rabitq_num_bit_base = std::stoi(strs[4]);
        }
        build_parameters_str = fmt::format(parameter_temp_reorder,
                                           param.data_type,
                                           param.metric_type,
                                           param.dim,
                                           param.extra_info_size,
                                           true, /* reorder */
                                           base_quantizer_str,
                                           param.thread_count,
                                           pq_dim,
                                           high_quantizer_str,
                                           precise_io_type,
                                           dir.GenerateRandomFile(),
                                           param.graph_type,
                                           param.graph_storage,
                                           param.support_remove,
                                           param.use_attr_filter,
                                           param.store_raw_vector,
                                           param.support_duplicate,
                                           param.graph_io_type,
                                           param.graph_file_path,
                                           rabitq_num_bit_base,
                                           rabitq_num_bit_query);
    } else {
        build_parameters_str = fmt::format(parameter_temp_origin,
                                           param.data_type,
                                           param.metric_type,
                                           param.dim,
                                           param.extra_info_size,
                                           base_quantizer_str,
                                           pq_dim,
                                           param.thread_count,
                                           param.graph_type,
                                           param.graph_storage,
                                           param.support_remove,
                                           param.use_attr_filter,
                                           param.store_raw_vector,
                                           param.support_duplicate,
                                           param.graph_io_type,
                                           param.graph_file_path,
                                           param.rabitq_num_bit_base,
                                           param.rabitq_num_bit_query);
    }
    return build_parameters_str;
}

bool
HGraphTestIndex::IsRaBitQ(const std::string& quantization_str) {
    return (quantization_str.find(vsag::QUANTIZATION_TYPE_VALUE_RABITQ) != std::string::npos);
}

void
HGraphTestIndex::TestGeneral(const TestIndex::IndexPtr& index,
                             const TestDatasetPtr& dataset,
                             const std::string& search_param,
                             float recall,
                             bool expect_success) {
    REQUIRE(index->GetIndexType() == vsag::IndexType::HGRAPH);
    TestGetMinAndMaxId(index, dataset);
    TestKnnSearch(index, dataset, search_param, recall, true);
    TestKnnSearchIter(index, dataset, search_param, recall, true);
    TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
    TestRangeSearch(index, dataset, search_param, recall, 10, true);
    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
    TestFilterSearch(index, dataset, search_param, recall, true, true);
    TestCheckIdExist(index, dataset);
    TestCalcDistanceById(index, dataset, 1e-5, expect_success);
    TestGetRawVectorByIds(index, dataset, expect_success);
    TestBatchCalcDistanceById(index, dataset, 1e-5, expect_success);
    TestSearchAllocator(index, dataset, search_param, recall, true);
    TestUpdateVector(index, dataset, search_param, false);
    TestUpdateId(index, dataset, search_param, true);
    TestMemoryUsageDetail(index);
    TestIndexStatus(index);
}

void
HGraphTestIndex::TestMemoryUsageDetail(const IndexPtr& index) {
    auto memory_detail = vsag::JsonType::Parse(index->GetMemoryUsageDetail());
    REQUIRE(memory_detail.Contains("basic_flatten_codes"));
    REQUIRE(memory_detail.Contains("bottom_graph"));
    REQUIRE(memory_detail.Contains("route_graph"));
}
}  // namespace fixtures

namespace {

static fixtures::TestIndex::DatasetPtr
MakeTinyDenseDataset(const std::vector<float>& vectors,
                     const std::vector<int64_t>& ids,
                     int64_t dim) {
    auto dataset = vsag::Dataset::Make();
    auto* copied_vectors = new float[vectors.size()];
    auto* copied_ids = new int64_t[ids.size()];
    std::copy(vectors.begin(), vectors.end(), copied_vectors);
    std::copy(ids.begin(), ids.end(), copied_ids);
    dataset->Dim(dim)
        ->NumElements(static_cast<int64_t>(ids.size()))
        ->Ids(copied_ids)
        ->Float32Vectors(copied_vectors)
        ->Owner(true);
    return dataset;
}

static std::string
GenerateTinyDuplicateHGraphParam(bool use_old_serial_format = false,
                                 bool support_duplicate = true,
                                 float duplicate_distance_threshold = 0.0F,
                                 const std::string& quantization_str = "fp32",
                                 bool store_raw_vector = false) {
    fixtures::HGraphTestIndex::HGraphBuildParam build_param("l2", 2, quantization_str);
    build_param.support_duplicate = support_duplicate;
    build_param.thread_count = 1;
    build_param.store_raw_vector = store_raw_vector;
    auto param = fixtures::HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
    auto json = vsag::JsonType::Parse(param);
    if (duplicate_distance_threshold > 0.0F) {
        json["index_param"]["duplicate_distance_threshold"].SetFloat(duplicate_distance_threshold);
    }
    if (use_old_serial_format) {
        json["use_old_serial_format"].SetBool(true);
    }
    return json.Dump();
}

static fixtures::TestIndex::IndexPtr
BuildTinyDuplicateHGraphIndex(const std::string& param) {
    auto index = fixtures::TestIndex::TestFactory(fixtures::HGraphTestIndex::name, param, true);
    auto first_duplicate = MakeTinyDenseDataset({1.0F, 0.0F}, {10}, 2);
    auto second_duplicate = MakeTinyDenseDataset({1.0F, 0.0F}, {11}, 2);
    REQUIRE(index->Add(first_duplicate).has_value());
    REQUIRE(index->Add(second_duplicate).has_value());
    return index;
}

static int64_t
GetBasicFlattenDetailSize(const fixtures::TestIndex::IndexPtr& index) {
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto memory_detail = vsag::JsonType::Parse(index->GetMemoryUsageDetail());
    return memory_detail["basic_flatten_codes"].GetInt();
}

static void
RequireTinyDuplicateVectorAccess(const fixtures::TestIndex::IndexPtr& index) {
    REQUIRE(index->CheckIdExist(10));
    REQUIRE(index->CheckIdExist(11));

    int64_t ids[2] = {10, 11};
    auto raw_vectors = index->GetRawVectorByIds(ids, 2);
    REQUIRE(raw_vectors.has_value());
    REQUIRE(raw_vectors.value()->GetDim() == 2);
    REQUIRE(raw_vectors.value()->GetNumElements() == 2);
    auto* vectors = raw_vectors.value()->GetFloat32Vectors();
    REQUIRE(std::fabs(vectors[0] - 1.0F) < 1e-6F);
    REQUIRE(std::fabs(vectors[1]) < 1e-6F);
    REQUIRE(std::fabs(vectors[2] - 1.0F) < 1e-6F);
    REQUIRE(std::fabs(vectors[3]) < 1e-6F);

    float query[2] = {1.0F, 0.0F};
    auto first_distance = index->CalcDistanceById(query, 10);
    auto second_distance = index->CalcDistanceById(query, 11);
    REQUIRE(first_distance.has_value());
    REQUIRE(second_distance.has_value());
    REQUIRE(std::fabs(first_distance.value()) < 1e-6F);
    REQUIRE(std::fabs(second_distance.value()) < 1e-6F);

    auto batch_distances = index->CalDistanceById(query, ids, 2);
    REQUIRE(batch_distances.has_value());
    auto* distances = batch_distances.value()->GetDistances();
    REQUIRE(std::fabs(distances[0]) < 1e-6F);
    REQUIRE(std::fabs(distances[1]) < 1e-6F);
}

static void
RequireTinyDuplicateSearch(const fixtures::TestIndex::IndexPtr& index) {
    auto query = MakeTinyDenseDataset({1.0F, 0.0F}, {10}, 2);
    auto graph_result = index->KnnSearch(query, 1, R"({"hgraph":{"ef_search":16}})");
    REQUIRE(graph_result.has_value());
    REQUIRE(graph_result.value()->GetDim() == 1);
    REQUIRE((graph_result.value()->GetIds()[0] == 10 || graph_result.value()->GetIds()[0] == 11));
    REQUIRE(std::fabs(graph_result.value()->GetDistances()[0]) < 1e-6F);

    auto brute_force_result =
        index->KnnSearch(query, 2, R"({"hgraph":{"ef_search":16,"brute_force_threshold":1.0}})");
    REQUIRE(brute_force_result.has_value());
    REQUIRE(brute_force_result.value()->GetDim() == 2);

    std::vector<int64_t> result_ids(brute_force_result.value()->GetIds(),
                                    brute_force_result.value()->GetIds() + 2);
    std::sort(result_ids.begin(), result_ids.end());
    REQUIRE(result_ids == std::vector<int64_t>{10, 11});

    auto* distances = brute_force_result.value()->GetDistances();
    REQUIRE(std::fabs(distances[0]) < 1e-6F);
    REQUIRE(std::fabs(distances[1]) < 1e-6F);
}

static void
RequireTinyDuplicateLogicalCorrectness(const fixtures::TestIndex::IndexPtr& index) {
    RequireTinyDuplicateVectorAccess(index);
    RequireTinyDuplicateSearch(index);
}

static void
RequireRangeSearchDisableReorderChangesResult(const fixtures::TestIndex::IndexPtr& index,
                                              const fixtures::TestDatasetPtr& dataset,
                                              const std::string& search_param_with_reorder,
                                              const std::string& search_param_without_reorder,
                                              int64_t limited_size = 10) {
    const auto queries = dataset->query_;
    const auto query_count = queries->GetNumElements();
    const auto dim = queries->GetDim();
    bool found_difference = false;
    for (int64_t i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(queries->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto with_reorder = index->RangeSearch(
            query, std::numeric_limits<float>::max(), search_param_with_reorder, limited_size);
        auto without_reorder = index->RangeSearch(
            query, std::numeric_limits<float>::max(), search_param_without_reorder, limited_size);
        REQUIRE(with_reorder.has_value());
        REQUIRE(without_reorder.has_value());
        if (with_reorder.value()->GetDim() != without_reorder.value()->GetDim()) {
            found_difference = true;
            break;
        }
        const auto result_dim = with_reorder.value()->GetDim();
        for (int64_t j = 0; j < result_dim; ++j) {
            if (with_reorder.value()->GetIds()[j] != without_reorder.value()->GetIds()[j] ||
                std::abs(with_reorder.value()->GetDistances()[j] -
                         without_reorder.value()->GetDistances()[j]) > 1e-6F) {
                found_difference = true;
                break;
            }
        }
        if (found_difference) {
            break;
        }
    }
    REQUIRE(found_difference);
}

template <typename Fn>
void
RunWithGeneratedBlockSizeLimit(Fn&& fn) {
    const auto origin_size = vsag::Options::Instance().block_size_limit();
    const auto size = GENERATE(1024 * 1024 * 2);
    vsag::Options::Instance().set_block_size_limit(size);
    fn();
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

template <typename Cases, typename Fn>
void
ForEachHGraphCase(const fixtures::HGraphResourcePtr& resource, const Cases& test_cases, Fn&& fn) {
    for (const auto& metric_type : resource->metric_types) {
        for (auto raw_dim : resource->dims) {
            for (const auto& [base_quantization_str, recall] : test_cases) {
                auto dim = raw_dim;
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (fixtures::HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                fn(metric_type, dim, base_quantization_str, recall);
            }
        }
    }
}

#define HGRAPH_PR_DAILY_CASE(title, tags, helper)                        \
    TEST_CASE("(PR) " title, tags "[pr]") {                              \
        auto test_index = std::make_shared<fixtures::HGraphTestIndex>(); \
        auto resource = test_index->GetResource(true);                   \
        helper(test_index, resource);                                    \
    }                                                                    \
    TEST_CASE("(Daily) " title, tags "[daily]") {                        \
        auto test_index = std::make_shared<fixtures::HGraphTestIndex>(); \
        auto resource = test_index->GetResource(false);                  \
        helper(test_index, resource);                                    \
    }

}  // namespace

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex,
                             "HGraph Factory Test With Exceptions",
                             "[ft][hgraph]") {
    SECTION("Empty parameters") {
        auto param = "{}";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("No dim param") {
        auto param = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid param") {
        auto metric = GENERATE("", "l4", "inner_product", "cosin", "hamming");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "{}",
            "dim": 23,
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        auto param = fmt::format(param_tmp, metric);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid datatype param") {
        auto datatype = GENERATE("fp32", "uint8_t", "binary", "", "float", "int8");
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "{}",
            "metric_type": "l2",
            "dim": 23,
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        auto param = fmt::format(param_tmp, datatype);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid dim param") {
        int dim = GENERATE(-12, -1, 0);
        constexpr const char* param_tmp = R"(
        {{
            "dtype": "float32",
            "metric_type": "l2",
            "dim": {},
            "index_param": {{
                "base_quantization_type": "sq8"
            }}
        }})";
        auto param = fmt::format(param_tmp, dim);
        REQUIRE_THROWS(TestFactory(name, param, false));
        auto float_param = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 3.51,
            "index_param": {
                "base_quantization_type": "sq8"
            }
        })";
        REQUIRE_THROWS(TestFactory(name, float_param, false));
    }

    SECTION("Miss hgraph param") {
        auto param = GENERATE(
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                }}
            }})",
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35
            }})");
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION(
        "Invalid hgraph param "
        "base_quantization_type") {
        auto base_quantization_types = GENERATE("fsa");
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                    "base_quantization_type": "{}"
                }}
            }})";
        auto param = fmt::format(param_temp, base_quantization_types);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION("Invalid hgraph param key") {
        auto param_keys = GENERATE("base_quantization_types", "base_quantization");
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                    "{}": "sq8"
                }}
            }})";
        auto param = fmt::format(param_temp, param_keys);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }

    SECTION(
        "Invalid hgraph param "
        "graph_storage_type") {
        auto graph_storage_type = "fsa";
        constexpr const char* param_temp =
            R"({{
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 35,
                "index_param": {{
                    "graph_storage_type": "{}"
                }}
            }})";
        auto param = fmt::format(param_temp, graph_storage_type);
        REQUIRE_THROWS(TestFactory(name, param, false));
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex,
                             "HGraph Factory Test With Correct Parameters",
                             "[ft][hgraph]") {
    // bug issue #883
    SECTION("Empty index_param") {
        auto param = R"(
        {
            "dtype": "float32",
            "dim": 128,
            "metric_type": "l2",
            "index_param": {
            }
        })";
        REQUIRE(TestFactory(name, param, true));
    }
    SECTION("pq index_param") {
        auto param = R"(
        {
            "dtype": "float32",
            "dim": 128,
            "metric_type": "l2",
            "index_param": {
                "base_quantization_type": "pq"
            }
        })";
        REQUIRE(TestFactory(name, param, true));
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex, "HGraph GetStatus", "[ft][hgraph]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.support_duplicate = true;
                build_param.support_remove = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                INFO(index->GetStats());
                vsag::SearchRequest request;
                request.topk_ = 100;
                request.params_str_ = fmt::format(fixtures::search_param_tmp, 200, false);
                request.query_ = dataset->query_;
                auto raw_num = dataset->query_->GetNumElements();
                dataset->query_->NumElements(10);
                INFO(index->AnalyzeIndexBySearch(request));
                dataset->query_->NumElements(raw_num);
            }
        }
    }
}

static void
TestHGraphBuildAndContinueAdd(const fixtures::HGraphTestIndexPtr& test_index,
                              const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    ForEachHGraphCase(
        resource,
        resource->test_cases,
        [&](const auto& metric_type, int64_t dim, const auto& base_quantization_str, float recall) {
            RunWithGeneratedBlockSizeLimit([&] {
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestContinueAdd(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                TestIndex::TestIndexDetailData(index);
            });
        });
}

HGRAPH_PR_DAILY_CASE("HGraph Build & ContinueAdd Test",
                     "[ft][build][hgraph]",
                     TestHGraphBuildAndContinueAdd)

static void
TestHGraphFactor(const fixtures::HGraphTestIndexPtr& test_index,
                 const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);

    constexpr static const char* search_param_template = R"(
        {{
            "hgraph": {{
                "ef_search": 200,
                "factor": {}
            }}
        }})";
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                float factors[4]{4, 0.5, -2.0F, 100};
                for (int i = 0; i < 4; i++) {
                    auto search_param = fmt::format(search_param_template, factors[i], false);
                    TestIndex::TestKnnSearch(index, dataset, search_param, recall, true);
                }
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("HGraph Factor Test", "[ft][factory][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphFactor(test_index, resource);
}

void
TestHGraphTrainAndAddTest(const fixtures::HGraphTestIndexPtr& test_index,
                          const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    ForEachHGraphCase(
        resource,
        resource->test_cases,
        [&](const auto& metric_type, int64_t dim, const auto& base_quantization_str, float recall) {
            RunWithGeneratedBlockSizeLimit([&] {
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestTrainAndAdd(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
            });
        });
}

HGRAPH_PR_DAILY_CASE("HGraph Train & Add Test", "[ft][build][hgraph]", TestHGraphTrainAndAddTest)

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex,
                             "HGraph Search Empty Index",
                             "[ft][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2", "ip", "cosine");
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    auto ex_search_param = fmt::format(fixtures::search_param_tmp, 200, true);
    auto dim = fixtures::get_common_used_dims(1, fixtures::RandomValue(0, 999))[0];
    auto& [base_quantization_str, recall] = all_test_cases[0];
    vsag::Options::Instance().set_block_size_limit(size);
    HGraphTestIndex::HGraphBuildParam build_param(metric_type, dim, base_quantization_str);
    auto param = GenerateHGraphBuildParametersString(build_param);
    auto index = TestFactory(name, param, true);
    auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
    TestGetMinAndMaxId(index, dataset, false);
    TestKnnSearch(index, dataset, search_param, recall, false);
    TestKnnSearchIter(index, dataset, search_param, recall, false);
    TestConcurrentKnnSearch(index, dataset, search_param, recall, false);
    TestRangeSearch(index, dataset, search_param, recall, 10, false);
    TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, false);
    TestFilterSearch(index, dataset, search_param, recall, false, true);
    TestCheckIdExist(index, dataset, false);
    TestCalcDistanceById(index, dataset, 2e-6, false);
    TestBatchCalcDistanceById(index, dataset, 2e-6, false);
    TestKnnSearchExFilter(index, dataset, ex_search_param, recall, false);
    TestKnnSearchIter(index, dataset, ex_search_param, recall, false, true);
    // with ex info empty index
    build_param.extra_info_size = 256;
    auto ex_param = GenerateHGraphBuildParametersString(build_param);
    auto ex_index = TestFactory(name, param, true);
    auto ex_dataset = pool.GetDatasetAndCreate(
        dim, base_count, metric_type, false, 0.8, build_param.extra_info_size);
    TestKnnSearchExFilter(ex_index, ex_dataset, ex_search_param, recall, false);
    TestKnnSearchIter(ex_index, ex_dataset, ex_search_param, recall, false, true);
    auto index2 = TestIndex::TestFactory(name, param, true);
    TestIndex::TestSerializeFile(index, index2, dataset, search_param, true);
    index2 = TestIndex::TestFactory(name, param, true);
    TestIndex::TestSerializeBinarySet(index, index2, dataset, search_param, true);
    index2 = TestIndex::TestFactory(name, param, true);
    TestIndex::TestSerializeReaderSet(index, index2, dataset, search_param, name, true);
    index2 = TestIndex::TestFactory(name, param, true);
    TestIndex::TestSerializeWriteFunc(index, index2, dataset, search_param, true);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

static void
TestHGraphBuild(const fixtures::HGraphTestIndexPtr& test_index,
                const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    ForEachHGraphCase(
        resource,
        resource->test_cases,
        [&](const auto& metric_type, int64_t dim, const auto& base_quantization_str, float recall) {
            RunWithGeneratedBlockSizeLimit([&] {
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);

                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestExportIDs(index, dataset);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
            });
        });
}

HGRAPH_PR_DAILY_CASE("HGraph Build Test", "[ft][build][hgraph]", TestHGraphBuild)
static void
TestHGraphWithAttr(const fixtures::HGraphTestIndexPtr& test_index,
                   const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    std::vector<std::pair<std::string, float>> tmp_test_cases = {
        {"fp32", 0.75},
    };
    ForEachHGraphCase(
        resource,
        tmp_test_cases,
        [&](const auto& metric_type, int64_t dim, const auto& base_quantization_str, float recall) {
            RunWithGeneratedBlockSizeLimit([&] {
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.use_attr_filter = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                if (not index->CheckFeature(vsag::SUPPORT_BUILD)) {
                    return;
                }
                auto build_result = index->Build(dataset->base_);
                REQUIRE(build_result.has_value());
                TestIndex::TestWithAttr(index, dataset, search_param, false);
                auto index2 = TestIndex::TestFactory(HGraphTestIndex::name, param, true);

                REQUIRE_NOTHROW(test_serializion_file(*index, *index2, "serialize_hgraph"));
                TestIndex::TestWithAttr(index2, dataset, search_param, true);
            });
        });
}

HGRAPH_PR_DAILY_CASE("HGraph With Attr", "[ft][filter_search][hgraph]", TestHGraphWithAttr)

TEST_CASE("(PR) HGraph SearchWithRequest Reasoning", "[ft][hgraph][pr]") {
    using namespace fixtures;

    HGraphTestIndex::HGraphBuildParam build_param("l2", 16, "fp32");
    auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

    auto index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req;
    req.topk_ = 5;
    req.params_str_ = fmt::format(fixtures::search_param_tmp, 32, false);
    req.query_ = query;
    req.expected_labels_ = {dataset->base_->GetIds()[0]};

    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE_FALSE(result.value()->GetReasoning().empty());
    REQUIRE(result.value()->GetReasoning().find("missed_targets") != std::string::npos);

    req.enable_filter_ = true;
    req.filter_ = std::make_shared<RejectAllFilter>();

    auto empty_result = index->SearchWithRequest(req);
    REQUIRE(empty_result.has_value());
    REQUIRE(empty_result.value()->GetDim() == 0);
    REQUIRE_FALSE(empty_result.value()->GetReasoning().empty());
    REQUIRE(empty_result.value()->GetReasoning().find("missed_targets") != std::string::npos);
    REQUIRE(empty_result.value()->GetReasoning().find("diagnosis") != std::string::npos);
}

TEST_CASE("(PR) HGraph Reasoning Found Verification", "[ft][hgraph][reasoning][pr]") {
    using namespace fixtures;

    HGraphTestIndex::HGraphBuildParam build_param("l2", 16, "fp32");
    auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

    auto index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req;
    req.topk_ = 10;
    req.params_str_ = fmt::format(fixtures::search_param_tmp, 200, false);
    req.query_ = query;

    auto baseline = index->SearchWithRequest(req);
    REQUIRE(baseline.has_value());
    REQUIRE(baseline.value()->GetDim() > 0);

    auto* ids = baseline.value()->GetIds();
    int64_t found_label = ids[0];

    req.expected_labels_ = {found_label};
    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE_FALSE(result.value()->GetReasoning().empty());

    auto reasoning = result.value()->GetReasoning();
    REQUIRE(reasoning.find("1/1") != std::string::npos);
    REQUIRE(reasoning.find("0 missed") != std::string::npos);
}

TEST_CASE("(PR) HGraph Reasoning Multiple Labels Mixed", "[ft][hgraph][reasoning][pr]") {
    using namespace fixtures;

    HGraphTestIndex::HGraphBuildParam build_param("l2", 16, "fp32");
    auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

    auto index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req;
    req.topk_ = 5;
    req.params_str_ = fmt::format(fixtures::search_param_tmp, 200, false);
    req.query_ = query;

    auto baseline = index->SearchWithRequest(req);
    REQUIRE(baseline.has_value());
    REQUIRE(baseline.value()->GetDim() > 0);

    auto* ids = baseline.value()->GetIds();
    int64_t found_label = ids[0];
    int64_t unlikely_label = dataset->base_->GetNumElements() + 99999;

    req.expected_labels_ = {found_label, unlikely_label};
    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE_FALSE(result.value()->GetReasoning().empty());

    auto reasoning = result.value()->GetReasoning();
    REQUIRE(reasoning.find("expected_analysis") != std::string::npos);
}

TEST_CASE("(PR) HGraph Reasoning Does Not Affect Results", "[ft][hgraph][reasoning][pr]") {
    using namespace fixtures;

    HGraphTestIndex::HGraphBuildParam build_param("l2", 16, "fp32");
    auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

    auto index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req_without;
    req_without.topk_ = 10;
    req_without.params_str_ = fmt::format(fixtures::search_param_tmp, 200, false);
    req_without.query_ = query;

    auto result_without = index->SearchWithRequest(req_without);
    REQUIRE(result_without.has_value());
    REQUIRE(result_without.value()->GetDim() > 0);

    vsag::SearchRequest req_with;
    req_with.topk_ = 10;
    req_with.params_str_ = fmt::format(fixtures::search_param_tmp, 200, false);
    req_with.query_ = query;
    req_with.expected_labels_ = {result_without.value()->GetIds()[0]};

    auto result_with = index->SearchWithRequest(req_with);
    REQUIRE(result_with.has_value());
    REQUIRE(result_with.value()->GetDim() == result_without.value()->GetDim());

    auto dim_without = result_without.value()->GetDim();
    for (int64_t i = 0; i < dim_without; ++i) {
        REQUIRE(result_with.value()->GetIds()[i] == result_without.value()->GetIds()[i]);
        REQUIRE(result_with.value()->GetDistances()[i] ==
                result_without.value()->GetDistances()[i]);
    }
}

TEST_CASE("(PR) HGraph Reasoning Empty Expected Labels", "[ft][hgraph][reasoning][pr]") {
    using namespace fixtures;

    HGraphTestIndex::HGraphBuildParam build_param("l2", 16, "fp32");
    auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

    auto index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(16, 256, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    vsag::SearchRequest req;
    req.topk_ = 5;
    req.params_str_ = fmt::format(fixtures::search_param_tmp, 32, false);
    req.query_ = query;
    req.expected_labels_ = {};

    auto result = index->SearchWithRequest(req);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetReasoning().find("expected_analysis") == std::string::npos);
}

TEST_CASE("(PR) HGraph Reasoning Zero Overhead When Disabled", "[ft][hgraph][reasoning][pr]") {
    using namespace fixtures;

    HGraphTestIndex::HGraphBuildParam build_param("l2", 64, "fp32");
    auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

    auto index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(64, 1000, "l2");
    TestIndex::TestBuildIndex(index, dataset, true);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dataset->base_->GetDim())
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);

    constexpr int warmup_rounds = 20;
    constexpr int measure_rounds = 100;

    std::string search_params = fmt::format(fixtures::search_param_tmp, 100, false);

    for (int i = 0; i < warmup_rounds; ++i) {
        index->KnnSearch(query, 10, search_params, vsag::BitsetPtr(nullptr));
    }

    auto start_baseline = std::chrono::steady_clock::now();
    for (int i = 0; i < measure_rounds; ++i) {
        auto r = index->KnnSearch(query, 10, search_params, vsag::BitsetPtr(nullptr));
        REQUIRE(r.has_value());
    }
    auto end_baseline = std::chrono::steady_clock::now();
    auto baseline_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_baseline - start_baseline)
            .count();

    vsag::SearchRequest req_no_reasoning;
    req_no_reasoning.topk_ = 10;
    req_no_reasoning.params_str_ = search_params;
    req_no_reasoning.query_ = query;
    req_no_reasoning.expected_labels_ = {};

    for (int i = 0; i < warmup_rounds; ++i) {
        index->SearchWithRequest(req_no_reasoning);
    }

    auto start_no_reasoning = std::chrono::steady_clock::now();
    for (int i = 0; i < measure_rounds; ++i) {
        auto r = index->SearchWithRequest(req_no_reasoning);
        REQUIRE(r.has_value());
    }
    auto end_no_reasoning = std::chrono::steady_clock::now();
    auto no_reasoning_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_no_reasoning - start_no_reasoning)
            .count();

    double ratio =
        static_cast<double>(no_reasoning_us) / static_cast<double>(std::max(baseline_us, 1L));
    REQUIRE(ratio < 1.5);
}

static void
TestHGraphGetRawVector(const fixtures::HGraphTestIndexPtr& test_index,
                       const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    const std::vector<std::pair<std::string, float>> test_cases = {
        {"fp32", 0.99}, {"sq8", 0.99}, {"sq4_uniform,fp32", 0.95}};
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    ForEachHGraphCase(
        resource,
        test_cases,
        [&](const auto& metric_type, int64_t dim, const auto& base_quantization_str, float recall) {
            RunWithGeneratedBlockSizeLimit([&] {
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.store_raw_vector = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                TestIndex::TestBuildIndex(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
            });
        });
}

HGRAPH_PR_DAILY_CASE("HGraph Support Get Raw Vector",
                     "[ft][update][hgraph]",
                     TestHGraphGetRawVector)

static void
TestHGraphTune(const fixtures::HGraphTestIndexPtr& test_index,
               const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    const std::vector<std::pair<std::string, std::string>> test_cases = {
        /* [case 1] tune basic */
        {"sq8", "sq8"},
        {"fp32", "bf16"},
        {"sq8", "fp32"},
        /* [case 2] tune precise */
        {"sq4,sq8", "sq4,sq8"},
        {"sq4,bf16", "sq4,fp16"},
        {"sq4,bf16", "sq4,fp32"},
        /* [case 3] add precise */
        {"sq4", "sq4,fp16"},
        {"sq4", "sq4,fp32"},
        /* [case 4] drop precise */
        {"sq4,fp32", "sq4"},
        {"sq4,bf16", "sq4"},
    };

    bool is_tested_disable_future_tuning = false;
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str1, base_quantization_str2] : test_cases) {
                INFO(
                    fmt::format("metric_type: {}, dim: {}, base_quantization_str1: {}, "
                                "base_quantization_str2: {}",
                                metric_type,
                                dim,
                                base_quantization_str1,
                                base_quantization_str2));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str1) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    continue;  // Skip invalid RaBitQ configurations
                }

                // Set block size limit for current test iteration
                vsag::Options::Instance().set_block_size_limit(size);

                // Generate index parameters with attribute support enabled
                HGraphTestIndex::HGraphBuildParam build_param1(
                    metric_type, dim, base_quantization_str1);
                build_param1.store_raw_vector = true;
                auto param1 = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param1);

                // Generate alter index param
                HGraphTestIndex::HGraphBuildParam build_param2(
                    metric_type, dim, base_quantization_str2);
                build_param2.store_raw_vector = true;
                auto param2 = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param2);

                // Create index and dataset
                auto index1 = TestIndex::TestFactory(
                    test_index->name, param1, true);  // non-empty, used for test tune
                auto index2 = TestIndex::TestFactory(
                    test_index->name, param2, true);  // empty, used for test serialize and general

                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index1, dataset, true);

                if (not is_tested_disable_future_tuning) {
                    auto index3 = TestIndex::TestFactory(test_index->name, param1, true);
                    TestIndex::TestBuildIndex(index3, dataset, true);
                    // set disable_future_tuning
                    auto set_result = index3->Tune(param2, true);
                    REQUIRE(set_result.has_value());
                    REQUIRE(set_result.value());

                    set_result = index3->Tune(param2, false);
                    REQUIRE(set_result.has_value());
                    REQUIRE_FALSE(set_result.value());
                    is_tested_disable_future_tuning = true;
                }

                // set index param
                auto set_result = index1->Tune(param2);
                REQUIRE(set_result.has_value());
                REQUIRE(set_result.value());

                // serialize test
                TestIndex::TestSerializeFile(index1, index2, dataset, search_param, true);

                // basic test
                HGraphTestIndex::TestGeneral(index1, dataset, search_param, 0.7);
                HGraphTestIndex::TestGeneral(index2, dataset, search_param, 0.7);

                // Restore original block size limit
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Tune", "[ft][search][hgraph]", TestHGraphTune)

TEST_CASE("(PR) HGraph Tune with ignore_reorder", "[ft][search][hgraph][pr]") {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = 1024 * 1024 * 2;
    vsag::Options::Instance().set_block_size_limit(size);

    int64_t dim = 128;
    auto metric_type = "l2";

    std::string param1 = fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 100,
            "build_thread_count": 0,
            "store_raw_vector": true
        }}
    }})",
                                     metric_type,
                                     dim);

    auto index = TestIndex::TestFactory("hgraph", param1, true);
    auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim, 200, metric_type);
    TestIndex::TestBuildIndex(index, dataset, true);

    std::string param2 = fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "use_reorder": true,
            "ignore_reorder": true,
            "base_quantization_type": "fp32",
            "precise_quantization_type": "fp32",
            "precise_io_type": "block_memory_io",
            "max_degree": 32,
            "ef_construction": 100,
            "build_thread_count": 0
        }}
    }})",
                                     metric_type,
                                     dim);

    auto tune_result = index->Tune(param2, true);
    REQUIRE(tune_result.has_value());
    REQUIRE(tune_result.value());

    auto base_range = index->GetMinAndMaxId();
    REQUIRE(base_range.has_value());

    int64_t query_id = dataset->base_->GetIds()[0];
    auto query_dataset = vsag::Dataset::Make();
    query_dataset->Dim(dim)
        ->NumElements(1)
        ->Ids(&query_id)
        ->Float32Vectors(dataset->base_->GetFloat32Vectors())
        ->Owner(false);
    std::string search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    vsag::SearchParam search_param_obj(false, search_param, nullptr, nullptr);
    auto search_result = index->KnnSearch(query_dataset, 5, search_param_obj);
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() > 0);

    vsag::Options::Instance().set_block_size_limit(origin_size);
}

static void
TestHGraphODescentBuild(const fixtures::HGraphTestIndexPtr& test_index,
                        const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));

                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }

                // Set block size limit for current test iteration
                vsag::Options::Instance().set_block_size_limit(size);

                // Generate index parameters with attribute support enabled
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.graph_type = "odescent";
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                // Create index and dataset
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                // Execute build test
                TestIndex::TestBuildIndex(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);

                // Restore original block size limit
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph ODescent Build", "[ft][build][hgraph]", TestHGraphODescentBuild)

static void
TestHGraphMarkRemove(const fixtures::HGraphTestIndexPtr& test_index,
                     const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    ForEachHGraphCase(
        resource,
        resource->test_cases,
        [&](const auto& metric_type, int64_t dim, const auto& base_quantization_str, float recall) {
            RunWithGeneratedBlockSizeLimit([&] {
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.support_remove = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestMarkRemoveIndex(index, dataset, search_param, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
            });
        });
}

HGRAPH_PR_DAILY_CASE("HGraph Mark Remove", "[ft][remove][hgraph]", TestHGraphMarkRemove)

static void
TestHGraphCompressedBuild(const fixtures::HGraphTestIndexPtr& test_index,
                          const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    ForEachHGraphCase(
        resource,
        resource->test_cases,
        [&](const auto& metric_type, int64_t dim, const auto& base_quantization_str, float recall) {
            RunWithGeneratedBlockSizeLimit([&] {
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.graph_storage = "compressed";
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
            });
        });
}

HGRAPH_PR_DAILY_CASE("HGraph Compressed Graph Build",
                     "[ft][build][hgraph]",
                     TestHGraphCompressedBuild)

static void
TestHGraphMerge(const fixtures::HGraphTestIndexPtr& test_index,
                const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto model = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                auto ret = model->Train(dataset->base_);
                REQUIRE(ret.has_value() == true);
                auto merge_index = TestIndex::TestMergeIndexWithSameModel(model, dataset, 5, true);
                HGraphTestIndex::TestGeneral(merge_index, dataset, search_param, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Merge", "[ft][merge][hgraph]", TestHGraphMerge)

static void
TestHGraphAdd(const fixtures::HGraphTestIndexPtr& test_index,
              const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestAddIndex(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
                    HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                }
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Add", "[ft][build][hgraph]", TestHGraphAdd)

static void
TestHGraphNonstandardID(const fixtures::HGraphTestIndexPtr& test_index,
                        const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, 10000, metric_type, false, 0.8, 0, 48);
                TestIndex::TestAddIndex(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_ADD_FROM_EMPTY)) {
                    HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                }
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

TEST_CASE("HGraph Test NonstandardID", "[ft][build][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphNonstandardID(test_index, resource);
}

static void
RunHGraphDuplicateChecks(const fixtures::HGraphTestIndexPtr& test_index,
                         const fixtures::HGraphResourcePtr& resource,
                         const std::string& graph_storage,
                         bool run_full_search_checks,
                         bool run_serialize_check) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto duplicate_pos = GENERATE("prefix", "suffix", "middle");
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                if (base_quantization_str == "sq8_uniform" or
                    base_quantization_str == "sq4_uniform") {
                    // The codes for sq8_uniform and sq4_uniform store the norm values. Even when
                    // vectors are identical, there may be precision errors in the norms, so it's
                    // not possible to determine duplicates based solely on the codes. Since the
                    // uniform version of quantization isn't used for building indexes, this step
                    // can be omitted here.
                    continue;
                }
                INFO(
                    fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}, "
                                "duplicate_pos: {}",
                                metric_type,
                                dim,
                                base_quantization_str,
                                recall,
                                duplicate_pos));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.support_duplicate = true;
                build_param.graph_storage = graph_storage;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDuplicateDataset(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildDuplicateIndex(index, dataset, duplicate_pos, true);
                if (run_full_search_checks) {
                    TestIndex::TestKnnSearch(index, dataset, search_param, recall, true);
                    // TODO(inabao): Fix knn search iter test
                    // TestIndex::TestKnnSearchIter(index, dataset, search_param, recall, true);
                    TestIndex::TestConcurrentKnnSearch(index, dataset, search_param, recall, true);
                    TestIndex::TestRangeSearch(index, dataset, search_param, recall, 10, true);
                    TestIndex::TestRangeSearch(index, dataset, search_param, recall / 2.0, 5, true);
                    TestIndex::TestFilterSearch(index, dataset, search_param, recall, true, true);
                    TestIndex::TestCheckIdExist(index, dataset);
                    TestIndex::TestCalcDistanceById(index, dataset);
                    TestIndex::TestGetRawVectorByIds(index, dataset);
                    TestIndex::TestBatchCalcDistanceById(index, dataset);
                    TestIndex::TestSearchAllocator(index, dataset, search_param, recall, true);
                }
                if (run_serialize_check) {
                    auto index2 = TestIndex::TestFactory(test_index->name, param, true);
                    TestIndex::TestSerializeFile(index, index2, dataset, search_param, true);
                }
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

static void
TestHGraphDuplicate(const fixtures::HGraphTestIndexPtr& test_index,
                    const fixtures::HGraphResourcePtr& resource) {
    RunHGraphDuplicateChecks(test_index, resource, "flat", true, true);
}

static void
TestHGraphDuplicateSerializeCompressed(const fixtures::HGraphTestIndexPtr& test_index,
                                       const fixtures::HGraphResourcePtr& resource) {
    RunHGraphDuplicateChecks(test_index, resource, "compressed", false, true);
}

TEST_CASE("HGraph duplicate stores shared vectors once", "[ft][build][duplicate][hgraph][pr]") {
    using namespace fixtures;

    auto param = GenerateTinyDuplicateHGraphParam();
    auto duplicate_index = BuildTinyDuplicateHGraphIndex(param);

    auto single_index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto single_vector = MakeTinyDenseDataset({1.0F, 0.0F}, {30}, 2);
    REQUIRE(single_index->Add(single_vector).has_value());

    auto distinct_index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto first_distinct = MakeTinyDenseDataset({1.0F, 0.0F}, {20}, 2);
    auto second_distinct = MakeTinyDenseDataset({0.0F, 1.0F}, {21}, 2);
    REQUIRE(distinct_index->Add(first_distinct).has_value());
    REQUIRE(distinct_index->Add(second_distinct).has_value());

    REQUIRE(GetBasicFlattenDetailSize(duplicate_index) == GetBasicFlattenDetailSize(single_index));
    REQUIRE(GetBasicFlattenDetailSize(duplicate_index) < GetBasicFlattenDetailSize(distinct_index));
    auto stats = vsag::JsonType::Parse(duplicate_index->GetStats());
    REQUIRE(stats["code_slots"]["logical_count"].GetInt() == 2);
    REQUIRE(stats["code_slots"]["physical_count"].GetInt() == 1);
    auto memory_detail = vsag::JsonType::Parse(duplicate_index->GetMemoryUsageDetail());
    REQUIRE(memory_detail["logical_code_count"].GetInt() == 2);
    REQUIRE(memory_detail["physical_code_count"].GetInt() == 1);
    RequireTinyDuplicateLogicalCorrectness(duplicate_index);
}

TEST_CASE("HGraph approximate duplicate shares representative stored vector",
          "[ft][build][duplicate][hgraph][pr]") {
    using namespace fixtures;

    auto param = GenerateTinyDuplicateHGraphParam(false, true, 1.0F, "sq8", true);
    auto approximate_index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto first_vector = MakeTinyDenseDataset({1.0F, 0.0F}, {10}, 2);
    auto approximate_duplicate = MakeTinyDenseDataset({1.05F, 0.0F}, {11}, 2);
    REQUIRE(approximate_index->Add(first_vector).has_value());
    REQUIRE(approximate_index->Add(approximate_duplicate).has_value());

    auto single_slot_index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    REQUIRE(single_slot_index->Add(first_vector).has_value());

    auto two_slot_index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto far_vector = MakeTinyDenseDataset({0.0F, 10.0F}, {12}, 2);
    REQUIRE(two_slot_index->Add(first_vector).has_value());
    REQUIRE(two_slot_index->Add(far_vector).has_value());

    REQUIRE(GetBasicFlattenDetailSize(approximate_index) ==
            GetBasicFlattenDetailSize(single_slot_index));
    REQUIRE(GetBasicFlattenDetailSize(approximate_index) <
            GetBasicFlattenDetailSize(two_slot_index));
    auto stats = vsag::JsonType::Parse(approximate_index->GetStats());
    REQUIRE(stats["code_slots"]["logical_count"].GetInt() == 2);
    REQUIRE(stats["code_slots"]["physical_count"].GetInt() == 1);

    int64_t ids[2] = {10, 11};
    auto raw_vectors = approximate_index->GetRawVectorByIds(ids, 2);
    REQUIRE(raw_vectors.has_value());
    auto* vectors = raw_vectors.value()->GetFloat32Vectors();
    REQUIRE(std::fabs(vectors[0] - 1.0F) < 1e-6F);
    REQUIRE(std::fabs(vectors[1]) < 1e-6F);
    REQUIRE(std::fabs(vectors[2] - 1.0F) < 1e-6F);
    REQUIRE(std::fabs(vectors[3]) < 1e-6F);

    float first_query[2] = {1.0F, 0.0F};
    float second_query[2] = {1.05F, 0.0F};
    auto distance_to_first = approximate_index->CalcDistanceById(first_query, 11);
    auto distance_to_second = approximate_index->CalcDistanceById(second_query, 11);
    REQUIRE(distance_to_first.has_value());
    REQUIRE(distance_to_second.has_value());
    REQUIRE(std::fabs(distance_to_first.value()) < 1e-6F);
    REQUIRE(distance_to_second.value() > 1e-4F);

    auto deferred_param = GenerateTinyDuplicateHGraphParam(false, true, 1.0F, "rabitq");
    auto deferred_index = TestIndex::TestFactory(HGraphTestIndex::name, deferred_param, true);
    auto deferred_vectors = MakeTinyDenseDataset({1.0F, 0.0F, 1.05F, 0.0F}, {20, 21}, 2);
    REQUIRE(deferred_index->Add(deferred_vectors).has_value());
    auto deferred_stats = vsag::JsonType::Parse(deferred_index->GetStats());
    REQUIRE(deferred_stats["code_slots"]["logical_count"].GetInt() == 2);
    REQUIRE(deferred_stats["code_slots"]["physical_count"].GetInt() == 1);
}

TEST_CASE("HGraph duplicate update detaches shared stored vector",
          "[ft][build][update][duplicate][hgraph][pr]") {
    using namespace fixtures;

    auto param = GenerateTinyDuplicateHGraphParam();
    auto index = BuildTinyDuplicateHGraphIndex(param);
    auto one_slot_size = GetBasicFlattenDetailSize(index);

    auto distinct_index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto first_vector = MakeTinyDenseDataset({1.0F, 0.0F}, {20}, 2);
    auto second_vector = MakeTinyDenseDataset({0.0F, 1.0F}, {21}, 2);
    REQUIRE(distinct_index->Add(first_vector).has_value());
    REQUIRE(distinct_index->Add(second_vector).has_value());
    auto two_slot_size = GetBasicFlattenDetailSize(distinct_index);
    REQUIRE(one_slot_size < two_slot_size);

    auto updated_vector = MakeTinyDenseDataset({0.0F, 1.0F}, {11}, 2);
    auto update_result = index->UpdateVector(11, updated_vector, true);
    REQUIRE(update_result.has_value());
    REQUIRE(update_result.value());

    REQUIRE(GetBasicFlattenDetailSize(index) == two_slot_size);
    auto stats = vsag::JsonType::Parse(index->GetStats());
    REQUIRE(stats["code_slots"]["logical_count"].GetInt() == 2);
    REQUIRE(stats["code_slots"]["physical_count"].GetInt() == 2);

    int64_t ids[2] = {10, 11};
    auto raw_vectors = index->GetRawVectorByIds(ids, 2);
    REQUIRE(raw_vectors.has_value());
    auto* vectors = raw_vectors.value()->GetFloat32Vectors();
    REQUIRE(std::fabs(vectors[0] - 1.0F) < 1e-6F);
    REQUIRE(std::fabs(vectors[1]) < 1e-6F);
    REQUIRE(std::fabs(vectors[2]) < 1e-6F);
    REQUIRE(std::fabs(vectors[3] - 1.0F) < 1e-6F);

    float first_query[2] = {1.0F, 0.0F};
    float second_query[2] = {0.0F, 1.0F};
    auto first_distance = index->CalcDistanceById(first_query, 10);
    auto old_query_to_updated = index->CalcDistanceById(first_query, 11);
    auto updated_distance = index->CalcDistanceById(second_query, 11);
    REQUIRE(first_distance.has_value());
    REQUIRE(old_query_to_updated.has_value());
    REQUIRE(updated_distance.has_value());
    REQUIRE(std::fabs(first_distance.value()) < 1e-6F);
    REQUIRE(old_query_to_updated.value() > 1.0F);
    REQUIRE(std::fabs(updated_distance.value()) < 1e-6F);

    auto updated_query = MakeTinyDenseDataset({0.0F, 1.0F}, {11}, 2);
    auto brute_force_result = index->KnnSearch(
        updated_query, 1, R"({"hgraph":{"ef_search":16,"brute_force_threshold":1.0}})");
    REQUIRE(brute_force_result.has_value());
    REQUIRE(brute_force_result.value()->GetIds()[0] == 11);
    REQUIRE(std::fabs(brute_force_result.value()->GetDistances()[0]) < 1e-6F);
}

TEST_CASE("HGraph duplicate storage dedup survives concurrent add and search",
          "[ft][build][concurrent][duplicate][hgraph][pr]") {
    using namespace fixtures;

    // The dedup mapping grows physical code storage lazily as new logical ids
    // are added. This test stresses that growth against concurrent searches:
    // a search resolving a logical id reads the flatten storage while another
    // thread may be appending a brand-new physical slot. A `memory_io` backend
    // is used on purpose because it reallocates (moves) the buffer on every
    // growth, so any read that races an unsynchronised realloc faults
    // deterministically instead of relying on the 128MB block boundary that
    // `block_memory_io` only crosses rarely.
    constexpr int64_t kDim = 16;
    constexpr int64_t kGroupCount = 400;

    HGraphTestIndex::HGraphBuildParam build_param("l2", kDim, "fp32");
    build_param.support_duplicate = true;
    build_param.thread_count = 4;
    auto json =
        vsag::JsonType::Parse(HGraphTestIndex::GenerateHGraphBuildParametersString(build_param));
    json["index_param"]["base_io_type"].SetString("memory_io");
    auto param = json.Dump();

    auto index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);

    // Deterministic per-group vectors; ids 2*g and 2*g+1 are exact duplicates so
    // half of the logical ids alias an existing physical slot and the other half
    // append a fresh one.
    auto make_group_vector = [&](int64_t group) {
        std::vector<float> vec(static_cast<size_t>(kDim), 0.0F);
        vec[group % kDim] = 1.0F + static_cast<float>(group);
        vec[(group + 1) % kDim] = 0.5F;
        return vec;
    };

    // Seed a small base so searches have a non-empty graph from the start.
    constexpr int64_t kSeedGroups = 20;
    for (int64_t g = 0; g < kSeedGroups; ++g) {
        auto vec = make_group_vector(g);
        REQUIRE(index->Add(MakeTinyDenseDataset(vec, {2 * g}, kDim)).has_value());
        REQUIRE(index->Add(MakeTinyDenseDataset(vec, {2 * g + 1}, kDim)).has_value());
    }

    std::atomic<bool> failed{false};
    std::atomic<bool> stop_search{false};

    auto adder = [&]() {
        for (int64_t g = kSeedGroups; g < kGroupCount; ++g) {
            auto vec = make_group_vector(g);
            if (not index->Add(MakeTinyDenseDataset(vec, {2 * g}, kDim)).has_value() or
                not index->Add(MakeTinyDenseDataset(vec, {2 * g + 1}, kDim)).has_value()) {
                failed.store(true);
                return;
            }
        }
    };

    auto searcher = [&]() {
        std::mt19937 rng(12345);
        while (not stop_search.load(std::memory_order_acquire)) {
            int64_t g = static_cast<int64_t>(rng() % kSeedGroups);
            auto vec = make_group_vector(g);
            auto query = MakeTinyDenseDataset(vec, {0}, kDim);
            // Graph search resolves every visited candidate through the dedup
            // code-slot mapping while the adder concurrently grows the physical
            // storage, exercising the mapped-flatten read path against storage
            // resize. (Brute-force search is intentionally avoided here: it
            // scans [0, total_count_) directly and would race the not-yet-written
            // codes of an in-flight Add independently of the dedup mapping.)
            auto result = index->KnnSearch(query, 2, R"({"hgraph":{"ef_search":32}})");
            if (not result.has_value()) {
                failed.store(true);
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.emplace_back(adder);
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(searcher);
    }
    threads.front().join();  // wait for the adder
    stop_search.store(true, std::memory_order_release);
    for (size_t i = 1; i < threads.size(); ++i) {
        threads[i].join();
    }
    REQUIRE_FALSE(failed.load());

    // Every logical id must remain individually addressable, and each duplicate
    // pair must resolve to the same stored vector after all the concurrent
    // growth.
    REQUIRE(index->GetNumElements() == 2 * kGroupCount);
    for (int64_t g = 0; g < kGroupCount; g += 37) {
        REQUIRE(index->CheckIdExist(2 * g));
        REQUIRE(index->CheckIdExist(2 * g + 1));
        auto vec = make_group_vector(g);
        auto first = index->CalcDistanceById(vec.data(), 2 * g);
        auto second = index->CalcDistanceById(vec.data(), 2 * g + 1);
        REQUIRE(first.has_value());
        REQUIRE(second.has_value());
        REQUIRE(std::fabs(first.value()) < 1e-4F);
        REQUIRE(std::fabs(second.value()) < 1e-4F);
    }
}

TEST_CASE("HGraph duplicate storage dedup is disabled without support_duplicate",
          "[ft][build][duplicate][compatibility][hgraph][pr]") {
    using namespace fixtures;

    auto param = GenerateTinyDuplicateHGraphParam(false, false);
    auto equal_vector_index = BuildTinyDuplicateHGraphIndex(param);

    auto distinct_index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto first_distinct = MakeTinyDenseDataset({1.0F, 0.0F}, {20}, 2);
    auto second_distinct = MakeTinyDenseDataset({0.0F, 1.0F}, {21}, 2);
    REQUIRE(distinct_index->Add(first_distinct).has_value());
    REQUIRE(distinct_index->Add(second_distinct).has_value());

    REQUIRE(GetBasicFlattenDetailSize(equal_vector_index) ==
            GetBasicFlattenDetailSize(distinct_index));
    RequireTinyDuplicateVectorAccess(equal_vector_index);
}

TEST_CASE("HGraph duplicate code-slot mapping survives serialize",
          "[ft][serialize][duplicate][hgraph][pr]") {
    using namespace fixtures;

    auto param = GenerateTinyDuplicateHGraphParam();
    auto index = BuildTinyDuplicateHGraphIndex(param);
    auto serialize_binary = index->Serialize();
    REQUIRE(serialize_binary.has_value());

    auto restored_index = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto deserialize_result = restored_index->Deserialize(serialize_binary.value());
    REQUIRE(deserialize_result.has_value());
    REQUIRE(GetBasicFlattenDetailSize(restored_index) == GetBasicFlattenDetailSize(index));
    RequireTinyDuplicateLogicalCorrectness(restored_index);
}

TEST_CASE("HGraph duplicate old serial layout keeps logical vector access",
          "[ft][serialize][duplicate][compatibility][hgraph][pr]") {
    using namespace fixtures;

    auto old_param = GenerateTinyDuplicateHGraphParam(true);
    auto old_index = BuildTinyDuplicateHGraphIndex(old_param);
    auto serialize_binary = old_index->Serialize();
    REQUIRE(serialize_binary.has_value());

    auto current_param = GenerateTinyDuplicateHGraphParam();
    auto restored_index = TestIndex::TestFactory(HGraphTestIndex::name, current_param, true);
    auto deserialize_result = restored_index->Deserialize(serialize_binary.value());
    REQUIRE(deserialize_result.has_value());
    // The old serial layout carries no code-slot mapping, so a current
    // mapping-capable reader rebuilds an identity mapping: every logical id keeps
    // its own physical slot (no dedup). Both labels must remain individually
    // addressable with their original vectors. Note: we intentionally do not
    // compare flatten sizes against `old_index` here — a current-format index is
    // shrunk to its physical count on Serialize() while an old-format index is
    // not, so the two sizes are not comparable.
    REQUIRE(restored_index->GetNumElements() == old_index->GetNumElements());
    auto restored_stats = vsag::JsonType::Parse(restored_index->GetStats());
    REQUIRE(restored_stats["code_slots"]["logical_count"].GetInt() == 2);
    REQUIRE(restored_stats["code_slots"]["physical_count"].GetInt() == 2);
    RequireTinyDuplicateVectorAccess(restored_index);
}

TEST_CASE("HGraph duplicate mapped serial data rejects non-mapping readers",
          "[ft][serialize][duplicate][compatibility][hgraph][pr]") {
    using namespace fixtures;

    auto param = GenerateTinyDuplicateHGraphParam();
    auto index = BuildTinyDuplicateHGraphIndex(param);
    auto serialize_binary = index->Serialize();
    REQUIRE(serialize_binary.has_value());

    auto old_param = GenerateTinyDuplicateHGraphParam(true);
    auto old_reader = TestIndex::TestFactory(HGraphTestIndex::name, old_param, true);
    auto old_reader_result = old_reader->Deserialize(serialize_binary.value());
    REQUIRE_FALSE(old_reader_result.has_value());
    REQUIRE(old_reader_result.error().type == vsag::ErrorType::INVALID_ARGUMENT);

    auto no_duplicate_param = GenerateTinyDuplicateHGraphParam(false, false);
    auto no_duplicate_reader =
        TestIndex::TestFactory(HGraphTestIndex::name, no_duplicate_param, true);
    auto no_duplicate_reader_result = no_duplicate_reader->Deserialize(serialize_binary.value());
    REQUIRE_FALSE(no_duplicate_reader_result.has_value());
    REQUIRE(no_duplicate_reader_result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("HGraph duplicate rejects unsupported ODescent and force remove paths",
          "[ft][build][duplicate][compatibility][hgraph][pr]") {
    using namespace fixtures;

    auto odescent_json = vsag::JsonType::Parse(GenerateTinyDuplicateHGraphParam());
    odescent_json["index_param"]["graph_type"].SetString("odescent");
    auto odescent_index = TestIndex::TestFactory(HGraphTestIndex::name, odescent_json.Dump(), true);
    auto build_data = MakeTinyDenseDataset({1.0F, 0.0F, 0.0F, 1.0F}, {10, 11}, 2);
    auto build_result = odescent_index->Build(build_data);
    REQUIRE_FALSE(build_result.has_value());
    REQUIRE(build_result.error().type == vsag::ErrorType::INVALID_ARGUMENT);

    auto force_remove_json = vsag::JsonType::Parse(GenerateTinyDuplicateHGraphParam());
    force_remove_json["index_param"]["support_remove"].SetBool(true);
    auto force_remove_index =
        TestIndex::TestFactory(HGraphTestIndex::name, force_remove_json.Dump(), true);
    REQUIRE(force_remove_index->Add(MakeTinyDenseDataset({1.0F, 0.0F}, {10}, 2)).has_value());
    auto remove_result = force_remove_index->Remove(10, vsag::RemoveMode::FORCE_REMOVE);
    REQUIRE_FALSE(remove_result.has_value());
    REQUIRE(remove_result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(force_remove_index->CheckIdExist(10));
}

TEST_CASE("HGraph duplicate merge fails before mixing logical and physical ids",
          "[ft][merge][duplicate][compatibility][hgraph][pr]") {
    using namespace fixtures;

    auto param = GenerateTinyDuplicateHGraphParam();
    auto target = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    auto source = BuildTinyDuplicateHGraphIndex(param);
    vsag::IdMapFunction id_map = [](int64_t id) { return std::make_tuple(true, id); };
    std::vector<vsag::MergeUnit> merge_units{{source, id_map}};
    auto merge_result = target->Merge(merge_units);
    REQUIRE_FALSE(merge_result.has_value());
    REQUIRE(merge_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
    REQUIRE(target->GetNumElements() == 0);
}

TEST_CASE("HGraph duplicate ignores imported build cache and uses normal build",
          "[ft][build][cache][duplicate][compatibility][hgraph][pr]") {
    using namespace fixtures;

    auto param = GenerateTinyDuplicateHGraphParam();
    std::vector<float> vectors{1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F};
    std::vector<int64_t> ids{10, 11, 12};
    std::vector<std::string> source_ids{"source-10", "source-11", "source-12"};
    auto make_dataset = [&]() {
        return vsag::Dataset::Make()
            ->Dim(2)
            ->NumElements(3)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->SourceID(source_ids.data())
            ->Owner(false);
    };

    auto baseline = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    REQUIRE(baseline->Build(make_dataset()).has_value());
    std::stringstream cache;
    REQUIRE(baseline->ExportCache(cache).has_value());

    auto rebuilt = TestIndex::TestFactory(HGraphTestIndex::name, param, true);
    REQUIRE(rebuilt->ImportCache(cache).has_value());
    REQUIRE(rebuilt->Build(make_dataset()).has_value());
    REQUIRE(rebuilt->GetNumElements() == 3);

    auto stats = vsag::JsonType::Parse(rebuilt->GetStats());
    REQUIRE(stats["build_cache_hit_rate"].Contains("skipped_reason"));
    REQUIRE(stats["code_slots"]["logical_count"].GetInt() == 3);
    REQUIRE(stats["code_slots"]["physical_count"].GetInt() == 2);
}

HGRAPH_PR_DAILY_CASE("HGraph Duplicate", "[ft][build][hgraph][duplicate]", TestHGraphDuplicate)

TEST_CASE("(PR) HGraph Duplicate Serialize Compressed",
          "[ft][build][duplicate][serialize][hgraph][pr]") {
    auto test_index = std::make_shared<fixtures::HGraphTestIndex>();
    auto resource = test_index->GetResource(true);
    TestHGraphDuplicateSerializeCompressed(test_index, resource);
}

static void
TestHGraphSearchWithDirtyVector(const fixtures::HGraphTestIndexPtr& test_index,
                                const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        auto dataset = HGraphTestIndex::pool.GetNanDataset(metric_type);
        auto dim = dataset->dim_;

        for (auto& [base_quantization_str, recall] : resource->test_cases) {
            INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                             metric_type,
                             dim,
                             base_quantization_str,
                             recall));
            if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                continue;  // Skip invalid RaBitQ configurations
            }
            vsag::Options::Instance().set_block_size_limit(size);
            HGraphTestIndex::HGraphBuildParam build_param(metric_type, dim, base_quantization_str);
            auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
            auto index = TestIndex::TestFactory(test_index->name, param, true);
            TestIndex::TestBuildIndex(index, dataset, true);
            TestIndex::TestSearchWithDirtyVector(index, dataset, search_param, true);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Search with Dirty Vector",
                     "[ft][search][hgraph]",
                     TestHGraphSearchWithDirtyVector)

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex,
                             "HGraph Search with Sparse Vector",
                             "[ft][concurrent][hgraph]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = "ip";
    INFO(fmt::format("metric_type: {}", metric_type));
    auto dim = 128;
    auto dataset = pool.GetSparseDatasetAndCreate(base_count, dim, 0.8);
    auto search_param = fmt::format(fixtures::search_param_tmp, 100, false);
    vsag::Options::Instance().set_block_size_limit(size);

    HGraphTestIndex::HGraphBuildParam build_param(metric_type, dim, "sparse");
    build_param.data_type = "sparse";
    auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
    auto index = TestFactory(name, param, true);
    TestConcurrentAdd(index, dataset, true);
    TestGeneral(index, dataset, search_param, true);
    auto index2 = TestIndex::TestFactory(name, param, true);
    TestIndex::TestSerializeFile(index, index2, dataset, search_param, true);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

static void
TestHGraphConcurrentAdd(const fixtures::HGraphTestIndexPtr& test_index,
                        const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }

                // Set block size limit for current test iteration
                vsag::Options::Instance().set_block_size_limit(size);

                // Generate index parameters with attribute support enabled
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                // Execute build test
                TestIndex::TestConcurrentAdd(index, dataset, true);
                if (index->CheckFeature(vsag::SUPPORT_ADD_CONCURRENT)) {
                    HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                }
                // Restore original block size limit
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Concurrent Add",
                     "[ft][build][concurrent][hgraph]",
                     TestHGraphConcurrentAdd)

static void
TestHGraphConcurrentAddSearchRemove(const fixtures::HGraphTestIndexPtr& test_index,
                                    const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }

                // Set block size limit for current test iteration
                vsag::Options::Instance().set_block_size_limit(size);

                // Generate index parameters with attribute support enabled
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.support_remove = true;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                // Execute build test
                TestIndex::TestConcurrentAddSearchRemove(index, dataset, search_param, true);
                // Restore original block size limit
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Concurrent Add Search Remove",
                     "[ft][build][concurrent][hgraph]",
                     TestHGraphConcurrentAddSearchRemove)

static void
TestHGraphSerialize(const fixtures::HGraphTestIndexPtr& test_index,
                    const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestBuildIndex(index, dataset, true);
                auto index2 = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeFile(index, index2, dataset, search_param, true);
                index2 = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeBinarySet(index, index2, dataset, search_param, true);
                index2 = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeReaderSet(
                    index, index2, dataset, search_param, test_index->name, true);
                index2 = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeWriteFunc(index, index2, dataset, search_param, true);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Serialize File",
                     "[ft][serialize][hgraph][serialization]",
                     TestHGraphSerialize)

static void
TestHGraphReaderIO(const fixtures::HGraphTestIndexPtr& test_index,
                   const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    (metric_type != "l2" || dim < fixtures::RABITQ_MIN_RACALL_DIM)) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);

                TestIndex::TestBuildIndex(index, dataset, true);
                if (base_quantization_str.find(',') != std::string::npos) {
                    build_param.quantization_str += ",reader_io";
                }
                auto reader_param =
                    HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index2 = TestIndex::TestFactory(test_index->name, reader_param, true);
                TestIndex::TestSerializeReaderSet(
                    index, index2, dataset, search_param, test_index->name, true);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Reader IO", "[ft][serialize][hgraph]", TestHGraphReaderIO)

static void
TestHGraphClone(const fixtures::HGraphTestIndexPtr& test_index,
                const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 32;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestClone(index, dataset, search_param);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Clone", "[ft][clone][hgraph]", TestHGraphClone)

static void
TestHGraphExportModel(const fixtures::HGraphTestIndexPtr& test_index,
                      const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto index2 = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestExportModel(index, index2, dataset, search_param);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Export Model", "[ft][export][hgraph]", TestHGraphExportModel)

static void
TestHGraphRandomAllocator(const fixtures::HGraphTestIndexPtr& test_index,
                          const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto allocator = std::make_shared<fixtures::RandomAllocator>();

    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.thread_count = 1;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = vsag::Factory::CreateIndex(test_index->name, param, allocator.get());
                if (not index.has_value()) {
                    continue;
                }
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestContinueAddIgnoreRequire(index.value(), dataset);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Build & ContinueAdd Test With Random Allocator",
                     "[ft][build][hgraph]",
                     TestHGraphRandomAllocator)

static void
TestHGraphDuplicateBuild(const fixtures::HGraphTestIndexPtr& test_index,
                         const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto graph_storage = GENERATE("flat", "compressed");
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);

                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.support_duplicate = true;
                build_param.graph_storage = graph_storage;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestDuplicateAdd(index, dataset);
                HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Duplicate Build",
                     "[ft][build][duplicate][hgraph]",
                     TestHGraphDuplicateBuild)

static void
TestHGraphEstimateMemoryAndGetMemoryUsage(const fixtures::HGraphTestIndexPtr& test_index,
                                          const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 64;
    uint64_t estimate_count = 1000;

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         estimate_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestEstimateMemory(test_index->name, param, dataset);
                TestIndex::TestGetMemoryUsage(test_index->name, param, dataset);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Estimate Memory And Get Memory Usage",
                     "[ft][memory][hgraph]",
                     TestHGraphEstimateMemoryAndGetMemoryUsage)

TEST_CASE_PERSISTENT_FIXTURE(fixtures::HGraphTestIndex,
                             "HGraph ELP Optimizer",
                             "[ft][build][hgraph]") {
    fixtures::logger::LoggerReplacer _;
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = fixtures::RandomSelect<std::string>({"l2", "ip", "cosine"})[0];
    INFO(fmt::format("metric_type: {}", metric_type));

    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "use_reorder": true,
            "use_elp_optimizer": {},
            "base_quantization_type": "sq4_uniform",
            "max_degree": 64,
            "ef_construction": 200,
            "precise_quantization_type": "fp32",
            "ignore_reorder": true
        }}
    }}
    )";

    auto dim = 128;
    vsag::Options::Instance().set_block_size_limit(size);
    auto base = pool.GetDatasetAndCreate(dim, 100, metric_type);
    std::string param_weak = fmt::format(parameter_temp, metric_type, dim, false);
    std::string param_strong = fmt::format(parameter_temp, metric_type, dim, true);
    auto index_weak = TestFactory(name, param_weak, true);
    TestBuildIndex(index_weak, base);
    auto index_strong = TestFactory(name, param_strong, true);
    TestBuildIndex(index_strong, base);
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

static void
TestHGraphIgnoreReorder(const fixtures::HGraphTestIndexPtr& test_index,
                        const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    constexpr auto parameter_temp_reorder = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "use_reorder": true,
            "base_quantization_type": "sq8",
            "max_degree": 96,
            "ef_construction": 400,
            "precise_quantization_type": "fp32",
            "ignore_reorder": true
        }}
    }}
    )";
    float recall = 0.95;
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            INFO(fmt::format("metric_type: {}, dim: {}, recall: {}", metric_type, dim, recall));
            vsag::Options::Instance().set_block_size_limit(size);
            auto dataset =
                HGraphTestIndex::pool.GetDatasetAndCreate(dim, resource->base_count, metric_type);
            std::string param = fmt::format(parameter_temp_reorder, metric_type, dim);
            auto index = TestIndex::TestFactory(test_index->name, param, true);
            TestIndex::TestBuildIndex(index, dataset);
            HGraphTestIndex::TestGeneral(index, dataset, search_param, recall);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Ignore Reorder", "[ft][search][hgraph]", TestHGraphIgnoreReorder)

static void
TestHGraphSearchDisableReorder(const fixtures::HGraphTestIndexPtr& test_index,
                               const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    constexpr const char* search_param_tmp_disable_reorder = R"({{
            "hgraph": {{
                "ef_search": 200,
                "enable_reorder": {}
            }}
        }})";

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            auto base_quantization_str = "sq4_uniform,fp32";
            float recall_with_reorder = 0.95F;
            float recall_without_reorder = 0.75F;
            INFO(
                fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, "
                            "recall_with_reorder: {}, recall_without_reorder: {}",
                            metric_type,
                            dim,
                            base_quantization_str,
                            recall_with_reorder,
                            recall_without_reorder));
            vsag::Options::Instance().set_block_size_limit(size);
            HGraphTestIndex::HGraphBuildParam build_param(metric_type, dim, base_quantization_str);
            auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
            auto index = TestIndex::TestFactory(test_index->name, param, true);
            auto dataset =
                HGraphTestIndex::pool.GetDatasetAndCreate(dim, resource->base_count, metric_type);
            TestIndex::TestBuildIndex(index, dataset, true);
            auto recall_result_with_reorder =
                TestIndex::TestKnnSearch(index,
                                         dataset,
                                         fmt::format(search_param_tmp_disable_reorder, true),
                                         recall_with_reorder,
                                         true);
            auto recall_result_without_reorder =
                TestIndex::TestKnnSearch(index,
                                         dataset,
                                         fmt::format(search_param_tmp_disable_reorder, false),
                                         recall_without_reorder,
                                         true);
            auto iter_recall_result_with_reorder =
                TestIndex::TestKnnSearchIter(index,
                                             dataset,
                                             fmt::format(search_param_tmp_disable_reorder, true),
                                             recall_with_reorder,
                                             true);
            auto iter_recall_result_without_reorder =
                TestIndex::TestKnnSearchIter(index,
                                             dataset,
                                             fmt::format(search_param_tmp_disable_reorder, false),
                                             recall_without_reorder,
                                             true);
            auto search_param_with_reorder = fmt::format(search_param_tmp_disable_reorder, true);
            auto search_param_without_reorder =
                fmt::format(search_param_tmp_disable_reorder, false);
            REQUIRE(recall_result_with_reorder > recall_result_without_reorder);
            REQUIRE(iter_recall_result_with_reorder > iter_recall_result_without_reorder);
            RequireRangeSearchDisableReorderChangesResult(
                index, dataset, search_param_with_reorder, search_param_without_reorder);
            vsag::Options::Instance().set_block_size_limit(origin_size);
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Search Disable Reorder",
                     "[ft][search][hgraph]",
                     TestHGraphSearchDisableReorder)

static void
TestHGraphWithExtraInfo(const fixtures::HGraphTestIndexPtr& test_index,
                        const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    uint64_t extra_info_size = 256;
    auto search_ex_filter_param = fmt::format(fixtures::search_param_tmp, 500, true);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.extra_info_size = extra_info_size;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(dim,
                                                                         resource->base_count,
                                                                         metric_type,
                                                                         false /*with_path*/,
                                                                         0.8 /*valid_ratio*/,
                                                                         extra_info_size);
                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestKnnSearch(index, dataset, search_param, recall, true);
                TestIndex::TestKnnSearchIter(index, dataset, search_param, recall, true);
                TestIndex::TestRangeSearch(index, dataset, search_param, recall, 10, true);
                TestIndex::TestGetExtraInfoById(index, dataset, extra_info_size);
                TestIndex::TestKnnSearchExFilter(
                    index, dataset, search_ex_filter_param, recall, true);
                TestIndex::TestUpdateExtraInfo(index, dataset, extra_info_size);
                TestIndex::TestKnnSearchExFilter(
                    index, dataset, search_ex_filter_param, recall, true);
                TestIndex::TestKnnSearchIter(
                    index, dataset, search_ex_filter_param, recall, true, true);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph With Extra Info", "[ft][search][hgraph]", TestHGraphWithExtraInfo)

static void
TestHGraphSearchOverTime(const fixtures::HGraphTestIndexPtr& test_index,
                         const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    constexpr const char* search_param = R"({
            "hgraph": {
                "ef_search": 200,
                "timeout_ms": 5.0
            }
        })";
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                TestIndex::TestSearchOvertime(index, dataset, search_param);
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Search Over Time", "[ft][search][hgraph]", TestHGraphSearchOverTime)

static void
TestHGraphDiskIOType(const fixtures::HGraphTestIndexPtr& test_index,
                     const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto search_param = fmt::format(fixtures::search_param_tmp, 200, false);
    float recall = 0.98;
    const std::vector<std::pair<std::string, std::string>> io_cases = {
        {"sq8_uniform,bf16", "sq8_uniform,bf16,buffer_io"},
        {"rabitq,fp16", "rabitq,fp16,async_io"},
        {"rabitq,fp16", "rabitq,fp16,mmap_io"},
    };
    const std::vector<std::string> graph_io_types = {"block_memory_io", "mmap_io", "async_io"};
    auto select_idx = 0;
    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [memory_io_str, disk_io_str] : io_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, memory_io_str: {}, disk_io_str: {}",
                                 metric_type,
                                 dim,
                                 memory_io_str,
                                 disk_io_str));
                if (HGraphTestIndex::IsRaBitQ(memory_io_str) &&
                    (dim < fixtures::RABITQ_MIN_RACALL_DIM)) {
                    continue;  // Skip invalid RaBitQ configurations
                }
                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(metric_type, dim, memory_io_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);
                TestIndex::TestBuildIndex(index, dataset, true);
                build_param.quantization_str = disk_io_str;

                auto graph_io_type = graph_io_types[select_idx];
                build_param.graph_io_type = graph_io_type;
                param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto disk_index = TestIndex::TestFactory(test_index->name, param, true);
                TestIndex::TestSerializeFile(index, disk_index, dataset, search_param, true);
                HGraphTestIndex::TestGeneral(disk_index, dataset, search_param, recall);
                ++select_idx;
                select_idx %= graph_io_types.size();
                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Disk IO Type Index", "[ft][serialize][hgraph]", TestHGraphDiskIOType)

TEST_CASE("HGraph Concurrent Read Write", "[ft][concurrent][hgraph]") {
    uint32_t op_num = 10000;
    uint32_t dim = 128;
    uint32_t top_k = 5;
    float read_ratio = 0.8;
    float thread_num = 5;

    std::vector<std::vector<float>> dataset;
    dataset.reserve(op_num);
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-10.0, 10.0);
    for (uint32_t i = 0; i < op_num; ++i) {
        std::vector<float> vector_data;
        vector_data.reserve(dim);
        for (uint32_t j = 0; j < dim; ++j) {
            vector_data.emplace_back(dist(rng));
        }
        dataset.emplace_back(std::move(vector_data));
    }

    std::string search_params = R"({
        "hgraph": {
          "ef_search": 100
        }
    })";

    std::string hgraph_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "base_io_type": "block_memory_io",
            "max_degree": 32,
            "ef_construction": 100,
            "alpha":1.2,
            "use_reorder": false
        }
    })";
    auto build_res = vsag::Factory::CreateIndex("hgraph", hgraph_params);
    auto vsag_index = std::move(build_res.value());

    std::atomic<uint32_t> actual_read_num{0};
    std::atomic<uint32_t> actual_write_num{0};
    uint32_t expect_read_num = op_num * read_ratio;
    uint32_t expect_write_num = op_num - expect_read_num;

    auto test_func = [&]() {
        // Decide whether each operation is a write or a read.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0, 1.0);

        uint32_t local_read_num{0};
        uint32_t local_write_num{0};

        auto write_func = [&]() {
            uint32_t old_value = actual_write_num.fetch_add(1);
            if (old_value >= expect_write_num) {
                return;
            }

            int64_t vec_id = static_cast<int64_t>(old_value);
            auto base = vsag::Dataset::Make();
            base->NumElements(1)
                ->Dim(dim)
                ->Ids(&vec_id)
                ->Float32Vectors(dataset[old_value].data())
                ->Owner(false);

            // Do hnsw add.
            auto res = vsag_index->Add(base);
            if (!res.has_value()) {
                std::cout << "put error: " << res.error().message << std::endl;
            }

            ++local_write_num;
        };

        auto read_func = [&]() {
            uint32_t old_value = actual_read_num.fetch_add(1);
            if (old_value >= expect_read_num) {
                return;
            }

            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(dim)
                ->Float32Vectors(dataset[old_value].data())
                ->Owner(false);

            // Do knn search.
            auto res = vsag_index->KnnSearch(query, top_k, search_params);
            if (!res.has_value()) {
                std::cout << "query error: " << res.error().message << std::endl;
            }

            ++local_read_num;
        };

        while (true) {
            if (actual_read_num >= expect_read_num && actual_write_num >= expect_write_num) {
                break;
            }

            if (actual_read_num >= expect_read_num) {
                write_func();
            } else if (actual_write_num >= expect_write_num) {
                read_func();
            } else if (dist(gen) > read_ratio) {
                write_func();
            } else {
                read_func();
            }
        }
    };

    auto threads = std::make_unique<std::vector<std::thread>>();
    threads->reserve(thread_num);
    for (uint32_t i = 0; i < thread_num; ++i) {
        threads->emplace_back(test_func);
    }

    // Wait write completed.
    for (auto& thread : *threads) {
        thread.join();
    }
}

// Tests for hops_limit search parameter
static void
TestHGraphHopsLimit(const fixtures::HGraphTestIndexPtr& test_index,
                    const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);

    // Test with valid hops_limit (> ef_search)
    constexpr static const char* search_param_with_hops_limit = R"({
        "hgraph": {
            "ef_search": 30,
            "hops_limit": 100
        }
    })";

    // Test with invalid hops_limit (<= ef_search) - should warn and ignore
    constexpr static const char* search_param_invalid_hops_limit = R"({
        "hgraph": {
            "ef_search": 100,
            "hops_limit": 50
        }
    })";

    // Test without hops_limit (default behavior)
    constexpr static const char* search_param_without_hops_limit = R"({
        "hgraph": {
            "ef_search": 30
        }
    })";

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}, recall: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str,
                                 recall));
                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }

                vsag::Options::Instance().set_block_size_limit(size);
                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);
                auto index = TestIndex::TestFactory(test_index->name, param, true);
                auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                    dim, resource->base_count, metric_type);

                TestIndex::TestBuildIndex(index, dataset, true);

                // Test with valid hops_limit - should work normally
                TestIndex::TestKnnSearch(
                    index, dataset, search_param_with_hops_limit, recall * 0.9, true);

                // Test without hops_limit - should work normally
                TestIndex::TestKnnSearch(
                    index, dataset, search_param_without_hops_limit, recall, true);

                // Test with invalid hops_limit - should warn but still work
                TestIndex::TestKnnSearch(
                    index, dataset, search_param_invalid_hops_limit, recall, true);

                vsag::Options::Instance().set_block_size_limit(origin_size);
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Hops Limit", "[ft][search][hgraph]", TestHGraphHopsLimit)

static void
TestHGraphReverseEdges(const fixtures::HGraphTestIndexPtr& test_index,
                       const fixtures::HGraphResourcePtr& resource) {
    using namespace fixtures;
    auto search_param = fmt::format(search_param_tmp, 100, false);

    for (auto metric_type : resource->metric_types) {
        for (auto dim : resource->dims) {
            for (auto& [base_quantization_str, recall] : resource->test_cases) {
                INFO(fmt::format("metric_type: {}, dim: {}, base_quantization_str: {}",
                                 metric_type,
                                 dim,
                                 base_quantization_str));

                if (HGraphTestIndex::IsRaBitQ(base_quantization_str) &&
                    dim < fixtures::RABITQ_MIN_RACALL_DIM) {
                    dim = fixtures::RABITQ_MIN_RACALL_DIM;
                }

                HGraphTestIndex::HGraphBuildParam build_param(
                    metric_type, dim, base_quantization_str);
                build_param.thread_count = 1;
                auto param = HGraphTestIndex::GenerateHGraphBuildParametersString(build_param);

                SECTION("Build with use_reverse_edges enabled") {
                    auto param_with_reverse = param;
                    uint64_t pos =
                        static_cast<uint64_t>(param_with_reverse.find("\"index_param\": {{"));
                    if (pos != static_cast<uint64_t>(std::string::npos)) {
                        param_with_reverse.insert(static_cast<size_t>(pos) + 17,
                                                  "\"use_reverse_edges\": true, ");
                    }

                    auto index = TestIndex::TestFactory(test_index->name, param_with_reverse, true);
                    auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);

                    TestIndex::TestBuildIndex(index, dataset, true);
                    TestIndex::TestKnnSearch(index, dataset, search_param, recall, true);
                }

                SECTION("Serialize and Deserialize with reverse edges") {
                    auto param_with_reverse = param;
                    uint64_t pos =
                        static_cast<uint64_t>(param_with_reverse.find("\"index_param\": {{"));
                    if (pos != static_cast<uint64_t>(std::string::npos)) {
                        param_with_reverse.insert(static_cast<size_t>(pos) + 17,
                                                  "\"use_reverse_edges\": true, ");
                    }

                    auto index = TestIndex::TestFactory(test_index->name, param_with_reverse, true);
                    auto dataset = HGraphTestIndex::pool.GetDatasetAndCreate(
                        dim, resource->base_count, metric_type);

                    TestIndex::TestBuildIndex(index, dataset, true);

                    fixtures::TempDir dir("hgraph_reverse_edge");
                    std::string path = dir.GenerateRandomFile();

                    std::ofstream out_file(path, std::ios::binary);
                    index->Serialize(out_file);
                    out_file.close();

                    std::ifstream in_file(path, std::ios::binary);
                    auto deserialized_index =
                        TestIndex::TestFactory(test_index->name, param_with_reverse, true);
                    deserialized_index->Deserialize(in_file);
                    in_file.close();

                    TestIndex::TestKnnSearch(
                        deserialized_index, dataset, search_param, recall, true);
                }
            }
        }
    }
}

HGRAPH_PR_DAILY_CASE("HGraph Reverse Edges", "[ft][build][hgraph]", TestHGraphReverseEdges)

namespace {

class ModuloFilter : public vsag::Filter {
public:
    ModuloFilter(int64_t modulus, int64_t residue, float valid_ratio)
        : modulus_(modulus), residue_(residue), valid_ratio_(valid_ratio) {
    }

    bool
    CheckValid(int64_t id) const override {
        return (id % modulus_) == residue_;
    }

    float
    ValidRatio() const override {
        return valid_ratio_;
    }

private:
    int64_t modulus_;
    int64_t residue_;
    float valid_ratio_;
};

}  // namespace

TEST_CASE("(PR) HGraph brute_force_threshold", "[ft][hgraph][pr][brute_force_threshold]") {
    constexpr int64_t dim = 16;
    constexpr int64_t base_count = 1000;
    constexpr int64_t modulus = 50;
    constexpr int64_t residue = 7;
    constexpr int64_t topk = 5;

    std::string hgraph_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 16,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 100,
            "use_reorder": false
        }
    })";
    auto factory_res = vsag::Factory::CreateIndex("hgraph", hgraph_params);
    REQUIRE(factory_res.has_value());
    auto index = std::move(factory_res.value());

    std::mt19937 rng(20260528);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> base_vectors(base_count * dim);
    std::vector<int64_t> ids(base_count);
    for (int64_t i = 0; i < base_count; ++i) {
        ids[i] = i;
        for (int64_t j = 0; j < dim; ++j) {
            base_vectors[i * dim + j] = dist(rng);
        }
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(base_count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);
    auto build_res = index->Build(base);
    REQUIRE(build_res.has_value());

    std::vector<float> query_vec(dim);
    for (int64_t j = 0; j < dim; ++j) {
        query_vec[j] = dist(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vec.data())->Owner(false);

    auto filter =
        std::make_shared<ModuloFilter>(modulus, residue, 1.0F / static_cast<float>(modulus));

    auto search_param_graph = R"({"hgraph": {"ef_search": 200}})";
    auto search_param_brute = R"({"hgraph": {"ef_search": 200, "brute_force_threshold": 0.5}})";

    auto res_graph = index->KnnSearch(query, topk, search_param_graph, filter);
    auto res_brute = index->KnnSearch(query, topk, search_param_brute, filter);

    REQUIRE(res_graph.has_value());
    REQUIRE(res_brute.has_value());
    REQUIRE(res_brute.value()->GetDim() == topk);

    // Independent reference: scan all base ids that pass the filter, compute L2.
    std::vector<std::pair<float, int64_t>> reference;
    for (int64_t i = 0; i < base_count; ++i) {
        if ((i % modulus) != residue) {
            continue;
        }
        float d = 0.0F;
        for (int64_t j = 0; j < dim; ++j) {
            float diff = base_vectors[i * dim + j] - query_vec[j];
            d += diff * diff;
        }
        reference.emplace_back(d, ids[i]);
    }
    std::sort(reference.begin(), reference.end());
    REQUIRE(static_cast<int64_t>(reference.size()) >= topk);

    const auto* brute_ids = res_brute.value()->GetIds();
    const auto* brute_dists = res_brute.value()->GetDistances();
    for (int64_t k = 0; k < topk; ++k) {
        REQUIRE(brute_ids[k] == reference[k].second);
        REQUIRE(std::abs(brute_dists[k] - reference[k].first) < 1e-5F);
    }
}

TEST_CASE("(PR) HGraph brute_force_threshold default is no-op",
          "[ft][hgraph][pr][brute_force_threshold]") {
    constexpr int64_t dim = 8;
    constexpr int64_t base_count = 200;
    constexpr int64_t topk = 3;

    std::string hgraph_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 8,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 64,
            "use_reorder": false
        }
    })";
    auto index = vsag::Factory::CreateIndex("hgraph", hgraph_params).value();

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> base_vectors(base_count * dim);
    std::vector<int64_t> ids(base_count);
    for (int64_t i = 0; i < base_count; ++i) {
        ids[i] = i;
        for (int64_t j = 0; j < dim; ++j) {
            base_vectors[i * dim + j] = dist(rng);
        }
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(base_count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> query_vec(dim);
    for (int64_t j = 0; j < dim; ++j) {
        query_vec[j] = dist(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vec.data())->Owner(false);

    auto baseline = index->KnnSearch(query, topk, R"({"hgraph": {"ef_search": 64}})");
    auto with_zero = index->KnnSearch(
        query, topk, R"({"hgraph": {"ef_search": 64, "brute_force_threshold": 0.0}})");
    REQUIRE(baseline.has_value());
    REQUIRE(with_zero.has_value());
    REQUIRE(baseline.value()->GetDim() == with_zero.value()->GetDim());
    for (int64_t k = 0; k < baseline.value()->GetDim(); ++k) {
        REQUIRE(baseline.value()->GetIds()[k] == with_zero.value()->GetIds()[k]);
        REQUIRE(std::abs(baseline.value()->GetDistances()[k] -
                         with_zero.value()->GetDistances()[k]) < 1e-6F);
    }
}

TEST_CASE("HGraph ExportCache + ImportCache + Build acceleration smoke test",
          "[ft][hgraph][cache][pr]") {
    // End-to-end smoke test for the cache-accelerated Build path:
    //   (1) Build a baseline HGraph with N points carrying source_id.
    //   (2) ExportCache to an in-memory stream.
    //   (3) Create a fresh HGraph, ImportCache, then Build the same dataset.
    //       Build() should automatically take the warm-start + two-phase
    //       refine path because cache_ has been populated.
    //   (4) Verify the warmed index returns reasonable knn results on a few
    //       random queries (we don't compare absolute recall against the
    //       baseline, only that the index is non-empty, searchable, and
    //       returns the inserted ids).
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 200;
    constexpr int64_t TOPK = 10;

    const auto* param = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 50
        }
    }
    )";

    // Prepare deterministic data and source_ids.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> vectors(TEST_DIM * TEST_COUNT);
    for (auto& v : vectors) {
        v = dist(rng);
    }
    std::vector<int64_t> ids(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        ids[i] = i + 1;
    }
    std::vector<std::string> source_ids(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        source_ids[i] = fmt::format("sid_{}", i);
    }

    auto make_dataset = [&]() {
        auto base = vsag::Dataset::Make();
        base->NumElements(TEST_COUNT)
            ->Dim(TEST_DIM)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->SourceID(source_ids.data())
            ->Owner(false);
        return base;
    };

    // ---- (1) baseline build ----
    auto baseline = vsag::Factory::CreateIndex("hgraph", param).value();
    auto baseline_build = baseline->Build(make_dataset());
    REQUIRE(baseline_build.has_value());
    REQUIRE(baseline->GetNumElements() == TEST_COUNT);

    // ---- (2) export cache ----
    std::stringstream cache_buf;
    auto export_result = baseline->ExportCache(cache_buf);
    REQUIRE(export_result.has_value());
    REQUIRE(cache_buf.tellp() > 0);

    // ---- (3) fresh index, import cache, build again ----
    auto warmed = vsag::Factory::CreateIndex("hgraph", param).value();
    auto import_result = warmed->ImportCache(cache_buf);
    REQUIRE(import_result.has_value());
    auto* logger_ptr = vsag::Options::Instance().logger();
    if (logger_ptr != nullptr) {
        logger_ptr->SetLevel(vsag::Logger::Level::kINFO);
    }
    auto warmed_build = warmed->Build(make_dataset());
    if (logger_ptr != nullptr) {
        logger_ptr->SetLevel(vsag::Logger::Level::kWARN);
    }
    REQUIRE(warmed_build.has_value());
    REQUIRE(warmed->GetNumElements() == TEST_COUNT);

    // ---- (4) sanity-check knn search on the warmed index ----
    std::vector<float> query_vec(TEST_DIM);
    std::copy(vectors.begin(), vectors.begin() + TEST_DIM, query_vec.begin());
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(TEST_DIM)->Float32Vectors(query_vec.data())->Owner(false);

    const auto* search_param = R"({"hgraph": {"ef_search": 50}})";
    auto search_result = warmed->KnnSearch(query, TOPK, search_param);
    REQUIRE(search_result.has_value());
    auto knn = search_result.value();
    REQUIRE(knn->GetNumElements() == 1);
    REQUIRE(knn->GetDim() > 0);
    REQUIRE(knn->GetDim() <= TOPK);
    // The first inserted vector (id=1) is identical to the query: it must be
    // returned and its distance must be (approximately) zero.
    bool found_self = false;
    for (int64_t i = 0; i < knn->GetDim(); ++i) {
        if (knn->GetIds()[i] == ids[0]) {
            found_self = true;
            REQUIRE(knn->GetDistances()[i] < 1e-4F);
            break;
        }
    }
    REQUIRE(found_self);
}

TEST_CASE("HGraph ExportCache + ImportCache + Build miss-only path", "[ft][hgraph][cache][pr]") {
    // Force every node to take the *missed* branch of build_with_cache:
    //   (1) Build baseline with source_id="sid_A" and ExportCache.
    //   (2) Fresh index, ImportCache (so cache_->neighbors_ has key "sid_A"),
    //       then Build with a DIFFERENT source_id "sid_B". Every inserted
    //       node will fail the warm_start lookup and go through the
    //       missed-refine loop. Verifies the missed path does not crash and
    //       still produces a searchable index.
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 200;
    constexpr int64_t TOPK = 10;

    const auto* param = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 50
        }
    }
    )";

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> vectors(TEST_DIM * TEST_COUNT);
    for (auto& v : vectors) {
        v = dist(rng);
    }
    std::vector<int64_t> ids(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        ids[i] = i + 1;
    }
    const std::string source_id_a = "sid_A";
    const std::string source_id_b = "sid_B";
    // Each vector must carry a unique source_id (semantics of source_id are
    // per-vector identifiers, not group tags). Use disjoint prefixes so the
    // second Build cannot match any source_id in the imported cache.
    std::vector<std::string> source_ids_a(TEST_COUNT);
    std::vector<std::string> source_ids_b(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        source_ids_a[i] = fmt::format("A_{}", i);
        source_ids_b[i] = fmt::format("B_{}", i);
    }

    auto make_dataset = [&](const std::vector<std::string>& sids) {
        auto base = vsag::Dataset::Make();
        base->NumElements(TEST_COUNT)
            ->Dim(TEST_DIM)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->SourceID(sids.data())
            ->Owner(false);
        return base;
    };

    // ---- (1) baseline build with source_id "sid_A" ----
    auto baseline = vsag::Factory::CreateIndex("hgraph", param).value();
    auto baseline_build = baseline->Build(make_dataset(source_ids_a));
    REQUIRE(baseline_build.has_value());
    REQUIRE(baseline->GetNumElements() == TEST_COUNT);

    // ---- (2) export cache (contains only "sid_A") ----
    std::stringstream cache_buf;
    auto export_result = baseline->ExportCache(cache_buf);
    REQUIRE(export_result.has_value());
    REQUIRE(cache_buf.tellp() > 0);

    // ---- (3) fresh index, import cache, build with DIFFERENT source_id ----
    auto warmed = vsag::Factory::CreateIndex("hgraph", param).value();
    auto import_result = warmed->ImportCache(cache_buf);
    REQUIRE(import_result.has_value());
    auto* logger_ptr = vsag::Options::Instance().logger();
    if (logger_ptr != nullptr) {
        logger_ptr->SetLevel(vsag::Logger::Level::kINFO);
    }
    // sid_B is not present in cache_->neighbors_ => 100% missed nodes.
    auto warmed_build = warmed->Build(make_dataset(source_ids_b));
    if (logger_ptr != nullptr) {
        logger_ptr->SetLevel(vsag::Logger::Level::kWARN);
    }
    REQUIRE(warmed_build.has_value());
    REQUIRE(warmed->GetNumElements() == TEST_COUNT);

    // ---- (4) sanity-check knn search on the warmed index ----
    std::vector<float> query_vec(TEST_DIM);
    std::copy(vectors.begin(), vectors.begin() + TEST_DIM, query_vec.begin());
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(TEST_DIM)->Float32Vectors(query_vec.data())->Owner(false);

    const auto* search_param = R"({"hgraph": {"ef_search": 50}})";
    auto search_result = warmed->KnnSearch(query, TOPK, search_param);
    REQUIRE(search_result.has_value());
    auto knn = search_result.value();
    REQUIRE(knn->GetNumElements() == 1);
    REQUIRE(knn->GetDim() > 0);
    REQUIRE(knn->GetDim() <= TOPK);
    // The first inserted vector (id=1) is identical to the query: it must be
    // returned and its distance must be (approximately) zero. This exercises
    // the all-missed code path end-to-end.
    bool found_self = false;
    for (int64_t i = 0; i < knn->GetDim(); ++i) {
        if (knn->GetIds()[i] == ids[0]) {
            found_self = true;
            REQUIRE(knn->GetDistances()[i] < 1e-4F);
            break;
        }
    }
    REQUIRE(found_self);
}

TEST_CASE("HGraph GetStats reports build cache hit-rate", "[ft][hgraph][cache][pr]") {
    // Verify that the build-time warm-start hit-rate computed during
    // build_with_cache() is surfaced through GetStats():
    //   (1) A normal Build() (no imported cache) reports a skipped_reason.
    //   (2) A Build() after ImportCache() reports a numeric hit-rate plus the
    //       hit / missed node counts, and the counts sum to the index size.
    constexpr int64_t TEST_DIM = 32;
    constexpr int64_t TEST_COUNT = 200;

    const auto* param = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 32,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 50
        }
    }
    )";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> vectors(TEST_DIM * TEST_COUNT);
    for (auto& v : vectors) {
        v = dist(rng);
    }
    std::vector<int64_t> ids(TEST_COUNT);
    std::vector<std::string> source_ids(TEST_COUNT);
    for (int64_t i = 0; i < TEST_COUNT; ++i) {
        ids[i] = i + 1;
        source_ids[i] = fmt::format("sid_{}", i);
    }

    auto make_dataset = [&]() {
        auto base = vsag::Dataset::Make();
        base->NumElements(TEST_COUNT)
            ->Dim(TEST_DIM)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->SourceID(source_ids.data())
            ->Owner(false);
        return base;
    };

    // ---- (1) normal build: hit-rate is skipped ----
    auto baseline = vsag::Factory::CreateIndex("hgraph", param).value();
    REQUIRE(baseline->Build(make_dataset()).has_value());
    auto baseline_stats = baseline->GetStats();
    INFO(baseline_stats);
    auto baseline_parsed = vsag::JsonType::Parse(baseline_stats);
    REQUIRE(baseline_parsed.Contains("build_cache_hit_rate"));
    // Without an imported cache the rate is not a number but a skipped_reason
    // object nested under the build_cache_hit_rate key.
    REQUIRE(baseline_parsed["build_cache_hit_rate"].Contains("skipped_reason"));
    REQUIRE(baseline_parsed["build_cache_hit_rate"]["skipped_reason"].GetString() ==
            std::string("index was not built from an imported cache"));
    REQUIRE_FALSE(baseline_parsed.Contains("build_cache_hit_nodes"));
    REQUIRE_FALSE(baseline_parsed.Contains("build_cache_missed_nodes"));

    // ---- (2) cache-accelerated build: hit-rate is reported ----
    std::stringstream cache_buf;
    REQUIRE(baseline->ExportCache(cache_buf).has_value());

    auto warmed = vsag::Factory::CreateIndex("hgraph", param).value();
    REQUIRE(warmed->ImportCache(cache_buf).has_value());
    REQUIRE(warmed->Build(make_dataset()).has_value());
    REQUIRE(warmed->GetNumElements() == TEST_COUNT);

    auto warmed_stats = warmed->GetStats();
    INFO(warmed_stats);
    auto parsed = vsag::JsonType::Parse(warmed_stats);
    REQUIRE(parsed.Contains("build_cache_hit_rate"));
    REQUIRE(parsed.Contains("build_cache_hit_nodes"));
    REQUIRE(parsed.Contains("build_cache_missed_nodes"));
    const float hit_rate = parsed["build_cache_hit_rate"].GetFloat();
    REQUIRE(hit_rate >= 0.0F);
    REQUIRE(hit_rate <= 1.0F);
    const auto hit_nodes = parsed["build_cache_hit_nodes"].GetInt();
    const auto missed_nodes = parsed["build_cache_missed_nodes"].GetInt();
    REQUIRE(hit_nodes + missed_nodes == TEST_COUNT);
}
TEST_CASE("HGraph Concurrent Tune and CalDistanceById", "[ft][concurrent][hgraph]") {
    constexpr uint32_t dim = 64;
    constexpr uint32_t num_vectors = 1000;

    std::string build_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 200,
            "build_thread_count": 0,
            "store_raw_vector": true
        }
    })";

    std::string tune_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 32,
            "ef_construction": 100,
            "build_thread_count": 0
        }
    })";

    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(num_vectors * dim);
    std::vector<int64_t> ids(num_vectors);
    for (uint32_t i = 0; i < num_vectors; ++i) {
        ids[i] = static_cast<int64_t>(i);
        for (uint32_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = dist(rng);
        }
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto index = vsag::Factory::CreateIndex("hgraph", build_params);
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(base).has_value());

    std::vector<float> query(vectors.begin(), vectors.begin() + dim);
    std::vector<int64_t> batch_ids = {0, 1, 2};

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> cal_count{0};

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                index.value()->CalDistanceById(query.data(), batch_ids.data(), 3);
                cal_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto tune_result = index.value()->Tune(tune_params, true);
    CHECK(tune_result.has_value());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    stop.store(true, std::memory_order_relaxed);
    for (auto& t : readers) {
        t.join();
    }

    REQUIRE(cal_count.load() > 0);
}

TEST_CASE("HGraph Concurrent Tune and CalcDistanceById (single id)", "[ft][concurrent][hgraph]") {
    constexpr uint32_t dim = 64;
    constexpr uint32_t num_vectors = 1000;

    std::string build_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 200,
            "build_thread_count": 0,
            "store_raw_vector": true
        }
    })";

    std::string tune_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 32,
            "ef_construction": 100,
            "build_thread_count": 0
        }
    })";

    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(num_vectors * dim);
    std::vector<int64_t> ids(num_vectors);
    for (uint32_t i = 0; i < num_vectors; ++i) {
        ids[i] = static_cast<int64_t>(i);
        for (uint32_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = dist(rng);
        }
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto index = vsag::Factory::CreateIndex("hgraph", build_params);
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(base).has_value());

    std::vector<float> query(vectors.begin(), vectors.begin() + dim);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> cal_count{0};

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                index.value()->CalcDistanceById(query.data(), 0);
                cal_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto tune_result = index.value()->Tune(tune_params, true);
    CHECK(tune_result.has_value());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    stop.store(true, std::memory_order_relaxed);
    for (auto& t : readers) {
        t.join();
    }

    REQUIRE(cal_count.load() > 0);
}

TEST_CASE("HGraph Concurrent Tune(disable_future_tuning=false) and CalDistanceById",
          "[ft][concurrent][hgraph]") {
    constexpr uint32_t dim = 64;
    constexpr uint32_t num_vectors = 1000;

    std::string build_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 200,
            "build_thread_count": 0,
            "store_raw_vector": true
        }
    })";

    std::string tune_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 32,
            "ef_construction": 100,
            "build_thread_count": 0,
            "store_raw_vector": true
        }
    })";

    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vectors(num_vectors * dim);
    std::vector<int64_t> ids(num_vectors);
    for (uint32_t i = 0; i < num_vectors; ++i) {
        ids[i] = static_cast<int64_t>(i);
        for (uint32_t j = 0; j < dim; ++j) {
            vectors[i * dim + j] = dist(rng);
        }
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto index = vsag::Factory::CreateIndex("hgraph", build_params);
    REQUIRE(index.has_value());
    REQUIRE(index.value()->Build(base).has_value());

    std::vector<float> query(vectors.begin(), vectors.begin() + dim);
    std::vector<int64_t> batch_ids = {0, 1, 2};

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> cal_count{0};

    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                index.value()->CalDistanceById(query.data(), batch_ids.data(), 3);
                cal_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto tune_result = index.value()->Tune(tune_params, false);
    CHECK(tune_result.has_value());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    stop.store(true, std::memory_order_relaxed);
    for (auto& t : readers) {
        t.join();
    }

    REQUIRE(cal_count.load() > 0);
}
