
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

#include "pyramid.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"

namespace {

constexpr int64_t PYRAMID_TEST_DIM = 4;

struct PyramidTestIndex {
    std::shared_ptr<vsag::Allocator> allocator;
    std::shared_ptr<vsag::Pyramid> index;
};

PyramidTestIndex
MakePyramidIndex(uint32_t index_min_size,
                 uint64_t build_thread_count = 1,
                 bool support_duplicate = false) {
    PyramidTestIndex result;
    vsag::IndexCommonParam common_param;
    common_param.dim_ = PYRAMID_TEST_DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    result.allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    common_param.allocator_ = result.allocator;

    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "base_io_type": "memory_io",
        "max_degree": 8,
        "ef_construction": 8,
        "alpha": 1.2,
        "graph_type": "nsw",
        "no_build_levels": [0],
        "index_min_size": 3
    })");
    external_param[vsag::PYRAMID_INDEX_MIN_SIZE].SetInt(index_min_size);
    external_param[vsag::PYRAMID_BUILD_THREAD_COUNT].SetUint64(build_thread_count);
    external_param[vsag::PYRAMID_SUPPORT_DUPLICATE].SetBool(support_duplicate);
    auto param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    result.index = std::make_shared<vsag::Pyramid>(param, common_param);
    return result;
}

vsag::DatasetPtr
MakePyramidDataset(float* vectors, int64_t* ids, std::string* paths, int64_t count) {
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(PYRAMID_TEST_DIM)
        ->Ids(ids)
        ->Float32Vectors(vectors)
        ->Paths(paths)
        ->Owner(false);
}

int64_t
GetPyramidSubindexCount(const std::shared_ptr<vsag::Pyramid>& index, const char* status) {
    auto stats = vsag::JsonType::Parse(index->GetStats());
    return stats["subindex_quality"][status].GetInt();
}

int64_t
GetPyramidNodeStatusCount(const std::shared_ptr<vsag::Pyramid>& index, const char* status) {
    auto stats = vsag::JsonType::Parse(index->GetStats());
    return stats["index_node_structure"]["status_distribution"][status].GetInt();
}

int64_t
GetPyramidTotalNodes(const std::shared_ptr<vsag::Pyramid>& index) {
    auto stats = vsag::JsonType::Parse(index->GetStats());
    return stats["index_node_structure"]["total_nodes"].GetInt();
}

void
FillPyramidTestVectors(std::vector<float>& vectors, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        auto* vector = vectors.data() + i * PYRAMID_TEST_DIM;
        vector[0] = static_cast<float>(i);
        vector[1] = static_cast<float>(i % 17);
        vector[2] = static_cast<float>(i % 31);
        vector[3] = static_cast<float>(i % 47);
    }
}

void
RequirePyramidSelfMatch(const std::shared_ptr<vsag::Pyramid>& index,
                        std::vector<float>& vectors,
                        const std::vector<int64_t>& ids,
                        std::string* query_path,
                        int64_t row,
                        const std::string& search_params =
                            R"({"pyramid":{"ef_search":1000,"subindex_ef_search":1000}})") {
    auto query =
        MakePyramidDataset(vectors.data() + row * PYRAMID_TEST_DIM, nullptr, query_path, 1);
    auto result = index->KnnSearch(query, 1, search_params, vsag::FilterPtr{});
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == ids[row]);
    REQUIRE(result->GetDistances()[0] == 0.0F);
}

}  // namespace

TEST_CASE("Split function tests", "[ut][pyramid]") {
    SECTION("Empty input string") {
        auto result = vsag::split("", ',');
        REQUIRE(result.empty());
    }

    SECTION("No delimiters in string") {
        auto result = vsag::split("hello", ',');
        REQUIRE(result == std::vector<std::string>{"hello"});
    }

    SECTION("Delimiter at start") {
        auto result = vsag::split(",hello,world", ',');
        REQUIRE(result == std::vector<std::string>{"hello", "world"});
    }

    SECTION("Delimiter at end") {
        auto result = vsag::split("hello,world,", ',');
        REQUIRE(result == std::vector<std::string>{"hello", "world"});
    }

    SECTION("Multiple consecutive delimiters") {
        auto result = vsag::split("a,,b,,,c", ',');
        REQUIRE(result == std::vector<std::string>{"a", "b", "c"});
    }

    SECTION("Normal split with multiple tokens") {
        auto result = vsag::split("one,two,three", ',');
        REQUIRE(result == std::vector<std::string>{"one", "two", "three"});
    }

    SECTION("All delimiters") {
        auto result = vsag::split(",,,", ',');
        REQUIRE(result.empty());
    }

    SECTION("Mixed delimiters and spaces") {
        auto result = vsag::split("  , hello,  world  ", ',');
        REQUIRE(result == std::vector<std::string>{"  ", " hello", "  world  "});
    }
}

TEST_CASE("Pyramid promotes flat node at index minimum size", "[ut][pyramid]") {
    auto test_index = MakePyramidIndex(3);
    const auto& index = test_index.index;
    std::vector<float> vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
        2.0F,
        2.0F,
        2.0F,
        2.0F,
    };
    std::vector<int64_t> ids = {100, 101, 102};
    std::vector<std::string> paths(3, "tenant");

    REQUIRE(index->Add(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 2)).empty());
    REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 0);

    REQUIRE(index
                ->Add(MakePyramidDataset(
                    vectors.data() + 2 * PYRAMID_TEST_DIM, ids.data() + 2, paths.data() + 2, 1))
                .empty());
    REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == 0);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == 3);

    for (int64_t i = 0; i < 3; ++i) {
        auto query =
            MakePyramidDataset(vectors.data() + i * PYRAMID_TEST_DIM, nullptr, paths.data() + i, 1);
        auto result =
            index->KnnSearch(query, 1, R"({"pyramid":{"ef_search":10}})", vsag::FilterPtr{});
        REQUIRE(result->GetIds()[0] == ids[i]);
    }
}

TEST_CASE("Pyramid NSW Build handles path-heavy data with few graph nodes",
          "[ut][pyramid][build]") {
    constexpr int64_t flat_leaf_count = 363;
    constexpr int64_t flat_leaf_size = 3;
    constexpr int64_t graph_leaf_count = 2;
    constexpr int64_t graph_leaf_size = 4;
    constexpr int64_t count = flat_leaf_count * flat_leaf_size + graph_leaf_count * graph_leaf_size;
    const uint64_t build_thread_count = GENERATE(1, 4);
    auto test_index = MakePyramidIndex(graph_leaf_size, build_thread_count);
    const auto& index = test_index.index;

    std::vector<float> vectors((count + 1) * PYRAMID_TEST_DIM);
    FillPyramidTestVectors(vectors, count + 1);
    std::vector<int64_t> ids(count + 1);
    std::iota(ids.begin(), ids.end(), 1000);
    std::vector<std::string> paths;
    paths.reserve(count + 1);
    for (int64_t leaf = 0; leaf < flat_leaf_count; ++leaf) {
        for (int64_t i = 0; i < flat_leaf_size; ++i) {
            paths.emplace_back("flat-" + std::to_string(leaf));
        }
    }
    const int64_t first_graph_row = static_cast<int64_t>(paths.size());
    for (int64_t leaf = 0; leaf < graph_leaf_count; ++leaf) {
        for (int64_t i = 0; i < graph_leaf_size; ++i) {
            paths.emplace_back("graph-" + std::to_string(leaf));
        }
    }
    paths.emplace_back("flat-0");

    REQUIRE(
        index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count)).empty());
    REQUIRE(GetPyramidNodeStatusCount(index, "NO_INDEX") == 1);
    REQUIRE(GetPyramidNodeStatusCount(index, "FLAT") == flat_leaf_count);
    REQUIRE(GetPyramidNodeStatusCount(index, "GRAPH") == graph_leaf_count);
    REQUIRE(GetPyramidTotalNodes(index) == 1 + flat_leaf_count + graph_leaf_count);
    REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == flat_leaf_count);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 2);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") ==
            graph_leaf_count * graph_leaf_size);
    RequirePyramidSelfMatch(index, vectors, ids, paths.data(), 0);
    RequirePyramidSelfMatch(index, vectors, ids, paths.data() + first_graph_row, first_graph_row);

    REQUIRE(index
                ->Add(MakePyramidDataset(vectors.data() + count * PYRAMID_TEST_DIM,
                                         ids.data() + count,
                                         paths.data() + count,
                                         1))
                .empty());
    REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == flat_leaf_count - 1);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 3);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") ==
            graph_leaf_count * graph_leaf_size + graph_leaf_size);
    RequirePyramidSelfMatch(index, vectors, ids, paths.data(), 0);
    RequirePyramidSelfMatch(index, vectors, ids, paths.data() + count, count);
}

TEST_CASE("Pyramid NSW Build preserves duplicate handling", "[ut][pyramid][build]") {
    constexpr int64_t count = 16;
    auto test_index = MakePyramidIndex(3, 4, true);
    const auto& index = test_index.index;

    std::vector<float> vectors(count * PYRAMID_TEST_DIM, 1.0F);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 4000);
    std::vector<std::string> paths(count, "duplicates");

    REQUIRE(
        index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count)).empty());
    REQUIRE(index->GetNumElements() == count);
    REQUIRE(GetPyramidNodeStatusCount(index, "GRAPH") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == 1);

    auto query = MakePyramidDataset(vectors.data(), nullptr, paths.data(), 1);
    auto result =
        index->KnnSearch(query, count, R"({"pyramid":{"ef_search":32}})", vsag::FilterPtr{});
    REQUIRE(result->GetDim() == 1);
    REQUIRE(std::find(ids.begin(), ids.end(), result->GetIds()[0]) != ids.end());
    REQUIRE(result->GetDistances()[0] == 0.0F);
}

TEST_CASE("Pyramid NSW Build keeps compact inner ids after rejected labels",
          "[ut][pyramid][build]") {
    auto test_index = MakePyramidIndex(3);
    const auto& index = test_index.index;
    std::vector<float> vectors(4 * PYRAMID_TEST_DIM);
    for (int64_t i = 0; i < 4; ++i) {
        std::fill_n(
            vectors.data() + i * PYRAMID_TEST_DIM, PYRAMID_TEST_DIM, static_cast<float>(i * 10));
    }
    std::vector<int64_t> ids = {7, 7, 8, 9};
    std::vector<std::string> paths = {"kept", "rejected", "kept", "kept"};

    auto failed = index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 4));
    REQUIRE(failed == std::vector<int64_t>{7});
    REQUIRE(index->GetNumElements() == 3);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == 3);

    auto first_query = MakePyramidDataset(vectors.data(), nullptr, paths.data(), 1);
    auto first_result =
        index->KnnSearch(first_query, 1, R"({"pyramid":{"ef_search":10}})", vsag::FilterPtr{});
    REQUIRE(first_result->GetDim() == 1);
    REQUIRE(first_result->GetIds()[0] == 7);

    auto rejected_query =
        MakePyramidDataset(vectors.data() + PYRAMID_TEST_DIM, nullptr, paths.data() + 1, 1);
    auto rejected_result =
        index->KnnSearch(rejected_query, 1, R"({"pyramid":{"ef_search":10}})", vsag::FilterPtr{});
    REQUIRE(rejected_result->GetDim() == 0);

    auto third_query =
        MakePyramidDataset(vectors.data() + 2 * PYRAMID_TEST_DIM, nullptr, paths.data() + 2, 1);
    auto third_result =
        index->KnnSearch(third_query, 1, R"({"pyramid":{"ef_search":10}})", vsag::FilterPtr{});
    REQUIRE(third_result->GetDim() == 1);
    REQUIRE(third_result->GetIds()[0] == 8);

    auto fourth_query =
        MakePyramidDataset(vectors.data() + 3 * PYRAMID_TEST_DIM, nullptr, paths.data() + 3, 1);
    auto fourth_result =
        index->KnnSearch(fourth_query, 1, R"({"pyramid":{"ef_search":10}})", vsag::FilterPtr{});
    REQUIRE(fourth_result->GetDim() == 1);
    REQUIRE(fourth_result->GetIds()[0] == 9);
}

TEST_CASE("Pyramid Build rejects an index containing only removed vectors",
          "[ut][pyramid][build]") {
    auto test_index = MakePyramidIndex(3);
    const auto& index = test_index.index;
    std::vector<float> vectors(PYRAMID_TEST_DIM, 1.0F);
    std::vector<int64_t> ids = {42};
    std::vector<std::string> paths = {"leaf"};
    auto dataset = MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 1);

    REQUIRE(index->Build(dataset).empty());
    REQUIRE(index->Remove(ids, vsag::RemoveMode::MARK_REMOVE) == 1);
    REQUIRE(index->GetNumElements() == 0);
    REQUIRE_THROWS(index->Build(dataset));
}
