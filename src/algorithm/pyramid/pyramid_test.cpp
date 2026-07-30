
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
#include <array>
#include <future>
#include <numeric>
#include <queue>
#include <sstream>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index_common_param.h"
#include "unittest.h"
#include "vsag/index.h"

namespace vsag {

class PyramidTestAccess {
public:
    static IndexNode*
    GetRoot(const Pyramid& index) {
        return index.hierarchies_.at("").get()->root.get();
    }

    static InnerIdType
    GetRootEntryPoint(const Pyramid& index) {
        return GetRoot(index)->entry_point_;
    }

    static std::vector<std::array<uint64_t, 3>>
    PlanNswBuildChunks(const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                       uint64_t build_thread_count,
                       Allocator* allocator) {
        Vector<Pyramid::NswBuildRange> build_ranges(allocator);
        build_ranges.reserve(ranges.size());
        for (const auto& [begin, end] : ranges) {
            build_ranges.push_back({begin, end});
        }

        const auto chunks =
            Pyramid::plan_nsw_build_chunks(build_ranges, build_thread_count, allocator);
        std::vector<std::array<uint64_t, 3>> result;
        result.reserve(chunks.size());
        for (const auto& chunk : chunks) {
            result.push_back({chunk.job_index, chunk.begin, chunk.end});
        }
        return result;
    }
};

}  // namespace vsag

namespace {

constexpr int64_t PYRAMID_TEST_DIM = 4;

struct PyramidTestIndex {
    std::shared_ptr<vsag::Allocator> allocator;
    std::shared_ptr<vsag::Pyramid> index;
};

PyramidTestIndex
MakePyramidIndex(uint32_t index_min_size,
                 uint64_t build_thread_count = 1,
                 uint64_t thread_pool_size = 0,
                 bool build_root = false,
                 bool support_duplicate = false,
                 bool use_reverse_edges = false,
                 uint64_t max_degree = 8,
                 uint64_t ef_construction = 8) {
    PyramidTestIndex result;
    vsag::IndexCommonParam common_param;
    common_param.dim_ = PYRAMID_TEST_DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    result.allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    common_param.allocator_ = result.allocator;
    if (thread_pool_size > 0) {
        common_param.thread_pool_ = vsag::SafeThreadPool::FactoryDefaultThreadPool();
        common_param.thread_pool_->SetPoolSize(thread_pool_size);
    }

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
    external_param[vsag::PYRAMID_GRAPH_MAX_DEGREE].SetUint64(max_degree);
    external_param[vsag::PYRAMID_EF_CONSTRUCTION].SetUint64(ef_construction);
    if (build_root) {
        external_param[vsag::PYRAMID_NO_BUILD_LEVELS].SetVector(std::vector<int32_t>{});
    }
    auto param = std::dynamic_pointer_cast<vsag::PyramidParameters>(
        vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param));
    param->graph_param->use_reverse_edges_ = use_reverse_edges;
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
FillPermutedPyramidTestVectors(std::vector<float>& vectors, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        const auto value = static_cast<int64_t>((static_cast<uint64_t>(i) * 499) % count);
        auto* vector = vectors.data() + i * PYRAMID_TEST_DIM;
        vector[0] = static_cast<float>(value);
        vector[1] = static_cast<float>(value % 17);
        vector[2] = static_cast<float>(value % 31);
        vector[3] = static_cast<float>(value % 47);
    }
}

bool
PyramidSelfMatches(const std::shared_ptr<vsag::Pyramid>& index,
                   std::vector<float>& vectors,
                   const std::vector<int64_t>& ids,
                   std::string* query_path,
                   int64_t row,
                   const std::string& search_params) {
    auto query =
        MakePyramidDataset(vectors.data() + row * PYRAMID_TEST_DIM, nullptr, query_path, 1);
    auto result = index->KnnSearch(query, 1, search_params, vsag::FilterPtr{});
    return result->GetDim() == 1 && result->GetIds()[0] == ids[row] &&
           result->GetDistances()[0] == 0.0F;
}

void
RequirePyramidSelfMatch(const std::shared_ptr<vsag::Pyramid>& index,
                        std::vector<float>& vectors,
                        const std::vector<int64_t>& ids,
                        std::string* query_path,
                        int64_t row,
                        const std::string& search_params =
                            R"({"pyramid":{"ef_search":1000,"subindex_ef_search":1000}})") {
    REQUIRE(PyramidSelfMatches(index, vectors, ids, query_path, row, search_params));
}

uint64_t
GetReachableNodeCount(const std::shared_ptr<vsag::Pyramid>& index) {
    const auto* root = vsag::PyramidTestAccess::GetRoot(*index);
    std::queue<vsag::InnerIdType> pending;
    std::vector<bool> visited(index->GetNumElements(), false);
    pending.push(root->entry_point_);
    visited[root->entry_point_] = true;
    uint64_t count = 0;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::Vector<vsag::InnerIdType> neighbors(allocator.get());
    while (not pending.empty()) {
        const auto current = pending.front();
        pending.pop();
        ++count;
        root->graph_->GetNeighbors(current, neighbors);
        for (const auto neighbor : neighbors) {
            if (not visited[neighbor]) {
                visited[neighbor] = true;
                pending.push(neighbor);
            }
        }
    }
    return count;
}

bool
HasConsistentRootReverseEdges(const std::shared_ptr<vsag::Pyramid>& index, uint64_t count) {
    const auto* root = vsag::PyramidTestAccess::GetRoot(*index);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::Vector<vsag::InnerIdType> neighbors(allocator.get());
    for (uint64_t i = 0; i < count; ++i) {
        root->graph_->GetNeighbors(static_cast<vsag::InnerIdType>(i), neighbors);
        for (const auto neighbor : neighbors) {
            vsag::Vector<vsag::InnerIdType> incoming(allocator.get());
            root->graph_->GetIncomingNeighbors(neighbor, incoming);
            if (std::find(incoming.begin(), incoming.end(), i) == incoming.end()) {
                return false;
            }
        }
    }
    return true;
}

vsag::InnerIdType
GetExpectedReservoirEntryPoint(uint64_t count) {
    std::default_random_engine generator{2021};
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    vsag::InnerIdType entry_point = 0;
    for (uint64_t total_count = 1; total_count < count; ++total_count) {
        if (static_cast<double>(total_count) * distribution(generator) < 1.0) {
            entry_point = static_cast<vsag::InnerIdType>(total_count);
        }
    }
    return entry_point;
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

TEST_CASE("Pyramid NSW Build constructs only planned graph nodes", "[ut][pyramid][build]") {
    const uint64_t build_thread_count = GENERATE(0, 4);
    auto test_index = MakePyramidIndex(3, build_thread_count, 4);
    const auto& index = test_index.index;
    std::vector<float> vectors(9 * PYRAMID_TEST_DIM);
    for (int64_t i = 0; i < 9; ++i) {
        std::fill_n(vectors.data() + i * PYRAMID_TEST_DIM, PYRAMID_TEST_DIM, static_cast<float>(i));
    }
    std::vector<int64_t> ids = {100, 101, 102, 103, 104, 105, 106, 107, 108};
    std::vector<std::string> paths = {
        "flat",
        "flat",
        "graph-a/child",
        "graph-a/child",
        "graph-a/child",
        "graph-b",
        "graph-b",
        "graph-b",
        "flat",
    };

    REQUIRE(index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), 8)).empty());
    REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 2);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == 6);
    REQUIRE(GetPyramidNodeStatusCount(index, "GRAPH") == 3);

    for (int64_t i = 0; i < 8; ++i) {
        auto query =
            MakePyramidDataset(vectors.data() + i * PYRAMID_TEST_DIM, nullptr, paths.data() + i, 1);
        auto result =
            index->KnnSearch(query, 1, R"({"pyramid":{"ef_search":10}})", vsag::FilterPtr{});
        REQUIRE(result->GetIds()[0] == ids[i]);
    }

    REQUIRE(index
                ->Add(MakePyramidDataset(
                    vectors.data() + 8 * PYRAMID_TEST_DIM, ids.data() + 8, paths.data() + 8, 1))
                .empty());
    REQUIRE(GetPyramidSubindexCount(index, "flat_subindexes") == 0);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 3);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == 9);
    REQUIRE(GetPyramidNodeStatusCount(index, "GRAPH") == 4);
}

TEST_CASE("Pyramid NSW Build parallelizes one large root graph", "[ut][pyramid][build]") {
    constexpr int64_t count = 1000;
    const uint64_t build_thread_count = GENERATE(1, 4);
    INFO("build_thread_count=" << build_thread_count);
    auto test_index = MakePyramidIndex(3, build_thread_count, 4, true);
    const auto& index = test_index.index;

    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillPyramidTestVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 1000);
    std::vector<std::string> paths(count, "");

    REQUIRE(
        index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count)).empty());
    REQUIRE(GetPyramidNodeStatusCount(index, "GRAPH") == 1);
    REQUIRE(GetPyramidTotalNodes(index) == 1);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == count);
    REQUIRE(GetReachableNodeCount(index) == count);
    REQUIRE(vsag::PyramidTestAccess::GetRootEntryPoint(*index) ==
            GetExpectedReservoirEntryPoint(count));

    for (const int64_t row : {int64_t{0}, count / 2, count - 1}) {
        RequirePyramidSelfMatch(index, vectors, ids, nullptr, row);
    }

    if (build_thread_count > 1) {
        vectors.resize((count + 1) * PYRAMID_TEST_DIM);
        FillPyramidTestVectors(vectors, count + 1);
        ids.push_back(9000);
        paths.emplace_back("");
        REQUIRE(index
                    ->Add(MakePyramidDataset(vectors.data() + count * PYRAMID_TEST_DIM,
                                             ids.data() + count,
                                             paths.data() + count,
                                             1))
                    .empty());
        RequirePyramidSelfMatch(index, vectors, ids, nullptr, count);

        auto binary_set = index->vsag::InnerIndexInterface::Serialize();
        auto loaded_test_index = MakePyramidIndex(3, build_thread_count, 4, true);
        loaded_test_index.index->vsag::InnerIndexInterface::Deserialize(binary_set);
        RequirePyramidSelfMatch(loaded_test_index.index, vectors, ids, nullptr, count);
    }
}

TEST_CASE("Pyramid NSW Build planner splits skewed graph nodes", "[ut][pyramid][build]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    std::vector<std::pair<uint64_t, uint64_t>> ranges{{64, 10064}};
    for (uint64_t i = 0; i < 9; ++i) {
        ranges.emplace_back(0, 1);
    }

    const uint64_t build_thread_count = 4;
    const auto chunks =
        vsag::PyramidTestAccess::PlanNswBuildChunks(ranges, build_thread_count, allocator.get());
    REQUIRE(chunks.size() == 25);
    REQUIRE(chunks.size() <= ranges.size() + 4 * build_thread_count);

    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> chunks_by_job(ranges.size());
    for (const auto& chunk : chunks) {
        REQUIRE(chunk[0] < ranges.size());
        REQUIRE(chunk[1] < chunk[2]);
        chunks_by_job[chunk[0]].emplace_back(chunk[1], chunk[2]);
    }
    REQUIRE(chunks_by_job[0].size() == 16);
    for (uint64_t job_index = 0; job_index < ranges.size(); ++job_index) {
        auto& job_chunks = chunks_by_job[job_index];
        std::sort(job_chunks.begin(), job_chunks.end());
        REQUIRE(job_chunks.front().first == ranges[job_index].first);
        REQUIRE(job_chunks.back().second == ranges[job_index].second);
        for (uint64_t i = 1; i < job_chunks.size(); ++i) {
            REQUIRE(job_chunks[i - 1].second == job_chunks[i].first);
        }
    }

    REQUIRE(vsag::PyramidTestAccess::PlanNswBuildChunks({}, 4, allocator.get()).empty());
    const auto small_chunks =
        vsag::PyramidTestAccess::PlanNswBuildChunks({{7, 8}}, 4, allocator.get());
    REQUIRE(small_chunks.size() == 1);
    const std::array<uint64_t, 3> expected_small_chunk{0, 7, 8};
    REQUIRE(small_chunks.front() == expected_small_chunk);
}

TEST_CASE("Pyramid parallel NSW Build stays reachable across repeated schedules",
          "[ut][pyramid][build]") {
    constexpr int64_t count = 1000;
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillPermutedPyramidTestVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 5000);
    std::vector<std::string> paths(count, "");

    const std::string low_ef_search = R"({"pyramid":{"ef_search":32,"subindex_ef_search":32}})";
    for (uint64_t iteration = 0; iteration < 16; ++iteration) {
        CAPTURE(iteration);
        auto test_index = MakePyramidIndex(3, 4, 4, true, false, true, 8, 32);
        const auto& index = test_index.index;
        REQUIRE(index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
                    .empty());
        REQUIRE(GetReachableNodeCount(index) == count);
        REQUIRE(HasConsistentRootReverseEdges(index, count));

        uint64_t self_matches = 0;
        for (uint64_t query = 0; query < 32; ++query) {
            const auto row = static_cast<int64_t>((iteration * 61 + query * 29) % count);
            self_matches +=
                PyramidSelfMatches(index, vectors, ids, nullptr, row, low_ef_search) ? 1 : 0;
        }
        CAPTURE(self_matches);
        REQUIRE(self_matches >= 28);
        RequirePyramidSelfMatch(index, vectors, ids, nullptr, iteration % count);
    }
}

TEST_CASE("Pyramid parallel NSW Build repairs connectivity at very low graph degree",
          "[ut][pyramid][build]") {
    constexpr int64_t count = 256;
    const uint64_t max_degree = GENERATE(1, 2);
    CAPTURE(max_degree);
    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillPyramidTestVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 6000);
    std::vector<std::string> paths(count, "");

    for (uint64_t iteration = 0; iteration < 8; ++iteration) {
        CAPTURE(iteration);
        auto test_index = MakePyramidIndex(3, 4, 4, true, false, false, max_degree);
        const auto& index = test_index.index;
        REQUIRE(index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count))
                    .empty());
        REQUIRE(GetReachableNodeCount(index) == count);
    }
}

TEST_CASE("Pyramid NSW Build weights chunks toward a large graph node", "[ut][pyramid][build]") {
    constexpr int64_t large_count = 512;
    constexpr int64_t small_node_count = 9;
    constexpr int64_t small_count = 3;
    constexpr int64_t count = large_count + small_node_count * small_count;
    auto test_index = MakePyramidIndex(3, 4, 4);
    const auto& index = test_index.index;

    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillPyramidTestVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 2000);
    std::vector<std::string> paths(count, "large");
    for (int64_t node = 0; node < small_node_count; ++node) {
        for (int64_t i = 0; i < small_count; ++i) {
            paths[large_count + node * small_count + i] = "small-" + std::to_string(node);
        }
    }

    REQUIRE(
        index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count)).empty());
    REQUIRE(GetPyramidNodeStatusCount(index, "GRAPH") == 10);
    REQUIRE(GetPyramidNodeStatusCount(index, "NO_INDEX") == 1);
    REQUIRE(GetPyramidTotalNodes(index) == 11);
    REQUIRE(GetPyramidSubindexCount(index, "graph_subindexes") == 10);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == count);

    for (const int64_t row : {int64_t{0}, large_count / 2, large_count - 1}) {
        RequirePyramidSelfMatch(index, vectors, ids, paths.data() + row, row);
    }
    for (int64_t node = 0; node < small_node_count; ++node) {
        const int64_t row = large_count + node * small_count;
        RequirePyramidSelfMatch(index, vectors, ids, paths.data() + row, row);
    }
}

TEST_CASE("Pyramid NSW Build safely parallelizes parent and child graphs", "[ut][pyramid][build]") {
    constexpr int64_t count = 512;
    auto test_index = MakePyramidIndex(3, 4, 4);
    const auto& index = test_index.index;

    std::vector<float> vectors(count * PYRAMID_TEST_DIM);
    FillPyramidTestVectors(vectors, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 3000);
    std::vector<std::string> paths(count, "parent/child");

    REQUIRE(
        index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count)).empty());
    REQUIRE(GetPyramidNodeStatusCount(index, "GRAPH") == 2);
    REQUIRE(GetPyramidNodeStatusCount(index, "NO_INDEX") == 1);
    REQUIRE(GetPyramidTotalNodes(index) == 3);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == count);

    for (const int64_t row : {int64_t{0}, count / 2, count - 1}) {
        RequirePyramidSelfMatch(index, vectors, ids, paths.data() + row, row);
        std::string parent_path = "parent";
        RequirePyramidSelfMatch(index, vectors, ids, &parent_path, row);
    }
}

TEST_CASE("Pyramid NSW Build keeps duplicate handling node-serial", "[ut][pyramid][build]") {
    constexpr int64_t count = 128;
    auto test_index = MakePyramidIndex(3, 4, 4, true, true);
    const auto& index = test_index.index;

    std::vector<float> vectors(count * PYRAMID_TEST_DIM, 1.0F);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 4000);
    std::vector<std::string> paths(count, "");

    REQUIRE(
        index->Build(MakePyramidDataset(vectors.data(), ids.data(), paths.data(), count)).empty());
    REQUIRE(index->GetNumElements() == count);
    REQUIRE(GetPyramidNodeStatusCount(index, "GRAPH") == 1);
    REQUIRE(GetPyramidSubindexCount(index, "total_vectors_in_graph") == 1);
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
