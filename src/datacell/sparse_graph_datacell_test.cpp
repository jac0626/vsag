
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

#include "sparse_graph_datacell.h"

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <future>
#include <thread>

#include "graph_interface_test.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "sparse_graph_datacell_parameter.h"
#include "unittest.h"
#include "utils/lock_strategy.h"
using namespace vsag;

void
TestSparseGraphDataCell(const GraphInterfaceParamPtr& param,
                        const IndexCommonParam& common_param,
                        bool test_delete) {
    auto count = GENERATE(1000, 2000);
    auto max_id = 10000;

    auto graph = GraphInterface::MakeInstance(param, common_param);
    GraphInterfaceTest test(graph);
    auto other = GraphInterface::MakeInstance(param, common_param);
    test.BasicTest(max_id, count, other, test_delete);
}

TEST_CASE("SparseGraphDataCell Basic Test", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 32, 64);
    auto is_support_delete = GENERATE(true, false);

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->support_delete_ = is_support_delete;
    TestSparseGraphDataCell(graph_param, common_param, is_support_delete);
}

TEST_CASE("SparseGraphDataCell Remove Test", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 32);
    auto is_support_delete = GENERATE(true);
    auto remove_flag_bit = GENERATE(4, 8);

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->support_delete_ = is_support_delete;
    graph_param->remove_flag_bit_ = remove_flag_bit;
    TestSparseGraphDataCell(graph_param, common_param, is_support_delete);
}

TEST_CASE("SparseGraphDataCell Merge Test", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 32, 64);
    auto is_support_delete = GENERATE(true, false);
    int count = 1000;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->support_delete_ = is_support_delete;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    auto other = GraphInterface::MakeInstance(graph_param, common_param);
    test.MergeTest(other, count);
}

TEST_CASE("SparseGraphDataCell Reverse Edges", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = 32;
    auto max_degree = 32;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->use_reverse_edges_ = true;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    test.ReverseEdgeTest(100);
}

TEST_CASE("SparseGraphDataCell Move", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = 32;
    auto max_degree = 32;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->use_reverse_edges_ = true;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);
    GraphInterfaceTest test(graph);
    test.MoveTest(100);
}

TEST_CASE("SparseGraphDataCell Move same id keeps neighbors", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = 8;
    graph_param->use_reverse_edges_ = true;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);

    Vector<InnerIdType> empty(allocator.get());
    Vector<InnerIdType> neighbors(allocator.get());
    neighbors.emplace_back(1);
    neighbors.emplace_back(2);

    graph->InsertNeighborsById(0, neighbors);
    graph->InsertNeighborsById(1, empty);
    graph->InsertNeighborsById(2, empty);

    graph->Move(0, 0);

    Vector<InnerIdType> moved_neighbors(allocator.get());
    graph->GetNeighbors(0, moved_neighbors);
    REQUIRE(moved_neighbors.size() == 2);
    REQUIRE(moved_neighbors[0] == 1);
    REQUIRE(moved_neighbors[1] == 2);

    Vector<InnerIdType> incoming(allocator.get());
    graph->GetIncomingNeighbors(1, incoming);
    REQUIRE(incoming.size() == 1);
    REQUIRE(incoming[0] == 0);
}

TEST_CASE("SparseGraphDataCell decodes stored neighbors for reverse-edge updates",
          "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = 8;
    graph_param->support_delete_ = true;
    graph_param->remove_flag_bit_ = 4;
    graph_param->use_reverse_edges_ = true;

    auto graph = GraphInterface::MakeInstance(graph_param, common_param);

    Vector<InnerIdType> empty(allocator.get());
    Vector<InnerIdType> to_two(allocator.get());
    Vector<InnerIdType> to_three(allocator.get());
    to_two.emplace_back(2);
    to_three.emplace_back(3);

    graph->InsertNeighborsById(2, empty);
    graph->InsertNeighborsById(3, empty);
    graph->DeleteNeighborsById(2);
    graph->InsertNeighborsById(1, to_two);

    Vector<InnerIdType> incoming(allocator.get());
    graph->GetIncomingNeighbors(2, incoming);
    REQUIRE(incoming.size() == 1);
    REQUIRE(incoming[0] == 1);

    graph->InsertNeighborsById(1, to_three);

    graph->GetIncomingNeighbors(2, incoming);
    REQUIRE(incoming.empty());

    graph->GetIncomingNeighbors(3, incoming);
    REQUIRE(incoming.size() == 1);
    REQUIRE(incoming[0] == 1);
}

TEST_CASE("SparseGraphDataCell supports frozen concurrent build updates",
          "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = 8;
    graph_param->support_delete_ = true;
    graph_param->use_reverse_edges_ = true;

    auto graph = std::make_shared<SparseGraphDataCell>(graph_param, common_param);
    Vector<InnerIdType> ids(allocator.get());
    ids.emplace_back(2);
    ids.emplace_back(101);
    ids.emplace_back(9);
    graph->PreallocateNodesForBuild(ids);

    REQUIRE(graph->TotalCount() == 3);
    REQUIRE(graph->MaxCapacity() == 102);
    for (const auto id : ids) {
        REQUIRE(graph->CheckIdExists(id));
    }

    Vector<InnerIdType> neighbors(allocator.get());
    neighbors.emplace_back(101);
    neighbors.emplace_back(9);
    graph->UpdateNeighborsForBuild(2, neighbors);

    Vector<InnerIdType> actual(allocator.get());
    graph->GetNeighbors(2, actual);
    REQUIRE(actual == neighbors);
    graph->GetNeighborsForBuild(2, actual);
    REQUIRE(actual == neighbors);

    Vector<InnerIdType> incoming(allocator.get());
    graph->GetIncomingNeighbors(101, incoming);
    Vector<InnerIdType> expected_incoming(allocator.get());
    expected_incoming.emplace_back(2);
    REQUIRE(incoming == expected_incoming);

    Vector<InnerIdType> missing_neighbor(allocator.get());
    missing_neighbor.emplace_back(77);
    REQUIRE_THROWS(graph->UpdateNeighborsForBuild(2, missing_neighbor));
    REQUIRE_THROWS(graph->UpdateNeighborsForBuild(77, neighbors));

    graph->FinishConcurrentBuild();
    graph->DeleteNeighborsById(101);
    graph->GetNeighbors(2, actual);
    Vector<InnerIdType> expected_after_delete(allocator.get());
    expected_after_delete.emplace_back(9);
    REQUIRE(actual == expected_after_delete);

    REQUIRE_THROWS(graph->UpdateNeighborsForBuild(2, neighbors));
    REQUIRE_THROWS(graph->GetNeighborsForBuild(2, actual));
}

TEST_CASE("SparseGraphDataCell updates frozen nodes concurrently", "[ut][SparseGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    IndexCommonParam common_param;
    common_param.dim_ = 32;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = 8;
    graph_param->support_delete_ = true;
    graph_param->use_reverse_edges_ = true;

    constexpr InnerIdType node_count = 128;
    constexpr uint64_t worker_count = 8;
    auto graph = std::make_shared<SparseGraphDataCell>(graph_param, common_param);
    Vector<InnerIdType> ids(allocator.get());
    ids.reserve(node_count);
    for (InnerIdType id = 0; id < node_count; ++id) {
        ids.emplace_back(id);
    }
    graph->PreallocateNodesForBuild(ids);

    MutexArrayPtr point_mutexes = std::make_shared<PointsMutex>(node_count, allocator.get());
    std::atomic<uint64_t> ready_count{0};
    std::atomic<bool> start{false};
    std::vector<std::future<void>> workers;
    workers.reserve(worker_count);
    for (uint64_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back(std::async(std::launch::async, [&, worker]() {
            ready_count.fetch_add(1, std::memory_order_release);
            while (not start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (InnerIdType id = static_cast<InnerIdType>(worker); id < node_count;
                 id += worker_count) {
                Vector<InnerIdType> neighbors(allocator.get());
                neighbors.emplace_back((id + 1) % node_count);
                neighbors.emplace_back((id + 2) % node_count);
                {
                    LockGuard lock(point_mutexes, id);
                    graph->UpdateNeighborsForBuild(id, neighbors);
                }

                const auto read_id = (id + node_count - 1) % node_count;
                Vector<InnerIdType> observed(allocator.get());
                SharedLock lock(point_mutexes, read_id);
                graph->GetNeighborsForBuild(read_id, observed);
            }
        }));
    }

    while (ready_count.load(std::memory_order_acquire) < worker_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.get();
    }
    graph->FinishConcurrentBuild();

    for (InnerIdType id = 0; id < node_count; ++id) {
        Vector<InnerIdType> actual(allocator.get());
        graph->GetNeighbors(id, actual);
        Vector<InnerIdType> expected(allocator.get());
        expected.emplace_back((id + 1) % node_count);
        expected.emplace_back((id + 2) % node_count);
        REQUIRE(actual == expected);

        Vector<InnerIdType> incoming(allocator.get());
        graph->GetIncomingNeighbors(id, incoming);
        std::sort(incoming.begin(), incoming.end());
        Vector<InnerIdType> expected_incoming(allocator.get());
        expected_incoming.emplace_back((id + node_count - 2) % node_count);
        expected_incoming.emplace_back((id + node_count - 1) % node_count);
        std::sort(expected_incoming.begin(), expected_incoming.end());
        REQUIRE(incoming == expected_incoming);
    }
}
