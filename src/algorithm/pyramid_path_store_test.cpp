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

#include "pyramid_path_store.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>

#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

TEST_CASE("PyramidPathStore records and reorders paths", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 4> source = {"root/a", "", "root/c", "root/d"};

    store.Record(source.data(), source.size());
    REQUIRE(store.Size() == source.size());

    vsag::Vector<vsag::InnerIdType> inner_ids(allocator.get());
    inner_ids.push_back(2);
    inner_ids.push_back(0);
    inner_ids.push_back(1);
    std::array<std::string, 3> restored;

    REQUIRE(store.GetPaths(inner_ids, restored.data()));
    REQUIRE(restored == std::array<std::string, 3>{"root/c", "root/a", ""});
}

TEST_CASE("PyramidPathStore records filtered paths with holes", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 4> source = {"zero", "one", "two", "three"};
    vsag::Vector<int64_t> data_biases(allocator.get());
    data_biases.push_back(3);
    data_biases.push_back(1);

    store.Record(source.data(), data_biases, 4);
    REQUIRE(store.Size() == 6);

    vsag::Vector<vsag::InnerIdType> present_ids(allocator.get());
    present_ids.push_back(5);
    present_ids.push_back(4);
    std::array<std::string, 2> restored;
    REQUIRE(store.GetPaths(present_ids, restored.data()));
    REQUIRE(restored == std::array<std::string, 2>{"one", "three"});

    vsag::Vector<vsag::InnerIdType> hole_ids(allocator.get());
    hole_ids.push_back(3);
    REQUIRE_FALSE(store.GetPaths(hole_ids, restored.data()));

    vsag::Vector<vsag::InnerIdType> out_of_range_ids(allocator.get());
    out_of_range_ids.push_back(6);
    REQUIRE_FALSE(store.GetPaths(out_of_range_ids, restored.data()));
}

TEST_CASE("PyramidPathStore serialization roundtrip", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 3> source = {"first", "unused", ""};
    vsag::Vector<int64_t> data_biases(allocator.get());
    data_biases.push_back(2);
    data_biases.push_back(0);
    store.Record(source.data(), data_biases, 2);

    std::stringstream stream;
    IOStreamWriter writer(stream);
    store.Serialize(writer);

    vsag::PyramidPathStore restored_store(allocator.get());
    IOStreamReader reader(stream);
    restored_store.Deserialize(reader, 4);
    REQUIRE(restored_store.Size() == 4);

    vsag::Vector<vsag::InnerIdType> present_ids(allocator.get());
    present_ids.push_back(2);
    present_ids.push_back(3);
    std::array<std::string, 2> restored;
    REQUIRE(restored_store.GetPaths(present_ids, restored.data()));
    REQUIRE(restored == std::array<std::string, 2>{"", "first"});

    vsag::Vector<vsag::InnerIdType> hole_ids(allocator.get());
    hole_ids.push_back(0);
    REQUIRE_FALSE(restored_store.GetPaths(hole_ids, restored.data()));
}

TEST_CASE("PyramidPathStore uses fixed-width path lengths", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 1> source = {"path"};
    store.Record(source.data(), source.size());

    std::stringstream stream;
    IOStreamWriter writer(stream);
    store.Serialize(writer);
    const auto payload = stream.str();
    REQUIRE(payload.size() == sizeof(uint64_t) + sizeof(uint8_t) + sizeof(uint64_t) + 4);

    uint64_t path_length = 0;
    std::memcpy(
        &path_length, payload.data() + sizeof(uint64_t) + sizeof(uint8_t), sizeof(path_length));
    REQUIRE(path_length == 4);

    vsag::PyramidPathStore restored_store(allocator.get());
    IOStreamReader reader(stream);
    restored_store.Deserialize(reader, 1);
    vsag::Vector<vsag::InnerIdType> inner_ids(allocator.get());
    inner_ids.push_back(0);
    std::array<std::string, 1> restored;
    REQUIRE(restored_store.GetPaths(inner_ids, restored.data()));
    REQUIRE(restored == source);
}

TEST_CASE("PyramidPathStore rejects malformed serialization", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    SECTION("slot count exceeds maximum") {
        std::stringstream stream;
        IOStreamWriter writer(stream);
        StreamWriter::WriteObj(writer, uint64_t{3});

        vsag::PyramidPathStore store(allocator.get());
        IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 2));
    }

    SECTION("presence byte is invalid") {
        std::stringstream stream;
        IOStreamWriter writer(stream);
        StreamWriter::WriteObj(writer, uint64_t{1});
        StreamWriter::WriteObj(writer, uint8_t{2});

        vsag::PyramidPathStore store(allocator.get());
        IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("slot count exceeds remaining payload") {
        std::stringstream stream;
        IOStreamWriter writer(stream);
        StreamWriter::WriteObj(writer, uint64_t{2});

        vsag::PyramidPathStore store(allocator.get());
        IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 2));
    }

    SECTION("path length exceeds remaining payload") {
        std::stringstream stream;
        IOStreamWriter writer(stream);
        StreamWriter::WriteObj(writer, uint64_t{1});
        StreamWriter::WriteObj(writer, uint8_t{1});
        StreamWriter::WriteObj(writer, std::numeric_limits<uint64_t>::max());

        vsag::PyramidPathStore store(allocator.get());
        IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }
}
