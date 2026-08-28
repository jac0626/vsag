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
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

TEST_CASE("PyramidPathStore inserts and reorders paths", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 4> source = {"root/a", "", "root/c", "root/d"};

    {
        auto writer = store.AcquireWriter();
        writer.Prepare(source.size(), source.size());
        for (uint64_t slot = 0; slot < source.size(); ++slot) {
            writer.Insert(static_cast<vsag::InnerIdType>(slot), source[slot]);
        }
    }
    REQUIRE(store.Size() == source.size());

    vsag::Vector<vsag::InnerIdType> inner_ids(allocator.get());
    inner_ids.push_back(2);
    inner_ids.push_back(0);
    inner_ids.push_back(1);
    std::array<std::string, 3> restored;

    REQUIRE(store.GetPaths(inner_ids, restored.data()));
    REQUIRE(restored == std::array<std::string, 3>{"root/c", "root/a", ""});

    {
        auto writer = store.AcquireWriter();
        REQUIRE_THROWS(writer.Insert(0, "duplicate"));
    }
    REQUIRE(store.Size() == source.size());
}

TEST_CASE("PyramidPathStore inserts paths with holes", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 4> source = {"zero", "one", "two", "three"};
    {
        auto writer = store.AcquireWriter();
        writer.Prepare(6, 2);
        writer.Prepare(3, 0);
        writer.Insert(5, source[1]);
        writer.Insert(4, source[3]);
    }
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

TEST_CASE("PyramidPathStore stores zero, one, and multiple paths", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 3> multiple_paths = {"root/a", "", "root/a"};

    {
        auto writer = store.AcquireWriter();
        writer.Prepare(5, 4);
        writer.Insert(0, nullptr, 0);
        writer.Insert(2, multiple_paths.data(), multiple_paths.size());
        writer.Insert(4, "");
    }

    vsag::Vector<vsag::InnerIdType> inner_ids(allocator.get());
    inner_ids.push_back(2);
    inner_ids.push_back(0);
    inner_ids.push_back(4);
    std::vector<std::vector<std::string>> restored_rows;
    REQUIRE(store.GetPathRows(inner_ids, restored_rows));
    const std::vector<std::vector<std::string>> expected_rows = {
        {"root/a", "", "root/a"}, {}, {""}};
    REQUIRE(restored_rows == expected_rows);

    std::array<std::string, 3> legacy_paths;
    REQUIRE_FALSE(store.GetPaths(inner_ids, legacy_paths.data()));

    vsag::Vector<vsag::InnerIdType> single_id(allocator.get());
    single_id.push_back(4);
    REQUIRE(store.GetPaths(single_id, legacy_paths.data()));
    REQUIRE(legacy_paths[0].empty());

    vsag::Vector<vsag::InnerIdType> hole_id(allocator.get());
    hole_id.push_back(1);
    REQUIRE_FALSE(store.GetPathRows(hole_id, restored_rows));
    REQUIRE(restored_rows == expected_rows);
}

TEST_CASE("PyramidPathStore validates path rows", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());

    {
        auto writer = store.AcquireWriter();
        REQUIRE_THROWS(writer.Insert(
            0, nullptr, static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1));
        REQUIRE_THROWS(writer.Insert(0, nullptr, 1));
    }
    REQUIRE(store.Size() == 0);
    {
        auto writer = store.AcquireWriter();
        writer.Insert(0, nullptr, 0);
        REQUIRE_THROWS(writer.Insert(0, nullptr, 0));
    }
    REQUIRE(store.Size() == 1);
}

TEST_CASE("PyramidPathStore serialization roundtrip", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 3> source = {"first", "unused", ""};
    {
        auto writer = store.AcquireWriter();
        writer.Insert(3, source[0]);
        writer.Insert(2, source[2]);
    }

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    store.Serialize(writer);

    std::stringstream legacy_stream;
    vsag::IOStreamWriter legacy_writer(legacy_stream);
    vsag::StreamWriter::WriteObj(legacy_writer, uint64_t{4});
    vsag::StreamWriter::WriteObj(legacy_writer, uint8_t{0});
    vsag::StreamWriter::WriteObj(legacy_writer, uint8_t{0});
    vsag::StreamWriter::WriteObj(legacy_writer, uint8_t{1});
    vsag::StreamWriter::WriteString(legacy_writer, "");
    vsag::StreamWriter::WriteObj(legacy_writer, uint8_t{1});
    vsag::StreamWriter::WriteString(legacy_writer, "first");
    REQUIRE(stream.str() == legacy_stream.str());

    vsag::PyramidPathStore restored_store(allocator.get());
    vsag::IOStreamReader reader(stream);
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

TEST_CASE("PyramidPathStore multi-path serialization roundtrip", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::PyramidPathStore store(allocator.get());
    const std::array<std::string, 3> multiple_paths = {"a", "", "a"};
    {
        auto writer = store.AcquireWriter();
        writer.Prepare(4, 4);
        writer.Insert(0, nullptr, 0);
        writer.Insert(2, multiple_paths.data(), multiple_paths.size());
        writer.Insert(3, "one");
    }

    std::stringstream stream;
    vsag::IOStreamWriter writer(stream);
    store.Serialize(writer);

    vsag::PyramidPathStore reversed_store(allocator.get());
    {
        auto reversed_writer = reversed_store.AcquireWriter();
        reversed_writer.Insert(3, "one");
        reversed_writer.Insert(2, multiple_paths.data(), multiple_paths.size());
        reversed_writer.Insert(0, nullptr, 0);
    }
    std::stringstream reversed_stream;
    vsag::IOStreamWriter reversed_stream_writer(reversed_stream);
    reversed_store.Serialize(reversed_stream_writer);
    REQUIRE(stream.str() == reversed_stream.str());

    std::stringstream expected_stream;
    vsag::IOStreamWriter expected_writer(expected_stream);
    vsag::StreamWriter::WriteObj(expected_writer, uint64_t{4});
    vsag::StreamWriter::WriteObj(expected_writer, uint8_t{2});
    vsag::StreamWriter::WriteObj(expected_writer, uint16_t{0});
    vsag::StreamWriter::WriteObj(expected_writer, uint8_t{0});
    vsag::StreamWriter::WriteObj(expected_writer, uint8_t{2});
    vsag::StreamWriter::WriteObj(expected_writer, uint16_t{3});
    for (const auto& path : multiple_paths) {
        vsag::StreamWriter::WriteString(expected_writer, path);
    }
    vsag::StreamWriter::WriteObj(expected_writer, uint8_t{1});
    vsag::StreamWriter::WriteString(expected_writer, "one");
    REQUIRE(stream.str() == expected_stream.str());

    vsag::PyramidPathStore restored_store(allocator.get());
    vsag::IOStreamReader reader(stream);
    restored_store.Deserialize(reader, 4);

    vsag::Vector<vsag::InnerIdType> inner_ids(allocator.get());
    inner_ids.push_back(0);
    inner_ids.push_back(2);
    inner_ids.push_back(3);
    std::vector<std::vector<std::string>> restored_rows;
    REQUIRE(restored_store.GetPathRows(inner_ids, restored_rows));
    const std::vector<std::vector<std::string>> expected_rows = {{}, {"a", "", "a"}, {"one"}};
    REQUIRE(restored_rows == expected_rows);

    vsag::Vector<vsag::InnerIdType> hole_id(allocator.get());
    hole_id.push_back(1);
    REQUIRE_FALSE(restored_store.GetPathRows(hole_id, restored_rows));
}

TEST_CASE("PyramidPathStore rejects malformed serialization", "[ut][pyramid][path_store]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    SECTION("slot count exceeds maximum") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{3});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 2));
    }

    SECTION("path state is invalid") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{1});
        vsag::StreamWriter::WriteObj(writer, uint8_t{3});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("multi-path state is missing its count") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{1});
        vsag::StreamWriter::WriteObj(writer, uint8_t{2});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("multi-path state uses the legacy single-path count") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{1});
        vsag::StreamWriter::WriteObj(writer, uint8_t{2});
        vsag::StreamWriter::WriteObj(writer, uint16_t{1});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("multi-path strings exceed remaining payload") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{1});
        vsag::StreamWriter::WriteObj(writer, uint8_t{2});
        vsag::StreamWriter::WriteObj(writer, uint16_t{2});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("slot count exceeds remaining payload") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{2});

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 2));
    }

    SECTION("path string length exceeds remaining payload") {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{1});
        vsag::StreamWriter::WriteObj(writer, uint8_t{1});
        vsag::StreamWriter::WriteObj(writer, std::numeric_limits<uint64_t>::max());

        vsag::PyramidPathStore store(allocator.get());
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));
    }

    SECTION("failed restore leaves existing paths unchanged") {
        vsag::PyramidPathStore store(allocator.get());
        {
            auto store_writer = store.AcquireWriter();
            store_writer.Insert(0, "existing");
        }

        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        vsag::StreamWriter::WriteObj(writer, uint64_t{1});
        vsag::StreamWriter::WriteObj(writer, uint8_t{3});
        vsag::IOStreamReader reader(stream);
        REQUIRE_THROWS(store.Deserialize(reader, 1));

        vsag::Vector<vsag::InnerIdType> inner_ids(allocator.get());
        inner_ids.push_back(0);
        std::array<std::string, 1> restored;
        REQUIRE(store.GetPaths(inner_ids, restored.data()));
        REQUIRE(restored[0] == "existing");
    }
}
