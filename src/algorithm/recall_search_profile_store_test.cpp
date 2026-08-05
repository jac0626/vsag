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

#include "recall_search_profile_store.h"

#include <limits>
#include <string>

#include "unittest.h"
#include "vsag/errors.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

RecallSearchProfileEntry
MakeEntry(int64_t top_k,
          double target_recall,
          double validated_recall,
          std::string parameters,
          std::string path = {}) {
    return RecallSearchProfileEntry{
        .top_k = top_k,
        .target_recall = target_recall,
        .validated_recall = validated_recall,
        .path = std::move(path),
        .search_parameters = std::move(parameters),
    };
}

void
RequireInvalidArgument(const std::function<void()>& function) {
    try {
        function();
        FAIL("expected INVALID_ARGUMENT");
    } catch (const VsagException& exception) {
        REQUIRE(exception.error_.type == ErrorType::INVALID_ARGUMENT);
    }
}

}  // namespace

TEST_CASE("recall search profile resolves the smallest validated point",
          "[ut][RecallSearchProfileStore]") {
    RecallSearchProfileStore store;
    store.Update(MakeEntry(10, 0.80, 0.82, R"({"hgraph":{"ef_search":40}})"));
    store.Update(MakeEntry(10, 0.90, 0.91, R"({"hgraph":{"ef_search":80}})"));
    store.Update(MakeEntry(10, 0.95, 0.96, R"({"hgraph":{"ef_search":160}})", "catalog/hard"));

    REQUIRE(*store.Resolve(10, 0.80, "") == R"({"hgraph":{"ef_search":40}})");
    REQUIRE(*store.Resolve(10, 0.85, "") == R"({"hgraph":{"ef_search":80}})");
    REQUIRE(*store.Resolve(10, 0.95, "catalog/hard") == R"({"hgraph":{"ef_search":160}})");

    RequireInvalidArgument([&store]() { (void)store.Resolve(100, 0.80, ""); });
    RequireInvalidArgument([&store]() { (void)store.Resolve(10, 0.92, ""); });
    RequireInvalidArgument([&store]() { (void)store.Resolve(10, 0.80, "missing"); });
}

TEST_CASE("recall search profile update publishes a new immutable snapshot",
          "[ut][RecallSearchProfileStore]") {
    RecallSearchProfileStore store;
    store.Update(MakeEntry(10, 0.90, 0.91, R"({"hgraph":{"ef_search":80}})"));
    auto previous = store.Resolve(10, 0.90, "");

    store.Update(MakeEntry(10, 0.90, 0.93, R"({"hgraph":{"ef_search":96}})"));

    REQUIRE(*previous == R"({"hgraph":{"ef_search":80}})");
    REQUIRE(*store.Resolve(10, 0.90, "") == R"({"hgraph":{"ef_search":96}})");
    REQUIRE(*store.Resolve(10, 0.92, "") == R"({"hgraph":{"ef_search":96}})");

    const auto one_third = 1.0 / 3.0;
    store.Update(MakeEntry(20, one_third, one_third, R"({"hgraph":{"ef_search":32}})"));
    REQUIRE(*store.Resolve(20, one_third, "") == R"({"hgraph":{"ef_search":32}})");
}

TEST_CASE("recall search profile validates entries and requests",
          "[ut][RecallSearchProfileStore]") {
    RecallSearchProfileStore store;
    RequireInvalidArgument(
        [&store]() { store.Update(MakeEntry(0, 0.80, 0.81, R"({"hgraph":{}})")); });
    RequireInvalidArgument(
        [&store]() { store.Update(MakeEntry(10, 0.90, 0.89, R"({"hgraph":{}})")); });
    RequireInvalidArgument([&store]() { store.Update(MakeEntry(10, 0.80, 0.81, "not-json")); });
    RequireInvalidArgument([&store]() {
        store.Update(
            MakeEntry(10, std::numeric_limits<double>::quiet_NaN(), 0.81, R"({"hgraph":{}})"));
    });
    RequireInvalidArgument([&store]() { (void)store.Resolve(10, 1.01, ""); });
}

TEST_CASE("recall search profile serializes and restores", "[ut][RecallSearchProfileStore]") {
    RecallSearchProfileStore store;
    store.Update(MakeEntry(10, 0.80, 0.82, R"({"hgraph":{"ef_search":40}})"));
    store.Update(MakeEntry(10, 0.95, 0.96, R"({"pyramid":{"ef_search":160}})", "catalog/hard"));

    JsonType basic_info;
    store.AppendTo(basic_info);
    REQUIRE(basic_info.Contains("recall_search_profiles"));

    RecallSearchProfileStore restored;
    restored.RestoreFrom(basic_info);
    REQUIRE(*restored.Resolve(10, 0.80, "") == R"({"hgraph":{"ef_search":40}})");
    REQUIRE(*restored.Resolve(10, 0.95, "catalog/hard") == R"({"pyramid":{"ef_search":160}})");

    JsonType empty_info;
    restored.RestoreFrom(empty_info);
    RequireInvalidArgument([&restored]() { (void)restored.Resolve(10, 0.80, ""); });

    auto invalid_info = JsonType::Parse(R"({"recall_search_profiles":{}})");
    try {
        restored.RestoreFrom(invalid_info);
        FAIL("expected INVALID_BINARY");
    } catch (const VsagException& exception) {
        REQUIRE(exception.error_.type == ErrorType::INVALID_BINARY);
    }
}

}  // namespace vsag
