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

#include <stdexcept>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

using Seconds = std::chrono::duration<double>;

}  // namespace

double
ElapsedSeconds(const Clock::time_point& start) {
    return std::chrono::duration_cast<Seconds>(Clock::now() - start).count();
}

void
Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string
GetString(const JsonType& object, const std::string& key, const std::string& default_value) {
    if (!object.is_object() || !object.contains(key)) {
        return default_value;
    }
    Require(object[key].is_string(), key + " must be a string");
    return object[key].get<std::string>();
}

int
GetInt(const JsonType& object, const std::string& key, int default_value) {
    if (!object.is_object() || !object.contains(key)) {
        return default_value;
    }
    Require(object[key].is_number_integer(), key + " must be an integer");
    return object[key].get<int>();
}

uint64_t
GetUInt64(const JsonType& object, const std::string& key, uint64_t default_value) {
    if (!object.is_object() || !object.contains(key)) {
        return default_value;
    }
    Require(object[key].is_number_unsigned() || object[key].is_number_integer(),
            key + " must be an unsigned integer");
    auto value = object[key].get<int64_t>();
    Require(value >= 0, key + " must be an unsigned integer");
    return static_cast<uint64_t>(value);
}

bool
GetBool(const JsonType& object, const std::string& key, bool default_value) {
    if (!object.is_object() || !object.contains(key)) {
        return default_value;
    }
    Require(object[key].is_boolean(), key + " must be a boolean");
    return object[key].get<bool>();
}

JsonType&
EnsureObject(JsonType& object, const std::string& key) {
    if (!object.contains(key) || object[key].is_null()) {
        object[key] = JsonType::object();
    }
    Require(object[key].is_object(), key + " must be an object");
    return object[key];
}

}  // namespace vsag::autotune::internal
