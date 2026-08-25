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

#pragma once

#include <shared_mutex>
#include <string>

#include "typing.h"

namespace vsag {

class StreamReader;
class StreamWriter;

std::string
ReadPyramidPathString(StreamReader& reader);

class PyramidPathStore {
public:
    explicit PyramidPathStore(Allocator* allocator)
        : paths_by_inner_id_(allocator), has_path_(allocator), allocator_(allocator) {
    }

    void
    Record(const std::string* paths, const Vector<int64_t>& data_biases, int64_t first_inner_id);

    void
    Record(const std::string* paths, uint64_t count);

    [[nodiscard]] bool
    GetPaths(const Vector<InnerIdType>& inner_ids, std::string* paths) const;

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader, uint64_t max_count);

    [[nodiscard]] uint64_t
    Size() const;

private:
    mutable std::shared_mutex mutex_;
    Vector<std::string> paths_by_inner_id_;
    Vector<uint8_t> has_path_;
    Allocator* allocator_{nullptr};
};

}  // namespace vsag
