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

#include <memory>
#include <vector>

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(DuplicateInterface);

class DuplicateInterface {
public:
    virtual ~DuplicateInterface() = default;

    virtual void
    SetDuplicateId(InnerIdType group_id, InnerIdType duplicate_id) = 0;

    [[nodiscard]] virtual auto
    GetDuplicateIds(InnerIdType id) const -> std::vector<InnerIdType> = 0;

    [[nodiscard]] virtual auto
    GetGroupId(InnerIdType id) const -> InnerIdType = 0;

    [[nodiscard]] virtual auto
    GetGroupSize(InnerIdType id) const -> uint64_t {
        auto group_id = this->GetGroupId(id);
        return static_cast<uint64_t>(this->GetDuplicateIds(group_id).size()) + 1;
    }

    virtual void
    Serialize(StreamWriter& writer) const = 0;

    virtual void
    Deserialize(StreamReader& reader) = 0;

    virtual void
    DeserializeFromLegacyFormat(StreamReader& reader, size_t total_size) = 0;

    virtual void
    MergeOther(const DuplicateInterface& other, InnerIdType bias, InnerIdType count) {
        std::vector<bool> visited(count, false);
        for (InnerIdType id = 0; id < count; ++id) {
            if (visited[id]) {
                continue;
            }

            const auto group_id = other.GetGroupId(id);
            if (group_id >= count || visited[group_id]) {
                visited[id] = true;
                continue;
            }

            visited[id] = true;
            visited[group_id] = true;
            for (auto duplicate_id : other.GetDuplicateIds(group_id)) {
                if (duplicate_id >= count || visited[duplicate_id]) {
                    continue;
                }
                visited[duplicate_id] = true;
                this->SetDuplicateId(group_id + bias, duplicate_id + bias);
            }
        }
    }

    virtual void
    Resize(InnerIdType new_size) = 0;
};

using DuplicateTrackerPtr = std::shared_ptr<DuplicateInterface>;

}  // namespace vsag
