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

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>

#include "datacell/flatten_interface.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

// HGraph protects slot-map lifetime and capacity changes with its outer locks. Slot bindings are
// published and read with atomics so the search hot path does not take an extra map lock.
class HGraphCodeSlotMap {
public:
    explicit HGraphCodeSlotMap(Allocator* allocator);

    ~HGraphCodeSlotMap();

    InnerIdType
    AllocateSlot();

    void
    PublishSlot(InnerIdType inner_id, InnerIdType code_slot_id);

    void
    RebindSlot(InnerIdType inner_id, InnerIdType code_slot_id);

    void
    RemoveLogical(InnerIdType inner_id);

    void
    MoveLogical(InnerIdType from, InnerIdType to);

    void
    CompactPhysicalSlotsAfter(InnerIdType removed_slot);

    [[nodiscard]] InnerIdType
    Resolve(InnerIdType inner_id) const;

    void
    ResolvePair(InnerIdType inner_id1,
                InnerIdType inner_id2,
                InnerIdType& code_slot_id1,
                InnerIdType& code_slot_id2) const;

    void
    ResolveBatch(const InnerIdType* inner_ids, InnerIdType count, InnerIdType* code_slot_ids) const;

    void
    ReserveLogicalSize(InnerIdType new_size);

    void
    MergeOther(const HGraphCodeSlotMap& other, InnerIdType logical_bias, InnerIdType physical_bias);

    [[nodiscard]] InnerIdType
    PhysicalCount() const;

    [[nodiscard]] InnerIdType
    PublishedLogicalCount() const;

    void
    Clear();

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    [[nodiscard]] uint64_t
    GetMemoryUsage() const;

private:
    void
    EnsureLogicalSize(InnerIdType new_size);

    void
    ReleaseSlots();

    Allocator* allocator_{nullptr};
    std::atomic<InnerIdType>* inner_to_slot_{nullptr};
    InnerIdType logical_capacity_{0};
    std::atomic<InnerIdType> physical_count_{0};
    std::atomic<InnerIdType> published_logical_count_{0};
    mutable std::shared_mutex mutex_;
};

FlattenInterfacePtr
MakeHGraphCodeSlotAdapter(FlattenInterfacePtr physical,
                          std::shared_ptr<const HGraphCodeSlotMap> slot_map,
                          Allocator* allocator,
                          const std::atomic<uint64_t>* logical_total_count);

// Capacity, movement, and compaction use physical slot ids and must bypass the logical-id adapter.
FlattenInterfacePtr
GetHGraphPhysicalFlatten(const FlattenInterfacePtr& flatten);

void
InsertVectorToHGraphPhysicalSlot(const FlattenInterfacePtr& flatten,
                                 const void* vector,
                                 InnerIdType code_slot_id);

}  // namespace vsag
