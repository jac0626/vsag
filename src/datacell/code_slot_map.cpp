// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "code_slot_map.h"

#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>

#include "common.h"
#include "storage/serialization.h"
#include "typing.h"

namespace vsag {

namespace {

constexpr CodeSlotIdType INVALID_CODE_SLOT = std::numeric_limits<CodeSlotIdType>::max();

}  // namespace

CodeSlotMap::CodeSlotMap(Allocator* allocator) : allocator_(allocator) {
}

CodeSlotMap::~CodeSlotMap() {
    this->ReleaseSlots();
}

CodeSlotIdType
CodeSlotMap::AllocateSlot() {
    auto slot_id = physical_count_.fetch_add(1, std::memory_order_acq_rel);
    if (slot_id >= logical_capacity_) {
        physical_count_.fetch_sub(1, std::memory_order_acq_rel);
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("code slot capacity is not reserved for slot {}", slot_id));
    }
    return slot_id;
}

void
// NOLINTNEXTLINE(readability-make-member-function-const): publishing mutates slot bindings.
CodeSlotMap::PublishSlot(InnerIdType inner_id, CodeSlotIdType code_slot_id) {
    auto physical_count = physical_count_.load(std::memory_order_acquire);
    CHECK_ARGUMENT(
        code_slot_id < physical_count,
        fmt::format(
            "code_slot_id({}) must be less than physical_count({})", code_slot_id, physical_count));
    if (inner_id >= logical_capacity_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("inner_id({}) has no reserved code slot", inner_id));
    }
    auto expected = INVALID_CODE_SLOT;
    auto published = inner_to_slot_[inner_id].compare_exchange_strong(
        expected, code_slot_id, std::memory_order_release, std::memory_order_acquire);
    CHECK_ARGUMENT(published, fmt::format("inner_id({}) is already bound", inner_id));
    published_logical_count_.fetch_add(1, std::memory_order_acq_rel);
}

CodeSlotIdType
CodeSlotMap::Resolve(InnerIdType inner_id) const {
    if (inner_id >= logical_capacity_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("inner_id({}) has no bound code slot", inner_id));
    }
    auto slot = inner_to_slot_[inner_id].load(std::memory_order_acquire);
    if (slot == INVALID_CODE_SLOT) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("inner_id({}) has no bound code slot", inner_id));
    }
    return slot;
}

void
CodeSlotMap::ResolvePair(InnerIdType inner_id1,
                         InnerIdType inner_id2,
                         CodeSlotIdType& code_slot_id1,
                         CodeSlotIdType& code_slot_id2) const {
    if (inner_id1 >= logical_capacity_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("inner_id({}) has no bound code slot", inner_id1));
    }
    if (inner_id2 >= logical_capacity_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("inner_id({}) has no bound code slot", inner_id2));
    }
    code_slot_id1 = inner_to_slot_[inner_id1].load(std::memory_order_acquire);
    code_slot_id2 = inner_to_slot_[inner_id2].load(std::memory_order_acquire);
    if (code_slot_id1 == INVALID_CODE_SLOT) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("inner_id({}) has no bound code slot", inner_id1));
    }
    if (code_slot_id2 == INVALID_CODE_SLOT) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("inner_id({}) has no bound code slot", inner_id2));
    }
}

void
CodeSlotMap::ResolveBatch(const InnerIdType* inner_ids,
                          InnerIdType count,
                          CodeSlotIdType* code_slot_ids) const {
    for (InnerIdType i = 0; i < count; ++i) {
        auto inner_id = inner_ids[i];
        if (inner_id >= logical_capacity_) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("inner_id({}) has no bound code slot", inner_id));
        }
        auto slot = inner_to_slot_[inner_id].load(std::memory_order_acquire);
        if (slot == INVALID_CODE_SLOT) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("inner_id({}) has no bound code slot", inner_id));
        }
        code_slot_ids[i] = slot;
    }
}

void
CodeSlotMap::ReserveLogicalSize(InnerIdType new_size) {
    std::unique_lock lock(mutex_);
    this->EnsureLogicalSize(new_size);
}

void
CodeSlotMap::RemoveAndSwapLast(InnerIdType removed_id, InnerIdType last_id) {
    std::unique_lock lock(mutex_);

    auto published_count = published_logical_count_.load(std::memory_order_acquire);
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        published_count > 0 && last_id == published_count - 1,
        fmt::format(
            "last_id({}) must be the last of {} published logical ids", last_id, published_count));
    CHECK_ARGUMENT(removed_id <= last_id,
                   fmt::format("removed_id({}) must not exceed last_id({})", removed_id, last_id));

    auto removed_slot = inner_to_slot_[removed_id].load(std::memory_order_acquire);
    auto last_slot = inner_to_slot_[last_id].load(std::memory_order_acquire);
    CHECK_ARGUMENT(removed_slot != INVALID_CODE_SLOT,
                   fmt::format("removed_id({}) has no bound code slot", removed_id));
    CHECK_ARGUMENT(last_slot != INVALID_CODE_SLOT,
                   fmt::format("last_id({}) has no bound code slot", last_id));

    if (removed_id != last_id) {
        inner_to_slot_[removed_id].store(last_slot, std::memory_order_release);
    }
    inner_to_slot_[last_id].store(INVALID_CODE_SLOT, std::memory_order_release);
    published_logical_count_.fetch_sub(1, std::memory_order_acq_rel);
}

CodeSlotIdType
CodeSlotMap::CompactSlots(const std::function<void(CodeSlotIdType, CodeSlotIdType)>& move_slot) {
    std::unique_lock lock(mutex_);

    const auto physical_count = physical_count_.load(std::memory_order_acquire);
    const auto logical_count = published_logical_count_.load(std::memory_order_acquire);
    Vector<CodeSlotIdType> slot_remap(physical_count, INVALID_CODE_SLOT, allocator_);
    CodeSlotIdType referenced_count = 0;
    for (InnerIdType inner_id = 0; inner_id < logical_count; ++inner_id) {
        const auto slot = inner_to_slot_[inner_id].load(std::memory_order_acquire);
        CHECK_ARGUMENT(slot < physical_count,
                       fmt::format("inner_id({}) has invalid code slot {} for physical count {}",
                                   inner_id,
                                   slot,
                                   physical_count));
        if (slot_remap[slot] == INVALID_CODE_SLOT) {
            slot_remap[slot] = slot;
            ++referenced_count;
        }
    }

    CodeSlotIdType source = physical_count;
    for (CodeSlotIdType target = 0; target < referenced_count; ++target) {
        if (slot_remap[target] != INVALID_CODE_SLOT) {
            continue;
        }
        do {
            --source;
        } while (source >= referenced_count && slot_remap[source] == INVALID_CODE_SLOT);
        CHECK_ARGUMENT(source >= referenced_count,
                       "failed to find a referenced tail code slot during compaction");
        move_slot(source, target);
        slot_remap[source] = target;
    }

    for (InnerIdType inner_id = 0; inner_id < logical_count; ++inner_id) {
        const auto old_slot = inner_to_slot_[inner_id].load(std::memory_order_acquire);
        const auto new_slot = slot_remap[old_slot];
        CHECK_ARGUMENT(new_slot != INVALID_CODE_SLOT,
                       fmt::format("inner_id({}) lost its code slot during compaction", inner_id));
        if (new_slot != old_slot) {
            inner_to_slot_[inner_id].store(new_slot, std::memory_order_release);
        }
    }
    physical_count_.store(referenced_count, std::memory_order_release);
    return referenced_count;
}

CodeSlotIdType
CodeSlotMap::PhysicalCount() const {
    return physical_count_.load(std::memory_order_acquire);
}

InnerIdType
CodeSlotMap::PublishedLogicalCount() const {
    return published_logical_count_.load(std::memory_order_acquire);
}

void
CodeSlotMap::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(mutex_);
    auto physical_count = physical_count_.load(std::memory_order_acquire);
    StreamWriter::WriteObj(writer, physical_count);
    auto serialized_slot_count = logical_capacity_;
    while (serialized_slot_count > 0) {
        auto slot = inner_to_slot_[serialized_slot_count - 1].load(std::memory_order_acquire);
        if (slot != INVALID_CODE_SLOT) {
            break;
        }
        --serialized_slot_count;
    }
    serialized_slot_count =
        std::max(serialized_slot_count, static_cast<InnerIdType>(physical_count));
    Vector<CodeSlotIdType> slots(allocator_);
    slots.resize(serialized_slot_count);
    for (InnerIdType inner_id = 0; inner_id < serialized_slot_count; ++inner_id) {
        slots[inner_id] = inner_to_slot_[inner_id].load(std::memory_order_acquire);
    }
    StreamWriter::WriteVector(writer, slots);
}

void
CodeSlotMap::Deserialize(StreamReader& reader) {
    std::unique_lock lock(mutex_);
    CodeSlotIdType physical_count = 0;
    StreamReader::ReadObj(reader, physical_count);
    Vector<CodeSlotIdType> slots(allocator_);
    StreamReader::ReadVector(reader, slots);
    if (physical_count > slots.size()) {
        throw VsagException(
            ErrorType::INVALID_BINARY,
            fmt::format("invalid code slot mapping: physical count {} exceeds slot count {}",
                        physical_count,
                        slots.size()));
    }
    this->ReleaseSlots();
    this->EnsureLogicalSize(static_cast<InnerIdType>(slots.size()));
    InnerIdType published_logical_count = 0;
    for (InnerIdType inner_id = 0; inner_id < slots.size(); ++inner_id) {
        auto slot = slots[inner_id];
        if (slot != INVALID_CODE_SLOT && slot >= physical_count) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format("invalid code slot mapping: slot {} >= physical count {}",
                            slot,
                            physical_count));
        }
        if (slot != INVALID_CODE_SLOT) {
            ++published_logical_count;
        }
        inner_to_slot_[inner_id].store(slot, std::memory_order_release);
    }
    physical_count_.store(physical_count, std::memory_order_release);
    published_logical_count_.store(published_logical_count, std::memory_order_release);
}

uint64_t
CodeSlotMap::GetMemoryUsage() const {
    std::shared_lock lock(mutex_);
    auto memory = static_cast<uint64_t>(sizeof(*this));
    memory += static_cast<uint64_t>(logical_capacity_) * sizeof(std::atomic<CodeSlotIdType>);
    return memory;
}

void
CodeSlotMap::EnsureLogicalSize(InnerIdType new_size) {
    if (new_size <= logical_capacity_) {
        return;
    }
    auto old_capacity = logical_capacity_;
    auto* new_slots = static_cast<std::atomic<CodeSlotIdType>*>(
        allocator_->Allocate(static_cast<uint64_t>(new_size) *
                             static_cast<uint64_t>(sizeof(std::atomic<CodeSlotIdType>))));
    if (new_slots == nullptr) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "failed to allocate code slot map");
    }
    for (InnerIdType inner_id = 0; inner_id < new_size; ++inner_id) {
        new (new_slots + inner_id) std::atomic<CodeSlotIdType>(INVALID_CODE_SLOT);
    }
    for (InnerIdType inner_id = 0; inner_id < old_capacity; ++inner_id) {
        new_slots[inner_id].store(inner_to_slot_[inner_id].load(std::memory_order_acquire),
                                  std::memory_order_release);
    }
    this->ReleaseSlots();
    inner_to_slot_ = new_slots;
    logical_capacity_ = new_size;
}

void
CodeSlotMap::ReleaseSlots() {
    if (inner_to_slot_ == nullptr) {
        logical_capacity_ = 0;
        return;
    }
    using AtomicSlot = std::atomic<CodeSlotIdType>;
    for (InnerIdType inner_id = 0; inner_id < logical_capacity_; ++inner_id) {
        inner_to_slot_[inner_id].~AtomicSlot();
    }
    allocator_->Deallocate(inner_to_slot_);
    inner_to_slot_ = nullptr;
    logical_capacity_ = 0;
}

}  // namespace vsag
