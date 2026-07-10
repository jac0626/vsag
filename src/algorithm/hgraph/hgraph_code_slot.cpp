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

#include "hgraph_code_slot.h"

#include <fmt/format.h>

#include <array>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

#include "common.h"
#include "index_common_param.h"

namespace vsag {

namespace {

constexpr InnerIdType INVALID_CODE_SLOT = std::numeric_limits<InnerIdType>::max();

class HGraphCodeSlotAdapter : public FlattenInterface {
public:
    HGraphCodeSlotAdapter(FlattenInterfacePtr physical,
                          std::shared_ptr<const HGraphCodeSlotMap> slot_map,
                          Allocator* allocator,
                          const std::atomic<uint64_t>* logical_total_count)
        : physical_(std::move(physical)),
          slot_map_(std::move(slot_map)),
          allocator_(allocator),
          logical_total_count_(logical_total_count) {
        this->refresh_code_size();
    }

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        this->with_mapped_ids(idx, id_count, ctx, [&](const InnerIdType* mapped_ids) {
            physical_->Query(result_dists, computer, mapped_ids, id_count, ctx);
        });
    }

    void
    QueryWithDistanceFilter(float* result_dists,
                            const ComputerInterfacePtr& computer,
                            const InnerIdType* idx,
                            InnerIdType id_count,
                            float threshold,
                            QueryContext* ctx = nullptr) override {
        this->with_mapped_ids(idx, id_count, ctx, [&](const InnerIdType* mapped_ids) {
            physical_->QueryWithDistanceFilter(
                result_dists, computer, mapped_ids, id_count, threshold, ctx);
        });
    }

    void
    QueryWithDistanceLowerBound(float* result_dists,
                                float* lower_bounds,
                                const ComputerInterfacePtr& computer,
                                const InnerIdType* idx,
                                InnerIdType id_count,
                                QueryContext* ctx = nullptr) override {
        this->with_mapped_ids(idx, id_count, ctx, [&](const InnerIdType* mapped_ids) {
            physical_->QueryWithDistanceLowerBound(
                result_dists, lower_bounds, computer, mapped_ids, id_count, ctx);
        });
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return physical_->FactoryComputer(query);
    }

    void
    Train(const void* data, uint64_t count) override {
        physical_->Train(data, count);
        this->refresh_code_size();
    }

    void
    InsertVector(const void* vector, InnerIdType idx) override {
        physical_->InsertVector(vector, slot_map_->Resolve(idx));
    }

    void
    InsertVectorToSlot(const void* vector, InnerIdType code_slot_id) {
        physical_->InsertVector(vector, code_slot_id);
    }

    bool
    UpdateVector(const void* vector, InnerIdType idx) override {
        return physical_->UpdateVector(vector, slot_map_->Resolve(idx));
    }

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override {
        if (idx_vec == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "HGraph code-slot adapter requires explicit logical ids");
        }
        Vector<InnerIdType> code_slot_ids(static_cast<uint64_t>(count), allocator_);
        for (InnerIdType i = 0; i < count; ++i) {
            code_slot_ids[i] = slot_map_->Resolve(idx_vec[i]);
        }
        physical_->BatchInsertVector(vectors, count, code_slot_ids.data());
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override {
        InnerIdType code_slot_id1 = 0;
        InnerIdType code_slot_id2 = 0;
        slot_map_->ResolvePair(id1, id2, code_slot_id1, code_slot_id2);
        return physical_->ComputePairVectors(code_slot_id1, code_slot_id2);
    }

    void
    Prefetch(InnerIdType id) override {
        physical_->Prefetch(slot_map_->Resolve(id));
    }

    std::string
    GetQuantizerName() override {
        return physical_->GetQuantizerName();
    }

    MetricType
    GetMetricType() override {
        return physical_->GetMetricType();
    }

    void
    Resize(InnerIdType capacity) override {
        (void)capacity;
        this->reject_physical_operation("Resize");
    }

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        auto other_adapter = std::dynamic_pointer_cast<HGraphCodeSlotAdapter>(other);
        if (other_adapter != nullptr) {
            physical_->ExportModel(other_adapter->physical_);
            return;
        }
        physical_->ExportModel(other);
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        physical_->InitIO(io_param);
        this->refresh_code_size();
    }

    uint64_t
    GetMemoryUsage() const override {
        // The adapter does not own the slot map. HGraph accounts for it once, separately from
        // every physical flatten that shares the same map.
        return physical_->GetMemoryUsage();
    }

    IndexCommonParam
    ExportCommonParam() override {
        return physical_->ExportCommonParam();
    }

    bool
    SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params) override {
        auto ret = physical_->SetRuntimeParameters(new_params);
        this->refresh_code_size();
        return ret;
    }

    bool
    Decode(const uint8_t* codes, float* vector) override {
        return physical_->Decode(codes, vector);
    }

    bool
    Encode(const float* vector, uint8_t* codes) override {
        return physical_->Encode(vector, codes);
    }

    const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override {
        return physical_->GetCodesById(slot_map_->Resolve(id), need_release);
    }

    void
    Release(const uint8_t* data) const override {
        physical_->Release(data);
    }

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override {
        return physical_->GetCodesById(slot_map_->Resolve(id), codes);
    }

    InnerIdType
    TotalCount() const override {
        // Callers of the adapter operate in HGraph's logical inner-id space.
        return static_cast<InnerIdType>(logical_total_count_->load(std::memory_order_acquire));
    }

    void
    Serialize(StreamWriter& writer) override {
        physical_->Serialize(writer);
    }

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override {
        physical_->Deserialize(std::move(reader));
        this->refresh_code_size();
    }

    bool
    InMemory() const override {
        return physical_->InMemory();
    }

    bool
    HoldMolds() const override {
        return physical_->HoldMolds();
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override {
        (void)other;
        (void)bias;
        this->reject_physical_operation("MergeOther");
    }

    void
    Move(InnerIdType from, InnerIdType to) override {
        (void)from;
        (void)to;
        this->reject_physical_operation("Move");
    }

    void
    ShrinkToFit(InnerIdType capacity) override {
        (void)capacity;
        this->reject_physical_operation("ShrinkToFit");
    }

    [[nodiscard]] FlattenInterfacePtr
    PhysicalFlatten() const {
        return this->physical_;
    }

private:
    [[noreturn]] void
    reject_physical_operation(const char* operation) const {
        throw VsagException(
            ErrorType::INTERNAL_ERROR,
            fmt::format("{} requires an explicit physical HGraph flatten", operation));
    }

    void
    refresh_code_size() {
        this->code_size_ = physical_->code_size_;
    }

    template <typename QueryFunc>
    void
    with_mapped_ids(const InnerIdType* idx,
                    InnerIdType id_count,
                    QueryContext* ctx,
                    QueryFunc&& query_func) const {
        if (id_count == 0) {
            query_func(idx);
            return;
        }
        if (id_count == 1) {
            InnerIdType mapped_id = 0;
            mapped_id = slot_map_->Resolve(idx[0]);
            query_func(&mapped_id);
            return;
        }
        constexpr InnerIdType stack_mapped_id_count = 128;
        if (id_count <= stack_mapped_id_count) {
            std::array<InnerIdType, stack_mapped_id_count> mapped_ids{};
            slot_map_->ResolveBatch(idx, id_count, mapped_ids.data());
            query_func(mapped_ids.data());
            return;
        }
        Allocator* allocator = select_query_allocator(ctx, allocator_);
        Vector<InnerIdType> mapped_ids(static_cast<uint64_t>(id_count), allocator);
        slot_map_->ResolveBatch(idx, id_count, mapped_ids.data());
        query_func(mapped_ids.data());
    }

    FlattenInterfacePtr physical_{nullptr};
    std::shared_ptr<const HGraphCodeSlotMap> slot_map_{nullptr};
    Allocator* allocator_{nullptr};
    const std::atomic<uint64_t>* logical_total_count_{nullptr};
};

}  // namespace

FlattenInterfacePtr
MakeHGraphCodeSlotAdapter(FlattenInterfacePtr physical,
                          std::shared_ptr<const HGraphCodeSlotMap> slot_map,
                          Allocator* allocator,
                          const std::atomic<uint64_t>* logical_total_count) {
    return std::make_shared<HGraphCodeSlotAdapter>(
        std::move(physical), std::move(slot_map), allocator, logical_total_count);
}

FlattenInterfacePtr
GetHGraphPhysicalFlatten(const FlattenInterfacePtr& flatten) {
    auto adapter = std::dynamic_pointer_cast<HGraphCodeSlotAdapter>(flatten);
    return adapter == nullptr ? flatten : adapter->PhysicalFlatten();
}

void
InsertVectorToHGraphPhysicalSlot(const FlattenInterfacePtr& flatten,
                                 const void* vector,
                                 InnerIdType code_slot_id) {
    auto adapter = std::dynamic_pointer_cast<HGraphCodeSlotAdapter>(flatten);
    if (adapter != nullptr) {
        adapter->InsertVectorToSlot(vector, code_slot_id);
        return;
    }
    flatten->InsertVector(vector, code_slot_id);
}

HGraphCodeSlotMap::HGraphCodeSlotMap(Allocator* allocator) : allocator_(allocator) {
}

HGraphCodeSlotMap::~HGraphCodeSlotMap() {
    this->ReleaseSlots();
}

InnerIdType
HGraphCodeSlotMap::AllocateSlot() {
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
HGraphCodeSlotMap::PublishSlot(InnerIdType inner_id, InnerIdType code_slot_id) {
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

void
HGraphCodeSlotMap::RebindSlot(InnerIdType inner_id, InnerIdType code_slot_id) {
    auto physical_count = physical_count_.load(std::memory_order_acquire);
    CHECK_ARGUMENT(
        code_slot_id < physical_count,
        fmt::format(
            "code_slot_id({}) must be less than physical_count({})", code_slot_id, physical_count));
    if (inner_id >= logical_capacity_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("inner_id({}) has no reserved code slot", inner_id));
    }
    auto old_slot = inner_to_slot_[inner_id].load(std::memory_order_acquire);
    while (old_slot != INVALID_CODE_SLOT &&
           not inner_to_slot_[inner_id].compare_exchange_weak(
               old_slot, code_slot_id, std::memory_order_release, std::memory_order_acquire)) {
    }
    CHECK_ARGUMENT(old_slot != INVALID_CODE_SLOT,
                   fmt::format("inner_id({}) has no bound code slot", inner_id));
}

void
HGraphCodeSlotMap::RemoveLogical(InnerIdType inner_id) {
    if (inner_id >= logical_capacity_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("inner_id({}) has no reserved code slot", inner_id));
    }
    auto old_slot = inner_to_slot_[inner_id].exchange(INVALID_CODE_SLOT, std::memory_order_acq_rel);
    CHECK_ARGUMENT(old_slot != INVALID_CODE_SLOT,
                   fmt::format("inner_id({}) has no bound code slot", inner_id));
    published_logical_count_.fetch_sub(1, std::memory_order_acq_rel);
}

void
HGraphCodeSlotMap::MoveLogical(InnerIdType from, InnerIdType to) {
    if (from == to) {
        this->RemoveLogical(to);
        return;
    }
    if (from >= logical_capacity_ || to >= logical_capacity_) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("cannot move code slot from inner_id({}) to inner_id({})", from, to));
    }

    auto from_slot = inner_to_slot_[from].exchange(INVALID_CODE_SLOT, std::memory_order_acq_rel);
    CHECK_ARGUMENT(from_slot != INVALID_CODE_SLOT,
                   fmt::format("inner_id({}) has no bound code slot", from));
    auto to_slot = inner_to_slot_[to].load(std::memory_order_acquire);
    CHECK_ARGUMENT(to_slot != INVALID_CODE_SLOT,
                   fmt::format("inner_id({}) has no bound code slot", to));
    inner_to_slot_[to].store(from_slot, std::memory_order_release);
    published_logical_count_.fetch_sub(1, std::memory_order_acq_rel);
}

void
HGraphCodeSlotMap::CompactPhysicalSlotsAfter(InnerIdType removed_slot) {
    auto physical_count = physical_count_.load(std::memory_order_acquire);
    CHECK_ARGUMENT(
        removed_slot < physical_count,
        fmt::format(
            "removed_slot({}) must be less than physical_count({})", removed_slot, physical_count));
    for (InnerIdType inner_id = 0; inner_id < logical_capacity_; ++inner_id) {
        auto slot = inner_to_slot_[inner_id].load(std::memory_order_acquire);
        if (slot != INVALID_CODE_SLOT && slot > removed_slot) {
            inner_to_slot_[inner_id].store(slot - 1, std::memory_order_release);
        }
    }
    physical_count_.store(physical_count - 1, std::memory_order_release);
}

InnerIdType
HGraphCodeSlotMap::Resolve(InnerIdType inner_id) const {
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
HGraphCodeSlotMap::ResolvePair(InnerIdType inner_id1,
                               InnerIdType inner_id2,
                               InnerIdType& code_slot_id1,
                               InnerIdType& code_slot_id2) const {
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
HGraphCodeSlotMap::ResolveBatch(const InnerIdType* inner_ids,
                                InnerIdType count,
                                InnerIdType* code_slot_ids) const {
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
HGraphCodeSlotMap::ReserveLogicalSize(InnerIdType new_size) {
    std::unique_lock lock(mutex_);
    this->EnsureLogicalSize(new_size);
}

void
HGraphCodeSlotMap::MergeOther(const HGraphCodeSlotMap& other,
                              InnerIdType logical_bias,
                              InnerIdType physical_bias) {
    if (this == &other) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "code slot map cannot merge itself");
    }

    auto other_logical_count = other.PublishedLogicalCount();
    auto other_physical_count = other.PhysicalCount();
    auto current_physical_count = this->PhysicalCount();
    CHECK_ARGUMENT(current_physical_count == physical_bias,
                   fmt::format("code slot physical bias({}) must match physical count({})",
                               physical_bias,
                               current_physical_count));

    this->ReserveLogicalSize(logical_bias + other_logical_count);
    for (InnerIdType slot_id = 0; slot_id < other_physical_count; ++slot_id) {
        auto allocated_slot = this->AllocateSlot();
        CHECK_ARGUMENT(allocated_slot == physical_bias + slot_id,
                       fmt::format("allocated code slot({}) must match expected slot({})",
                                   allocated_slot,
                                   physical_bias + slot_id));
    }
    for (InnerIdType inner_id = 0; inner_id < other_logical_count; ++inner_id) {
        auto code_slot_id = other.Resolve(inner_id);
        this->PublishSlot(logical_bias + inner_id, physical_bias + code_slot_id);
    }
}

InnerIdType
HGraphCodeSlotMap::PhysicalCount() const {
    return physical_count_.load(std::memory_order_acquire);
}

InnerIdType
HGraphCodeSlotMap::PublishedLogicalCount() const {
    return published_logical_count_.load(std::memory_order_acquire);
}

void
HGraphCodeSlotMap::Clear() {
    std::unique_lock lock(mutex_);
    this->ReleaseSlots();
    physical_count_.store(0, std::memory_order_release);
    published_logical_count_.store(0, std::memory_order_release);
}

void
HGraphCodeSlotMap::Serialize(StreamWriter& writer) const {
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
    Vector<InnerIdType> slots(allocator_);
    slots.resize(serialized_slot_count);
    for (InnerIdType inner_id = 0; inner_id < serialized_slot_count; ++inner_id) {
        slots[inner_id] = inner_to_slot_[inner_id].load(std::memory_order_acquire);
    }
    StreamWriter::WriteVector(writer, slots);
}

void
HGraphCodeSlotMap::Deserialize(StreamReader& reader) {
    std::unique_lock lock(mutex_);
    InnerIdType physical_count = 0;
    StreamReader::ReadObj(reader, physical_count);
    Vector<InnerIdType> slots(allocator_);
    StreamReader::ReadVector(reader, slots);
    this->ReleaseSlots();
    this->EnsureLogicalSize(static_cast<InnerIdType>(slots.size()));
    InnerIdType published_logical_count = 0;
    for (InnerIdType inner_id = 0; inner_id < slots.size(); ++inner_id) {
        auto slot = slots[inner_id];
        if (slot != INVALID_CODE_SLOT && slot >= physical_count) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
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
HGraphCodeSlotMap::GetMemoryUsage() const {
    std::shared_lock lock(mutex_);
    auto memory = static_cast<uint64_t>(sizeof(*this));
    memory += static_cast<uint64_t>(logical_capacity_) * sizeof(std::atomic<InnerIdType>);
    return memory;
}

void
HGraphCodeSlotMap::EnsureLogicalSize(InnerIdType new_size) {
    if (new_size <= logical_capacity_) {
        return;
    }
    auto old_capacity = logical_capacity_;
    auto* new_slots = static_cast<std::atomic<InnerIdType>*>(allocator_->Allocate(
        static_cast<uint64_t>(new_size) * static_cast<uint64_t>(sizeof(std::atomic<InnerIdType>))));
    if (new_slots == nullptr) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "failed to allocate HGraph code slot map");
    }
    for (InnerIdType inner_id = 0; inner_id < new_size; ++inner_id) {
        new (new_slots + inner_id) std::atomic<InnerIdType>(INVALID_CODE_SLOT);
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
HGraphCodeSlotMap::ReleaseSlots() {
    if (inner_to_slot_ == nullptr) {
        logical_capacity_ = 0;
        return;
    }
    using AtomicSlot = std::atomic<InnerIdType>;
    for (InnerIdType inner_id = 0; inner_id < logical_capacity_; ++inner_id) {
        inner_to_slot_[inner_id].~AtomicSlot();
    }
    allocator_->Deallocate(inner_to_slot_);
    inner_to_slot_ = nullptr;
    logical_capacity_ = 0;
}

}  // namespace vsag
