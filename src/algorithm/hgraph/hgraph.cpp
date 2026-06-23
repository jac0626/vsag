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

#include "hgraph.h"

#include <datacell/compressed_graph_datacell_parameter.h>
#include <fmt/format.h>

#include <atomic>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "algorithm/inner_index_interface.h"
#include "analyzer/analyzer.h"
#include "attr/argparse.h"
#include "common.h"
#include "datacell/flatten_interface.h"
#include "datacell/sparse_graph_datacell.h"
#include "dataset_impl.h"
#include "impl/filter/filter_headers.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "impl/reasoning/search_reasoning.h"
#include "impl/reorder/flatten_reorder.h"
#include "index/index_impl.h"
#include "index/iterator_filter.h"
#include "io/reader_io_parameter.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/util_functions.h"
#include "utils/visited_list.h"
#include "vsag/options.h"

namespace vsag {

namespace {

constexpr InnerIdType INVALID_CODE_SLOT = std::numeric_limits<InnerIdType>::max();

class HGraphLogicalCodeView : public FlattenInterface {
public:
    HGraphLogicalCodeView(FlattenInterfacePtr base,
                          std::shared_ptr<const HGraphCodeSlotMapping> mapping,
                          Allocator* allocator,
                          const std::atomic<uint64_t>* logical_total_count,
                          std::shared_mutex* storage_mutex)
        : base_(std::move(base)),
          mapping_(std::move(mapping)),
          allocator_(allocator),
          logical_total_count_(logical_total_count),
          storage_mutex_(storage_mutex) {
        this->code_size_ = base_->code_size_;
        this->max_capacity_ = base_->max_capacity_;
        this->total_count_ = base_->total_count_;
        this->prefetch_stride_code_ = base_->prefetch_stride_code_;
        this->prefetch_depth_code_ = base_->prefetch_depth_code_;
    }

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          QueryContext* ctx = nullptr) override {
        this->with_mapped_ids(idx, id_count, ctx, [&](const InnerIdType* mapped_ids) {
            base_->Query(result_dists, computer, mapped_ids, id_count, ctx);
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
            base_->QueryWithDistanceFilter(
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
            base_->QueryWithDistanceLowerBound(
                result_dists, lower_bounds, computer, mapped_ids, id_count, ctx);
        });
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        return base_->FactoryComputer(query);
    }

    void
    Train(const void* data, uint64_t count) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support Train");
    }

    void
    InsertVector(const void* vector, InnerIdType idx) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support InsertVector");
    }

    bool
    UpdateVector(const void* vector, InnerIdType idx) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support UpdateVector");
    }

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support BatchInsertVector");
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override {
        std::shared_lock storage_lock(*storage_mutex_);
        return base_->ComputePairVectors(mapping_->Resolve(id1), mapping_->Resolve(id2));
    }

    bool
    CompareVector(const void* vector, InnerIdType id) override {
        std::shared_lock storage_lock(*storage_mutex_);
        return base_->CompareVector(vector, mapping_->Resolve(id));
    }

    void
    Prefetch(InnerIdType id) override {
        std::shared_lock storage_lock(*storage_mutex_);
        base_->Prefetch(mapping_->Resolve(id));
    }

    std::string
    GetQuantizerName() override {
        return base_->GetQuantizerName();
    }

    MetricType
    GetMetricType() override {
        return base_->GetMetricType();
    }

    void
    Resize(InnerIdType capacity) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support Resize");
    }

    void
    ExportModel(const FlattenInterfacePtr& other) const override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support ExportModel");
    }

    void
    InitIO(const IOParamPtr& io_param) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support InitIO");
    }

    int64_t
    GetMemoryUsage() const override {
        return base_->GetMemoryUsage();
    }

    IndexCommonParam
    ExportCommonParam() override {
        return base_->ExportCommonParam();
    }

    bool
    SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params) override {
        bool ret = base_->SetRuntimeParameters(new_params);
        this->prefetch_stride_code_ = base_->prefetch_stride_code_;
        this->prefetch_depth_code_ = base_->prefetch_depth_code_;
        return ret;
    }

    bool
    Decode(const uint8_t* codes, float* vector) override {
        return base_->Decode(codes, vector);
    }

    bool
    Encode(const float* vector, uint8_t* codes) override {
        return base_->Encode(vector, codes);
    }

    const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override {
        std::shared_lock storage_lock(*storage_mutex_);
        auto* codes = static_cast<uint8_t*>(allocator_->Allocate(this->code_size_));
        if (codes == nullptr || not base_->GetCodesById(mapping_->Resolve(id), codes)) {
            if (codes != nullptr) {
                allocator_->Deallocate(codes);
            }
            need_release = false;
            return nullptr;
        }
        need_release = true;
        return codes;
    }

    void
    Release(const uint8_t* data) const override {
        allocator_->Deallocate(const_cast<uint8_t*>(data));
    }

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override {
        std::shared_lock storage_lock(*storage_mutex_);
        return base_->GetCodesById(mapping_->Resolve(id), codes);
    }

    InnerIdType
    TotalCount() const override {
        return static_cast<InnerIdType>(logical_total_count_->load(std::memory_order_acquire));
    }

    void
    Serialize(StreamWriter& writer) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support Serialize");
    }

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support Deserialize");
    }

    bool
    InMemory() const override {
        return base_->InMemory();
    }

    bool
    HoldMolds() const override {
        return base_->HoldMolds();
    }

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support MergeOther");
    }

    void
    Move(InnerIdType from, InnerIdType to) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support Move");
    }

    void
    ShrinkToFit(InnerIdType capacity) override {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "HGraph logical code view does not support ShrinkToFit");
    }

private:
    template <typename QueryFunc>
    void
    with_mapped_ids(const InnerIdType* idx,
                    InnerIdType id_count,
                    QueryContext* ctx,
                    QueryFunc&& query_func) const {
        std::shared_lock storage_lock(*storage_mutex_);
        if (id_count == 0) {
            query_func(idx);
            return;
        }
        if (id_count == 1) {
            InnerIdType mapped_id = mapping_->Resolve(idx[0]);
            query_func(&mapped_id);
            return;
        }
        Allocator* allocator = select_query_allocator(ctx, allocator_);
        Vector<InnerIdType> mapped_ids(static_cast<uint64_t>(id_count), allocator);
        mapping_->ResolveBatch(idx, mapped_ids.data(), id_count);
        query_func(mapped_ids.data());
    }

    FlattenInterfacePtr base_{nullptr};
    std::shared_ptr<const HGraphCodeSlotMapping> mapping_{nullptr};
    Allocator* allocator_{nullptr};
    const std::atomic<uint64_t>* logical_total_count_{nullptr};
    std::shared_mutex* storage_mutex_{nullptr};
};

}  // namespace

HGraphCodeSlotMapping::HGraphCodeSlotMapping(Allocator* allocator)
    : allocator_(allocator), logical_to_code_ids_(allocator), code_slot_ref_counts_(allocator) {
}

void
HGraphCodeSlotMapping::ResizeLogical(InnerIdType new_size) {
    std::unique_lock lock(mutex_);
    this->ResizeLogicalNoLock(new_size);
}

void
HGraphCodeSlotMapping::ResizeLogicalNoLock(InnerIdType new_size) {
    if (new_size <= logical_to_code_ids_.size()) {
        return;
    }
    auto old_size = logical_to_code_ids_.size();
    logical_to_code_ids_.resize(new_size);
    for (uint64_t i = old_size; i < logical_to_code_ids_.size(); ++i) {
        logical_to_code_ids_[i] = INVALID_CODE_SLOT;
    }
}

void
HGraphCodeSlotMapping::CommitNewSlot(InnerIdType inner_id, InnerIdType code_slot_id) {
    std::unique_lock lock(mutex_);
    this->ResizeLogicalNoLock(inner_id + 1);
    if (code_slot_id != physical_count_) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("invalid HGraph code slot commit: expected {}, got {}",
                                        physical_count_,
                                        code_slot_id));
    }
    this->DecreaseSlotRefNoLock(logical_to_code_ids_[inner_id]);
    ++physical_count_;
    code_slot_ref_counts_.push_back(1);
    logical_to_code_ids_[inner_id] = code_slot_id;
}

void
HGraphCodeSlotMapping::BindExistingSlot(InnerIdType inner_id, InnerIdType code_slot_id) {
    std::unique_lock lock(mutex_);
    this->ResizeLogicalNoLock(inner_id + 1);
    auto old_slot = logical_to_code_ids_[inner_id];
    if (old_slot == code_slot_id) {
        return;
    }
    if (code_slot_id >= code_slot_ref_counts_.size() || code_slot_ref_counts_[code_slot_id] == 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("invalid HGraph code slot alias: {}", code_slot_id));
    }
    this->DecreaseSlotRefNoLock(old_slot);
    ++code_slot_ref_counts_[code_slot_id];
    logical_to_code_ids_[inner_id] = code_slot_id;
}

InnerIdType
HGraphCodeSlotMapping::Resolve(InnerIdType inner_id) const {
    std::shared_lock lock(mutex_);
    return this->ResolveNoLock(inner_id);
}

void
HGraphCodeSlotMapping::ResolveBatch(const InnerIdType* inner_ids,
                                    InnerIdType* code_slot_ids,
                                    InnerIdType count) const {
    std::shared_lock lock(mutex_);
    for (InnerIdType i = 0; i < count; ++i) {
        code_slot_ids[i] = this->ResolveNoLock(inner_ids[i]);
    }
}

InnerIdType
HGraphCodeSlotMapping::PhysicalCount() const {
    std::shared_lock lock(mutex_);
    return physical_count_;
}

InnerIdType
HGraphCodeSlotMapping::LogicalCount() const {
    std::shared_lock lock(mutex_);
    return static_cast<InnerIdType>(logical_to_code_ids_.size());
}

bool
HGraphCodeSlotMapping::IsMapped(InnerIdType inner_id) const {
    std::shared_lock lock(mutex_);
    return inner_id < logical_to_code_ids_.size() &&
           logical_to_code_ids_[inner_id] != INVALID_CODE_SLOT;
}

bool
HGraphCodeSlotMapping::IsSharedSlot(InnerIdType inner_id) const {
    std::shared_lock lock(mutex_);
    if (inner_id >= logical_to_code_ids_.size()) {
        return false;
    }
    auto code_slot_id = logical_to_code_ids_[inner_id];
    if (code_slot_id == INVALID_CODE_SLOT || code_slot_id >= code_slot_ref_counts_.size()) {
        return false;
    }
    return code_slot_ref_counts_[code_slot_id] > 1;
}

void
HGraphCodeSlotMapping::Serialize(StreamWriter& writer, InnerIdType logical_count) const {
    std::shared_lock lock(mutex_);
    StreamWriter::WriteObj(writer, logical_count);
    StreamWriter::WriteObj(writer, physical_count_);
    Vector<InnerIdType> logical_to_code_ids(logical_count, allocator_);
    for (InnerIdType i = 0; i < logical_count; ++i) {
        logical_to_code_ids[i] =
            i < logical_to_code_ids_.size() ? logical_to_code_ids_[i] : INVALID_CODE_SLOT;
    }
    StreamWriter::WriteVector(writer, logical_to_code_ids);
}

void
HGraphCodeSlotMapping::Deserialize(StreamReader& reader) {
    std::unique_lock lock(mutex_);
    InnerIdType logical_count = 0;
    StreamReader::ReadObj(reader, logical_count);
    StreamReader::ReadObj(reader, physical_count_);
    StreamReader::ReadVector(reader, logical_to_code_ids_);
    if (logical_to_code_ids_.size() != logical_count) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "corrupted HGraph code-slot mapping logical count");
    }
    code_slot_ref_counts_.resize(physical_count_);
    for (auto& code_slot_ref_count : code_slot_ref_counts_) {
        code_slot_ref_count = 0;
    }
    for (const auto code_slot_id : logical_to_code_ids_) {
        if (code_slot_id == INVALID_CODE_SLOT) {
            continue;
        }
        if (code_slot_id >= code_slot_ref_counts_.size()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "corrupted HGraph code-slot mapping slot id");
        }
        ++code_slot_ref_counts_[code_slot_id];
    }
}

void
HGraphCodeSlotMapping::InitIdentity(InnerIdType logical_count) {
    std::unique_lock lock(mutex_);
    physical_count_ = logical_count;
    logical_to_code_ids_.resize(logical_count);
    code_slot_ref_counts_.resize(logical_count);
    for (InnerIdType i = 0; i < logical_count; ++i) {
        logical_to_code_ids_[i] = i;
        code_slot_ref_counts_[i] = 1;
    }
}

int64_t
HGraphCodeSlotMapping::GetMemoryUsage() const {
    std::shared_lock lock(mutex_);
    return static_cast<int64_t>(sizeof(HGraphCodeSlotMapping)) +
           static_cast<int64_t>(logical_to_code_ids_.size() * sizeof(InnerIdType)) +
           static_cast<int64_t>(code_slot_ref_counts_.size() * sizeof(uint32_t));
}

InnerIdType
HGraphCodeSlotMapping::ResolveNoLock(InnerIdType inner_id) const {
    if (inner_id >= logical_to_code_ids_.size()) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("HGraph logical id {} is out of range", inner_id));
    }
    auto code_slot_id = logical_to_code_ids_[inner_id];
    if (code_slot_id == INVALID_CODE_SLOT) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("HGraph logical id {} has no code slot", inner_id));
    }
    return code_slot_id;
}

void
HGraphCodeSlotMapping::DecreaseSlotRefNoLock(InnerIdType code_slot_id) {
    if (code_slot_id == INVALID_CODE_SLOT || code_slot_id >= code_slot_ref_counts_.size() ||
        code_slot_ref_counts_[code_slot_id] == 0) {
        return;
    }
    --code_slot_ref_counts_[code_slot_id];
}

class HGraphAnalyzer;

HGraph::HGraph(const HGraphParameterPtr& hgraph_param, const vsag::IndexCommonParam& common_param)
    : InnerIndexInterface(hgraph_param, common_param),
      route_graphs_(common_param.allocator_.get()),
      cache_(std::make_unique<HGraphCache>(common_param.allocator_.get())),
      use_elp_optimizer_(hgraph_param->use_elp_optimizer),
      ignore_reorder_(hgraph_param->ignore_reorder),
      build_by_base_(hgraph_param->build_by_base),
      reorder_by_base_(hgraph_param->reorder_source == HGRAPH_REORDER_SOURCE_BASE),
      ef_construct_(hgraph_param->ef_construction),
      alpha_(hgraph_param->alpha),
      duplicate_distance_threshold_(hgraph_param->duplicate_distance_threshold),
      support_force_remove_(hgraph_param->support_force_remove),
      odescent_param_(hgraph_param->odescent_param),
      graph_type_(hgraph_param->graph_type),
      hierarchical_datacell_param_(hgraph_param->hierarchical_graph_param),
      use_old_serial_format_(common_param.use_old_serial_format_) {
    this->support_duplicate_ = hgraph_param->support_duplicate;
    this->persist_source_id_ = hgraph_param->persist_source_id;
    neighbors_mutex_ = std::make_shared<PointsMutex>(0, common_param.allocator_.get());
    this->basic_flatten_codes_ =
        FlattenInterface::MakeInstance(hgraph_param->base_codes_param, common_param);
    if (has_precise_reorder()) {
        this->high_precise_codes_ =
            FlattenInterface::MakeInstance(hgraph_param->precise_codes_param, common_param);
    }
    this->searcher_ = std::make_shared<BasicSearcher>(common_param, neighbors_mutex_);

    this->bottom_graph_ =
        GraphInterface::MakeInstance(hgraph_param->bottom_graph_param, common_param);
    if (this->support_duplicate_) {
        this->label_table_->SetDuplicateTracker(this->bottom_graph_->GetDuplicateTracker());
    }
    mult_ = 1 / log(1.0 * static_cast<double>(this->bottom_graph_->MaximumDegree()));

    this->parallel_searcher_ =
        std::make_shared<ParallelSearcher>(common_param, thread_pool_, neighbors_mutex_);

    UnorderedMap<std::string, float> default_param(common_param.allocator_.get());
    default_param.insert(
        {PREFETCH_DEPTH_CODE, (this->basic_flatten_codes_->code_size_ + 63.0) / 64.0});
    this->basic_flatten_codes_->SetRuntimeParameters(default_param);

    if (use_elp_optimizer_) {
        optimizer_ = std::make_shared<Optimizer<BasicSearcher>>(common_param);
    }
    check_and_init_raw_vector(hgraph_param->raw_vector_param, common_param);
    this->init_code_slot_mapping();
    this->refresh_code_read_views();
    init_resize_bit_and_reorder();
    resize(bottom_graph_->max_capacity_);
}

void
HGraph::init_code_slot_mapping() {
    if (not this->support_duplicate_ || this->use_old_serial_format_) {
        return;
    }
    this->code_slot_mapping_ = std::make_shared<HGraphCodeSlotMapping>(this->allocator_);
}

void
HGraph::refresh_code_read_views() {
    if (not this->has_code_slot_mapping()) {
        this->basic_read_codes_ = this->basic_flatten_codes_;
        this->precise_read_codes_ = this->high_precise_codes_;
        this->raw_read_codes_ = this->raw_vector_;
        return;
    }
    this->basic_read_codes_ =
        std::make_shared<HGraphLogicalCodeView>(this->basic_flatten_codes_,
                                                this->code_slot_mapping_,
                                                this->allocator_,
                                                &this->total_count_,
                                                &this->code_slot_storage_mutex_);
    if (this->high_precise_codes_ != nullptr) {
        this->precise_read_codes_ =
            std::make_shared<HGraphLogicalCodeView>(this->high_precise_codes_,
                                                    this->code_slot_mapping_,
                                                    this->allocator_,
                                                    &this->total_count_,
                                                    &this->code_slot_storage_mutex_);
    } else {
        this->precise_read_codes_.reset();
    }
    if (this->raw_vector_ == nullptr) {
        this->raw_read_codes_.reset();
    } else if (this->raw_vector_ == this->basic_flatten_codes_) {
        this->raw_read_codes_ = this->basic_read_codes_;
    } else if (this->raw_vector_ == this->high_precise_codes_) {
        this->raw_read_codes_ = this->precise_read_codes_;
    } else {
        this->raw_read_codes_ =
            std::make_shared<HGraphLogicalCodeView>(this->raw_vector_,
                                                    this->code_slot_mapping_,
                                                    this->allocator_,
                                                    &this->total_count_,
                                                    &this->code_slot_storage_mutex_);
    }
}

InnerIdType
HGraph::resolve_code_slot_id(InnerIdType inner_id) const {
    if (not this->has_code_slot_mapping()) {
        return inner_id;
    }
    return this->code_slot_mapping_->Resolve(inner_id);
}

void
HGraph::resize_persistent_code_storage_unlocked(InnerIdType new_size) {
    auto current_capacity = this->basic_flatten_codes_->max_capacity_;
    if (new_size <= current_capacity) {
        return;
    }
    auto new_capacity = static_cast<InnerIdType>(
        next_multiple_of_power_of_two(new_size, this->resize_increase_count_bit_));
    this->basic_flatten_codes_->Resize(new_capacity);
    if (has_precise_reorder()) {
        this->high_precise_codes_->Resize(new_capacity);
    }
    if (create_new_raw_vector_) {
        this->raw_vector_->Resize(new_capacity);
    }
}

void
HGraph::write_persistent_codes_unlocked(const void* data, InnerIdType code_slot_id) {
    this->resize_persistent_code_storage_unlocked(code_slot_id + 1);
    this->basic_flatten_codes_->InsertVector(data, code_slot_id);
    if (has_precise_reorder()) {
        this->high_precise_codes_->InsertVector(data, code_slot_id);
    }
    if (create_new_raw_vector_) {
        this->raw_vector_->InsertVector(data, code_slot_id);
    }
}

void
HGraph::shrink_persistent_code_storage() const {
    if (not this->has_code_slot_mapping()) {
        return;
    }
    std::scoped_lock<std::shared_mutex> global_lock(this->global_mutex_);
    std::unique_lock lock(this->code_slot_storage_mutex_);
    auto physical_count = this->code_slot_mapping_->PhysicalCount();
    this->basic_flatten_codes_->ShrinkToFit(physical_count);
    if (has_precise_reorder()) {
        this->high_precise_codes_->ShrinkToFit(physical_count);
    }
    if (create_new_raw_vector_) {
        this->raw_vector_->ShrinkToFit(physical_count);
    }
}

bool
HGraph::Tune(const std::string& parameters, bool disable_future_tuning) {
    if (not this->index_feature_list_->CheckFeature(IndexFeature::SUPPORT_TUNE) or
        not this->has_raw_vector_) {
        return false;
    }

    // parse
    auto parsed_params = JsonType::Parse(parameters);
    JsonType hgraph_json;
    if (parsed_params.Contains(INDEX_PARAM)) {
        hgraph_json = parsed_params[INDEX_PARAM];
    }

    // map
    auto inner_json = map_hgraph_param(hgraph_json);

    // construct param obj
    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->FromJson(inner_json);
    auto inner_parameter = std::make_shared<InnerIndexParameter>();
    inner_parameter->FromJson(inner_json);

    // init new_basic_code obj
    auto common_param = this->basic_flatten_codes_->ExportCommonParam();
    auto new_basic_code =
        FlattenInterface::MakeInstance(hgraph_parameter->base_codes_param, common_param);
    FlattenInterfacePtr new_precise_code;
    bool new_reorder_by_base = inner_parameter->reorder_source == HGRAPH_REORDER_SOURCE_BASE;
    if (inner_parameter->use_reorder && not new_reorder_by_base) {
        new_precise_code =
            FlattenInterface::MakeInstance(hgraph_parameter->precise_codes_param, common_param);
    }

    std::scoped_lock lock(this->add_mutex_);
    if (this->immutable_.load(std::memory_order_acquire)) {
        return false;
    }

    // check which code need to tune and update create_param_ptr_
    bool is_tune_base_code = false;
    bool is_tune_precise_code = false;
    bool new_use_reorder = use_reorder_;
    bool drop_precise_codes = false;
    auto param = std::dynamic_pointer_cast<HGraphParameter>(create_param_ptr_);
    if (basic_flatten_codes_->GetQuantizerName() != new_basic_code->GetQuantizerName()) {
        // [case 1] base_code is not same
        is_tune_base_code = true;
    }
    if (has_precise_reorder() and inner_parameter->use_reorder and not new_reorder_by_base and
        this->high_precise_codes_->GetQuantizerName() != new_precise_code->GetQuantizerName()) {
        // [case 2] precise code is not same
        is_tune_precise_code = true;
    }
    if (not inner_parameter->use_reorder or new_reorder_by_base) {
        // [case 3] drop precise_code
        new_use_reorder = inner_parameter->use_reorder;
        drop_precise_codes = true;
        param->precise_codes_param.reset();
        is_tune_precise_code = false;
    }
    if (not new_use_reorder and inner_parameter->use_reorder and not new_reorder_by_base) {
        // [case 4] assign new precise_code
        new_use_reorder = true;
        is_tune_precise_code = true;
    }

    // update create_param_ptr_
    if (is_tune_base_code) {
        param->base_codes_param = hgraph_parameter->base_codes_param;
    }
    if (is_tune_precise_code) {
        param->precise_codes_param = hgraph_parameter->precise_codes_param;
    }
    param->use_reorder = new_use_reorder;
    param->reorder_source = inner_parameter->reorder_source;

    // export train data and train new_basic_code
    auto train_count = std::min(this->train_sample_count_, this->GetNumElements());
    Vector<float> train_data(train_count * dim_, 0, allocator_);
    if (is_tune_base_code or is_tune_precise_code) {
        for (InnerIdType i = 0; i < train_count; i++) {
            this->GetVectorByInnerId(i, (train_data.data() + i * dim_));
        }
    }

    auto tune_and_rebuild = [&](bool need_tune,
                                FlattenInterfacePtr old_code,
                                FlattenInterfacePtr new_code) {
        if (not need_tune) {
            return old_code;
        }

        new_code->Train(train_data.data(), train_count);

        Vector<float> insert_buffer(dim_, 0, allocator_);
        UnorderedSet<InnerIdType> inserted_code_slots(this->allocator_);
        for (int64_t i = 0; i < total_count_; ++i) {
            auto code_slot_id = this->resolve_code_slot_id(static_cast<InnerIdType>(i));
            if (this->has_code_slot_mapping() &&
                not inserted_code_slots.emplace(code_slot_id).second) {
                continue;
            }
            GetVectorByInnerId(i, insert_buffer.data());
            new_code->InsertVector(static_cast<const void*>(insert_buffer.data()), code_slot_id);
        }
        return new_code;
    };

    auto new_basic = tune_and_rebuild(is_tune_base_code, basic_flatten_codes_, new_basic_code);
    auto new_precise =
        tune_and_rebuild(is_tune_precise_code, high_precise_codes_, new_precise_code);

    // Acquire exclusive global lock to atomically swap flatten codes,
    // preventing concurrent searches from accessing partially updated state.
    {
        std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
        basic_flatten_codes_ = new_basic;
        if (drop_precise_codes) {
            high_precise_codes_.reset();
        } else {
            high_precise_codes_ = new_precise;
        }
        use_reorder_ = new_use_reorder;
        reorder_by_base_ = new_reorder_by_base;
        param->use_reorder = new_use_reorder;

        check_and_init_raw_vector(param->raw_vector_param, common_param, false);

        // set status
        if (disable_future_tuning) {
            this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_TUNE, false);
            this->raw_vector_.reset();
            has_raw_vector_ = false;
            create_new_raw_vector_ = false;
        }
        this->refresh_code_read_views();
        init_resize_bit_and_reorder();
    }
    return true;
}

uint64_t
HGraph::EstimateMemory(uint64_t num_elements) const {
    uint64_t estimate_memory = 0;
    auto block_size = Options::Instance().block_size_limit();
    auto element_count =
        next_multiple_of_power_of_two(num_elements, this->resize_increase_count_bit_);

    auto block_memory_ceil = [](uint64_t memory, uint64_t block_size) -> uint64_t {
        return static_cast<uint64_t>(
            std::ceil(static_cast<double>(memory) / static_cast<double>(block_size)) *
            static_cast<double>(block_size));
    };

    if (this->basic_flatten_codes_->InMemory()) {
        auto base_memory = this->basic_flatten_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(base_memory, block_size);
    }

    if (bottom_graph_->InMemory()) {
        auto bottom_graph_memory =
            (this->bottom_graph_->maximum_degree_ + 1) * sizeof(InnerIdType) * element_count;
        estimate_memory += block_memory_ceil(bottom_graph_memory, block_size);
    }

    if (has_precise_reorder() && this->high_precise_codes_->InMemory() &&
        not this->ignore_reorder_) {
        auto precise_memory = this->high_precise_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(precise_memory, block_size);
    }

    if (extra_info_size_ > 0 && this->extra_infos_ != nullptr && this->extra_infos_->InMemory()) {
        auto extra_info_memory = this->extra_infos_->ExtraInfoSize() * element_count;
        estimate_memory += block_memory_ceil(extra_info_memory, block_size);
    }

    auto label_map_memory =
        element_count * (sizeof(std::pair<LabelType, InnerIdType>) + 2 * sizeof(void*));
    estimate_memory += label_map_memory;

    auto sparse_graph_memory = (this->mult_ * 0.05 * static_cast<double>(element_count)) *
                               sizeof(InnerIdType) *
                               (static_cast<double>(this->bottom_graph_->maximum_degree_) / 2 + 1);
    estimate_memory += static_cast<uint64_t>(sparse_graph_memory);

    auto other_memory = element_count * (sizeof(LabelType) + sizeof(std::shared_mutex) +
                                         sizeof(std::shared_ptr<std::shared_mutex>));
    estimate_memory += other_memory;

    return estimate_memory;
}

GraphInterfacePtr
HGraph::generate_one_route_graph() {
    return std::make_shared<SparseGraphDataCell>(hierarchical_datacell_param_, this->allocator_);
}

float
HGraph::CalcDistanceById(const float* query, int64_t id, bool calculate_precise_distance) const {
    FlattenInterfacePtr flat;
    {
        std::shared_lock<std::shared_mutex> lock;
        if (!this->immutable_.load(std::memory_order_acquire)) {
            lock = std::shared_lock<std::shared_mutex>(this->global_mutex_);
        }
        flat = this->basic_read_codes_;
        if (has_precise_reorder() && calculate_precise_distance) {
            flat = this->precise_read_codes_;
        }
        if (create_new_raw_vector_ && calculate_precise_distance) {
            flat = this->raw_read_codes_;
        }
    }
    return InnerIndexInterface::calc_distance_by_id(query, id, flat);
}

DatasetPtr
HGraph::CalDistanceById(const float* query,
                        const int64_t* ids,
                        int64_t count,
                        bool calculate_precise_distance) const {
    FlattenInterfacePtr flat;
    {
        std::shared_lock<std::shared_mutex> lock;
        if (!this->immutable_.load(std::memory_order_acquire)) {
            lock = std::shared_lock<std::shared_mutex>(this->global_mutex_);
        }
        flat = this->basic_read_codes_;
        if (has_precise_reorder() && calculate_precise_distance) {
            flat = this->precise_read_codes_;
        }
        if (create_new_raw_vector_ && calculate_precise_distance) {
            flat = this->raw_read_codes_;
        }
    }
    return InnerIndexInterface::cal_distance_by_id(query, ids, count, flat);
}

std::pair<int64_t, int64_t>
HGraph::GetMinAndMaxId() const {
    int64_t min_id = INT64_MAX;
    int64_t max_id = INT64_MIN;
    std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
    if (this->total_count_ == 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Label map size is zero");
    }
    for (int i = 0; i < this->total_count_; ++i) {
        if (this->label_table_->IsRemoved(i)) {
            continue;
        }
        auto label = this->label_table_->GetLabelById(i);
        max_id = std::max(label, max_id);
        min_id = std::min(label, min_id);
    }
    return {min_id, max_id};
}

InnerIndexPtr
HGraph::ExportModel(const IndexCommonParam& param) const {
    auto index = std::make_shared<HGraph>(this->create_param_ptr_, param);
    this->basic_flatten_codes_->ExportModel(index->basic_flatten_codes_);
    if (has_precise_reorder()) {
        this->high_precise_codes_->ExportModel(index->high_precise_codes_);
    }
    return index;
}
void
HGraph::GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const {
    if (raw_read_codes_ != nullptr) {
        raw_read_codes_->GetCodesById(inner_id, data);
        return;
    }

    if (has_precise_reorder()) {
        precise_read_codes_->GetCodesById(inner_id, data);
    } else {
        basic_read_codes_->GetCodesById(inner_id, data);
    }
}

void
HGraph::Merge(const std::vector<MergeUnit>& merge_units) {
    int64_t total_count = this->GetNumElements();
    for (const auto& unit : merge_units) {
        total_count += unit.index->GetNumElements();
    }
    if (max_capacity_ < total_count) {
        this->resize(total_count);
    }
    for (const auto& merge_unit : merge_units) {
        const auto other_index = std::dynamic_pointer_cast<HGraph>(
            std::dynamic_pointer_cast<IndexImpl<HGraph>>(merge_unit.index)->GetInnerIndex());
        if (this->support_duplicate_ || other_index->support_duplicate_) {
            throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                "HGraph merge does not support duplicate detection yet");
        }
        if (total_count_ == 0) {
            this->entry_point_id_ = other_index->entry_point_id_;
        }
        basic_flatten_codes_->MergeOther(other_index->basic_flatten_codes_, this->total_count_);
        label_table_->MergeOther(other_index->label_table_, merge_unit.id_map_func);
        if (has_precise_reorder()) {
            high_precise_codes_->MergeOther(other_index->high_precise_codes_, this->total_count_);
        }
        bottom_graph_->MergeOther(other_index->bottom_graph_, this->total_count_);
        if (route_graphs_.size() < other_index->route_graphs_.size()) {
            route_graphs_.push_back(this->generate_one_route_graph());
        }
        for (int j = 0; j < std::min(other_index->route_graphs_.size(), route_graphs_.size());
             ++j) {
            route_graphs_[j]->MergeOther(other_index->route_graphs_[j], this->total_count_);
        }
        this->total_count_ += other_index->GetNumElements();
    }
    if (this->odescent_param_ == nullptr) {
        odescent_param_ = std::make_shared<ODescentParameter>();
    }

    auto build_data = (has_precise_reorder() and not build_by_base_) ? this->high_precise_codes_
                                                                     : this->basic_flatten_codes_;
    for (InnerIdType inner_id = 0; inner_id < this->total_count_; ++inner_id) {
        Vector<InnerIdType> neighbors(this->allocator_);
        this->bottom_graph_->GetNeighbors(inner_id, neighbors);
        neighbors.resize(neighbors.size() / 2);
        this->bottom_graph_->InsertNeighborsById(inner_id, neighbors);
    }
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        odescent_builder.Build(bottom_graph_);
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& graph : route_graphs_) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto ids = graph->GetIds();
        sparse_odescent_builder.Build(ids, graph);
        sparse_odescent_builder.SaveGraph(graph);
        this->entry_point_id_ = ids.back();
    }
}

void
HGraph::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    auto codes = (has_precise_reorder()) ? precise_read_codes_ : basic_read_codes_;
    codes = (create_new_raw_vector_) ? raw_read_codes_ : codes;
    bool release;
    const auto* buffer = codes->GetCodesById(inner_id, release);
    if (buffer == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("failed to get vector by inner id {}", inner_id));
    }
    codes->Decode(buffer, data);
    if (release) {
        codes->Release(buffer);
    }
}

void
HGraph::SetImmutable() {
    if (this->immutable_.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock<std::shared_mutex> add_lock(this->add_mutex_);
    std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
    this->neighbors_mutex_.reset();
    this->neighbors_mutex_ = std::make_shared<EmptyMutex>();
    this->searcher_->SetMutexArray(this->neighbors_mutex_);
    this->immutable_.store(true, std::memory_order_release);
}

void
HGraph::SetIO(const std::shared_ptr<Reader> reader) {
    auto reader_param = std::make_shared<ReaderIOParameter>();
    reader_param->reader = reader;
    if (has_precise_reorder()) {
        high_precise_codes_->InitIO(reader_param);
    }
    basic_flatten_codes_->InitIO(reader_param);
    bottom_graph_->InitIO(reader_param);
}

const static uint64_t QUERY_SAMPLE_SIZE = 10;
const static int64_t DEFAULT_TOPK = 100;

std::string
HGraph::GetStats() const {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = DEFAULT_TOPK;
    analyzer_param.base_sample_size = std::min(QUERY_SAMPLE_SIZE, this->total_count_.load());
    analyzer_param.search_params =
        fmt::format(R"({{"hgraph": {{"ef_search": {}}}}})", ef_construct_);
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    JsonType stats = analyzer->GetStats();
    // Build-time cache hit-rate is a transient property of the
    // build_with_cache() path (taken only after ImportCache()), so it lives on
    // HGraph rather than in the post-hoc analyzer. A negative rate means this
    // index was not built from an imported cache.
    if (this->build_cache_hit_rate_ >= 0.0F) {
        stats["build_cache_hit_rate"].SetFloat(this->build_cache_hit_rate_);
        stats["build_cache_hit_nodes"].SetInt(this->build_cache_hit_nodes_);
        stats["build_cache_missed_nodes"].SetInt(this->build_cache_missed_nodes_);
    } else {
        stats["build_cache_hit_rate"]["skipped_reason"].SetString(
            "index was not built from an imported cache");
    }
    if (this->has_code_slot_mapping()) {
        stats["code_slots"]["logical_count"].SetInt(this->total_count_.load());
        stats["code_slots"]["physical_count"].SetInt(this->code_slot_mapping_->PhysicalCount());
        stats["code_slots"]["mapping_memory_bytes"].SetInt(
            this->code_slot_mapping_->GetMemoryUsage());
    }
    return stats.Dump(4);
}

void
HGraph::init_resize_bit_and_reorder() {
    auto step_block_size = Options::Instance().block_size_limit();
    auto block_size_per_vector = this->basic_flatten_codes_->code_size_;
    block_size_per_vector =
        std::max(block_size_per_vector,
                 static_cast<uint32_t>(this->bottom_graph_->maximum_degree_ * sizeof(InnerIdType)));
    if (use_reorder_) {
        auto reorder_codes = this->get_reorder_read_codes();
        block_size_per_vector = std::max(block_size_per_vector, reorder_codes->code_size_);
        reorder_ = std::make_shared<FlattenReorder>(reorder_codes, allocator_);
    } else {
        reorder_.reset();
    }
    if (this->extra_infos_ != nullptr) {
        block_size_per_vector =
            std::max<int64_t>(block_size_per_vector, static_cast<uint32_t>(this->extra_info_size_));
    }
    auto increase_count = step_block_size / block_size_per_vector;
    this->resize_increase_count_bit_ = std::max(
        DEFAULT_RESIZE_BIT, static_cast<uint64_t>(log2(static_cast<double>(increase_count))));
}

void
HGraph::check_and_init_raw_vector(const FlattenInterfaceParamPtr& raw_vector_param,
                                  const IndexCommonParam& common_param,
                                  bool is_create_new) {
    if (raw_vector_param == nullptr) {
        return;
    }

    if (is_create_new) {
        raw_vector_ = FlattenInterface::MakeInstance(raw_vector_param, common_param);
    }

    if (basic_flatten_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32 and
        high_precise_codes_ == nullptr) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }
    if (basic_flatten_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32 and
        high_precise_codes_ != nullptr and
        high_precise_codes_->GetQuantizerName() != QUANTIZATION_TYPE_VALUE_FP32) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }

    auto io_type_name = raw_vector_param->io_parameter->GetTypeName();
    if (io_type_name != IO_TYPE_VALUE_BLOCK_MEMORY_IO and io_type_name != IO_TYPE_VALUE_MEMORY_IO) {
        create_new_raw_vector_ = true;
        has_raw_vector_ = true;
        return;
    }

    if (basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        raw_vector_ = basic_flatten_codes_;
        has_raw_vector_ = true;
        return;
    }

    if (high_precise_codes_ != nullptr and
        high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        raw_vector_ = high_precise_codes_;
        has_raw_vector_ = true;
        return;
    }
}

bool
HGraph::UpdateVector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    std::shared_lock<std::shared_mutex> force_remove_rlock;
    if (this->support_force_remove()) {
        force_remove_rlock = std::shared_lock<std::shared_mutex>(this->force_remove_mutex_);
    }
    // check if id exists and get copied base data
    uint32_t inner_id = 0;
    {
        std::shared_lock label_lock(this->label_lookup_mutex_);
        inner_id = this->label_table_->GetIdByLabel(id);
    }

    // the validation of the new vector
    void* new_base_vec = nullptr;
    uint64_t data_size = 0;
    get_vectors(data_type_, dim_, new_base, &new_base_vec, &data_size);

    if (not force_update) {
        std::shared_lock label_lock(this->label_lookup_mutex_);

        // 1. check whether vectors are same
        Vector<int8_t> base_data(data_size, allocator_);
        GetVectorByInnerId(inner_id, (float*)base_data.data());
        float old_self_dist = this->CalcDistanceById((float*)base_data.data(), id);
        float self_dist = this->CalcDistanceById((float*)new_base_vec, id);
        if (std::abs(old_self_dist - self_dist) < 1e-3) {
            return true;
        }

        // 2. check whether the neighborhood relationship is same
        Vector<InnerIdType> neighbors(allocator_);
        this->bottom_graph_->GetNeighbors(inner_id, neighbors);
        for (auto neighbor_inner_id : neighbors) {
            // don't compare with itself
            if (neighbor_inner_id == inner_id) {
                continue;
            }

            float neighbor_dist = 0;
            try {
                neighbor_dist =
                    this->CalcDistanceById(static_cast<float*>(new_base_vec),
                                           this->label_table_->GetLabelById(neighbor_inner_id));
            } catch (const std::runtime_error& e) {
                // incase that neighbor has been deleted
                continue;
            }
            if (neighbor_dist < self_dist) {
                return false;
            }
        }
    }

    // note that only modify vector need to obtain unique lock
    // and the lock has been obtained inside datacell
    auto code_slot_id = static_cast<InnerIdType>(inner_id);
    if (this->has_code_slot_mapping()) {
        auto mapping_inner_id = static_cast<InnerIdType>(inner_id);
        std::unique_lock storage_lock(this->code_slot_storage_mutex_);
        bool insert_new_slot = not this->code_slot_mapping_->IsMapped(mapping_inner_id) ||
                               this->code_slot_mapping_->IsSharedSlot(mapping_inner_id);
        if (insert_new_slot) {
            code_slot_id = this->code_slot_mapping_->PhysicalCount();
            this->write_persistent_codes_unlocked(new_base_vec, code_slot_id);
            this->code_slot_mapping_->CommitNewSlot(mapping_inner_id, code_slot_id);
            return true;
        }

        code_slot_id = this->code_slot_mapping_->Resolve(mapping_inner_id);
        bool update_status = basic_flatten_codes_->UpdateVector(new_base_vec, code_slot_id);
        if (has_precise_reorder()) {
            update_status =
                update_status && high_precise_codes_->UpdateVector(new_base_vec, code_slot_id);
        }
        if (create_new_raw_vector_) {
            update_status = update_status && raw_vector_->UpdateVector(new_base_vec, code_slot_id);
        }
        return update_status;
    }
    code_slot_id = this->resolve_code_slot_id(static_cast<InnerIdType>(inner_id));
    bool update_status = basic_flatten_codes_->UpdateVector(new_base_vec, code_slot_id);
    if (has_precise_reorder()) {
        update_status =
            update_status && high_precise_codes_->UpdateVector(new_base_vec, code_slot_id);
    }
    if (create_new_raw_vector_) {
        update_status = update_status && raw_vector_->UpdateVector(new_base_vec, code_slot_id);
    }
    return update_status;
}

std::string
HGraph::AnalyzeIndexBySearch(const SearchRequest& request) {
    AnalyzerParam analyzer_param(allocator_);
    analyzer_param.topk = request.topk_;
    auto analyzer = CreateAnalyzer(this, analyzer_param);
    JsonType stats = analyzer->AnalyzeIndexBySearch(request);
    return stats.Dump(4);
}

void
HGraph::GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
    this->attr_filter_index_->GetAttribute(0, inner_id, attr);
}

void
HGraph::cal_memory_usage() {
    auto memory = sizeof(HGraph);
    memory += this->neighbors_mutex_->GetMemoryUsage();
    memory += this->pool_->GetMemoryUsage();
    memory += this->label_table_->GetMemoryUsage();
    memory += this->basic_flatten_codes_->GetMemoryUsage();
    memory += this->bottom_graph_->GetMemoryUsage();
    for (auto& graph : this->route_graphs_) {
        memory += graph->GetMemoryUsage();
    }
    if (has_precise_reorder()) {
        memory += this->high_precise_codes_->GetMemoryUsage();
    }

    if (this->extra_infos_ != nullptr and this->extra_info_size_ > 0) {
        memory += this->extra_infos_->GetMemoryUsage();
    }

    if (this->create_new_raw_vector_ and this->raw_vector_ != nullptr) {
        memory += raw_vector_->GetMemoryUsage();
    }
    if (this->has_code_slot_mapping()) {
        memory += this->code_slot_mapping_->GetMemoryUsage();
    }

    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(static_cast<int64_t>(memory));
}

}  // namespace vsag
