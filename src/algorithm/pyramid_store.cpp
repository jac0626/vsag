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

#include "pyramid_store.h"

#include <algorithm>
#include <cstring>

#include "common.h"
#include "datacell/flatten_interface.h"
#include "impl/logger/logger.h"

namespace vsag {

PyramidStore::PyramidStore(const PyramidParamPtr& pyramid_param,
                           const IndexCommonParam& common_param)
    : label_table_(std::make_shared<LabelTable>(
          common_param.allocator_.get(), true, false, pyramid_param->label_remap_type)),
      allocator_(common_param.allocator_.get()),
      dim_(common_param.dim_),
      use_reorder_(pyramid_param->use_reorder) {
    base_codes_ = FlattenInterface::MakeInstance(pyramid_param->base_codes_param, common_param);
    if (use_reorder_) {
        precise_codes_ =
            FlattenInterface::MakeInstance(pyramid_param->precise_codes_param, common_param);
        reorder_ = std::make_shared<FlattenReorder>(precise_codes_, allocator_);
    }
}

void
PyramidStore::Train(const DatasetPtr& base) {
    CHECK_ARGUMENT(base != nullptr, "base dataset is required");
    base_codes_->Train(base->GetFloat32Vectors(), base->GetNumElements());
    if (use_reorder_) {
        precise_codes_->Train(base->GetFloat32Vectors(), base->GetNumElements());
    }
}

void
PyramidStore::resize_locked(int64_t new_max_capacity, const ResizeCallback& on_resize) {
    if (new_max_capacity <= max_capacity_) {
        return;
    }

    label_table_->Resize(new_max_capacity);
    base_codes_->Resize(new_max_capacity);
    if (use_reorder_) {
        precise_codes_->Resize(new_max_capacity);
    }
    if (on_resize != nullptr) {
        on_resize(new_max_capacity);
    }
    max_capacity_ = new_max_capacity;
}

void
PyramidStore::Resize(int64_t new_max_capacity, const ResizeCallback& on_resize) {
    std::unique_lock<std::shared_mutex> lock(resize_mutex_);
    resize_locked(new_max_capacity, on_resize);
}

PyramidStore::InsertResult
PyramidStore::Insert(const DatasetPtr& base,
                     bool check_duplicate,
                     const ResizeCallback& on_resize) {
    CHECK_ARGUMENT(base != nullptr, "base dataset is required");
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    CHECK_ARGUMENT(data_vectors != nullptr, "base vectors are required");
    CHECK_ARGUMENT(data_ids != nullptr, "base ids are required");

    InsertResult result(allocator_);
    int64_t data_num = base->GetNumElements();
    {
        std::lock_guard lock(cur_element_count_mutex_);
        auto local_cur_element_count = cur_element_count_;
        if (max_capacity_ == 0) {
            Resize(std::max(INIT_CAPACITY, data_num), on_resize);
        } else if (max_capacity_ < data_num + cur_element_count_) {
            auto new_capacity = std::min(MAX_CAPACITY_EXTEND, max_capacity_);
            new_capacity = std::max(data_num + cur_element_count_ - max_capacity_, new_capacity) +
                           max_capacity_;
            Resize(new_capacity, on_resize);
        }

        int64_t valid_id_count = 0;
        for (int64_t i = 0; i < data_num; ++i) {
            if (check_duplicate && label_table_->CheckLabel(data_ids[i])) {
                logger::warn("Label {} already exists, skip adding.", data_ids[i]);
                result.failed_ids.push_back(data_ids[i]);
                continue;
            }
            auto inner_id = static_cast<InnerIdType>(valid_id_count + local_cur_element_count);
            label_table_->Insert(inner_id, data_ids[i]);
            base_codes_->InsertVector(data_vectors + dim_ * i, inner_id);
            if (use_reorder_) {
                precise_codes_->InsertVector(data_vectors + dim_ * i, inner_id);
            }
            valid_id_count++;
            result.data_biases.push_back(i);
            result.inner_ids.push_back(inner_id);
        }
        cur_element_count_ += valid_id_count;
    }
    return result;
}

void
PyramidStore::InsertForBuild(const DatasetPtr& base, const ResizeCallback& on_resize) {
    CHECK_ARGUMENT(base != nullptr, "base dataset is required");
    const auto* data_vectors = base->GetFloat32Vectors();
    const auto* data_ids = base->GetIds();
    CHECK_ARGUMENT(data_vectors != nullptr, "base vectors are required");
    CHECK_ARGUMENT(data_ids != nullptr, "base ids are required");

    int64_t data_num = base->GetNumElements();
    {
        std::lock_guard lock(cur_element_count_mutex_);
        Resize(data_num, on_resize);
        std::memcpy(label_table_->label_table_.data(), data_ids, sizeof(LabelType) * data_num);

        base_codes_->BatchInsertVector(data_vectors, data_num);
        if (use_reorder_) {
            precise_codes_->BatchInsertVector(data_vectors, data_num);
        }
        cur_element_count_ = data_num;
    }
}

void
PyramidStore::Serialize(StreamWriter& writer) const {
    label_table_->Serialize(writer);
    base_codes_->Serialize(writer);
    if (use_reorder_) {
        precise_codes_->Serialize(writer);
    }
}

void
PyramidStore::Deserialize(StreamReader& reader) {
    label_table_->Deserialize(reader);
    base_codes_->Deserialize(reader);
    if (use_reorder_) {
        precise_codes_->Deserialize(reader);
    }
    cur_element_count_ = base_codes_->TotalCount();
}

void
PyramidStore::ExportModel(PyramidStore& target) const {
    if (target.use_reorder_ != this->use_reorder_) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Export model's pyramid reorder config mismatched");
    }
    this->base_codes_->ExportModel(target.base_codes_);
    if (use_reorder_) {
        if (target.precise_codes_ == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Export model's pyramid precise codes is empty");
        }
        this->precise_codes_->ExportModel(target.precise_codes_);
    }
}

void
PyramidStore::SetImmutable() {
    label_table_->SetImmutable();
}

}  // namespace vsag
