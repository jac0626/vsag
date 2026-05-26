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

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "algorithm/pyramid_zparameters.h"
#include "datacell/flatten_interface.h"
#include "impl/label_table.h"
#include "impl/reorder/flatten_reorder.h"
#include "impl/reorder/reorder.h"
#include "index_common_param.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/dataset.h"

namespace vsag {

class PyramidStore {
public:
    using ResizeCallback = std::function<void(int64_t)>;

    struct InsertResult {
        explicit InsertResult(Allocator* allocator) : data_biases(allocator), inner_ids(allocator) {
        }

        std::vector<int64_t> failed_ids;
        Vector<int64_t> data_biases;
        Vector<InnerIdType> inner_ids;
    };

public:
    PyramidStore(const PyramidParamPtr& pyramid_param, const IndexCommonParam& common_param);

    void
    Train(const DatasetPtr& base);

    InsertResult
    Insert(const DatasetPtr& base, bool check_duplicate, const ResizeCallback& on_resize);

    void
    InsertForBuild(const DatasetPtr& base, const ResizeCallback& on_resize);

    void
    Resize(int64_t new_max_capacity, const ResizeCallback& on_resize);

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    void
    ExportModel(PyramidStore& target) const;

    [[nodiscard]] std::shared_lock<std::shared_mutex>
    ResizeLock() const {
        return std::shared_lock<std::shared_mutex>(resize_mutex_);
    }

    [[nodiscard]] int64_t
    GetNumElements() const {
        return base_codes_->TotalCount();
    }

    void
    SetImmutable();

    [[nodiscard]] const LabelTablePtr&
    label_table() const {
        return label_table_;
    }

    [[nodiscard]] const FlattenInterfacePtr&
    base_codes() const {
        return base_codes_;
    }

    [[nodiscard]] const FlattenInterfacePtr&
    precise_codes() const {
        return precise_codes_;
    }

    [[nodiscard]] const ReorderInterfacePtr&
    reorder() const {
        return reorder_;
    }

    [[nodiscard]] bool
    use_reorder() const {
        return use_reorder_;
    }

    [[nodiscard]] int64_t
    max_capacity() const {
        return max_capacity_;
    }

    [[nodiscard]] FlattenInterfacePtr
    DefaultCodes() const {
        return use_reorder_ ? precise_codes_ : base_codes_;
    }

private:
    void
    resize_locked(int64_t new_max_capacity, const ResizeCallback& on_resize);

private:
    LabelTablePtr label_table_{nullptr};
    FlattenInterfacePtr base_codes_{nullptr};
    FlattenInterfacePtr precise_codes_{nullptr};
    ReorderInterfacePtr reorder_{nullptr};
    Allocator* allocator_{nullptr};
    uint64_t dim_{0};
    mutable std::shared_mutex resize_mutex_;
    std::mutex cur_element_count_mutex_;
    int64_t max_capacity_{0};
    int64_t cur_element_count_{0};
    bool use_reorder_{false};
};

}  // namespace vsag
