
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

#include "./build_eval_case.h"

#include <algorithm>
#include <filesystem>
#include <utility>

#include "../monitor/duration_monitor.h"
#include "../monitor/memory_peak_monitor.h"
#include "vsag_exception.h"

namespace vsag::eval {

BuildEvalCase::BuildEvalCase(const std::string& dataset_path,
                             const std::string& index_path,
                             vsag::IndexPtr index,
                             EvalConfig config,
                             EvalDatasetPtr dataset)
    : EvalCase(dataset_path, index_path, std::move(index), std::move(dataset)),
      config_(std::move(config)) {
    this->init_monitors();
}

void
BuildEvalCase::init_monitors() {
    if (config_.enable_memory) {
        auto memory_peak_monitor = std::make_shared<MemoryPeakMonitor>("build");
        this->monitors_.emplace_back(std::move(memory_peak_monitor));
    }
    if (config_.enable_tps) {
        auto duration_monitor = std::make_shared<DurationMonitor>();
        this->monitors_.emplace_back(std::move(duration_monitor));
    }
}

JsonType
BuildEvalCase::Run() {
    this->do_build();
    this->serialize();
    auto result = this->process_result();
    return result;
}
void
BuildEvalCase::do_build() {
    auto base = vsag::Dataset::Make();
    int64_t total_base = this->dataset_ptr_->GetNumberOfBase();
    std::vector<int64_t> ids(total_base);
    std::iota(ids.begin(), ids.end(), 0);
    base->NumElements(total_base)->Dim(this->dataset_ptr_->GetDim())->Ids(ids.data())->Owner(false);
    if (this->dataset_ptr_->GetVectorType() == DENSE_VECTORS) {
        if (this->dataset_ptr_->GetTrainDataType() == vsag::DATATYPE_FLOAT32) {
            base->Float32Vectors((const float*)this->dataset_ptr_->GetTrain());
        } else if (this->dataset_ptr_->GetTrainDataType() == vsag::DATATYPE_INT8) {
            base->Int8Vectors((const int8_t*)this->dataset_ptr_->GetTrain());
        }
    } else {
        base->SparseVectors((const SparseVector*)this->dataset_ptr_->GetTrain());
    }
    uint64_t started_monitor_count = 0;
    try {
        for (auto& monitor : monitors_) {
            monitor->Start();
            ++started_monitor_count;
        }
        auto build_index = index_->Build(base);
        if (not build_index.has_value()) {
            throw std::runtime_error(build_index.error().message);
        }
        for (uint64_t i = 0; i < started_monitor_count; ++i) {
            monitors_[i]->Record();
            monitors_[i]->Stop();
        }
    } catch (...) {
        for (uint64_t i = 0; i < started_monitor_count; ++i) {
            monitors_[i]->Stop();
        }
        throw;
    }
}
void
BuildEvalCase::serialize() {
    std::filesystem::path dir_path(index_path_);
    dir_path = dir_path.parent_path();
    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }
    std::ofstream outfile(this->index_path_, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("failed to open index path for serialization: " + index_path_);
    }
    auto result = this->index_->Serialize(outfile);
    if (not result.has_value()) {
        throw std::runtime_error("failed to serialize index: " + result.error().message);
    }
    outfile.flush();
    if (not outfile.good()) {
        throw std::runtime_error("failed to serialize index: index stream write failed");
    }
}

JsonType
BuildEvalCase::process_result() {
    JsonType result;
    JsonType eval_result;
    for (auto& monitor : this->monitors_) {
        const auto& one_result = monitor->GetResult();
        EvalCase::MergeJsonType(one_result, eval_result);
    }
    result = eval_result;
    const double duration_seconds = result.value("duration(s)", 0.0);
    result["tps"] =
        duration_seconds > 0.0
            ? static_cast<double>(this->dataset_ptr_->GetNumberOfBase()) / duration_seconds
            : 0.0;
    EvalCase::MergeJsonType(this->basic_info_, result);
    result["index_info"] = JsonType::parse(config_.build_param);
    result["action"] = "build";
    result["index"] = config_.index_name;
    result["index_memory(B)"] = this->index_->GetMemoryUsage();
    try {
        auto detail = this->index_->GetMemoryUsageDetail();
        for (const auto& [name, size] : detail) {
            result["memory_detail(B)"][name] = size;
        }
    } catch (const std::exception& e) {
        result["memory_detail_error"] = e.what();
    }
    return result;
}

}  // namespace vsag::eval
