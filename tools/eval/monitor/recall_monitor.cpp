
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

#include "recall_monitor.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

#include "../eval_dataset.h"
#include "search_record.h"
namespace vsag::eval {

namespace {

constexpr double THRESHOLD_ERROR = 2e-6;

double
get_recall(const SearchRecord& record) {
    if (record.requested_k == 0) {
        return 0.0;
    }
    if (record.dataset == nullptr || record.query_data == nullptr ||
        record.ground_truth_neighbors == nullptr) {
        throw std::invalid_argument("recall monitor received an incomplete search record");
    }
    if (record.requested_k > record.ground_truth_count) {
        throw std::invalid_argument("requested k exceeds the ground-truth width");
    }

    const auto base_count = record.dataset->GetNumberOfBase();
    const auto distance_func = record.dataset->GetDistanceFunc();
    auto dim = static_cast<uint64_t>(record.dataset->GetDim());
    std::vector<float> ground_truth_distances;
    ground_truth_distances.reserve(record.requested_k);
    for (uint64_t i = 0; i < record.requested_k; ++i) {
        const int64_t id = record.ground_truth_neighbors[i];
        if (id < 0 || id >= base_count) {
            throw std::invalid_argument("ground truth contains an invalid base id");
        }
        ground_truth_distances.emplace_back(
            distance_func(record.query_data, record.dataset->GetOneTrain(id), &dim));
    }
    const float threshold =
        *std::max_element(ground_truth_distances.begin(), ground_truth_distances.end());

    uint64_t hit_count = 0;
    std::unordered_set<int64_t> seen;
    const uint64_t candidate_count = std::min(record.returned_count, record.requested_k);
    if (record.neighbors != nullptr) {
        for (uint64_t i = 0; i < candidate_count; ++i) {
            const int64_t id = record.neighbors[i];
            if (id < 0 || id >= base_count || not seen.emplace(id).second) {
                continue;
            }
            const float distance =
                distance_func(record.query_data, record.dataset->GetOneTrain(id), &dim);
            if (distance <= threshold + THRESHOLD_ERROR) {
                ++hit_count;
            }
        }
    }
    return static_cast<double>(hit_count) / static_cast<double>(record.requested_k);
}

}  // namespace

RecallMonitor::RecallMonitor(uint64_t max_record_counts) : Monitor("recall_monitor") {
    if (max_record_counts > 0) {
        this->recall_records_.reserve(max_record_counts);
    }
}
void
RecallMonitor::Start() {
    std::lock_guard<std::mutex> lock(record_mutex_);
    recall_records_.clear();
}

void
RecallMonitor::Stop() {
}

Monitor::JsonType
RecallMonitor::GetResult() {
    JsonType result;
    for (auto& metric : metrics_) {
        this->cal_and_set_result(metric, result);
    }
    return result;
}
void
RecallMonitor::Record(void* input) {
    if (input == nullptr) {
        throw std::invalid_argument("recall monitor requires a search record");
    }
    const auto* record = static_cast<const SearchRecord*>(input);
    const double recall = get_recall(*record);
    std::lock_guard<std::mutex> lock(record_mutex_);
    this->recall_records_.emplace_back(recall);
}
void
RecallMonitor::SetMetrics(std::string metric) {
    this->metrics_.emplace_back(std::move(metric));
}
void
RecallMonitor::cal_and_set_result(const std::string& metric, Monitor::JsonType& result) {
    if (metric == "avg_recall") {
        auto val = this->cal_avg_recall();
        result["recall_avg"] = val;
    } else if (metric == "percent_recall") {
        std::vector<double> percents = {0, 10, 30, 50, 70, 90};
        for (auto& percent : percents) {
            auto val = this->cal_recall_rate(percent * 0.01);
            result["recall_detail"]["p" + std::to_string(int(percent))] = val;
        }
    }
}

double
RecallMonitor::cal_avg_recall() {
    if (recall_records_.empty()) {
        return 0.0;
    }
    double sum =
        std::accumulate(this->recall_records_.begin(), this->recall_records_.end(), double(0));
    return sum / static_cast<double>(recall_records_.size());
}

double
RecallMonitor::cal_recall_rate(double rate) {
    if (recall_records_.empty()) {
        return 0.0;
    }
    std::sort(this->recall_records_.begin(), this->recall_records_.end());
    auto pos = static_cast<uint64_t>(rate * static_cast<double>(this->recall_records_.size() - 1));
    return recall_records_[pos];
}
}  // namespace vsag::eval
