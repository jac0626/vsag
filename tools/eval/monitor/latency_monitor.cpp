
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

#include "latency_monitor.h"

#include <cmath>
#include <mutex>
#include <stdexcept>

#include "search_record.h"

namespace vsag::eval {

LatencyMonitor::LatencyMonitor(uint64_t max_record_counts) : Monitor("latency_monitor") {
    if (max_record_counts > 0) {
        this->latency_records_.reserve(max_record_counts);
    }
}

void
LatencyMonitor::Start() {
    std::lock_guard<std::mutex> lock(record_mutex_);
    latency_records_.clear();
    batch_duration_seconds_ = 0.0;
    batch_start_ = Clock::now();
}
void
LatencyMonitor::Stop() {
    const auto batch_end = Clock::now();
    std::lock_guard<std::mutex> lock(record_mutex_);
    batch_duration_seconds_ = std::chrono::duration<double>(batch_end - batch_start_).count();
}

void
LatencyMonitor::Stop(double batch_duration_seconds) {
    if (batch_duration_seconds < 0.0 || not std::isfinite(batch_duration_seconds)) {
        throw std::invalid_argument("batch duration must be finite and non-negative");
    }
    std::lock_guard<std::mutex> lock(record_mutex_);
    batch_duration_seconds_ = batch_duration_seconds;
}

Monitor::JsonType
LatencyMonitor::GetResult() {
    JsonType result;
    result["duration(s)"] = batch_duration_seconds_;
    for (auto& metric : metrics_) {
        this->cal_and_set_result(metric, result);
    }
    return result;
}
void
LatencyMonitor::Record(void* input) {
    if (input == nullptr) {
        throw std::invalid_argument("latency monitor requires a search record");
    }
    const auto* record = static_cast<const SearchRecord*>(input);
    if (record->latency_ms < 0.0 || not std::isfinite(record->latency_ms)) {
        throw std::invalid_argument("search latency must be finite and non-negative");
    }
    std::lock_guard<std::mutex> lock(record_mutex_);
    this->latency_records_.emplace_back(record->latency_ms);
}
void
LatencyMonitor::SetMetrics(std::string metric) {
    this->metrics_.emplace_back(std::move(metric));
}
void
LatencyMonitor::cal_and_set_result(const std::string& metric, Monitor::JsonType& result) {
    if (metric == "qps") {
        auto val = this->cal_qps();
        result["qps"] = val;
    } else if (metric == "avg_latency") {
        auto val = this->cal_avg_latency();
        result["latency_avg(ms)"] = val;
    } else if (metric == "percent_latency") {
        std::vector<double> percents = {50, 80, 90, 95, 99};
        for (auto& percent : percents) {
            auto val = this->cal_latency_rate(percent * 0.01);
            result["latency_detail(ms)"]["p" + std::to_string(int(percent))] = val;
        }
    }
}

double
LatencyMonitor::cal_qps() {
    if (batch_duration_seconds_ <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(latency_records_.size()) / batch_duration_seconds_;
}

double
LatencyMonitor::cal_avg_latency() {
    if (latency_records_.empty()) {
        return 0.0;
    }
    double total_time_cost =
        std::accumulate(this->latency_records_.begin(), this->latency_records_.end(), double(0));
    return total_time_cost / static_cast<double>(latency_records_.size());
}
double
LatencyMonitor::cal_latency_rate(double rate) {
    if (latency_records_.empty()) {
        return 0.0;
    }
    std::sort(this->latency_records_.begin(), this->latency_records_.end());
    auto pos = static_cast<uint64_t>(rate * static_cast<double>(this->latency_records_.size() - 1));
    return latency_records_[pos];
}
}  // namespace vsag::eval
