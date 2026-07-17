
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

#include "memory_peak_monitor.h"

#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>

namespace vsag::eval {

namespace {

constexpr auto SAMPLE_INTERVAL = std::chrono::milliseconds(5);

#if !defined(__APPLE__)
std::string
get_proc_file_name(pid_t pid) {
    return "/proc/" + std::to_string(pid) + "/statm";
}
#endif

uint64_t
page_size_bytes() {
    const auto page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? static_cast<uint64_t>(page_size) : uint64_t{4096};
}

}  // namespace

MemoryPeakMonitor::MemoryPeakMonitor(std::string name)
    : Monitor("memory_peak_monitor"), process_name_(std::move(name)) {
    init_memory_ = GetCurrentResidentPages();
    max_memory_ = init_memory_;
}

MemoryPeakMonitor::MemoryPeakMonitor(std::string name, uint64_t init_memory_pages)
    : Monitor("memory_peak_monitor"),
      init_memory_(init_memory_pages),
      process_name_(std::move(name)) {
    max_memory_ = init_memory_;
}

MemoryPeakMonitor::~MemoryPeakMonitor() {
    this->StopSampling();
}

uint64_t
MemoryPeakMonitor::GetCurrentResidentPages() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const auto status = task_info(
        mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count);
    if (status != KERN_SUCCESS) {
        return 0;
    }
    const auto page_size = page_size_bytes();
    return (static_cast<uint64_t>(info.resident_size) + page_size - 1) / page_size;
#else
    std::ifstream infile(get_proc_file_name(getpid()));
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    infile >> total_pages >> resident_pages;
    return resident_pages;
#endif
}

void
MemoryPeakMonitor::Start() {
    this->StopSampling();
    sampling_active_.store(true, std::memory_order_release);
    sampling_thread_ = std::thread([this]() {
        std::unique_lock<std::mutex> lock(sampling_wait_mutex_);
        while (sampling_active_.load(std::memory_order_acquire)) {
            sampling_cv_.wait_for(lock, SAMPLE_INTERVAL, [this]() {
                return not sampling_active_.load(std::memory_order_acquire);
            });
            if (not sampling_active_.load(std::memory_order_acquire)) {
                break;
            }
            lock.unlock();
            this->SampleCurrentResidentPages();
            lock.lock();
        }
    });
}
void
MemoryPeakMonitor::Stop() {
    this->StopSampling();
}

void
MemoryPeakMonitor::StopSampling() {
    sampling_active_.store(false, std::memory_order_release);
    sampling_cv_.notify_all();
    if (sampling_thread_.joinable()) {
        sampling_thread_.join();
    }
    this->SampleCurrentResidentPages();
}
Monitor::JsonType
MemoryPeakMonitor::GetResult() {
    std::lock_guard<std::mutex> lock(record_mutex_);
    JsonType result;
    std::vector<std::string> metrics = {"B", "KB", "MB", "GB", "TB"};
    const uint64_t peak_delta_pages =
        this->max_memory_ > this->init_memory_ ? this->max_memory_ - this->init_memory_ : 0;
    auto size = static_cast<float>(peak_delta_pages * page_size_bytes());
    size_t i = 0;
    while (size >= 1024.0F && i < metrics.size() - 1) {
        size /= 1024;
        i++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size;
    result["memory_peak(" + process_name_ + ")"] = oss.str() + " " + metrics[i];
    return result;
}
void
MemoryPeakMonitor::Record(void* input) {
    static_cast<void>(input);
    this->SampleCurrentResidentPages();
}

void
MemoryPeakMonitor::SampleCurrentResidentPages() {
    std::lock_guard<std::mutex> lock(record_mutex_);
    const auto resident_pages = GetCurrentResidentPages();
    if (max_memory_ < resident_pages) {
        max_memory_ = resident_pages;
    }
}

}  // namespace vsag::eval
