
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
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "monitor.h"

namespace vsag::eval {

class MemoryPeakMonitor : public Monitor {
public:
    explicit MemoryPeakMonitor(std::string name);

    MemoryPeakMonitor(std::string name, uint64_t init_memory_pages);

    ~MemoryPeakMonitor() override;

    static uint64_t
    GetCurrentResidentPages();

    void
    Start() override;

    void
    Stop() override;

    JsonType
    GetResult() override;

    void
    Record(void* input) override;

private:
    void
    StopSampling();

    void
    SampleCurrentResidentPages();

    uint64_t max_memory_{0};
    uint64_t init_memory_{0};
    std::string process_name_{};

    std::atomic<bool> sampling_active_{false};
    std::thread sampling_thread_{};
    std::condition_variable sampling_cv_{};
    std::mutex sampling_wait_mutex_{};
};

}  // namespace vsag::eval
