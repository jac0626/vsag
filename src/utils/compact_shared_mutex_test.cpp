
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

#include "compact_shared_mutex.h"

#if defined(__linux__)

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "unittest.h"

using namespace vsag;

TEST_CASE("CompactSharedMutex Basic", "[ut][CompactSharedMutex]") {
    CompactSharedMutex mutex;

    REQUIRE(sizeof(mutex) == sizeof(uint32_t));
    mutex.lock();
    mutex.unlock();
    mutex.lock_shared();
    mutex.unlock_shared();
}

TEST_CASE("CompactSharedMutex blocks writers behind readers", "[ut][CompactSharedMutex]") {
    CompactSharedMutex mutex;
    std::atomic<bool> writer_started{false};
    std::atomic<bool> writer_acquired{false};

    mutex.lock_shared();
    std::thread writer([&]() {
        writer_started.store(true, std::memory_order_release);
        mutex.lock();
        writer_acquired.store(true, std::memory_order_release);
        mutex.unlock();
    });

    while (not writer_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    REQUIRE_FALSE(writer_acquired.load(std::memory_order_acquire));
    mutex.unlock_shared();
    writer.join();
    REQUIRE(writer_acquired.load(std::memory_order_acquire));
}

TEST_CASE("CompactSharedMutex excludes readers and writers", "[ut][CompactSharedMutex]") {
    CompactSharedMutex mutex;
    std::atomic<uint64_t> active_readers{0};
    std::atomic<uint64_t> active_writers{0};
    std::atomic<uint64_t> violations{0};
    std::atomic<uint64_t> writes{0};
    constexpr uint64_t kReaderThreads = 8;
    constexpr uint64_t kWriterThreads = 4;
    constexpr uint64_t kIterations = 20000;
    std::vector<std::thread> threads;

    for (uint64_t i = 0; i < kReaderThreads; ++i) {
        threads.emplace_back([&]() {
            for (uint64_t iteration = 0; iteration < kIterations; ++iteration) {
                mutex.lock_shared();
                active_readers.fetch_add(1, std::memory_order_relaxed);
                if (active_writers.load(std::memory_order_relaxed) != 0) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                active_readers.fetch_sub(1, std::memory_order_relaxed);
                mutex.unlock_shared();
            }
        });
    }
    for (uint64_t i = 0; i < kWriterThreads; ++i) {
        threads.emplace_back([&]() {
            for (uint64_t iteration = 0; iteration < kIterations; ++iteration) {
                mutex.lock();
                if (active_writers.fetch_add(1, std::memory_order_relaxed) != 0 ||
                    active_readers.load(std::memory_order_relaxed) != 0) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
                writes.fetch_add(1, std::memory_order_relaxed);
                active_writers.fetch_sub(1, std::memory_order_relaxed);
                mutex.unlock();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
    REQUIRE(violations.load(std::memory_order_relaxed) == 0);
    REQUIRE(writes.load(std::memory_order_relaxed) == kWriterThreads * kIterations);
}

TEST_CASE("CompactSharedMutex gives waiting writers progress", "[ut][CompactSharedMutex]") {
    CompactSharedMutex mutex;
    std::atomic<bool> stop{false};
    std::atomic<bool> writer_acquired{false};
    constexpr uint64_t kReaderThreads = 16;
    std::vector<std::thread> readers;

    for (uint64_t i = 0; i < kReaderThreads; ++i) {
        readers.emplace_back([&]() {
            while (not stop.load(std::memory_order_relaxed)) {
                mutex.lock_shared();
                std::this_thread::yield();
                mutex.unlock_shared();
            }
        });
    }

    std::thread writer([&]() {
        mutex.lock();
        writer_acquired.store(true, std::memory_order_release);
        mutex.unlock();
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (not writer_acquired.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const auto made_progress = writer_acquired.load(std::memory_order_acquire);
    stop.store(true, std::memory_order_relaxed);
    for (auto& reader : readers) {
        reader.join();
    }
    writer.join();

    REQUIRE(made_progress);
}

#endif
