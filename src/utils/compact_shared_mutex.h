
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

#if defined(__linux__)

#include <atomic>
#include <cstdint>

namespace vsag {

class CompactSharedMutex {
public:
    CompactSharedMutex() = default;

    CompactSharedMutex(const CompactSharedMutex&) = delete;
    CompactSharedMutex&
    operator=(const CompactSharedMutex&) = delete;

    void
    lock() {
        uint32_t expected = 0;
        if (state_.compare_exchange_strong(
                expected, kWriterLocked, std::memory_order_acquire, std::memory_order_relaxed)) {
            return;
        }
        LockSlow();
    }

    void
    unlock() {
        const auto old = state_.fetch_and(~kWriterLocked, std::memory_order_release);
        if ((old & kWaitingWritersMask) != 0) {
            WakeWriter();
        } else {
            WakeReaders();
        }
    }

    void
    lock_shared() {
        auto current = state_.load(std::memory_order_relaxed);
        if ((current & (kWriterLocked | kWaitingWritersMask)) == 0 &&
            (current & kReaderCountMask) != kReaderCountMask &&
            state_.compare_exchange_strong(
                current, current + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            return;
        }
        LockSharedSlow();
    }

    void
    unlock_shared() {
        const auto old = state_.fetch_sub(1, std::memory_order_release);
        if ((old & kReaderCountMask) == 1 && (old & kWaitingWritersMask) != 0) {
            WakeWriter();
        }
    }

private:
    static constexpr uint32_t kReaderCountBits = 20;
    static constexpr uint32_t kReaderCountMask = (1U << kReaderCountBits) - 1;
    static constexpr uint32_t kWaitingWriterOne = 1U << kReaderCountBits;
    static constexpr uint32_t kWaitingWritersMask = 0x7FFU << kReaderCountBits;
    static constexpr uint32_t kWriterLocked = 1U << 31;

    static constexpr uint32_t kReaderWaiterBit = 1U;
    static constexpr uint32_t kWriterWaiterBit = 2U;

    void
    LockSlow();

    void
    LockSharedSlow();

    void
    Wait(uint32_t expected, uint32_t waiter_bit);

    void
    WakeWriter();

    void
    WakeReaders();

    std::atomic<uint32_t> state_{0};
};

static_assert(sizeof(CompactSharedMutex) == sizeof(uint32_t));
static_assert(std::atomic<uint32_t>::is_always_lock_free);

}  // namespace vsag

#endif
