
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

#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <system_error>
#include <thread>

namespace vsag {
namespace {

constexpr uint64_t kSpinCount = 64;

void
CpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}

}  // namespace

void
CompactSharedMutex::LockSlow() {
    auto current = state_.load(std::memory_order_relaxed);
    while (true) {
        if ((current & kWaitingWritersMask) == kWaitingWritersMask) {
            throw std::system_error(
                std::make_error_code(std::errc::resource_unavailable_try_again));
        }
        if (state_.compare_exchange_weak(current,
                                         current + kWaitingWriterOne,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
            break;
        }
    }

    while (true) {
        current = state_.load(std::memory_order_acquire);
        if ((current & (kWriterLocked | kReaderCountMask)) == 0) {
            const auto desired = (current - kWaitingWriterOne) | kWriterLocked;
            if (state_.compare_exchange_weak(
                    current, desired, std::memory_order_acquire, std::memory_order_relaxed)) {
                return;
            }
            continue;
        }
        Wait(current, kWriterWaiterBit);
    }
}

void
CompactSharedMutex::LockSharedSlow() {
    while (true) {
        auto current = state_.load(std::memory_order_acquire);
        if ((current & (kWriterLocked | kWaitingWritersMask)) == 0) {
            if ((current & kReaderCountMask) == kReaderCountMask) {
                std::this_thread::yield();
                continue;
            }
            if (state_.compare_exchange_weak(
                    current, current + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
                return;
            }
            continue;
        }
        Wait(current, kReaderWaiterBit);
    }
}

void
CompactSharedMutex::Wait(uint32_t expected, uint32_t waiter_bit) {
    for (uint64_t spin = 0; spin < kSpinCount; ++spin) {
        if (state_.load(std::memory_order_relaxed) != expected) {
            return;
        }
        CpuRelax();
    }

    auto* address = reinterpret_cast<uint32_t*>(&state_);
    const auto result = syscall(
        SYS_futex, address, FUTEX_WAIT_BITSET_PRIVATE, expected, nullptr, nullptr, waiter_bit);
    if (result == -1 && errno != EAGAIN && errno != EINTR) {
        std::this_thread::yield();
    }
}

void
CompactSharedMutex::WakeWriter() {
    auto* address = reinterpret_cast<uint32_t*>(&state_);
    syscall(SYS_futex, address, FUTEX_WAKE_BITSET_PRIVATE, 1, nullptr, nullptr, kWriterWaiterBit);
}

void
CompactSharedMutex::WakeReaders() {
    auto* address = reinterpret_cast<uint32_t*>(&state_);
    syscall(
        SYS_futex, address, FUTEX_WAKE_BITSET_PRIVATE, INT_MAX, nullptr, nullptr, kReaderWaiterBit);
}

}  // namespace vsag

#endif
