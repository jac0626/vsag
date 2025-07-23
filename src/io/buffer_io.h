
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

#include "basic_io.h"
#include "buffer_io_parameter.h"
#include "index/index_common_param.h"

namespace vsag {

class BufferIO : public BasicIO<BufferIO> {
public:
    BufferIO(std::string filename, Allocator* allocator);

    explicit BufferIO(const BufferIOParameterPtr& io_param, const IndexCommonParam& common_param);

    explicit BufferIO(const IOParamPtr& param, const IndexCommonParam& common_param);

    ~BufferIO() override {
        close(this->fd_);
        // remove file
        std::filesystem::remove(this->filepath_);
    }

    void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const;

    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const;

    void
    ReleaseImpl(const uint8_t* data) const;

    bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

    void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64);

    static bool
    InMemoryImpl();

    void
    InitIOImpl(const IOParamPtr& io_param) {
    }

private:
    std::string filepath_{};

    int fd_{-1};
};
}  // namespace vsag
