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

#include "pyramid_path_store.h"

#include <limits>
#include <mutex>

#include "common.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag_exception.h"

namespace vsag {

std::string
ReadPyramidPathString(StreamReader& reader) {
    uint64_t length = 0;
    StreamReader::ReadObj(reader, length);
    const auto cursor = reader.GetCursor();
    const auto reader_length = reader.Length();
    if (cursor > reader_length || length > reader_length - cursor) {
        throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path string length");
    }
    // StreamReader::ReadString allocates from the serialized length before reading the payload.
    // Validate against the bounded reader first so malformed input cannot request an oversized
    // allocation.
    std::string value(length, '\0');
    if (length > 0) {
        reader.Read(value.data(), length);
    }
    return value;
}

void
PyramidPathStore::Record(const std::string* paths,
                         const Vector<int64_t>& data_biases,
                         int64_t first_inner_id) {
    CHECK_ARGUMENT(paths != nullptr, "paths must not be null");
    CHECK_ARGUMENT(first_inner_id >= 0, "first inner id must not be negative");
    if (data_biases.empty()) {
        return;
    }

    const auto begin = static_cast<uint64_t>(first_inner_id);
    CHECK_ARGUMENT(begin <= std::numeric_limits<uint64_t>::max() - data_biases.size(),
                   "Pyramid path count overflow");
    const auto end = begin + data_biases.size();
    std::unique_lock lock(mutex_);
    if (paths_by_inner_id_.size() < end) {
        paths_by_inner_id_.resize(end);
        has_path_.resize(end, 0);
    }
    for (uint64_t offset = 0; offset < data_biases.size(); ++offset) {
        const auto inner_id = begin + offset;
        paths_by_inner_id_[inner_id] = paths[data_biases[offset]];
        has_path_[inner_id] = 1;
    }
}

void
PyramidPathStore::Record(const std::string* paths, uint64_t count) {
    CHECK_ARGUMENT(paths != nullptr, "paths must not be null");
    std::unique_lock lock(mutex_);
    paths_by_inner_id_.resize(count);
    has_path_.resize(count, 0);
    for (uint64_t inner_id = 0; inner_id < count; ++inner_id) {
        paths_by_inner_id_[inner_id] = paths[inner_id];
        has_path_[inner_id] = 1;
    }
}

bool
PyramidPathStore::GetPaths(const Vector<InnerIdType>& inner_ids, std::string* paths) const {
    std::shared_lock lock(mutex_);
    for (uint64_t offset = 0; offset < inner_ids.size(); ++offset) {
        const auto inner_id = static_cast<uint64_t>(inner_ids[offset]);
        if (inner_id >= has_path_.size() || has_path_[inner_id] == 0) {
            return false;
        }
        paths[offset] = paths_by_inner_id_[inner_id];
    }
    return true;
}

void
PyramidPathStore::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(mutex_);
    if (paths_by_inner_id_.size() != has_path_.size()) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Pyramid path store size mismatch");
    }
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(paths_by_inner_id_.size()));
    for (uint64_t inner_id = 0; inner_id < has_path_.size(); ++inner_id) {
        const auto present = has_path_[inner_id];
        if (present > 1) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Pyramid path store has invalid presence value");
        }
        StreamWriter::WriteObj(writer, present);
        if (present != 0) {
            StreamWriter::WriteString(writer, paths_by_inner_id_[inner_id]);
        }
    }
}

void
PyramidPathStore::Deserialize(StreamReader& reader, uint64_t max_count) {
    uint64_t slot_count = 0;
    StreamReader::ReadObj(reader, slot_count);
    if (slot_count > max_count) {
        throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path slot count");
    }
    const auto cursor = reader.GetCursor();
    const auto reader_length = reader.Length();
    if (cursor > reader_length || slot_count > reader_length - cursor) {
        throw VsagException(ErrorType::READ_ERROR,
                            "corrupted Pyramid path slots exceed remaining payload");
    }

    Vector<std::string> restored_paths(slot_count, std::string{}, allocator_);
    Vector<uint8_t> restored_has_path(slot_count, 0, allocator_);
    for (uint64_t inner_id = 0; inner_id < slot_count; ++inner_id) {
        uint8_t present = 0;
        StreamReader::ReadObj(reader, present);
        if (present > 1) {
            throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path presence value");
        }
        restored_has_path[inner_id] = present;
        if (present != 0) {
            restored_paths[inner_id] = ReadPyramidPathString(reader);
        }
    }

    std::unique_lock lock(mutex_);
    paths_by_inner_id_.swap(restored_paths);
    has_path_.swap(restored_has_path);
}

uint64_t
PyramidPathStore::Size() const {
    std::shared_lock lock(mutex_);
    return paths_by_inner_id_.size();
}

}  // namespace vsag
