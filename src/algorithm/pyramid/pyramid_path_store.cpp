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

#include <algorithm>
#include <limits>
#include <mutex>

#include "common.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

enum class PathRowState : uint8_t {
    HOLE = 0,
    SINGLE_PATH = 1,
    PATH_LIST = 2,
};

constexpr uint64_t MISSING_PATH_OFFSET = std::numeric_limits<uint64_t>::max();

void
reserve_paths_geometrically(Vector<std::string>& paths, uint64_t required_capacity) {
    const auto current_capacity = static_cast<uint64_t>(paths.capacity());
    if (required_capacity <= current_capacity) {
        return;
    }

    const auto max_capacity = static_cast<uint64_t>(paths.max_size());
    auto target_capacity = std::max<uint64_t>(current_capacity, 1);
    if (target_capacity <= max_capacity / 2) {
        target_capacity *= 2;
    } else {
        target_capacity = max_capacity;
    }
    paths.reserve(std::max(required_capacity, target_capacity));
}

}  // namespace

std::string
ReadPyramidPathString(StreamReader& reader) {
    uint64_t length = 0;
    StreamReader::ReadObj(reader, length);
    const auto cursor = reader.GetCursor();
    const auto reader_length = reader.Length();
    if (cursor > reader_length || length > reader_length - cursor) {
        throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path string length");
    }
    if (length > std::string{}.max_size()) {
        throw VsagException(ErrorType::READ_ERROR, "Pyramid path string is too large");
    }
    std::string value(length, '\0');
    if (length > 0) {
        reader.Read(value.data(), length);
    }
    return value;
}

void
PyramidPathStore::Writer::Prepare(uint64_t slot_count, uint64_t additional_path_count) {
    constexpr uint64_t max_slot_count =
        static_cast<uint64_t>(std::numeric_limits<InnerIdType>::max()) + 1;
    CHECK_ARGUMENT(slot_count <= max_slot_count, "Pyramid path slot count is too large");
    CHECK_ARGUMENT(slot_count <= store_.offsets_.max_size(),
                   "Pyramid path slot count is too large");
    CHECK_ARGUMENT(slot_count <= store_.counts_.max_size(), "Pyramid path slot count is too large");

    const auto path_count = static_cast<uint64_t>(store_.paths_.size());
    CHECK_ARGUMENT(additional_path_count <= store_.paths_.max_size() - path_count,
                   "Pyramid path count is too large");

    const auto old_offset_count = store_.offsets_.size();
    const auto old_count_count = store_.counts_.size();
    try {
        if (slot_count > old_offset_count) {
            store_.offsets_.resize(slot_count, MISSING_PATH_OFFSET);
        }
        if (slot_count > old_count_count) {
            store_.counts_.resize(slot_count, 0);
        }
        reserve_paths_geometrically(store_.paths_, path_count + additional_path_count);
    } catch (...) {
        store_.offsets_.resize(old_offset_count);
        store_.counts_.resize(old_count_count);
        throw;
    }
}

void
PyramidPathStore::Writer::Insert(InnerIdType inner_id, const std::string& path) {
    Insert(inner_id, &path, 1);
}

void
PyramidPathStore::Writer::Insert(InnerIdType inner_id,
                                 const std::string* paths,
                                 uint64_t path_count) {
    CHECK_ARGUMENT(path_count <= std::numeric_limits<uint16_t>::max(),
                   "too many Pyramid paths for one inner id");
    if (path_count > 0) {
        CHECK_ARGUMENT(paths != nullptr, "Pyramid paths must not be null");
    }
    const auto slot = static_cast<uint64_t>(inner_id);
    if (slot < store_.offsets_.size()) {
        CHECK_ARGUMENT(store_.offsets_[slot] == MISSING_PATH_OFFSET,
                       "inner id already has Pyramid paths");
    }

    const auto old_path_count = static_cast<uint64_t>(store_.paths_.size());
    const auto old_offset_count = store_.offsets_.size();
    const auto old_count_count = store_.counts_.size();
    try {
        Prepare(slot + 1, path_count);
        for (uint64_t offset = 0; offset < path_count; ++offset) {
            store_.paths_.emplace_back(paths[offset]);
        }
        store_.offsets_[slot] = old_path_count;
        store_.counts_[slot] = static_cast<uint16_t>(path_count);
    } catch (...) {
        store_.paths_.resize(old_path_count);
        store_.offsets_.resize(old_offset_count);
        store_.counts_.resize(old_count_count);
        throw;
    }
}

PyramidPathStore::Writer
PyramidPathStore::AcquireWriter() {
    return Writer(*this);
}

bool
PyramidPathStore::GetPaths(const Vector<InnerIdType>& inner_ids, std::string* paths) const {
    std::shared_lock lock(mutex_);
    if (offsets_.size() != counts_.size()) {
        return false;
    }
    for (const auto id : inner_ids) {
        const auto inner_id = static_cast<uint64_t>(id);
        if (inner_id >= offsets_.size() || counts_[inner_id] != 1 ||
            offsets_[inner_id] >= paths_.size()) {
            return false;
        }
    }
    for (uint64_t offset = 0; offset < inner_ids.size(); ++offset) {
        paths[offset] = paths_[offsets_[inner_ids[offset]]];
    }
    return true;
}

bool
PyramidPathStore::GetPathRows(const Vector<InnerIdType>& inner_ids,
                              std::vector<std::vector<std::string>>& path_rows) const {
    std::shared_lock lock(mutex_);
    if (offsets_.size() != counts_.size()) {
        return false;
    }
    std::vector<std::vector<std::string>> restored_rows;
    restored_rows.reserve(inner_ids.size());
    for (const auto inner_id : inner_ids) {
        const auto slot = static_cast<uint64_t>(inner_id);
        if (slot >= offsets_.size()) {
            return false;
        }
        const auto path_offset = offsets_[slot];
        const auto path_count = static_cast<uint64_t>(counts_[slot]);
        if (path_offset == MISSING_PATH_OFFSET || path_offset > paths_.size() ||
            path_count > paths_.size() - path_offset) {
            return false;
        }
        const auto begin_offset = static_cast<Vector<std::string>::difference_type>(path_offset);
        const auto end_offset =
            static_cast<Vector<std::string>::difference_type>(path_offset + path_count);
        restored_rows.emplace_back(paths_.begin() + begin_offset, paths_.begin() + end_offset);
    }
    path_rows = std::move(restored_rows);
    return true;
}

void
PyramidPathStore::Serialize(StreamWriter& writer) const {
    std::shared_lock lock(mutex_);
    if (offsets_.size() != counts_.size()) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "Pyramid path store has inconsistent slot arrays");
    }
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(offsets_.size()));
    for (uint64_t inner_id = 0; inner_id < offsets_.size(); ++inner_id) {
        const auto path_offset = offsets_[inner_id];
        const auto path_count = counts_[inner_id];
        if (path_offset == MISSING_PATH_OFFSET) {
            StreamWriter::WriteObj(writer, static_cast<uint8_t>(PathRowState::HOLE));
            continue;
        }
        if (path_offset > paths_.size() || path_count > paths_.size() - path_offset) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "Pyramid path store has an invalid path range");
        }

        if (path_count == 1) {
            StreamWriter::WriteObj(writer, static_cast<uint8_t>(PathRowState::SINGLE_PATH));
            StreamWriter::WriteString(writer, paths_[path_offset]);
            continue;
        }

        StreamWriter::WriteObj(writer, static_cast<uint8_t>(PathRowState::PATH_LIST));
        StreamWriter::WriteObj(writer, path_count);
        for (uint64_t offset = 0; offset < path_count; ++offset) {
            StreamWriter::WriteString(writer, paths_[path_offset + offset]);
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
    constexpr uint64_t max_slot_count =
        static_cast<uint64_t>(std::numeric_limits<InnerIdType>::max()) + 1;
    if (slot_count > max_slot_count) {
        throw VsagException(ErrorType::READ_ERROR, "Pyramid path slot count is too large");
    }
    const auto cursor = reader.GetCursor();
    const auto reader_length = reader.Length();
    if (cursor > reader_length || slot_count > reader_length - cursor) {
        throw VsagException(ErrorType::READ_ERROR,
                            "corrupted Pyramid path slots exceed remaining payload");
    }

    Vector<uint64_t> restored_offsets(allocator_);
    Vector<uint16_t> restored_counts(allocator_);
    if (slot_count > restored_offsets.max_size() || slot_count > restored_counts.max_size()) {
        throw VsagException(ErrorType::READ_ERROR, "Pyramid path slot count is too large");
    }
    restored_offsets.resize(slot_count, MISSING_PATH_OFFSET);
    restored_counts.resize(slot_count, 0);
    Vector<std::string> restored_paths(allocator_);
    for (uint64_t inner_id = 0; inner_id < slot_count; ++inner_id) {
        uint8_t raw_state = 0;
        StreamReader::ReadObj(reader, raw_state);
        const auto state = static_cast<PathRowState>(raw_state);
        if (state == PathRowState::HOLE) {
            continue;
        }

        const auto path_offset = static_cast<uint64_t>(restored_paths.size());
        if (state == PathRowState::SINGLE_PATH) {
            reserve_paths_geometrically(restored_paths, path_offset + 1);
            restored_paths.emplace_back(ReadPyramidPathString(reader));
            restored_offsets[inner_id] = path_offset;
            restored_counts[inner_id] = 1;
        } else if (state == PathRowState::PATH_LIST) {
            uint16_t path_count = 0;
            StreamReader::ReadObj(reader, path_count);
            if (path_count == 1) {
                throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path row state");
            }
            const auto path_cursor = reader.GetCursor();
            const auto path_reader_length = reader.Length();
            if (path_cursor > path_reader_length ||
                path_count > (path_reader_length - path_cursor) / sizeof(uint64_t) ||
                path_count > restored_paths.max_size() - path_offset) {
                throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path row count");
            }
            reserve_paths_geometrically(restored_paths, path_offset + path_count);
            for (uint64_t offset = 0; offset < path_count; ++offset) {
                restored_paths.emplace_back(ReadPyramidPathString(reader));
            }
            restored_offsets[inner_id] = path_offset;
            restored_counts[inner_id] = path_count;
        } else {
            throw VsagException(ErrorType::READ_ERROR, "corrupted Pyramid path row state");
        }
    }

    std::unique_lock lock(mutex_);
    offsets_.swap(restored_offsets);
    counts_.swap(restored_counts);
    paths_.swap(restored_paths);
}

uint64_t
PyramidPathStore::Size() const {
    std::shared_lock lock(mutex_);
    return offsets_.size();
}

}  // namespace vsag
