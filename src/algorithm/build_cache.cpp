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

#include "build_cache.h"

#include "impl/allocator/default_allocator.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

uint64_t
remaining_bytes(StreamReader& reader) {
    const auto cursor = reader.GetCursor();
    const auto length = reader.Length();
    if (cursor > length) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted build cache cursor");
    }
    return length - cursor;
}

void
require_remaining(StreamReader& reader, uint64_t size, const char* field) {
    if (size > remaining_bytes(reader)) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            fmt::format("corrupted build cache {} length", field));
    }
}

std::string
read_string(StreamReader& reader) {
    uint64_t size = 0;
    StreamReader::ReadObj(reader, size);
    require_remaining(reader, size, "string");
    std::vector<char> buffer(size);
    reader.Read(buffer.data(), size);
    return {buffer.data(), size};
}

void
read_neighbors(StreamReader& reader, Vector<InnerIdType>& neighbors) {
    uint64_t size = 0;
    StreamReader::ReadObj(reader, size);
    if (size > remaining_bytes(reader) / sizeof(InnerIdType)) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted build cache neighbor count");
    }
    neighbors.resize(size);
    reader.Read(reinterpret_cast<char*>(neighbors.data()), size * sizeof(InnerIdType));
}

struct CachedCodeDescriptor {
    std::string quantizer_name;
    uint64_t code_size{0};
    uint64_t payload_size{0};
};

}  // namespace

BuildCache::BuildCache(Allocator* allocator)
    : allocator_(allocator), source_ids_(allocator_), neighbors_(allocator_) {
}

void
BuildCache::Serialize(StreamWriter& writer) const {
    uint64_t source_ids_size = source_ids_.size();
    StreamWriter::WriteObj(writer, source_ids_size);
    Vector<InnerIdType> empty(allocator_);
    for (uint64_t i = 0; i < source_ids_size; ++i) {
        const auto& source_id = source_ids_[i];
        StreamWriter::WriteString(writer, source_id);
        auto it = neighbors_.find(source_id);
        if (it != neighbors_.end()) {
            StreamWriter::WriteVector(writer, it->second);
        } else {
            StreamWriter::WriteVector(writer, empty);
        }
    }
}

void
BuildCache::Deserialize(StreamReader& reader) {
    uint64_t source_ids_size = 0;
    StreamReader::ReadObj(reader, source_ids_size);
    if (source_ids_size > remaining_bytes(reader) / (sizeof(uint64_t) * 2)) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted build cache source-id count");
    }
    source_ids_.clear();
    source_ids_.reserve(source_ids_size);
    neighbors_.clear();
    for (uint64_t i = 0; i < source_ids_size; ++i) {
        auto source_id = read_string(reader);
        source_ids_.push_back(source_id);
        Vector<InnerIdType> neighbors(allocator_);
        read_neighbors(reader, neighbors);
        if (!neighbors.empty()) {
            neighbors_.emplace(std::move(source_id), std::move(neighbors));
        }
    }
}

const Vector<InnerIdType>*
BuildCache::FindNeighborInnerIds(const std::string& source_id) const {
    auto it = neighbors_.find(source_id);
    if (it == neighbors_.end()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<std::string>
BuildCache::GetNeighbors(const std::string& source_id) const {
    std::vector<std::string> result;
    auto it = neighbors_.find(source_id);
    if (it == neighbors_.end()) {
        return result;
    }
    const auto& inner_ids = it->second;
    result.reserve(inner_ids.empty() ? 0 : inner_ids.size() - 1);
    for (uint64_t i = 1; i < inner_ids.size(); ++i) {
        const auto& inner_id = inner_ids[i];
        if (static_cast<uint64_t>(inner_id) < source_ids_.size()) {
            result.push_back(source_ids_[inner_id]);
        }
    }
    return result;
}

void
SerializeBuildCacheCodes(StreamWriter& writer,
                         const std::vector<FlattenInterfacePtr>& codes,
                         uint64_t expected_count) {
    bool valid = not codes.empty();
    for (const auto& cell : codes) {
        valid = valid && cell != nullptr && cell->TotalCount() == expected_count;
    }

    const uint64_t cell_count = valid ? codes.size() : 0;
    StreamWriter::WriteObj(writer, cell_count);
    if (not valid) {
        return;
    }

    std::vector<uint64_t> payload_sizes;
    payload_sizes.reserve(codes.size());
    for (const auto& cell : codes) {
        StreamWriter::WriteString(writer, cell->GetQuantizerName());
        StreamWriter::WriteObj(writer, cell->GetQuantizerCodeSize());
        const auto payload_size = cell->CalcSerializeSize();
        StreamWriter::WriteObj(writer, payload_size);
        payload_sizes.push_back(payload_size);
    }
    for (uint64_t i = 0; i < codes.size(); ++i) {
        const auto begin = writer.GetCursor();
        codes[i]->Serialize(writer);
        if (writer.GetCursor() - begin != payload_sizes[i]) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                "build cache code serialization size changed");
        }
    }
}

bool
DeserializeBuildCacheCodes(StreamReader& reader,
                           const std::vector<FlattenInterfacePtr>& codes,
                           uint64_t expected_count) {
    uint64_t cell_count = 0;
    StreamReader::ReadObj(reader, cell_count);
    if (cell_count > 16) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted build cache code-cell count");
    }

    std::vector<CachedCodeDescriptor> descriptors;
    descriptors.reserve(cell_count);
    for (uint64_t i = 0; i < cell_count; ++i) {
        CachedCodeDescriptor descriptor;
        descriptor.quantizer_name = read_string(reader);
        StreamReader::ReadObj(reader, descriptor.code_size);
        StreamReader::ReadObj(reader, descriptor.payload_size);
        descriptors.emplace_back(std::move(descriptor));
    }

    bool compatible = cell_count > 0 && cell_count == codes.size();
    if (compatible) {
        for (uint64_t i = 0; i < cell_count; ++i) {
            compatible = compatible && codes[i] != nullptr &&
                         descriptors[i].quantizer_name == codes[i]->GetQuantizerName() &&
                         descriptors[i].code_size == codes[i]->GetQuantizerCodeSize();
        }
    }

    for (uint64_t i = 0; i < cell_count; ++i) {
        const auto payload_size = descriptors[i].payload_size;
        require_remaining(reader, payload_size, "code payload");
        if (not compatible) {
            SkipForward(reader, payload_size);
            continue;
        }
        auto payload = reader.Slice(payload_size);
        codes[i]->Deserialize(payload);
        if (payload.GetCursor() != payload.Length()) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                "build cache code payload was not fully consumed");
        }
    }

    if (compatible) {
        for (const auto& cell : codes) {
            if (cell->TotalCount() != expected_count) {
                throw VsagException(ErrorType::INVALID_BINARY,
                                    "build cache code count does not match source ids");
            }
        }
    }
    return compatible;
}

void
SerializeBuildCacheSourceIds(StreamWriter& writer, const Vector<std::string>& source_ids) {
    const uint64_t count = source_ids.size();
    StreamWriter::WriteObj(writer, count);
    for (const auto& source_id : source_ids) {
        StreamWriter::WriteString(writer, source_id);
    }
}

void
DeserializeBuildCacheSourceIds(StreamReader& reader, Vector<std::string>& source_ids) {
    uint64_t count = 0;
    StreamReader::ReadObj(reader, count);
    if (count > remaining_bytes(reader) / sizeof(uint64_t)) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted build cache source-id count");
    }
    source_ids.clear();
    source_ids.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        source_ids.emplace_back(read_string(reader));
    }
}

bool
BuildCacheSourceIdsMatch(const Vector<std::string>& cached_source_ids,
                         const std::string* source_ids,
                         uint64_t count) {
    if (source_ids == nullptr || cached_source_ids.size() != count) {
        return false;
    }
    for (uint64_t i = 0; i < count; ++i) {
        if (cached_source_ids[i].empty() || cached_source_ids[i] != source_ids[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace vsag
