
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

#include "fixtures.h"

#include <assert.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>

#include "fmt/format.h"
#include "simd/simd.h"

namespace fixtures {

const int RABITQ_MIN_RACALL_DIM = 960;

std::vector<int>
get_common_used_dims(uint64_t count, int seed) {
    const std::vector<int> dims = {
        7,    8,   9,    // generic (dim < 32)
        32,   33,  48,   // sse(32) + generic(dim < 16)
        64,   65,  70,   // avx(64) + generic(dim < 16)
        96,   97,  109,  // avx(64) + sse(32) + generic(dim < 16)
        128,  129,       // avx512(128) + generic(dim < 16)
        160,  161,       // avx512(128) + sse(32) + generic(dim < 16)
        192,  193,       // avx512(128) + avx(64) + generic(dim < 16)
        224,  225,       // avx512(128) + avx(64) + sse(32) + generic(dim < 16)
        256,  512,       // common used dims
        784,  960,       // common used dims
        1024, 1536};     // common used dims
    if (count == -1 || count >= dims.size()) {
        return dims;
    }
    std::vector<int> result(dims.begin(), dims.end());
    std::shuffle(result.begin(), result.end(), std::mt19937(seed));
    result.resize(count);
    return result;
}

std::pair<std::vector<float>, std::vector<uint8_t>>
GenerateBinaryVectorsAndCodes(uint32_t count, uint32_t dim, int seed) {
    assert(count % 2 == 0);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distrib_real(-1, 1);
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim));

    uint32_t code_size = (dim + 7) / 8;
    std::vector<uint8_t> codes(count * code_size);
    std::vector<float> vectors(count * dim);

    for (uint32_t i = 0; i < count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            if (distrib_real(rng) >= 0.0f) {
                codes[i * code_size + d / 8] |= (1 << (d % 8));
                vectors[i * dim + d] = inv_sqrt_d;
            } else {
                vectors[i * dim + d] = -inv_sqrt_d;
            }
        }
    }

    return {vectors, codes};
}

std::vector<float>
generate_vectors(uint64_t count, uint32_t dim, bool need_normalize, int seed) {
    return std::move(GenerateVectors<float>(count, dim, seed, need_normalize));
}

std::vector<int8_t>
generate_int8_codes(uint64_t count, uint32_t dim, int seed) {
    return GenerateVectors<int8_t>(count, dim, seed);
}

std::vector<uint8_t>
generate_int4_codes(uint64_t count, uint32_t dim, int seed) {
    return generate_uint8_codes(count, dim, seed);
}

std::vector<uint8_t>
generate_uint8_codes(uint64_t count, uint32_t dim, int seed) {
    return GenerateVectors<uint8_t>(count, dim, seed);
}

std::tuple<std::vector<int64_t>, std::vector<float>>
generate_ids_and_vectors(int64_t num_vectors, int64_t dim, bool need_normalize, int seed) {
    std::vector<int64_t> ids(num_vectors);
    std::iota(ids.begin(), ids.end(), 0);
    return {ids, generate_vectors(num_vectors, dim, need_normalize, seed)};
}

std::vector<char>
generate_extra_infos(uint64_t count, uint32_t size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int8_t> distrib_real(std::numeric_limits<int8_t>::lowest(),
                                                       std::numeric_limits<int8_t>::max());
    std::vector<char> vectors(size * count);
    for (int64_t i = 0; i < size * count; ++i) {
        vectors[i] = static_cast<char>(distrib_real(rng));
    }
    return vectors;
}

std::string
generate_hnsw_build_parameters_string(const std::string& metric_type, int64_t dim) {
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 64,
            "ef_construction": 500
        }}
    }}
    )";
    auto build_parameters_str = fmt::format(parameter_temp, metric_type, dim);
    return build_parameters_str;
}

std::vector<IOItem>
GenTestItems(uint64_t count, uint64_t max_length, uint64_t max_index) {
    std::vector<IOItem> result(count);
    std::unordered_set<uint64_t> maps;
    for (auto& item : result) {
        while (true) {
            item.start_ = (random() % max_index) * max_length;
            if (not maps.count(item.start_)) {
                maps.insert(item.start_);
                break;
            }
        };
        item.length_ = random() % max_length + 1;
        item.data_ = new uint8_t[item.length_];
        auto vec = fixtures::generate_vectors(1, max_length, false, random());
        memcpy(item.data_, vec.data(), item.length_);
    }
    return result;
}

uint64_t
GetFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    return static_cast<uint64_t>(file.tellg());
}

std::vector<std::string>
SplitString(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(s);

    while (std::getline(ss, token, delimiter)) {
        tokens.emplace_back(token);
    }

    return tokens;
}

}  // namespace fixtures
