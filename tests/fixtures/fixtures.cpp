
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


template <typename T, typename Gen>
void
FillIntegerValues(vsag::AttributeValue<T>* attr, uint32_t count, Gen& gen) {
    using Limits = std::numeric_limits<T>;
    std::uniform_int_distribution<int64_t> dist(Limits::min(), Limits::max());
    for (uint32_t i = 0; i < count; ++i) {
        attr->GetValue().emplace_back(static_cast<T>(dist(gen)));
    }
}

template <typename Gen>
void
FillStringValues(vsag::AttributeValue<std::string>* attr,
                 uint32_t count,
                 uint32_t max_len,
                 Gen& gen) {
    std::uniform_int_distribution<uint32_t> len_dist(1, max_len);
    std::uniform_int_distribution<int> char_dist('a', 'z');
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t len = len_dist(gen);
        std::string str;
        str.reserve(len);
        for (uint32_t c = 0; c < len; ++c) {
            str += char_dist(gen);
        }
        attr->GetValue().emplace_back(str);
    }
}

static std::vector<int>
select_k_numbers(int64_t n, int k) {
    std::vector<int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<> dist(i, static_cast<int>(n - 1));
        std::swap(numbers[i], numbers[dist(gen)]);
    }
    numbers.resize(k);
    return numbers;
}

template <typename Gen>
vsag::Attribute*
CreateAttribute(std::string term_name,
                vsag::AttrValueType type,
                uint32_t term_count,
                uint32_t max_str_len,
                Gen& gen) {
    switch (type) {
        case vsag::AttrValueType::INT32: {
            auto attr = new vsag::AttributeValue<int32_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::UINT32: {
            auto attr = new vsag::AttributeValue<uint32_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::INT64: {
            auto attr = new vsag::AttributeValue<int64_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::UINT64: {
            auto attr = new vsag::AttributeValue<uint64_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::INT8: {
            auto attr = new vsag::AttributeValue<int8_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::UINT8: {
            auto attr = new vsag::AttributeValue<uint8_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::INT16: {
            auto attr = new vsag::AttributeValue<int16_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::UINT16: {
            auto attr = new vsag::AttributeValue<uint16_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::STRING: {
            auto attr = new vsag::AttributeValue<std::string>();
            attr->name_ = term_name;
            FillStringValues(attr, term_count, max_str_len, gen);
            return attr;
        }
        default:
            return nullptr;
    }
}

vsag::AttributeSet*
generate_attributes(uint64_t count, uint32_t max_term_count, uint32_t max_value_count, int seed) {
    auto* results = new vsag::AttributeSet[count];
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> term_count_dist(1, max_term_count);
    auto term_count = term_count_dist(gen);
    std::vector<std::pair<std::string, vsag::AttrValueType>> terms(term_count);

    std::uniform_int_distribution<int> type_dist(1, 9);
    std::uniform_int_distribution<int> value_count_dist(1, max_value_count);

    for (uint64_t i = 0; i < term_count; ++i) {
        std::string term_name = fmt::format("term_{}", i);
        auto term_type = static_cast<vsag::AttrValueType>(type_dist(gen));
        if (term_type == vsag::AttrValueType::UINT64) {
            term_type = vsag::AttrValueType::INT64;
        }
        terms[i] = {term_name, term_type};
    }

    for (uint32_t i = 0; i < count; ++i) {
        auto cur_term_count = RandomValue(1, term_count);
        results[i].attrs_.reserve(cur_term_count);
        auto idxes = select_k_numbers(term_count, cur_term_count);

        for (uint32_t j = 0; j < cur_term_count; ++j) {
            auto term_id = idxes[j];
            auto& term_name = terms[term_id].first;
            auto& term_type = terms[term_id].second;
            auto attr = CreateAttribute(term_name, term_type, value_count_dist(gen), 10, gen);
            results[i].attrs_.emplace_back(attr);
        }
    }
    return results;
}

float
test_knn_recall(const vsag::IndexPtr& index,
                const std::string& search_parameters,
                int64_t num_vectors,
                int64_t dim,
                std::vector<int64_t>& ids,
                std::vector<float>& vectors) {
    int64_t correct = 0;
    for (int64_t i = 0; i < num_vectors; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + i * dim)->Owner(false);
        auto result = index->KnnSearch(query, 10, search_parameters).value();
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            if (ids[i] == result->GetIds()[j]) {
                ++correct;
                break;
            }
        }
    }

    float recall = 1.0 * correct / num_vectors;
    return recall;
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
