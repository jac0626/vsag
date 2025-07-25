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

#if defined(ENABLE_SVE)
#include <arm_sve.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

constexpr auto
generate_bit_lookup_table() {
    std::array<std::array<uint8_t, 8>, 256> table{};
    for (int byte_value = 0; byte_value < 256; ++byte_value) {
        for (int bit_pos = 0; bit_pos < 8; ++bit_pos) {
            table[byte_value][bit_pos] = ((byte_value >> bit_pos) & 1) ? 1 : 0;
        }
    }
    return table;
}

// 在编译时生成并初始化全局查找表，无任何运行时开销
static constexpr auto g_bit_lookup_table = generate_bit_lookup_table();
#include "simd.h"

#if defined(__ARM_FEATURE_SVE_BF16)
#include <arm_bf16.h>
#endif
#if defined(__ARM_FEATURE_SVE_FP16)
#include <arm_fp16.h>
#endif

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace vsag::sve {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((size_t*)qty_ptr);
    return sve::FP32ComputeL2Sqr(pVect1, pVect2, qty);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* pVect1 = (float*)pVect1v;
    auto* pVect2 = (float*)pVect2v;
    auto qty = *((size_t*)qty_ptr);
    return sve::FP32ComputeIP(pVect1, pVect2, qty);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - sve::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(ENABLE_SVE)
    auto* pVect1 = (const int8_t*)pVect1v;
    auto* pVect2 = (const int8_t*)pVect2v;
    auto qty = *((size_t*)qty_ptr);

    svint32_t sum_vec = svdup_s32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t pg = svwhilelt_b8(i, qty);
    do {
        svint8_t v1 = svld1_s8(pg, pVect1 + i);
        svint8_t v2 = svld1_s8(pg, pVect2 + i);
        sum_vec = svdot_s32(sum_vec, v1, v2);
        i += step;
        pg = svwhilelt_b8(i, qty);
    } while (svptest_first(svptrue_b8(), pg));

    return static_cast<float>(svaddv_s32(svptrue_b32(), sum_vec));
#else
    return neon::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
#endif
}

float
INT8InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -sve::INT8InnerProduct(pVect1, pVect2, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_SVE)
    const auto* float_centers = (const float*)single_dim_centers;
    auto* float_result = (float*)result;
    uint64_t num_floats_per_vector = svcntw();
    svfloat32_t val_vec = svdup_f32(single_dim_val);
    int i = 0;
    do {
        svbool_t pg = svwhilelt_b32(i, 256);
        svfloat32_t centers_vec = svld1_f32(pg, float_centers + i);
        svfloat32_t result_vec = svld1_f32(pg, float_result + i);
        svfloat32_t diff_vec = svsub_f32_m(pg, centers_vec, val_vec);
        result_vec = svmad_f32_m(pg, diff_vec, diff_vec, result_vec);
        svst1_f32(pg, float_result + i, result_vec);
        i += num_floats_per_vector;
    } while (i < 256);
#else
    neon::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
#endif
}

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;

    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svprfw(svptrue_b32(), query + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes + i + step, SV_PLDL1KEEP);

        svfloat32_t q_vec = svld1_f32(pg, query + i);
        svfloat32_t c_vec = svld1_f32(pg, codes + i);

        sum_vec = svmla_f32_m(pg, sum_vec, q_vec, c_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svprfw(svptrue_b32(), query + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes + i + step, SV_PLDL1KEEP);

        svfloat32_t q_vec = svld1_f32(pg, query + i);
        svfloat32_t c_vec = svld1_f32(pg, codes + i);

        svfloat32_t diff = svsub_f32_z(pg, q_vec, c_vec);

        sum_vec = svmla_f32_m(pg, sum_vec, diff, diff);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4) {
#if defined(ENABLE_SVE)

    svfloat32_t res1_vec = svdup_f32(0.0f);
    svfloat32_t res2_vec = svdup_f32(0.0f);
    svfloat32_t res3_vec = svdup_f32(0.0f);
    svfloat32_t res4_vec = svdup_f32(0.0f);

    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svprfw(svptrue_b32(), query + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes1 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes2 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes3 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes4 + i + step, SV_PLDL1KEEP);

        svfloat32_t q_vec = svld1_f32(pg, query + i);

        svfloat32_t c1_vec = svld1_f32(pg, codes1 + i);
        res1_vec = svmla_f32_m(pg, res1_vec, q_vec, c1_vec);

        svfloat32_t c2_vec = svld1_f32(pg, codes2 + i);
        res2_vec = svmla_f32_m(pg, res2_vec, q_vec, c2_vec);

        svfloat32_t c3_vec = svld1_f32(pg, codes3 + i);
        res3_vec = svmla_f32_m(pg, res3_vec, q_vec, c3_vec);

        svfloat32_t c4_vec = svld1_f32(pg, codes4 + i);
        res4_vec = svmla_f32_m(pg, res4_vec, q_vec, c4_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));

    result1 = svaddv_f32(svptrue_b32(), res1_vec);
    result2 = svaddv_f32(svptrue_b32(), res2_vec);
    result3 = svaddv_f32(svptrue_b32(), res3_vec);
    result4 = svaddv_f32(svptrue_b32(), res4_vec);
#else
    neon::FP32ComputeIPBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4) {
#if defined(ENABLE_SVE)
    svfloat32_t res1_vec = svdup_f32(0.0f);
    svfloat32_t res2_vec = svdup_f32(0.0f);
    svfloat32_t res3_vec = svdup_f32(0.0f);
    svfloat32_t res4_vec = svdup_f32(0.0f);

    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svprfw(svptrue_b32(), query + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes1 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes2 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes3 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes4 + i + step, SV_PLDL1KEEP);

        svfloat32_t q_vec = svld1_f32(pg, query + i);

        svfloat32_t c1_vec = svld1_f32(pg, codes1 + i);
        svfloat32_t d1_vec = svsub_f32_z(pg, q_vec, c1_vec);
        res1_vec = svmla_f32_m(pg, res1_vec, d1_vec, d1_vec);

        svfloat32_t c2_vec = svld1_f32(pg, codes2 + i);
        svfloat32_t d2_vec = svsub_f32_z(pg, q_vec, c2_vec);
        res2_vec = svmla_f32_m(pg, res2_vec, d2_vec, d2_vec);

        svfloat32_t c3_vec = svld1_f32(pg, codes3 + i);
        svfloat32_t d3_vec = svsub_f32_z(pg, q_vec, c3_vec);
        res3_vec = svmla_f32_m(pg, res3_vec, d3_vec, d3_vec);

        svfloat32_t c4_vec = svld1_f32(pg, codes4 + i);
        svfloat32_t d4_vec = svsub_f32_z(pg, q_vec, c4_vec);
        res4_vec = svmla_f32_m(pg, res4_vec, d4_vec, d4_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));

    result1 = svaddv_f32(svptrue_b32(), res1_vec);
    result2 = svaddv_f32(svptrue_b32(), res2_vec);
    result3 = svaddv_f32(svptrue_b32(), res3_vec);
    result4 = svaddv_f32(svptrue_b32(), res4_vec);
#else
    neon::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(pg, x + i);
        svfloat32_t y_vec = svld1_f32(pg, y + i);
        svfloat32_t z_vec = svsub_f32_z(pg, x_vec, y_vec);
        svst1_f32(pg, z + i, z_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));
#else
    neon::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(pg, x + i);
        svfloat32_t y_vec = svld1_f32(pg, y + i);
        svfloat32_t z_vec = svadd_f32_z(pg, x_vec, y_vec);
        svst1_f32(pg, z + i, z_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));
#else
    neon::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(pg, x + i);
        svfloat32_t y_vec = svld1_f32(pg, y + i);
        svfloat32_t z_vec = svmul_f32_z(pg, x_vec, y_vec);
        svst1_f32(pg, z + i, z_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));
#else
    neon::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(pg, x + i);
        svfloat32_t y_vec = svld1_f32(pg, y + i);
        svfloat32_t z_vec = svdiv_f32_z(pg, x_vec, y_vec);
        svst1_f32(pg, z + i, z_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));
#else
    neon::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svfloat32_t x_vec = svld1_f32(pg, x + i);

        sum_vec = svadd_f32_m(pg, sum_vec, x_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::FP32ReduceAdd(x, dim);
#endif
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(__ARM_FEATURE_SVE_BF16)
    const bfloat16_t* query_bf16 = (const bfloat16_t*)query;
    const bfloat16_t* codes_bf16 = (const bfloat16_t*)codes;

    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;

    const uint64_t step = svcnth();

    svbool_t pg = svwhilelt_b16(i, dim);
    do {
        svprfw(svptrue_b16(), query_bf16 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b16(), codes_bf16 + i + step, SV_PLDL1KEEP);

        svbfloat16_t q_vec = svld1_bf16(pg, query_bf16 + i);
        svbfloat16_t c_vec = svld1_bf16(pg, codes_bf16 + i);

        sum_vec =
            svmla_f32_m(pg, sum_vec, svcvt_f32_bf16_z(pg, q_vec), svcvt_f32_bf16_z(pg, c_vec));

        i += step;
        pg = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b16(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(__ARM_FEATURE_SVE_BF16)
    const bfloat16_t* query_bf16 = (const bfloat16_t*)query;
    const bfloat16_t* codes_bf16 = (const bfloat16_t*)codes;
    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcnth();

    svbool_t pg = svwhilelt_b16(i, dim);
    do {
        svprfw(svptrue_b16(), query_bf16 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b16(), codes_bf16 + i + step, SV_PLDL1KEEP);

        svbfloat16_t q_vec = svld1_bf16(pg, query_bf16 + i);
        svbfloat16_t c_vec = svld1_bf16(pg, codes_bf16 + i);

        svbfloat16_t diff = svsub_bf16_z(pg, q_vec, c_vec);
        svfloat32_t diff_f32 = svcvt_f32_bf16_z(pg, diff);

        sum_vec = svmla_f32_m(pg, sum_vec, diff_f32, diff_f32);

        i += step;
        pg = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b16(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(__ARM_FEATURE_SVE_FP16)
    const _Float16* query_f16 = (const _Float16*)query;
    const _Float16* codes_f16 = (const _Float16*)codes;

    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;

    const uint64_t step = svcnth();

    svbool_t pg = svwhilelt_b16(i, dim);
    do {
        svprfw(svptrue_b16(), query_f16 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b16(), codes_f16 + i + step, SV_PLDL1KEEP);

        svfloat16_t q_vec = svld1_f16(pg, query_f16 + i);
        svfloat16_t c_vec = svld1_f16(pg, codes_f16 + i);

        sum_vec = svmla_f32_m(pg, sum_vec, svcvt_f32_f16_z(pg, q_vec), svcvt_f32_f16_z(pg, c_vec));

        i += step;
        pg = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b16(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(__ARM_FEATURE_SVE_FP16)
    const _Float16* query_f16 = (const _Float16*)query;
    const _Float16* codes_f16 = (const _Float16*)codes;
    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcnth();

    svbool_t pg = svwhilelt_b16(i, dim);
    do {
        svprfw(svptrue_b16(), query_f16 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b16(), codes_f16 + i + step, SV_PLDL1KEEP);

        svfloat16_t q_vec = svld1_f16(pg, query_f16 + i);
        svfloat16_t c_vec = svld1_f16(pg, codes_f16 + i);

        svfloat16_t diff = svsub_f16_z(pg, q_vec, c_vec);
        svfloat32_t diff_f32 = svcvt_f32_f16_z(pg, diff);

        sum_vec = svmla_f32_m(pg, sum_vec, diff_f32, diff_f32);

        i += step;
        pg = svwhilelt_b16(i, dim);
    } while (svptest_first(svptrue_b16(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::FP16ComputeL2Sqr(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum_vec = svdup_f32(0.0f);
    svfloat32_t inv_255_vec = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svprfw(svptrue_b32(), query + i + step, SV_PLDL1KEEP);
        svprfb(svptrue_b8(), codes + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), lower_bound + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), diff + i + step, SV_PLDL1KEEP);

        svuint32_t c_u32_vec = svld1ub_u32(pg, codes + i);
        svfloat32_t c_f32_vec = svcvt_f32_u32_z(pg, c_u32_vec);

        svfloat32_t q_vec = svld1_f32(pg, query + i);
        svfloat32_t lb_vec = svld1_f32(pg, lower_bound + i);
        svfloat32_t d_vec = svld1_f32(pg, diff + i);

        svfloat32_t dequant_vec =
            svmla_f32_m(pg, lb_vec, svmul_f32_m(pg, c_f32_vec, inv_255_vec), d_vec);

        sum_vec = svmla_f32_m(pg, sum_vec, q_vec, dequant_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum_vec = svdup_f32(0.0f);
    svfloat32_t inv_255_vec = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svprfw(svptrue_b32(), query + i + step, SV_PLDL1KEEP);
        svprfb(svptrue_b8(), codes + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), lower_bound + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), diff + i + step, SV_PLDL1KEEP);

        svuint32_t c_u32_vec = svld1ub_u32(pg, codes + i);
        svfloat32_t c_f32_vec = svcvt_f32_u32_z(pg, c_u32_vec);

        svfloat32_t q_vec = svld1_f32(pg, query + i);
        svfloat32_t lb_vec = svld1_f32(pg, lower_bound + i);
        svfloat32_t d_vec = svld1_f32(pg, diff + i);

        svfloat32_t dequant_vec =
            svmla_f32_m(pg, lb_vec, svmul_f32_m(pg, c_f32_vec, inv_255_vec), d_vec);
        svfloat32_t diff_vec = svsub_f32_z(pg, q_vec, dequant_vec);
        sum_vec = svmla_f32_m(pg, sum_vec, diff_vec, diff_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum_vec = svdup_f32(0.0f);
    svfloat32_t inv_255_vec = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svprfb(svptrue_b8(), codes1 + i + step, SV_PLDL1KEEP);
        svprfb(svptrue_b8(), codes2 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), lower_bound + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), diff + i + step, SV_PLDL1KEEP);

        svuint32_t c1_u32_vec = svld1ub_u32(pg, codes1 + i);
        svfloat32_t c1_f32_vec = svcvt_f32_u32_z(pg, c1_u32_vec);
        svuint32_t c2_u32_vec = svld1ub_u32(pg, codes2 + i);
        svfloat32_t c2_f32_vec = svcvt_f32_u32_z(pg, c2_u32_vec);

        svfloat32_t lb_vec = svld1_f32(pg, lower_bound + i);
        svfloat32_t d_vec = svld1_f32(pg, diff + i);

        svfloat32_t dequant1_vec =
            svmla_f32_m(pg, lb_vec, svmul_f32_m(pg, c1_f32_vec, inv_255_vec), d_vec);
        svfloat32_t dequant2_vec =
            svmla_f32_m(pg, lb_vec, svmul_f32_m(pg, c2_f32_vec, inv_255_vec), d_vec);

        sum_vec = svmla_f32_m(pg, sum_vec, dequant1_vec, dequant2_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t sum_vec = svdup_f32(0.0f);
    svfloat32_t inv_255_vec = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svprfb(svptrue_b8(), codes1 + i + step, SV_PLDL1KEEP);
        svprfb(svptrue_b8(), codes2 + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), lower_bound + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), diff + i + step, SV_PLDL1KEEP);

        svuint32_t c1_u32_vec = svld1ub_u32(pg, codes1 + i);
        svfloat32_t c1_f32_vec = svcvt_f32_u32_z(pg, c1_u32_vec);
        svuint32_t c2_u32_vec = svld1ub_u32(pg, codes2 + i);
        svfloat32_t c2_f32_vec = svcvt_f32_u32_z(pg, c2_u32_vec);

        svfloat32_t lb_vec = svld1_f32(pg, lower_bound + i);
        svfloat32_t d_vec = svld1_f32(pg, diff + i);

        svfloat32_t dequant1_vec =
            svmla_f32_m(pg, lb_vec, svmul_f32_m(pg, c1_f32_vec, inv_255_vec), d_vec);
        svfloat32_t dequant2_vec =
            svmla_f32_m(pg, lb_vec, svmul_f32_m(pg, c2_f32_vec, inv_255_vec), d_vec);

        svfloat32_t diff_vec = svsub_f32_z(pg, dequant1_vec, dequant2_vec);
        sum_vec = svmla_f32_m(pg, sum_vec, diff_vec, diff_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t z_result_vec = svdup_f32(0.0f);
    const svfloat32_t z_inv_15 = svdup_f32(1.0f / 15.0f);
    const uint64_t vl = svcntw();
    const svbool_t pg_all = svptrue_b32();

    const svuint32_t z_base_indices = svindex_u32(0, 1);
    const svuint32_t z_even_elem_indices = svmul_n_u32_x(pg_all, z_base_indices, 2);
    const svuint32_t z_even_byte_offsets =
        svmul_n_u32_x(pg_all, z_even_elem_indices, sizeof(float));
    const svuint32_t z_odd_elem_indices = svadd_n_u32_x(pg_all, z_even_elem_indices, 1);
    const svuint32_t z_odd_byte_offsets = svmul_n_u32_x(pg_all, z_odd_elem_indices, sizeof(float));

    uint64_t d = 0;
    for (; d + 2 * vl <= dim; d += 2 * vl) {
        const svfloat32_t z_query_even =
            svld1_gather_u32offset_f32(pg_all, &query[d], z_even_byte_offsets);
        const svfloat32_t z_query_odd =
            svld1_gather_u32offset_f32(pg_all, &query[d], z_odd_byte_offsets);
        const svfloat32_t z_lower_even =
            svld1_gather_u32offset_f32(pg_all, &lower_bound[d], z_even_byte_offsets);
        const svfloat32_t z_lower_odd =
            svld1_gather_u32offset_f32(pg_all, &lower_bound[d], z_odd_byte_offsets);
        const svfloat32_t z_diff_even =
            svld1_gather_u32offset_f32(pg_all, &diff[d], z_even_byte_offsets);

        const svfloat32_t z_diff_odd =
            svld1_gather_u32offset_f32(pg_all, &diff[d], z_odd_byte_offsets);

        const svuint32_t z_packed_u32 = svld1ub_u32(pg_all, &codes[d / 2]);

        const svuint32_t z_codes_even_u32 = svand_n_u32_x(pg_all, z_packed_u32, 0x0F);
        const svuint32_t z_codes_odd_u32 = svlsr_n_u32_x(pg_all, z_packed_u32, 4);

        const svfloat32_t z_codes_even_f32 = svcvt_f32_u32_x(pg_all, z_codes_even_u32);
        const svfloat32_t z_codes_odd_f32 = svcvt_f32_u32_x(pg_all, z_codes_odd_u32);

        svfloat32_t z_y_even = svmla_f32_x(
            pg_all, z_lower_even, svmul_f32_x(pg_all, z_codes_even_f32, z_inv_15), z_diff_even);
        svfloat32_t z_y_odd = svmla_f32_x(
            pg_all, z_lower_odd, svmul_f32_x(pg_all, z_codes_odd_f32, z_inv_15), z_diff_odd);

        z_result_vec = svmla_f32_x(pg_all, z_result_vec, z_query_even, z_y_even);
        z_result_vec = svmla_f32_x(pg_all, z_result_vec, z_query_odd, z_y_odd);
    }

    if (d < dim) {
        return svaddv_f32(pg_all, z_result_vec) +
               neon::SQ4ComputeIP(&query[d], &codes[d / 2], &lower_bound[d], &diff[d], dim - d);
    }

    return svaddv_f32(pg_all, z_result_vec);
#else
    return neon::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t z_result_vec = svdup_f32(0.0f);
    const svfloat32_t z_inv_15 = svdup_f32(1.0f / 15.0f);
    const uint64_t vl = svcntw();
    const svbool_t pg_all = svptrue_b32();

    const svuint32_t z_base_indices = svindex_u32(0, 1);
    const svuint32_t z_even_elem_indices = svmul_n_u32_x(pg_all, z_base_indices, 2);
    const svuint32_t z_even_byte_offsets =
        svmul_n_u32_x(pg_all, z_even_elem_indices, sizeof(float));
    const svuint32_t z_odd_elem_indices = svadd_n_u32_x(pg_all, z_even_elem_indices, 1);
    const svuint32_t z_odd_byte_offsets = svmul_n_u32_x(pg_all, z_odd_elem_indices, sizeof(float));

    uint64_t d = 0;
    for (; d + 2 * vl <= dim; d += 2 * vl) {
        const svfloat32_t z_query_even =
            svld1_gather_u32offset_f32(pg_all, &query[d], z_even_byte_offsets);
        const svfloat32_t z_query_odd =
            svld1_gather_u32offset_f32(pg_all, &query[d], z_odd_byte_offsets);
        const svfloat32_t z_lower_even =
            svld1_gather_u32offset_f32(pg_all, &lower_bound[d], z_even_byte_offsets);
        const svfloat32_t z_lower_odd =
            svld1_gather_u32offset_f32(pg_all, &lower_bound[d], z_odd_byte_offsets);
        const svfloat32_t z_diff_even_param =
            svld1_gather_u32offset_f32(pg_all, &diff[d], z_even_byte_offsets);
        const svfloat32_t z_diff_odd_param =
            svld1_gather_u32offset_f32(pg_all, &diff[d], z_odd_byte_offsets);

        const svuint32_t z_packed_u32 = svld1ub_u32(pg_all, &codes[d / 2]);

        const svuint32_t z_codes_even_u32 = svand_n_u32_x(pg_all, z_packed_u32, 0x0F);
        const svuint32_t z_codes_odd_u32 = svlsr_n_u32_x(pg_all, z_packed_u32, 4);

        const svfloat32_t z_codes_even_f32 = svcvt_f32_u32_x(pg_all, z_codes_even_u32);
        const svfloat32_t z_codes_odd_f32 = svcvt_f32_u32_x(pg_all, z_codes_odd_u32);

        svfloat32_t z_y_even = svmla_f32_x(pg_all,
                                           z_lower_even,
                                           svmul_f32_x(pg_all, z_codes_even_f32, z_inv_15),
                                           z_diff_even_param);
        svfloat32_t z_y_odd = svmla_f32_x(
            pg_all, z_lower_odd, svmul_f32_x(pg_all, z_codes_odd_f32, z_inv_15), z_diff_odd_param);

        svfloat32_t z_diff_even = svsub_f32_x(pg_all, z_query_even, z_y_even);
        svfloat32_t z_diff_odd = svsub_f32_x(pg_all, z_query_odd, z_y_odd);

        z_result_vec = svmla_f32_x(pg_all, z_result_vec, z_diff_even, z_diff_even);
        z_result_vec = svmla_f32_x(pg_all, z_result_vec, z_diff_odd, z_diff_odd);
    }

    if (d < dim) {
        return svaddv_f32(pg_all, z_result_vec) +
               neon::SQ4ComputeL2Sqr(&query[d], &codes[d / 2], &lower_bound[d], &diff[d], dim - d);
    }

    return svaddv_f32(pg_all, z_result_vec);
#else
    return neon::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t z_result_vec = svdup_f32(0.0f);
    const svfloat32_t z_inv_15 = svdup_f32(1.0f / 15.0f);
    const uint64_t vl = svcntw();
    const svbool_t pg_all = svptrue_b32();

    const svuint32_t z_base_indices = svindex_u32(0, 1);
    const svuint32_t z_even_elem_indices = svmul_n_u32_x(pg_all, z_base_indices, 2);
    const svuint32_t z_even_byte_offsets =
        svmul_n_u32_x(pg_all, z_even_elem_indices, sizeof(float));
    const svuint32_t z_odd_elem_indices = svadd_n_u32_x(pg_all, z_even_elem_indices, 1);
    const svuint32_t z_odd_byte_offsets = svmul_n_u32_x(pg_all, z_odd_elem_indices, sizeof(float));

    uint64_t d = 0;
    for (; d + 2 * vl <= dim; d += 2 * vl) {
        const svfloat32_t z_lower_even =
            svld1_gather_u32offset_f32(pg_all, &lower_bound[d], z_even_byte_offsets);
        const svfloat32_t z_lower_odd =
            svld1_gather_u32offset_f32(pg_all, &lower_bound[d], z_odd_byte_offsets);
        const svfloat32_t z_diff_even =
            svld1_gather_u32offset_f32(pg_all, &diff[d], z_even_byte_offsets);
        const svfloat32_t z_diff_odd =
            svld1_gather_u32offset_f32(pg_all, &diff[d], z_odd_byte_offsets);

        const svuint32_t z_packed1_u32 = svld1ub_u32(pg_all, &codes1[d / 2]);
        const svuint32_t z_packed2_u32 = svld1ub_u32(pg_all, &codes2[d / 2]);

        const svuint32_t z_codes1_even_u32 = svand_n_u32_x(pg_all, z_packed1_u32, 0x0F);
        const svuint32_t z_codes1_odd_u32 = svlsr_n_u32_x(pg_all, z_packed1_u32, 4);
        const svuint32_t z_codes2_even_u32 = svand_n_u32_x(pg_all, z_packed2_u32, 0x0F);
        const svuint32_t z_codes2_odd_u32 = svlsr_n_u32_x(pg_all, z_packed2_u32, 4);

        const svfloat32_t z_codes1_even_f32 = svcvt_f32_u32_x(pg_all, z_codes1_even_u32);
        const svfloat32_t z_codes1_odd_f32 = svcvt_f32_u32_x(pg_all, z_codes1_odd_u32);
        const svfloat32_t z_codes2_even_f32 = svcvt_f32_u32_x(pg_all, z_codes2_even_u32);
        const svfloat32_t z_codes2_odd_f32 = svcvt_f32_u32_x(pg_all, z_codes2_odd_u32);

        svfloat32_t z_y1_even = svmla_f32_x(
            pg_all, z_lower_even, svmul_f32_x(pg_all, z_codes1_even_f32, z_inv_15), z_diff_even);
        svfloat32_t z_y1_odd = svmla_f32_x(
            pg_all, z_lower_odd, svmul_f32_x(pg_all, z_codes1_odd_f32, z_inv_15), z_diff_odd);
        svfloat32_t z_y2_even = svmla_f32_x(
            pg_all, z_lower_even, svmul_f32_x(pg_all, z_codes2_even_f32, z_inv_15), z_diff_even);
        svfloat32_t z_y2_odd = svmla_f32_x(
            pg_all, z_lower_odd, svmul_f32_x(pg_all, z_codes2_odd_f32, z_inv_15), z_diff_odd);

        z_result_vec = svmla_f32_x(pg_all, z_result_vec, z_y1_even, z_y2_even);
        z_result_vec = svmla_f32_x(pg_all, z_result_vec, z_y1_odd, z_y2_odd);
    }

    if (d < dim) {
        return svaddv_f32(pg_all, z_result_vec) +
               neon::SQ4ComputeCodesIP(
                   &codes1[d / 2], &codes2[d / 2], &lower_bound[d], &diff[d], dim - d);
    }

    return svaddv_f32(pg_all, z_result_vec);
#else
    return neon::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
#if defined(ENABLE_SVE)
    svfloat32_t z_result_vec = svdup_f32(0.0f);
    const svfloat32_t z_inv_15 = svdup_f32(1.0f / 15.0f);
    const uint64_t vl = svcntw();
    const svbool_t pg_all = svptrue_b32();

    const svuint32_t z_base_indices = svindex_u32(0, 1);
    const svuint32_t z_even_elem_indices = svmul_n_u32_x(pg_all, z_base_indices, 2);
    const svuint32_t z_even_byte_offsets =
        svmul_n_u32_x(pg_all, z_even_elem_indices, sizeof(float));
    const svuint32_t z_odd_elem_indices = svadd_n_u32_x(pg_all, z_even_elem_indices, 1);
    const svuint32_t z_odd_byte_offsets = svmul_n_u32_x(pg_all, z_odd_elem_indices, sizeof(float));

    uint64_t d = 0;
    for (; d + 2 * vl <= dim; d += 2 * vl) {
        const svfloat32_t z_lower_even =
            svld1_gather_u32offset_f32(pg_all, &lower_bound[d], z_even_byte_offsets);
        const svfloat32_t z_lower_odd =
            svld1_gather_u32offset_f32(pg_all, &lower_bound[d], z_odd_byte_offsets);
        const svfloat32_t z_diff_even =
            svld1_gather_u32offset_f32(pg_all, &diff[d], z_even_byte_offsets);
        const svfloat32_t z_diff_odd =
            svld1_gather_u32offset_f32(pg_all, &diff[d], z_odd_byte_offsets);

        const svuint32_t z_packed1_u32 = svld1ub_u32(pg_all, &codes1[d / 2]);
        const svuint32_t z_packed2_u32 = svld1ub_u32(pg_all, &codes2[d / 2]);

        const svuint32_t z_codes1_even_u32 = svand_n_u32_x(pg_all, z_packed1_u32, 0x0F);
        const svuint32_t z_codes1_odd_u32 = svlsr_n_u32_x(pg_all, z_packed1_u32, 4);
        const svuint32_t z_codes2_even_u32 = svand_n_u32_x(pg_all, z_packed2_u32, 0x0F);
        const svuint32_t z_codes2_odd_u32 = svlsr_n_u32_x(pg_all, z_packed2_u32, 4);

        const svfloat32_t z_codes1_even_f32 = svcvt_f32_u32_x(pg_all, z_codes1_even_u32);
        const svfloat32_t z_codes1_odd_f32 = svcvt_f32_u32_x(pg_all, z_codes1_odd_u32);
        const svfloat32_t z_codes2_even_f32 = svcvt_f32_u32_x(pg_all, z_codes2_even_u32);
        const svfloat32_t z_codes2_odd_f32 = svcvt_f32_u32_x(pg_all, z_codes2_odd_u32);

        svfloat32_t z_y1_even = svmla_f32_x(
            pg_all, z_lower_even, svmul_f32_x(pg_all, z_codes1_even_f32, z_inv_15), z_diff_even);
        svfloat32_t z_y1_odd = svmla_f32_x(
            pg_all, z_lower_odd, svmul_f32_x(pg_all, z_codes1_odd_f32, z_inv_15), z_diff_odd);
        svfloat32_t z_y2_even = svmla_f32_x(
            pg_all, z_lower_even, svmul_f32_x(pg_all, z_codes2_even_f32, z_inv_15), z_diff_even);
        svfloat32_t z_y2_odd = svmla_f32_x(
            pg_all, z_lower_odd, svmul_f32_x(pg_all, z_codes2_odd_f32, z_inv_15), z_diff_odd);

        svfloat32_t z_diff_res_even = svsub_f32_x(pg_all, z_y1_even, z_y2_even);
        svfloat32_t z_diff_res_odd = svsub_f32_x(pg_all, z_y1_odd, z_y2_odd);

        z_result_vec = svmla_f32_x(pg_all, z_result_vec, z_diff_res_even, z_diff_res_even);
        z_result_vec = svmla_f32_x(pg_all, z_result_vec, z_diff_res_odd, z_diff_res_odd);
    }

    if (d < dim) {
        return svaddv_f32(pg_all, z_result_vec) +
               neon::SQ4ComputeCodesL2Sqr(
                   &codes1[d / 2], &codes2[d / 2], &lower_bound[d], &diff[d], dim - d);
    }

    return svaddv_f32(pg_all, z_result_vec);
#else
    return neon::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    svuint32_t sum_vec = svdup_u32(0);
    uint64_t d = 0;
    const uint64_t step = svcntb() * 2;  // 修正：每次处理的字节数

    while (d < dim) {
        svbool_t pg_bytes = svwhilelt_b8(d / 2, (dim + 1) / 2);

        svuint8_t packed_codes1 = svld1_u8(pg_bytes, codes1 + d / 2);
        svuint8_t packed_codes2 = svld1_u8(pg_bytes, codes2 + d / 2);

        svuint8_t c1_low_u8 = svand_u8_z(pg_bytes, packed_codes1, svdup_u8(0x0F));
        svuint8_t c1_high_u8 = svlsr_n_u8_z(pg_bytes, packed_codes1, 4);
        svuint8_t c2_low_u8 = svand_u8_z(pg_bytes, packed_codes2, svdup_u8(0x0F));
        svuint8_t c2_high_u8 = svlsr_n_u8_z(pg_bytes, packed_codes2, 4);

        sum_vec = svdot_u32(sum_vec, c1_low_u8, c2_low_u8);
        sum_vec = svdot_u32(sum_vec, c1_high_u8, c2_high_u8);

        d += step;
    }

    return static_cast<float>(svaddv_u32(svptrue_b32(), sum_vec));
#else
    return neon::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    svuint32_t sum_vec = svdup_u32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t pg = svwhilelt_b8(i, dim);
    do {
        svuint8_t c1_u8_vec = svld1_u8(pg, codes1 + i);
        svuint8_t c2_u8_vec = svld1_u8(pg, codes2 + i);

        sum_vec = svdot_u32(sum_vec, c1_u8_vec, c2_u8_vec);

        i += step;
        pg = svwhilelt_b8(i, dim);
    } while (svptest_first(svptrue_b8(), pg));

    return static_cast<float>(svaddv_u32(svptrue_b32(), sum_vec));
#else
    return neon::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 0.0f;
    }

    auto predicate_array = std::make_unique<uint8_t[]>(dim);

    const uint64_t num_bytes = dim / 8;
    for (uint64_t i = 0; i < num_bytes; ++i) {
        memcpy(&predicate_array[i * 8], g_bit_lookup_table[bits[i]].data(), 8);
    }

    if (dim % 8 != 0) {
        const uint64_t remaining_bits = dim % 8;
        memcpy(&predicate_array[num_bytes * 8],
               g_bit_lookup_table[bits[num_bytes]].data(),
               remaining_bits);
    }

    uint64_t d = 0;
    const uint64_t vl = svcntw();
    svfloat32_t vec_sum = svdup_f32(0.0f);

    const svfloat32_t v_inv = svdup_f32(inv_sqrt_d);
    const svfloat32_t v_neg_inv = svdup_f32(-inv_sqrt_d);

    while (d < dim) {
        const svbool_t pg = svwhilelt_b32(d, dim);

        const svuint32_t v_preds_extended = svld1ub_u32(pg, &predicate_array[d]);

        const svbool_t bit_mask = svcmpne_n_u32(pg, v_preds_extended, 0);

        const svfloat32_t vec_b = svsel_f32(bit_mask, v_inv, v_neg_inv);
        const svfloat32_t vec_v = svld1_f32(pg, &vector[d]);
        vec_sum = svmla_f32_m(pg, vec_sum, vec_v, vec_b);

        d += vl;
    }

    return svaddv_f32(svptrue_b32(), vec_sum);
#else
    return neon::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
#endif
}

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return 0;
    }

    uint32_t result = 0;
    size_t num_bytes = (dim + 7) / 8;
    uint64_t num_u64_lanes = svcntd();

    uint64_t chunk_size_bytes = num_u64_lanes * 8;

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        const uint8_t* cur_codes = codes + bit_pos * num_bytes;
        uint64_t plane_sum = 0;
        size_t i = 0;

        svuint64_t acc = svdup_u64(0);

        for (; i + chunk_size_bytes <= num_bytes; i += chunk_size_bytes) {
            svuint64_t vec_codes =
                svld1_u64(svptrue_b64(), reinterpret_cast<const uint64_t*>(cur_codes + i));
            svuint64_t vec_bits =
                svld1_u64(svptrue_b64(), reinterpret_cast<const uint64_t*>(bits + i));

            svuint64_t and_result = svand_u64_z(svptrue_b64(), vec_codes, vec_bits);

            svuint64_t popcnt_result = svcnt_u64_z(svptrue_b64(), and_result);

            acc = svadd_u64_m(svptrue_b64(), acc, popcnt_result);
        }

        plane_sum += svaddv_u64(svptrue_b64(), acc);

        for (; i < num_bytes; ++i) {
            uint8_t bitwise_and = cur_codes[i] & bits[i];
            plane_sum += __builtin_popcount(bitwise_and);
        }

        result += plane_sum << bit_pos;
    }

    return result;
#else
    return neon::RaBitQSQ4UBinaryIP(codes, bits, dim);
#endif
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t pg = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(pg, x + i);
        svuint8_t y_vec = svld1_u8(pg, y + i);
        svuint8_t res_vec = svand_u8_z(pg, x_vec, y_vec);
        svst1_u8(pg, result + i, res_vec);

        i += step;
        pg = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), pg));
#else
    neon::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t pg = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(pg, x + i);
        svuint8_t y_vec = svld1_u8(pg, y + i);
        svuint8_t res_vec = svorr_u8_z(pg, x_vec, y_vec);
        svst1_u8(pg, result + i, res_vec);

        i += step;
        pg = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), pg));
#else
    neon::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t pg = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(pg, x + i);
        svuint8_t y_vec = svld1_u8(pg, y + i);
        svuint8_t res_vec = sveor_u8_z(pg, x_vec, y_vec);
        svst1_u8(pg, result + i, res_vec);

        i += step;
        pg = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), pg));
#else
    neon::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntb();
    svbool_t pg = svwhilelt_b8(i, num_byte);
    do {
        svuint8_t x_vec = svld1_u8(pg, x + i);
        svuint8_t res_vec = svnot_u8_z(pg, x_vec);
        svst1_u8(pg, result + i, res_vec);

        i += step;
        pg = svwhilelt_b8(i, num_byte);
    } while (svptest_first(svptrue_b8(), pg));
#else
    neon::BitNot(x, num_byte, result);
#endif
}

void
Prefetch(const void* data) {
#if defined(ENABLE_SVE)
    svprfw(svptrue_b32(), data, SV_PLDL1KEEP);
#else
    neon::Prefetch(data);
#endif
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_SVE)
    if (dim == 0) {
        return;
    }
    if (scalar == 0) {
        scalar = 1.0f;
    }
    svfloat32_t scalar_vec = svdup_f32(scalar);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svfloat32_t from_vec = svld1_f32(pg, from + i);
        svfloat32_t to_vec = svdiv_f32_z(pg, from_vec, scalar_vec);
        svst1_f32(pg, to + i, to_vec);
        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));
#else
    neon::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
#if defined(ENABLE_SVE)
    float norm = std::sqrt(sve::FP32ComputeIP(from, from, dim));
    if (norm == 0) {
        norm = 1.0f;
    }
    sve::DivScalar(from, to, dim, norm);
    return norm;
#else
    return neon::Normalize(from, to, dim);
#endif
}

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t total_bytes = pq_dim * 16;
    auto step = svcntb();

    const svuint8_t mask4 = svdup_u8(0x0F);
    const svuint16_t mask8 = svdup_u16(0x00FF);
    svuint16x4_t accumulators =
        svcreate4_u16(svdup_u16(0), svdup_u16(0), svdup_u16(0), svdup_u16(0));

    uint8_t offsets_data[svcntb()];
    for (uint64_t c = 0; c < svcntb() / 16; ++c) std::memset(offsets_data + c * 16, c * 16, 16);
    const svuint8_t index_offsets = svld1_u8(svptrue_b8(), offsets_data);

    while (i < total_bytes) {
        svbool_t pg_bytes = svwhilelt_b8(i, total_bytes);

        svuint8_t super_table = svld1_u8(pg_bytes, lookup_table + i);
        svuint8_t super_codes = svld1_u8(pg_bytes, codes + i);

        svuint8_t low_nibbles = svand_u8_z(pg_bytes, super_codes, mask4);
        svuint8_t high_nibbles = svlsr_n_u8_z(pg_bytes, super_codes, 4);

        svuint8_t adjusted_low_indices = svadd_u8_z(pg_bytes, low_nibbles, index_offsets);
        svuint8_t adjusted_high_indices = svadd_u8_z(pg_bytes, high_nibbles, index_offsets);

        svuint8_t low_vals = svtbl_u8(super_table, adjusted_low_indices);
        svuint8_t high_vals = svtbl_u8(super_table, adjusted_high_indices);

        svbool_t pg_u16 = svwhilelt_b16(i / 2, total_bytes / 2);

        svuint16_t acc0 = svget4_u16(accumulators, 0);
        svuint16_t acc1 = svget4_u16(accumulators, 1);
        svuint16_t acc2 = svget4_u16(accumulators, 2);
        svuint16_t acc3 = svget4_u16(accumulators, 3);

        acc0 =
            svadd_u16_m(pg_u16, acc0, svand_u16_z(pg_u16, svreinterpret_u16_u8(low_vals), mask8));
        acc1 = svadd_u16_m(pg_u16, acc1, svlsr_n_u16_z(pg_u16, svreinterpret_u16_u8(low_vals), 8));
        acc2 =
            svadd_u16_m(pg_u16, acc2, svand_u16_z(pg_u16, svreinterpret_u16_u8(high_vals), mask8));
        acc3 = svadd_u16_m(pg_u16, acc3, svlsr_n_u16_z(pg_u16, svreinterpret_u16_u8(high_vals), 8));

        accumulators = svset4_u16(accumulators, 0, acc0);
        accumulators = svset4_u16(accumulators, 1, acc1);
        accumulators = svset4_u16(accumulators, 2, acc2);
        accumulators = svset4_u16(accumulators, 3, acc3);

        i += step;
    }

    uint16_t temp[svcntb() / 2];
    {
        // Segment 0

        svst1_u16(svptrue_b16(), temp, svget4_u16(accumulators, 0));
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < svcntb() / 16; k++) result[0 * 8 + j] += temp[j + 8 * (k)];

        // Segment 1

        svst1_u16(svptrue_b16(), temp, svget4_u16(accumulators, 1));
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < svcntb() / 16; k++) result[1 * 8 + j] += temp[j + 8 * k];

        // Segment 2

        svst1_u16(svptrue_b16(), temp, svget4_u16(accumulators, 2));
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < svcntb() / 16; k++) result[2 * 8 + j] += temp[j + 8 * k];

        // Segment 3

        svst1_u16(svptrue_b16(), temp, svget4_u16(accumulators, 3));
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < svcntb() / 16; k++) result[3 * 8 + j] += temp[j + 8 * k];
        ;
    }

#else
    neon::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
KacsWalk(float* data, size_t len) {
#if defined(ENABLE_SVE)
    size_t n = len / 2;
    size_t offset = (len % 2) + n;
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, n);
    do {
        svfloat32_t v1 = svld1_f32(pg, data + i);
        svfloat32_t v2 = svld1_f32(pg, data + i + offset);
        svfloat32_t add = svadd_f32_z(pg, v1, v2);
        svfloat32_t sub = svsub_f32_z(pg, v1, v2);
        svst1_f32(pg, data + i, add);
        svst1_f32(pg, data + i + offset, sub);
        i += step;
        pg = svwhilelt_b32(i, n);
    } while (svptest_first(svptrue_b32(), pg));

    if (len % 2 != 0) {
        data[n] *= std::sqrt(2.0F);
    }
#else
    neon::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, size_t dim) {
#if defined(ENABLE_SVE)
    auto predicate_array = std::make_unique<uint8_t[]>(dim);
    const uint64_t num_bytes = dim / 8;
    for (uint64_t j = 0; j < num_bytes; ++j) {
        memcpy(&predicate_array[j * 8], g_bit_lookup_table[flip[j]].data(), 8);
    }
    if (dim % 8 != 0) {
        const uint64_t remaining_bits = dim % 8;
        memcpy(&predicate_array[num_bytes * 8],
               g_bit_lookup_table[flip[num_bytes]].data(),
               remaining_bits);
    }

    uint64_t d = 0;
    const uint64_t vl = svcntw();
    while (d < dim) {
        const svbool_t pg = svwhilelt_b32(d, dim);
        const svuint32_t v_preds_extended = svld1ub_u32(pg, &predicate_array[d]);
        const svbool_t bit_mask = svcmpne_n_u32(pg, v_preds_extended, 0);

        svfloat32_t data_vec = svld1_f32(pg, data + d);
        svfloat32_t result_vec = svneg_f32_m(data_vec, bit_mask, data_vec);
        svst1_f32(pg, data + d, result_vec);

        d += vl;
    }
#else
    neon::FlipSign(flip, data, dim);
#endif
}

void
VecRescale(float* data, size_t dim, float val) {
#if defined(ENABLE_SVE)
    svfloat32_t val_vec = svdup_f32(val);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svfloat32_t data_vec = svld1_f32(pg, data + i);
        svfloat32_t res_vec = svmul_f32_z(pg, data_vec, val_vec);
        svst1_f32(pg, data + i, res_vec);
        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_first(svptrue_b32(), pg));
#else
    neon::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_SVE)
    for (int i = idx; i < dim_; i += 2 * step) {
        uint64_t j = 0;
        const uint64_t sve_step = svcntw();
        svbool_t pg = svwhilelt_b32(j, (uint64_t)step);
        do {
            svfloat32_t x = svld1_f32(pg, data + i + j);
            svfloat32_t y = svld1_f32(pg, data + i + j + step);
            svst1_f32(pg, data + i + j, svadd_f32_z(pg, x, y));
            svst1_f32(pg, data + i + j + step, svsub_f32_z(pg, x, y));
            j += sve_step;
            pg = svwhilelt_b32(j, (uint64_t)step);
        } while (svptest_first(svptrue_b32(), pg));
    }
#else
    neon::RotateOp(data, idx, dim_, step);
#endif
}

void
FHTRotate(float* data, size_t dim_) {
#if defined(ENABLE_SVE)
    size_t n = dim_;
    size_t step = 1;
    while (step < n) {
        sve::RotateOp(data, 0, dim_, step);
        step *= 2;
    }
#else
    neon::FHTRotate(data, dim_);
#endif
}

}  // namespace vsag::sve
