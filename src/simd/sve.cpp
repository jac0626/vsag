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

#include <cmath>
#include <cstdint>

#include "simd.h"

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace vsag::sve {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(ENABLE_SVE)
    auto* pVect1 = (const float*)pVect1v;
    auto* pVect2 = (const float*)pVect2v;
    uint64_t qty = *((uint64_t*)qty_ptr);

    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, qty);
    do {
        svprfd(svptrue_b32(), pVect1 + i + step, 0);
        svprfd(svptrue_b32(), pVect2 + i + step, 0);

        svfloat32_t v1 = svld1_f32(pg, pVect1 + i);
        svfloat32_t v2 = svld1_f32(pg, pVect2 + i);
        svfloat32_t diff = svsub_f32_z(pg, v1, v2);
        sum_vec = svmla_f32_m(pg, sum_vec, diff, diff);

        i += step;
        pg = svwhilelt_b32(i, qty);
    } while (svptest_any(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::L2Sqr(pVect1v, pVect2v, qty_ptr);
#endif
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(ENABLE_SVE)
    auto* pVect1 = (const float*)pVect1v;
    auto* pVect2 = (const float*)pVect2v;
    uint64_t qty = *((uint64_t*)qty_ptr);

    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, qty);
    do {
        svprfd(svptrue_b32(), pVect1 + i + step, 0);
        svprfd(svptrue_b32(), pVect2 + i + step, 0);

        svfloat32_t v1 = svld1_f32(pg, pVect1 + i);
        svfloat32_t v2 = svld1_f32(pg, pVect2 + i);
        sum_vec = svmla_f32_m(pg, sum_vec, v1, v2);

        i += step;
        pg = svwhilelt_b32(i, qty);
    } while (svptest_any(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::InnerProduct(pVect1v, pVect2v, qty_ptr);
#endif
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(ENABLE_SVE) && defined(__ARM_FEATURE_SVE2__)
    auto* pVect1 = (const int8_t*)pVect1v;
    auto* pVect2 = (const int8_t*)pVect2v;
    uint64_t qty = *((uint64_t*)qty_ptr);

    svint32_t sum_vec = svdup_s32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t pg = svwhilelt_b8(i, qty);
    while (svptest_any(svptrue_b8(), pg)) {
        svint8_t v1 = svld1_s8(pg, pVect1 + i);
        svint8_t v2 = svld1_s8(pg, pVect2 + i);
        sum_vec = svdot_s32_m(pg, sum_vec, v1, v2);

        i += step;
        pg = svwhilelt_b8(i, qty);
    }

    return (float)svaddv_s32(svptrue_b32(), sum_vec);
#else
    return neon::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
#endif
}

float
INT8InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -INT8InnerProduct(pVect1, pVect2, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
#if defined(ENABLE_SVE)
    const float* centers = (const float*)single_dim_centers;
    float* res = (float*)result;
    const svfloat32_t val_vec = svdup_f32(single_dim_val);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, 256);
    do {
        svfloat32_t centers_vec = svld1_f32(pg, centers + i);
        svfloat32_t res_vec = svld1_f32(pg, res + i);
        svfloat32_t diff = svsub_f32_z(pg, centers_vec, val_vec);
        svfloat32_t diff_sq = svmul_f32_z(pg, diff, diff);
        svst1_f32(pg, res + i, svadd_f32_m(pg, res_vec, diff_sq));

        i += step;
        pg = svwhilelt_b32(i, 256);
    } while (svptest_any(svptrue_b32(), pg));
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
        svprfd(svptrue_b32(), query + i + step, 0);
        svprfd(svptrue_b32(), codes + i + step, 0);

        svfloat32_t q_vec = svld1_f32(pg, query + i);
        svfloat32_t c_vec = svld1_f32(pg, codes + i);
        sum_vec = svmla_f32_m(pg, sum_vec, q_vec, c_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg));

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
        svprfd(svptrue_b32(), query + i + step, 0);
        svprfd(svptrue_b32(), codes + i + step, 0);

        svfloat32_t q_vec = svld1_f32(pg, query + i);
        svfloat32_t c_vec = svld1_f32(pg, codes + i);
        svfloat32_t diff = svsub_f32_z(pg, q_vec, c_vec);
        sum_vec = svmla_f32_m(pg, sum_vec, diff, diff);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg));

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
        svprfd(svptrue_b32(), query + i + step, 0);
        svprfd(svptrue_b32(), codes1 + i + step, 0);
        svprfd(svptrue_b32(), codes2 + i + step, 0);
        svprfd(svptrue_b32(), codes3 + i + step, 0);
        svprfd(svptrue_b32(), codes4 + i + step, 0);

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
    } while (svptest_any(svptrue_b32(), pg));

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
        svprfd(svptrue_b32(), query + i + step, 0);
        svprfd(svptrue_b32(), codes1 + i + step, 0);
        svprfd(svptrue_b32(), codes2 + i + step, 0);
        svprfd(svptrue_b32(), codes3 + i + step, 0);
        svprfd(svptrue_b32(), codes4 + i + step, 0);

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
    } while (svptest_any(svptrue_b32(), pg));

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
    } while (svptest_any(svptrue_b32(), pg));
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
    } while (svptest_any(svptrue_b32(), pg));
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
    } while (svptest_any(svptrue_b32(), pg));
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
    } while (svptest_any(svptrue_b32(), pg));
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
    } while (svptest_any(svptrue_b32(), pg));
    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::FP32ReduceAdd(x, dim);
#endif
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE) && defined(__ARM_FEATURE_SVE_BF16__)
    const bfloat16_t* query_bf16 = reinterpret_cast<const bfloat16_t*>(query);
    const bfloat16_t* codes_bf16 = reinterpret_cast<const bfloat16_t*>(codes);

    svbfloat16_t sum_vec = svdup_bf16(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcnth();
    svbool_t pg = svwhilelt_b16(i, dim);
    do {
        svprfd(svptrue_b16(), query_bf16 + i + step, 0);
        svprfd(svptrue_b16(), codes_bf16 + i + step, 0);

        svbfloat16_t q_vec = svld1_bf16(pg, query_bf16 + i);
        svbfloat16_t c_vec = svld1_bf16(pg, codes_bf16 + i);
        sum_vec = svmla_bf16_m(pg, sum_vec, q_vec, c_vec);

        i += step;
        pg = svwhilelt_b16(i, dim);
    } while (svptest_any(svptrue_b16(), pg));

    return static_cast<float>(svaddv_bf16(svptrue_b16(), sum_vec));
#else
    return neon::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE) && defined(__ARM_FEATURE_SVE_BF16__)
    const bfloat16_t* query_bf16 = reinterpret_cast<const bfloat16_t*>(query);
    const bfloat16_t* codes_bf16 = reinterpret_cast<const bfloat16_t*>(codes);

    svbfloat16_t sum_vec = svdup_bf16(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcnth();
    svbool_t pg = svwhilelt_b16(i, dim);
    do {
        svprfd(svptrue_b16(), query_bf16 + i + step, 0);
        svprfd(svptrue_b16(), codes_bf16 + i + step, 0);

        svbfloat16_t q_vec = svld1_bf16(pg, query_bf16 + i);
        svbfloat16_t c_vec = svld1_bf16(pg, codes_bf16 + i);
        svbfloat16_t diff = svsub_bf16_z(pg, q_vec, c_vec);
        sum_vec = svmla_bf16_m(pg, sum_vec, diff, diff);

        i += step;
        pg = svwhilelt_b16(i, dim);
    } while (svptest_any(svptrue_b16(), pg));

    return static_cast<float>(svaddv_bf16(svptrue_b16(), sum_vec));
#else
    return neon::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE) && defined(__ARM_FEATURE_SVE_FP16__)
    const __fp16* query_fp16 = reinterpret_cast<const __fp16*>(query);
    const __fp16* codes_fp16 = reinterpret_cast<const __fp16*>(codes);

    svfloat16_t sum_vec = svdup_f16(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcnth();
    svbool_t pg = svwhilelt_b16(i, dim);
    do {
        svprfd(svptrue_b16(), query_fp16 + i + step, 0);
        svprfd(svptrue_b16(), codes_fp16 + i + step, 0);

        svfloat16_t q_vec = svld1_f16(pg, query_fp16 + i);
        svfloat16_t c_vec = svld1_f16(pg, codes_fp16 + i);
        sum_vec = svmla_f16_m(pg, sum_vec, q_vec, c_vec);

        i += step;
        pg = svwhilelt_b16(i, dim);
    } while (svptest_any(svptrue_b16(), pg));

    return static_cast<float>(svaddv_f16(svptrue_b16(), sum_vec));
#else
    return neon::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE) && defined(__ARM_FEATURE_SVE_FP16__)
    const __fp16* query_fp16 = reinterpret_cast<const __fp16*>(query);
    const __fp16* codes_fp16 = reinterpret_cast<const __fp16*>(codes);

    svfloat16_t sum_vec = svdup_f16(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcnth();
    svbool_t pg = svwhilelt_b16(i, dim);
    do {
        svprfd(svptrue_b16(), query_fp16 + i + step, 0);
        svprfd(svptrue_b16(), codes_fp16 + i + step, 0);

        svfloat16_t q_vec = svld1_f16(pg, query_fp16 + i);
        svfloat16_t c_vec = svld1_f16(pg, codes_fp16 + i);
        svfloat16_t diff = svsub_f16_z(pg, q_vec, c_vec);
        sum_vec = svmla_f16_m(pg, sum_vec, diff, diff);

        i += step;
        pg = svwhilelt_b16(i, dim);
    } while (svptest_any(svptrue_b16(), pg));

    return static_cast<float>(svaddv_f16(svptrue_b16(), sum_vec));
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
    const svfloat32_t inv_255_vec = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg_w = svwhilelt_b32(i, dim);
    do {
        svbool_t pg_b = svwhilelt_b8(i, dim);
        svfloat32_t q_vec = svld1_f32(pg_w, query + i);
        svuint8_t c_u8_vec = svld1_u8(pg_b, codes + i);
        svuint32_t c_u32_vec = svuxtb_u32(pg_w, c_u8_vec);
        svfloat32_t c_f32_vec = svcvt_f32_u32_z(pg_w, c_u32_vec);

        svfloat32_t lb_vec = svld1_f32(pg_w, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(pg_w, diff + i);

        svfloat32_t scaled_codes = svmul_f32_z(pg_w, c_f32_vec, inv_255_vec);
        svfloat32_t adjusted_codes = svmla_f32_m(pg_w, lb_vec, scaled_codes, diff_vec);
        sum_vec = svmla_f32_m(pg_w, sum_vec, q_vec, adjusted_codes);

        i += step;
        pg_w = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg_w));

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
    const svfloat32_t inv_255_vec = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg_w = svwhilelt_b32(i, dim);
    do {
        svbool_t pg_b = svwhilelt_b8(i, dim);
        svfloat32_t q_vec = svld1_f32(pg_w, query + i);
        svuint8_t c_u8_vec = svld1_u8(pg_b, codes + i);
        svuint32_t c_u32_vec = svuxtb_u32(pg_w, c_u8_vec);
        svfloat32_t c_f32_vec = svcvt_f32_u32_z(pg_w, c_u32_vec);

        svfloat32_t lb_vec = svld1_f32(pg_w, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(pg_w, diff + i);

        svfloat32_t scaled_codes = svmul_f32_z(pg_w, c_f32_vec, inv_255_vec);
        svfloat32_t adjusted_codes = svmla_f32_m(pg_w, lb_vec, scaled_codes, diff_vec);
        svfloat32_t diff_res = svsub_f32_z(pg_w, q_vec, adjusted_codes);
        sum_vec = svmla_f32_m(pg_w, sum_vec, diff_res, diff_res);

        i += step;
        pg_w = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg_w));

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
    const svfloat32_t inv_255_vec = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg_w = svwhilelt_b32(i, dim);
    do {
        svbool_t pg_b = svwhilelt_b8(i, dim);
        svuint8_t c1_u8_vec = svld1_u8(pg_b, codes1 + i);
        svuint8_t c2_u8_vec = svld1_u8(pg_b, codes2 + i);
        svuint32_t c1_u32_vec = svuxtb_u32(pg_w, c1_u8_vec);
        svuint32_t c2_u32_vec = svuxtb_u32(pg_w, c2_u8_vec);
        svfloat32_t c1_f32_vec = svcvt_f32_u32_z(pg_w, c1_u32_vec);
        svfloat32_t c2_f32_vec = svcvt_f32_u32_z(pg_w, c2_u32_vec);

        svfloat32_t lb_vec = svld1_f32(pg_w, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(pg_w, diff + i);

        svfloat32_t scaled1 = svmul_f32_z(pg_w, c1_f32_vec, inv_255_vec);
        svfloat32_t adjusted1 = svmla_f32_m(pg_w, lb_vec, scaled1, diff_vec);

        svfloat32_t scaled2 = svmul_f32_z(pg_w, c2_f32_vec, inv_255_vec);
        svfloat32_t adjusted2 = svmla_f32_m(pg_w, lb_vec, scaled2, diff_vec);

        sum_vec = svmla_f32_m(pg_w, sum_vec, adjusted1, adjusted2);

        i += step;
        pg_w = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg_w));

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
    const svfloat32_t inv_255_vec = svdup_f32(1.0f / 255.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg_w = svwhilelt_b32(i, dim);
    do {
        svbool_t pg_b = svwhilelt_b8(i, dim);
        svuint8_t c1_u8_vec = svld1_u8(pg_b, codes1 + i);
        svuint8_t c2_u8_vec = svld1_u8(pg_b, codes2 + i);
        svuint32_t c1_u32_vec = svuxtb_u32(pg_w, c1_u8_vec);
        svuint32_t c2_u32_vec = svuxtb_u32(pg_w, c2_u8_vec);
        svfloat32_t c1_f32_vec = svcvt_f32_u32_z(pg_w, c1_u32_vec);
        svfloat32_t c2_f32_vec = svcvt_f32_u32_z(pg_w, c2_u32_vec);

        svfloat32_t lb_vec = svld1_f32(pg_w, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(pg_w, diff + i);

        svfloat32_t scaled1 = svmul_f32_z(pg_w, c1_f32_vec, inv_255_vec);
        svfloat32_t adjusted1 = svmla_f32_m(pg_w, lb_vec, scaled1, diff_vec);

        svfloat32_t scaled2 = svmul_f32_z(pg_w, c2_f32_vec, inv_255_vec);
        svfloat32_t adjusted2 = svmla_f32_m(pg_w, lb_vec, scaled2, diff_vec);

        svfloat32_t diff_res = svsub_f32_z(pg_w, adjusted1, adjusted2);
        sum_vec = svmla_f32_m(pg_w, sum_vec, diff_res, diff_res);

        i += step;
        pg_w = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg_w));

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
    svfloat32_t sum_vec = svdup_f32(0.0f);
    const svfloat32_t inv_15_vec = svdup_f32(1.0f / 15.0f);
    const svuint8_t low_mask = svdup_u8(0x0F);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svuint8_t c_packed = svld1_u8(svwhilelt_b8(i / 2, (dim + 1) / 2), codes + i / 2);

        svuint8_t c_lo = svand_u8_z(pg, c_packed, low_mask);
        svuint8_t c_hi = svlsr_n_u8_z(pg, c_packed, 4);

        svuint8_t c_deinterleaved = svuzp1(c_lo, c_hi);

        svuint32_t c_u32 = svuxtb_u32(pg, c_deinterleaved);
        svfloat32_t c_f32 = svcvt_f32_u32_z(pg, c_u32);

        svfloat32_t q_vec = svld1_f32(pg, query + i);
        svfloat32_t lb_vec = svld1_f32(pg, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(pg, diff + i);

        svfloat32_t scaled_codes = svmul_f32_z(pg, c_f32, inv_15_vec);
        svfloat32_t adjusted_codes = svmla_f32_m(pg, lb_vec, scaled_codes, diff_vec);
        sum_vec = svmla_f32_m(pg, sum_vec, q_vec, adjusted_codes);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
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
    svfloat32_t sum_vec = svdup_f32(0.0f);
    const svfloat32_t inv_15_vec = svdup_f32(1.0f / 15.0f);
    const svuint8_t low_mask = svdup_u8(0x0F);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svuint8_t c_packed = svld1_u8(svwhilelt_b8(i / 2, (dim + 1) / 2), codes + i / 2);

        svuint8_t c_lo = svand_u8_z(pg, c_packed, low_mask);
        svuint8_t c_hi = svlsr_n_u8_z(pg, c_packed, 4);

        svuint8_t c_deinterleaved = svuzp1(c_lo, c_hi);

        svuint32_t c_u32 = svuxtb_u32(pg, c_deinterleaved);
        svfloat32_t c_f32 = svcvt_f32_u32_z(pg, c_u32);

        svfloat32_t q_vec = svld1_f32(pg, query + i);
        svfloat32_t lb_vec = svld1_f32(pg, lower_bound + i);
        svfloat32_t diff_vec = svld1_f32(pg, diff + i);

        svfloat32_t scaled_codes = svmul_f32_z(pg, c_f32, inv_15_vec);
        svfloat32_t adjusted_codes = svmla_f32_m(pg, lb_vec, scaled_codes, diff_vec);
        svfloat32_t diff_res = svsub_f32_z(pg, q_vec, adjusted_codes);
        sum_vec = svmla_f32_m(pg, sum_vec, diff_res, diff_res);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
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
    svfloat32_t sum_vec = svdup_f32(0.0f);
    const svfloat32_t inv_15_vec = svdup_f32(1.0f / 15.0f);
    const svuint8_t low_mask = svdup_u8(0x0F);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svuint8_t c1_packed = svld1_u8(svwhilelt_b8(i / 2, (dim + 1) / 2), codes1 + i / 2);
        svuint8_t c2_packed = svld1_u8(svwhilelt_b8(i / 2, (dim + 1) / 2), codes2 + i / 2);

        svuint8_t c1_lo = svand_u8_z(pg, c1_packed, low_mask);
        svuint8_t c1_hi = svlsr_n_u8_z(pg, c1_packed, 4);
        svuint8_t c2_lo = svand_u8_z(pg, c2_packed, low_mask);
        svuint8_t c2_hi = svlsr_n_u8_z(pg, c2_packed, 4);

        svuint8_t c1_deinterleaved = svuzp1(c1_lo, c1_hi);
        svuint8_t c2_deinterleaved = svuzp1(c2_lo, c2_hi);

        svuint32_t c1_u32 = svuxtb_u32(pg, c1_deinterleaved);
        svuint32_t c2_u32 = svuxtb_u32(pg, c2_deinterleaved);
        svfloat32_t c1_f32 = svcvt_f32_u32_z(pg, c1_u32);
        svfloat32_t c2_f32 = svcvt_f32_u32_z(pg, c2_u32);

        svfloat32_t lb_vec = svld1_f32(pg, lower_bound + i);
        // Note: generic.cpp has a bug, using diff[d] for both nibbles. We replicate it.
        svfloat32_t diff_vec = svld1_f32(pg, diff + i);

        svfloat32_t scaled1 = svmul_f32_z(pg, c1_f32, inv_15_vec);
        svfloat32_t adjusted1 = svmla_f32_m(pg, lb_vec, scaled1, diff_vec);

        svfloat32_t scaled2 = svmul_f32_z(pg, c2_f32, inv_15_vec);
        svfloat32_t adjusted2 = svmla_f32_m(pg, lb_vec, scaled2, diff_vec);

        sum_vec = svmla_f32_m(pg, sum_vec, adjusted1, adjusted2);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
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
    svfloat32_t sum_vec = svdup_f32(0.0f);
    const svfloat32_t inv_15_vec = svdup_f32(1.0f / 15.0f);
    const svuint8_t low_mask = svdup_u8(0x0F);
    uint64_t i = 0;
    const uint64_t step = svcntw();

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svuint8_t c1_packed = svld1_u8(svwhilelt_b8(i / 2, (dim + 1) / 2), codes1 + i / 2);
        svuint8_t c2_packed = svld1_u8(svwhilelt_b8(i / 2, (dim + 1) / 2), codes2 + i / 2);

        svuint8_t c1_lo = svand_u8_z(pg, c1_packed, low_mask);
        svuint8_t c1_hi = svlsr_n_u8_z(pg, c1_packed, 4);
        svuint8_t c2_lo = svand_u8_z(pg, c2_packed, low_mask);
        svuint8_t c2_hi = svlsr_n_u8_z(pg, c2_packed, 4);

        svuint8_t c1_deinterleaved = svuzp1(c1_lo, c1_hi);
        svuint8_t c2_deinterleaved = svuzp1(c2_lo, c2_hi);

        svuint32_t c1_u32 = svuxtb_u32(pg, c1_deinterleaved);
        svuint32_t c2_u32 = svuxtb_u32(pg, c2_deinterleaved);
        svfloat32_t c1_f32 = svcvt_f32_u32_z(pg, c1_u32);
        svfloat32_t c2_f32 = svcvt_f32_u32_z(pg, c2_u32);

        svfloat32_t lb_vec = svld1_f32(pg, lower_bound + i);
        // Note: generic.cpp has a bug, using diff[d] for both nibbles. We replicate it.
        svfloat32_t diff_vec = svld1_f32(pg, diff + i);

        svfloat32_t scaled1 = svmul_f32_z(pg, c1_f32, inv_15_vec);
        svfloat32_t adjusted1 = svmla_f32_m(pg, lb_vec, scaled1, diff_vec);

        svfloat32_t scaled2 = svmul_f32_z(pg, c2_f32, inv_15_vec);
        svfloat32_t adjusted2 = svmla_f32_m(pg, lb_vec, scaled2, diff_vec);

        svfloat32_t diff_res = svsub_f32_z(pg, adjusted1, adjusted2);
        sum_vec = svmla_f32_m(pg, sum_vec, diff_res, diff_res);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
#else
    return neon::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE) && defined(__ARM_FEATURE_SVE2__)
    svint32_t sum_vec = svdup_s32(0);
    const svuint8_t low_mask = svdup_u8(0x0F);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t pg_b = svwhilelt_b8(i, dim);
    do {
        svuint8_t c1_packed = svld1_u8(svwhilelt_b8(i / 2, (dim + 1) / 2), codes1 + i / 2);
        svuint8_t c2_packed = svld1_u8(svwhilelt_b8(i / 2, (dim + 1) / 2), codes2 + i / 2);

        svuint8_t c1_lo = svand_u8_z(pg_b, c1_packed, low_mask);
        svuint8_t c1_hi = svlsr_n_u8_z(pg_b, c1_packed, 4);
        svuint8_t c2_lo = svand_u8_z(pg_b, c2_packed, low_mask);
        svuint8_t c2_hi = svlsr_n_u8_z(pg_b, c2_packed, 4);

        svuint8_t c1_deinterleaved = svuzp1(c1_lo, c1_hi);
        svuint8_t c2_deinterleaved = svuzp1(c2_lo, c2_hi);

        sum_vec = svdot_s32_m(
            pg_b, sum_vec, svmov_s8_z(pg_b, c1_deinterleaved), svmov_s8_z(pg_b, c2_deinterleaved));

        i += step;
        pg_b = svwhilelt_b8(i, dim);
    } while (svptest_any(svptrue_b8(), pg_b));

    return (float)svaddv_s32(svptrue_b32(), sum_vec);
#else
    return neon::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE) && defined(__ARM_FEATURE_SVE2__)
    svint32_t sum_vec = svdup_s32(0);
    uint64_t i = 0;
    const uint64_t step = svcntb();

    svbool_t pg = svwhilelt_b8(i, dim);
    do {
        svint8_t c1_vec = svld1_s8(pg, (const int8_t*)codes1 + i);
        svint8_t c2_vec = svld1_s8(pg, (const int8_t*)codes2 + i);
        sum_vec = svdot_s32_m(pg, sum_vec, c1_vec, c2_vec);

        i += step;
        pg = svwhilelt_b8(i, dim);
    } while (svptest_any(svptrue_b8(), pg));

    return (float)svaddv_s32(svptrue_b32(), sum_vec);
#else
    return neon::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_SVE)
    svfloat32_t sum_vec = svdup_f32(0.0f);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    float lane_vals[step];

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        const uint64_t current_limit = i + step;
        for (uint64_t j = 0; i + j < current_limit; ++j) {
            if (i + j < dim) {
                bool bit = ((bits[(i + j) / 8] >> ((i + j) % 8)) & 1) != 0;
                lane_vals[j] = bit ? inv_sqrt_d : -inv_sqrt_d;
            } else {
                lane_vals[j] = 0.0f;
            }
        }

        svfloat32_t v_vec = svld1_f32(pg, vector + i);
        svfloat32_t b_vec = svld1_f32(pg, lane_vals);

        sum_vec = svmla_f32_m(pg, sum_vec, v_vec, b_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg));

    return svaddv_f32(svptrue_b32(), sum_vec);
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

    for (uint64_t bit_pos = 0; bit_pos < 4; ++bit_pos) {
        uint64_t total_popcount = 0;
        uint64_t i = 0;
        const uint64_t step = svcntb();
        svbool_t pg = svwhilelt_b8(i, num_bytes);
        do {
            svuint8_t codes_vec = svld1_u8(pg, codes + (bit_pos * num_bytes) + i);
            svuint8_t bits_vec = svld1_u8(pg, bits + i);
            svuint8_t and_vec = svand_u8_z(pg, codes_vec, bits_vec);
            svuint8_t popcount_vec = svcnt_u8_z(pg, and_vec);
            total_popcount += svaddv_u8(pg, popcount_vec);

            i += step;
            pg = svwhilelt_b8(i, num_bytes);
        } while (svptest_any(svptrue_b8(), pg));
        result += total_popcount << bit_pos;
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
    } while (svptest_any(svptrue_b8(), pg));
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
    } while (svptest_any(svptrue_b8(), pg));
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
    } while (svptest_any(svptrue_b8(), pg));
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
    } while (svptest_any(svptrue_b8(), pg));
#else
    neon::BitNot(x, num_byte, result);
#endif
}

void
Prefetch(const void* data) {
#if defined(ENABLE_SVE)
    svprfd(svptrue_b8(), (const int8_t*)data, 0);
#else
    neon::Prefetch(data);
#endif
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_SVE)
    if (dim == 0)
        return;
    if (scalar == 0)
        scalar = 1.0f;
    svfloat32_t scalar_vec = svdup_f32(scalar);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        svfloat32_t from_vec = svld1_f32(pg, from + i);
        svfloat32_t res_vec = svdiv_f32_z(pg, from_vec, scalar_vec);
        svst1_f32(pg, to + i, res_vec);
        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
#if defined(ENABLE_SVE)
    float norm = std::sqrt(FP32ComputeIP(from, from, dim));
    DivScalar(from, to, dim, norm);
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
    for (size_t i = 0; i < pq_dim; i++) {
        const auto* dict = lookup_table;
        lookup_table += 16;
        const auto* code = codes;
        codes += 16;
        for (size_t j = 0; j < 16; j++) {
            uint8_t c = code[j];
            uint8_t lo_idx = c & 0x0F;
            uint8_t hi_idx = c >> 4;
            uint32_t lo_val = dict[lo_idx];
            uint32_t hi_val = dict[hi_idx];

            if (j % 2 == 0) {
                result[j / 2] += lo_val;
                result[16 + j / 2] += hi_val;
            } else {
                result[8 + j / 2] += lo_val;
                result[24 + j / 2] += hi_val;
            }
        }
    }
#else
    neon::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
KacsWalk(float* data, size_t len) {
#if defined(ENABLE_SVE)
    size_t base = len % 2;
    size_t offset = base + (len / 2);
    uint64_t i = 0;
    const uint64_t step = svcntw();
    svbool_t pg = svwhilelt_b32(i, len / 2);
    do {
        svfloat32_t vec1 = svld1_f32(pg, data + i);
        svfloat32_t vec2 = svld1_f32(pg, data + i + offset);
        svst1_f32(pg, data + i, svadd_f32_z(pg, vec1, vec2));
        svst1_f32(pg, data + i + offset, svsub_f32_z(pg, vec1, vec2));
        i += step;
        pg = svwhilelt_b32(i, len / 2);
    } while (svptest_any(svptrue_b32(), pg));

    if (base != 0) {
        data[len / 2] *= std::sqrt(2.0F);
    }
#else
    neon::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, size_t dim) {
#if defined(ENABLE_SVE)
    uint64_t i = 0;
    const uint64_t step = svcntw();
    float sign_lanes[step];

    svbool_t pg = svwhilelt_b32(i, dim);
    do {
        const uint64_t current_limit = i + step;
        for (uint64_t j = 0; i + j < current_limit; ++j) {
            if (i + j < dim) {
                bool should_flip = (flip[(i + j) / 8] & (1 << ((i + j) % 8))) != 0;
                sign_lanes[j] = should_flip ? -1.0f : 1.0f;
            } else {
                sign_lanes[j] = 1.0f;
            }
        }
        svfloat32_t sign_vec = svld1_f32(pg, sign_lanes);
        svfloat32_t data_vec = svld1_f32(pg, data + i);
        svfloat32_t res_vec = svmul_f32_z(pg, data_vec, sign_vec);
        svst1_f32(pg, data + i, res_vec);

        i += step;
        pg = svwhilelt_b32(i, dim);
    } while (svptest_any(svptrue_b32(), pg));
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
    } while (svptest_any(svptrue_b32(), pg));
#else
    neon::VecRescale(data, dim, val);
#endif
}

void
FHTRotate(float* data, size_t dim_) {
#if defined(ENABLE_SVE)
    size_t n = dim_;
    size_t step = 1;
    const uint64_t vec_step = svcntw();
    while (step < n) {
        for (size_t i = 0; i < n; i += 2 * step) {
            uint64_t j = 0;
            svbool_t pg = svwhilelt_b32(j, step);
            do {
                svfloat32_t x = svld1_f32(pg, data + i + j);
                svfloat32_t y = svld1_f32(pg, data + i + j + step);
                svst1_f32(pg, data + i + j, svadd_f32_z(pg, x, y));
                svst1_f32(pg, data + i + j + step, svsub_f32_z(pg, x, y));
                j += vec_step;
                pg = svwhilelt_b32(j, step);
            } while (svptest_any(svptrue_b32(), pg));
        }
        step *= 2;
    }
#else
    neon::FHTRotate(data, dim_);
#endif
}

}  // namespace vsag::sve