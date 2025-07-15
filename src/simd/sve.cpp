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
    // TODO: SVE implementation here
    return neon::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
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
    // TODO: SVE implementation here
    neon::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
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
        svprfw(svptrue_b32(), query + i + step, SV_PLDL1KEEP);
        svprfw(svptrue_b32(), codes + i + step, SV_PLDL1KEEP);

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

        sum_vec = svmla_f32_m(pg, sum_vec, svcvt_f32_bf16_z(pg, q_vec), svcvt_f32_bf16_z(pg, c_vec));

        i += step;
        pg = svwhilelt_b16(i, dim);
    } while (svptest_any(svptrue_b16(), pg));

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
    } while (svptest_any(svptrue_b16(), pg));

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
    } while (svptest_any(svptrue_b16(), pg));

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
    } while (svptest_any(svptrue_b16(), pg));

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
    // TODO: SVE implementation here
    return neon::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
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
    // TODO: SVE implementation here
    return neon::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
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
    // TODO: SVE implementation here
    return neon::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
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
    // TODO: SVE implementation here
    return neon::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
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
    // TODO: SVE implementation here
    return neon::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
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
    // TODO: SVE implementation here
    return neon::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
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
    // TODO: SVE implementation here
    return neon::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
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
    // TODO: SVE implementation here
    return neon::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#else
    return neon::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#else
    return neon::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#else
    return neon::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
#else
    return neon::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
#endif
}

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::RaBitQSQ4UBinaryIP(codes, bits, dim);
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
    // TODO: SVE implementation here
    neon::Prefetch(data);
#else
    neon::Prefetch(data);
#endif
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::DivScalar(from, to, dim, scalar);
#else
    neon::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::Normalize(from, to, dim);
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
    // TODO: SVE implementation here
    neon::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#else
    neon::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
#endif
}

void
KacsWalk(float* data, size_t len) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::KacsWalk(data, len);
#else
    neon::KacsWalk(data, len);
#endif
}

void
FlipSign(const uint8_t* flip, float* data, size_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::FlipSign(flip, data, dim);
#else
    neon::FlipSign(flip, data, dim);
#endif
}

void
VecRescale(float* data, size_t dim, float val) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::VecRescale(data, dim, val);
#else
    neon::VecRescale(data, dim, val);
#endif
}

void
RotateOp(float* data, int idx, int dim_, int step) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::RotateOp(data, idx, dim_, step);
#else
    neon::RotateOp(data, idx, dim_, step);
#endif
}


void
FHTRotate(float* data, size_t dim_) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::FHTRotate(data, dim_);
#else
    neon::FHTRotate(data, dim_);
#endif
}

}  // namespace vsag::sve
