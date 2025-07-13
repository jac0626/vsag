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
    // TODO: SVE implementation here
    return neon::FP32ComputeIP(query, codes, dim);
#else
    return neon::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::FP32ComputeL2Sqr(query, codes, dim);
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
    // TODO: SVE implementation here
    neon::FP32ComputeIPBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
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
    // TODO: SVE implementation here
    neon::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#else
    neon::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
#endif
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::FP32Sub(x, y, z, dim);
#else
    neon::FP32Sub(x, y, z, dim);
#endif
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::FP32Add(x, y, z, dim);
#else
    neon::FP32Add(x, y, z, dim);
#endif
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::FP32Mul(x, y, z, dim);
#else
    neon::FP32Mul(x, y, z, dim);
#endif
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::FP32Div(x, y, z, dim);
#else
    neon::FP32Div(x, y, z, dim);
#endif
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::FP32ReduceAdd(x, dim);
#else
    return neon::FP32ReduceAdd(x, dim);
#endif
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::BF16ComputeIP(query, codes, dim);
#else
    return neon::BF16ComputeIP(query, codes, dim);
#endif
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::BF16ComputeL2Sqr(query, codes, dim);
#else
    return neon::BF16ComputeL2Sqr(query, codes, dim);
#endif
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::FP16ComputeIP(query, codes, dim);
#else
    return neon::FP16ComputeIP(query, codes, dim);
#endif
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    return neon::FP16ComputeL2Sqr(query, codes, dim);
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
    // TODO: SVE implementation here
    neon::BitAnd(x, y, num_byte, result);
#else
    neon::BitAnd(x, y, num_byte, result);
#endif
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::BitOr(x, y, num_byte, result);
#else
    neon::BitOr(x, y, num_byte, result);
#endif
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::BitXor(x, y, num_byte, result);
#else
    neon::BitXor(x, y, num_byte, result);
#endif
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::BitNot(x, num_byte, result);
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
FHTRotate(float* data, size_t dim_) {
#if defined(ENABLE_SVE)
    // TODO: SVE implementation here
    neon::FHTRotate(data, dim_);
#else
    neon::FHTRotate(data, dim_);
#endif
}

}  // namespace vsag::sve
