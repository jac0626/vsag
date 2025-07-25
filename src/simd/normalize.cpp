
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

#include "normalize.h"

#include "simd_status.h"

namespace vsag {

static NormalizeType
GetNormalize() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::Normalize;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::Normalize;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::Normalize;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::Normalize;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::Normalize;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::Normalize;
#endif
    }
    return generic::Normalize;
}
NormalizeType Normalize = GetNormalize();

static NormalizeWithCentroidType
GetNormalizeWithCentroid() {
    return generic::NormalizeWithCentroid;
}
NormalizeWithCentroidType NormalizeWithCentroid = GetNormalizeWithCentroid();

static InverseNormalizeWithCentroidType
GetInverseNormalizeWithCentroid() {
    return generic::InverseNormalizeWithCentroid;
}
InverseNormalizeWithCentroidType InverseNormalizeWithCentroid = GetInverseNormalizeWithCentroid();

static DivScalarType
GetDivScalar() {
    if (SimdStatus::SupportAVX512()) {
#if defined(ENABLE_AVX512)
        return avx512::DivScalar;
#endif
    } else if (SimdStatus::SupportAVX2()) {
#if defined(ENABLE_AVX2)
        return avx2::DivScalar;
#endif
    } else if (SimdStatus::SupportAVX()) {
#if defined(ENABLE_AVX)
        return avx::DivScalar;
#endif
    } else if (SimdStatus::SupportSSE()) {
#if defined(ENABLE_SSE)
        return sse::DivScalar;
#endif
    } else if (SimdStatus::SupportSVE()) {
#if defined(ENABLE_SVE)
        return sve::DivScalar;
#endif
    } else if (SimdStatus::SupportNEON()) {
#if defined(ENABLE_NEON)
        return neon::DivScalar;
#endif
    }
    return generic::DivScalar;
}
DivScalarType DivScalar = GetDivScalar();

}  // namespace vsag
