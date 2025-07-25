
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

#pragma once

#include <cstdint>

namespace vsag {
namespace generic {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);

float
NormalizeWithCentroid(const float* from, const float* centroid, float* to, uint64_t dim);

void
InverseNormalizeWithCentroid(
    const float* from, const float* centroid, float* to, uint64_t dim, float norm);
}  // namespace generic

namespace sse {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace sse

namespace avx {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace avx

namespace avx2 {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace avx2

namespace avx512 {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace avx512

namespace neon {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace neon

namespace sve {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace sve

using NormalizeType = float (*)(const float* from, float* to, uint64_t dim);
extern NormalizeType Normalize;

using NormalizeWithCentroidType = float (*)(const float* from,
                                            const float* centroid,
                                            float* to,
                                            uint64_t dim);
extern NormalizeWithCentroidType NormalizeWithCentroid;

using InverseNormalizeWithCentroidType =
    void (*)(const float* from, const float* centroid, float* to, uint64_t dim, float norm);
extern InverseNormalizeWithCentroidType InverseNormalizeWithCentroid;

using DivScalarType = void (*)(const float* from, float* to, uint64_t dim, float scalar);
extern DivScalarType DivScalar;

}  // namespace vsag
