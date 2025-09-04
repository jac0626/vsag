
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
#include <string>

#include "simd_marco.h"

namespace vsag {

namespace generic {
void
PQLookUp(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result);
}  // namespace generic

namespace sse {
void
PQLookUp(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result);
}  // namespace sse

namespace avx {
void
PQLookUp(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result);
}  // namespace avx

namespace avx2 {
void
PQLookUp(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result);
}  // namespace avx2

namespace avx512 {
void
PQLookUp(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result);
}  // namespace avx512

namespace neon {
void
PQLookUp(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result);
}  // namespace neon

namespace sve {
void
PQLookUp(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result);
}  // namespace sve

using PQLookUpType = void (*)(const uint8_t* RESTRICT lookup_table,
                                        const uint8_t* RESTRICT codes,
                                        uint64_t pq_dim,
                                        int32_t* RESTRICT result);
extern PQLookUpType PQLookUp;


namespace generic {
void
PQLookUpBatch4(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes1,const uint8_t* RESTRICT codes2,const uint8_t* RESTRICT codes3,const uint8_t* RESTRICT codes4,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result1,int32_t* RESTRICT result2,int32_t* RESTRICT result3,int32_t* RESTRICT result4);
}  // namespace generic

namespace sse {
void
PQLookUpBatch4(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes1,const uint8_t* RESTRICT codes2,const uint8_t* RESTRICT codes3,const uint8_t* RESTRICT codes4,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result1,int32_t* RESTRICT result2,int32_t* RESTRICT result3,int32_t* RESTRICT result4);
}  // namespace sse

namespace avx {
void
PQLookUpBatch4(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes1,const uint8_t* RESTRICT codes2,const uint8_t* RESTRICT codes3,const uint8_t* RESTRICT codes4,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result1,int32_t* RESTRICT result2,int32_t* RESTRICT result3,int32_t* RESTRICT result4);
}  // namespace avx

namespace avx2 {
void
PQLookUpBatch4(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes1,const uint8_t* RESTRICT codes2,const uint8_t* RESTRICT codes3,const uint8_t* RESTRICT codes4,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result1,int32_t* RESTRICT result2,int32_t* RESTRICT result3,int32_t* RESTRICT result4);
}  // namespace avx2

namespace avx512 {
void
PQLookUpBatch4(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes1,const uint8_t* RESTRICT codes2,const uint8_t* RESTRICT codes3,const uint8_t* RESTRICT codes4,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result1,int32_t* RESTRICT result2,int32_t* RESTRICT result3,int32_t* RESTRICT result4);
}  // namespace avx512

namespace neon {
void
PQLookUpBatch4(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes1,const uint8_t* RESTRICT codes2,const uint8_t* RESTRICT codes3,const uint8_t* RESTRICT codes4,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result1,int32_t* RESTRICT result2,int32_t* RESTRICT result3,int32_t* RESTRICT result4);
}  // namespace neon

namespace sve {
void
PQLookUpBatch4(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes1,const uint8_t* RESTRICT codes2,const uint8_t* RESTRICT codes3,const uint8_t* RESTRICT codes4,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result1,int32_t* RESTRICT result2,int32_t* RESTRICT result3,int32_t* RESTRICT result4);
}  // namespace sve

using PQLookUpBatch4Type = void (*)(const uint8_t* RESTRICT lookup_table,
                                        const uint8_t* RESTRICT codes1,const uint8_t* RESTRICT codes2,const uint8_t* RESTRICT codes3,const uint8_t* RESTRICT codes4,
                                        uint64_t pq_dim,
                                        int32_t* RESTRICT result1,int32_t* RESTRICT result2,int32_t* RESTRICT result3,int32_t* RESTRICT result4);
extern PQLookUpBatch4Type PQLookUpBatch4;

}  // namespace vsag
