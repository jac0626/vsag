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

#include "autotune_environment.h"

#include <cpuinfo.h>

#include <cstdint>
#include <exception>
#include <thread>

#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

#include "simd/simd.h"
#include "vsag/vsag.h"

namespace vsag::autotune::internal {

namespace {

JsonType
make_simd_feature(bool compiled, bool runtime) {
    return JsonType{
        {"compiled", compiled}, {"runtime", runtime}, {"effective", compiled && runtime}};
}

void
set_platform_evidence(JsonType& result) {
#if defined(__unix__) || defined(__APPLE__)
    struct utsname platform_info {};
    if (uname(&platform_info) == 0) {
        result["os"] = platform_info.sysname;
        result["os_release"] = platform_info.release;
        result["architecture"] = platform_info.machine;
    }
#endif

#if defined(__linux__)
    struct sysinfo memory_info {};
    if (sysinfo(&memory_info) == 0) {
        const auto total_memory_bytes = static_cast<uint64_t>(memory_info.totalram) *
                                        static_cast<uint64_t>(memory_info.mem_unit);
        result["total_memory_bytes"] = total_memory_bytes;
    }
#endif
}

void
set_cpu_evidence(JsonType& result) {
    const bool initialized = cpuinfo_initialize();
    result["cpuinfo_initialized"] = initialized;
    if (initialized) {
        result["physical_cores"] = cpuinfo_get_cores_count();
        result["logical_cores"] = cpuinfo_get_processors_count();
        if (cpuinfo_get_packages_count() > 0 && cpuinfo_get_package(0) != nullptr) {
            result["cpu_model"] = cpuinfo_get_package(0)->name;
        }
    } else {
        result["logical_cores"] = std::thread::hardware_concurrency();
    }
}

void
set_simd_evidence(JsonType& result) {
    const auto status = vsag::setup_simd();
    auto& simd = result["simd"];
    simd["sse"] = make_simd_feature(status.dist_support_sse, status.runtime_has_sse);
    simd["avx"] = make_simd_feature(status.dist_support_avx, status.runtime_has_avx);
    simd["avx2"] = make_simd_feature(status.dist_support_avx2, status.runtime_has_avx2);
    simd["avx512"] =
        make_simd_feature(status.dist_support_avx512f && status.dist_support_avx512dq &&
                              status.dist_support_avx512bw && status.dist_support_avx512vl,
                          status.runtime_has_avx512f && status.runtime_has_avx512dq &&
                              status.runtime_has_avx512bw && status.runtime_has_avx512vl);
    simd["avx512_vpopcntdq"] =
        make_simd_feature(status.dist_support_avx512vpopcntdq, status.runtime_has_avx512vpopcntdq);
    simd["amx"] = make_simd_feature(status.dist_support_amx, status.runtime_has_amx);
    simd["amx_bf16"] = make_simd_feature(status.dist_support_amx_bf16, status.runtime_has_amx_bf16);
    simd["neon"] = make_simd_feature(status.dist_support_neon, status.runtime_has_neon);
    simd["sve"] = make_simd_feature(status.dist_support_sve, status.runtime_has_sve);
}

}  // namespace

JsonType
CollectEnvironmentEvidence() {
    JsonType result = JsonType::object();
    try {
        result["vsag_version"] = vsag::version();
        set_platform_evidence(result);
        set_cpu_evidence(result);
        set_simd_evidence(result);
    } catch (const std::exception& error) {
        result["collection_error"] = error.what();
    } catch (...) {
        result["collection_error"] = "unknown environment evidence error";
    }
    return result;
}

}  // namespace vsag::autotune::internal
