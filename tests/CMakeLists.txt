
#  unittests
file (GLOB_RECURSE UNIT_TESTS "../src/simd/*_test.cpp")
add_executable (unittests ${UNIT_TESTS}
        test_main.cpp
        fixtures/fixtures.cpp
)
if (DIST_CONTAINS_SSE)
    target_compile_definitions (unittests PRIVATE ENABLE_SSE=1)
endif ()
if (DIST_CONTAINS_AVX)
    target_compile_definitions (unittests PRIVATE ENABLE_AVX=1)
endif ()
if (DIST_CONTAINS_AVX2)
    target_compile_definitions (unittests PRIVATE ENABLE_AVX2=1)
endif ()
if (DIST_CONTAINS_AVX512)
    target_compile_definitions (unittests PRIVATE ENABLE_AVX512=1)
endif ()
if (DIST_CONTAINS_AVX512VPOPCNTDQ)
    target_compile_definitions (unittests PRIVATE DENABLE_AVX512VPOPCNTDQ=1)
endif ()
if (DIST_CONTAINS_NEON)
    target_compile_definitions (unittests PRIVATE ENABLE_NEON=1)
endif ()
target_include_directories (unittests PRIVATE "./fixtures")
target_link_libraries (unittests PRIVATE Catch2::Catch2  fmt::fmt simd)
add_dependencies (unittests fmt Catch2)

