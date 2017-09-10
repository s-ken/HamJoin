#ifndef INCLUDED_CONFIG
#define INCLUDED_CONFIG

constexpr int NUM_THREADS = 4;
constexpr float THRESHOLD = 0.9;

#if defined( __AVX512F__ )
    constexpr int SIMD_WIDTH = 64;
    constexpr int SIMD_WIDTH_INT = SIMD_WIDTH / 4;

#elif defined( __AVX2__ )
    constexpr int SIMD_WIDTH = 32;
    constexpr int SIMD_WIDTH_INT = SIMD_WIDTH / 4;

#endif

#endif
