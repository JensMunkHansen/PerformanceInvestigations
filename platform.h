#pragma once

#ifdef _MSC_VER
#include <malloc.h> // For _aligned_malloc
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#define BENCH_SCALE 1
#else
#include <cstdlib> // For std::aligned_alloc
#define ALIGNED_ALLOC(alignment, size) std::aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) std::free(ptr)
#define BENCH_SCALE 2
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(addr) __builtin_prefetch((addr), 0, 3)
#elif defined(_MSC_VER)
#include <xmmintrin.h>
#define PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define PREFETCH(addr) ((void)0) // fallback: no-op
#endif

#if defined(__clang__)
    #define PRAGMA_VECTORIZE_ENABLE _Pragma("clang loop vectorize(enable)")
    #define PRAGMA_IVDEP            _Pragma("ivdep") \
                                    _Pragma("clang loop vectorize(enable)")             
                                     

#elif defined(__GNUC__) || defined(__GNUG__)
    #define PRAGMA_VECTORIZE_ENABLE _Pragma("GCC ivdep") // GCC only supports ivdep
    #define PRAGMA_IVDEP            _Pragma("GCC ivdep")

#elif defined(_MSC_VER)
    #define PRAGMA_VECTORIZE_ENABLE __pragma(loop(ivdep)) __pragma(loop(vector))
    #define PRAGMA_IVDEP            __pragma(loop(ivdep))

#else
    // Unknown compiler: do nothing
    #define PRAGMA_VECTORIZE_ENABLE
    #define PRAGMA_IVDEP
#endif

