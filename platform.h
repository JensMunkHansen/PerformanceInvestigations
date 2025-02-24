#pragma once

#ifdef _MSC_VER
#include <malloc.h> // For _aligned_malloc
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <cstdlib> // For std::aligned_alloc
#define ALIGNED_ALLOC(alignment, size) std::aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) std::free(ptr)
#endif
