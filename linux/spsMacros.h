#pragma once

#if defined(__GNUG__)
#define GCC_SPLIT_BLOCK(str) __asm__("//\n\t// " str "\n\t//\n");
#elif defined(_WIN64)
// Only 64-bit windows forbid inline assembly
#define GCC_SPLIT_BLOCK(str)
#else
#define GCC_SPLIT_BLOCK(str) __asm {}
#endif
