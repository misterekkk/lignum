// The CatBoost block kernel compiled for AVX2 (x86-64 GCC and Clang builds), chosen at run time
// by catboost.cpp when the CPU has AVX2. Every header is included before the target switch, so
// the standard library's inline functions keep the generic instruction set even where a copy of
// them is emitted here; only the kernel's own functions, internal to the avx2 namespace, use AVX2.
// tests/cpp/check_kernel_isa.py checks that no weakly linked function of this object uses VEX
// (AVX) instructions.

#include "catboost_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__clang__)
    #pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
    #pragma GCC push_options
    #pragma GCC target("avx2")
#endif

#define LIGNUM_CATBOOST_ISA avx2
#include "catboost_kernel.inc"
#undef LIGNUM_CATBOOST_ISA

#if defined(__clang__)
    #pragma clang attribute pop
#elif defined(__GNUC__)
    #pragma GCC pop_options
#endif
