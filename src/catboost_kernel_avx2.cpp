// The CatBoost block kernel compiled for AVX2 (x86-64 builds), chosen at run time by catboost.cpp
// when the CPU has AVX2. GCC and Clang: every header is included before the target switch, so
// the standard library's inline functions keep the generic instruction set even where a copy of
// them is emitted here; only the kernel's own functions, internal to the avx2 namespace, use AVX2.
// MSVC: the whole file is compiled with /arch:AVX2 (CMakeLists.txt), so no function outside the
// avx2 namespace may be emitted here with external linkage (the linker could pick this copy for
// every caller). tests/cpp/check_kernel_isa.py checks both: the linked library (objdump) and this
// object file (dumpbin).

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
#define LIGNUM_CATBOOST_ISA_AVX2 1
#include "catboost_kernel.inc"
#undef LIGNUM_CATBOOST_ISA_AVX2
#undef LIGNUM_CATBOOST_ISA

#if defined(__clang__)
    #pragma clang attribute pop
#elif defined(__GNUC__)
    #pragma GCC pop_options
#endif
