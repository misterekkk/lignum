#pragma once

// The CatBoost kernel's switches and per-thread scratch, shared by every instruction-set build of
// the block kernel (catboost_kernel.inc: generic in catboost.cpp, AVX2 in catboost_kernel_avx2.cpp).
// Only constants and plain data here: inline functions compiled once per instruction set would be
// merged by the linker, and an AVX2 copy could end up called on a CPU without AVX2.

#include "catboost.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lignum::catboost_kernel {

    // How a float feature's bucket (the number of its borders below the value) is computed;
    // LIGNUM_CATBOOST_FLOATS chooses for experiments (catboost-profile.yml measures each on
    // x86-64 and aarch64):
    //   0: bucket_search, a binary search written as conditional moves; ARM compilers keep
    //      them (csel), x86 ones make a data-dependent branch of each (clang: ucomiss; ja);
    //   1: count the borders one by one (vectorised over a block's rows, no select at all);
    //      features with more than kCountBorders borders search;
    //   2: bucket_two_level, every 16th border and then 16 borders;
    //   3 (default): per feature, up to kSmallFeature borders the faster of 0 / 1 on the
    //      architecture (ARM 0, x86 1), more borders 2.
    // CI (94bd651), 1 thread, 200k rows, u8 buckets: california (~123 borders per feature)
    // EPYC 7763 search 0.132 s, count 0.107, two levels 0.090; Neoverse-N2 0.071 / 0.086 /
    // 0.064. covtype (~21 per feature) EPYC 0.299 / 0.251 / 0.290; N2 0.182 / 0.193 / 0.185.
    #ifndef LIGNUM_CATBOOST_FLOATS
        #define LIGNUM_CATBOOST_FLOATS 3
    #endif
    constexpr int kFloatMode = LIGNUM_CATBOOST_FLOATS;
    constexpr size_t kCountBorders = 256;
    constexpr size_t kSmallFeature = 32;
    #if defined(__aarch64__) || defined(_M_ARM64)
        constexpr int kSmallFeatureMode = 0;
    #else
        constexpr int kSmallFeatureMode = 1;
    #endif

    // Buckets of a block as bytes when every source fits (ObliviousForest::small_buckets):
    // twice as many per vector, and unsigned byte compares exist in SSE2 (16-bit ones do not).
    // CI (94bd651), 1 thread, 200k rows: california EPYC 0.101 -> 0.090 s, Neoverse-N2 0.086 ->
    // 0.064 s; covtype 0.288 -> 0.290 / 0.206 -> 0.185 (two levels); M4 within noise.
    #ifndef LIGNUM_CATBOOST_U8
        #define LIGNUM_CATBOOST_U8 1
    #endif
    constexpr bool kSmallBuckets = LIGNUM_CATBOOST_U8 != 0;

    // Trees whose leaves a block adds together (phase2): each row's sums stay in registers for
    // a group instead of being loaded and stored per tree; still one addition per tree, in
    // model order.
    #ifndef LIGNUM_CATBOOST_GROUP
        #define LIGNUM_CATBOOST_GROUP 4
    #endif
    constexpr int kGroup = LIGNUM_CATBOOST_GROUP;
    static_assert(kGroup >= 1 && kGroup <= 16, "LIGNUM_CATBOOST_GROUP: 1 to 16 trees");

    // Leaf indices as bytes when every tree has depth <= 8 (twice as many per vector, like
    // CatBoost's evaluator); LIGNUM_CATBOOST_INDEX8=0 keeps 16 bits.
    #ifndef LIGNUM_CATBOOST_INDEX8
        #define LIGNUM_CATBOOST_INDEX8 1
    #endif
    constexpr bool kByteIndex = LIGNUM_CATBOOST_INDEX8 != 0;

    // LIGNUM_CATBOOST_ASM (diagnostic builds): the hot loops as their own functions, so their
    // assembly can be read with objdump (benchmarks/catboost_profile.py).
    #ifdef LIGNUM_CATBOOST_ASM
        #if defined(_MSC_VER)
            #define LIGNUM_HOT_LOOP __declspec(noinline)
        #else
            #define LIGNUM_HOT_LOOP __attribute__((noinline))
        #endif
    #else
        #define LIGNUM_HOT_LOOP
    #endif

    // Trees without one-hot splits get their leaf index from plain `bucket >= t` tests, the
    // bits gathered from the deepest level up (v = 2 v + bit): one constant per level instead
    // of three (threshold, range, bit mask). With depth 6 that is 6 vector registers instead
    // of 18, which SSE2's 16 could not hold (GCC reloaded 8 of them from the stack in the hot
    // loop). LIGNUM_CATBOOST_GE=0 uses the range test for every tree. CI (ba6d01b), 1 thread,
    // 200k rows, range test -> plain: EPYC 9V45 california 0.058 -> 0.054 s (no stack accesses
    // left in the loop), Neoverse-N2 0.070 -> 0.062, covtype 0.203 -> 0.190.
    #ifndef LIGNUM_CATBOOST_GE
        #define LIGNUM_CATBOOST_GE 1
    #endif
    constexpr bool kPlainSplits = LIGNUM_CATBOOST_GE != 0;


    // Rows per block (bucket and index arrays are this many rows long per source and tree);
    // LIGNUM_CATBOOST_BLOCK for experiments. CI (ba6d01b), 1 thread, 200k rows, 128 -> 256:
    // EPYC 9V45 california 0.054 -> 0.048 s, adult 0.149 -> 0.139, covtype 0.113 -> 0.104;
    // Neoverse-N2 0.062 -> 0.059, 0.242 -> 0.228, 0.190 -> 0.174.
    #ifndef LIGNUM_CATBOOST_BLOCK
        #define LIGNUM_CATBOOST_BLOCK 256
    #endif
    constexpr size_t kBlock = LIGNUM_CATBOOST_BLOCK;
    // Batches below this many rows take the per-row path (see predict_row).
    #ifndef LIGNUM_CATBOOST_ROW_PATH
        #define LIGNUM_CATBOOST_ROW_PATH 4
    #endif
    constexpr size_t kRowPathMax = LIGNUM_CATBOOST_ROW_PATH;

    // Scratch of one thread: buckets of every source for a block (column-major, so the
    // per-tree loops over rows vectorise), category hashes, table entries, leaf indices.
    // Kept per thread and only grown, so a single row allocates and clears nothing. Arrays
    // are [source][row] with a stride of the block's row count.
    struct Scratch {
        std::vector<uint16_t> buckets;
        std::vector<uint32_t> hashes;
        std::vector<uint32_t> entries;
        std::vector<uint64_t> combination;
        std::vector<uint64_t> prefixes;  // [level][row]: hash of the current prefix-tree node per level
        std::vector<uint16_t> index;
        std::vector<double> sums;
        std::vector<uint8_t> conditions;  // single-row path
        std::vector<float> values;        // one float feature of the block
        std::vector<uint32_t> counts;     // per row: coarse borders below the value (two levels)
        std::vector<uint8_t> small;       // buckets as bytes (ObliviousForest::small_buckets)

        void prepare(const ObliviousForest& f, size_t rows);  // grows the arrays (catboost.cpp)
    };


    using BlockKernel = void (*)(const ObliviousForest& f, const double* X, size_t n_features, size_t rows, double* out, Scratch& s);

    // The block kernel per instruction set: rows of X (n_features columns) into out; s prepared
    // for `rows` rows.
    namespace generic {
        void predict_block(const ObliviousForest& f, const double* X, size_t n_features, size_t rows, double* out, Scratch& s);
    }
    namespace avx2 {
        void predict_block(const ObliviousForest& f, const double* X, size_t n_features, size_t rows, double* out, Scratch& s);
    }

} // namespace lignum::catboost_kernel
