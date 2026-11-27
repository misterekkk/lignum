#include "catboost.hpp"
#include "catboost_internal.hpp"
#include "catboost_kernel.hpp"

#include <lignum/model.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    #include <intrin.h>
    #include <immintrin.h>
#endif

// The block kernel for any CPU of the target (see catboost_kernel_avx2.cpp for AVX2).
#define LIGNUM_CATBOOST_ISA generic
#include "catboost_kernel.inc"
#undef LIGNUM_CATBOOST_ISA

namespace lignum {

    using namespace catboost_kernel;
    using namespace catboost_kernel::generic;

    namespace catboost_detail {

        bool cpu_has_avx2() {
            static const bool supported = [] {
            #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                #if defined(_MSC_VER)
                    int info[4];
                    __cpuid(info, 1);
                    const bool avx = (info[2] & (1 << 28)) && (info[2] & (1 << 27)) && ((_xgetbv(0) & 6) == 6);
                    __cpuidex(info, 7, 0);
                    return avx && (info[1] & (1 << 5));
                #else
                    __builtin_cpu_init();
                    return __builtin_cpu_supports("avx") && __builtin_cpu_supports("avx2");
                #endif
            #else
                return false;
            #endif
            }();
            return supported;
        }

        namespace {

            // Float borders padded with +inf to a power of two, for bucket_search.
            void pad_borders(ObliviousForest& f) {
                constexpr float inf = std::numeric_limits<float>::infinity();
                for (ObliviousForest::Float& x : f.floats) {
                    size_t n = 1;
                    while (n < x.borders.size()) n <<= 1;
                    x.search = x.borders;
                    x.search.resize(n, inf);
                    // Two levels (bucket_two_level): count the coarse borders below the value, then the
                    // 16 borders of the block that follows. The extra block of +inf keeps that block in
                    // range when the value is above every coarse border.
                    const size_t blocks = (x.borders.size() + 15) / 16;
                    x.fine = x.borders;
                    x.fine.resize((blocks + 1) * 16, inf);
                    x.coarse.clear();
                    for (size_t k = 0; k < blocks; k++) x.coarse.push_back(x.fine[16 * k + 15]);
                }
            }

            // Fills the forest's prefix tree from its projections (see ObliviousForest::prefix_cat):
            // projections sorted by category list, one node per category not shared with the previous
            // list. In this order every node's parent is the latest node one level up.
            void index_prefixes(ObliviousForest& f) {
                std::vector<uint32_t> order(f.projections.size());
                for (uint32_t p = 0; p < order.size(); p++) order[p] = p;
                std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return f.projections[a].cats < f.projections[b].cats; });
                f.prefix_cat.clear();
                f.prefix_level.clear();
                f.prefix_projection_begin.assign(1, 0);
                f.prefix_projections.clear();
                f.prefix_levels = 0;
                const std::vector<uint32_t>* previous = nullptr;
                for (uint32_t p : order) {
                    const std::vector<uint32_t>& cats = f.projections[p].cats;
                    if (cats.empty() || cats.size() > 255) throw std::runtime_error("CatBoost projection with " + std::to_string(cats.size()) + " categories.");
                    if (!previous || cats != *previous) {  // new nodes from the first category not shared with the previous list
                        size_t common = 0;
                        while (previous && common + 1 < cats.size() && common < previous->size() && cats[common] == (*previous)[common]) common++;
                        for (size_t level = common; level < cats.size(); level++) {
                            f.prefix_cat.push_back(cats[level]);
                            f.prefix_level.push_back(static_cast<uint8_t>(level));
                            f.prefix_projection_begin.push_back(static_cast<uint32_t>(f.prefix_projections.size()));
                        }
                    }
                    f.prefix_projections.push_back(p);  // to the latest node: its category list is this one
                    f.prefix_projection_begin.back() = static_cast<uint32_t>(f.prefix_projections.size());
                    f.prefix_levels = std::max(f.prefix_levels, static_cast<uint32_t>(cats.size()));
                    previous = &cats;
                }
            }

            // The condition bytes of the single-row path (see ObliviousForest::condition_begin), the
            // range test of every split, and the runs of trees of one depth.
            void index_conditions(ObliviousForest& f) {
                const size_t n_sources = f.n_sources();
                std::vector<uint32_t> count(n_sources, 0);
                for (size_t k = 0; k < f.split_source.size(); k++) {
                    const uint16_t s = f.split_source[k];
                    if (s >= n_sources) throw std::runtime_error("CatBoost split on an unknown source.");
                    count[s] = std::max<uint32_t>(count[s], f.split_threshold[k]);
                }
                f.condition_begin.assign(1, 0);
                for (size_t s = 0; s < n_sources; s++) f.condition_begin.push_back(f.condition_begin.back() + count[s]);
                f.split_condition.resize(f.split_source.size());
                for (size_t k = 0; k < f.split_source.size(); k++) {
                    f.split_condition[k] = f.condition_begin[f.split_source[k]] + f.split_threshold[k] - 1;
                }
                f.split_range.resize(f.split_source.size());
                for (size_t k = 0; k < f.split_source.size(); k++) {
                    f.split_range[k] = f.split_equal[k] ? uint16_t{0} : static_cast<uint16_t>(0xffff - f.split_threshold[k]);
                }
                // Condition bytes pay when sources are few next to splits: every source writes its
                // bytes per row. M4, 500 trees, bytes vs comparing buckets: california (8 sources, 3000
                // splits) 0.62 vs 0.92 µs, covtype (53) 1.02 vs 1.23, bank (268, 2969) 1.60 vs 1.67;
                // adult (364, 2981) 2.25 vs 2.08, diabetes (1228, 2943) 6.7 vs 5.1.
                f.row_conditions = n_sources * 10 <= f.split_source.size();
                f.tree_equal.assign(f.n_trees(), 0);
                for (size_t t = 0; t < f.n_trees(); t++) {
                    for (uint32_t k = f.split_begin[t]; k < f.split_begin[t + 1]; k++) f.tree_equal[t] |= f.split_equal[k];
                }
                f.small_buckets = true;
                for (const auto& x : f.floats) f.small_buckets &= x.borders.size() <= 255;
                for (const auto& x : f.one_hots) f.small_buckets &= x.values.size() <= 255;
                for (const auto& c : f.ctrs) {
                    for (uint16_t b : c.entry_buckets) f.small_buckets &= b <= 255;
                }
                for (uint16_t t : f.split_threshold) f.small_buckets &= t <= 255;
                f.run_end.clear();
                f.run_depth.clear();
                f.max_depth = 0;
                for (size_t t = 0; t < f.n_trees(); t++) {
                    const uint32_t depth = f.split_begin[t + 1] - f.split_begin[t];
                    if (depth > 255) throw std::runtime_error("CatBoost tree deeper than 255.");
                    if (f.run_depth.empty() || f.run_depth.back() != depth) {
                        f.run_end.push_back(static_cast<uint32_t>(t));
                        f.run_depth.push_back(static_cast<uint8_t>(depth));
                    }
                    f.run_end.back() = static_cast<uint32_t>(t + 1);
                    f.max_depth = std::max(f.max_depth, depth);
                }
            }

        } // namespace

        void derive_layout(ObliviousForest& f) {
            pad_borders(f);
            index_prefixes(f);
            index_conditions(f);
            size_t table_bytes = 0;
            for (const auto& t : f.tables) table_bytes += t.keys.size() * (sizeof(uint64_t) + sizeof(uint32_t));
            const size_t table_weight = table_bytes > (2u << 20) ? 5 : 2;
            f.row_work = f.n_trees() * std::max<size_t>(1, f.dim) / 8 + f.floats.size() * 3 + f.tables.size() * table_weight;
        }

    } // namespace catboost_detail

    // ---------------------------------------------------------------------------------------------
    // Kernel

    void catboost_kernel::Scratch::prepare(const ObliviousForest& f, size_t rows) {
        auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
        grow(buckets, f.n_sources() * rows);
        grow(hashes, f.cat_columns.size() * rows);
        grow(entries, f.tables.size() * rows);
        grow(combination, rows);
        grow(prefixes, f.prefix_levels * rows);
        grow(index, kGroup * rows);
        grow(sums, rows * f.dim);
        grow(values, rows);
        grow(counts, rows);
        if (kSmallBuckets && f.small_buckets) grow(small, f.n_sources() * rows);
        if (rows == 1 && f.row_conditions) grow(conditions, f.condition_begin.back() + 8);  // + 8: stores of 8 bytes
    }

    namespace {

        // The block kernel for this CPU: AVX2 where the CPU and the build have it, else generic.
        // LIGNUM_ISA=generic or avx2 chooses (tests run both); avx2 without AVX2 support falls back.
        BlockKernel select_block_kernel() {
            [[maybe_unused]] bool avx2 = false;  // only read where the AVX2 kernel is built
        #if defined(LIGNUM_CATBOOST_AVX2)
            avx2 = catboost_detail::cpu_has_avx2();
        #endif
            const char* isa = std::getenv("LIGNUM_ISA");
            if (isa && std::string_view(isa) == "generic") avx2 = false;
        #if defined(LIGNUM_CATBOOST_AVX2)
            if (avx2) return &catboost_kernel::avx2::predict_block;
        #endif
            return &catboost_kernel::generic::predict_block;
        }

        void run_block(const ObliviousForest& f, const double* X, size_t n_features, size_t rows, double* out, Scratch& s) {
            static const BlockKernel kernel = select_block_kernel();
            s.prepare(f, rows);
            kernel(f, X, n_features, rows, out, s);
        }

        Scratch& thread_scratch() {
            thread_local Scratch scratch;
            return scratch;
        }

        // Trees [t0, t1) of one row, all of depth Depth (known at compile time: a tree's leaf index
        // is Depth steps without a loop). Bytes: each split is one byte of the row's condition array
        // (ObliviousForest::condition_begin); otherwise it compares the source's bucket with one
        // unsigned range test. Dim = 0: any number of outputs. The sums carry over from the
        // previous run, so leaves are still added in model order.
        template <int Depth, int Dim, bool Bytes>
        void row_trees(const ObliviousForest& f, size_t t0, size_t t1, const uint16_t* buckets, const uint8_t* bits, double* io) {
            const size_t dim = Dim > 0 ? Dim : f.dim;
            double local[Dim > 0 ? Dim : 1];
            double* sums = Dim > 0 ? local : io;
            if constexpr (Dim > 0) std::copy(io, io + Dim, local);
            const double* leaves = f.leaves.data() + f.leaf_begin[t0];
            const size_t leaves_per_tree = dim << Depth;
            for (size_t t = t0, k = f.split_begin[t0]; t < t1; t++, k += Depth, leaves += leaves_per_tree) {
                uint32_t index = 0;
                for (int d = 0; d < Depth; d++) {
                    if constexpr (Bytes) {
                        index |= static_cast<uint32_t>(bits[f.split_condition[k + d]]) << d;
                    } else {
                        const uint16_t v = static_cast<uint16_t>(buckets[f.split_source[k + d]] - f.split_threshold[k + d]);
                        index |= static_cast<uint32_t>(v <= f.split_range[k + d]) << d;
                    }
                }
                const double* l = leaves + index * dim;
                for (size_t d = 0; d < dim; d++) sums[d] += l[d];
            }
            if constexpr (Dim > 0) std::copy(local, local + Dim, io);
        }

        // Trees [t0, t1) of any depths (deeper than the specialised ones).
        void row_trees_any(const ObliviousForest& f, size_t t0, size_t t1, const uint16_t* buckets, double* sums) {
            const size_t dim = f.dim;
            for (size_t t = t0; t < t1; t++) {
                uint32_t index = 0;
                for (uint32_t k = f.split_begin[t], bit = 0; k < f.split_begin[t + 1]; k++, bit++) {
                    const uint16_t v = static_cast<uint16_t>(buckets[f.split_source[k]] - f.split_threshold[k]);
                    index |= static_cast<uint32_t>(v <= f.split_range[k]) << bit;
                }
                const double* l = f.leaves.data() + f.leaf_begin[t] + index * dim;
                for (size_t d = 0; d < dim; d++) sums[d] += l[d];
            }
        }

        template <int Depth, bool Bytes>
        void row_trees_dim(const ObliviousForest& f, size_t t0, size_t t1, const uint16_t* buckets, const uint8_t* bits, double* sums) {
            switch (f.dim) {
                case 1: row_trees<Depth, 1, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 2: row_trees<Depth, 2, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 3: row_trees<Depth, 3, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 4: row_trees<Depth, 4, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 5: row_trees<Depth, 5, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 6: row_trees<Depth, 6, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 7: row_trees<Depth, 7, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 8: row_trees<Depth, 8, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 9: row_trees<Depth, 9, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 10: row_trees<Depth, 10, Bytes>(f, t0, t1, buckets, bits, sums); break;
                default: row_trees<Depth, 0, Bytes>(f, t0, t1, buckets, bits, sums); break;
            }
        }

        template <int Depth>
        void row_trees_kind(const ObliviousForest& f, size_t t0, size_t t1, const uint16_t* buckets, const uint8_t* bits, double* sums) {
            if (bits) row_trees_dim<Depth, true>(f, t0, t1, buckets, bits, sums);
            else row_trees_dim<Depth, false>(f, t0, t1, buckets, bits, sums);
        }

        // All trees of one row, run by run. bits: the row's condition bytes, or null to compare buckets.
        void row_trees_all(const ObliviousForest& f, const uint16_t* buckets, const uint8_t* bits, double* out, Scratch& s) {
            double* sums = s.sums.data();
            std::fill(sums, sums + f.dim, 0.0);
            for (size_t r = 0, t0 = 0; r < f.run_end.size(); t0 = f.run_end[r], r++) {
                const size_t t1 = f.run_end[r];
                switch (f.run_depth[r]) {
                    case 0: row_trees_kind<0>(f, t0, t1, buckets, bits, sums); break;
                    case 1: row_trees_kind<1>(f, t0, t1, buckets, bits, sums); break;
                    case 2: row_trees_kind<2>(f, t0, t1, buckets, bits, sums); break;
                    case 3: row_trees_kind<3>(f, t0, t1, buckets, bits, sums); break;
                    case 4: row_trees_kind<4>(f, t0, t1, buckets, bits, sums); break;
                    case 5: row_trees_kind<5>(f, t0, t1, buckets, bits, sums); break;
                    case 6: row_trees_kind<6>(f, t0, t1, buckets, bits, sums); break;
                    case 7: row_trees_kind<7>(f, t0, t1, buckets, bits, sums); break;
                    case 8: row_trees_kind<8>(f, t0, t1, buckets, bits, sums); break;
                    case 9: row_trees_kind<9>(f, t0, t1, buckets, bits, sums); break;
                    case 10: row_trees_kind<10>(f, t0, t1, buckets, bits, sums); break;
                    default: row_trees_any(f, t0, t1, buckets, sums); break;
                }
            }
            for (size_t d = 0; d < f.dim; d++) out[d] = f.scale * sums[d] + f.bias[d];
        }

        // One row without the block layout: a few rows would pay a loop over rows per split.
        void predict_row(const ObliviousForest& f, const double* x, double* out, Scratch& s) {
            const size_t F = f.floats.size(), H = f.one_hots.size();
            s.prepare(f, 1);
            uint16_t* buckets = s.buckets.data();
            uint32_t* hashes = s.hashes.data();
            uint32_t* entries = s.entries.data();
            for (size_t k = 0; k < F; k++) {
                const ObliviousForest::Float& ff = f.floats[k];
                const float v = static_cast<float>(x[ff.column]);
                if (std::isnan(v)) {
                    buckets[k] = ff.nan_as_true ? static_cast<uint16_t>(ff.borders.size()) : 0;
                } else {
                    buckets[k] = float_bucket(ff, v);
                }
            }
            for (size_t c = 0; c < f.cat_columns.size(); c++) hashes[c] = static_cast<uint32_t>(x[f.cat_columns[c]]);
            for (size_t k = 0; k < H; k++) {
                const ObliviousForest::OneHot& oh = f.one_hots[k];
                const uint32_t h = hashes[oh.cat];
                uint16_t v = 0;
                for (size_t j = 0; j < oh.values.size(); j++) v = oh.values[j] == h ? static_cast<uint16_t>(j + 1) : v;
                buckets[F + k] = v;
            }
            uint64_t* prefix = s.prefixes.data();
            for (size_t n = 0; n < f.prefix_cat.size(); n++) {
                const uint32_t level = f.prefix_level[n];
                prefix[level] = combine_hash(level ? prefix[level - 1] : 0, static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(hashes[f.prefix_cat[n]]))));
                for (uint32_t i = f.prefix_projection_begin[n]; i < f.prefix_projection_begin[n + 1]; i++) {
                    const ObliviousForest::Projection& proj = f.projections[f.prefix_projections[i]];
                    uint64_t hash = prefix[level];
                    for (const auto& [source, threshold] : proj.floats) hash = combine_hash(hash, buckets[source] >= threshold ? 1 : 0);
                    for (const auto& [source, value] : proj.one_hots) hash = combine_hash(hash, buckets[source] == value ? 1 : 0);
                    for (uint32_t t : proj.tables) {
                        const ObliviousForest::Table& table = f.tables[t];
                        uint64_t slot = slot_of(hash, table.mask);
                        uint32_t found = 0;
                        while (table.slots[slot] != 0) {
                            if (table.keys[slot] == hash) { found = table.slots[slot]; break; }
                            slot = (slot + 1) & table.mask;
                        }
                        entries[t] = found;
                    }
                }
            }
            for (size_t k = 0; k < f.ctrs.size(); k++) {
                const ObliviousForest::Ctr& ctr = f.ctrs[k];
                const uint32_t e = entries[ctr.table];
                buckets[F + H + k] = ctr.entry_buckets[e];
            }

            if (!f.row_conditions) {
                row_trees_all(f, buckets, nullptr, out, s);
                return;
            }
            static_assert(std::endian::native == std::endian::little, "condition bytes are written as little-endian words");
            uint8_t* bits = s.conditions.data();
            constexpr uint64_t kOnes = 0x0101010101010101ULL;
            for (size_t k = 0; k + 1 < f.condition_begin.size(); k++) {
                const uint32_t v = buckets[k];
                uint8_t* b = bits + f.condition_begin[k];
                const uint32_t n = f.condition_begin[k + 1] - f.condition_begin[k];
                const bool one_hot = k >= F && k < F + H;
                for (uint32_t j = 0; j < n; j += 8, b += 8) {
                    uint64_t word;
                    if (one_hot) {  // byte v - 1 - j
                        word = v > j && v - j <= 8 ? uint64_t{1} << (8 * (v - 1 - j)) : 0;
                    } else {        // bytes below v - j
                        const uint32_t below = v > j ? std::min<uint32_t>(v - j, 8) : 0;
                        word = below == 8 ? kOnes : kOnes & ((uint64_t{1} << (8 * below)) - 1);
                    }
                    std::memcpy(b, &word, 8);  // little-endian byte order: byte i is bits 8i..8i+7
                }
            }
            row_trees_all(f, buckets, bits, out, s);
        }

        void check_categories(const ObliviousForest& f, const double* X, size_t n_samples, size_t n_features) {
            for (uint32_t column : f.cat_columns) {
                for (size_t i = 0; i < n_samples; i++) {
                    const double v = X[i * n_features + column];
                    if (!(v >= 0.0 && v <= 4294967295.0 && v == std::floor(v))) {
                        throw std::invalid_argument("Column " + std::to_string(column) +
                                                    " is categorical: it must hold category hashes (lignum.catboost_hash), got " + std::to_string(v) + ".");
                    }
                }
            }
        }

        void finish_oblivious_block(double* dst, size_t rows, size_t dim, uint8_t tr, double sigmoid, bool raw_score) {
            if (raw_score || tr == transform::Identity) return;
            if (tr == transform::Exp) {
                catboost_exp(dst, rows * dim);
            } else if (tr == transform::Softmax) {
                catboost_softmax(dst, rows, dim);
            } else if (tr == transform::Sigmoid) {
                const size_t n = rows * dim;
                for (size_t i = 0; i < n; i++) dst[i] = 1.0 / (1.0 + std::exp(-sigmoid * dst[i]));
            }
        }

    } // namespace

    void predict_oblivious(const ObliviousForest& f, const double* X, size_t n_samples, size_t n_features, double* out,
                           int n_threads, Strategy strategy, uint8_t tr, double sigmoid, bool raw_score) {
        if (n_samples < kRowPathMax) {
            check_categories(f, X, n_samples, n_features);
            Scratch& s = thread_scratch();
            for (size_t i = 0; i < n_samples; i++) predict_row(f, X + i * n_features, out + i * f.dim, s);
            finish_oblivious_block(out, n_samples, f.dim, tr, sigmoid, raw_score);
            return;
        }
        constexpr size_t kMinAutoParallelRows = 64;
        constexpr size_t kMinAutoParallelWorkSmall = 200000;
        constexpr size_t kMinAutoParallelWork = 140000;
        const size_t min_work = n_samples < kBlock ? kMinAutoParallelWorkSmall : kMinAutoParallelWork;
        const bool auto_too_small = strategy == Strategy::Auto &&
                                    (n_samples < kMinAutoParallelRows || n_samples * f.row_work < min_work);
        const bool serial = strategy == Strategy::Serial || n_threads <= 1 || auto_too_small;
        if (serial) {
            const size_t n_blocks = (n_samples + kBlock - 1) / kBlock;
            Scratch& s = thread_scratch();
            for (size_t b = 0; b < n_blocks; b++) {
                const size_t i = b * kBlock;
                const size_t count = std::min(kBlock, n_samples - i);
                check_categories(f, X + i * n_features, count, n_features);
                run_block(f, X + i * n_features, n_features, count, out + i * f.dim, s);
                finish_oblivious_block(out + i * f.dim, count, f.dim, tr, sigmoid, raw_score);
            }
            return;
        }
        const size_t per_thread = (n_samples + 2 * static_cast<size_t>(n_threads) - 1) / (2 * static_cast<size_t>(n_threads));
        const size_t block = std::clamp<size_t>((per_thread + 3) / 4 * 4, size_t{8}, kBlock);
        const size_t n_blocks = (n_samples + block - 1) / block;
        if (n_blocks == 1) {
            Scratch& s = thread_scratch();
            check_categories(f, X, n_samples, n_features);
            run_block(f, X, n_features, n_samples, out, s);
            finish_oblivious_block(out, n_samples, f.dim, tr, sigmoid, raw_score);
            return;
        }
        std::exception_ptr failure;
        #pragma omp parallel num_threads(n_threads)
        {
            Scratch& s = thread_scratch();
            #pragma omp for schedule(dynamic)
            for (int64_t b = 0; b < static_cast<int64_t>(n_blocks); b++) {
                try {
                    const size_t i = static_cast<size_t>(b) * block;
                    const size_t count = std::min(block, n_samples - i);
                    check_categories(f, X + i * n_features, count, n_features);
                    run_block(f, X + i * n_features, n_features, count, out + i * f.dim, s);
                    finish_oblivious_block(out + i * f.dim, count, f.dim, tr, sigmoid, raw_score);
                } catch (...) {
                    #pragma omp critical
                    if (!failure) failure = std::current_exception();
                }
            }
        }
        if (failure) std::rethrow_exception(failure);
    }

} // namespace lignum
