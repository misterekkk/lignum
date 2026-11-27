#include <lignum/model.hpp>

#include "binned.hpp"
#include "columns.hpp"
#include "compiled.hpp"
#include "catboost.hpp"
#include "ir.hpp"

#include <algorithm>
#include <bit>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef LIGNUM_WALK8
    #define LIGNUM_WALK8 1
#endif

#ifdef _MSC_VER
    using omp_iterator_t = ptrdiff_t;
#else
    using omp_iterator_t = size_t;
#endif

#if defined(__APPLE__) && defined(__ARM_NEON__)
    #define N_BLOCK 256
    #define T_BLOCK 32
#elif defined(__x86_64__) || defined(_M_X64)
    #define N_BLOCK 128
    #define T_BLOCK 16
#else
    #define N_BLOCK 64
    #define T_BLOCK 16
#endif

#if defined(_MSC_VER)
    #define LIGNUM_INLINE __forceinline
    #define LIGNUM_LAMBDA_INLINE
#else
    #define LIGNUM_INLINE inline __attribute__((always_inline))
    #define LIGNUM_LAMBDA_INLINE __attribute__((always_inline))
#endif

namespace lignum {

    namespace {

        template <bool ZeroMissing>
        inline int32_t cmp_dir(double x, double threshold, uint8_t dirs) {
            if constexpr (ZeroMissing) {
                if ((dirs & dir_bits::ZeroIsMissing) && std::fabs(x) <= kZeroThreshold) {
                    return dirs & dir_bits::DefaultRight;
                }
            }
            return std::isnan(x) ? (dirs & dir_bits::DefaultRight) : (x >= threshold);
        }

        // Transforms mirror the source library's arithmetic so outputs match exactly.
        inline double sigmoid_miter(double z) {
            if (z >= 0.0) {
                const double ez = std::exp(-z);
                return 1.0 / (1.0 + ez);
            }
            const double ez = std::exp(z);
            return ez / (1.0 + ez);
        }

        inline double apply_transform(double x, uint8_t transform, double sigmoid) {
            switch (transform) {
                case transform::Sigmoid:  return 1.0 / (1.0 + std::exp(-sigmoid * x));
                case transform::Exp:      return std::exp(x);
                case transform::Log1pExp: return std::log1p(std::exp(x));
                // LightGBM: Common::Sign(x) * x * x, an int sign times x, then times x (NaN gives NaN).
                case transform::SignedSquare: return static_cast<double>((x > 0.0) - (x < 0.0)) * x * x;
                default:                  return x;
            }
        }

        inline float apply_transform(float x, uint8_t transform) {
            switch (transform) {
                case transform::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
                case transform::Exp:     return std::exp(x);
                case transform::Hinge:   return x > 0.0f ? 1.0f : 0.0f;  // XGBoost: NaN gives 0
                default:                 return x;
            }
        }

        int resolve_threads(int n_jobs) {
            #ifdef _OPENMP
                const int max_threads = omp_get_max_threads();
                return (n_jobs <= 0 || n_jobs > max_threads) ? max_threads : n_jobs;
            #else
                (void)n_jobs;
                return 1;
            #endif
        }

        size_t round_up(size_t x, size_t multiple) {
            return (x + multiple - 1) / multiple * multiple;
        }

        // Categorical slot. NaN follows the default direction. Otherwise (see slot_kind):
        //  - LightGBM: valid iff -1 < x < 2^31; int(x) in the set goes left, everything else right.
        //  - XGBoost:  x is first rounded to float32; valid iff 0 <= x < 2^24; int(x) in the set
        //              goes right, everything else left.
        //
        // Branch-free on purpose: NaN, the valid range and the category value depend on the data,
        // so branches on them mispredict and stall the interleaved rows (measured: ~10% slower).
        // Word 0 of cat_bitsets is always zero, so non-bitset slots read a harmless address.
        LIGNUM_INLINE int32_t cat_dir(double x, double encoded, uint8_t kind, uint8_t dirs, const uint32_t* LIGNUM_RESTRICT bitsets) {
            const bool xgb = (dirs & dir_bits::XGBoostCategory) != 0;
            const double v = xgb ? static_cast<double>(static_cast<float>(x)) : x;
            const bool valid = xgb ? (v >= 0.0 && v < 16777216.0) : (v > -1.0 && v < 2147483648.0);  // false for NaN, +-inf
            const uint32_t c = static_cast<uint32_t>(valid ? v : 0.0);
            const uint64_t payload = std::bit_cast<uint64_t>(encoded);
            const uint32_t lo = static_cast<uint32_t>(payload);
            const uint32_t hi = static_cast<uint32_t>(payload >> 32);

            const bool in_mask = (c < 64) & static_cast<bool>((payload >> (c & 63)) & 1);
            const bool in_range = (c - lo) <= hi;
            const uint32_t word = c >> 5;
            const bool word_ok = (kind == slot_kind::CatBitset) & (word < hi);
            const bool in_bitset = static_cast<bool>((bitsets[word_ok ? lo + word : 0] >> (c & 31)) & 1);

            const bool in_set = kind == slot_kind::CatMask ? in_mask : (kind == slot_kind::CatRange ? in_range : in_bitset);
            const bool goes_right = xgb ? (valid & in_set) : !(valid & in_set);
            const int32_t nan_dir = dirs & dir_bits::DefaultRight;
            return std::isnan(x) ? nan_dir : static_cast<int32_t>(goes_right);
        }

        template <uint8_t Kind>
        LIGNUM_INLINE int32_t slot_dir(const KNode& kn, int k, const double* LIGNUM_RESTRICT row, const uint32_t* LIGNUM_RESTRICT bitsets) {
            const double x = row[kn.features[k]];
            if constexpr ((Kind & tree_kind::Categorical) != 0) {
                if (kn.kinds[k] != slot_kind::Numeric) return cat_dir(x, kn.thresholds[k], kn.kinds[k], kn.dirs[k], bitsets);
            }
            return cmp_dir<(Kind & tree_kind::ZeroMissing) != 0>(x, kn.thresholds[k], kn.dirs[k]);
        }

        // miter Q16 encoder (affine or quantile knots): matches miter::Q16Quantizer::encode_u16 bit-for-bit.
        LIGNUM_INLINE uint16_t q16_code_affine(double x, const FeatureQuantizer& q, bool constant) {
            if (std::isnan(x)) return kNanBin;
            if (constant) return 0;
            const double v = (x - q.lo) * q.scale;
            if (v <= 0.0) return 0;
            if (v >= 65534.0) return 65534;
            const double rounded = std::floor(v + 0.5);
            if (rounded <= 0.0) return 0;
            if (rounded >= 65534.0) return 65534;
            return static_cast<uint16_t>(rounded);
        }

        LIGNUM_INLINE uint16_t q16_code_quantile(double x, const double* LIGNUM_RESTRICT knots, uint32_t n_knots) {
            if (x <= knots[0]) return 0;
            if (x >= knots[n_knots - 1]) return 65534;
            const double* it = std::upper_bound(knots, knots + n_knots, x);
            const size_t idx = static_cast<size_t>(it - knots) - 1;
            const double denom = knots[idx + 1] - knots[idx];
            const double raw = denom > 0.0 ? (x - knots[idx]) / denom : 0.0;
            const double frac = raw < 0.0 ? 0.0 : (raw > 1.0 ? 1.0 : raw);
            const double m_inv = 65534.0 / static_cast<double>(n_knots - 1);
            const double pos = (static_cast<double>(idx) + frac) * m_inv;
            const double rounded = std::floor(pos + 0.5);
            if (rounded <= 0.0) return 0;
            if (rounded >= 65534.0) return 65534;
            return static_cast<uint16_t>(rounded);
        }

        LIGNUM_INLINE uint16_t q16_code(const FeatureQuantizer* LIGNUM_RESTRICT quant,
                                        const uint8_t* LIGNUM_RESTRICT quant_constant,
                                        const double* LIGNUM_RESTRICT knots,
                                        const uint32_t* LIGNUM_RESTRICT knot_offsets,
                                        uint32_t col, double x) {
            if (std::isnan(x)) return kNanBin;
            if (quant_constant[col]) return 0;
            if (knot_offsets) {
                const uint32_t lo = knot_offsets[col];
                const uint32_t hi = knot_offsets[col + 1];
                if (hi > lo) return q16_code_quantile(x, knots + lo, hi - lo);
            }
            return q16_code_affine(x, quant[col], false);
        }

        LIGNUM_INLINE uint32_t read_cat_hash(double x) {
            constexpr uint32_t kMissingCatHash = 0x9e3779b9U;
            if (std::isnan(x) || x < 0.0 || x >= 4294967296.0) return kMissingCatHash;
            return static_cast<uint32_t>(x);
        }

        LIGNUM_INLINE float lookup_ctr(const CtrTableHeader* LIGNUM_RESTRICT tables,
                                       const uint64_t* LIGNUM_RESTRICT keys,
                                       const uint32_t* LIGNUM_RESTRICT slots,
                                       const float* LIGNUM_RESTRICT values,
                                       uint32_t table_idx, const double* LIGNUM_RESTRICT row) {
            constexpr uint64_t kCtrHashMagic = 0x4906ba494954cb65ULL;
            const CtrTableHeader& t = tables[table_idx];
            if (t.n_slots == 0) return t.default_value;
            uint64_t h = static_cast<uint64_t>(read_cat_hash(row[t.columns[0]]));
            for (uint8_t c = 1; c < t.n_cols; c++) {
                h = kCtrHashMagic * (h + kCtrHashMagic * static_cast<uint64_t>(read_cat_hash(row[t.columns[c]])));
            }
            const uint64_t* LIGNUM_RESTRICT tkeys = keys + t.keys_offset;
            const uint32_t* LIGNUM_RESTRICT tslots = slots + t.slots_offset;
            const float* LIGNUM_RESTRICT tvals = values + t.values_offset;
            uint64_t idx = (h ^ (h >> 29U)) & t.mask;
            for (;;) {
                const uint32_t entry = tslots[idx];
                if (entry == 0U) return t.default_value;
                if (tkeys[idx] == h) return tvals[entry - 1U];
                idx = (idx + 1U) & t.mask;
            }
        }

        // Where the branchy traversal is used (measured with bench-lleaves.yml, 300k rows, 500 trees):
        //   LIGNUM_BRANCHY_TAIL (default 1): leftover rows (fewer than four, e.g. a single row);
        //     faster everywhere (one row: -16..-18% on x86-64/aarch64 Linux, -16% on M4).
        //   LIGNUM_BRANCHY_CAT (default 1, 0 on Apple): 4-row walks in trees with categorical
        //     splits; -17% on x86-64, -14% on aarch64 Linux, but ~+20% on M4.
        // Override with -DLIGNUM_BRANCHY_TAIL=0/1 and -DLIGNUM_BRANCHY_CAT=0/1.
        #ifndef LIGNUM_BRANCHY_TAIL
            #define LIGNUM_BRANCHY_TAIL 1
        #endif
        #ifndef LIGNUM_BRANCHY_CAT
            #if defined(__APPLE__)
                #define LIGNUM_BRANCHY_CAT 0
            #else
                #define LIGNUM_BRANCHY_CAT 1
            #endif
        #endif
        constexpr bool kBranchyTail = LIGNUM_BRANCHY_TAIL != 0;
        constexpr bool kBranchyCat = LIGNUM_BRANCHY_CAT != 0;

        template <uint8_t Kind>
        constexpr bool branchy_rows = kBranchyCat && (Kind & tree_kind::Categorical) != 0;
        template <uint8_t Kind>
        constexpr bool branchy_tail = kBranchyTail || branchy_rows<Kind>;

        // One KNode = two tree levels. `Kind` is the tree_kind of the tree, so numeric trees
        // compile to plain threshold comparisons.
        //  - Branch-free (default): all three comparisons are evaluated, then the grandchild is
        //    picked without a dependency between the two levels.
        //  - Branchy: branch on the root comparison like compiled code, so the CPU speculates down
        //    the predicted path and only the taken child's comparison is evaluated. Cheaper when a
        //    comparison is expensive (categorical) or there is nothing to interleave (one row).
        template <uint8_t Kind, bool Branchy>
        LIGNUM_INLINE int32_t step(const KNode& kn, const double* LIGNUM_RESTRICT row, const uint32_t* LIGNUM_RESTRICT bitsets) {
            if constexpr (Branchy) {
                if (slot_dir<Kind>(kn, 0, row, bitsets)) {
                    return kn.children[2 | slot_dir<Kind>(kn, 2, row, bitsets)];
                }
                return kn.children[slot_dir<Kind>(kn, 1, row, bitsets)];
            } else {
                int32_t r = slot_dir<Kind>(kn, 0, row, bitsets);
                int32_t l = slot_dir<Kind>(kn, 1, row, bitsets);
                int32_t c = slot_dir<Kind>(kn, 2, row, bitsets);
                return kn.children[(r << 1) | (r ? c : l)];
            }
        }

        struct Forest {
            const KNode* LIGNUM_RESTRICT nodes;
            const uint32_t* LIGNUM_RESTRICT bitsets;
            const ObliqueSplit* LIGNUM_RESTRICT oblique;     // miter models only
            const FeatureQuantizer* LIGNUM_RESTRICT quant;
            const uint8_t* LIGNUM_RESTRICT quant_constant;
            const double* LIGNUM_RESTRICT quant_knots;
            const uint32_t* LIGNUM_RESTRICT quant_knot_offsets;
            const CtrTableHeader* LIGNUM_RESTRICT ctr_tables;
            const uint64_t* LIGNUM_RESTRICT ctr_keys;
            const uint32_t* LIGNUM_RESTRICT ctr_slots;
            const float* LIGNUM_RESTRICT ctr_values;
        };

        // Oblique slot (see ObliqueSplit): any NaN input -> default direction, else P >= threshold.
        LIGNUM_INLINE int32_t oblique_dir(const KNode& kn, int k, const double* LIGNUM_RESTRICT row, Forest f) {
            const ObliqueSplit& o = f.oblique[std::bit_cast<uint64_t>(kn.thresholds[k])];
            const double x0 = row[o.features[0]], x1 = row[o.features[1]], x2 = row[o.features[2]];
            if (std::isnan(x0) || std::isnan(x1) || std::isnan(x2)) return kn.dirs[k] & dir_bits::DefaultRight;
            const int32_t c0 = static_cast<int32_t>(q16_code(f.quant, f.quant_constant, f.quant_knots, f.quant_knot_offsets, o.features[0], x0)) - 32768;
            const int32_t c1 = static_cast<int32_t>(q16_code(f.quant, f.quant_constant, f.quant_knots, f.quant_knot_offsets, o.features[1], x1)) - 32768;
            const int32_t c2 = static_cast<int32_t>(q16_code(f.quant, f.quant_constant, f.quant_knots, f.quant_knot_offsets, o.features[2], x2)) - 32768;
            const int32_t p = static_cast<int32_t>(o.weights[0]) * c0 + static_cast<int32_t>(o.weights[1]) * c1 +
                              static_cast<int32_t>(o.weights[2]) * c2;
            return static_cast<int32_t>(p >= o.threshold);
        }

        // Generic extended slot (used only for miter models with oblique splits or CTR tables).
        LIGNUM_INLINE int32_t extended_slot_dir(const KNode& kn, int k, const double* LIGNUM_RESTRICT row, Forest f) {
            if (kn.kinds[k] == slot_kind::Oblique) return oblique_dir(kn, k, row, f);
            if (kn.kinds[k] == slot_kind::Ctr) {
                const float val = lookup_ctr(f.ctr_tables, f.ctr_keys, f.ctr_slots, f.ctr_values, kn.features[k], row);
                return static_cast<int32_t>(static_cast<double>(val) >= kn.thresholds[k]);
            }
            return slot_dir<tree_kind::ZeroMissing | tree_kind::Categorical>(kn, k, row, f.bitsets);
        }

        template <bool Branchy>
        LIGNUM_INLINE int32_t extended_step(const KNode& kn, const double* LIGNUM_RESTRICT row, Forest f) {
            if constexpr (Branchy) {
                if (extended_slot_dir(kn, 0, row, f)) return kn.children[2 | extended_slot_dir(kn, 2, row, f)];
                return kn.children[extended_slot_dir(kn, 1, row, f)];
            } else {
                int32_t r = extended_slot_dir(kn, 0, row, f);
                int32_t l = extended_slot_dir(kn, 1, row, f);
                int32_t c = extended_slot_dir(kn, 2, row, f);
                return kn.children[(r << 1) | (r ? c : l)];
            }
        }

        template <uint8_t Kind, bool Branchy>
        LIGNUM_INLINE int32_t step_any(const KNode& kn, const double* LIGNUM_RESTRICT row, Forest f) {
            if constexpr ((Kind & (tree_kind::Oblique | tree_kind::Ctr)) != 0) {
                return extended_step<Branchy>(kn, row, f);
            } else {
                return step<Kind, Branchy>(kn, row, f.bitsets);
            }
        }

        // Eight rows walk the same tree interleaved to hide memory latency on numeric trees.
        template <uint8_t Kind>
        LIGNUM_INLINE void walk8(Forest f, int32_t root,
                                 const double* r0, const double* r1, const double* r2, const double* r3,
                                 const double* r4, const double* r5, const double* r6, const double* r7,
                                 int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3,
                                 int32_t& n4, int32_t& n5, int32_t& n6, int32_t& n7) {
            constexpr bool B = branchy_rows<Kind>;
            if (root < 0) { n0 = n1 = n2 = n3 = n4 = n5 = n6 = n7 = root; return; }
            const KNode& rn = f.nodes[root];
            n0 = step_any<Kind, B>(rn, r0, f);
            n1 = step_any<Kind, B>(rn, r1, f);
            n2 = step_any<Kind, B>(rn, r2, f);
            n3 = step_any<Kind, B>(rn, r3, f);
            n4 = step_any<Kind, B>(rn, r4, f);
            n5 = step_any<Kind, B>(rn, r5, f);
            n6 = step_any<Kind, B>(rn, r6, f);
            n7 = step_any<Kind, B>(rn, r7, f);
            while ((n0 & n1 & n2 & n3 & n4 & n5 & n6 & n7) >= 0) {
                if (n0 >= 0) n0 = step_any<Kind, B>(f.nodes[n0], r0, f);
                if (n1 >= 0) n1 = step_any<Kind, B>(f.nodes[n1], r1, f);
                if (n2 >= 0) n2 = step_any<Kind, B>(f.nodes[n2], r2, f);
                if (n3 >= 0) n3 = step_any<Kind, B>(f.nodes[n3], r3, f);
                if (n4 >= 0) n4 = step_any<Kind, B>(f.nodes[n4], r4, f);
                if (n5 >= 0) n5 = step_any<Kind, B>(f.nodes[n5], r5, f);
                if (n6 >= 0) n6 = step_any<Kind, B>(f.nodes[n6], r6, f);
                if (n7 >= 0) n7 = step_any<Kind, B>(f.nodes[n7], r7, f);
            }
        }

        // Four rows walk the same tree interleaved to hide memory latency.
        template <uint8_t Kind>
        LIGNUM_INLINE void walk4(Forest f, int32_t root,
                                 const double* r0, const double* r1, const double* r2, const double* r3,
                                 int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) {
            constexpr bool B = branchy_rows<Kind>;
            if (root < 0) { n0 = n1 = n2 = n3 = root; return; }
            const KNode& rn = f.nodes[root];
            n0 = step_any<Kind, B>(rn, r0, f);
            n1 = step_any<Kind, B>(rn, r1, f);
            n2 = step_any<Kind, B>(rn, r2, f);
            n3 = step_any<Kind, B>(rn, r3, f);
            while ((n0 & n1 & n2 & n3) >= 0) {
                if (n0 >= 0) n0 = step_any<Kind, B>(f.nodes[n0], r0, f);
                if (n1 >= 0) n1 = step_any<Kind, B>(f.nodes[n1], r1, f);
                if (n2 >= 0) n2 = step_any<Kind, B>(f.nodes[n2], r2, f);
                if (n3 >= 0) n3 = step_any<Kind, B>(f.nodes[n3], r3, f);
            }
        }

        // Four trees of the same row, interleaved (used when there are fewer than four rows).
        template <uint8_t Kind>
        LIGNUM_INLINE void walk4_trees(Forest f, const int32_t* roots, const double* row,
                                       int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) {
            constexpr bool B = branchy_tail<Kind>;
            n0 = roots[0]; n1 = roots[1]; n2 = roots[2]; n3 = roots[3];
            while ((n0 & n1 & n2 & n3) >= 0) {
                if (n0 >= 0) n0 = step_any<Kind, B>(f.nodes[n0], row, f);
                if (n1 >= 0) n1 = step_any<Kind, B>(f.nodes[n1], row, f);
                if (n2 >= 0) n2 = step_any<Kind, B>(f.nodes[n2], row, f);
                if (n3 >= 0) n3 = step_any<Kind, B>(f.nodes[n3], row, f);
            }
        }

        template <uint8_t Kind>
        LIGNUM_INLINE int32_t walk1(Forest f, int32_t node, const double* row) {
            constexpr bool B = branchy_tail<Kind>;
            while (node >= 0) node = step_any<Kind, B>(f.nodes[node], row, f);
            return node;
        }

        // Calls fn(std::integral_constant<uint8_t, kind>) so the traversal is specialised per tree
        // kind. Models with only numeric trees (Mixed = false) never branch. Extended = true only
        // for models with oblique or CTR splits (miter): kept in a separate instantiation so other
        // models compile identically.
        template <bool Mixed, bool Extended = false, typename Fn>
        LIGNUM_INLINE void with_kind(uint8_t kind, Fn&& fn) {
            using K0 = std::integral_constant<uint8_t, 0>;
            if constexpr (!Mixed) {
                fn(K0{});
            } else if constexpr (Extended) {
                if ((kind & (tree_kind::Oblique | tree_kind::Ctr)) != 0) {
                    fn(std::integral_constant<uint8_t, tree_kind::All>{});
                } else {
                    with_kind<true, false>(kind, fn);
                }
            } else {
                switch (kind & 3) {
                    case 0:  fn(K0{}); break;
                    case 1:  fn(std::integral_constant<uint8_t, 1>{}); break;
                    case 2:  fn(std::integral_constant<uint8_t, 2>{}); break;
                    default: fn(std::integral_constant<uint8_t, 3>{}); break;
                }
            }
        }

        LIGNUM_INLINE double leaf_value(const double* LIGNUM_RESTRICT leaves, int32_t node) {
            return leaves[(-node) - 1];
        }

        // Tree traversal shared by all strategies. Leaves are always added to a row's running
        // sum (kept in out_preds between tiles) in model order, exactly like the source library,
        // so every strategy produces bit-identical results.
        //
        // A tree-parallel strategy (threads fill a per-tree leaf buffer, rows sum it in order) was
        // measured and lost to Serial below 8 rows and to Rows above, so it is not offered.
        // LightGBM: everything in double. XGBoost: max and exp in float, the sum in double, the
        // division by float(sum) in float (verified bit-exact).
        template <typename Acc>
        void softmax(double* v, size_t K) {
            if constexpr (std::is_same_v<Acc, float>) {
                float wmax = static_cast<float>(v[0]);
                for (size_t c = 1; c < K; c++) wmax = std::max(wmax, static_cast<float>(v[c]));
                double wsum = 0.0;
                for (size_t c = 0; c < K; c++) {
                    const float e = std::exp(static_cast<float>(v[c]) - wmax);
                    v[c] = static_cast<double>(e);
                    wsum += e;
                }
                const float fsum = static_cast<float>(wsum);
                for (size_t c = 0; c < K; c++) v[c] = static_cast<double>(static_cast<float>(v[c]) / fsum);
            } else {
                double wmax = v[0];
                for (size_t c = 1; c < K; c++) wmax = std::max(wmax, v[c]);
                double wsum = 0.0;
                for (size_t c = 0; c < K; c++) {
                    v[c] = std::exp(v[c] - wmax);
                    wsum += v[c];
                }
                for (size_t c = 0; c < K; c++) v[c] /= wsum;
            }
        }

        template <typename Acc>
        void softmax_miter(double* v, size_t K) {
            double wmax = v[0];
            for (size_t c = 1; c < K; c++) wmax = std::max(wmax, v[c]);
            double wsum = 0.0;
            for (size_t c = 0; c < K; c++) {
                v[c] = std::exp(v[c] - wmax);
                wsum += v[c];
            }
            const double inv_sum = 1.0 / wsum;
            for (size_t c = 0; c < K; c++) v[c] *= inv_sum;
        }

        // Output transform of rows [i_begin, i_end); rows hold K = n_outputs values when Multi.
        template <typename Acc, bool Multi>
        void finish_rows(const Model& m, bool raw_score, size_t i_begin, size_t i_end, double* LIGNUM_RESTRICT out) {
            const uint8_t tr = m.transform;
            if (raw_score || tr == transform::Identity || tr == transform::Argmax) return;

            const size_t K = Multi ? m.n_outputs : 1;
            if (tr == transform::Softmax) {
                if constexpr (std::is_same_v<Acc, double>) {
                    if (m.source == model_source::Miter) {
                        for (size_t i = i_begin; i < i_end; i++) softmax_miter<Acc>(&out[i * K], K);
                        return;
                    }
                }
                for (size_t i = i_begin; i < i_end; i++) softmax<Acc>(&out[i * K], K);
                return;
            }
            if constexpr (std::is_same_v<Acc, double>) {
                if (m.source == model_source::Miter && tr == transform::Sigmoid && m.sigmoid == 1.0) {
                    for (size_t j = i_begin * K; j < i_end * K; j++) out[j] = sigmoid_miter(out[j]);
                    return;
                }
            }
            for (size_t j = i_begin * K; j < i_end * K; j++) {
                if constexpr (std::is_same_v<Acc, float>) {
                    out[j] = static_cast<double>(apply_transform(static_cast<float>(out[j]), tr));
                } else {
                    out[j] = apply_transform(out[j], tr, m.sigmoid);
                }
            }
        }

        // `Mixed`: the model has trees that are not plain numeric (zero-as-missing or categorical).
        // Only then does the kernel branch per tree; numeric trees always take the plain traversal.
        // `Multi`: the model has several outputs (classes). Row i's outputs live in
        // out[i * K .. i * K + K), and each tree adds to its own output, still in model order.
        template <typename Acc, bool Mixed, bool Multi, bool Extended = false>
        class Kernel {
            public:
                Kernel(const Model& model, const double* X, size_t n_features, bool raw_score)
                    : m_(model), X_(X), n_features_(n_features), raw_score_(raw_score) {}

                // Rows [i_begin, i_end): base scores, all trees in tiles of T_BLOCK, transform.
                void predict_rows(size_t i_begin, size_t i_end, double* LIGNUM_RESTRICT out) const {
                    const size_t count = i_end - i_begin;
                    if constexpr (Multi) {
                        const size_t K = m_.n_outputs;
                        const bool argmax = !raw_score_ && m_.transform == transform::Argmax;
                        thread_local std::vector<double> block_margins;
                        double* LIGNUM_RESTRICT dst = nullptr;
                        if (argmax) {
                            block_margins.resize(count * K);
                            dst = block_margins.data();
                        } else {
                            dst = &out[i_begin * K];
                        }
                        for (size_t r = 0; r < count; r++) {
                            for (size_t c = 0; c < K; c++) dst[r * K + c] = static_cast<double>(static_cast<Acc>(m_.base_scores[c]));
                        }
                        for (size_t t_start = 0; t_start < m_.n_offsets; t_start += T_BLOCK) {
                            const size_t t_end = std::min(t_start + T_BLOCK, m_.n_offsets);
                            accumulate_multi(i_begin, i_end, t_start, t_end, dst);
                        }
                        if (argmax) {
                            for (size_t r = 0; r < count; r++) {
                                const double* row_m = &dst[r * K];
                                out[i_begin + r] = static_cast<double>(std::max_element(row_m, row_m + K) - row_m);
                            }
                        } else {
                            finish(i_begin, i_end, out);
                        }
                    } else {
                        const double base = static_cast<double>(static_cast<Acc>(m_.base_scores[0]));
                        for (size_t i = i_begin; i < i_end; i++) out[i] = base;
                        for (size_t t_start = 0; t_start < m_.n_offsets; t_start += T_BLOCK) {
                            const size_t t_end = std::min(t_start + T_BLOCK, m_.n_offsets);
                            accumulate(i_begin, i_end, t_start, t_end, out);
                        }
                        if (!raw_score_ && m_.transform == transform::Argmax) {
                            for (size_t i = i_begin; i < i_end; i++) out[i] = 0.0;
                        } else {
                            finish(i_begin, i_end, out);
                        }
                    }
                }

            private:
                const Model& m_;
                const double* X_;
                size_t n_features_;
                bool raw_score_;

                // Union of the kinds of trees [t, t + count): their traversal must handle all of them.
                static LIGNUM_INLINE uint8_t kind_of(const uint8_t* kinds, size_t t, size_t count) {
                    if constexpr (!Mixed) {
                        return 0;
                    } else {
                        uint8_t any = 0;
                        for (size_t k = 0; k < count; k++) any |= kinds[t + k];
                        return any;
                    }
                }

                Forest forest() const {
                    return Forest{m_.k_nodes, m_.cat_bitsets, m_.oblique_splits, m_.quantizers,
                                  m_.quantizer_constant, m_.quantizer_knots, m_.quantizer_knot_offsets,
                                  m_.ctr_tables, m_.ctr_keys, m_.ctr_slots, m_.ctr_values};
                }

                const double* row(size_t i) const { return &X_[i * n_features_]; }

                // Adds trees [t_begin, t_end) to the running sums of rows [i_begin, i_end).
                void accumulate(size_t i_begin, size_t i_end, size_t t_begin, size_t t_end, double* LIGNUM_RESTRICT out) const {
                    const Forest f = forest();
                    const int32_t* LIGNUM_RESTRICT roots = m_.tree_offsets;
                    const uint8_t* LIGNUM_RESTRICT kinds = m_.tree_kinds;
                    const double* LIGNUM_RESTRICT leaf_values = m_.leaf_values;
                    size_t i_cur = i_begin;
                    if constexpr (!Mixed && LIGNUM_WALK8) {
                        const size_t unrolled8_end = i_begin + ((i_end - i_begin) / 8) * 8;
                        for (size_t i = i_begin; i < unrolled8_end; i += 8) {
                            const double *r0 = row(i), *r1 = row(i + 1), *r2 = row(i + 2), *r3 = row(i + 3);
                            const double *r4 = row(i + 4), *r5 = row(i + 5), *r6 = row(i + 6), *r7 = row(i + 7);
                            Acc sum0 = static_cast<Acc>(out[i]),     sum1 = static_cast<Acc>(out[i + 1]);
                            Acc sum2 = static_cast<Acc>(out[i + 2]), sum3 = static_cast<Acc>(out[i + 3]);
                            Acc sum4 = static_cast<Acc>(out[i + 4]), sum5 = static_cast<Acc>(out[i + 5]);
                            Acc sum6 = static_cast<Acc>(out[i + 6]), sum7 = static_cast<Acc>(out[i + 7]);
                            for (size_t t = t_begin; t < t_end; t++) {
                                int32_t n0, n1, n2, n3, n4, n5, n6, n7;
                                walk8<0>(f, roots[t], r0, r1, r2, r3, r4, r5, r6, r7, n0, n1, n2, n3, n4, n5, n6, n7);
                                sum0 += static_cast<Acc>(leaf_value(leaf_values, n0));
                                sum1 += static_cast<Acc>(leaf_value(leaf_values, n1));
                                sum2 += static_cast<Acc>(leaf_value(leaf_values, n2));
                                sum3 += static_cast<Acc>(leaf_value(leaf_values, n3));
                                sum4 += static_cast<Acc>(leaf_value(leaf_values, n4));
                                sum5 += static_cast<Acc>(leaf_value(leaf_values, n5));
                                sum6 += static_cast<Acc>(leaf_value(leaf_values, n6));
                                sum7 += static_cast<Acc>(leaf_value(leaf_values, n7));
                            }
                            out[i]     = static_cast<double>(sum0); out[i + 1] = static_cast<double>(sum1);
                            out[i + 2] = static_cast<double>(sum2); out[i + 3] = static_cast<double>(sum3);
                            out[i + 4] = static_cast<double>(sum4); out[i + 5] = static_cast<double>(sum5);
                            out[i + 6] = static_cast<double>(sum6); out[i + 7] = static_cast<double>(sum7);
                        }
                        i_cur = unrolled8_end;
                    }
                    const size_t unrolled_end = i_begin + ((i_end - i_begin) / 4) * 4;
                    for (size_t i = i_cur; i < unrolled_end; i += 4) {
                        const double *r0 = row(i), *r1 = row(i + 1), *r2 = row(i + 2), *r3 = row(i + 3);
                        Acc sum0 = static_cast<Acc>(out[i]);
                        Acc sum1 = static_cast<Acc>(out[i + 1]);
                        Acc sum2 = static_cast<Acc>(out[i + 2]);
                        Acc sum3 = static_cast<Acc>(out[i + 3]);
                        for (size_t t = t_begin; t < t_end; t++) {
                            int32_t n0, n1, n2, n3;
                            with_kind<Mixed, Extended>(kind_of(kinds, t, 1), [&](auto K) LIGNUM_LAMBDA_INLINE {
                                walk4<decltype(K)::value>(f, roots[t], r0, r1, r2, r3, n0, n1, n2, n3);
                            });
                            sum0 += static_cast<Acc>(leaf_value(leaf_values, n0));
                            sum1 += static_cast<Acc>(leaf_value(leaf_values, n1));
                            sum2 += static_cast<Acc>(leaf_value(leaf_values, n2));
                            sum3 += static_cast<Acc>(leaf_value(leaf_values, n3));
                        }
                        out[i] = static_cast<double>(sum0);
                        out[i + 1] = static_cast<double>(sum1);
                        out[i + 2] = static_cast<double>(sum2);
                        out[i + 3] = static_cast<double>(sum3);
                    }
                    // Leftover rows: interleave four trees of the same row instead. Leaves are
                    // still added in tree order, so the result does not change.
                    const size_t trees_unrolled_end = t_begin + ((t_end - t_begin) / 4) * 4;
                    for (size_t i = unrolled_end; i < i_end; i++) {
                        const double* r = row(i);
                        Acc sum = static_cast<Acc>(out[i]);
                        for (size_t t = t_begin; t < trees_unrolled_end; t += 4) {
                            int32_t n0, n1, n2, n3;
                            with_kind<Mixed, Extended>(kind_of(kinds, t, 4), [&](auto K) LIGNUM_LAMBDA_INLINE {
                                walk4_trees<decltype(K)::value>(f, &roots[t], r, n0, n1, n2, n3);
                            });
                            sum += static_cast<Acc>(leaf_value(leaf_values, n0));
                            sum += static_cast<Acc>(leaf_value(leaf_values, n1));
                            sum += static_cast<Acc>(leaf_value(leaf_values, n2));
                            sum += static_cast<Acc>(leaf_value(leaf_values, n3));
                        }
                        for (size_t t = trees_unrolled_end; t < t_end; t++) {
                            int32_t leaf;
                            with_kind<Mixed, Extended>(kind_of(kinds, t, 1), [&](auto K) LIGNUM_LAMBDA_INLINE { leaf = walk1<decltype(K)::value>(f, roots[t], r); });
                            sum += static_cast<Acc>(leaf_value(leaf_values, leaf));
                        }
                        out[i] = static_cast<double>(sum);
                    }
                }

                // Multi-output version of accumulate(): each leaf goes to its tree's output.
                // `dst` holds (i_end - i_begin) * K outputs starting at row i_begin.
                void accumulate_multi(size_t i_begin, size_t i_end, size_t t_begin, size_t t_end, double* LIGNUM_RESTRICT dst) const {
                    const Forest f = forest();
                    const int32_t* LIGNUM_RESTRICT roots = m_.tree_offsets;
                    const uint8_t* LIGNUM_RESTRICT kinds = m_.tree_kinds;
                    const uint16_t* LIGNUM_RESTRICT outputs = m_.tree_outputs;
                    const double* LIGNUM_RESTRICT leaf_values = m_.leaf_values;
                    const size_t K = m_.n_outputs;

                    auto add = [&](size_t i, size_t t, int32_t node) LIGNUM_LAMBDA_INLINE {
                        double& slot = dst[(i - i_begin) * K + outputs[t]];
                        slot = static_cast<double>(static_cast<Acc>(slot) + static_cast<Acc>(leaf_value(leaf_values, node)));
                    };

                    size_t i_cur = i_begin;
                    if constexpr (!Mixed && LIGNUM_WALK8) {
                        const size_t unrolled8_end = i_begin + ((i_end - i_begin) / 8) * 8;
                        for (size_t i = i_begin; i < unrolled8_end; i += 8) {
                            const double *r0 = row(i), *r1 = row(i + 1), *r2 = row(i + 2), *r3 = row(i + 3);
                            const double *r4 = row(i + 4), *r5 = row(i + 5), *r6 = row(i + 6), *r7 = row(i + 7);
                            for (size_t t = t_begin; t < t_end; t++) {
                                int32_t n0, n1, n2, n3, n4, n5, n6, n7;
                                walk8<0>(f, roots[t], r0, r1, r2, r3, r4, r5, r6, r7, n0, n1, n2, n3, n4, n5, n6, n7);
                                add(i, t, n0);     add(i + 1, t, n1); add(i + 2, t, n2); add(i + 3, t, n3);
                                add(i + 4, t, n4); add(i + 5, t, n5); add(i + 6, t, n6); add(i + 7, t, n7);
                            }
                        }
                        i_cur = unrolled8_end;
                    }
                    const size_t unrolled_end = i_begin + ((i_end - i_begin) / 4) * 4;
                    for (size_t i = i_cur; i < unrolled_end; i += 4) {
                        const double *r0 = row(i), *r1 = row(i + 1), *r2 = row(i + 2), *r3 = row(i + 3);
                        for (size_t t = t_begin; t < t_end; t++) {
                            int32_t n0, n1, n2, n3;
                            with_kind<Mixed, Extended>(kind_of(kinds, t, 1), [&](auto K) LIGNUM_LAMBDA_INLINE {
                                walk4<decltype(K)::value>(f, roots[t], r0, r1, r2, r3, n0, n1, n2, n3);
                            });
                            add(i, t, n0);
                            add(i + 1, t, n1);
                            add(i + 2, t, n2);
                            add(i + 3, t, n3);
                        }
                    }
                    const size_t trees_unrolled_end = t_begin + ((t_end - t_begin) / 4) * 4;
                    for (size_t i = unrolled_end; i < i_end; i++) {
                        const double* r = row(i);
                        for (size_t t = t_begin; t < trees_unrolled_end; t += 4) {
                            int32_t n0, n1, n2, n3;
                            with_kind<Mixed, Extended>(kind_of(kinds, t, 4), [&](auto K) LIGNUM_LAMBDA_INLINE {
                                walk4_trees<decltype(K)::value>(f, &roots[t], r, n0, n1, n2, n3);
                            });
                            add(i, t, n0);
                            add(i, t + 1, n1);
                            add(i, t + 2, n2);
                            add(i, t + 3, n3);
                        }
                        for (size_t t = trees_unrolled_end; t < t_end; t++) {
                            int32_t leaf;
                            with_kind<Mixed, Extended>(kind_of(kinds, t, 1), [&](auto K) LIGNUM_LAMBDA_INLINE { leaf = walk1<decltype(K)::value>(f, roots[t], r); });
                            add(i, t, leaf);
                        }
                    }
                }

                void finish(size_t i_begin, size_t i_end, double* LIGNUM_RESTRICT out) const {
                    finish_rows<Acc, Multi>(m_, raw_score_, i_begin, i_end, out);
                }
        };

        template <typename K>
        void run_serial(const K& kernel, size_t n_samples, double* out) {
            for (size_t i_start = 0; i_start < n_samples; i_start += N_BLOCK) {
                kernel.predict_rows(i_start, std::min<size_t>(i_start + N_BLOCK, n_samples), out);
            }
        }

        template <typename K>
        void run_rows(const K& kernel, size_t n_samples, double* out, int n_threads, size_t max_block = N_BLOCK) {
            // Enough blocks for every thread (two each, for balance across P/E cores), but never
            // larger than max_block, which keeps a tile of trees hot in cache for the whole block.
            const size_t per_thread = (n_samples + 2 * n_threads - 1) / (2 * n_threads);
            const size_t block = std::clamp<size_t>(round_up(per_thread, 4), 4, max_block);

            std::exception_ptr failure;
            #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
            for (omp_iterator_t i_start = 0; i_start < static_cast<omp_iterator_t>(n_samples); i_start += block) {
                try {
                    kernel.predict_rows(i_start, std::min<size_t>(i_start + block, n_samples), out);
                } catch (...) {
                    #pragma omp critical
                    if (!failure) failure = std::current_exception();
                }
            }
            if (failure) std::rethrow_exception(failure);
        }

        // Below ~8 rows waking the thread pool costs more than the work (measured on an M4:
        // 4 rows 27 us serial vs 36 us parallel; 8 rows 47 us vs 40 us).
        constexpr size_t kMinRowsForThreads = 8;

        // Smallest batch that uses the binned kernel. With four trees interleaved per leftover row
        // it wins from one row on (M4: numeric 6.9 -> 6.6 us, categorical 14.4 -> 8.3 us per row).
        #ifndef LIGNUM_BINNED_MIN_ROWS
            #define LIGNUM_BINNED_MIN_ROWS 1
        #endif
        constexpr size_t kMinRowsForBinned = LIGNUM_BINNED_MIN_ROWS;

        // Row chunk of the compiled mode, in bytes of per-row buffers (M4, single thread:
        // Covertype 545 ms with 256 rows -> 517 ms with 2 MB; 8 MB gains only 1% more).
        #ifndef LIGNUM_COMPILED_CHUNK_KB
            #define LIGNUM_COMPILED_CHUNK_KB 2048
        #endif
        constexpr size_t kCompiledChunkBytes = size_t{LIGNUM_COMPILED_CHUNK_KB} * 1024;

        Strategy choose_strategy(size_t n_samples, int n_threads) {
            if (n_threads <= 1 || n_samples < kMinRowsForThreads) return Strategy::Serial;
            return Strategy::Rows;
        }

        template <typename Acc, bool Mixed, bool Multi, bool Extended = false>
        void run(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                 int n_jobs, Strategy strategy, bool raw_score) {
            const Kernel<Acc, Mixed, Multi, Extended> kernel(model, X, n_features, raw_score);
            const int n_threads = resolve_threads(n_jobs);
            if (strategy == Strategy::Auto) strategy = choose_strategy(n_samples, n_threads);

            switch (strategy) {
                case Strategy::Serial: run_serial(kernel, n_samples, out); break;
                default:               run_rows(kernel, n_samples, out, n_threads); break;
            }
        }

        template <typename Acc, bool Mixed, bool Extended = false>
        void run_outputs(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                         int n_jobs, Strategy strategy, bool raw_score) {
            if (model.n_outputs > 1) run<Acc, Mixed, true, Extended>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
            else                     run<Acc, Mixed, false, Extended>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
        }

        void run_any(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                     int n_jobs, Strategy strategy, bool raw_score) {
            if (model.n_oblique > 0 || model.n_ctr_tables > 0) {
                run_outputs<double, true, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
            } else if (model.float32) {
                if (model.has_special_trees) run_outputs<float, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
                else                        run_outputs<float, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
            } else {
                if (model.has_special_trees) run_outputs<double, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
                else                        run_outputs<double, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
            }
        }

        // ---- Binned kernel (see src/binned.hpp) ----------------------------------------------

        // Number of cuts <= x (branch-free binary search, linear count for small len); NaN -> max(CodeT).
        template <typename CodeT = uint16_t>
        LIGNUM_INLINE CodeT bin_of(const double* LIGNUM_RESTRICT cuts, uint32_t len, double x) {
            constexpr CodeT kNan = std::numeric_limits<CodeT>::max();
            if (std::isnan(x)) return kNan;
            if (len <= 8) {
                uint32_t b = 0;
                for (uint32_t k = 0; k < len; k++) b += (cuts[k] <= x);
                return static_cast<CodeT>(b);
            }
            const double* base = cuts;
            uint32_t n = len;
            while (n > 1) {
                const uint32_t half = n / 2;
                base = (base[half] <= x) ? base + half : base;
                n -= half;
            }
            return static_cast<CodeT>((base - cuts) + (n == 1 && *base <= x));
        }

        // Fixed-step branch-free power-of-2 binary search (Khuong & Morin reduction):
        // p = 1 << LogP <= len < 2*p; zero loop branches, constant offsets, and branch-free NaN.
        template <int LogP, typename CodeT = uint16_t>
        LIGNUM_INLINE CodeT bin_pow2(const double* LIGNUM_RESTRICT cuts, uint32_t rem, double x) {
            constexpr CodeT kNan = std::numeric_limits<CodeT>::max();
            const double* base = (cuts[rem] <= x) ? cuts + rem : cuts;
            if constexpr (LogP >= 15) base = (base[16384] <= x) ? base + 16384 : base;
            if constexpr (LogP >= 14) base = (base[8192] <= x) ? base + 8192 : base;
            if constexpr (LogP >= 13) base = (base[4096] <= x) ? base + 4096 : base;
            if constexpr (LogP >= 12) base = (base[2048] <= x) ? base + 2048 : base;
            if constexpr (LogP >= 11) base = (base[1024] <= x) ? base + 1024 : base;
            if constexpr (LogP >= 10) base = (base[512] <= x) ? base + 512 : base;
            if constexpr (LogP >= 9)  base = (base[256] <= x) ? base + 256 : base;
            if constexpr (LogP >= 8)  base = (base[128] <= x) ? base + 128 : base;
            if constexpr (LogP >= 7)  base = (base[64] <= x) ? base + 64 : base;
            if constexpr (LogP >= 6)  base = (base[32] <= x) ? base + 32 : base;
            if constexpr (LogP >= 5)  base = (base[16] <= x) ? base + 16 : base;
            if constexpr (LogP >= 4)  base = (base[8] <= x) ? base + 8 : base;
            if constexpr (LogP >= 3)  base = (base[4] <= x) ? base + 4 : base;
            if constexpr (LogP >= 2)  base = (base[2] <= x) ? base + 2 : base;
            if constexpr (LogP >= 1)  base = (base[1] <= x) ? base + 1 : base;
            const CodeT bin = static_cast<CodeT>((base - cuts) + (*base <= x));
            return std::isnan(x) ? kNan : bin;
        }

        // Four rows binned in parallel with interleaved fixed-step binary search to hide L1D latency.
        template <int LogP, typename CodeT = uint16_t>
        LIGNUM_INLINE void bin_pow2_4(const double* LIGNUM_RESTRICT cuts, uint32_t rem,
                                      const double* LIGNUM_RESTRICT src, size_t src_stride,
                                      CodeT* LIGNUM_RESTRICT dst, size_t dst_stride) {
            constexpr CodeT kNan = std::numeric_limits<CodeT>::max();
            const double x0 = src[0];
            const double x1 = src[src_stride];
            const double x2 = src[2 * src_stride];
            const double x3 = src[3 * src_stride];
            const double* b0 = (cuts[rem] <= x0) ? cuts + rem : cuts;
            const double* b1 = (cuts[rem] <= x1) ? cuts + rem : cuts;
            const double* b2 = (cuts[rem] <= x2) ? cuts + rem : cuts;
            const double* b3 = (cuts[rem] <= x3) ? cuts + rem : cuts;
            auto step_bin = [&](uint32_t s) LIGNUM_LAMBDA_INLINE {
                b0 = (b0[s] <= x0) ? b0 + s : b0;
                b1 = (b1[s] <= x1) ? b1 + s : b1;
                b2 = (b2[s] <= x2) ? b2 + s : b2;
                b3 = (b3[s] <= x3) ? b3 + s : b3;
            };
            if constexpr (LogP >= 15) step_bin(16384);
            if constexpr (LogP >= 14) step_bin(8192);
            if constexpr (LogP >= 13) step_bin(4096);
            if constexpr (LogP >= 12) step_bin(2048);
            if constexpr (LogP >= 11) step_bin(1024);
            if constexpr (LogP >= 10) step_bin(512);
            if constexpr (LogP >= 9)  step_bin(256);
            if constexpr (LogP >= 8)  step_bin(128);
            if constexpr (LogP >= 7)  step_bin(64);
            if constexpr (LogP >= 6)  step_bin(32);
            if constexpr (LogP >= 5)  step_bin(16);
            if constexpr (LogP >= 4)  step_bin(8);
            if constexpr (LogP >= 3)  step_bin(4);
            if constexpr (LogP >= 2)  step_bin(2);
            if constexpr (LogP >= 1)  step_bin(1);
            dst[0]              = std::isnan(x0) ? kNan : static_cast<CodeT>((b0 - cuts) + (*b0 <= x0));
            dst[dst_stride]     = std::isnan(x1) ? kNan : static_cast<CodeT>((b1 - cuts) + (*b1 <= x1));
            dst[2 * dst_stride] = std::isnan(x2) ? kNan : static_cast<CodeT>((b2 - cuts) + (*b2 <= x2));
            dst[3 * dst_stride] = std::isnan(x3) ? kNan : static_cast<CodeT>((b3 - cuts) + (*b3 <= x3));
        }

        // Bins `count` values from src[r * src_stride] into dst[r * dst_stride], dispatching ONCE on len.
        template <typename CodeT>
        void bin_feature_strided(const double* LIGNUM_RESTRICT cuts, uint32_t len, const double* LIGNUM_RESTRICT src, size_t src_stride,
                                 size_t count, CodeT* LIGNUM_RESTRICT dst, size_t dst_stride) {
            constexpr CodeT kNan = std::numeric_limits<CodeT>::max();
            if (len == 0) {
                for (size_t r = 0; r < count; r++) dst[r * dst_stride] = std::isnan(src[r * src_stride]) ? kNan : CodeT{0};
                return;
            }
            if (len <= 8) {
                for (size_t r = 0; r < count; r++) {
                    const double x = src[r * src_stride];
                    uint32_t b = 0;
                    for (uint32_t k = 0; k < len; k++) b += (cuts[k] <= x);
                    dst[r * dst_stride] = std::isnan(x) ? kNan : static_cast<CodeT>(b);
                }
                return;
            }
            const int logp = std::bit_width(len) - 1;
            const uint32_t rem = len - (uint32_t{1} << logp);
            const size_t n4 = (count / 4) * 4;
            auto run = [&]<int LP>() LIGNUM_LAMBDA_INLINE {
                for (size_t r = 0; r < n4; r += 4) {
                    bin_pow2_4<LP, CodeT>(cuts, rem, src + r * src_stride, src_stride, dst + r * dst_stride, dst_stride);
                }
                for (size_t r = n4; r < count; r++) {
                    dst[r * dst_stride] = bin_pow2<LP, CodeT>(cuts, rem, src[r * src_stride]);
                }
            };
            switch (logp) {
                case 3:  run.template operator()<3>();  break;
                case 4:  run.template operator()<4>();  break;
                case 5:  run.template operator()<5>();  break;
                case 6:  run.template operator()<6>();  break;
                case 7:  run.template operator()<7>();  break;
                case 8:  run.template operator()<8>();  break;
                case 9:  run.template operator()<9>();  break;
                case 10: run.template operator()<10>(); break;
                case 11: run.template operator()<11>(); break;
                case 12: run.template operator()<12>(); break;
                case 13: run.template operator()<13>(); break;
                case 14: run.template operator()<14>(); break;
                case 15: run.template operator()<15>(); break;
                default:
                    for (size_t r = 0; r < count; r++) dst[r * dst_stride] = bin_of<CodeT>(cuts, len, src[r * src_stride]);
                    break;
            }
        }

        // Category code under LightGBM rules (valid iff -1 < x < 2^31, truncation) or XGBoost rules
        // (float32 first, valid iff 0 <= x < 2^24). Codes >= kInvalidCat never belong to a set.
        LIGNUM_INLINE uint16_t category_code(double x, bool xgboost) {
            if (std::isnan(x)) return kNanBin;
            const double v = xgboost ? static_cast<double>(static_cast<float>(x)) : x;
            const bool valid = xgboost ? (v >= 0.0 && v < 16777216.0) : (v > -1.0 && v < 2147483648.0);
            if (!valid) return kInvalidCat;
            const uint32_t c = static_cast<uint32_t>(v);
            return static_cast<uint16_t>(std::min<uint32_t>(c, kInvalidCat));
        }

        // Per row: 8-bit or 16-bit codes (bins / categories) and precomputed categorical decision bits.
        template <typename CodeT>
        struct BinnedRow {
            const CodeT* LIGNUM_RESTRICT codes;
            const uint64_t* LIGNUM_RESTRICT decisions;
            const uint16_t* LIGNUM_RESTRICT zero_bin;  // per dense feature (model-wide)
        };

        // Kind: tree_kind bits of the tree (Categorical, ZeroMissing).
        template <uint8_t Kind, typename CodeT = uint16_t>
        LIGNUM_INLINE int32_t bdir(const BNode& b, int s, const BinnedRow<CodeT>& r) {
            constexpr CodeT kNan = std::numeric_limits<CodeT>::max();
            const CodeT code = r.codes[b.features[s]];
            if constexpr ((Kind & tree_kind::Categorical) != 0) {
                if ((b.cat_slots >> s) & 1) {
                    // The answer was computed before the walk: one bit from L1.
                    const uint32_t bit = b.thresholds[s];
                    const bool in_set = (r.decisions[bit >> 6] >> (bit & 63)) & 1;
                    if ((b.xgb_slots >> s) & 1) {
                        return static_cast<int32_t>(in_set | ((code == kNan) & static_cast<bool>((b.nan_left >> s) & 1)));
                    }
                    return static_cast<int32_t>(!in_set);
                }
            }
            const uint32_t shift = (b.nan_left >> s) & 1;
            const int32_t right = static_cast<CodeT>(code + shift) >= b.thresholds[s];
            if constexpr ((Kind & tree_kind::ZeroMissing) != 0) {
                // Branch-free: the zero band takes the NaN direction (right iff no shift).
                const bool in_band = static_cast<bool>((b.zero_slots >> s) & 1) & (static_cast<uint16_t>(code) == r.zero_bin[b.features[s]]);
                return in_band ? static_cast<int32_t>(shift ^ 1) : right;
            }
            return right;
        }

        // Three tree levels per step, comparing only along the path (3 dependent comparisons):
        // evaluating all seven speculatively was 2x slower (the kernel is compute bound).
        template <uint8_t Kind, typename CodeT = uint16_t>
        LIGNUM_INLINE int32_t bstep(const BNode& b, const BinnedRow<CodeT>& r) {
            const int32_t d0 = bdir<Kind>(b, 0, r);
            const int32_t d1 = bdir<Kind>(b, 1 + d0, r);
            const int32_t g = (d0 << 1) | d1;
            const int32_t d2 = bdir<Kind>(b, 3 + g, r);
            return b.children[(g << 1) | d2];
        }

        template <uint8_t Kind, typename CodeT>
        LIGNUM_INLINE void bwalk8(const BNode* LIGNUM_RESTRICT nodes, int32_t root, const BinnedRow<CodeT>* r,
                                  int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3,
                                  int32_t& n4, int32_t& n5, int32_t& n6, int32_t& n7) {
            if (root < 0) { n0 = n1 = n2 = n3 = n4 = n5 = n6 = n7 = root; return; }
            const BNode& rn = nodes[root];
            n0 = bstep<Kind>(rn, r[0]);
            n1 = bstep<Kind>(rn, r[1]);
            n2 = bstep<Kind>(rn, r[2]);
            n3 = bstep<Kind>(rn, r[3]);
            n4 = bstep<Kind>(rn, r[4]);
            n5 = bstep<Kind>(rn, r[5]);
            n6 = bstep<Kind>(rn, r[6]);
            n7 = bstep<Kind>(rn, r[7]);
            while ((n0 & n1 & n2 & n3 & n4 & n5 & n6 & n7) >= 0) {
                if (n0 >= 0) n0 = bstep<Kind>(nodes[n0], r[0]);
                if (n1 >= 0) n1 = bstep<Kind>(nodes[n1], r[1]);
                if (n2 >= 0) n2 = bstep<Kind>(nodes[n2], r[2]);
                if (n3 >= 0) n3 = bstep<Kind>(nodes[n3], r[3]);
                if (n4 >= 0) n4 = bstep<Kind>(nodes[n4], r[4]);
                if (n5 >= 0) n5 = bstep<Kind>(nodes[n5], r[5]);
                if (n6 >= 0) n6 = bstep<Kind>(nodes[n6], r[6]);
                if (n7 >= 0) n7 = bstep<Kind>(nodes[n7], r[7]);
            }
        }

        template <uint8_t Kind, typename CodeT>
        LIGNUM_INLINE void bwalk4(const BNode* LIGNUM_RESTRICT nodes, int32_t root, const BinnedRow<CodeT>* r,
                                  int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) {
            if (root < 0) { n0 = n1 = n2 = n3 = root; return; }
            const BNode& rn = nodes[root];
            n0 = bstep<Kind>(rn, r[0]);
            n1 = bstep<Kind>(rn, r[1]);
            n2 = bstep<Kind>(rn, r[2]);
            n3 = bstep<Kind>(rn, r[3]);
            while ((n0 & n1 & n2 & n3) >= 0) {
                if (n0 >= 0) n0 = bstep<Kind>(nodes[n0], r[0]);
                if (n1 >= 0) n1 = bstep<Kind>(nodes[n1], r[1]);
                if (n2 >= 0) n2 = bstep<Kind>(nodes[n2], r[2]);
                if (n3 >= 0) n3 = bstep<Kind>(nodes[n3], r[3]);
            }
        }

        // Four trees of the same row, interleaved (rows outside the 4-row groups, e.g. one row).
        template <uint8_t Kind, typename CodeT = uint16_t>
        LIGNUM_INLINE void bwalk4_trees(const BNode* LIGNUM_RESTRICT nodes, const int32_t* roots, const BinnedRow<CodeT>& r,
                                        int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) {
            n0 = roots[0]; n1 = roots[1]; n2 = roots[2]; n3 = roots[3];
            while ((n0 & n1 & n2 & n3) >= 0) {
                if (n0 >= 0) n0 = bstep<Kind>(nodes[n0], r);
                if (n1 >= 0) n1 = bstep<Kind>(nodes[n1], r);
                if (n2 >= 0) n2 = bstep<Kind>(nodes[n2], r);
                if (n3 >= 0) n3 = bstep<Kind>(nodes[n3], r);
            }
        }

        template <uint8_t Kind, typename CodeT = uint16_t>
        LIGNUM_INLINE int32_t bwalk1(const BNode* LIGNUM_RESTRICT nodes, int32_t node, const BinnedRow<CodeT>& r) {
            while (node >= 0) node = bstep<Kind>(nodes[node], r);
            return node;
        }

        // Copies precomputed categorical decision bits for every categorical feature of one row.
        LIGNUM_INLINE void copy_decisions(const BinnedForest& f, const uint16_t* LIGNUM_RESTRICT rc, uint64_t* LIGNUM_RESTRICT rd) {
            for (size_t k = 0; k < f.n_cat_features; k++) {
                const BinnedCatFeature& cf = f.cat_features[k];
                const uint16_t code = rc[cf.dense];
                if (code < cf.n_categories) {
                    std::memcpy(rd + cf.row_offset, &f.cat_tables[cf.table_offset + uint64_t{code} * cf.words], cf.words * sizeof(uint64_t));
                } else {
                    std::memset(rd + cf.row_offset, 0, cf.words * sizeof(uint64_t));
                }
            }
        }

        // Bins all CTR dense features for row-major X[i_begin .. i_begin + count).
        void bin_ctr_rows(const Model& m, const BinnedForest& f, const double* X, size_t n_features,
                          size_t i_begin, size_t count, uint16_t* LIGNUM_RESTRICT codes, size_t stride) {
            for (size_t j = 0; j < f.n_features; j++) {
                if (f.feature_kind[j] != feature_kind::Ctr) continue;
                const uint32_t tbl_idx = f.feature_of[j];
                const uint32_t lo = f.cut_offsets[j];
                const uint32_t len = f.cut_offsets[j + 1] - lo;
                const double* cuts = &f.cuts[lo];
                for (size_t r = 0; r < count; r++) {
                    const double* row = &X[(i_begin + r) * n_features];
                    const float val = lookup_ctr(m.ctr_tables, m.ctr_keys, m.ctr_slots, m.ctr_values, tbl_idx, row);
                    codes[r * stride + j] = bin_of<uint16_t>(cuts, len, static_cast<double>(val));
                }
            }
        }

        // Bins all CTR dense features for columnar input.
        void bin_ctr_columns(const Model& m, const BinnedForest& f, const ColumnSource& columns,
                             size_t i_begin, size_t count, uint16_t* LIGNUM_RESTRICT codes, size_t stride) {
            std::vector<uint32_t> cols;
            for (size_t j = 0; j < f.n_features; j++) {
                if (f.feature_kind[j] != feature_kind::Ctr) continue;
                const CtrTableHeader& t = m.ctr_tables[f.feature_of[j]];
                for (uint8_t c = 0; c < t.n_cols; c++) cols.push_back(t.columns[c]);
            }
            if (cols.empty()) return;
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

            thread_local std::vector<double> col_buf;
            col_buf.resize(cols.size() * count);
            for (size_t ci = 0; ci < cols.size(); ci++) {
                columns.read(cols[ci], i_begin, count, &col_buf[ci * count]);
            }
            constexpr uint32_t kMissingCatHash = 0x9e3779b9U;
            constexpr uint64_t kCtrHashMagic = 0x4906ba494954cb65ULL;
            auto slot_of = [&](uint32_t col) -> size_t {
                return static_cast<size_t>(std::lower_bound(cols.begin(), cols.end(), col) - cols.begin());
            };
            auto hash_at = [&](size_t slot, size_t r) -> uint64_t {
                const double x = col_buf[slot * count + r];
                if (std::isnan(x) || x < 0.0 || x >= 4294967296.0) return kMissingCatHash;
                return static_cast<uint64_t>(static_cast<uint32_t>(x));
            };
            for (size_t j = 0; j < f.n_features; j++) {
                if (f.feature_kind[j] != feature_kind::Ctr) continue;
                const CtrTableHeader& t = m.ctr_tables[f.feature_of[j]];
                const uint32_t lo = f.cut_offsets[j];
                const uint32_t len = f.cut_offsets[j + 1] - lo;
                const double* cuts = &f.cuts[lo];
                const size_t s0 = slot_of(t.columns[0]);
                const size_t s1 = t.n_cols > 1 ? slot_of(t.columns[1]) : 0;
                const size_t s2 = t.n_cols > 2 ? slot_of(t.columns[2]) : 0;
                const uint64_t* LIGNUM_RESTRICT tkeys = m.ctr_keys + t.keys_offset;
                const uint32_t* LIGNUM_RESTRICT tslots = m.ctr_slots + t.slots_offset;
                const float* LIGNUM_RESTRICT tvals = m.ctr_values + t.values_offset;
                for (size_t r = 0; r < count; r++) {
                    float val = t.default_value;
                    if (t.n_slots > 0) {
                        uint64_t h = hash_at(s0, r);
                        if (t.n_cols > 1) h = kCtrHashMagic * (h + kCtrHashMagic * hash_at(s1, r));
                        if (t.n_cols > 2) h = kCtrHashMagic * (h + kCtrHashMagic * hash_at(s2, r));
                        uint64_t idx = (h ^ (h >> 29U)) & t.mask;
                        for (;;) {
                            const uint32_t entry = tslots[idx];
                            if (entry == 0U) break;
                            if (tkeys[idx] == h) { val = tvals[entry - 1U]; break; }
                            idx = (idx + 1U) & t.mask;
                        }
                    }
                    codes[r * stride + j] = bin_of<uint16_t>(cuts, len, static_cast<double>(val));
                }
            }
        }

        // Columnar input: each feature column of the chunk is read once into a small contiguous
        // buffer and binned in one loop, then categorical decision bits are copied per row.
        template <typename CodeT, bool Extended = false>
        void bin_columns_into(const Model& m, const BinnedForest& f, const ColumnSource& columns, size_t i_begin, size_t i_end,
                              CodeT* LIGNUM_RESTRICT codes, uint64_t* LIGNUM_RESTRICT decisions) {
            const size_t stride = Extended ? f.row_codes() : f.n_features;
            const size_t nw = f.decision_words, count = i_end - i_begin;
            thread_local std::vector<double> values;
            values.resize(count);
            if constexpr (Extended && sizeof(CodeT) == 2) {
                for (size_t q = 0; q < f.n_q16; q++) {
                    const uint32_t col = f.q16_columns[q];
                    columns.read(col, i_begin, count, values.data());
                    for (size_t i = 0; i < count; i++) {
                        codes[i * stride + f.n_features + q] = q16_code(
                            m.quantizers, m.quantizer_constant, m.quantizer_knots, m.quantizer_knot_offsets, col, values[i]);
                    }
                }
            }
            for (size_t j = 0; j < f.n_features; j++) {
                const uint8_t kind = f.feature_kind[j];
                if (kind == feature_kind::Ctr) continue;
                columns.read(f.feature_of[j], i_begin, count, values.data());
                if (kind == feature_kind::Numeric) {
                    const uint32_t lo = f.cut_offsets[j];
                    bin_feature_strided<CodeT>(&f.cuts[lo], f.cut_offsets[j + 1] - lo,
                                               values.data(), 1, count, codes + j, stride);
                } else if constexpr (sizeof(CodeT) == 2) {
                    const bool xgboost = kind == feature_kind::XGBoostCategory;
                    for (size_t i = 0; i < count; i++) codes[i * stride + j] = category_code(values[i], xgboost);
                }
            }
            if constexpr (sizeof(CodeT) == 2) {
                if (f.has_ctr_features) bin_ctr_columns(m, f, columns, i_begin, count, codes, stride);
                if (f.n_cat_features > 0) {
                    for (size_t i = 0; i < count; i++) copy_decisions(f, &codes[i * stride], &decisions[i * nw]);
                }
            } else {
                (void)nw; (void)decisions; (void)m;
            }
        }

        // Shared row-major / columnar Phase 1 binning for BinnedKernel / ObliqueBinnedKernel / SymBinnedKernel.
        template <typename CodeT, bool Extended>
        void bin_rows_into(const Model& m, const BinnedForest& f, const ColumnSource* columns, const double* X,
                           size_t n_features, size_t i_begin, size_t i_end, CodeT* LIGNUM_RESTRICT codes,
                           uint64_t* LIGNUM_RESTRICT decisions) {
            const size_t count = i_end - i_begin;
            const size_t stride = Extended ? f.row_codes() : f.n_features;
            const size_t nw = f.decision_words;
            if (columns) {
                bin_columns_into<CodeT, Extended>(m, f, *columns, i_begin, i_end, codes, decisions);
                return;
            }
            if (count >= 16) {
                const double* base_X = &X[i_begin * n_features];
                if constexpr (Extended && sizeof(CodeT) == 2) {
                    for (size_t q = 0; q < f.n_q16; q++) {
                        const uint32_t col = f.q16_columns[q];
                        for (size_t r = 0; r < count; r++) {
                            codes[r * stride + f.n_features + q] = q16_code(
                                m.quantizers, m.quantizer_constant, m.quantizer_knots, m.quantizer_knot_offsets,
                                col, base_X[r * n_features + col]);
                        }
                    }
                }
                for (size_t j = 0; j < f.n_features; j++) {
                    const uint8_t kind = f.feature_kind[j];
                    if (kind == feature_kind::Ctr) continue;
                    const uint32_t col = f.feature_of[j];
                    if (kind == feature_kind::Numeric) {
                        const uint32_t lo = f.cut_offsets[j];
                        bin_feature_strided<CodeT>(&f.cuts[lo], f.cut_offsets[j + 1] - lo,
                                                   base_X + col, n_features, count, codes + j, stride);
                    } else if constexpr (sizeof(CodeT) == 2) {
                        const bool xgboost = kind == feature_kind::XGBoostCategory;
                        for (size_t r = 0; r < count; r++) {
                            codes[r * stride + j] = category_code(base_X[r * n_features + col], xgboost);
                        }
                    }
                }
                if constexpr (sizeof(CodeT) == 2) {
                    if (f.has_ctr_features) bin_ctr_rows(m, f, X, n_features, i_begin, count, codes, stride);
                    if (f.n_cat_features > 0) {
                        for (size_t r = 0; r < count; r++) {
                            copy_decisions(f, &codes[r * stride], &decisions[r * nw]);
                        }
                    }
                }
                return;
            }
            for (size_t i = i_begin; i < i_end; i++) {
                const double* row = &X[i * n_features];
                CodeT* rc = &codes[(i - i_begin) * stride];
                if constexpr (Extended && sizeof(CodeT) == 2) {
                    for (size_t q = 0; q < f.n_q16; q++) {
                        const uint32_t col = f.q16_columns[q];
                        rc[f.n_features + q] = q16_code(
                            m.quantizers, m.quantizer_constant, m.quantizer_knots, m.quantizer_knot_offsets, col, row[col]);
                    }
                }
                for (size_t j = 0; j < f.n_features; j++) {
                    const uint8_t kind = f.feature_kind[j];
                    if (kind == feature_kind::Ctr) continue;
                    const double x = row[f.feature_of[j]];
                    if (kind == feature_kind::Numeric) {
                        const uint32_t lo = f.cut_offsets[j];
                        rc[j] = bin_of<CodeT>(&f.cuts[lo], f.cut_offsets[j + 1] - lo, x);
                    } else if constexpr (sizeof(CodeT) == 2) {
                        rc[j] = category_code(x, kind == feature_kind::XGBoostCategory);
                    }
                }
                if constexpr (sizeof(CodeT) == 2) {
                    if (f.n_cat_features > 0) {
                        copy_decisions(f, rc, &decisions[(i - i_begin) * nw]);
                    }
                }
            }
            if constexpr (sizeof(CodeT) == 2) {
                if (f.has_ctr_features) bin_ctr_rows(m, f, X, n_features, i_begin, count, codes, stride);
            }
        }

        // Initializes the per-row margin buffer for a block with base_scores.
        template <typename Acc, bool Multi>
        LIGNUM_INLINE double* prepare_block_dst(const Model& m, size_t i_begin, size_t count, bool argmax,
                                                double* LIGNUM_RESTRICT out, double* LIGNUM_RESTRICT margins_buf) {
            const size_t K = Multi ? m.n_outputs : 1;
            double* LIGNUM_RESTRICT dst = (Multi && argmax) ? margins_buf : &out[i_begin * K];
            for (size_t r = 0; r < count; r++) {
                for (size_t c = 0; c < K; c++) dst[r * K + c] = static_cast<double>(static_cast<Acc>(m.base_scores[c]));
            }
            return dst;
        }

        // Applies Argmax narrowing or finish_rows transform at the end of a block.
        template <typename Acc, bool Multi>
        LIGNUM_INLINE void finish_block_dst(const Model& m, size_t i_begin, size_t count, bool raw_score, bool argmax,
                                            const double* LIGNUM_RESTRICT dst, double* LIGNUM_RESTRICT out) {
            if (argmax) {
                if constexpr (Multi) {
                    const size_t K = m.n_outputs;
                    for (size_t r = 0; r < count; r++) {
                        const double* row_m = &dst[r * K];
                        out[i_begin + r] = static_cast<double>(std::max_element(row_m, row_m + K) - row_m);
                    }
                } else {
                    for (size_t r = 0; r < count; r++) out[i_begin + r] = 0.0;
                }
            } else {
                finish_rows<Acc, Multi>(m, raw_score, i_begin, i_begin + count, out);
            }
        }

        template <typename Acc, bool Multi>
        class CompiledKernel {
            public:
                CompiledKernel(const Model& model, const BinnedForest& forest, const CompiledForest& compiled,
                               const double* X, size_t n_features, bool raw_score, const ColumnSource* columns = nullptr)
                    : m_(model), f_(forest), compiled_(compiled), columns_(columns), X_(X), n_features_(n_features), raw_score_(raw_score) {}

                void predict_rows(size_t i_begin, size_t i_end, double* LIGNUM_RESTRICT out) const {
                    const size_t count = i_end - i_begin;
                    const size_t stride = f_.row_codes();
                    const size_t nw = f_.decision_words;
                    thread_local std::vector<uint16_t> buffer;
                    thread_local std::vector<uint64_t> decision_buffer;
                    buffer.resize(count * stride);
                    decision_buffer.resize(count * nw);
                    uint16_t* codes = buffer.data();
                    uint64_t* decisions = decision_buffer.data();
                    if (columns_) {
                        bin_columns_into<uint16_t, true>(m_, f_, *columns_, i_begin, i_end, codes, decisions);
                    } else {
                        // Compiled trees come with their own binning of numeric and Q16 features.
                        compiled_.bin(&X_[i_begin * n_features_], n_features_, count, codes);
                        if (f_.has_ctr_features) bin_ctr_rows(m_, f_, X_, n_features_, i_begin, count, codes, stride);
                        if (f_.n_cat_features > 0) {
                            for (size_t r = 0; r < count; r++) {
                                const double* row = &X_[(i_begin + r) * n_features_];
                                uint16_t* rc = &codes[r * stride];
                                for (size_t k = 0; k < f_.n_cat_features; k++) {
                                    const uint32_t j = f_.cat_features[k].dense;
                                    rc[j] = category_code(row[f_.feature_of[j]], f_.feature_kind[j] == feature_kind::XGBoostCategory);
                                }
                                copy_decisions(f_, rc, &decisions[r * nw]);
                            }
                        }
                    }
                    const bool argmax = !raw_score_ && m_.transform == transform::Argmax;
                    double* LIGNUM_RESTRICT dst = nullptr;
                    if constexpr (Multi) {
                        if (argmax) {
                            thread_local std::vector<double> block_margins;
                            block_margins.resize(count * m_.n_outputs);
                            dst = prepare_block_dst<Acc, Multi>(m_, i_begin, count, true, out, block_margins.data());
                        } else {
                            dst = prepare_block_dst<Acc, Multi>(m_, i_begin, count, false, out, nullptr);
                        }
                    } else {
                        dst = prepare_block_dst<Acc, Multi>(m_, i_begin, count, argmax, out, nullptr);
                    }
                    // Compiled trees: blocks in model order, each adding its trees to every row.
                    for (const auto block : compiled_.blocks) block(codes, decisions, count, dst);
                    finish_block_dst<Acc, Multi>(m_, i_begin, count, raw_score_, argmax, dst, out);
                }

            private:
                const Model& m_;
                const BinnedForest& f_;
                const CompiledForest& compiled_;
                const ColumnSource* columns_;
                const double* X_;
                size_t n_features_;
                bool raw_score_;
        };

        template <typename Acc, bool Mixed, bool Multi, typename CodeT = uint16_t>
        class BinnedKernel {
            public:
                // columns: columnar input to bin instead of the row-major X (then X is null).
                BinnedKernel(const Model& model, const BinnedForest& forest, const double* X,
                             size_t n_features, bool raw_score, const ColumnSource* columns = nullptr)
                    : m_(model), f_(forest), columns_(columns), X_(X), n_features_(n_features), raw_score_(raw_score) {}

                void predict_rows(size_t i_begin, size_t i_end, double* LIGNUM_RESTRICT out) const {
                    const size_t count = i_end - i_begin;
                    const size_t nd = f_.n_features;
                    const size_t nw = f_.decision_words;
                    thread_local std::vector<CodeT> buffer;
                    thread_local std::vector<uint64_t> decision_buffer;
                    buffer.resize(count * nd);
                    decision_buffer.resize(count * nw);
                    CodeT* codes = buffer.data();
                    uint64_t* decisions = decision_buffer.data();
                    bin_rows_into<CodeT, false>(m_, f_, columns_, X_, n_features_, i_begin, i_end, codes, decisions);

                    const bool argmax = !raw_score_ && m_.transform == transform::Argmax;
                    double* LIGNUM_RESTRICT dst = nullptr;
                    if constexpr (Multi) {
                        if (argmax) {
                            thread_local std::vector<double> block_margins;
                            block_margins.resize(count * m_.n_outputs);
                            dst = prepare_block_dst<Acc, Multi>(m_, i_begin, count, true, out, block_margins.data());
                        } else {
                            dst = prepare_block_dst<Acc, Multi>(m_, i_begin, count, false, out, nullptr);
                        }
                    } else {
                        dst = prepare_block_dst<Acc, Multi>(m_, i_begin, count, argmax, out, nullptr);
                    }
                    for (size_t t_start = 0; t_start < f_.n_trees; t_start += T_BLOCK) {
                        accumulate(codes, nd, decisions, nw, count, t_start, std::min(t_start + T_BLOCK, f_.n_trees), dst);
                    }
                    finish_block_dst<Acc, Multi>(m_, i_begin, count, raw_score_, argmax, dst, out);
                }

            private:
                const Model& m_;
                const BinnedForest& f_;
                const ColumnSource* columns_;
                const double* X_;
                size_t n_features_;
                bool raw_score_;

                // Adds a leaf to the running sum of local row r's output for tree t, in model order.
                LIGNUM_INLINE void add(double* LIGNUM_RESTRICT dst, size_t r, size_t t, int32_t node) const {
                    const size_t slot = Multi ? r * m_.n_outputs + m_.tree_outputs[t] : r;
                    dst[slot] = static_cast<double>(static_cast<Acc>(dst[slot]) + static_cast<Acc>(leaf_value(m_.leaf_values, node)));
                }

                void accumulate(const CodeT* LIGNUM_RESTRICT codes, size_t nd, const uint64_t* LIGNUM_RESTRICT decisions, size_t nw,
                                size_t count, size_t t_begin, size_t t_end, double* LIGNUM_RESTRICT dst) const {
                    static constexpr uint64_t kZeroDecision = 0;
                    const BNode* LIGNUM_RESTRICT nodes = f_.nodes;
                    const int32_t* LIGNUM_RESTRICT roots = f_.roots;
                    const uint8_t* LIGNUM_RESTRICT kinds = f_.tree_kind;
                    const double* LIGNUM_RESTRICT leaf_values = m_.leaf_values;
                    auto row = [&](size_t r) LIGNUM_LAMBDA_INLINE {
                        return BinnedRow<CodeT>{&codes[r * nd], nw > 0 ? &decisions[r * nw] : &kZeroDecision, f_.zero_bin};
                    };
                    size_t i_cur = 0;
                    if constexpr (!Mixed && LIGNUM_WALK8) {
                        const size_t unrolled8_end = (count / 8) * 8;
                        for (size_t i = 0; i < unrolled8_end; i += 8) {
                            const BinnedRow<CodeT> r[8] = {row(i), row(i + 1), row(i + 2), row(i + 3), row(i + 4), row(i + 5), row(i + 6), row(i + 7)};
                            if constexpr (Multi) {
                                for (size_t t = t_begin; t < t_end; t++) {
                                    int32_t n0, n1, n2, n3, n4, n5, n6, n7;
                                    with_kind<Mixed>(Mixed ? kinds[t] : 0, [&](auto K) LIGNUM_LAMBDA_INLINE {
                                        bwalk8<decltype(K)::value>(nodes, roots[t], r, n0, n1, n2, n3, n4, n5, n6, n7);
                                    });
                                    add(dst, i, t, n0);     add(dst, i + 1, t, n1); add(dst, i + 2, t, n2); add(dst, i + 3, t, n3);
                                    add(dst, i + 4, t, n4); add(dst, i + 5, t, n5); add(dst, i + 6, t, n6); add(dst, i + 7, t, n7);
                                }
                            } else {
                                Acc sum0 = static_cast<Acc>(dst[i]),     sum1 = static_cast<Acc>(dst[i + 1]);
                                Acc sum2 = static_cast<Acc>(dst[i + 2]), sum3 = static_cast<Acc>(dst[i + 3]);
                                Acc sum4 = static_cast<Acc>(dst[i + 4]), sum5 = static_cast<Acc>(dst[i + 5]);
                                Acc sum6 = static_cast<Acc>(dst[i + 6]), sum7 = static_cast<Acc>(dst[i + 7]);
                                for (size_t t = t_begin; t < t_end; t++) {
                                    int32_t n0, n1, n2, n3, n4, n5, n6, n7;
                                    with_kind<Mixed>(Mixed ? kinds[t] : 0, [&](auto K) LIGNUM_LAMBDA_INLINE {
                                        bwalk8<decltype(K)::value>(nodes, roots[t], r, n0, n1, n2, n3, n4, n5, n6, n7);
                                    });
                                    sum0 += static_cast<Acc>(leaf_value(leaf_values, n0));
                                    sum1 += static_cast<Acc>(leaf_value(leaf_values, n1));
                                    sum2 += static_cast<Acc>(leaf_value(leaf_values, n2));
                                    sum3 += static_cast<Acc>(leaf_value(leaf_values, n3));
                                    sum4 += static_cast<Acc>(leaf_value(leaf_values, n4));
                                    sum5 += static_cast<Acc>(leaf_value(leaf_values, n5));
                                    sum6 += static_cast<Acc>(leaf_value(leaf_values, n6));
                                    sum7 += static_cast<Acc>(leaf_value(leaf_values, n7));
                                }
                                dst[i]     = static_cast<double>(sum0); dst[i + 1] = static_cast<double>(sum1);
                                dst[i + 2] = static_cast<double>(sum2); dst[i + 3] = static_cast<double>(sum3);
                                dst[i + 4] = static_cast<double>(sum4); dst[i + 5] = static_cast<double>(sum5);
                                dst[i + 6] = static_cast<double>(sum6); dst[i + 7] = static_cast<double>(sum7);
                            }
                        }
                        i_cur = unrolled8_end;
                    }
                    const size_t unrolled_end = (count / 4) * 4;
                    for (size_t i = i_cur; i < unrolled_end; i += 4) {
                        const BinnedRow<CodeT> r[4] = {row(i), row(i + 1), row(i + 2), row(i + 3)};
                        if constexpr (Multi) {
                            for (size_t t = t_begin; t < t_end; t++) {
                                int32_t n0, n1, n2, n3;
                                with_kind<Mixed>(Mixed ? kinds[t] : 0, [&](auto K) LIGNUM_LAMBDA_INLINE {
                                    bwalk4<decltype(K)::value>(nodes, roots[t], r, n0, n1, n2, n3);
                                });
                                add(dst, i, t, n0);
                                add(dst, i + 1, t, n1);
                                add(dst, i + 2, t, n2);
                                add(dst, i + 3, t, n3);
                            }
                        } else {
                            Acc sum0 = static_cast<Acc>(dst[i]);
                            Acc sum1 = static_cast<Acc>(dst[i + 1]);
                            Acc sum2 = static_cast<Acc>(dst[i + 2]);
                            Acc sum3 = static_cast<Acc>(dst[i + 3]);
                            for (size_t t = t_begin; t < t_end; t++) {
                                int32_t n0, n1, n2, n3;
                                with_kind<Mixed>(Mixed ? kinds[t] : 0, [&](auto K) LIGNUM_LAMBDA_INLINE {
                                    bwalk4<decltype(K)::value>(nodes, roots[t], r, n0, n1, n2, n3);
                                });
                                sum0 += static_cast<Acc>(leaf_value(leaf_values, n0));
                                sum1 += static_cast<Acc>(leaf_value(leaf_values, n1));
                                sum2 += static_cast<Acc>(leaf_value(leaf_values, n2));
                                sum3 += static_cast<Acc>(leaf_value(leaf_values, n3));
                            }
                            dst[i] = static_cast<double>(sum0);
                            dst[i + 1] = static_cast<double>(sum1);
                            dst[i + 2] = static_cast<double>(sum2);
                            dst[i + 3] = static_cast<double>(sum3);
                        }
                    }
                    // Leftover rows: interleave four trees of the same row; leaves still in tree order.
                    const size_t trees_unrolled_end = t_begin + ((t_end - t_begin) / 4) * 4;
                    for (size_t i = unrolled_end; i < count; i++) {
                        const BinnedRow<CodeT> r = row(i);
                        if constexpr (!Multi) {
                            Acc sum = static_cast<Acc>(dst[i]);
                            for (size_t t = t_begin; t < trees_unrolled_end; t += 4) {
                                int32_t n0, n1, n2, n3;
                                const uint8_t k4 = Mixed ? static_cast<uint8_t>(kinds[t] | kinds[t + 1] | kinds[t + 2] | kinds[t + 3]) : 0;
                                with_kind<Mixed>(k4, [&](auto K) LIGNUM_LAMBDA_INLINE {
                                    bwalk4_trees<decltype(K)::value>(nodes, &roots[t], r, n0, n1, n2, n3);
                                });
                                sum += static_cast<Acc>(leaf_value(leaf_values, n0));
                                sum += static_cast<Acc>(leaf_value(leaf_values, n1));
                                sum += static_cast<Acc>(leaf_value(leaf_values, n2));
                                sum += static_cast<Acc>(leaf_value(leaf_values, n3));
                            }
                            for (size_t t = trees_unrolled_end; t < t_end; t++) {
                                int32_t leaf;
                                with_kind<Mixed>(Mixed ? kinds[t] : 0, [&](auto K) LIGNUM_LAMBDA_INLINE {
                                    leaf = bwalk1<decltype(K)::value>(nodes, roots[t], r);
                                });
                                sum += static_cast<Acc>(leaf_value(leaf_values, leaf));
                            }
                            dst[i] = static_cast<double>(sum);
                        } else {
                            for (size_t t = t_begin; t < trees_unrolled_end; t += 4) {
                                int32_t n0, n1, n2, n3;
                                const uint8_t k4 = Mixed ? static_cast<uint8_t>(kinds[t] | kinds[t + 1] | kinds[t + 2] | kinds[t + 3]) : 0;
                                with_kind<Mixed>(k4, [&](auto K) LIGNUM_LAMBDA_INLINE {
                                    bwalk4_trees<decltype(K)::value>(nodes, &roots[t], r, n0, n1, n2, n3);
                                });
                                add(dst, i, t, n0);
                                add(dst, i, t + 1, n1);
                                add(dst, i, t + 2, n2);
                                add(dst, i, t + 3, n3);
                            }
                            for (size_t t = trees_unrolled_end; t < t_end; t++) {
                                int32_t leaf;
                                with_kind<Mixed>(Mixed ? kinds[t] : 0, [&](auto K) LIGNUM_LAMBDA_INLINE {
                                    leaf = bwalk1<decltype(K)::value>(nodes, roots[t], r);
                                });
                                add(dst, i, t, leaf);
                            }
                        }
                    }
                }
        };

        // ---- ONode walk (trees with oblique splits; see ONode in src/binned.hpp) ----
        template <bool Cat>
        LIGNUM_INLINE int32_t odir(const ONodeSlot& s, const BinnedRow<uint16_t>& r) {
            if constexpr (Cat) {
                if ((s.w[0] | s.w[1] | s.w[2]) == 0) {
                    const uint32_t bit = static_cast<uint32_t>(s.thr);
                    return static_cast<int32_t>(((r.decisions[bit >> 6] >> (bit & 63)) & 1) == 0);
                }
            }
            const uint16_t c0 = r.codes[s.f[0] & 0x7FFF], c1 = r.codes[s.f[1]], c2 = r.codes[s.f[2]];
            const bool nan = (c0 == kNanBin) | (c1 == kNanBin) | (c2 == kNanBin);
            const int32_t p = static_cast<int32_t>(s.w[0]) * (static_cast<int32_t>(c0) - 32768) +
                              static_cast<int32_t>(s.w[1]) * (static_cast<int32_t>(c1) - 32768) +
                              static_cast<int32_t>(s.w[2]) * (static_cast<int32_t>(c2) - 32768);
            return nan ? static_cast<int32_t>(s.f[0] >> 15) : static_cast<int32_t>(p >= s.thr);
        }

        template <bool Cat>
        LIGNUM_INLINE int32_t ostep(const ONode& o, const BinnedRow<uint16_t>& r) {
            // All three slots speculatively (no dependency between the two levels), as in KNode.
            const int32_t d0 = odir<Cat>(o.s[0], r);
            const int32_t l = odir<Cat>(o.s[1], r);
            const int32_t c = odir<Cat>(o.s[2], r);
            return o.children[(d0 << 1) | (d0 ? c : l)];
        }

        template <bool Cat>
        LIGNUM_INLINE void owalk4(const ONode* LIGNUM_RESTRICT nodes, int32_t root, const BinnedRow<uint16_t>* r,
                                  int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) {
            n0 = n1 = n2 = n3 = root;
            while ((n0 & n1 & n2 & n3) >= 0) {
                if (n0 >= 0) n0 = ostep<Cat>(nodes[n0], r[0]);
                if (n1 >= 0) n1 = ostep<Cat>(nodes[n1], r[1]);
                if (n2 >= 0) n2 = ostep<Cat>(nodes[n2], r[2]);
                if (n3 >= 0) n3 = ostep<Cat>(nodes[n3], r[3]);
            }
        }

        template <bool Cat>
        LIGNUM_INLINE void owalk4_trees(const ONode* LIGNUM_RESTRICT nodes, const int32_t* roots, const BinnedRow<uint16_t>& r,
                                        int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) {
            n0 = roots[0]; n1 = roots[1]; n2 = roots[2]; n3 = roots[3];
            while ((n0 & n1 & n2 & n3) >= 0) {
                if (n0 >= 0) n0 = ostep<Cat>(nodes[n0], r);
                if (n1 >= 0) n1 = ostep<Cat>(nodes[n1], r);
                if (n2 >= 0) n2 = ostep<Cat>(nodes[n2], r);
                if (n3 >= 0) n3 = ostep<Cat>(nodes[n3], r);
            }
        }

        template <bool Cat>
        LIGNUM_INLINE int32_t owalk1(const ONode* LIGNUM_RESTRICT nodes, int32_t node, const BinnedRow<uint16_t>& r) {
            while (node >= 0) node = ostep<Cat>(nodes[node], r);
            return node;
        }

        // Binned kernel for models with ONode trees (miter): isolated from BinnedKernel so
        // non-oblique models keep their exact codegen and cache footprint.
        template <bool Multi>
        class ObliqueBinnedKernel {
            public:
                ObliqueBinnedKernel(const Model& model, const BinnedForest& forest, const double* X,
                                    size_t n_features, bool raw_score, const ColumnSource* columns = nullptr)
                    : m_(model), f_(forest), columns_(columns), X_(X), n_features_(n_features), raw_score_(raw_score) {}

                void predict_rows(size_t i_begin, size_t i_end, double* LIGNUM_RESTRICT out) const {
                    const size_t count = i_end - i_begin;
                    const size_t stride = f_.row_codes();
                    const size_t nw = f_.decision_words;
                    thread_local std::vector<uint16_t> buffer;
                    thread_local std::vector<uint64_t> decision_buffer;
                    buffer.resize(count * stride);
                    decision_buffer.resize(count * nw);
                    uint16_t* codes = buffer.data();
                    uint64_t* decisions = decision_buffer.data();
                    bin_rows_into<uint16_t, true>(m_, f_, columns_, X_, n_features_, i_begin, i_end, codes, decisions);

                    const bool argmax = !raw_score_ && m_.transform == transform::Argmax;
                    double* LIGNUM_RESTRICT dst = nullptr;
                    if constexpr (Multi) {
                        if (argmax) {
                            thread_local std::vector<double> block_margins;
                            block_margins.resize(count * m_.n_outputs);
                            dst = prepare_block_dst<double, true>(m_, i_begin, count, true, out, block_margins.data());
                        } else {
                            dst = prepare_block_dst<double, true>(m_, i_begin, count, false, out, nullptr);
                        }
                    } else {
                        dst = prepare_block_dst<double, false>(m_, i_begin, count, argmax, out, nullptr);
                    }
                    for (size_t t_start = 0; t_start < f_.n_trees; t_start += T_BLOCK) {
                        accumulate(codes, stride, decisions, nw, count, t_start, std::min(t_start + T_BLOCK, f_.n_trees), dst);
                    }
                    finish_block_dst<double, Multi>(m_, i_begin, count, raw_score_, argmax, dst, out);
                }

            private:
                const Model& m_;
                const BinnedForest& f_;
                const ColumnSource* columns_;
                const double* X_;
                size_t n_features_;
                bool raw_score_;

                LIGNUM_INLINE void add(double* LIGNUM_RESTRICT dst, size_t r, size_t t, int32_t node) const {
                    const size_t slot = Multi ? r * m_.n_outputs + m_.tree_outputs[t] : r;
                    dst[slot] += leaf_value(m_.leaf_values, node);
                }

                void accumulate(const uint16_t* LIGNUM_RESTRICT codes, size_t stride, const uint64_t* LIGNUM_RESTRICT decisions,
                                size_t nw, size_t count, size_t t_begin, size_t t_end, double* LIGNUM_RESTRICT dst) const {
                    static constexpr uint64_t kZeroDecision = 0;
                    const BNode* LIGNUM_RESTRICT nodes = f_.nodes;
                    const ONode* LIGNUM_RESTRICT onodes = f_.onodes;
                    const int32_t* LIGNUM_RESTRICT roots = f_.roots;
                    const uint8_t* LIGNUM_RESTRICT kinds = f_.tree_kind;
                    const double* LIGNUM_RESTRICT leaf_values = m_.leaf_values;
                    auto row = [&](size_t r) LIGNUM_LAMBDA_INLINE {
                        return BinnedRow<uint16_t>{&codes[r * stride], nw > 0 ? &decisions[r * nw] : &kZeroDecision, f_.zero_bin};
                    };
                    auto walk4 = [&](size_t t, const BinnedRow<uint16_t>* r, int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) LIGNUM_LAMBDA_INLINE {
                        const uint8_t k = kinds[t];
                        if ((k & tree_kind::Oblique) != 0) {
                            if ((k & tree_kind::Categorical) != 0) owalk4<true>(onodes, roots[t], r, n0, n1, n2, n3);
                            else                                   owalk4<false>(onodes, roots[t], r, n0, n1, n2, n3);
                        } else {
                            with_kind<true>(k, [&](auto K) LIGNUM_LAMBDA_INLINE { bwalk4<decltype(K)::value>(nodes, roots[t], r, n0, n1, n2, n3); });
                        }
                    };
                    auto walk1 = [&](size_t t, const BinnedRow<uint16_t>& r) LIGNUM_LAMBDA_INLINE {
                        const uint8_t k = kinds[t];
                        int32_t leaf;
                        if ((k & tree_kind::Oblique) != 0) {
                            leaf = (k & tree_kind::Categorical) != 0 ? owalk1<true>(onodes, roots[t], r) : owalk1<false>(onodes, roots[t], r);
                        } else {
                            with_kind<true>(k, [&](auto K) LIGNUM_LAMBDA_INLINE { leaf = bwalk1<decltype(K)::value>(nodes, roots[t], r); });
                        }
                        return leaf;
                    };
                    const size_t unrolled_end = (count / 4) * 4;
                    for (size_t i = 0; i < unrolled_end; i += 4) {
                        const BinnedRow<uint16_t> r[4] = {row(i), row(i + 1), row(i + 2), row(i + 3)};
                        if constexpr (Multi) {
                            for (size_t t = t_begin; t < t_end; t++) {
                                int32_t n0, n1, n2, n3;
                                walk4(t, r, n0, n1, n2, n3);
                                add(dst, i, t, n0); add(dst, i + 1, t, n1); add(dst, i + 2, t, n2); add(dst, i + 3, t, n3);
                            }
                        } else {
                            double sum0 = dst[i], sum1 = dst[i + 1], sum2 = dst[i + 2], sum3 = dst[i + 3];
                            for (size_t t = t_begin; t < t_end; t++) {
                                int32_t n0, n1, n2, n3;
                                walk4(t, r, n0, n1, n2, n3);
                                sum0 += leaf_value(leaf_values, n0);
                                sum1 += leaf_value(leaf_values, n1);
                                sum2 += leaf_value(leaf_values, n2);
                                sum3 += leaf_value(leaf_values, n3);
                            }
                            dst[i] = sum0; dst[i + 1] = sum1; dst[i + 2] = sum2; dst[i + 3] = sum3;
                        }
                    }
                    const size_t trees_unrolled_end = t_begin + ((t_end - t_begin) / 4) * 4;
                    for (size_t i = unrolled_end; i < count; i++) {
                        const BinnedRow<uint16_t> r = row(i);
                        for (size_t t = t_begin; t < trees_unrolled_end; t += 4) {
                            const uint8_t all = static_cast<uint8_t>(kinds[t] & kinds[t + 1] & kinds[t + 2] & kinds[t + 3]);
                            const uint8_t any = static_cast<uint8_t>(kinds[t] | kinds[t + 1] | kinds[t + 2] | kinds[t + 3]);
                            if ((all & tree_kind::Oblique) != 0) {
                                int32_t n0, n1, n2, n3;
                                if ((any & tree_kind::Categorical) != 0) owalk4_trees<true>(onodes, &roots[t], r, n0, n1, n2, n3);
                                else                                     owalk4_trees<false>(onodes, &roots[t], r, n0, n1, n2, n3);
                                add(dst, i, t, n0); add(dst, i, t + 1, n1); add(dst, i, t + 2, n2); add(dst, i, t + 3, n3);
                            } else {
                                for (size_t u = t; u < t + 4; u++) add(dst, i, u, walk1(u, r));
                            }
                        }
                        for (size_t t = trees_unrolled_end; t < t_end; t++) add(dst, i, t, walk1(t, r));
                    }
                }
        };

        // Branch-free symmetric (oblivious) miter kernel: one ONodeSlot condition per tree level
        // builds the leaf index directly without chasing ONode/BNode child pointers.
        template <bool Multi>
        class SymBinnedKernel {
            public:
                SymBinnedKernel(const Model& model, const BinnedForest& forest, const double* X,
                                size_t n_features, bool raw_score, const ColumnSource* columns = nullptr)
                    : m_(model), f_(forest), sym_(*forest.sym), columns_(columns), X_(X),
                      n_features_(n_features), raw_score_(raw_score) {}

                void predict_rows(size_t i_begin, size_t i_end, double* LIGNUM_RESTRICT out) const {
                    const size_t count = i_end - i_begin;
                    const size_t stride = f_.row_codes();
                    const size_t nw = f_.decision_words;
                    thread_local std::vector<uint16_t> buffer;
                    thread_local std::vector<uint64_t> decision_buffer;
                    buffer.resize(count * stride);
                    decision_buffer.resize(count * nw);
                    uint16_t* codes = buffer.data();
                    uint64_t* decisions = decision_buffer.data();
                    bin_rows_into<uint16_t, true>(m_, f_, columns_, X_, n_features_, i_begin, i_end, codes, decisions);

                    const bool argmax = !raw_score_ && m_.transform == transform::Argmax;
                    double* LIGNUM_RESTRICT dst = nullptr;
                    if constexpr (Multi) {
                        if (argmax) {
                            thread_local std::vector<double> block_margins;
                            block_margins.resize(count * m_.n_outputs);
                            dst = prepare_block_dst<double, true>(m_, i_begin, count, true, out, block_margins.data());
                        } else {
                            dst = prepare_block_dst<double, true>(m_, i_begin, count, false, out, nullptr);
                        }
                    } else {
                        dst = prepare_block_dst<double, false>(m_, i_begin, count, argmax, out, nullptr);
                    }
                    for (size_t t_start = 0; t_start < sym_.trees.size(); t_start += T_BLOCK) {
                        accumulate(codes, stride, decisions, nw, count, t_start, std::min(t_start + T_BLOCK, sym_.trees.size()), dst);
                    }
                    finish_block_dst<double, Multi>(m_, i_begin, count, raw_score_, argmax, dst, out);
                }

            private:
                const Model& m_;
                const BinnedForest& f_;
                const SymForest& sym_;
                const ColumnSource* columns_;
                const double* X_;
                size_t n_features_;
                bool raw_score_;

                void accumulate(const uint16_t* LIGNUM_RESTRICT codes, size_t stride, const uint64_t* LIGNUM_RESTRICT decisions,
                                size_t nw, size_t count, size_t t_begin, size_t t_end, double* LIGNUM_RESTRICT dst) const {
                    static constexpr uint64_t kZeroDecision = 0;
                    const SymTree* LIGNUM_RESTRICT trees = sym_.trees.data();
                    const ONodeSlot* LIGNUM_RESTRICT slots = sym_.slots.data();
                    const double* LIGNUM_RESTRICT leaves = sym_.leaves.data();
                    const size_t K = Multi ? m_.n_outputs : 1;
                    auto row = [&](size_t r) LIGNUM_LAMBDA_INLINE {
                        return BinnedRow<uint16_t>{&codes[r * stride], nw > 0 ? &decisions[r * nw] : &kZeroDecision, f_.zero_bin};
                    };

                    const size_t unrolled_end = (count / 4) * 4;
                    for (size_t i = 0; i < unrolled_end; i += 4) {
                        const BinnedRow<uint16_t> r0 = row(i), r1 = row(i + 1), r2 = row(i + 2), r3 = row(i + 3);
                        for (size_t t = t_begin; t < t_end; t++) {
                            const SymTree& st = trees[t];
                            const ONodeSlot* LIGNUM_RESTRICT ts = slots + st.slot_offset;
                            const double* LIGNUM_RESTRICT tl = leaves + st.leaf_offset;
                            auto step4 = [&](auto cat_tag) LIGNUM_LAMBDA_INLINE {
                                constexpr bool Cat = decltype(cat_tag)::value;
                                uint32_t idx0 = 0, idx1 = 0, idx2 = 0, idx3 = 0;
                                for (uint32_t lv = 0; lv < st.depth; lv++) {
                                    const ONodeSlot& sl = ts[lv];
                                    idx0 = (idx0 << 1) | static_cast<uint32_t>(odir<Cat>(sl, r0));
                                    idx1 = (idx1 << 1) | static_cast<uint32_t>(odir<Cat>(sl, r1));
                                    idx2 = (idx2 << 1) | static_cast<uint32_t>(odir<Cat>(sl, r2));
                                    idx3 = (idx3 << 1) | static_cast<uint32_t>(odir<Cat>(sl, r3));
                                }
                                const size_t out_col = Multi ? st.output : 0;
                                dst[i * K + out_col]       += tl[idx0];
                                dst[(i + 1) * K + out_col] += tl[idx1];
                                dst[(i + 2) * K + out_col] += tl[idx2];
                                dst[(i + 3) * K + out_col] += tl[idx3];
                            };
                            if (st.has_cat) step4(std::true_type{});
                            else            step4(std::false_type{});
                        }
                    }
                    for (size_t i = unrolled_end; i < count; i++) {
                        const BinnedRow<uint16_t> r = row(i);
                        for (size_t t = t_begin; t < t_end; t++) {
                            const SymTree& st = trees[t];
                            const ONodeSlot* LIGNUM_RESTRICT ts = slots + st.slot_offset;
                            const double* LIGNUM_RESTRICT tl = leaves + st.leaf_offset;
                            uint32_t idx = 0;
                            if (st.has_cat) {
                                for (uint32_t lv = 0; lv < st.depth; lv++) idx = (idx << 1) | static_cast<uint32_t>(odir<true>(ts[lv], r));
                            } else {
                                for (uint32_t lv = 0; lv < st.depth; lv++) idx = (idx << 1) | static_cast<uint32_t>(odir<false>(ts[lv], r));
                            }
                            const size_t out_col = Multi ? st.output : 0;
                            dst[i * K + out_col] += tl[idx];
                        }
                    }
                }
        };

        template <typename Acc, bool Mixed, bool Multi>
        void run_binned(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                        int n_jobs, Strategy strategy, bool raw_score, const ColumnSource* columns) {
            // Hold a local copy of compiled across the call so decompile() cannot unload it mid-run.
            const auto compiled_guard = n_samples >= model.compiled_min_rows ? get_compiled(model) : nullptr;
            const CompiledForest* compiled = compiled_guard.get();
            const int n_threads = resolve_threads(n_jobs);
            if (strategy == Strategy::Auto) strategy = choose_strategy(n_samples, n_threads);
            const BinnedForest& f = *model.binned;
            if (compiled) {
                const CompiledKernel<Acc, Multi> kernel(model, f, *compiled, X, n_features, raw_score, columns);
                const size_t per_row = f.row_codes() * sizeof(uint16_t) + f.decision_words * sizeof(uint64_t) + model.n_outputs * sizeof(double);
                const size_t rows = std::clamp<size_t>(kCompiledChunkBytes / per_row / 4 * 4, N_BLOCK, 1 << 16);
                if (strategy != Strategy::Serial) {
                    run_rows(kernel, n_samples, out, n_threads, rows);
                    return;
                }
                for (size_t i = 0; i < n_samples; i += rows) kernel.predict_rows(i, std::min(i + rows, n_samples), out);
                return;
            }
            if (f.sym) {
                const SymBinnedKernel<Multi> kernel(model, f, X, n_features, raw_score, columns);
                if (strategy == Strategy::Serial) run_serial(kernel, n_samples, out);
                else                              run_rows(kernel, n_samples, out, n_threads);
                return;
            }
            if (f.n_onodes > 0) {
                const ObliqueBinnedKernel<Multi> kernel(model, f, X, n_features, raw_score, columns);
                if (strategy == Strategy::Serial) run_serial(kernel, n_samples, out);
                else                              run_rows(kernel, n_samples, out, n_threads);
                return;
            }
            auto exec = [&](auto code_tag) {
                using CodeT = decltype(code_tag);
                const BinnedKernel<Acc, Mixed, Multi, CodeT> kernel(model, f, X, n_features, raw_score, columns);
                if (strategy == Strategy::Serial) run_serial(kernel, n_samples, out);
                else                              run_rows(kernel, n_samples, out, n_threads);
            };
            if (f.fits_u8_codes) exec(uint8_t{});
            else                 exec(uint16_t{});
        }

        template <typename Acc, bool Mixed>
        void run_binned_outputs(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                                int n_jobs, Strategy strategy, bool raw_score, const ColumnSource* columns) {
            if (model.n_outputs > 1) run_binned<Acc, Mixed, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score, columns);
            else                     run_binned<Acc, Mixed, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score, columns);
        }

        void run_binned_any(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                            int n_jobs, Strategy strategy, bool raw_score, const ColumnSource* columns = nullptr) {
            if (model.float32) {
                if (model.has_special_trees) run_binned_outputs<float, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score, columns);
                else                         run_binned_outputs<float, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score, columns);
            } else {
                if (model.has_special_trees) run_binned_outputs<double, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score, columns);
                else                         run_binned_outputs<double, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score, columns);
            }
        }

        void dispatch(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                      int n_jobs, Strategy strategy, bool raw_score) {
            if (!model.oblivious && !model.base_scores) {
                throw std::logic_error("Model is empty (not loaded).");
            }
            if (n_features < model.n_features) {
                throw std::invalid_argument("Input has " + std::to_string(n_features) + " features, but the model requires at least " +
                                            std::to_string(model.n_features) + ".");
            }
            if (n_samples == 0) return;

            if (model.oblivious) {
                predict_oblivious(*model.oblivious, X, n_samples, n_features, out, resolve_threads(n_jobs), strategy,
                                  model.transform, model.sigmoid, raw_score);
                return;
            }

            // The binned kernel when the model has one (from LIGNUM_BINNED_MIN_ROWS rows on, by default
            // one), otherwise the direct traversal.
            const bool binned = model.binned && n_samples >= kMinRowsForBinned;
            if (binned) run_binned_any(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
            else        run_any(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
        }

    } // namespace

    void predict_columns(const Model& model, const ColumnSource& source, size_t n_rows, size_t n_columns, double* out,
                         bool raw_score, int n_jobs, Strategy strategy) {
        if (!model.oblivious && !model.base_scores) throw std::logic_error("Model is empty (not loaded).");
        if (!model.binned) throw std::logic_error("predict_columns needs the binned kernel.");
        if (n_columns < model.n_features) {
            throw std::invalid_argument("Input has " + std::to_string(n_columns) + " features, but the model requires at least " +
                                        std::to_string(model.n_features) + ".");
        }
        if (n_rows == 0) return;
        run_binned_any(model, nullptr, n_rows, n_columns, out, n_jobs, strategy, raw_score, &source);
    }

    void Model::predict(const double* LIGNUM_RESTRICT X, size_t n_samples, size_t n_features, double* LIGNUM_RESTRICT out_preds,
                        int n_jobs, Strategy strategy) const {
        dispatch(*this, X, n_samples, n_features, out_preds, n_jobs, strategy, false);
    }

    void Model::predict_raw(const double* LIGNUM_RESTRICT X, size_t n_samples, size_t n_features, double* LIGNUM_RESTRICT out_preds,
                            int n_jobs, Strategy strategy) const {
        dispatch(*this, X, n_samples, n_features, out_preds, n_jobs, strategy, true);
    }

    size_t Model::n_trees() const {
        return oblivious ? oblivious->n_trees() : n_offsets;
    }

    size_t Model::total_leaves() const {
        return oblivious ? oblivious->leaves.size() / std::max<size_t>(1, oblivious->dim) : n_leaves;
    }

} // namespace lignum
