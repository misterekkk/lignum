#include <lignum/model.hpp>

#include <algorithm>
#include <bit>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _MSC_VER
    using omp_iterator_t = ptrdiff_t;
#else
    using omp_iterator_t = size_t;
#endif

#ifndef CACHE_TILLING_BLOCK_SIZE
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

        constexpr double kZeroThreshold = static_cast<double>(1e-35f);

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
        inline double apply_transform(double x, uint8_t transform, double sigmoid) {
            switch (transform) {
                case transform::Sigmoid:  return 1.0 / (1.0 + std::exp(-sigmoid * x));
                case transform::Exp:      return std::exp(x);
                case transform::Log1pExp: return std::log1p(std::exp(x));
                default:                  return x;
            }
        }

        inline float apply_transform(float x, uint8_t transform) {
            switch (transform) {
                case transform::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
                case transform::Exp:     return std::exp(x);
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
        LIGNUM_INLINE int32_t cat_dir(double x, double encoded, uint8_t kind, uint8_t dirs, const uint32_t* __restrict__ bitsets) {
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
        LIGNUM_INLINE int32_t slot_dir(const KNode& kn, int k, const double* __restrict__ row, const uint32_t* __restrict__ bitsets) {
            const double x = row[kn.features[k]];
            if constexpr ((Kind & tree_kind::Categorical) != 0) {
                if (kn.kinds[k] != slot_kind::Numeric) return cat_dir(x, kn.thresholds[k], kn.kinds[k], kn.dirs[k], bitsets);
            }
            return cmp_dir<(Kind & tree_kind::ZeroMissing) != 0>(x, kn.thresholds[k], kn.dirs[k]);
        }

        // One KNode = two tree levels. All three comparisons are evaluated, then the
        // grandchild is picked without a dependency between the two levels. `Kind` is the
        // tree_kind of the tree, so numeric trees compile to plain threshold comparisons.
        template <uint8_t Kind>
        LIGNUM_INLINE int32_t step(const KNode& kn, const double* __restrict__ row, const uint32_t* __restrict__ bitsets) {
        #ifdef LIGNUM_BRANCHY
            // Experimental: branch on the root comparison like compiled code, so the CPU can
            // speculate down the predicted path; only the taken child's comparison is evaluated.
            if (slot_dir<Kind>(kn, 0, row, bitsets)) {
                return kn.children[2 | slot_dir<Kind>(kn, 2, row, bitsets)];
            }
            return kn.children[slot_dir<Kind>(kn, 1, row, bitsets)];
        #else
            int32_t r = slot_dir<Kind>(kn, 0, row, bitsets);
            int32_t l = slot_dir<Kind>(kn, 1, row, bitsets);
            int32_t c = slot_dir<Kind>(kn, 2, row, bitsets);
            return kn.children[(r << 1) | (r ? c : l)];
        #endif
        }

        struct Forest {
            const KNode* __restrict__ nodes;
            const uint32_t* __restrict__ bitsets;
        };

        // Four rows walk the same tree interleaved to hide memory latency.
        template <uint8_t Kind>
        LIGNUM_INLINE void walk4(Forest f, int32_t root,
                                 const double* r0, const double* r1, const double* r2, const double* r3,
                                 int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) {
            n0 = n1 = n2 = n3 = root;
            while (n0 >= 0 || n1 >= 0 || n2 >= 0 || n3 >= 0) {
                if (n0 >= 0) n0 = step<Kind>(f.nodes[n0], r0, f.bitsets);
                if (n1 >= 0) n1 = step<Kind>(f.nodes[n1], r1, f.bitsets);
                if (n2 >= 0) n2 = step<Kind>(f.nodes[n2], r2, f.bitsets);
                if (n3 >= 0) n3 = step<Kind>(f.nodes[n3], r3, f.bitsets);
            }
        }

        // Four trees of the same row, interleaved (used when there are fewer than four rows).
        template <uint8_t Kind>
        LIGNUM_INLINE void walk4_trees(Forest f, const int32_t* roots, const double* row,
                                       int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) {
            n0 = roots[0]; n1 = roots[1]; n2 = roots[2]; n3 = roots[3];
            while (n0 >= 0 || n1 >= 0 || n2 >= 0 || n3 >= 0) {
                if (n0 >= 0) n0 = step<Kind>(f.nodes[n0], row, f.bitsets);
                if (n1 >= 0) n1 = step<Kind>(f.nodes[n1], row, f.bitsets);
                if (n2 >= 0) n2 = step<Kind>(f.nodes[n2], row, f.bitsets);
                if (n3 >= 0) n3 = step<Kind>(f.nodes[n3], row, f.bitsets);
            }
        }

        template <uint8_t Kind>
        LIGNUM_INLINE int32_t walk1(Forest f, int32_t node, const double* row) {
            while (node >= 0) node = step<Kind>(f.nodes[node], row, f.bitsets);
            return node;
        }

        // Calls fn(std::integral_constant<uint8_t, kind>) so the traversal is specialised per tree
        // kind. Models with only numeric trees (Mixed = false) never branch.
        template <bool Mixed, typename Fn>
        LIGNUM_INLINE void with_kind(uint8_t kind, Fn&& fn) {
            using K0 = std::integral_constant<uint8_t, 0>;
            if constexpr (!Mixed) {
                fn(K0{});
            } else {
                switch (kind & 3) {
                    case 0:  fn(K0{}); break;
                    case 1:  fn(std::integral_constant<uint8_t, 1>{}); break;
                    case 2:  fn(std::integral_constant<uint8_t, 2>{}); break;
                    default: fn(std::integral_constant<uint8_t, 3>{}); break;
                }
            }
        }

        LIGNUM_INLINE double leaf_value(const double* __restrict__ leaves, int32_t node) {
            return leaves[(-node) - 1];
        }

        // Tree traversal shared by all strategies. Leaves are always added to a row's running
        // sum (kept in out_preds between tiles) in model order, exactly like the source library,
        // so every strategy produces bit-identical results.
        //
        // A tree-parallel strategy (threads fill a per-tree leaf buffer, rows sum it in order) was
        // measured and lost to Serial below 8 rows and to Rows above, so it is not offered.
        // `Mixed`: the model has trees that are not plain numeric (zero-as-missing or categorical).
        // Only then does the kernel branch per tree; numeric trees always take the plain traversal.
        // `Multi`: the model has several outputs (classes). Row i's outputs live in
        // out[i * K .. i * K + K), and each tree adds to its own output, still in model order.
        template <typename Acc, bool Mixed, bool Multi>
        class Kernel {
            public:
                Kernel(const Model& model, const double* X, size_t n_features, bool raw_score)
                    : m_(model), X_(X), n_features_(n_features), raw_score_(raw_score) {}

                // Rows [i_begin, i_end): base scores, all trees in tiles of T_BLOCK, transform.
                void predict_rows(size_t i_begin, size_t i_end, double* __restrict__ out) const {
                    if constexpr (Multi) {
                        const size_t K = m_.n_outputs;
                        for (size_t i = i_begin; i < i_end; i++) {
                            for (size_t c = 0; c < K; c++) out[i * K + c] = static_cast<double>(static_cast<Acc>(m_.base_scores[c]));
                        }
                    } else {
                        const double base = static_cast<double>(static_cast<Acc>(m_.base_scores[0]));
                        for (size_t i = i_begin; i < i_end; i++) out[i] = base;
                    }
                    for (size_t t_start = 0; t_start < m_.n_offsets; t_start += T_BLOCK) {
                        const size_t t_end = std::min(t_start + T_BLOCK, m_.n_offsets);
                        if constexpr (Multi) accumulate_multi(i_begin, i_end, t_start, t_end, out);
                        else                 accumulate(i_begin, i_end, t_start, t_end, out);
                    }
                    finish(i_begin, i_end, out);
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

                Forest forest() const { return Forest{m_.k_nodes, m_.cat_bitsets}; }

                const double* row(size_t i) const { return &X_[i * n_features_]; }

                // Adds trees [t_begin, t_end) to the running sums of rows [i_begin, i_end).
                void accumulate(size_t i_begin, size_t i_end, size_t t_begin, size_t t_end, double* __restrict__ out) const {
                    const Forest f = forest();
                    const int32_t* __restrict__ roots = m_.tree_offsets;
                    const uint8_t* __restrict__ kinds = m_.tree_kinds;
                    const double* __restrict__ leaf_values = m_.leaf_values;
                    const size_t unrolled_end = i_begin + ((i_end - i_begin) / 4) * 4;
                    for (size_t i = i_begin; i < unrolled_end; i += 4) {
                        const double *r0 = row(i), *r1 = row(i + 1), *r2 = row(i + 2), *r3 = row(i + 3);
                        Acc sum0 = static_cast<Acc>(out[i]);
                        Acc sum1 = static_cast<Acc>(out[i + 1]);
                        Acc sum2 = static_cast<Acc>(out[i + 2]);
                        Acc sum3 = static_cast<Acc>(out[i + 3]);
                        for (size_t t = t_begin; t < t_end; t++) {
                            int32_t n0, n1, n2, n3;
                            with_kind<Mixed>(kind_of(kinds, t, 1), [&](auto K) LIGNUM_LAMBDA_INLINE {
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
                            with_kind<Mixed>(kind_of(kinds, t, 4), [&](auto K) LIGNUM_LAMBDA_INLINE {
                                walk4_trees<decltype(K)::value>(f, &roots[t], r, n0, n1, n2, n3);
                            });
                            sum += static_cast<Acc>(leaf_value(leaf_values, n0));
                            sum += static_cast<Acc>(leaf_value(leaf_values, n1));
                            sum += static_cast<Acc>(leaf_value(leaf_values, n2));
                            sum += static_cast<Acc>(leaf_value(leaf_values, n3));
                        }
                        for (size_t t = trees_unrolled_end; t < t_end; t++) {
                            int32_t leaf;
                            with_kind<Mixed>(kind_of(kinds, t, 1), [&](auto K) LIGNUM_LAMBDA_INLINE { leaf = walk1<decltype(K)::value>(f, roots[t], r); });
                            sum += static_cast<Acc>(leaf_value(leaf_values, leaf));
                        }
                        out[i] = static_cast<double>(sum);
                    }
                }

                // Multi-output version of accumulate(): each leaf goes to its tree's output.
                void accumulate_multi(size_t i_begin, size_t i_end, size_t t_begin, size_t t_end, double* __restrict__ out) const {
                    const Forest f = forest();
                    const int32_t* __restrict__ roots = m_.tree_offsets;
                    const uint8_t* __restrict__ kinds = m_.tree_kinds;
                    const uint16_t* __restrict__ outputs = m_.tree_outputs;
                    const double* __restrict__ leaf_values = m_.leaf_values;
                    const size_t K = m_.n_outputs;

                    auto add = [&](size_t i, size_t t, int32_t node) LIGNUM_LAMBDA_INLINE {
                        double& slot = out[i * K + outputs[t]];
                        slot = static_cast<double>(static_cast<Acc>(slot) + static_cast<Acc>(leaf_value(leaf_values, node)));
                    };

                    const size_t unrolled_end = i_begin + ((i_end - i_begin) / 4) * 4;
                    for (size_t i = i_begin; i < unrolled_end; i += 4) {
                        const double *r0 = row(i), *r1 = row(i + 1), *r2 = row(i + 2), *r3 = row(i + 3);
                        for (size_t t = t_begin; t < t_end; t++) {
                            int32_t n0, n1, n2, n3;
                            with_kind<Mixed>(kind_of(kinds, t, 1), [&](auto K) LIGNUM_LAMBDA_INLINE {
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
                            with_kind<Mixed>(kind_of(kinds, t, 4), [&](auto K) LIGNUM_LAMBDA_INLINE {
                                walk4_trees<decltype(K)::value>(f, &roots[t], r, n0, n1, n2, n3);
                            });
                            add(i, t, n0);
                            add(i, t + 1, n1);
                            add(i, t + 2, n2);
                            add(i, t + 3, n3);
                        }
                        for (size_t t = trees_unrolled_end; t < t_end; t++) {
                            int32_t leaf;
                            with_kind<Mixed>(kind_of(kinds, t, 1), [&](auto K) LIGNUM_LAMBDA_INLINE { leaf = walk1<decltype(K)::value>(f, roots[t], r); });
                            add(i, t, leaf);
                        }
                    }
                }

                void finish(size_t i_begin, size_t i_end, double* __restrict__ out) const {
                    const uint8_t tr = m_.transform;
                    if (raw_score_ || tr == transform::Identity || tr == transform::Argmax) return;

                    const size_t K = Multi ? m_.n_outputs : 1;
                    if (Multi && tr == transform::Softmax) {
                        for (size_t i = i_begin; i < i_end; i++) softmax(&out[i * K], K);
                        return;
                    }
                    for (size_t j = i_begin * K; j < i_end * K; j++) {
                        if constexpr (std::is_same_v<Acc, float>) {
                            out[j] = static_cast<double>(apply_transform(static_cast<float>(out[j]), tr));
                        } else {
                            out[j] = apply_transform(out[j], tr, m_.sigmoid);
                        }
                    }
                }

                // LightGBM: everything in double. XGBoost: max and exp in float, the sum in
                // double, the division by float(sum) in float (verified bit-exact).
                static void softmax(double* v, size_t K) {
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
        };

        template <typename K>
        void run_serial(const K& kernel, size_t n_samples, double* out) {
            for (size_t i_start = 0; i_start < n_samples; i_start += N_BLOCK) {
                kernel.predict_rows(i_start, std::min<size_t>(i_start + N_BLOCK, n_samples), out);
            }
        }

        template <typename K>
        void run_rows(const K& kernel, size_t n_samples, double* out, int n_threads) {
            // Enough blocks for every thread (two each, for balance across P/E cores), but never
            // larger than N_BLOCK, which keeps a tile of trees hot in cache for the whole block.
            const size_t per_thread = (n_samples + 2 * n_threads - 1) / (2 * n_threads);
            const size_t block = std::clamp<size_t>(round_up(per_thread, 4), 4, N_BLOCK);

            #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
            for (omp_iterator_t i_start = 0; i_start < static_cast<omp_iterator_t>(n_samples); i_start += block) {
                kernel.predict_rows(i_start, std::min<size_t>(i_start + block, n_samples), out);
            }
        }

        // Below ~8 rows waking the thread pool costs more than the work (measured on an M4:
        // 4 rows 27 us serial vs 36 us parallel; 8 rows 47 us vs 40 us).
        constexpr size_t kMinRowsForThreads = 8;

        Strategy choose_strategy(size_t n_samples, int n_threads) {
            if (n_threads <= 1 || n_samples < kMinRowsForThreads) return Strategy::Serial;
            return Strategy::Rows;
        }

        template <typename Acc, bool Mixed, bool Multi>
        void run(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                 int n_jobs, Strategy strategy, bool raw_score) {
            const Kernel<Acc, Mixed, Multi> kernel(model, X, n_features, raw_score);
            const int n_threads = resolve_threads(n_jobs);
            if (strategy == Strategy::Auto) strategy = choose_strategy(n_samples, n_threads);

            switch (strategy) {
                case Strategy::Serial: run_serial(kernel, n_samples, out); break;
                default:               run_rows(kernel, n_samples, out, n_threads); break;
            }
        }

        template <typename Acc, bool Mixed>
        void run_outputs(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                         int n_jobs, Strategy strategy, bool raw_score) {
            if (model.n_outputs > 1) run<Acc, Mixed, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
            else                     run<Acc, Mixed, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
        }

        void run_any(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                     int n_jobs, Strategy strategy, bool raw_score) {
            if (model.float32) {
                if (model.has_special_trees) run_outputs<float, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
                else                        run_outputs<float, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
            } else {
                if (model.has_special_trees) run_outputs<double, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
                else                        run_outputs<double, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
            }
        }

        void dispatch(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                      int n_jobs, Strategy strategy, bool raw_score) {
            if (n_features < model.n_features) {
                throw std::invalid_argument("Input has " + std::to_string(n_features) + " features, but the model requires at least " +
                                            std::to_string(model.n_features) + ".");
            }
            if (n_samples == 0) return;

            if (raw_score || model.transform != transform::Argmax) {
                run_any(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
                return;
            }

            // Argmax narrows K margins to one column, so it cannot be done in place.
            const size_t K = model.n_outputs;
            std::vector<double> margins(n_samples * K);
            run_any(model, X, n_samples, n_features, margins.data(), n_jobs, strategy, true);
            for (size_t i = 0; i < n_samples; i++) {
                const double* row = &margins[i * K];
                out[i] = static_cast<double>(std::max_element(row, row + K) - row);
            }
        }

    } // namespace

    void Model::predict(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds,
                        int n_jobs, Strategy strategy) const {
        dispatch(*this, X, n_samples, n_features, out_preds, n_jobs, strategy, false);
    }

    void Model::predict_raw(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds,
                            int n_jobs, Strategy strategy) const {
        dispatch(*this, X, n_samples, n_features, out_preds, n_jobs, strategy, true);
    }

} // namespace lignum
