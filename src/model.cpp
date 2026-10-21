#include <lignum/model.hpp>

#include <algorithm>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

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
#else
    #define LIGNUM_INLINE inline __attribute__((always_inline))
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

        // One KNode = two tree levels. All three comparisons are evaluated, then the
        // grandchild is picked without a dependency between the two levels.
        template <bool ZeroMissing>
        LIGNUM_INLINE int32_t step(const KNode& kn, const double* __restrict__ row) {
            int32_t r = cmp_dir<ZeroMissing>(row[kn.features[0]], kn.thresholds[0], kn.dirs[0]);
            int32_t l = cmp_dir<ZeroMissing>(row[kn.features[1]], kn.thresholds[1], kn.dirs[1]);
            int32_t c = cmp_dir<ZeroMissing>(row[kn.features[2]], kn.thresholds[2], kn.dirs[2]);
            return kn.children[(r << 1) | (r ? c : l)];
        }

        // Four rows walk the same tree interleaved to hide memory latency.
        template <bool ZeroMissing>
        LIGNUM_INLINE void walk4(const KNode* __restrict__ nodes, int32_t root,
                                 const double* r0, const double* r1, const double* r2, const double* r3,
                                 int32_t& n0, int32_t& n1, int32_t& n2, int32_t& n3) {
            n0 = n1 = n2 = n3 = root;
            while (n0 >= 0 || n1 >= 0 || n2 >= 0 || n3 >= 0) {
                if (n0 >= 0) n0 = step<ZeroMissing>(nodes[n0], r0);
                if (n1 >= 0) n1 = step<ZeroMissing>(nodes[n1], r1);
                if (n2 >= 0) n2 = step<ZeroMissing>(nodes[n2], r2);
                if (n3 >= 0) n3 = step<ZeroMissing>(nodes[n3], r3);
            }
        }

        template <bool ZeroMissing>
        LIGNUM_INLINE int32_t walk1(const KNode* __restrict__ nodes, int32_t node, const double* row) {
            while (node >= 0) node = step<ZeroMissing>(nodes[node], row);
            return node;
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
        template <typename Acc, bool ZeroMissing>
        class Kernel {
            public:
                Kernel(const Model& model, const double* X, size_t n_features, bool raw_score)
                    : m_(model), X_(X), n_features_(n_features), raw_score_(raw_score) {}

                // Rows [i_begin, i_end): base score, all trees in tiles of T_BLOCK, transform.
                void predict_rows(size_t i_begin, size_t i_end, double* __restrict__ out) const {
                    for (size_t i = i_begin; i < i_end; i++) out[i] = static_cast<double>(base());
                    for (size_t t_start = 0; t_start < m_.n_offsets; t_start += T_BLOCK) {
                        accumulate(i_begin, i_end, t_start, std::min(t_start + T_BLOCK, m_.n_offsets), out);
                    }
                    finish(i_begin, i_end, out);
                }

            private:
                const Model& m_;
                const double* X_;
                size_t n_features_;
                bool raw_score_;

                Acc base() const { return static_cast<Acc>(m_.base_score); }
                const double* row(size_t i) const { return &X_[i * n_features_]; }

                // Adds trees [t_begin, t_end) to the running sums of rows [i_begin, i_end).
                void accumulate(size_t i_begin, size_t i_end, size_t t_begin, size_t t_end, double* __restrict__ out) const {
                    const KNode* __restrict__ nodes = m_.k_nodes;
                    const int32_t* __restrict__ roots = m_.tree_offsets;
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
                            walk4<ZeroMissing>(nodes, roots[t], r0, r1, r2, r3, n0, n1, n2, n3);
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
                            int32_t n0 = roots[t], n1 = roots[t + 1], n2 = roots[t + 2], n3 = roots[t + 3];
                            while (n0 >= 0 || n1 >= 0 || n2 >= 0 || n3 >= 0) {
                                if (n0 >= 0) n0 = step<ZeroMissing>(nodes[n0], r);
                                if (n1 >= 0) n1 = step<ZeroMissing>(nodes[n1], r);
                                if (n2 >= 0) n2 = step<ZeroMissing>(nodes[n2], r);
                                if (n3 >= 0) n3 = step<ZeroMissing>(nodes[n3], r);
                            }
                            sum += static_cast<Acc>(leaf_value(leaf_values, n0));
                            sum += static_cast<Acc>(leaf_value(leaf_values, n1));
                            sum += static_cast<Acc>(leaf_value(leaf_values, n2));
                            sum += static_cast<Acc>(leaf_value(leaf_values, n3));
                        }
                        for (size_t t = trees_unrolled_end; t < t_end; t++) {
                            sum += static_cast<Acc>(leaf_value(leaf_values, walk1<ZeroMissing>(nodes, roots[t], r)));
                        }
                        out[i] = static_cast<double>(sum);
                    }
                }

                void finish(size_t i_begin, size_t i_end, double* __restrict__ out) const {
                    if (raw_score_ || m_.transform == transform::Identity) return;
                    for (size_t i = i_begin; i < i_end; i++) {
                        if constexpr (std::is_same_v<Acc, float>) {
                            out[i] = static_cast<double>(apply_transform(static_cast<float>(out[i]), m_.transform));
                        } else {
                            out[i] = apply_transform(out[i], m_.transform, m_.sigmoid);
                        }
                    }
                }
        };

        template <typename Acc, bool ZeroMissing>
        void run_serial(const Kernel<Acc, ZeroMissing>& kernel, size_t n_samples, double* out) {
            for (size_t i_start = 0; i_start < n_samples; i_start += N_BLOCK) {
                kernel.predict_rows(i_start, std::min<size_t>(i_start + N_BLOCK, n_samples), out);
            }
        }

        template <typename Acc, bool ZeroMissing>
        void run_rows(const Kernel<Acc, ZeroMissing>& kernel, size_t n_samples, double* out, int n_threads) {
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

        template <typename Acc, bool ZeroMissing>
        void run(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                 int n_jobs, Strategy strategy, bool raw_score) {
            const Kernel<Acc, ZeroMissing> kernel(model, X, n_features, raw_score);
            const int n_threads = resolve_threads(n_jobs);
            if (strategy == Strategy::Auto) strategy = choose_strategy(n_samples, n_threads);

            switch (strategy) {
                case Strategy::Serial: run_serial(kernel, n_samples, out); break;
                default:               run_rows(kernel, n_samples, out, n_threads); break;
            }
        }

        void dispatch(const Model& model, const double* X, size_t n_samples, size_t n_features, double* out,
                      int n_jobs, Strategy strategy, bool raw_score) {
            if (n_features < model.n_features) {
                throw std::invalid_argument("Input has " + std::to_string(n_features) + " features, but the model requires at least " +
                                            std::to_string(model.n_features) + ".");
            }
            if (n_samples == 0) return;

            if (model.float32) {
                if (model.has_zero_missing) run<float, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
                else                        run<float, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
            } else {
                if (model.has_zero_missing) run<double, true>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
                else                        run<double, false>(model, X, n_samples, n_features, out, n_jobs, strategy, raw_score);
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
