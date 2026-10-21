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

    } // namespace

    template <typename Acc, bool ZeroMissing, bool RawScore>
    void Model::predict_impl(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs) const noexcept {

        int32_t n_threads = 1;
        #ifdef _OPENMP
            int32_t max_num_threads = omp_get_max_threads();
            if (n_jobs <= 0 || n_jobs > max_num_threads) {
                n_threads = max_num_threads;
            } else {
                n_threads = n_jobs;
            }
        #endif
        // Leaves are summed onto out_preds tree by tree, in model order, exactly like the
        // source library; tiling only changes which rows/trees are visited when.
        const Acc base = static_cast<Acc>(base_score);

        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (omp_iterator_t i_start = 0; i_start < static_cast<omp_iterator_t>(n_samples); i_start += N_BLOCK) {
            size_t i_end = std::min<size_t>(i_start + N_BLOCK, n_samples);

            for (size_t i = i_start; i < i_end; i++) {
                out_preds[i] = static_cast<double>(base);
            }

            for (size_t t_start = 0; t_start < n_offsets; t_start += T_BLOCK) {
                size_t t_end = std::min(t_start + T_BLOCK, n_offsets);
                size_t unrolled_end = i_start + ((i_end - i_start) / 4) * 4;

                for (size_t i = i_start; i < unrolled_end; i += 4) {

                    const double* row0 = &X[i * n_features];
                    const double* row1 = &X[(i + 1) * n_features];
                    const double* row2 = &X[(i + 2) * n_features];
                    const double* row3 = &X[(i + 3) * n_features];

                    Acc sum0 = static_cast<Acc>(out_preds[i]);
                    Acc sum1 = static_cast<Acc>(out_preds[i + 1]);
                    Acc sum2 = static_cast<Acc>(out_preds[i + 2]);
                    Acc sum3 = static_cast<Acc>(out_preds[i + 3]);

                    for (size_t t = t_start; t < t_end; t++) {
                        int32_t node0 = tree_offsets[t];
                        int32_t node1 = tree_offsets[t];
                        int32_t node2 = tree_offsets[t];
                        int32_t node3 = tree_offsets[t];

                        while (node0 >= 0 || node1 >= 0 || node2 >= 0 || node3 >= 0) {
                            if (node0 >= 0) {
                                const KNode &kn0 = k_nodes[node0];

                                double xr = row0[kn0.features[0]];
                                double xl = row0[kn0.features[1]];
                                double xc = row0[kn0.features[2]];

                                int32_t r = cmp_dir<ZeroMissing>(xr, kn0.thresholds[0], kn0.dirs[0]);
                                int32_t l = cmp_dir<ZeroMissing>(xl, kn0.thresholds[1], kn0.dirs[1]);
                                int32_t c = cmp_dir<ZeroMissing>(xc, kn0.thresholds[2], kn0.dirs[2]);

                                node0 = kn0.children[(r << 1) | (r ? c : l)];
                            }

                            if (node1 >= 0) {
                                const KNode &kn1 = k_nodes[node1];

                                double xr = row1[kn1.features[0]];
                                double xl = row1[kn1.features[1]];
                                double xc = row1[kn1.features[2]];

                                int32_t r = cmp_dir<ZeroMissing>(xr, kn1.thresholds[0], kn1.dirs[0]);
                                int32_t l = cmp_dir<ZeroMissing>(xl, kn1.thresholds[1], kn1.dirs[1]);
                                int32_t c = cmp_dir<ZeroMissing>(xc, kn1.thresholds[2], kn1.dirs[2]);

                                node1 = kn1.children[(r << 1) | (r ? c : l)];
                            }

                            if (node2 >= 0) {
                                const KNode &kn2 = k_nodes[node2];

                                double xr = row2[kn2.features[0]];
                                double xl = row2[kn2.features[1]];
                                double xc = row2[kn2.features[2]];

                                int32_t r = cmp_dir<ZeroMissing>(xr, kn2.thresholds[0], kn2.dirs[0]);
                                int32_t l = cmp_dir<ZeroMissing>(xl, kn2.thresholds[1], kn2.dirs[1]);
                                int32_t c = cmp_dir<ZeroMissing>(xc, kn2.thresholds[2], kn2.dirs[2]);

                                node2 = kn2.children[(r << 1) | (r ? c : l)];
                            }

                            if (node3 >= 0) {
                                const KNode &kn3 = k_nodes[node3];

                                double xr = row3[kn3.features[0]];
                                double xl = row3[kn3.features[1]];
                                double xc = row3[kn3.features[2]];

                                int32_t r = cmp_dir<ZeroMissing>(xr, kn3.thresholds[0], kn3.dirs[0]);
                                int32_t l = cmp_dir<ZeroMissing>(xl, kn3.thresholds[1], kn3.dirs[1]);
                                int32_t c = cmp_dir<ZeroMissing>(xc, kn3.thresholds[2], kn3.dirs[2]);

                                node3 = kn3.children[(r << 1) | (r ? c : l)];
                            }
                        }
                        sum0 += static_cast<Acc>(leaf_values[(-node0) - 1]);
                        sum1 += static_cast<Acc>(leaf_values[(-node1) - 1]);
                        sum2 += static_cast<Acc>(leaf_values[(-node2) - 1]);
                        sum3 += static_cast<Acc>(leaf_values[(-node3) - 1]);
                    }
                    out_preds[i] = static_cast<double>(sum0);
                    out_preds[i + 1] = static_cast<double>(sum1);
                    out_preds[i + 2] = static_cast<double>(sum2);
                    out_preds[i + 3] = static_cast<double>(sum3);
                }

                for (size_t i = unrolled_end; i < i_end; i++) {
                    const double* row = &X[i * n_features];
                    Acc sum = static_cast<Acc>(out_preds[i]);
                    for (size_t t = t_start; t < t_end; t++) {
                        int32_t node = tree_offsets[t];
                        while (node >= 0) {
                            const KNode &kn = k_nodes[node];

                            double xr = row[kn.features[0]];
                            double xl = row[kn.features[1]];
                            double xc = row[kn.features[2]];

                            int32_t r = cmp_dir<ZeroMissing>(xr, kn.thresholds[0], kn.dirs[0]);
                            int32_t l = cmp_dir<ZeroMissing>(xl, kn.thresholds[1], kn.dirs[1]);
                            int32_t c = cmp_dir<ZeroMissing>(xc, kn.thresholds[2], kn.dirs[2]);

                            node = kn.children[(r << 1) | (r ? c : l)];
                        }
                        sum += static_cast<Acc>(leaf_values[(-node) - 1]);
                    }
                    out_preds[i] = static_cast<double>(sum);
                }
            }
            if constexpr (!RawScore) {
                if (transform != transform::Identity) {
                    for (size_t i = i_start; i < i_end; i++) {
                        if constexpr (std::is_same_v<Acc, float>) {
                            out_preds[i] = static_cast<double>(apply_transform(static_cast<float>(out_preds[i]), transform));
                        } else {
                            out_preds[i] = apply_transform(out_preds[i], transform, sigmoid);
                        }
                    }
                }
            }
        }
    }

    template <bool RawScore>
    void Model::dispatch(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs) const {
        if (n_features < this->n_features) {
            throw std::invalid_argument("Input has " + std::to_string(n_features) + " features, but the model requires at least " +
                                        std::to_string(this->n_features) + ".");
        }
        if (n_samples == 0) return;

        if (float32) {
            if (has_zero_missing) predict_impl<float, true, RawScore>(X, n_samples, n_features, out_preds, n_jobs);
            else                  predict_impl<float, false, RawScore>(X, n_samples, n_features, out_preds, n_jobs);
        } else {
            if (has_zero_missing) predict_impl<double, true, RawScore>(X, n_samples, n_features, out_preds, n_jobs);
            else                  predict_impl<double, false, RawScore>(X, n_samples, n_features, out_preds, n_jobs);
        }
    }

    void Model::predict(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs) const {
        dispatch<false>(X, n_samples, n_features, out_preds, n_jobs);
    }

    void Model::predict_raw(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs) const {
        dispatch<true>(X, n_samples, n_features, out_preds, n_jobs);
    }

} // namespace lignum
