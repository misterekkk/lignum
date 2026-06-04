#include "model.hpp"

#include <algorithm>
#include <omp.h>

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

        inline int32_t cmp_dir(double x, double threshold, int default_dir) {
            uint64_t bits;
            std::memcpy(&bits, &x, sizeof(bits));
            if ((bits & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL) {
                return default_dir;
            }
            
            return x >= threshold;
}

    } // namespace

    void Model::predict(const double* X, size_t n_samples, size_t n_features, double* out_preds, int n_jobs) const {
        int32_t n_threads = 1;

        #ifdef _OPENMP
            int32_t max_num_threads = omp_get_max_threads();
            if (n_jobs <= 0 || n_jobs > max_num_threads) {
                n_threads = max_num_threads;
            } else {
                n_threads = n_jobs;
            }
        #endif

        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (size_t i_start = 0; i_start < n_samples; i_start += N_BLOCK) {
            size_t i_end = std::min(i_start + N_BLOCK, n_samples);

            for (size_t i = i_start; i < i_end; i++) {
                out_preds[i] = base_score;
            }

            for (size_t t_start = 0; t_start < n_offsets; t_start += T_BLOCK) {
                size_t t_end = std::min(t_start + T_BLOCK, n_offsets);
                size_t unrolled_end = i_start + ((i_end - i_start) / 4) * 4;
                
                for (size_t i = i_start; i < unrolled_end; i += 4) {
                    
                    const double* row0 = &X[i * n_features];
                    const double* row1 = &X[(i + 1) * n_features];
                    const double* row2 = &X[(i + 2) * n_features];
                    const double* row3 = &X[(i + 3) * n_features];

                    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    
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

                                int32_t r = cmp_dir(xr, kn0.thresholds[0], kn0.dirs[0]);
                                int32_t l = cmp_dir(xl, kn0.thresholds[1], kn0.dirs[1]);
                                int32_t c = cmp_dir(xc, kn0.thresholds[2], kn0.dirs[2]);

                                node0 = kn0.children[(r << 1) | (r ? c : l)];
                            }

                            if (node1 >= 0) {
                                const KNode &kn1 = k_nodes[node1];
                                
                                double xr = row1[kn1.features[0]];
                                double xl = row1[kn1.features[1]];
                                double xc = row1[kn1.features[2]];

                                int32_t r = cmp_dir(xr, kn1.thresholds[0], kn1.dirs[0]);
                                int32_t l = cmp_dir(xl, kn1.thresholds[1], kn1.dirs[1]);
                                int32_t c = cmp_dir(xc, kn1.thresholds[2], kn1.dirs[2]);

                                node1 = kn1.children[(r << 1) | (r ? c : l)];
                            }

                            if (node2 >= 0) {
                                const KNode &kn2 = k_nodes[node2];
                                
                                double xr = row2[kn2.features[0]];
                                double xl = row2[kn2.features[1]];
                                double xc = row2[kn2.features[2]];

                                int32_t r = cmp_dir(xr, kn2.thresholds[0], kn2.dirs[0]);
                                int32_t l = cmp_dir(xl, kn2.thresholds[1], kn2.dirs[1]);
                                int32_t c = cmp_dir(xc, kn2.thresholds[2], kn2.dirs[2]);

                                node2 = kn2.children[(r << 1) | (r ? c : l)];
                            }

                            if (node3 >= 0) {
                                const KNode &kn3 = k_nodes[node3];
                                
                                double xr = row3[kn3.features[0]];
                                double xl = row3[kn3.features[1]];
                                double xc = row3[kn3.features[2]];

                                int32_t r = cmp_dir(xr, kn3.thresholds[0], kn3.dirs[0]);
                                int32_t l = cmp_dir(xl, kn3.thresholds[1], kn3.dirs[1]);
                                int32_t c = cmp_dir(xc, kn3.thresholds[2], kn3.dirs[2]);

                                node3 = kn3.children[(r << 1) | (r ? c : l)];
                            }
                        }
                        sum0 += leaf_values[(-node0) - 1];
                        sum1 += leaf_values[(-node1) - 1];
                        sum2 += leaf_values[(-node2) - 1];
                        sum3 += leaf_values[(-node3) - 1];
                    }
                    out_preds[i] += sum0;
                    out_preds[i + 1] += sum1;
                    out_preds[i + 2] += sum2;
                    out_preds[i + 3] += sum3;
                }

                for (size_t i = unrolled_end; i < i_end; i++) {
                    const double* row = &X[i * n_features];
                    double sum = 0.0;
                    for (size_t t = t_start; t < t_end; t++) {
                        int32_t node = tree_offsets[t];
                        while (node >= 0) {
                            const KNode &kn = k_nodes[node];
                            
                            double xr = row[kn.features[0]];
                            double xl = row[kn.features[1]];
                            double xc = row[kn.features[2]];

                            int32_t r = cmp_dir(xr, kn.thresholds[0], kn.dirs[0]);
                            int32_t l = cmp_dir(xl, kn.thresholds[1], kn.dirs[1]);
                            int32_t c = cmp_dir(xc, kn.thresholds[2], kn.dirs[2]);
                            
                            node = kn.children[(r << 1) | (r ? c : l)];
                        }
                        sum += leaf_values[(-node) - 1];
                    }
                    out_preds[i] += sum;
                }
            }
        }
    }

} // namespace lignum