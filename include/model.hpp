#pragma once

#if defined(_MSC_VER) && !defined(__restrict__)
    #define __restrict__ __restrict
#endif

#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>

namespace lignum {

    struct alignas(64) KNode {
        double thresholds[4];
        int32_t children[4];
        uint16_t features[4];
        uint8_t dirs[4];
        uint8_t padding[4];
    };

    class Data {
        public:
            virtual ~Data() = default;
    };

    class VectorData : public Data {
        public:
            std::vector<KNode> k_nodes;
            std::vector<int32_t> tree_offsets;
            std::vector<double> leaf_values;
    };

    class Model {
        public:
            const KNode* k_nodes = nullptr;
            size_t n_nodes = 0;

            const int32_t* tree_offsets = nullptr;
            size_t n_offsets = 0;

            const double* leaf_values = nullptr;
            size_t n_leaves = 0;

            double base_score = 0.0;
            uint8_t transform = 2;

            std::shared_ptr<const Data> data;

            Model() = default;

            void predict(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs = -1) const noexcept;
            void predict_raw(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs = -1) const noexcept;
            void save(const std::string& filepath) const;
        
        private:
            template <bool RawScore>
            void predict_impl(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs) const noexcept;
    };

} // namespace lignum