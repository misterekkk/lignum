#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>

namespace lignum {

    struct alignas(64) KNode {
        double thresholds[4];
        int children[4];
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
            std::vector<int> tree_offsets;
            std::vector<double> leaf_values;
    };

    class Model {
        public:
            const KNode* k_nodes = nullptr;
            size_t n_nodes = 0;

            const int* tree_offsets = nullptr;
            size_t n_offsets = 0;

            const double* leaf_values = nullptr;
            size_t n_leaves = 0;

            double base_score = 0.0;

            std::shared_ptr<const Data> data;

            Model() = default;

            void predict(const double* X, size_t n_samples, size_t n_features, double* out_preds) const;
            void save(const std::string& filepath) const;
    };

} // namespace lignum