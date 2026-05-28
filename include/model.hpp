#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace lignum {

    struct alignas(64) KNode {
        double thresholds[4];
        int children[4];
        uint16_t features[4];
        uint8_t dirs[4];
        uint8_t padding[4];
    };

    class Model {
        public:
            std::vector<KNode> k_nodes;
            std::vector<int> tree_offsets;
            std::vector<double> leaf_values;
            double base_score = 0.0;

            Model() = default;

            Model(std::vector<KNode> k_nodes, std::vector<int> tree_offsets, std::vector<double> leaf_values, double base_score);

            void predict(const double* X, size_t n_samples, size_t n_features, double* out_preds) const;
    };

} // namespace lignum