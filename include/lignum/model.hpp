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

    // Output transforms; the values are part of the binary format.
    namespace transform {
        constexpr uint8_t Sigmoid = 0;
        constexpr uint8_t Exp = 1;
        constexpr uint8_t Identity = 2;
        constexpr uint8_t Log1pExp = 3;
    } // namespace transform

    // How predict() spreads work across threads. All strategies return identical results.
    enum class Strategy : uint8_t {
        Auto,    // pick from the batch size and thread count
        Serial,  // everything on the calling thread; best for one or a few rows
        Rows,    // threads take blocks of rows, sized so every thread gets work
    };

    // Bits of KNode::dirs.
    namespace dir_bits {
        constexpr uint8_t DefaultRight = 1 << 0;   // where missing values go
        constexpr uint8_t ZeroIsMissing = 1 << 1;  // LightGBM missing_type=Zero: |x| <= 1e-35 is missing too
    } // namespace dir_bits

    // Per-tree kinds: the kernel picks a traversal per tree, so special cases only cost
    // in the trees that need them. Values are part of the binary format.
    namespace tree_kind {
        constexpr uint8_t Numeric = 0;
        constexpr uint8_t ZeroMissing = 1;  // has splits with dir_bits::ZeroIsMissing
    } // namespace tree_kind

    // Two levels of a binary tree in one cache line. Slot 0 is the root split, slots 1 and 2
    // its left and right children; children[0..3] are the four grandchildren (negative = leaf).
    // Every split goes right iff  isnan(x) ? (dirs & DefaultRight) : (x >= threshold).
    struct alignas(64) KNode {
        double thresholds[4];
        int32_t children[4];
        uint16_t features[4];
        uint8_t dirs[4];
        uint8_t padding[4];
    };
    static_assert(sizeof(KNode) == 64, "KNode must fill exactly one cache line");

    class Data {
        public:
            virtual ~Data() = default;
    };

    class VectorData : public Data {
        public:
            std::vector<KNode> k_nodes;
            std::vector<int32_t> tree_offsets;
            std::vector<uint8_t> tree_kinds;
            std::vector<double> leaf_values;
    };

    class Model {
        public:
            const KNode* k_nodes = nullptr;
            size_t n_nodes = 0;

            const int32_t* tree_offsets = nullptr;
            const uint8_t* tree_kinds = nullptr;  // n_offsets entries, see tree_kind
            size_t n_offsets = 0;

            const double* leaf_values = nullptr;
            size_t n_leaves = 0;

            double base_score = 0.0;
            double sigmoid = 1.0;
            uint8_t transform = transform::Identity;
            bool float32 = false;           // accumulate and transform in float32 (XGBoost semantics)
            bool has_zero_missing = false;  // some tree is tree_kind::ZeroMissing
            uint32_t n_features = 0;        // minimum number of input columns

            std::shared_ptr<const Data> data;

            Model() = default;

            // Throws std::invalid_argument if n_features is smaller than the model requires.
            void predict(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds,
                         int n_jobs = -1, Strategy strategy = Strategy::Auto) const;
            void predict_raw(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds,
                             int n_jobs = -1, Strategy strategy = Strategy::Auto) const;
            void save(const std::string& filepath) const;
    };

} // namespace lignum
