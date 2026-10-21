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
        constexpr uint8_t Softmax = 4;  // across outputs
        constexpr uint8_t Argmax = 5;   // index of the largest margin (XGBoost multi:softmax); one output column
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
        constexpr uint8_t ZeroMissing = 1 << 0;  // has splits with dir_bits::ZeroIsMissing
        constexpr uint8_t Categorical = 1 << 1;  // has categorical splits (see slot_kind)
    } // namespace tree_kind

    // Kind of each KNode slot. A categorical slot stores its category set in the 64 bits of
    // `thresholds[slot]` and goes LEFT when the category is in the set (LightGBM semantics:
    // NaN and values <= -1 go right; other values are truncated to an integer category).
    namespace slot_kind {
        constexpr uint8_t Numeric = 0;
        constexpr uint8_t CatMask = 1;    // bits: categories 0..63 (bit c set = c in the set)
        constexpr uint8_t CatRange = 2;   // low 32 bits: first category, high 32 bits: last - first
        constexpr uint8_t CatBitset = 3;  // low 32 bits: offset into Model::cat_bitsets, high 32 bits: words
    } // namespace slot_kind

    // Two levels of a binary tree in one cache line. Slot 0 is the root split, slots 1 and 2
    // its left and right children; children[0..3] are the four grandchildren (negative = leaf).
    // A numeric split goes right iff  isnan(x) ? (dirs & DefaultRight) : (x >= threshold).
    struct alignas(64) KNode {
        double thresholds[4];  // numeric threshold, or the category set of a categorical slot
        int32_t children[4];
        uint16_t features[4];
        uint8_t dirs[4];
        uint8_t kinds[4];      // slot_kind of each slot
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
            std::vector<uint16_t> tree_outputs;
            std::vector<double> leaf_values;
            std::vector<double> base_scores;
            std::vector<uint32_t> cat_bitsets;
    };

    class Model {
        public:
            const KNode* k_nodes = nullptr;
            size_t n_nodes = 0;

            const int32_t* tree_offsets = nullptr;
            const uint8_t* tree_kinds = nullptr;     // n_offsets entries, see tree_kind
            const uint16_t* tree_outputs = nullptr;  // n_offsets entries when n_outputs > 1: output (class) of each tree
            size_t n_offsets = 0;

            const double* leaf_values = nullptr;
            size_t n_leaves = 0;

            const uint32_t* cat_bitsets = nullptr;  // category sets too large for one slot
            size_t n_cat_words = 0;

            const double* base_scores = nullptr;  // n_outputs entries, margin space
            uint32_t n_outputs = 1;                // classes for multiclass models, otherwise 1
            double sigmoid = 1.0;
            uint8_t transform = transform::Identity;
            bool float32 = false;           // accumulate and transform in float32 (XGBoost semantics)
            bool has_special_trees = false;  // some tree is not tree_kind::Numeric
            uint32_t n_features = 0;        // minimum number of input columns

            std::shared_ptr<const Data> data;

            Model() = default;

            // Values per row written by predict (raw_score=false) or predict_raw (raw_score=true).
            // out_preds must hold n_samples * output_width(raw_score) doubles, row-major.
            size_t output_width(bool raw_score) const {
                return (!raw_score && transform == transform::Argmax) ? 1 : n_outputs;
            }

            // Throws std::invalid_argument if n_features is smaller than the model requires.
            void predict(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds,
                         int n_jobs = -1, Strategy strategy = Strategy::Auto) const;
            void predict_raw(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds,
                             int n_jobs = -1, Strategy strategy = Strategy::Auto) const;
            void save(const std::string& filepath) const;
    };

} // namespace lignum
