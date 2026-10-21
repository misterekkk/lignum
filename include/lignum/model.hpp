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

    // Bits of KNode::dirs.
    namespace dir_bits {
        constexpr uint8_t DefaultRight = 1 << 0;   // where missing values go
        constexpr uint8_t ZeroIsMissing = 1 << 1;  // LightGBM missing_type=Zero: |x| <= 1e-35 is missing too
    } // namespace dir_bits

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
            double sigmoid = 1.0;
            uint8_t transform = transform::Identity;
            bool float32 = false;           // accumulate and transform in float32 (XGBoost semantics)
            bool has_zero_missing = false;  // some split has dir_bits::ZeroIsMissing
            uint32_t n_features = 0;        // minimum number of input columns

            std::shared_ptr<const Data> data;

            Model() = default;

            // Throws std::invalid_argument if n_features is smaller than the model requires.
            void predict(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs = -1) const;
            void predict_raw(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs = -1) const;
            void save(const std::string& filepath) const;

        private:
            template <typename Acc, bool ZeroMissing, bool RawScore>
            void predict_impl(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs) const noexcept;

            template <bool RawScore>
            void dispatch(const double* __restrict__ X, size_t n_samples, size_t n_features, double* __restrict__ out_preds, int n_jobs) const;
    };

} // namespace lignum
