#pragma once

#include <lignum/model.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace lignum {

    // Experimental "binned" representation (LIGNUM_BINNED). Every input value is mapped once per
    // row to a 16-bit code:
    //
    //   numeric feature:      bin(x) = number of the feature's split thresholds t with t <= x,
    //                         so `x >= t_k` is exactly `bin(x) >= k + 1`
    //   categorical feature:  the integer category under the source library's rules
    //                         (kInvalidCat when it cannot belong to any set)
    //   NaN:                  kNanBin
    //
    // Comparisons become small integer comparisons, categorical slots skip the float->int
    // conversion, and a 64-byte node holds three tree levels instead of two.
    constexpr uint16_t kNanBin = 0xFFFF;
    constexpr uint16_t kInvalidCat = 0xFFFE;

    // Three levels of a binary tree. Slots in heap order: 0 is the root, 1 and 2 its children,
    // 3..6 the grandchildren (3, 4 under 1; 5, 6 under 2). children[2 * (s - 3) + dir] is the
    // subtree under slot s (negative = leaf, same encoding as KNode). Unused slots compare
    // against threshold 0 and point both ways to the same subtree.
    //
    // Numeric slot s: go right iff uint16(code + shift) >= thresholds[s], where shift = 1 for
    // slots whose NaN goes left (kNanBin wraps to 0) and the threshold already includes it.
    // Categorical slot s (bit s of cat_slots): thresholds[s] indexes BinnedForest::cat_slots.
    struct alignas(64) BNode {
        int32_t children[8];
        uint16_t thresholds[7];
        uint16_t features[7];    // dense feature index (see BinnedForest::feature_of)
        uint8_t nan_left;        // bit s: shift = 1 at numeric slot s
        uint8_t cat_slots;       // bit s: slot s is categorical
        uint8_t padding[2];
    };
    static_assert(sizeof(BNode) == 64, "BNode must fill exactly one cache line");

    // A categorical split: the set encoding of KNode (slot_kind, 64-bit payload into
    // Model::cat_bitsets) plus its semantics.
    struct BinnedCatSlot {
        uint64_t payload;
        uint8_t kind;       // slot_kind::CatMask / CatRange / CatBitset
        uint8_t xgboost;    // 1: in set -> right, NaN follows nan_right; 0 (LightGBM): in set -> left, NaN right
        uint8_t nan_right;
        uint8_t padding[5];
    };

    namespace feature_kind {
        constexpr uint8_t Numeric = 0;
        constexpr uint8_t LightGBMCategory = 1;
        constexpr uint8_t XGBoostCategory = 2;
    } // namespace feature_kind

    struct BinnedForest {
        std::vector<BNode> nodes;
        std::vector<int32_t> roots;          // per tree: BNode index, or a leaf (negative)
        std::vector<uint8_t> tree_has_cat;   // per tree: has categorical slots
        std::vector<uint32_t> feature_of;    // dense feature -> input column
        std::vector<uint8_t> feature_kind;   // dense feature -> feature_kind
        std::vector<uint32_t> cut_offsets;   // numeric dense feature j: cuts[cut_offsets[j] .. cut_offsets[j + 1])
        std::vector<double> cuts;            // sorted unique thresholds of every numeric dense feature
        std::vector<BinnedCatSlot> cat_slots;
    };

    // Builds the binned form of a model, or returns nullptr when the model uses something the
    // prototype does not cover (zero-as-missing splits, category ids >= kInvalidCat, too many
    // thresholds per feature).
    std::shared_ptr<const BinnedForest> build_binned(const Model& model);

} // namespace lignum
