#pragma once

#include <lignum/model.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace lignum {

    // Binned representation. Every input value is mapped once per row to a 16-bit code:
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
    //
    // Categorical slot s (bit s of cat_slots): thresholds[s] is a bit index into the row's
    // precomputed decision bits ("is this row's category in the set of this slot"), so the
    // tree walk reads one bit from L1 instead of converting the value and loading a set.
    //   LightGBM: in set -> left, anything else (NaN, invalid, unseen) -> right.
    //   XGBoost (bit s of xgb_slots): in set -> right; NaN -> right iff bit s of nan_left.
    struct alignas(64) BNode {
        int32_t children[8];
        uint16_t thresholds[7];
        uint16_t features[7];    // dense feature index (see BinnedForest::feature_of)
        uint8_t nan_left;        // numeric slot s: shift = 1; XGBoost categorical slot s: NaN goes right
        uint8_t cat_slots;       // bit s: slot s is categorical
        uint8_t xgb_slots;       // bit s: categorical slot s uses XGBoost semantics
        uint8_t padding;
    };
    static_assert(sizeof(BNode) == 64, "BNode must fill exactly one cache line");

    // A categorical split: the set encoding of KNode (slot_kind, 64-bit payload into
    // Model::cat_bitsets) plus its semantics.
    struct BinnedCatSlot {
        uint64_t payload;
        uint8_t kind;       // slot_kind::CatMask / CatRange / CatBitset
        uint8_t xgboost;    // 1: in set -> right, NaN follows nan_right; 0 (LightGBM): in set -> left, NaN right
        uint8_t nan_right;
        uint8_t padding;
        uint16_t dense;     // dense feature of the split
        uint16_t padding2;
    };

    namespace feature_kind {
        constexpr uint8_t Numeric = 0;
        constexpr uint8_t LightGBMCategory = 1;
        constexpr uint8_t XGBoostCategory = 2;
    } // namespace feature_kind

    // Per categorical feature: table[c] = decision bits of category c for every categorical slot
    // of this feature (bit l = c is in the set of the feature's l-th slot).
    struct BinnedCatFeature {
        uint16_t dense;          // dense feature index
        uint32_t n_categories;   // codes >= n_categories are in no set
        uint32_t words;          // 64-bit words per category row
        uint32_t row_offset;     // first word of this feature in a row's decision bits
        uint64_t table_offset;   // first word of this feature's table in BinnedForest::cat_tables
    };

    struct BinnedForest {
        std::vector<BNode> nodes;
        std::vector<int32_t> roots;          // per tree: BNode index, or a leaf (negative)
        std::vector<uint8_t> tree_has_cat;   // per tree: has categorical slots
        std::vector<uint32_t> feature_of;    // dense feature -> input column
        std::vector<uint8_t> feature_kind;   // dense feature -> feature_kind
        std::vector<uint32_t> cut_offsets;   // numeric dense feature j: cuts[cut_offsets[j] .. cut_offsets[j + 1])
        std::vector<double> cuts;            // sorted unique thresholds of every numeric dense feature
        std::vector<BinnedCatSlot> cat_slots;    // build-time only
        std::vector<BinnedCatFeature> cat_features;
        std::vector<uint64_t> cat_tables;
        uint32_t decision_words = 0;              // 64-bit words of decision bits per row
    };

    // Builds the binned form of a model, or returns nullptr when the model uses something the
    // prototype does not cover (zero-as-missing splits, category ids >= kInvalidCat, too many
    // thresholds per feature, more than 65535 categorical slots, decision tables > 256 MB).
    std::shared_ptr<const BinnedForest> build_binned(const Model& model);

} // namespace lignum
