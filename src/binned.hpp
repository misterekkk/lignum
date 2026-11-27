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
    //
    // Zero-as-missing slot s (bit s of zero_slots): the feature's cuts include both edges of the
    // zero band [-kZero, kZero], so the band is exactly one bin (BinnedForest::zero_bin); a code
    // equal to it takes the NaN direction (right iff the slot has no shift).
    struct alignas(64) BNode {
        int32_t children[8];
        uint16_t thresholds[7];
        uint16_t features[7];    // dense feature index (see BinnedForest::feature_of)
        uint8_t nan_left;        // numeric slot s: shift = 1; XGBoost categorical slot s: NaN goes right
        uint8_t cat_slots;       // bit s: slot s is categorical
        uint8_t xgb_slots;       // bit s: categorical slot s uses XGBoost semantics
        uint8_t zero_slots;      // bit s: numeric slot s treats the zero band as missing (LightGBM
                                 // missing_type=Zero), going where NaN goes
    };
    static_assert(sizeof(BNode) == 64, "BNode must fill exactly one cache line");

    // Trees with oblique splits (miter models, tree_kind::Oblique): two tree levels per node, three
    // slots of one uniform form over the row's 16-bit codes (NaN = 0xFFFF in every code kind):
    //   any of c[f0], c[f1], c[f2] is NaN ? default (bit 15 of f[0]) : P >= thr,
    //   P = w0 * (c[f0] - 32768) + w1 * (c[f1] - 32768) + w2 * (c[f2] - 32768)  (int32)
    // - oblique split: f = Q16 code columns (after the dense features), w = the split's weights;
    // - axis / CTR:    f = (j, j, j) on dense feature j, w = (1, 0, 0), thr = k + 1 - 32768, where k is
    //                  the threshold's index among the feature's cuts (x >= cut_k  <=>  bin >= k + 1);
    // - categorical (only in trees with tree_kind::Categorical; LightGBM semantics): all weights 0,
    //                  thr = decision bit index, goes right iff the bit is clear (NaN: right).
    // children[(d0 << 1) | d1] as in KNode (negative = leaf). Slots of leaf children are never
    // used for routing (both grandchildren are the same leaf).
    struct ONodeSlot {
        uint16_t f[3];
        int16_t w[3];
        int32_t thr;
    };
    static_assert(sizeof(ONodeSlot) == 16, "ONodeSlot is part of the binary format");

    struct alignas(64) ONode {
        ONodeSlot s[3];
        int32_t children[4];
    };
    static_assert(sizeof(ONode) == 64, "ONode must fill exactly one cache line");

    // Symmetric (oblivious) miter trees recovered at load/build time: every tree has fixed depth d
    // and one ONodeSlot per level, indexing 2^d leaves branch-free via bit shifts.
    struct SymTree {
        uint32_t slot_offset = 0;
        uint32_t leaf_offset = 0;
        uint8_t depth = 0;
        uint8_t has_cat = 0;
        uint16_t output = 0;
    };

    struct SymForest {
        std::vector<SymTree> trees;
        std::vector<ONodeSlot> slots;
        std::vector<double> leaves;
    };

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
        constexpr uint8_t Ctr = 3;  // dense feature whose cuts bin CTR lookup floats (`feature_of` = CTR table index)
    } // namespace feature_kind

    // Per categorical feature: table[c] = decision bits of category c for every categorical slot
    // of this feature (bit l = c is in the set of the feature's l-th slot).
    struct BinnedCatFeature {
        uint32_t dense;          // dense feature index
        uint32_t n_categories;   // codes >= n_categories are in no set
        uint32_t words;          // 64-bit words per category row
        uint32_t row_offset;     // first word of this feature in a row's decision bits
        uint64_t table_offset;   // first word of this feature's table in BinnedForest::cat_tables
    };
    static_assert(sizeof(BinnedCatFeature) == 24, "BinnedCatFeature is part of the binary format");

    // View of a binned model; the memory belongs to `owner` (vectors built at load time, or the
    // mapped model file). Same ownership scheme as Model.
    struct BinnedForest {
        const BNode* nodes = nullptr;
        size_t n_nodes = 0;
        const int32_t* roots = nullptr;          // per tree: BNode index, or a leaf (negative)
        const uint8_t* tree_kind = nullptr;      // per tree: tree_kind bits (Categorical, ZeroMissing)
        size_t n_trees = 0;
        const uint32_t* feature_of = nullptr;    // dense feature -> input column
        const uint8_t* feature_kind = nullptr;   // dense feature -> feature_kind
        const uint16_t* zero_bin = nullptr;      // dense feature -> bin of the zero band (zero-as-missing features)
        const uint32_t* cut_offsets = nullptr;   // n_features + 1: numeric feature j uses cuts[cut_offsets[j] .. cut_offsets[j + 1])
        size_t n_features = 0;
        const double* cuts = nullptr;            // sorted unique thresholds of every numeric dense feature
        size_t n_cuts = 0;
        const BinnedCatFeature* cat_features = nullptr;
        size_t n_cat_features = 0;
        const uint64_t* cat_tables = nullptr;
        size_t n_table_words = 0;
        uint32_t decision_words = 0;             // 64-bit words of decision bits per row
        // Oblique trees (miter): their roots index `onodes`; rows carry n_q16 extra codes after the
        // n_features dense codes: the Q16 code of input column q16_columns[q] (Model::quantizers).
        const ONode* onodes = nullptr;
        size_t n_onodes = 0;
        const uint32_t* q16_columns = nullptr;
        size_t n_q16 = 0;
        size_t row_codes() const { return n_features + n_q16; }
        bool has_ctr_features = false;           // runtime cache: any dense feature has feature_kind::Ctr
        bool fits_u8_codes = false;              // runtime cache: numeric-only with <= 254 cuts per feature
        std::shared_ptr<const SymForest> sym;    // runtime cache: symmetric miter trees (null otherwise)
        std::shared_ptr<const void> owner;
    };

    inline bool compute_has_ctr_features(const BinnedForest& f) {
        for (size_t j = 0; j < f.n_features; j++) {
            if (f.feature_kind[j] == feature_kind::Ctr) return true;
        }
        return false;
    }

    inline bool compute_fits_u8_codes(const BinnedForest& f) {
        if (f.n_cat_features > 0 || f.n_onodes > 0 || f.n_q16 > 0 || f.has_ctr_features || f.sym != nullptr) return false;
        for (size_t j = 0; j < f.n_features; j++) {
            if (f.cut_offsets[j + 1] - f.cut_offsets[j] > 254) return false;
        }
        return true;
    }

    // Detects when every tree in a miter model is symmetric (level-wise oblivious) of uniform
    // depth <= 10 and packs them into a compact branch-free SymForest (or returns nullptr).
    std::shared_ptr<const SymForest> detect_symmetric_forest(const Model& model, const BinnedForest& f);

    // Builds the binned form of a model, or returns nullptr when it does not fit the 16-bit codes
    // (category ids >= kInvalidCat, too many thresholds per feature, more than 65535 categorical
    // slots) or the decision tables would exceed 256 MB.
    std::shared_ptr<const BinnedForest> build_binned(const Model& model);

    // Binary format (see io.cpp): header + 64-byte aligned sections.
    struct alignas(64) BinnedFileHeader {
        uint64_t n_nodes = 0, n_trees = 0, n_features = 0, n_cuts = 0, n_cat_features = 0, n_table_words = 0;
        uint32_t decision_words = 0;
        uint32_t n_q16 = 0;     // format 0.10 (0 before): Q16 code columns, then their section
        uint64_t n_onodes = 0;  // format 0.10 (0 before): ONodes, after the other binned sections
    };
    static_assert(sizeof(BinnedFileHeader) == 64, "BinnedFileHeader must stay 64 bytes");

} // namespace lignum
