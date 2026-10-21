#pragma once

#include <lignum/model.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace lignum {

    // Experimental "binned" representation (LIGNUM_BINNED). Every input value is first mapped
    // to the index of its interval between the model's split thresholds for that feature:
    //
    //     bin(x) = number of thresholds t of the feature with t <= x      (NaN -> kNanBin)
    //
    // so `x >= t_k` is exactly `bin(x) >= k + 1`. Comparisons become small integer comparisons
    // and a 64-byte node holds three tree levels instead of two.
    constexpr uint16_t kNanBin = 0xFFFF;

    // Three levels of a binary tree. Slots in heap order: 0 is the root, 1 and 2 its children,
    // 3..6 the grandchildren (3, 4 under 1; 5, 6 under 2). children[2 * (s - 3) + dir] is the
    // subtree under slot s (negative = leaf, same encoding as KNode). Unused slots compare
    // against threshold 0 and point both ways to the same subtree.
    //
    // NaN without a branch: kNanBin (0xFFFF) is >= every threshold, so it goes right. Slots where
    // NaN goes left add 1 to the bin (0xFFFF wraps to 0) and store threshold + 1.
    struct alignas(64) BNode {
        int32_t children[8];
        uint16_t thresholds[7];  // go right iff uint16(bin + shift) >= threshold (threshold already includes shift)
        uint16_t features[7];    // dense feature index (see BinnedForest::feature_of)
        uint8_t nan_left;        // bit s: shift = 1 at slot s, so kNanBin wraps to 0 and goes left
        uint8_t padding[3];
    };
    static_assert(sizeof(BNode) == 64, "BNode must fill exactly one cache line");

    struct BinnedForest {
        std::vector<BNode> nodes;
        std::vector<int32_t> roots;          // per tree: BNode index, or a leaf (negative)
        std::vector<uint32_t> feature_of;    // dense feature -> input column
        std::vector<uint32_t> cut_offsets;   // dense feature -> [cut_offsets[j], cut_offsets[j + 1]) in cuts
        std::vector<double> cuts;            // sorted unique thresholds of every dense feature
    };

    // Builds the binned form of a model, or returns nullptr when the model uses features the
    // prototype does not cover (categorical or zero-as-missing splits, several outputs).
    std::shared_ptr<const BinnedForest> build_binned(const Model& model);

} // namespace lignum
