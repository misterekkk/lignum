#include "binned.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <unordered_map>

namespace lignum {

    namespace {

        // Plain binary tree recovered from the KNode form.
        struct BTNode {
            bool leaf = false;
            int32_t leaf_ref = 0;   // negative leaf reference (as stored in KNode::children)
            uint32_t feature = 0;   // input column
            double threshold = 0.0;
            uint8_t kind = slot_kind::Numeric;
            uint8_t dirs = 0;
            int32_t left = -1, right = -1;  // indices into the BTNode vector
        };

        class Unpacker {
            public:
                explicit Unpacker(const Model& m) : m_(m) {}
                std::vector<BTNode> nodes;

                int32_t from_ref(int32_t ref) {
                    if (ref < 0) return leaf(ref);
                    const KNode& kn = m_.k_nodes[ref];
                    // A child slot is a leaf iff both of its grandchild pointers are the same leaf.
                    const int32_t left = kn.children[0] == kn.children[1] ? leaf(kn.children[0])
                                                                          : split(kn, 1, from_ref(kn.children[0]), from_ref(kn.children[1]));
                    const int32_t right = kn.children[2] == kn.children[3] ? leaf(kn.children[2])
                                                                           : split(kn, 2, from_ref(kn.children[2]), from_ref(kn.children[3]));
                    return split(kn, 0, left, right);
                }

            private:
                const Model& m_;

                int32_t leaf(int32_t ref) {
                    BTNode n;
                    n.leaf = true;
                    n.leaf_ref = ref;
                    nodes.push_back(n);
                    return static_cast<int32_t>(nodes.size() - 1);
                }

                int32_t split(const KNode& kn, int slot, int32_t left, int32_t right) {
                    BTNode n;
                    n.feature = kn.features[slot];
                    n.threshold = kn.thresholds[slot];
                    n.kind = kn.kinds[slot];
                    n.dirs = kn.dirs[slot];
                    n.left = left;
                    n.right = right;
                    nodes.push_back(n);
                    return static_cast<int32_t>(nodes.size() - 1);
                }
        };

        // Largest category a slot's set can contain + 1 (sets never reach kInvalidCat).
        uint64_t category_bound(uint8_t kind, uint64_t payload) {
            const uint64_t lo = payload & 0xFFFFFFFFu, hi = payload >> 32;
            switch (kind) {
                case slot_kind::CatMask:  return 64;
                case slot_kind::CatRange: return lo + hi + 1;
                default:                  return hi * 32;
            }
        }

        class Packer {
            public:
                Packer(const std::vector<BTNode>& bt, BinnedForest& out, const std::unordered_map<uint32_t, uint16_t>& dense,
                       const std::vector<std::vector<double>>& cuts)
                    : bt_(bt), out_(out), dense_(dense), cuts_(cuts) {}

                bool tree_has_cat = false;

                int32_t pack(int32_t i) {
                    if (bt_[i].leaf) return bt_[i].leaf_ref;
                    const int32_t idx = static_cast<int32_t>(out_.nodes.size());
                    out_.nodes.push_back(BNode{});
                    BNode b{};
                    fill(b, 0, i);
                    pack_side(b, 1, bt_[i].left, 0);
                    pack_side(b, 2, bt_[i].right, 4);
                    out_.nodes[idx] = b;
                    return idx;
                }

            private:
                const std::vector<BTNode>& bt_;
                BinnedForest& out_;
                const std::unordered_map<uint32_t, uint16_t>& dense_;
                const std::vector<std::vector<double>>& cuts_;

                void fill(BNode& b, int slot, int32_t i) {
                    const BTNode& n = bt_[i];
                    const uint16_t j = dense_.at(n.feature);
                    b.features[slot] = j;
                    const bool nan_right = (n.dirs & dir_bits::DefaultRight) != 0;

                    if (n.kind != slot_kind::Numeric) {
                        if (out_.cat_slots.size() >= 0xFFFF) throw std::length_error("too many categorical slots");
                        BinnedCatSlot c{};
                        c.payload = std::bit_cast<uint64_t>(n.threshold);
                        c.kind = n.kind;
                        c.xgboost = (n.dirs & dir_bits::XGBoostCategory) != 0;
                        c.nan_right = nan_right;
                        b.thresholds[slot] = static_cast<uint16_t>(out_.cat_slots.size());
                        b.cat_slots |= static_cast<uint8_t>(1u << slot);
                        out_.cat_slots.push_back(c);
                        tree_has_cat = true;
                        return;
                    }
                    const auto& c = cuts_[j];
                    const auto k = std::lower_bound(c.begin(), c.end(), n.threshold) - c.begin();
                    b.thresholds[slot] = static_cast<uint16_t>(k + 1 + (nan_right ? 0 : 1));
                    if (!nan_right) b.nan_left |= static_cast<uint8_t>(1u << slot);
                }

                // Child `i` of the root goes to `slot` (1 or 2); its children to slots 2 * slot + 1
                // and 2 * slot + 2, whose subtrees start at children[first].
                void pack_side(BNode& b, int slot, int32_t i, int first) {
                    if (bt_[i].leaf) {
                        for (int k = 0; k < 4; k++) b.children[first + k] = bt_[i].leaf_ref;
                        return;
                    }
                    fill(b, slot, i);
                    pack_grandchild(b, 2 * slot + 1, bt_[i].left, first);
                    pack_grandchild(b, 2 * slot + 2, bt_[i].right, first + 2);
                }

                void pack_grandchild(BNode& b, int slot, int32_t i, int first) {
                    if (bt_[i].leaf) {
                        b.children[first] = b.children[first + 1] = bt_[i].leaf_ref;
                        return;
                    }
                    fill(b, slot, i);
                    // Recursion may grow out_.nodes; b is a local copy, written back by pack().
                    b.children[first] = pack(bt_[i].left);
                    b.children[first + 1] = pack(bt_[i].right);
                }
        };

    } // namespace

    std::shared_ptr<const BinnedForest> build_binned(const Model& model) {
        Unpacker unpacker(model);
        std::vector<int32_t> bt_roots;
        for (size_t t = 0; t < model.n_offsets; t++) bt_roots.push_back(unpacker.from_ref(model.tree_offsets[t]));
        const auto& bt = unpacker.nodes;

        // Dense features with their kind; sorted unique thresholds for numeric ones.
        std::unordered_map<uint32_t, uint16_t> dense;
        std::vector<std::vector<double>> cuts;
        auto forest = std::make_shared<BinnedForest>();
        for (const BTNode& n : bt) {
            if (n.leaf) continue;
            if (n.dirs & dir_bits::ZeroIsMissing) return nullptr;

            uint8_t kind = feature_kind::Numeric;
            if (n.kind != slot_kind::Numeric) {
                kind = (n.dirs & dir_bits::XGBoostCategory) ? feature_kind::XGBoostCategory : feature_kind::LightGBMCategory;
                if (category_bound(n.kind, std::bit_cast<uint64_t>(n.threshold)) > kInvalidCat) return nullptr;
            }

            auto [it, inserted] = dense.try_emplace(n.feature, static_cast<uint16_t>(cuts.size()));
            if (inserted) {
                if (cuts.size() >= 0xFFFF) return nullptr;
                cuts.emplace_back();
                forest->feature_of.push_back(n.feature);
                forest->feature_kind.push_back(kind);
            } else if (forest->feature_kind[it->second] != kind) {
                return nullptr;  // the same column used with different semantics
            }
            if (kind == feature_kind::Numeric) cuts[it->second].push_back(n.threshold);
        }
        for (auto& c : cuts) {
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
            if (c.size() >= kInvalidCat - 1) return nullptr;  // bin + 1 must stay below kInvalidCat
        }
        forest->cut_offsets.push_back(0);
        for (const auto& c : cuts) {
            forest->cuts.insert(forest->cuts.end(), c.begin(), c.end());
            forest->cut_offsets.push_back(static_cast<uint32_t>(forest->cuts.size()));
        }

        Packer packer(bt, *forest, dense, cuts);
        try {
            for (int32_t root : bt_roots) {
                packer.tree_has_cat = false;
                forest->roots.push_back(packer.pack(root));
                forest->tree_has_cat.push_back(packer.tree_has_cat);
            }
        } catch (const std::length_error&) {
            return nullptr;  // more categorical slots than a 16-bit index can address
        }
        return forest;
    }

} // namespace lignum
