#include "binned.hpp"

#include <algorithm>
#include <cmath>
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

                uint8_t tree_kind = 0;

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
                        c.dense = j;
                        // Slot id for now; build_binned() rewrites it to a decision bit index.
                        b.thresholds[slot] = static_cast<uint16_t>(out_.cat_slots.size());
                        b.cat_slots |= static_cast<uint8_t>(1u << slot);
                        if (c.xgboost) {
                            b.xgb_slots |= static_cast<uint8_t>(1u << slot);
                            if (nan_right) b.nan_left |= static_cast<uint8_t>(1u << slot);
                        }
                        out_.cat_slots.push_back(c);
                        tree_kind |= tree_kind::Categorical;
                        return;
                    }
                    const auto& c = cuts_[j];
                    const auto k = std::lower_bound(c.begin(), c.end(), n.threshold) - c.begin();
                    if (n.dirs & dir_bits::ZeroIsMissing) {
                        // Encoded like any numeric slot (NaN via the shift); the kernel overrides
                        // the result for the zero band.
                        b.zero_slots |= static_cast<uint8_t>(1u << slot);
                        tree_kind |= tree_kind::ZeroMissing;
                    }
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

        // Calls fn(c) for every category c in a slot's set.
        template <typename Fn>
        void for_each_category(const BinnedCatSlot& slot, const uint32_t* bitsets, Fn&& fn) {
            const uint32_t lo = static_cast<uint32_t>(slot.payload), hi = static_cast<uint32_t>(slot.payload >> 32);
            switch (slot.kind) {
                case slot_kind::CatMask:
                    for (uint32_t c = 0; c < 64; c++) if ((slot.payload >> c) & 1) fn(c);
                    break;
                case slot_kind::CatRange:
                    for (uint32_t c = lo; c <= lo + hi; c++) fn(c);
                    break;
                default:
                    for (uint32_t w = 0; w < hi; w++) {
                        for (uint32_t b = 0; b < 32; b++) if ((bitsets[lo + w] >> b) & 1) fn(w * 32 + b);
                    }
                    break;
            }
        }

        // Per categorical feature, a table category -> decision bits of all its slots; the slots'
        // BNode thresholds become bit indices into a row's decision bits.
        bool build_decision_tables(const Model& model, BinnedForest& f) {
            std::vector<std::vector<uint32_t>> slots_of(f.feature_of.size());
            for (uint32_t id = 0; id < f.cat_slots.size(); id++) slots_of[f.cat_slots[id].dense].push_back(id);

            std::vector<uint32_t> bit_of(f.cat_slots.size());
            uint32_t row_words = 0;
            for (uint16_t j = 0; j < slots_of.size(); j++) {
                const auto& ids = slots_of[j];
                if (ids.empty()) continue;
                BinnedCatFeature cf{};
                cf.dense = j;
                cf.words = static_cast<uint32_t>((ids.size() + 63) / 64);
                cf.row_offset = row_words;
                for (uint32_t id : ids) {
                    cf.n_categories = std::max<uint32_t>(cf.n_categories, static_cast<uint32_t>(
                        category_bound(f.cat_slots[id].kind, f.cat_slots[id].payload)));
                }
                cf.table_offset = f.cat_tables.size();
                const uint64_t table_words = uint64_t{cf.n_categories} * cf.words;
                if (cf.table_offset + table_words > (uint64_t{256} << 20) / 8) return false;  // 256 MB
                f.cat_tables.resize(cf.table_offset + table_words, 0);
                for (uint32_t l = 0; l < ids.size(); l++) {
                    bit_of[ids[l]] = row_words * 64 + l;
                    for_each_category(f.cat_slots[ids[l]], model.cat_bitsets, [&](uint32_t c) {
                        f.cat_tables[cf.table_offset + uint64_t{c} * cf.words + l / 64] |= uint64_t{1} << (l % 64);
                    });
                }
                row_words += cf.words;
                f.cat_features.push_back(cf);
            }
            if (uint64_t{row_words} * 64 > 0xFFFF) return false;  // bit indices are 16-bit
            f.decision_words = row_words;

            for (BNode& b : f.nodes) {
                for (int s = 0; s < 7; s++) {
                    if ((b.cat_slots >> s) & 1) b.thresholds[s] = static_cast<uint16_t>(bit_of[b.thresholds[s]]);
                }
            }
            return true;
        }

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
        std::vector<bool> zero_feature;
        for (const BTNode& n : bt) {
            if (n.leaf) continue;

            uint8_t kind = feature_kind::Numeric;
            if (n.kind != slot_kind::Numeric) {
                kind = (n.dirs & dir_bits::XGBoostCategory) ? feature_kind::XGBoostCategory : feature_kind::LightGBMCategory;
                if (category_bound(n.kind, std::bit_cast<uint64_t>(n.threshold)) > kInvalidCat) return nullptr;
            }

            auto [it, inserted] = dense.try_emplace(n.feature, static_cast<uint16_t>(cuts.size()));
            if (inserted) {
                if (cuts.size() >= 0xFFFF) return nullptr;
                cuts.emplace_back();
                zero_feature.push_back(false);
                forest->feature_of.push_back(n.feature);
                forest->feature_kind.push_back(kind);
            } else if (forest->feature_kind[it->second] != kind) {
                return nullptr;  // the same column used with different semantics
            }
            if (kind == feature_kind::Numeric) cuts[it->second].push_back(n.threshold);
            if (n.dirs & dir_bits::ZeroIsMissing) zero_feature[it->second] = true;
        }
        // LightGBM zeroes |x| <= kZero; parsers never place a threshold inside (-kZero, kZero], so
        // with both band edges as cuts the band is exactly one bin.
        constexpr double kZero = static_cast<double>(1e-35f);
        forest->zero_bin.assign(cuts.size(), kNanBin);
        for (size_t j = 0; j < cuts.size(); j++) {
            auto& c = cuts[j];
            if (zero_feature[j]) {
                c.push_back(-kZero);
                c.push_back(std::nextafter(kZero, INFINITY));
            }
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
            if (c.size() >= kInvalidCat - 1) return nullptr;  // bin + 1 must stay below kInvalidCat
            if (zero_feature[j]) {
                const auto lo = std::lower_bound(c.begin(), c.end(), -kZero) - c.begin();
                const auto hi = std::lower_bound(c.begin(), c.end(), std::nextafter(kZero, INFINITY)) - c.begin();
                if (hi != lo + 1) return nullptr;  // a threshold inside the band: not representable
                forest->zero_bin[j] = static_cast<uint16_t>(lo + 1);
            }
        }
        forest->cut_offsets.push_back(0);
        for (const auto& c : cuts) {
            forest->cuts.insert(forest->cuts.end(), c.begin(), c.end());
            forest->cut_offsets.push_back(static_cast<uint32_t>(forest->cuts.size()));
        }

        Packer packer(bt, *forest, dense, cuts);
        try {
            for (int32_t root : bt_roots) {
                packer.tree_kind = 0;
                forest->roots.push_back(packer.pack(root));
                forest->tree_kind.push_back(packer.tree_kind);
            }
        } catch (const std::length_error&) {
            return nullptr;  // more categorical slots than a 16-bit index can address
        }
        if (!build_decision_tables(model, *forest)) return nullptr;
        return forest;
    }

} // namespace lignum
