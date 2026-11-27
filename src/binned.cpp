#include "binned.hpp"
#include "ir.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace lignum {

    namespace {

        // Owning storage of a binned forest built at load time.
        struct BinnedStorage {
            std::vector<BNode> nodes;
            std::vector<int32_t> roots;
            std::vector<uint8_t> tree_kind;
            std::vector<uint32_t> feature_of;
            std::vector<uint8_t> feature_kind;
            std::vector<uint16_t> zero_bin;
            std::vector<uint32_t> cut_offsets;
            std::vector<double> cuts;
            std::vector<BinnedCatSlot> cat_slots;  // build-time only
            std::vector<BinnedCatFeature> cat_features;
            std::vector<uint64_t> cat_tables;
            uint32_t decision_words = 0;
            std::vector<ONode> onodes;
            std::vector<uint32_t> q16_columns;
        };

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
                Packer(const std::vector<BTNode>& bt, BinnedStorage& out, const std::vector<uint16_t>& dense,
                       const std::vector<uint16_t>& dense_ctr, const std::vector<std::vector<double>>& cuts)
                    : bt_(bt), out_(out), dense_(dense), dense_ctr_(dense_ctr), cuts_(cuts) {}

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
                BinnedStorage& out_;
                const std::vector<uint16_t>& dense_;
                const std::vector<uint16_t>& dense_ctr_;
                const std::vector<std::vector<double>>& cuts_;

                void fill(BNode& b, int slot, int32_t i) {
                    const BTNode& n = bt_[i];
                    if (n.kind == slot_kind::Ctr) {
                        const uint16_t j = dense_ctr_[n.feature];
                        b.features[slot] = j;
                        const auto& c = cuts_[j];
                        const auto k = std::lower_bound(c.begin(), c.end(), n.threshold) - c.begin();
                        b.thresholds[slot] = static_cast<uint16_t>(k + 1);
                        tree_kind |= tree_kind::Ctr;
                        return;
                    }
                    const uint16_t j = dense_[n.feature];
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

        // Packs a tree with oblique splits into ONodes (see binned.hpp). Throws std::domain_error
        // for what the ONode form does not cover (the model then keeps the KNode kernel).
        class OPacker {
            public:
                OPacker(const Model& m, const std::vector<BTNode>& bt, BinnedStorage& out,
                        const std::vector<uint16_t>& dense, const std::vector<uint16_t>& dense_ctr,
                        const std::vector<std::vector<double>>& cuts, const std::vector<uint16_t>& q16, size_t n_dense)
                    : m_(m), bt_(bt), out_(out), dense_(dense), dense_ctr_(dense_ctr),
                      cuts_(cuts), q16_(q16), n_dense_(n_dense) {}

                uint8_t tree_kind = 0;

                int32_t pack(int32_t i) {
                    if (bt_[i].leaf) return bt_[i].leaf_ref;
                    const int32_t idx = static_cast<int32_t>(out_.onodes.size());
                    out_.onodes.push_back(ONode{});
                    ONode o{};
                    fill(o.s[0], i);
                    side(o, 1, bt_[i].left, 0);
                    side(o, 2, bt_[i].right, 2);
                    out_.onodes[idx] = o;
                    return idx;
                }

            private:
                const Model& m_;
                const std::vector<BTNode>& bt_;
                BinnedStorage& out_;
                const std::vector<uint16_t>& dense_;
                const std::vector<uint16_t>& dense_ctr_;
                const std::vector<std::vector<double>>& cuts_;
                const std::vector<uint16_t>& q16_;
                size_t n_dense_;

                static uint16_t code_index(size_t j) {
                    if (j >= 0x8000) throw std::domain_error("too many code columns for ONode");
                    return static_cast<uint16_t>(j);
                }

                void fill(ONodeSlot& s, int32_t i) {
                    const BTNode& n = bt_[i];
                    const bool right = (n.dirs & dir_bits::DefaultRight) != 0;
                    if (n.kind == slot_kind::Oblique) {
                        const ObliqueSplit& o = m_.oblique_splits[std::bit_cast<uint64_t>(n.threshold)];
                        if (o.weights[0] == 0 && o.weights[1] == 0 && o.weights[2] == 0) {
                            throw std::domain_error("oblique split without weights");
                        }
                        for (int q = 0; q < 3; q++) {
                            s.f[q] = code_index(n_dense_ + q16_[o.features[q]]);
                            s.w[q] = o.weights[q];
                        }
                        s.thr = o.threshold;
                        tree_kind |= tree_kind::Oblique;
                    } else if (n.kind == slot_kind::Numeric) {
                        if (n.dirs & dir_bits::ZeroIsMissing) throw std::domain_error("zero-as-missing in an oblique tree");
                        const uint16_t j = code_index(dense_[n.feature]);
                        const auto& c = cuts_[j];
                        const auto k = std::lower_bound(c.begin(), c.end(), n.threshold) - c.begin();
                        s.f[0] = s.f[1] = s.f[2] = j;
                        s.w[0] = 1;
                        s.w[1] = s.w[2] = 0;
                        s.thr = static_cast<int32_t>(k + 1) - 32768;
                    } else if (n.kind == slot_kind::Ctr) {
                        const uint16_t j = code_index(dense_ctr_[n.feature]);
                        const auto& c = cuts_[j];
                        const auto k = std::lower_bound(c.begin(), c.end(), n.threshold) - c.begin();
                        s.f[0] = s.f[1] = s.f[2] = j;
                        s.w[0] = 1;
                        s.w[1] = s.w[2] = 0;
                        s.thr = static_cast<int32_t>(k + 1) - 32768;
                        tree_kind |= tree_kind::Ctr;
                        return;  // CTR never produces NaN; bit 15 stays 0
                    } else {
                        if (n.dirs & dir_bits::XGBoostCategory) throw std::domain_error("XGBoost categories in an oblique tree");
                        if (out_.cat_slots.size() >= 0xFFFF) throw std::length_error("too many categorical slots");
                        const uint16_t j = code_index(dense_[n.feature]);
                        BinnedCatSlot c{};
                        c.payload = std::bit_cast<uint64_t>(n.threshold);
                        c.kind = n.kind;
                        c.nan_right = right;
                        c.dense = j;
                        s.f[0] = s.f[1] = s.f[2] = j;
                        s.w[0] = s.w[1] = s.w[2] = 0;
                        s.thr = static_cast<int32_t>(out_.cat_slots.size());  // rewritten to a decision bit
                        out_.cat_slots.push_back(c);
                        tree_kind |= tree_kind::Categorical;
                        return;  // NaN goes right (decision bits are clear), bit 15 unused
                    }
                    if (right) s.f[0] = static_cast<uint16_t>(s.f[0] | 0x8000);
                }

                // Child `i` of the root goes to slot 1 or 2; its children to children[first], [first + 1].
                void side(ONode& o, int slot, int32_t i, int first) {
                    if (bt_[i].leaf) {
                        o.s[slot] = ONodeSlot{{0, 0, 0}, {1, 0, 0}, 0};  // never decides: both children equal
                        o.children[first] = o.children[first + 1] = bt_[i].leaf_ref;
                        return;
                    }
                    fill(o.s[slot], i);
                    // Recursion may grow out_.onodes; `o` is a local copy, written back by pack().
                    o.children[first] = pack(bt_[i].left);
                    o.children[first + 1] = pack(bt_[i].right);
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

        // Per categorical feature, a table category -> decision bits of all its unique slot sets;
        // duplicate sets on the same feature share one bit, shrinking decision_words and cat_tables.
        bool build_decision_tables(const Model& model, BinnedStorage& f) {
            std::vector<std::vector<uint32_t>> slots_of(f.feature_of.size());
            for (uint32_t id = 0; id < f.cat_slots.size(); id++) slots_of[f.cat_slots[id].dense].push_back(id);

            std::vector<uint32_t> bit_of(f.cat_slots.size());
            uint32_t row_words = 0;
            std::vector<uint32_t> unique_ids;
            std::vector<uint32_t> tmp_words;
            std::unordered_map<uint64_t, uint32_t> mask_bit;
            std::unordered_map<std::string, uint32_t> wide_bit;
            std::string key_buf;

            for (size_t j = 0; j < slots_of.size(); j++) {  // at most 0xFFFF dense features (see build)
                const auto& ids = slots_of[j];
                if (ids.empty()) continue;
                unique_ids.clear();
                mask_bit.clear();
                wide_bit.clear();

                auto register_mask = [&](uint64_t m, uint32_t id) {
                    auto [it, ins] = mask_bit.try_emplace(m, static_cast<uint32_t>(unique_ids.size()));
                    if (ins) unique_ids.push_back(id);
                    bit_of[id] = row_words * 64 + it->second;
                };

                for (uint32_t id : ids) {
                    const BinnedCatSlot& slot = f.cat_slots[id];
                    const uint32_t lo = static_cast<uint32_t>(slot.payload);
                    const uint32_t hi = static_cast<uint32_t>(slot.payload >> 32);
                    if (slot.kind == slot_kind::CatMask) {
                        register_mask(slot.payload, id);
                    } else if (slot.kind == slot_kind::CatRange && uint64_t{lo} + hi < 64) {
                        const uint64_t width = uint64_t{hi} + 1;
                        const uint64_t m = (width == 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1)) << lo;
                        register_mask(m, id);
                    } else if (slot.kind == slot_kind::CatBitset) {
                        uint32_t trimmed = hi;
                        while (trimmed > 0 && model.cat_bitsets[lo + trimmed - 1] == 0) trimmed--;
                        if (trimmed <= 2) {
                            uint64_t m = 0;
                            if (trimmed >= 1) m |= model.cat_bitsets[lo];
                            if (trimmed == 2) m |= uint64_t{model.cat_bitsets[lo + 1]} << 32;
                            register_mask(m, id);
                        } else {
                            key_buf.assign(reinterpret_cast<const char*>(&model.cat_bitsets[lo]), trimmed * sizeof(uint32_t));
                            auto [it, ins] = wide_bit.try_emplace(key_buf, static_cast<uint32_t>(unique_ids.size()));
                            if (ins) unique_ids.push_back(id);
                            bit_of[id] = row_words * 64 + it->second;
                        }
                    } else {
                        // CatRange extending beyond bit 63: canonical bitset representation.
                        const uint32_t last = lo + hi;
                        const uint32_t words32 = last / 32 + 1;
                        tmp_words.assign(words32, 0);
                        for (uint32_t c = lo; c <= last; c++) tmp_words[c >> 5] |= uint32_t{1} << (c & 31);
                        key_buf.assign(reinterpret_cast<const char*>(tmp_words.data()), tmp_words.size() * sizeof(uint32_t));
                        auto [it, ins] = wide_bit.try_emplace(key_buf, static_cast<uint32_t>(unique_ids.size()));
                        if (ins) unique_ids.push_back(id);
                        bit_of[id] = row_words * 64 + it->second;
                    }
                }

                BinnedCatFeature cf{};
                cf.dense = static_cast<uint32_t>(j);
                cf.words = static_cast<uint32_t>((unique_ids.size() + 63) / 64);
                cf.row_offset = row_words;
                for (uint32_t id : unique_ids) {
                    cf.n_categories = std::max<uint32_t>(cf.n_categories, static_cast<uint32_t>(
                        category_bound(f.cat_slots[id].kind, f.cat_slots[id].payload)));
                }
                cf.table_offset = f.cat_tables.size();
                const uint64_t table_words = uint64_t{cf.n_categories} * cf.words;
                if (cf.table_offset + table_words > (uint64_t{256} << 20) / 8) return false;  // 256 MB
                f.cat_tables.resize(cf.table_offset + table_words, 0);
                for (uint32_t l = 0; l < unique_ids.size(); l++) {
                    for_each_category(f.cat_slots[unique_ids[l]], model.cat_bitsets, [&](uint32_t c) {
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
            for (ONode& o : f.onodes) {
                for (ONodeSlot& s : o.s) {
                    if (s.w[0] == 0 && s.w[1] == 0 && s.w[2] == 0) {
                        s.thr = static_cast<int32_t>(bit_of[static_cast<uint32_t>(s.thr)]);
                    }
                }
            }
            return true;
        }

        constexpr int kMaxSymDepth = 10;

        bool bslot_to_oslot(const BNode& b, int s, ONodeSlot& out, uint8_t& has_cat) {
            if (((b.xgb_slots >> s) & 1) || ((b.zero_slots >> s) & 1)) return false;
            const uint16_t j = b.features[s];
            if (j >= 0x8000) return false;
            if ((b.cat_slots >> s) & 1) {
                out = ONodeSlot{{j, j, j}, {0, 0, 0}, static_cast<int32_t>(b.thresholds[s])};
                has_cat = 1;
                return true;
            }
            const uint8_t shift = (b.nan_left >> s) & 1;
            const uint16_t nan_right = static_cast<uint16_t>(shift ^ 1);
            const int32_t k1 = static_cast<int32_t>(b.thresholds[s]) - static_cast<int32_t>(shift);
            if (k1 < 0) return false;
            out = ONodeSlot{{static_cast<uint16_t>(j | (nan_right << 15)), j, j}, {1, 0, 0}, k1 - 32768};
            return true;
        }

        struct SymCollector {
            const Model& model;
            const BinnedForest& f;
            std::vector<ONodeSlot> level_slots;
            std::vector<uint8_t> has_slot;
            std::vector<double> leaves;
            int leaf_depth = -1;
            uint8_t has_cat = 0;

            bool record_slot(int level, const ONodeSlot& sl) {
                if (level >= kMaxSymDepth) return false;
                if (static_cast<size_t>(level) >= level_slots.size()) {
                    level_slots.resize(static_cast<size_t>(level) + 1);
                    has_slot.resize(static_cast<size_t>(level) + 1, 0);
                }
                if (!has_slot[level]) {
                    level_slots[level] = sl;
                    has_slot[level] = 1;
                } else if (std::memcmp(&level_slots[level], &sl, sizeof(ONodeSlot)) != 0) {
                    return false;
                }
                if ((sl.w[0] | sl.w[1] | sl.w[2]) == 0) has_cat = 1;
                return true;
            }

            bool emit_leaf(int32_t ref, int level) {
                if (ref >= 0) return false;
                if (leaf_depth < 0) leaf_depth = level;
                else if (leaf_depth != level) return false;
                leaves.push_back(model.leaf_values[(-ref) - 1]);
                return true;
            }

            bool walk_bnode(int32_t node, int sub, int level) {
                if (node < 0) return emit_leaf(node, level);
                const BNode& b = f.nodes[node];
                ONodeSlot sl{};
                if (!bslot_to_oslot(b, sub, sl, has_cat) || !record_slot(level, sl)) return false;

                if (sub == 0) {
                    const int32_t* cl = &b.children[0];
                    if (cl[0] == cl[1] && cl[0] == cl[2] && cl[0] == cl[3]) {
                        if (!emit_leaf(cl[0], level + 1)) return false;
                    } else if (!walk_bnode(node, 1, level + 1)) {
                        return false;
                    }
                    const int32_t* cr = &b.children[4];
                    if (cr[0] == cr[1] && cr[0] == cr[2] && cr[0] == cr[3]) {
                        return emit_leaf(cr[0], level + 1);
                    }
                    return walk_bnode(node, 2, level + 1);
                }
                if (sub <= 2) {
                    const int first = (sub == 1) ? 0 : 4;
                    if (b.children[first] == b.children[first + 1]) {
                        if (!emit_leaf(b.children[first], level + 1)) return false;
                    } else if (!walk_bnode(node, 2 * sub + 1, level + 1)) {
                        return false;
                    }
                    if (b.children[first + 2] == b.children[first + 3]) {
                        return emit_leaf(b.children[first + 2], level + 1);
                    }
                    return walk_bnode(node, 2 * sub + 2, level + 1);
                }
                const int first = 2 * (sub - 3);
                if (b.children[first] < 0) {
                    if (!emit_leaf(b.children[first], level + 1)) return false;
                } else if (!walk_bnode(b.children[first], 0, level + 1)) {
                    return false;
                }
                if (b.children[first + 1] < 0) {
                    return emit_leaf(b.children[first + 1], level + 1);
                }
                return walk_bnode(b.children[first + 1], 0, level + 1);
            }

            bool walk_onode(int32_t node, int sub, int level) {
                if (node < 0) return emit_leaf(node, level);
                const ONode& o = f.onodes[node];
                if (!record_slot(level, o.s[sub])) return false;

                if (sub == 0) {
                    if (o.children[0] == o.children[1]) {
                        if (!emit_leaf(o.children[0], level + 1)) return false;
                    } else if (!walk_onode(node, 1, level + 1)) {
                        return false;
                    }
                    if (o.children[2] == o.children[3]) {
                        return emit_leaf(o.children[2], level + 1);
                    }
                    return walk_onode(node, 2, level + 1);
                }
                const int first = (sub == 1) ? 0 : 2;
                if (o.children[first] < 0) {
                    if (!emit_leaf(o.children[first], level + 1)) return false;
                } else if (!walk_onode(o.children[first], 0, level + 1)) {
                    return false;
                }
                if (o.children[first + 1] < 0) {
                    return emit_leaf(o.children[first + 1], level + 1);
                }
                return walk_onode(o.children[first + 1], 0, level + 1);
            }
        };

    } // namespace

    std::shared_ptr<const SymForest> detect_symmetric_forest(const Model& model, const BinnedForest& f) {
        if (model.source != model_source::Miter || model.float32 || f.n_trees == 0 || f.row_codes() > 0x8000) {
            return nullptr;
        }
        auto sym = std::make_shared<SymForest>();
        sym->trees.reserve(f.n_trees);
        int max_depth = 0;
        for (size_t t = 0; t < f.n_trees; t++) {
            SymCollector col{model, f, {}, {}, {}, -1, 0};
            const bool is_oblique = (f.tree_kind[t] & tree_kind::Oblique) != 0;
            const bool ok = is_oblique ? col.walk_onode(f.roots[t], 0, 0)
                                       : col.walk_bnode(f.roots[t], 0, 0);
            if (!ok || col.leaf_depth < 0 || col.leaf_depth > kMaxSymDepth) return nullptr;
            if (col.leaves.size() != (size_t{1} << col.leaf_depth)) return nullptr;
            if (col.level_slots.size() != static_cast<size_t>(col.leaf_depth)) return nullptr;

            SymTree st;
            st.slot_offset = static_cast<uint32_t>(sym->slots.size());
            st.leaf_offset = static_cast<uint32_t>(sym->leaves.size());
            st.depth = static_cast<uint8_t>(col.leaf_depth);
            st.has_cat = col.has_cat;
            st.output = model.n_outputs > 1 ? model.tree_outputs[t] : 0;
            sym->slots.insert(sym->slots.end(), col.level_slots.begin(), col.level_slots.end());
            sym->leaves.insert(sym->leaves.end(), col.leaves.begin(), col.leaves.end());
            sym->trees.push_back(st);
            max_depth = std::max(max_depth, col.leaf_depth);
        }
        return max_depth >= 2 ? sym : nullptr;
    }

    std::shared_ptr<const BinnedForest> build_binned(const Model& model) {
        // miter oblique / CTR trees are always double precision.
        if ((model.n_oblique > 0 || model.n_ctr_tables > 0) && model.float32) return nullptr;
        Unpacker unpacker(model);
        unpacker.nodes.reserve(model.n_nodes * 3 + model.n_leaves);
        std::vector<int32_t> bt_roots;
        bt_roots.reserve(model.n_offsets);
        for (size_t t = 0; t < model.n_offsets; t++) bt_roots.push_back(unpacker.from_ref(model.tree_offsets[t]));
        const auto& bt = unpacker.nodes;

        // Dense features with their kind; sorted unique thresholds for numeric and CTR ones.
        std::vector<uint16_t> dense(model.n_features, 0xFFFF);
        std::vector<uint16_t> dense_ctr(model.n_ctr_tables, 0xFFFF);
        std::vector<uint16_t> q16_map(model.n_features, 0xFFFF);
        std::vector<std::vector<double>> cuts;
        auto forest = std::make_shared<BinnedStorage>();
        forest->nodes.reserve(model.n_nodes);
        forest->roots.reserve(model.n_offsets);
        forest->tree_kind.reserve(model.n_offsets);
        std::vector<bool> zero_feature;
        for (const BTNode& n : bt) {
            if (n.leaf) continue;
            if (n.kind == slot_kind::Oblique) {
                const ObliqueSplit& o = model.oblique_splits[std::bit_cast<uint64_t>(n.threshold)];
                for (uint32_t col : o.features) {
                    if (q16_map[col] == 0xFFFF) {
                        if (forest->q16_columns.size() >= 0x7FFF) return nullptr;
                        q16_map[col] = static_cast<uint16_t>(forest->q16_columns.size());
                        forest->q16_columns.push_back(col);
                    }
                }
                continue;
            }
            if (n.kind == slot_kind::Ctr) {
                uint16_t& slot = dense_ctr[n.feature];
                if (slot == 0xFFFF) {
                    if (cuts.size() >= 0xFFFF) return nullptr;
                    slot = static_cast<uint16_t>(cuts.size());
                    cuts.emplace_back();
                    zero_feature.push_back(false);
                    forest->feature_of.push_back(n.feature);
                    forest->feature_kind.push_back(feature_kind::Ctr);
                }
                cuts[slot].push_back(n.threshold);
                continue;
            }

            uint8_t kind = feature_kind::Numeric;
            if (n.kind != slot_kind::Numeric) {
                kind = (n.dirs & dir_bits::XGBoostCategory) ? feature_kind::XGBoostCategory : feature_kind::LightGBMCategory;
                if (category_bound(n.kind, std::bit_cast<uint64_t>(n.threshold)) > kInvalidCat) return nullptr;
            }

            uint16_t& slot = dense[n.feature];
            if (slot == 0xFFFF) {
                if (cuts.size() >= 0xFFFF) return nullptr;
                slot = static_cast<uint16_t>(cuts.size());
                cuts.emplace_back();
                zero_feature.push_back(false);
                forest->feature_of.push_back(n.feature);
                forest->feature_kind.push_back(kind);
            } else if (forest->feature_kind[slot] != kind) {
                return nullptr;  // the same column used with different semantics
            }
            if (kind == feature_kind::Numeric) {
                if (std::isnan(n.threshold)) return nullptr;
                cuts[slot].push_back(n.threshold);
            }
            if (n.dirs & dir_bits::ZeroIsMissing) zero_feature[slot] = true;
        }
        // LightGBM zeroes |x| <= kZeroThreshold; parsers never place a threshold inside (-kZeroThreshold, kZeroThreshold],
        // so with both band edges as cuts the band is exactly one bin.
        forest->zero_bin.assign(cuts.size(), kNanBin);
        for (size_t j = 0; j < cuts.size(); j++) {
            auto& c = cuts[j];
            if (zero_feature[j]) {
                c.push_back(-kZeroThreshold);
                c.push_back(std::nextafter(kZeroThreshold, INFINITY));
            }
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
            if (c.size() >= kInvalidCat - 1) return nullptr;  // bin + 1 must stay below kInvalidCat
            if (zero_feature[j]) {
                const auto lo = std::lower_bound(c.begin(), c.end(), -kZeroThreshold) - c.begin();
                const auto hi = std::lower_bound(c.begin(), c.end(), std::nextafter(kZeroThreshold, INFINITY)) - c.begin();
                if (hi != lo + 1) return nullptr;  // a threshold inside the band: not representable
                forest->zero_bin[j] = static_cast<uint16_t>(lo + 1);
            }
        }
        forest->cut_offsets.push_back(0);
        for (const auto& c : cuts) {
            forest->cuts.insert(forest->cuts.end(), c.begin(), c.end());
            forest->cut_offsets.push_back(static_cast<uint32_t>(forest->cuts.size()));
        }

        Packer packer(bt, *forest, dense, dense_ctr, cuts);
        OPacker opacker(model, bt, *forest, dense, dense_ctr, cuts, q16_map, cuts.size());
        try {
            for (size_t t = 0; t < bt_roots.size(); t++) {
                const bool oblique = model.tree_kinds && (model.tree_kinds[t] & tree_kind::Oblique) != 0;
                if (oblique) {
                    opacker.tree_kind = 0;
                    forest->roots.push_back(opacker.pack(bt_roots[t]));
                    forest->tree_kind.push_back(opacker.tree_kind);
                } else {
                    packer.tree_kind = 0;
                    forest->roots.push_back(packer.pack(bt_roots[t]));
                    forest->tree_kind.push_back(packer.tree_kind);
                }
            }
        } catch (const std::length_error&) {
            return nullptr;  // more categorical slots than a 16-bit index can address
        } catch (const std::domain_error&) {
            return nullptr;  // an oblique tree the ONode form does not cover: KNode kernel
        }
        if (!build_decision_tables(model, *forest)) return nullptr;

        auto view = std::make_shared<BinnedForest>();
        view->nodes = forest->nodes.data();
        view->n_nodes = forest->nodes.size();
        view->roots = forest->roots.data();
        view->tree_kind = forest->tree_kind.data();
        view->n_trees = forest->roots.size();
        view->feature_of = forest->feature_of.data();
        view->feature_kind = forest->feature_kind.data();
        view->zero_bin = forest->zero_bin.data();
        view->cut_offsets = forest->cut_offsets.data();
        view->n_features = forest->feature_of.size();
        view->cuts = forest->cuts.data();
        view->n_cuts = forest->cuts.size();
        view->cat_features = forest->cat_features.data();
        view->n_cat_features = forest->cat_features.size();
        view->cat_tables = forest->cat_tables.data();
        view->n_table_words = forest->cat_tables.size();
        view->decision_words = forest->decision_words;
        view->onodes = forest->onodes.data();
        view->n_onodes = forest->onodes.size();
        view->q16_columns = forest->q16_columns.data();
        view->n_q16 = forest->q16_columns.size();
        forest->cat_slots.clear();
        forest->cat_slots.shrink_to_fit();
        view->has_ctr_features = compute_has_ctr_features(*view);
        view->sym = detect_symmetric_forest(model, *view);
        view->fits_u8_codes = compute_fits_u8_codes(*view);
        view->owner = forest;
        return view;
    }

} // namespace lignum
