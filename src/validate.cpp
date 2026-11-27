#include "validate.hpp"
#include "binned.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace lignum {

    namespace {

        [[noreturn]] void fail(const std::string& what) {
            throw std::runtime_error("Invalid model file: " + what + ".");
        }

        // A child reference is a later node, or a leaf.
        void check_child(int64_t parent, int32_t child, size_t n_nodes, size_t n_leaves, const char* where) {
            if (child >= 0) {
                if (child <= parent || static_cast<size_t>(child) >= n_nodes) fail(std::string(where) + " child out of range");
            } else if (static_cast<size_t>(-(static_cast<int64_t>(child) + 1)) >= n_leaves) {
                fail(std::string(where) + " leaf out of range");
            }
        }

        // Every node is referenced once, by one slot of one parent or as a tree's root: the writers
        // lay out trees, never shared subtrees. The binned packer, the unpacking into the binned
        // form and the code generator recurse per reference, so a node referenced twice is
        // expanded twice (and a chain of them exponentially often), and the unpacking takes two
        // equal grandchild pointers for a leaf (a node index taken for a leaf index loops).
        class SingleReferences {
            public:
                explicit SingleReferences(size_t n_nodes, const char* what) : seen_(n_nodes, 0), what_(what) {}
                void add(int32_t child) {
                    if (child < 0) return;  // leaves may repeat: a leaf child fills both grandchildren
                    if (seen_[static_cast<size_t>(child)]++) fail(std::string(what_) + " referenced more than once");
                }
            private:
                std::vector<uint8_t> seen_;
                const char* what_;
        };

        constexpr int kMaxTreeDepth = 1024;

        void validate_knodes(const Model& m) {
            if (m.n_outputs == 0) fail("no outputs");
            if (m.transform > transform::Last) fail("unknown transform");
            // Each transform is implemented for its library's arithmetic only (Hinge: XGBoost,
            // float32; SignedSquare, Log1pExp: LightGBM, double).
            if ((m.transform == transform::Hinge && !m.float32) ||
                ((m.transform == transform::SignedSquare || m.transform == transform::Log1pExp) && m.float32)) {
                fail("transform does not match the model's precision");
            }
            if (m.n_nodes > 0 && m.n_features == 0) fail("splits without features");
            // Word 0 of the bitsets is the zero target of branch-free categorical lookups (the
            // builder always writes it), so the kernel may read it for any categorical slot.
            if (m.n_cat_words == 0 || m.cat_bitsets[0] != 0) fail("invalid categorical bitset words");

            if (m.n_oblique > 0 && m.n_quantizers < m.n_features) fail("oblique splits without quantizers");
            if ((m.n_oblique > 0 || m.n_ctr_tables > 0) && m.float32) fail("oblique or CTR splits in a float32 model");
            for (size_t j = 0; j < m.n_quantizers; j++) {
                if (m.quantizer_constant[j] > 1) fail("unknown quantizer flag");
                if (!std::isfinite(m.quantizers[j].lo) || !std::isfinite(m.quantizers[j].scale)) fail("quantizer not finite");
            }
            if (m.n_quantizer_knots > 0) {
                if (m.n_quantizers == 0 || !m.quantizer_knot_offsets ||
                    m.quantizer_knot_offsets[0] != 0 || m.quantizer_knot_offsets[m.n_quantizers] != m.n_quantizer_knots) {
                    fail("quantizer knot offsets");
                }
                for (size_t j = 0; j < m.n_quantizers; j++) {
                    const uint32_t lo = m.quantizer_knot_offsets[j];
                    const uint32_t hi = m.quantizer_knot_offsets[j + 1];
                    if (hi < lo) fail("quantizer knot offsets");
                    const uint32_t count = hi - lo;
                    if (count != 0 && (count < 2 || count > 257)) fail("invalid quantizer knot count");
                    for (uint32_t k = lo; k < hi; k++) {
                        if (!std::isfinite(m.quantizer_knots[k])) fail("quantizer knot not finite");
                        if (k > lo && !(m.quantizer_knots[k - 1] < m.quantizer_knots[k])) fail("quantizer knots not strictly increasing");
                    }
                }
            }
            for (size_t i = 0; i < m.n_oblique; i++) {
                const ObliqueSplit& o = m.oblique_splits[i];
                if (o.k < 2 || o.k > 3) fail("oblique split with an unsupported number of inputs");
                // |w| <= 2^14 keeps P = sum w * (code - 32768) inside int32 (3 * 2^14 * 2^15 < 2^31).
                for (int16_t w : o.weights) {
                    if (w < -16384 || w > 16384) fail("oblique weight out of range");
                }
                for (uint32_t f : o.features) {
                    if (f >= m.n_features) fail("oblique input out of range");
                }
            }
            for (size_t i = 0; i < m.n_ctr_tables; i++) {
                const CtrTableHeader& t = m.ctr_tables[i];
                if (t.n_cols == 0 || t.n_cols > 3) fail("CTR table projection out of range");
                for (uint8_t c = 0; c < t.n_cols; c++) {
                    if (t.columns[c] >= m.n_features) fail("CTR table column out of range");
                }
                if (!std::isfinite(t.default_value)) fail("CTR default value not finite");
                if (t.keys_offset > m.n_ctr_keys || t.n_slots > m.n_ctr_keys - t.keys_offset ||
                    t.slots_offset > m.n_ctr_slots || t.n_slots > m.n_ctr_slots - t.slots_offset ||
                    t.values_offset > m.n_ctr_values || t.n_values > m.n_ctr_values - t.values_offset) {
                    fail("CTR table offsets out of range");
                }
                if (t.n_slots == 0) {
                    if (t.mask != 0 || t.n_values != 0) fail("empty CTR table has non-zero mask or values");
                } else {
                    if (t.n_slots < 2 || !std::has_single_bit(t.n_slots) || t.mask != t.n_slots - 1) {
                        fail("CTR table hash capacity invalid");
                    }
                    bool has_empty = false;
                    for (size_t s = 0; s < t.n_slots; s++) {
                        const uint32_t entry = m.ctr_slots[t.slots_offset + s];
                        if (entry == 0) has_empty = true;
                        else if (entry > t.n_values) fail("CTR slot value index out of range");
                    }
                    if (!has_empty) fail("CTR table has no empty slot");
                }
                for (size_t v = 0; v < t.n_values; v++) {
                    if (!std::isfinite(m.ctr_values[t.values_offset + v])) fail("CTR table value not finite");
                }
            }

            for (size_t i = 0; i < m.n_nodes; i++) {
                const KNode& kn = m.k_nodes[i];
                for (int c = 0; c < 4; c++) check_child(static_cast<int64_t>(i), kn.children[c], m.n_nodes, m.n_leaves, "node");
                for (int s = 0; s < 3; s++) {
                    if (kn.kinds[s] > slot_kind::Ctr) fail("unknown slot kind");
                    if (kn.kinds[s] == slot_kind::Ctr) {
                        if (kn.features[s] >= m.n_ctr_tables) fail("CTR table index out of range");
                        if (!std::isfinite(kn.thresholds[s])) fail("non-finite CTR split threshold");
                    } else if (kn.kinds[s] != slot_kind::Oblique && kn.features[s] >= m.n_features) {
                        fail("node feature out of range");
                    }
                    if (!m.has_special_trees &&
                        (kn.kinds[s] != slot_kind::Numeric || (kn.dirs[s] & dir_bits::ZeroIsMissing) != 0)) {
                        fail("special split without special-trees flag");
                    }
                    if (kn.kinds[s] == slot_kind::Numeric && std::isnan(kn.thresholds[s])) {
                        fail("NaN split threshold");
                    }
                    if (kn.kinds[s] == slot_kind::Oblique) {
                        uint64_t index;
                        std::memcpy(&index, &kn.thresholds[s], sizeof(index));
                        if (index >= m.n_oblique) fail("oblique split out of range");
                    }
                    if (kn.kinds[s] == slot_kind::CatBitset) {
                        uint64_t payload;
                        std::memcpy(&payload, &kn.thresholds[s], sizeof(payload));
                        const uint64_t lo = payload & 0xFFFFFFFFu, words = payload >> 32;
                        if (lo > m.n_cat_words || words > m.n_cat_words - lo) fail("categorical bitset out of range");
                    }
                }
            }
            if (m.n_offsets > 0 && !m.tree_kinds) fail("missing tree kinds");
            for (size_t t = 0; t < m.n_offsets; t++) {
                check_child(-1, m.tree_offsets[t], m.n_nodes, m.n_leaves, "tree root");
                if ((m.tree_kinds[t] & ~tree_kind::All) != 0) fail("unknown tree kind");
                if (!m.has_special_trees && m.tree_kinds[t] != tree_kind::Numeric) fail("special tree without special-trees flag");
                if (m.n_outputs > 1 && m.tree_outputs[t] >= m.n_outputs) fail("tree output out of range");
            }
            // In range now (checked above).
            SingleReferences refs(m.n_nodes, "a node");
            std::vector<int> depth(m.n_nodes, 0);
            for (size_t t = 0; t < m.n_offsets; t++) {
                refs.add(m.tree_offsets[t]);
                if (m.tree_offsets[t] >= 0) depth[static_cast<size_t>(m.tree_offsets[t])] = 1;
            }
            for (size_t i = 0; i < m.n_nodes; i++) {
                const int d = depth[i];
                if (d > kMaxTreeDepth) fail("tree deeper than " + std::to_string(kMaxTreeDepth) + " KNodes");
                for (int c = 0; c < 4; c++) {
                    const int32_t child = m.k_nodes[i].children[c];
                    refs.add(child);
                    if (child >= 0) {
                        depth[static_cast<size_t>(child)] = std::max(depth[static_cast<size_t>(child)], d + 1);
                    }
                }
            }
            // A tree carries Oblique / Ctr iff a reachable used slot in it has that kind (kernels
            // and the binned packer choose the traversal from tree_kinds).
            if (m.tree_kinds) {
                std::vector<int32_t> stack;
                for (size_t t = 0; t < m.n_offsets; t++) {
                    bool oblique = false, ctr = false;
                    stack.assign(1, m.tree_offsets[t]);
                    while (!stack.empty() && (!oblique || !ctr)) {
                        const int32_t i = stack.back();
                        stack.pop_back();
                        if (i < 0) continue;
                        const KNode& kn = m.k_nodes[i];
                        auto check_slot = [&](int s) {
                            oblique |= kn.kinds[s] == slot_kind::Oblique;
                            ctr |= kn.kinds[s] == slot_kind::Ctr;
                        };
                        check_slot(0);
                        if (kn.children[0] != kn.children[1]) check_slot(1);
                        if (kn.children[2] != kn.children[3]) check_slot(2);
                        for (int c = 0; c < 4; c++) stack.push_back(kn.children[c]);
                    }
                    if (oblique != ((m.tree_kinds[t] & tree_kind::Oblique) != 0)) {
                        fail("tree kind does not match its oblique splits");
                    }
                    if (ctr != ((m.tree_kinds[t] & tree_kind::Ctr) != 0)) {
                        fail("tree kind does not match its CTR splits");
                    }
                }
            }
        }

        void validate_binned(const Model& m, const BinnedForest& f) {
            if (f.n_trees != m.n_offsets) fail("binned tree count differs from the model");
            if (uint64_t{f.decision_words} * 64 > 0x10000) fail("too many categorical decision bits");
            if ((f.n_nodes > 0 || f.n_onodes > 0) && f.n_features == 0 && f.n_q16 == 0) fail("binned splits without features");

            for (size_t j = 0; j < f.n_features; j++) {
                if (f.feature_kind[j] > feature_kind::Ctr) fail("unknown binned feature kind");
                if (f.feature_kind[j] == feature_kind::Ctr) {
                    if (m.float32) fail("binned CTR feature in a float32 model");
                    if (f.feature_of[j] >= m.n_ctr_tables) fail("binned CTR feature out of range");
                } else if (f.feature_of[j] >= m.n_features) {
                    fail("binned feature out of range");
                }
            }
            if (f.cut_offsets[0] != 0 || f.cut_offsets[f.n_features] != f.n_cuts) fail("binned cut offsets");
            for (size_t j = 0; j < f.n_features; j++) {
                if (f.cut_offsets[j + 1] < f.cut_offsets[j]) fail("binned cut offsets");
                if (f.cut_offsets[j + 1] - f.cut_offsets[j] >= kInvalidCat - 1) fail("too many cuts for a feature");
                for (uint32_t k = f.cut_offsets[j]; k < f.cut_offsets[j + 1]; k++) {
                    if (std::isnan(f.cuts[k])) fail("NaN binned cut");
                    if (k > f.cut_offsets[j] && !(f.cuts[k - 1] < f.cuts[k])) fail("binned cuts are not strictly increasing");
                }
            }
            for (size_t k = 0; k < f.n_cat_features; k++) {
                const BinnedCatFeature& cf = f.cat_features[k];
                if (cf.dense >= f.n_features ||
                    (f.feature_kind[cf.dense] != feature_kind::LightGBMCategory &&
                     f.feature_kind[cf.dense] != feature_kind::XGBoostCategory)) {
                    fail("categorical feature out of range");
                }
                if (cf.n_categories > kInvalidCat) fail("too many categories");
                if (uint64_t{cf.row_offset} + cf.words > f.decision_words) fail("categorical decision row out of range");
                // Written so that a huge table_offset cannot wrap the sum around.
                if (cf.table_offset > f.n_table_words || uint64_t{cf.n_categories} * cf.words > f.n_table_words - cf.table_offset) {
                    fail("categorical table out of range");
                }
            }
            bool any_cat_tree = false;
            auto in_onodes = [&](size_t t) { return (f.tree_kind[t] & tree_kind::Oblique) != 0; };
            for (size_t t = 0; t < f.n_trees; t++) {
                if ((f.tree_kind[t] & ~tree_kind::All) != 0) fail("unknown binned tree kind");
                if (in_onodes(t)) check_child(-1, f.roots[t], f.n_onodes, m.n_leaves, "binned oblique tree root");
                else              check_child(-1, f.roots[t], f.n_nodes, m.n_leaves, "binned tree root");
                if (f.tree_kind[t] & tree_kind::Categorical) any_cat_tree = true;
            }
            if (any_cat_tree && f.decision_words == 0) fail("categorical trees require decision words");
            for (size_t i = 0; i < f.n_nodes; i++) {
                const BNode& b = f.nodes[i];
                for (int c = 0; c < 8; c++) check_child(static_cast<int64_t>(i), b.children[c], f.n_nodes, m.n_leaves, "binned node");
                for (int s = 0; s < 7; s++) {
                    if (b.features[s] >= f.n_features) fail("binned node feature out of range");
                    if (((b.cat_slots >> s) & 1) && b.thresholds[s] >= uint32_t{f.decision_words} * 64) fail("decision bit out of range");
                }
            }
            if (f.n_onodes > 0 || f.n_q16 > 0) {
                if (m.n_oblique == 0 || m.float32) fail("binned oblique trees in a model without oblique splits");
                const size_t row_codes = f.row_codes();
                for (size_t q = 0; q < f.n_q16; q++) {
                    if (f.q16_columns[q] >= m.n_features || f.q16_columns[q] >= m.n_quantizers) fail("Q16 column out of range");
                }
                for (size_t i = 0; i < f.n_onodes; i++) {
                    const ONode& o = f.onodes[i];
                    for (int c = 0; c < 4; c++) check_child(static_cast<int64_t>(i), o.children[c], f.n_onodes, m.n_leaves, "binned oblique node");
                    for (const ONodeSlot& sl : o.s) {
                        if ((sl.f[0] & 0x7FFFu) >= row_codes || sl.f[1] >= row_codes || sl.f[2] >= row_codes) {
                            fail("ONode code column out of range");
                        }
                        for (int16_t w : sl.w) {
                            if (w < -16384 || w > 16384) fail("ONode weight out of range");
                        }
                        if (sl.w[0] == 0 && sl.w[1] == 0 && sl.w[2] == 0 &&
                            (sl.thr < 0 || static_cast<uint64_t>(sl.thr) >= uint64_t{f.decision_words} * 64)) {
                            fail("ONode decision bit out of range");
                        }
                    }
                }
                SingleReferences orefs(f.n_onodes, "a binned oblique node");
                std::vector<int> odepth(f.n_onodes, 0);
                for (size_t t = 0; t < f.n_trees; t++) {
                    if (in_onodes(t)) {
                        orefs.add(f.roots[t]);
                        if (f.roots[t] >= 0) odepth[static_cast<size_t>(f.roots[t])] = 1;
                    }
                }
                for (size_t i = 0; i < f.n_onodes; i++) {
                    const int d = odepth[i];
                    if (d > kMaxTreeDepth) fail("binned oblique tree deeper than " + std::to_string(kMaxTreeDepth) + " ONodes");
                    for (int c = 0; c < 4; c++) {
                        const int32_t child = f.onodes[i].children[c];
                        orefs.add(child);
                        if (child >= 0) {
                            odepth[static_cast<size_t>(child)] = std::max(odepth[static_cast<size_t>(child)], d + 1);
                        }
                    }
                }
            }
            SingleReferences refs(f.n_nodes, "a binned node");
            std::vector<int> depth(f.n_nodes, 0);
            for (size_t t = 0; t < f.n_trees; t++) {
                if (!in_onodes(t)) {
                    refs.add(f.roots[t]);
                    if (f.roots[t] >= 0) depth[static_cast<size_t>(f.roots[t])] = 1;
                }
            }
            for (size_t i = 0; i < f.n_nodes; i++) {
                const int d = depth[i];
                if (d > kMaxTreeDepth) fail("binned tree deeper than " + std::to_string(kMaxTreeDepth) + " BNodes");
                for (int c = 0; c < 8; c++) {
                    const int32_t child = f.nodes[i].children[c];
                    refs.add(child);
                    if (child >= 0) {
                        depth[static_cast<size_t>(child)] = std::max(depth[static_cast<size_t>(child)], d + 1);
                    }
                }
            }
        }

    } // namespace

    void validate_model(const Model& model) {
        validate_knodes(model);
        if (model.binned) validate_binned(model, *model.binned);
    }

    std::vector<std::string> model_feature_names(std::vector<std::string> names, uint32_t n_features, uint8_t source) {
        if (names.empty() || std::any_of(names.begin(), names.end(), [](const std::string& n) { return n.empty(); })) return {};
        if (source == model_source::LightGBM) {
            bool defaults = true;
            for (size_t i = 0; i < names.size() && defaults; i++) defaults = names[i] == "Column_" + std::to_string(i);
            if (defaults) return {};
        }
        if (names.size() < n_features) {
            fail("the model names " + std::to_string(names.size()) + " features but its trees use " + std::to_string(n_features));
        }
        return names;
    }

    void check_feature_names(const Model& model, const std::vector<std::string>& columns) {
        const auto& names = model.feature_names;
        if (names.empty()) return;
        auto same = [&](const std::string& column, const std::string& name) {
            if (column == name) return true;
            if (model.source != model_source::LightGBM || column.size() != name.size()) return false;
            std::string spaced = column;
            std::replace(spaced.begin(), spaced.end(), ' ', '_');
            return spaced == name;
        };
        size_t k = 0;
        while (k < names.size() && k < columns.size() && same(columns[k], names[k])) k++;
        if (k == names.size()) return;
        std::string message = k < columns.size()
            ? "Column " + std::to_string(k) + " is '" + columns[k] + "' but the model expects '" + names[k] + "' there"
            : "The input has " + std::to_string(columns.size()) + " columns but the model has " + std::to_string(names.size()) +
                  " features (the next one is '" + names[k] + "')";
        throw std::invalid_argument(message + "; the model's columns are model.feature_names, in that order. Select them with "
                                    "X[model.feature_names] (pandas) or table.select(model.feature_names) (pyarrow), or pass "
                                    "validate_features=False to use the columns by position.");
    }

} // namespace lignum
