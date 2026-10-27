#include "validate.hpp"
#include "binned.hpp"

#include <algorithm>
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

        void validate_knodes(const Model& m) {
            if (m.n_outputs == 0) fail("no outputs");
            if (m.transform > transform::Argmax) fail("unknown transform");
            if (m.n_nodes > 0 && m.n_features == 0) fail("splits without features");

            for (size_t i = 0; i < m.n_nodes; i++) {
                const KNode& kn = m.k_nodes[i];
                for (int c = 0; c < 4; c++) check_child(static_cast<int64_t>(i), kn.children[c], m.n_nodes, m.n_leaves, "node");
                for (int s = 0; s < 3; s++) {
                    if (kn.features[s] >= m.n_features) fail("node feature out of range");
                    if (kn.kinds[s] > slot_kind::CatBitset) fail("unknown slot kind");
                    if (kn.kinds[s] == slot_kind::CatBitset) {
                        uint64_t payload;
                        std::memcpy(&payload, &kn.thresholds[s], sizeof(payload));
                        const uint64_t lo = payload & 0xFFFFFFFFu, words = payload >> 32;
                        if (lo + words > m.n_cat_words) fail("categorical bitset out of range");
                    }
                }
            }
            for (size_t t = 0; t < m.n_offsets; t++) {
                check_child(-1, m.tree_offsets[t], m.n_nodes, m.n_leaves, "tree root");
                if (m.n_outputs > 1 && m.tree_outputs[t] >= m.n_outputs) fail("tree output out of range");
            }
        }

        void validate_binned(const Model& m, const BinnedForest& f) {
            if (f.n_trees != m.n_offsets) fail("binned tree count differs from the model");
            if (uint64_t{f.decision_words} * 64 > 0x10000) fail("too many categorical decision bits");
            if (f.n_nodes > 0 && f.n_features == 0) fail("binned splits without features");

            for (size_t j = 0; j < f.n_features; j++) {
                if (f.feature_of[j] >= m.n_features) fail("binned feature out of range");
                if (f.feature_kind[j] > feature_kind::XGBoostCategory) fail("unknown binned feature kind");
            }
            if (f.cut_offsets[0] != 0 || f.cut_offsets[f.n_features] != f.n_cuts) fail("binned cut offsets");
            for (size_t j = 0; j < f.n_features; j++) {
                if (f.cut_offsets[j + 1] < f.cut_offsets[j]) fail("binned cut offsets");
                if (f.cut_offsets[j + 1] - f.cut_offsets[j] >= kInvalidCat - 1) fail("too many cuts for a feature");
            }
            for (size_t k = 0; k < f.n_cat_features; k++) {
                const BinnedCatFeature& cf = f.cat_features[k];
                if (cf.dense >= f.n_features || f.feature_kind[cf.dense] == feature_kind::Numeric) fail("categorical feature out of range");
                if (cf.n_categories > kInvalidCat) fail("too many categories");
                if (uint64_t{cf.row_offset} + cf.words > f.decision_words) fail("categorical decision row out of range");
                if (cf.table_offset + uint64_t{cf.n_categories} * cf.words > f.n_table_words) fail("categorical table out of range");
            }
            for (size_t i = 0; i < f.n_nodes; i++) {
                const BNode& b = f.nodes[i];
                for (int c = 0; c < 8; c++) check_child(static_cast<int64_t>(i), b.children[c], f.n_nodes, m.n_leaves, "binned node");
                for (int s = 0; s < 7; s++) {
                    if (b.features[s] >= f.n_features) fail("binned node feature out of range");
                    if (((b.cat_slots >> s) & 1) && b.thresholds[s] >= uint32_t{f.decision_words} * 64) fail("decision bit out of range");
                }
            }
            for (size_t t = 0; t < f.n_trees; t++) check_child(-1, f.roots[t], f.n_nodes, m.n_leaves, "binned tree root");
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
