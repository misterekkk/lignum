#include <lignum/build.hpp>

#include "binned.hpp"
#include "builder.hpp"
#include "ir.hpp"
#include "validate.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lignum {

    namespace {

        [[noreturn]] void invalid(const std::string& what) {
            throw std::invalid_argument("build_model: " + what);
        }

        // Children come after their parent, so the walk below always ends and every node has
        // at most one parent (checked: a node reached twice is rejected).
        std::unique_ptr<TempBinNode> convert(const ir::Forest& f, const ir::Tree& tree, uint32_t i,
                                             int depth, std::vector<uint8_t>& seen) {
            if (depth > 1024) invalid("tree deeper than 1024 nodes");
            if (i >= tree.nodes.size()) invalid("child index out of range");
            if (seen[i]++) invalid("node referenced more than once");
            const ir::Node& n = tree.nodes[i];
            auto out = std::make_unique<TempBinNode>();
            if (n.kind == ir::Node::Kind::Leaf) {
                if (!std::isfinite(n.value)) invalid("leaf value is not finite");
                out->is_leaf = true;
                out->threshold_or_value = n.value;
                return out;
            }
            if (n.left <= i || n.right <= i) invalid("children must come after their parent");
            out->default_dir = n.default_right ? 1 : 0;
            switch (n.kind) {
                case ir::Node::Kind::Numeric:
                    if (n.feature >= f.n_features) invalid("split feature out of range");
                    if (std::isnan(n.value)) invalid("NaN threshold");
                    out->feature_id = static_cast<int32_t>(n.feature);
                    out->threshold_or_value = n.value;
                    break;
                case ir::Node::Kind::Categorical:
                    if (n.feature >= f.n_features) invalid("split feature out of range");
                    out->feature_id = static_cast<int32_t>(n.feature);
                    out->categorical = true;
                    out->categories = n.categories;
                    out->default_dir = 1;  // LightGBM semantics: NaN goes right (as in every kernel)
                    break;
                case ir::Node::Kind::Oblique: {
                    const ObliqueSplit& o = n.oblique;
                    if (o.k < 2 || o.k > 3) invalid("oblique splits have 2 or 3 inputs");
                    for (uint32_t c : o.features) {
                        if (c >= f.n_features) invalid("oblique input out of range");
                    }
                    if (o.k == 2 && (o.weights[2] != 0 || o.features[2] != o.features[0])) {
                        invalid("an oblique split with k = 2 must repeat features[0] with weight 0");
                    }
                    if (o.weights[0] == 0 && o.weights[1] == 0 && o.weights[2] == 0) {
                        invalid("oblique split must have at least one non-zero weight");
                    }
                    for (int16_t w : o.weights) {
                        if (w < -16384 || w > 16384) invalid("oblique weight outside [-2^14, 2^14]");
                    }
                    out->oblique = true;
                    out->oblique_split = o;
                    out->feature_id = static_cast<int32_t>(o.features[0]);
                    break;
                }
                case ir::Node::Kind::Ctr: {
                    if (n.feature >= f.ctr_tables.size() || n.feature > 65535u) {
                        invalid("CTR table index out of range");
                    }
                    if (!std::isfinite(n.value) || !std::isfinite(static_cast<float>(n.value))) {
                        invalid("CTR threshold is not finite");
                    }
                    out->ctr = true;
                    out->default_dir = 0;
                    out->feature_id = static_cast<int32_t>(n.feature);
                    out->threshold_or_value = static_cast<double>(static_cast<float>(n.value));
                    break;
                }
                default:
                    invalid("unknown node kind");
            }
            out->left = convert(f, tree, n.left, depth + 1, seen);
            out->right = convert(f, tree, n.right, depth + 1, seen);
            return out;
        }

        void populate_ctr_tables(const ir::Forest& forest, ModelInfo& info) {
            if (forest.ctr_tables.empty()) return;
            if (forest.ctr_tables.size() > 65536) invalid("too many CTR tables");
            for (const ir::CtrTable& t : forest.ctr_tables) {
                if (t.columns.empty() || t.columns.size() > 3) invalid("CTR table projection must have 1 to 3 columns");
                for (uint32_t col : t.columns) {
                    if (col >= forest.n_features) invalid("CTR table column out of range");
                }
                if (!std::isfinite(t.default_value)) invalid("CTR default value is not finite");
                if (t.keys.size() != t.slots.size()) invalid("CTR table keys and slots lengths differ");
                if (t.keys.empty()) {
                    if (t.mask != 0 || !t.values.empty()) invalid("empty CTR table must have empty values and zero mask");
                } else {
                    if (t.keys.size() < 2 || !std::has_single_bit(t.keys.size())) invalid("CTR table size must be a power of 2 >= 2");
                    if (t.mask != t.keys.size() - 1) invalid("CTR table mask does not match table size");
                    bool has_empty = false;
                    for (uint32_t s : t.slots) {
                        if (s == 0) has_empty = true;
                        else if (s > t.values.size()) invalid("CTR slot value index out of range");
                    }
                    if (!has_empty) invalid("CTR hash table must contain at least one empty slot");
                }
                for (float v : t.values) {
                    if (!std::isfinite(v)) invalid("CTR table value is not finite");
                }

                CtrTableHeader hdr{};
                hdr.n_cols = static_cast<uint8_t>(t.columns.size());
                for (size_t c = 0; c < t.columns.size(); c++) hdr.columns[c] = t.columns[c];
                hdr.default_value = t.default_value;
                hdr.mask = t.mask;
                hdr.n_slots = static_cast<uint32_t>(t.slots.size());
                hdr.n_values = static_cast<uint32_t>(t.values.size());
                hdr.keys_offset = info.ctr_keys.size();
                hdr.slots_offset = info.ctr_slots.size();
                hdr.values_offset = info.ctr_values.size();
                info.ctr_tables.push_back(hdr);
                info.ctr_keys.insert(info.ctr_keys.end(), t.keys.begin(), t.keys.end());
                info.ctr_slots.insert(info.ctr_slots.end(), t.slots.begin(), t.slots.end());
                info.ctr_values.insert(info.ctr_values.end(), t.values.begin(), t.values.end());
            }
        }

    } // namespace

    Model build_model(const ir::Forest& forest, bool binned) {
        if (forest.source > model_source::Miter) invalid("unknown model source");
        const bool supported_transform = forest.transform == transform::Identity || forest.transform == transform::Sigmoid ||
                                         forest.transform == transform::Softmax || forest.transform == transform::Exp ||
                                         forest.transform == transform::Log1pExp || forest.transform == transform::Argmax;
        if (!supported_transform) invalid("unsupported output transform");

        const std::vector<double> single_base{forest.base_score};
        const std::vector<double>& base_scores = forest.base_scores.empty() ? single_base : forest.base_scores;
        if (base_scores.empty() || base_scores.size() > 65535) invalid("base_scores count out of range");
        for (double b : base_scores) {
            if (!std::isfinite(b)) invalid("base score is not finite");
        }
        if (base_scores.size() == 1) {
            if (!forest.tree_outputs.empty()) invalid("single-output model must not provide tree_outputs");
        } else {
            if (forest.tree_outputs.size() != forest.trees.size()) invalid("tree_outputs must match tree count");
            for (uint16_t o : forest.tree_outputs) {
                if (o >= base_scores.size()) invalid("tree output out of range");
            }
        }

        if (!forest.quantizers.empty() &&
            (forest.quantizers.size() != forest.n_features || forest.quantizer_constant.size() != forest.n_features)) {
            invalid("quantizers must have one entry per input column");
        }
        for (const FeatureQuantizer& q : forest.quantizers) {
            if (!std::isfinite(q.lo) || !std::isfinite(q.scale)) invalid("quantizer is not finite");
        }
        for (uint8_t c : forest.quantizer_constant) {
            if (c > 1) invalid("quantizer_constant must be 0 or 1");
        }
        if (!forest.quantizer_knots.empty() && forest.quantizer_knots.size() != forest.n_features) {
            invalid("quantizer_knots must be empty or have n_features entries");
        }

        ModelInfo info;
        info.base_scores = base_scores;
        info.tree_outputs = forest.tree_outputs;
        info.transform = forest.transform;
        info.feature_names = forest.feature_names;
        info.objective = forest.objective;
        populate_ctr_tables(forest, info);

        std::vector<std::unique_ptr<TempBinNode>> temp;
        temp.reserve(forest.trees.size());
        bool any_oblique = false;
        for (const ir::Tree& tree : forest.trees) {
            if (tree.nodes.empty()) invalid("empty tree");
            for (const ir::Node& n : tree.nodes) any_oblique |= n.kind == ir::Node::Kind::Oblique;
            std::vector<uint8_t> seen(tree.nodes.size(), 0);
            temp.push_back(convert(forest, tree, 0, 0, seen));
        }
        if (any_oblique && forest.quantizers.empty()) invalid("oblique splits need quantizers");

        if (any_oblique) {
            info.quantizers = forest.quantizers;
            info.quantizer_constant = forest.quantizer_constant;
            if (!forest.quantizer_knots.empty()) {
                bool any_knots = false;
                for (const auto& knots : forest.quantizer_knots) any_knots |= !knots.empty();
                if (any_knots) {
                    info.quantizer_knot_offsets.reserve(forest.n_features + 1);
                    info.quantizer_knot_offsets.push_back(0);
                    for (const auto& knots : forest.quantizer_knots) {
                        if (!knots.empty()) {
                            if (knots.size() < 2 || knots.size() > 257) invalid("quantizer knots must have 2 to 257 entries");
                            for (size_t k = 0; k < knots.size(); k++) {
                                if (!std::isfinite(knots[k])) invalid("quantizer knot is not finite");
                                if (k > 0 && !(knots[k - 1] < knots[k])) invalid("quantizer knots must be strictly increasing");
                            }
                            info.quantizer_knots.insert(info.quantizer_knots.end(), knots.begin(), knots.end());
                        }
                        info.quantizer_knot_offsets.push_back(static_cast<uint32_t>(info.quantizer_knots.size()));
                    }
                }
            }
        }

        Model model = Builder::build(temp, info, forest.source, forest.n_features);
        validate_model(model);
        if (binned) model.binned = build_binned(model);
        return model;
    }

} // namespace lignum
