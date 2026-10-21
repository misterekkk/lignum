#include "builder.hpp"
#include "parsers/parsers.hpp"

#include <algorithm>
#include <stdexcept>

namespace lignum {

    Model Builder::load_json(const std::string& filepath, const std::string& model_format) {
        auto data = std::make_shared<VectorData>();
        ModelInfo info;

        std::vector<std::unique_ptr<TempBinNode>> temp_forest;

        if (model_format == "lightgbm" || model_format == "lgbm" || model_format == "lgb") {
            temp_forest = parsers::load_lightgbm_json(filepath, info);
        }
        else if (model_format == "xgboost" || model_format == "xgb") {
            temp_forest = parsers::load_xgboost_json(filepath, info);
        }
        else {
            throw std::invalid_argument("This model format is not supported.");
        }

        BuildStats stats;
        for (const auto& root : temp_forest) {
            stats.tree_zero_missing = false;
            data->tree_offsets.push_back(build_knode(root.get(), *data, stats));
            data->tree_kinds.push_back(stats.tree_zero_missing ? tree_kind::ZeroMissing : tree_kind::Numeric);
        }

        data->base_scores = info.base_scores;
        data->tree_outputs = info.tree_outputs;
        if (data->base_scores.empty() || (data->base_scores.size() > 1 && data->tree_outputs.size() != data->tree_offsets.size())) {
            throw std::runtime_error("Inconsistent multi-output model.");
        }

        Model model;
        model.n_outputs = static_cast<uint32_t>(data->base_scores.size());
        model.base_scores = data->base_scores.data();
        model.tree_outputs = model.n_outputs > 1 ? data->tree_outputs.data() : nullptr;
        model.sigmoid = info.sigmoid;
        model.transform = info.transform;
        model.float32 = info.precision == Precision::Float32;
        model.has_zero_missing = stats.has_zero_missing;
        model.n_features = static_cast<uint32_t>(stats.max_feature + 1);
        model.n_nodes = data->k_nodes.size();
        model.k_nodes = data->k_nodes.data();
        model.n_offsets = data->tree_offsets.size();
        model.tree_offsets = data->tree_offsets.data();
        model.tree_kinds = data->tree_kinds.data();
        model.n_leaves = data->leaf_values.size();
        model.leaf_values = data->leaf_values.data();
        model.data = data;

        return model;
    }

    int32_t Builder::process_leaf(TempBinNode* node, VectorData& data) {
        data.leaf_values.push_back(node->threshold_or_value);
        return -static_cast<int32_t>(data.leaf_values.size());
    }

    void Builder::fill_slot(KNode& kn, int slot, const TempBinNode* split, BuildStats& stats) {
        if (split->feature_id < 0 || split->feature_id > 65535) {
            throw std::out_of_range("Features are stored as uint16_t.");
        }
        kn.thresholds[slot] = split->threshold_or_value;
        kn.features[slot] = static_cast<uint16_t>(split->feature_id);
        kn.dirs[slot] = static_cast<uint8_t>((split->default_dir ? dir_bits::DefaultRight : 0) |
                                             (split->zero_is_missing ? dir_bits::ZeroIsMissing : 0));
        stats.max_feature = std::max(stats.max_feature, split->feature_id);
        stats.has_zero_missing |= split->zero_is_missing;
        stats.tree_zero_missing |= split->zero_is_missing;
    }

    int32_t Builder::build_knode(TempBinNode* node, VectorData& data, BuildStats& stats) {
        if (node->is_leaf) return process_leaf(node, data);

        int32_t current_idx = data.k_nodes.size();
        data.k_nodes.push_back(KNode{});
        KNode kn = {};

        fill_slot(kn, 0, node, stats);

        TempBinNode* left = node->left.get();
        if (left->is_leaf) {
            int32_t leaf_idx = process_leaf(left, data);
            kn.children[0] = leaf_idx;
            kn.children[1] = leaf_idx;
        } else {
            fill_slot(kn, 1, left, stats);
            kn.children[0] = build_knode(left->left.get(), data, stats);
            kn.children[1] = build_knode(left->right.get(), data, stats);
        }

        TempBinNode* right = node->right.get();
        if (right->is_leaf) {
            int32_t leaf_idx = process_leaf(right, data);
            kn.children[2] = leaf_idx;
            kn.children[3] = leaf_idx;
        } else {
            fill_slot(kn, 2, right, stats);
            kn.children[2] = build_knode(right->left.get(), data, stats);
            kn.children[3] = build_knode(right->right.get(), data, stats);
        }

        data.k_nodes[current_idx] = kn;

        return current_idx;
    }

} // namespace lignum
