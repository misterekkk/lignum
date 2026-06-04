#include "builder.hpp"
#include "parsers.hpp"

#include <stdexcept>

namespace lignum {

    Model Builder::load_json(const std::string& filepath, const std::string& model_format) {
        auto data = std::make_shared<VectorData>();
        double base_score = 0.0;

        std::vector<std::unique_ptr<TempBinNode>> temp_forest;

        if (model_format == "lightgbm" || model_format == "lgbm" || model_format == "lgb") {
            temp_forest = parsers::load_lightgbm_json(filepath);
        }
        else if (model_format == "xgboost" || model_format == "xgb") {
            temp_forest = parsers::load_xgboost_json(filepath, base_score);
        }
        else {
            throw std::invalid_argument("This model format is not supported.");
        }

        for (const auto& root : temp_forest) {
            data->tree_offsets.push_back(build_knode(root.get(), *data));
        }

        Model model;
        model.base_score = base_score;
        model.n_nodes = data->k_nodes.size();
        model.k_nodes = data->k_nodes.data();
        model.n_offsets = data->tree_offsets.size();
        model.tree_offsets = data->tree_offsets.data();
        model.n_leaves = data->leaf_values.size();
        model.leaf_values = data->leaf_values.data();
        model.data = data;

        return model;
    }

    int32_t Builder::process_leaf(TempBinNode* node, VectorData& data) {
            data.leaf_values.push_back(node->threshold_or_value);
            return -static_cast<int32_t>(data.leaf_values.size());
        }

    int32_t Builder::build_knode(TempBinNode* node, VectorData& data) {
        if (node->is_leaf) return process_leaf(node, data);

        int32_t current_idx = data.k_nodes.size();
        data.k_nodes.push_back(KNode{});
        KNode kn = {};

        kn.thresholds[0] = node->threshold_or_value;
        if (node->feature_id < 0 || node->feature_id > 65535) {
            throw std::out_of_range("Features are stored as uint16_t.");
        }
        kn.features[0] = static_cast<uint16_t>(node->feature_id);
        kn.dirs[0] = static_cast<uint8_t>(node->default_dir);

        TempBinNode* left = node->left.get();
        if (left->is_leaf) {
            int32_t leaf_idx = process_leaf(left, data);
            kn.children[0] = leaf_idx;
            kn.children[1] = leaf_idx;
        } else {
            kn.thresholds[1] = left->threshold_or_value;
            kn.features[1] = static_cast<uint16_t>(left->feature_id);
            kn.dirs[1] = static_cast<uint8_t>(left->default_dir);
            kn.children[0] = build_knode(left->left.get(), data);
            kn.children[1] = build_knode(left->right.get(), data);
        }

        TempBinNode* right = node->right.get();
        if (right->is_leaf) {
            int32_t leaf_idx = process_leaf(right, data);
            kn.children[2] = leaf_idx;
            kn.children[3] = leaf_idx;
        } else {
            kn.thresholds[2] = right->threshold_or_value;
            kn.features[2] = static_cast<uint16_t>(right->feature_id);
            kn.dirs[2] = static_cast<uint8_t>(right->default_dir);
            kn.children[2] = build_knode(right->left.get(), data);
            kn.children[3] = build_knode(right->right.get(), data);
        }

        data.k_nodes[current_idx] = kn;
        
        return current_idx;
    }

} // namespace lignum