#include "builder.hpp"

#include <stdexcept>

namespace lignum {

    Model Builder::load(const std::string& filepath, const std::string& model_format) {
        Model m;

        std::vector<std::unique_ptr<TempBinNode>> temp_forest;

        if (model_format == "lightgbm" || model_format == "lgb") {
            temp_forest = parsers::load_lightgbm_json(filepath);
        }
        else if (model_format == "xgboost" || model_format == "xgb") {
            temp_forest = parsers::load_xgboost_json(filepath, m.base_score);
        }
        else {
            throw std::invalid_argument("This model format is not supported.");
        }

        for (const auto& root : temp_forest) {
            m.tree_offsets.push_back(build_knode(root.get(), m));
        }

        return m;
    }

    int Builder::process_leaf(TempBinNode* node, Model& m) {
            m.leaf_values.push_back(node->threshold_or_value);
            return -static_cast<int>(m.leaf_values.size());
        }

    int Builder::build_knode(TempBinNode* node, Model& m) {
        if (node->is_leaf) return process_leaf(node, m);

        int current_idx = m.k_nodes.size();
        m.k_nodes.push_back(KNode{});
        KNode kn = {};

        kn.thresholds[0] = node->threshold_or_value;
        kn.features[0] = static_cast<uint16_t>(node->feature_id);
        kn.dirs[0] = static_cast<uint8_t>(node->default_dir);

        TempBinNode* left = node->left.get();
        if (left->is_leaf) {
            int leaf_idx = process_leaf(left, m);
            kn.children[0] = leaf_idx;
            kn.children[1] = leaf_idx;
        } else {
            kn.thresholds[1] = left->threshold_or_value;
            kn.features[1] = static_cast<uint16_t>(left->feature_id);
            kn.dirs[1] = static_cast<uint8_t>(left->default_dir);
            kn.children[0] = build_knode(left->left.get(), m);
            kn.children[1] = build_knode(left->right.get(), m);
        }

        TempBinNode* right = node->right.get();
        if (right->is_leaf) {
            int leaf_idx = process_leaf(right, m);
            kn.children[2] = leaf_idx;
            kn.children[3] = leaf_idx;
        } else {
            kn.thresholds[2] = right->threshold_or_value;
            kn.features[2] = static_cast<uint16_t>(right->feature_id);
            kn.dirs[2] = static_cast<uint8_t>(right->default_dir);
            kn.children[2] = build_knode(right->left.get(), m);
            kn.children[3] = build_knode(right->right.get(), m);
        }

        m.k_nodes[current_idx] = kn;
        
        return current_idx;
    }

} // namespace lignum