#include "parsers.hpp"
#include "simdjson.h"

namespace lignum {

    namespace parsers {

        namespace {

            std::unique_ptr<TempBinNode> parse_lgbm_node(simdjson::dom::object obj) {
                auto node = std::make_unique<TempBinNode>();
                auto left_child_result = obj["left_child"];
                
                if (left_child_result.error() == simdjson::SUCCESS) {
                    node->is_leaf = false;
                    node->threshold_or_value = obj["threshold"].get_double();
                    node->feature_id = static_cast<int32_t>(obj["split_feature"].get_int64());
                    bool default_left = obj["default_left"].get_bool();
                    node->default_dir = default_left ? 0 : 1;

                    node->left = parse_lgbm_node(obj["left_child"].get_object());
                    node->right = parse_lgbm_node(obj["right_child"].get_object());
                } else {
                    node->is_leaf = true;
                    node->threshold_or_value = obj["leaf_value"].get_double();
                }
                return node;
            }

        } // namespace

        std::vector<std::unique_ptr<TempBinNode>> load_lightgbm_json(const std::string& filepath) {
            simdjson::dom::parser parser;
            std::vector<std::unique_ptr<TempBinNode>> forest;
            simdjson::dom::element doc = parser.load(filepath);

            for (simdjson::dom::element tree_obj : doc["tree_info"]) {
                simdjson::dom::object tree_structure = tree_obj["tree_structure"].get_object();
                forest.push_back(parse_lgbm_node(tree_structure));
            }
            return forest;
        }

    } // namespace parsers

} // namespace lignum