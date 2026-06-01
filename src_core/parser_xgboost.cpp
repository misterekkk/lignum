#include "parsers.hpp"
#include "simdjson.h"

namespace lignum {

    namespace parsers {

        namespace {

            std::unique_ptr<TempBinNode> parse_xgboost_tree(size_t node_id, simdjson::dom::object tree_obj) {
                auto node = std::make_unique<TempBinNode>();
                
                simdjson::dom::array left_children = tree_obj["left_children"].get_array();
                simdjson::dom::array right_children = tree_obj["right_children"].get_array();

                int left_idx = left_children.at(node_id).get_int64();
                int right_idx = right_children.at(node_id).get_int64();

                if (left_idx == -1) {
                    node->is_leaf = true;
                    node->threshold_or_value = tree_obj["split_conditions"].get_array().at(node_id).get_double();
                } else {
                    node->is_leaf = false;
                    node->threshold_or_value = tree_obj["split_conditions"].get_array().at(node_id).get_double();
                    node->feature_id = static_cast<int>(tree_obj["split_indices"].get_array().at(node_id).get_int64());

                    bool default_left = (tree_obj["default_left"].get_array().at(node_id).get_int64() == 1);
                    node->default_dir = default_left ? 0 : 1;

                    node->left = parse_xgboost_tree(left_idx, tree_obj);
                    node->right = parse_xgboost_tree(right_idx, tree_obj);
                }

                return node;
            }
        
        } // namespace

        std::vector<std::unique_ptr<TempBinNode>> load_xgboost_json(const std::string& filepath, double& out_base_score) {
            simdjson::dom::parser parser;
            std::vector<std::unique_ptr<TempBinNode>> forest;

            simdjson::dom::element doc = parser.load(filepath);

            double base_score = 0.0;
            auto base_score_element = doc["learner"]["learner_model_param"]["base_score"];
            if (base_score_element.error() == simdjson::SUCCESS) {
                auto base_score_string = std::string(base_score_element.get_string().value());
                base_score_string.erase(std::remove(base_score_string.begin(), base_score_string.end(), ']'), base_score_string.end());
                base_score_string.erase(std::remove(base_score_string.begin(), base_score_string.end(), '['), base_score_string.end());

                base_score = std::stod(base_score_string);
            }

            auto loss_func_element = doc["learner"]["objective"]["name"];
            if (loss_func_element.error() == simdjson::SUCCESS) {
                auto loss_func_string = std::string(loss_func_element.get_string().value());
                if (loss_func_string == "binary:logistic") {
                    double eps = 1e-15;
                    base_score = std::min(std::max(base_score, eps), 1 - eps);
                    out_base_score = std::log(base_score / (1 - base_score));
                } else if (loss_func_string == "reg:squarederror" || loss_func_string == "reg:pseudohubererror" || loss_func_string == "reg:linear") {
                    out_base_score = base_score;
                }
            }

            simdjson::dom::array trees = doc["learner"]["gradient_booster"]["model"]["trees"].get_array();

            for (simdjson::dom::element tree_elem : trees) {
                simdjson::dom::object tree_obj = tree_elem.get_object();

                forest.push_back(parse_xgboost_tree(0, tree_obj));
            }

            return forest;
        }

    } // namespace parsers

} // namespace lignum