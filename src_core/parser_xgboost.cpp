#include "parsers.hpp"
#include "simdjson.h"

#include <cmath>

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
                    node->feature_id = static_cast<int32_t>(tree_obj["split_indices"].get_array().at(node_id).get_int64());

                    bool default_left = (tree_obj["default_left"].get_array().at(node_id).get_int64() == 1);
                    node->default_dir = default_left ? 0 : 1;

                    node->left = parse_xgboost_tree(left_idx, tree_obj);
                    node->right = parse_xgboost_tree(right_idx, tree_obj);
                }

                return node;
            }
        
        } // namespace

        std::vector<std::unique_ptr<TempBinNode>> load_xgboost_json(const std::string& filepath, double& out_base_score, uint8_t& transform) {
            simdjson::dom::parser parser;
            std::vector<std::unique_ptr<TempBinNode>> forest;

            simdjson::dom::element doc = parser.load(filepath);

            auto param_element = doc["learner"]["learner_model_param"];
            if (param_element.error() == simdjson::SUCCESS) {
                int num_class = 0;
                int num_target = 1;

                auto num_class_elem = param_element["num_class"];
                if (num_class_elem.error() == simdjson::SUCCESS) {
                    num_class = std::stoi(std::string(num_class_elem.get_string().value()));
                }

                auto num_target_elem = param_element["num_target"];
                if (num_target_elem.error() == simdjson::SUCCESS) {
                    num_target = std::stoi(std::string(num_target_elem.get_string().value()));
                }

                if (num_class > 1 || num_target > 1) {
                    throw std::runtime_error("Multiclass and multi-target models are not supported yet.");
                }
            }

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
                if (loss_func_string == "binary:logistic" || loss_func_string == "reg:logistic") {
                    double eps = 1e-15;
                    base_score = std::min(std::max(base_score, eps), 1 - eps);
                    out_base_score = std::log(base_score / (1 - base_score));
                    transform = 0;
                } else if (loss_func_string == "count:poisson" || loss_func_string == "reg:gamma" || loss_func_string == "reg:tweedie" || loss_func_string == "survival:cox" || loss_func_string == "survival:aft") {
                    out_base_score = (base_score > 0.0) ? std::log(base_score) : 0.0;
                    transform = 1;
                } else {
                    out_base_score = base_score;
                    transform = 2;
                }
            } else {
                out_base_score = base_score;
                transform = 2;
            }

            auto trees_element = doc["learner"]["gradient_booster"]["model"]["trees"];
            if (trees_element.error() == simdjson::SUCCESS) {
                simdjson::dom::array trees = trees_element.get_array();
                for (simdjson::dom::element tree_elem : trees) {
                    simdjson::dom::object tree_obj = tree_elem.get_object();
                    
                    auto cat_nodes_elem = tree_obj["categories_nodes"];
                    if (cat_nodes_elem.error() == simdjson::SUCCESS && cat_nodes_elem.get_array().size() > 0) {
                        throw std::runtime_error("Models with native categorical splits are not supported by this engine.");
                    }

                    forest.push_back(parse_xgboost_tree(0, tree_obj));
                }
            }
            
            return forest;
        }

    } // namespace parsers

} // namespace lignum