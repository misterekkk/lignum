#include "parsers.hpp"
#include "simdjson.h"

#include <cmath>

namespace lignum {

    namespace parsers {

        namespace {

            std::unique_ptr<TempBinNode> parse_lgbm_node(simdjson::dom::object obj) {
                auto node = std::make_unique<TempBinNode>();
                auto left_child_result = obj["left_child"];
                
                if (left_child_result.error() == simdjson::SUCCESS) {
                    node->is_leaf = false;
                    node->feature_id = static_cast<int32_t>(obj["split_feature"].get_int64());
                    
                    bool default_left = obj["default_left"].get_bool();
                    node->default_dir = default_left ? 0 : 1;

                    std::string decision_type = "";
                    auto dec_type_elem = obj["decision_type"];
                    if (dec_type_elem.error() == simdjson::SUCCESS) {
                        decision_type = std::string(dec_type_elem.get_string().value());
                    }

                    if (decision_type == "==") {
                        throw std::runtime_error("Native string-based categorical splits ('==') are not supported yet.");
                    }

                    double thresh = obj["threshold"].get_double();
                    node->threshold_or_value = std::nextafter(thresh, INFINITY);

                    node->left = parse_lgbm_node(obj["left_child"].get_object());
                    node->right = parse_lgbm_node(obj["right_child"].get_object());
                } else {
                    node->is_leaf = true;
                    node->threshold_or_value = obj["leaf_value"].get_double();
                }
                return node;
            }

        } // namespace

        std::vector<std::unique_ptr<TempBinNode>> load_lightgbm_json(const std::string& filepath, double& out_base_score, uint8_t& transform) {
            simdjson::dom::parser parser;
            std::vector<std::unique_ptr<TempBinNode>> forest;
            simdjson::dom::element doc = parser.load(filepath);

            auto num_class_elem = doc["num_class"];
            if (num_class_elem.error() == simdjson::SUCCESS) {
                int64_t num_class = num_class_elem.get_int64();
                if (num_class > 1) {
                    throw std::runtime_error("Multiclass models are not supported yet.");
                }
            }

            out_base_score = 0.0;
            transform = 2;
            
            auto obj_elem = doc["objective"];
            if (obj_elem.error() == simdjson::SUCCESS) {
                std::string obj_str = std::string(obj_elem.get_string().value());
                
                if (obj_str.find("binary") != std::string::npos || obj_str.find("cross_entropy") != std::string::npos) {
                    transform = 0;
                } else if (obj_str.find("poisson") != std::string::npos || obj_str.find("gamma") != std::string::npos || obj_str.find("tweedie") != std::string::npos) {
                    transform = 1;
                } else {
                    transform = 2;
                }
            }

            for (simdjson::dom::element tree_obj : doc["tree_info"]) {
                simdjson::dom::object tree_structure = tree_obj["tree_structure"].get_object();
                forest.push_back(parse_lgbm_node(tree_structure));
            }
            
            return forest;
        }

    } // namespace parsers

} // namespace lignum