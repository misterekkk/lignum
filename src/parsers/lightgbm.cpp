#include "parsers.hpp"
#include "simdjson.h"

#include <lignum/model.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lignum {

    namespace parsers {

        namespace {

            // LightGBM drops |x| <= kZeroThreshold from dense input, so such values behave as 0.0.
            constexpr double kZeroThreshold = static_cast<double>(1e-35f);

            // LightGBM goes right iff z(x) > t, where z zeroes the band [-kZero, kZero].
            // Returns the smallest double `thr` such that (x >= thr) gives the same answer.
            double lightgbm_threshold(double t) {
                if (t >= kZeroThreshold || t < -kZeroThreshold) return std::nextafter(t, INFINITY);
                if (t >= 0.0) return std::nextafter(kZeroThreshold, INFINITY);  // band -> 0 <= t: left
                return -kZeroThreshold;                                         // band -> 0 > t: right
            }

            std::unique_ptr<TempBinNode> parse_lgbm_node(simdjson::dom::object obj) {
                auto node = std::make_unique<TempBinNode>();

                if (obj["left_child"].error() != simdjson::SUCCESS) {
                    node->is_leaf = true;
                    node->threshold_or_value = obj["leaf_value"].get_double();
                    return node;
                }

                node->feature_id = static_cast<int32_t>(obj["split_feature"].get_int64());

                std::string_view decision_type = "<=";
                auto dec_type_elem = obj["decision_type"];
                if (dec_type_elem.error() == simdjson::SUCCESS) {
                    decision_type = dec_type_elem.get_string().value();
                }
                if (decision_type == "==") {
                    throw std::runtime_error("Native string-based categorical splits ('==') are not supported yet.");
                }

                std::string_view missing_type = "NaN";
                auto missing_elem = obj["missing_type"];
                if (missing_elem.error() == simdjson::SUCCESS) {
                    missing_type = missing_elem.get_string().value();
                }

                const double threshold = lightgbm_threshold(obj["threshold"].get_double());
                const int32_t natural_zero_dir = 0.0 >= threshold;
                const int32_t default_dir = obj["default_left"].get_bool() ? 0 : 1;

                node->threshold_or_value = threshold;
                if (missing_type == "None") {
                    // NaN is replaced by 0.0 and compared normally.
                    node->default_dir = natural_zero_dir;
                } else if (missing_type == "Zero") {
                    // NaN becomes 0.0, and 0.0 is missing. Only flag nodes where that changes the outcome.
                    node->default_dir = default_dir;
                    node->zero_is_missing = (default_dir != natural_zero_dir);
                } else {
                    node->default_dir = default_dir;
                }

                node->left = parse_lgbm_node(obj["left_child"].get_object());
                node->right = parse_lgbm_node(obj["right_child"].get_object());
                return node;
            }

            double parse_sigmoid(std::string_view objective) {
                const auto pos = objective.find("sigmoid:");
                if (pos == std::string_view::npos) return 1.0;
                return std::stod(std::string(objective.substr(pos + 8)));
            }

            void parse_objective(std::string_view objective, ModelInfo& info) {
                const std::string_view name = objective.substr(0, objective.find(' '));

                if (name == "binary" || name == "multiclassova" || name == "ovr" || name == "multiclass_ova" || name == "ova") {
                    info.transform = transform::Sigmoid;  // per class for one-vs-all
                    info.sigmoid = parse_sigmoid(objective);
                } else if (name == "multiclass" || name == "softmax") {
                    info.transform = transform::Softmax;
                } else if (name == "cross_entropy" || name == "xentropy") {
                    info.transform = transform::Sigmoid;
                } else if (name == "cross_entropy_lambda" || name == "xentlambda") {
                    info.transform = transform::Log1pExp;
                } else if (name == "poisson" || name == "gamma" || name == "tweedie") {
                    info.transform = transform::Exp;
                } else {
                    info.transform = transform::Identity;
                }
            }

        } // namespace

        std::vector<std::unique_ptr<TempBinNode>> load_lightgbm_json(const std::string& filepath, ModelInfo& info) {
            simdjson::dom::parser parser;
            std::vector<std::unique_ptr<TempBinNode>> forest;
            simdjson::dom::element doc = parser.load(filepath);

            info = ModelInfo{};
            info.precision = Precision::Float64;

            int64_t n_outputs = 1;
            auto per_iteration = doc["num_tree_per_iteration"];
            if (per_iteration.error() == simdjson::SUCCESS) {
                n_outputs = per_iteration.get_int64();
            } else if (doc["num_class"].error() == simdjson::SUCCESS) {
                n_outputs = doc["num_class"].get_int64();
            }
            if (n_outputs < 1 || n_outputs > 65535) {
                throw std::runtime_error("Unsupported number of trees per iteration: " + std::to_string(n_outputs) + ".");
            }

            auto obj_elem = doc["objective"];
            if (obj_elem.error() == simdjson::SUCCESS) {
                parse_objective(obj_elem.get_string().value(), info);
            }

            for (simdjson::dom::element tree_obj : doc["tree_info"]) {
                simdjson::dom::object tree_structure = tree_obj["tree_structure"].get_object();
                forest.push_back(parse_lgbm_node(tree_structure));
            }

            // Trees are stored iteration by iteration, one per class: tree t belongs to class t % K.
            info.base_scores.assign(static_cast<size_t>(n_outputs), 0.0);
            if (n_outputs > 1) {
                for (size_t t = 0; t < forest.size(); t++) {
                    info.tree_outputs.push_back(static_cast<uint16_t>(t % static_cast<size_t>(n_outputs)));
                }
            }

            return forest;
        }

    } // namespace parsers

} // namespace lignum
