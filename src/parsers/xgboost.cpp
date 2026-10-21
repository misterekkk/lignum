#include "parsers.hpp"
#include "simdjson.h"

#include <lignum/model.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lignum {

    namespace parsers {

        namespace {

            // XGBoost goes left iff float(x) < t (t is a float32). Returns the smallest double
            // `thr` such that float(x) >= t  <=>  x >= thr, so the kernel can compare doubles.
            double xgboost_threshold(double condition) {
                const float t = static_cast<float>(condition);
                double below = static_cast<double>(std::nextafter(t, -INFINITY));
                if (std::isinf(below)) {
                    below = static_cast<double>(t) - (static_cast<double>(std::nextafter(t, INFINITY)) - static_cast<double>(t));
                }
                // Exact in double: floats have 24-bit significands, the midpoint needs 25.
                const double mid = (below + static_cast<double>(t)) / 2.0;
                return static_cast<float>(mid) == t ? mid : std::nextafter(mid, INFINITY);
            }

            struct TreeArrays {
                std::vector<int64_t> left, right, split_index;
                std::vector<double> split_condition;
                std::vector<bool> default_left;
            };

            TreeArrays read_tree_arrays(simdjson::dom::object tree_obj) {
                TreeArrays arrays;
                for (int64_t v : tree_obj["left_children"].get_array()) arrays.left.push_back(v);
                for (int64_t v : tree_obj["right_children"].get_array()) arrays.right.push_back(v);
                for (int64_t v : tree_obj["split_indices"].get_array()) arrays.split_index.push_back(v);
                for (simdjson::dom::element v : tree_obj["split_conditions"].get_array()) arrays.split_condition.push_back(v.get_double());
                for (simdjson::dom::element v : tree_obj["default_left"].get_array()) {
                    arrays.default_left.push_back(v.is_bool() ? v.get_bool().value() : v.get_int64().value() != 0);
                }
                return arrays;
            }

            std::unique_ptr<TempBinNode> build_tree(const TreeArrays& t, size_t node_id) {
                auto node = std::make_unique<TempBinNode>();

                if (t.left.at(node_id) == -1) {
                    node->is_leaf = true;
                    // Leaves are float32 in XGBoost; the JSON value round-trips to the same float.
                    node->threshold_or_value = static_cast<float>(t.split_condition.at(node_id));
                    return node;
                }

                node->feature_id = static_cast<int32_t>(t.split_index.at(node_id));
                node->threshold_or_value = xgboost_threshold(t.split_condition.at(node_id));
                node->default_dir = t.default_left.at(node_id) ? 0 : 1;
                node->left = build_tree(t, static_cast<size_t>(t.left[node_id]));
                node->right = build_tree(t, static_cast<size_t>(t.right.at(node_id)));
                return node;
            }

            int read_int_param(simdjson::dom::element params, const char* key, int fallback) {
                auto elem = params[key];
                if (elem.error() != simdjson::SUCCESS) return fallback;
                return std::stoi(std::string(elem.get_string().value()));
            }

            // base_score is "[v]" or, for multi-output models, "[v0,v1,...]".
            std::vector<float> read_base_scores(simdjson::dom::element params) {
                auto elem = params["base_score"];
                if (elem.error() != simdjson::SUCCESS) return {0.5f};
                std::string text(elem.get_string().value());
                std::replace(text.begin(), text.end(), '[', ' ');
                std::replace(text.begin(), text.end(), ']', ' ');
                std::replace(text.begin(), text.end(), ',', ' ');
                std::vector<float> values;
                size_t pos = 0;
                while (pos < text.size()) {
                    size_t used = 0;
                    const auto start = text.find_first_not_of(' ', pos);
                    if (start == std::string::npos) break;
                    values.push_back(std::stof(text.substr(start), &used));
                    pos = start + used;
                }
                if (values.empty()) throw std::runtime_error("Invalid base_score.");
                return values;
            }

        } // namespace

        std::vector<std::unique_ptr<TempBinNode>> load_xgboost_json(const std::string& filepath, ModelInfo& info) {
            simdjson::dom::parser parser;
            std::vector<std::unique_ptr<TempBinNode>> forest;
            simdjson::dom::element doc = parser.load(filepath);

            info = ModelInfo{};
            info.precision = Precision::Float32;

            simdjson::dom::element params = doc["learner"]["learner_model_param"];
            if (read_int_param(params, "num_target", 1) > 1) {
                throw std::runtime_error("Multi-target models are not supported yet.");
            }
            const int n_outputs = std::max(1, read_int_param(params, "num_class", 0));

            std::vector<float> base_scores = read_base_scores(params);
            if (base_scores.size() == 1 && n_outputs > 1) base_scores.assign(static_cast<size_t>(n_outputs), base_scores[0]);
            if (base_scores.size() != static_cast<size_t>(n_outputs)) {
                throw std::runtime_error("base_score has " + std::to_string(base_scores.size()) + " values for " +
                                         std::to_string(n_outputs) + " outputs.");
            }

            auto objective_elem = doc["learner"]["objective"]["name"];
            const std::string_view objective = objective_elem.error() == simdjson::SUCCESS ? objective_elem.get_string().value() : "";

            // XGBoost converts base_score to margin space in float32 arithmetic.
            // binary:logitraw keeps base_score as is (no logit) and applies no transform;
            // multi:softprob / multi:softmax store it in margin space already.
            info.transform = transform::Identity;
            for (float& base_score : base_scores) {
                if (objective == "binary:logistic" || objective == "reg:logistic") {
                    if (!(base_score > 0.0f && base_score < 1.0f)) {
                        throw std::runtime_error("base_score must be in (0, 1) for logistic objectives.");
                    }
                    base_score = -std::log(1.0f / base_score - 1.0f);
                    info.transform = transform::Sigmoid;
                } else if (objective == "count:poisson" || objective == "reg:gamma" || objective == "reg:tweedie" ||
                           objective == "survival:cox" || objective == "survival:aft") {
                    if (!(base_score > 0.0f)) {
                        throw std::runtime_error("base_score must be positive for log-link objectives.");
                    }
                    base_score = std::log(base_score);
                    info.transform = transform::Exp;
                }
            }
            if (objective == "multi:softprob") info.transform = transform::Softmax;
            if (objective == "multi:softmax") info.transform = transform::Argmax;
            info.base_scores.assign(base_scores.begin(), base_scores.end());

            auto trees_element = doc["learner"]["gradient_booster"]["model"]["trees"];
            if (trees_element.error() == simdjson::SUCCESS) {
                for (simdjson::dom::element tree_elem : trees_element.get_array()) {
                    simdjson::dom::object tree_obj = tree_elem.get_object();

                    auto cat_nodes_elem = tree_obj["categories_nodes"];
                    if (cat_nodes_elem.error() == simdjson::SUCCESS && cat_nodes_elem.get_array().size() > 0) {
                        throw std::runtime_error("Models with native categorical splits are not supported by this engine.");
                    }

                    forest.push_back(build_tree(read_tree_arrays(tree_obj), 0));
                }
            }

            if (n_outputs > 1) {
                for (int64_t output : doc["learner"]["gradient_booster"]["model"]["tree_info"].get_array()) {
                    if (output < 0 || output >= n_outputs) throw std::runtime_error("tree_info refers to an unknown class.");
                    info.tree_outputs.push_back(static_cast<uint16_t>(output));
                }
                if (info.tree_outputs.size() != forest.size()) throw std::runtime_error("tree_info does not match the number of trees.");
            }

            return forest;
        }

    } // namespace parsers

} // namespace lignum
