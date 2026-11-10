#include "parsers.hpp"
#include "simdjson.h"

#include <lignum/model.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
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
                std::vector<int64_t> split_type;                      // 1 = categorical
                std::unordered_map<int64_t, std::vector<uint32_t>> categories;  // node -> categories going right
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

                auto split_type = tree_obj["split_type"];
                if (split_type.error() == simdjson::SUCCESS) {
                    for (int64_t v : split_type.get_array()) arrays.split_type.push_back(v);
                }
                auto cat_nodes = tree_obj["categories_nodes"];
                if (cat_nodes.error() == simdjson::SUCCESS && cat_nodes.get_array().size() > 0) {
                    std::vector<int64_t> nodes, segments, sizes, values;
                    for (int64_t v : cat_nodes.get_array()) nodes.push_back(v);
                    for (int64_t v : tree_obj["categories_segments"].get_array()) segments.push_back(v);
                    for (int64_t v : tree_obj["categories_sizes"].get_array()) sizes.push_back(v);
                    for (int64_t v : tree_obj["categories"].get_array()) values.push_back(v);
                    if (segments.size() != nodes.size() || sizes.size() != nodes.size()) {
                        throw std::runtime_error("Malformed categorical split arrays.");
                    }
                    for (size_t k = 0; k < nodes.size(); k++) {
                        if (segments[k] < 0 || sizes[k] < 0 || static_cast<uint64_t>(segments[k]) > values.size() ||
                            static_cast<uint64_t>(sizes[k]) > values.size() - static_cast<uint64_t>(segments[k])) {
                            throw std::runtime_error("Malformed categorical split arrays.");
                        }
                        // An input is a category only when 0 <= float(x) < 2^24, so other members can
                        // never be matched: dropped, which changes no prediction (and keeps a value
                        // like 2^32 + 1 from wrapping to category 1 in a 32-bit cast).
                        std::vector<uint32_t>& set = arrays.categories[nodes[k]];
                        for (int64_t j = segments[k]; j < segments[k] + sizes[k]; j++) {
                            if (values[static_cast<size_t>(j)] >= 0 && values[static_cast<size_t>(j)] < (int64_t{1} << 24)) {
                                set.push_back(static_cast<uint32_t>(values[static_cast<size_t>(j)]));
                            }
                        }
                    }
                }
                return arrays;
            }

            // As deep as the LightGBM parsers allow: deeper recursion here and in the builder could
            // overflow the stack (1 MB on Windows) on a crafted file.
            constexpr int kMaxDepth = 1024;

            // Every node is reached once: children shared between parents would multiply the tree
            // (a file of a few dozen nodes can describe 2^28 paths).
            std::unique_ptr<TempBinNode> build_tree(const TreeArrays& t, size_t node_id, int depth, std::vector<char>& visited) {
                if (depth > kMaxDepth) {
                    throw std::runtime_error("Malformed XGBoost tree: deeper than " + std::to_string(kMaxDepth) + " levels.");
                }
                if (visited.at(node_id)) throw std::runtime_error("Malformed XGBoost tree: a node has two parents.");
                visited[node_id] = 1;
                auto node = std::make_unique<TempBinNode>();

                if (t.left.at(node_id) == -1) {
                    node->is_leaf = true;
                    // Leaves are float32 in XGBoost; the JSON value round-trips to the same float.
                    node->threshold_or_value = static_cast<float>(t.split_condition.at(node_id));
                    return node;
                }

                node->feature_id = static_cast<int32_t>(t.split_index.at(node_id));
                node->default_dir = t.default_left.at(node_id) ? 0 : 1;
                if (node_id < t.split_type.size() && t.split_type[node_id] == 1) {
                    // XGBoost: float(x) is the category; categories in the set go RIGHT, invalid
                    // ones (negative or too large) go LEFT, NaN follows default_left.
                    const auto found = t.categories.find(static_cast<int64_t>(node_id));
                    node->categorical = true;
                    node->cat_set_goes_right = true;
                    if (found != t.categories.end()) node->categories = found->second;
                } else {
                    node->threshold_or_value = xgboost_threshold(t.split_condition.at(node_id));
                }
                // XGBoost numbers children after their parent; anything else would be a cycle.
                const int64_t left = t.left[node_id], right = t.right.at(node_id);
                if (left <= static_cast<int64_t>(node_id) || right <= static_cast<int64_t>(node_id)) {
                    throw std::runtime_error("Malformed XGBoost tree: child index does not follow its parent.");
                }
                node->left = build_tree(t, static_cast<size_t>(left), depth + 1, visited);
                node->right = build_tree(t, static_cast<size_t>(right), depth + 1, visited);
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

            // DART tree weights (absent for gbtree, all 1 when DART dropped nothing).
            void reject_weighted_trees(simdjson::simdjson_result<simdjson::dom::element> weights) {
                if (weights.error() != simdjson::SUCCESS) return;
                for (simdjson::dom::element weight : weights.get_array()) {
                    if (static_cast<float>(weight.get_double().value()) != 1.0f) {
                        throw std::runtime_error("DART models with dropped trees (tree weights other than 1) are not supported yet.");
                    }
                }
            }

            std::vector<std::unique_ptr<TempBinNode>> load_document(simdjson::dom::element doc, ModelInfo& info);

        } // namespace

        std::vector<std::unique_ptr<TempBinNode>> load_xgboost_json(const std::string& filepath, ModelInfo& info) {
            simdjson::dom::parser parser;
            try {
                return load_document(parser.load(filepath), info);
            } catch (const simdjson::simdjson_error& e) {
                // A missing or mistyped field: name the library, not only simdjson's error code.
                throw std::runtime_error("Malformed XGBoost model file " + filepath + ": " + e.what());
            }
        }

        std::vector<std::unique_ptr<TempBinNode>> load_xgboost_ubj(const std::string& filepath, ModelInfo& info) {
            const std::string json = ubjson_to_json(read_file(filepath));
            simdjson::dom::parser parser;
            try {
                return load_document(parser.parse(json), info);
            } catch (const simdjson::simdjson_error& e) {
                throw std::runtime_error("Malformed XGBoost model file " + filepath + ": " + e.what());
            }
        }

        namespace {

        std::vector<std::unique_ptr<TempBinNode>> load_document(simdjson::dom::element doc, ModelInfo& info) {
            std::vector<std::unique_ptr<TempBinNode>> forest;

            info = ModelInfo{};
            info.precision = Precision::Float32;

            simdjson::dom::element params = doc["learner"]["learner_model_param"];
            if (read_int_param(params, "num_target", 1) > 1) {
                throw std::runtime_error("Multi-target models are not supported yet.");
            }
            const int n_outputs = std::max(1, read_int_param(params, "num_class", 0));

            std::vector<float> base_scores = read_base_scores(params);
            // One base_score for every class (XGBoost before 2.0 writes "[5E-1]" for multiclass models).
            // A copy: assign(n, base_scores[0]) would read the element after freeing the old storage.
            if (base_scores.size() == 1 && n_outputs > 1) {
                const float base_score = base_scores[0];
                base_scores.assign(static_cast<size_t>(n_outputs), base_score);
            }
            if (base_scores.size() != static_cast<size_t>(n_outputs)) {
                throw std::runtime_error("base_score has " + std::to_string(base_scores.size()) + " values for " +
                                         std::to_string(n_outputs) + " outputs.");
            }

            auto objective_elem = doc["learner"]["objective"]["name"];
            const std::string_view objective = objective_elem.error() == simdjson::SUCCESS ? objective_elem.get_string().value() : "";

            // Every XGBoost objective with predict()'s transform and the conversion of base_score to
            // margin space (ProbToMargin), both in float32. An unknown objective is rejected: its
            // transform is not known, so predictions could differ from XGBoost's.
            enum class Link { Identity, Logit, Log };
            Link link = Link::Identity;
            if (objective.empty()) throw std::runtime_error("Malformed XGBoost model: no objective (learner.objective.name).");
            if (objective == "reg:squarederror" || objective == "reg:linear" || objective == "reg:squaredlogerror" ||
                objective == "reg:pseudohubererror" || objective == "reg:absoluteerror" || objective == "reg:quantileerror" ||
                objective == "reg:expectileerror" ||
                objective == "binary:logitraw" || objective == "rank:ndcg" || objective == "rank:map" || objective == "rank:pairwise") {
                info.transform = transform::Identity;
            } else if (objective == "binary:logistic" || objective == "reg:logistic") {
                info.transform = transform::Sigmoid;
                link = Link::Logit;
            } else if (objective == "count:poisson" || objective == "reg:gamma" || objective == "reg:tweedie" ||
                       objective == "survival:cox" || objective == "survival:aft") {
                info.transform = transform::Exp;
                link = Link::Log;
            } else if (objective == "binary:hinge") {
                info.transform = transform::Hinge;  // base_score is a margin already
            } else if (objective == "multi:softprob") {
                info.transform = transform::Softmax;  // base scores are stored in margin space
            } else if (objective == "multi:softmax") {
                info.transform = transform::Argmax;
            } else {
                throw std::runtime_error("Unsupported XGBoost objective '" + std::string(objective) +
                                         "': lignum does not know how XGBoost transforms its raw scores, so predictions "
                                         "could differ. Please report it at https://github.com/misterekkk/lignum/issues.");
            }
            for (float& base_score : base_scores) {
                if (link == Link::Logit) {
                    if (!(base_score > 0.0f && base_score < 1.0f)) {
                        throw std::runtime_error("base_score must be in (0, 1) for logistic objectives.");
                    }
                    base_score = -std::log(1.0f / base_score - 1.0f);
                } else if (link == Link::Log) {
                    if (!(base_score > 0.0f)) {
                        throw std::runtime_error("base_score must be positive for log-link objectives.");
                    }
                    base_score = std::log(base_score);
                }
            }
            info.base_scores.assign(base_scores.begin(), base_scores.end());
            info.objective = std::string(objective);

            simdjson::dom::array names;  // empty for models trained on a matrix
            if (doc["learner"]["feature_names"].get_array().get(names) == simdjson::SUCCESS) {
                for (simdjson::dom::element name : names) info.feature_names.emplace_back(name.get_string().value());
            }

            // gbtree keeps the trees in "model". DART from XGBoost < 3.4 nests a gbtree and keeps one
            // weight per tree beside it; 3.4 saves DART as gbtree with "weight_drop" in "model".
            // Weighted trees are not supported yet: reject them rather than predict wrongly.
            simdjson::dom::element booster = doc["learner"]["gradient_booster"];
            auto name_elem = booster["name"];
            const std::string_view booster_name = name_elem.error() == simdjson::SUCCESS ? name_elem.get_string().value() : "";
            simdjson::dom::element model;
            if (booster_name == "gbtree") {
                if (booster["model"].get(model) != simdjson::SUCCESS) throw std::runtime_error("Malformed XGBoost model: no gbtree model.");
            } else if (booster_name == "dart") {
                if (booster["gbtree"]["model"].get(model) != simdjson::SUCCESS) {
                    throw std::runtime_error("Malformed XGBoost model: DART without its gbtree model.");
                }
                reject_weighted_trees(booster["weight_drop"]);
            } else if (booster_name == "gblinear") {
                throw std::runtime_error("Linear boosters (booster=gblinear) are not supported.");
            } else {
                throw std::runtime_error("Unknown XGBoost booster '" + std::string(booster_name) + "'.");
            }
            reject_weighted_trees(model["weight_drop"]);

            simdjson::dom::array trees;
            if (model["trees"].get_array().get(trees) != simdjson::SUCCESS) {
                throw std::runtime_error("Malformed XGBoost model: no trees in gradient_booster.");
            }
            for (simdjson::dom::element tree_elem : trees) {
                const TreeArrays arrays = read_tree_arrays(tree_elem.get_object());
                std::vector<char> visited(arrays.left.size(), 0);
                forest.push_back(build_tree(arrays, 0, 0, visited));
            }

            if (n_outputs > 1) {
                for (int64_t output : model["tree_info"].get_array()) {
                    if (output < 0 || output >= n_outputs) throw std::runtime_error("tree_info refers to an unknown class.");
                    info.tree_outputs.push_back(static_cast<uint16_t>(output));
                }
                if (info.tree_outputs.size() != forest.size()) throw std::runtime_error("tree_info does not match the number of trees.");
            }

            return forest;
        }

        } // namespace

    } // namespace parsers

} // namespace lignum
