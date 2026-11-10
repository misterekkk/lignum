#include "parsers.hpp"
#include "simdjson.h"

#include <lignum/model.hpp>

#include <charconv>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

            enum class Missing { None, Zero, NaN };

            // A numeric split in the IR's terms (shared by the JSON and the text format).
            void numeric_split(TempBinNode& node, double raw_threshold, bool default_left, Missing missing) {
                const double threshold = lightgbm_threshold(raw_threshold);
                const int32_t natural_zero_dir = 0.0 >= threshold;
                const int32_t default_dir = default_left ? 0 : 1;
                node.threshold_or_value = threshold;
                if (missing == Missing::None) {
                    // NaN is replaced by 0.0 and compared normally.
                    node.default_dir = natural_zero_dir;
                } else if (missing == Missing::Zero) {
                    // NaN becomes 0.0, and 0.0 is missing. Only flag nodes where that changes the outcome.
                    node.default_dir = default_dir;
                    node.zero_is_missing = (default_dir != natural_zero_dir);
                } else {
                    node.default_dir = default_dir;
                }
            }

            // Adds category c to a LightGBM set if an input can match it: inputs are categories only
            // when -1 < x < 2^31 (int(x) truncates), so c is in [0, 2^31). Other members can never
            // be matched and are dropped, which changes no prediction (and keeps a crafted category
            // such as 4294967295 from allocating a bitset of 512 MB).
            void add_category(int64_t c, std::vector<uint32_t>& categories) {
                if (c >= 0 && c <= int64_t{2147483647}) categories.push_back(static_cast<uint32_t>(c));
            }

            // Categorical thresholds are strings such as "1||3||5".
            std::vector<uint32_t> parse_categories(simdjson::dom::element threshold) {
                std::vector<uint32_t> categories;
                if (threshold.is_string()) {
                    std::string_view text = threshold.get_string().value();
                    size_t start = 0;
                    while (start <= text.size()) {
                        size_t end = text.find("||", start);
                        if (end == std::string_view::npos) end = text.size();
                        if (end > start) {
                            const std::string_view token = text.substr(start, end - start);
                            int64_t c = 0;
                            const auto r = std::from_chars(token.data(), token.data() + token.size(), c);
                            if (r.ec != std::errc() || r.ptr != token.data() + token.size()) {
                                throw std::runtime_error("Malformed LightGBM model file: invalid category '" + std::string(token) + "'.");
                            }
                            add_category(c, categories);
                        }
                        start = end + 2;
                    }
                } else {
                    add_category(threshold.get_int64().value(), categories);
                }
                return categories;
            }

            std::unique_ptr<TempBinNode> parse_lgbm_node(simdjson::dom::object obj) {
                auto node = std::make_unique<TempBinNode>();

                if (obj["left_child"].error() != simdjson::SUCCESS) {
                    // Linear trees (linear_tree=true) add a linear model per leaf: not supported.
                    if (obj["leaf_const"].error() == simdjson::SUCCESS || obj["leaf_coeff"].error() == simdjson::SUCCESS) {
                        throw std::runtime_error("Malformed LightGBM model file: linear trees (linear_tree=true) are not supported.");
                    }
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
                    // LightGBM 4: NaN and values <= -1 go right, anything else is truncated to an
                    // integer category and goes left if it is in the set (verified empirically).
                    node->categorical = true;
                    node->default_dir = 1;
                    node->categories = parse_categories(obj["threshold"]);
                    node->left = parse_lgbm_node(obj["left_child"].get_object());
                    node->right = parse_lgbm_node(obj["right_child"].get_object());
                    return node;
                }

                std::string_view missing_type = "NaN";
                auto missing_elem = obj["missing_type"];
                if (missing_elem.error() == simdjson::SUCCESS) {
                    missing_type = missing_elem.get_string().value();
                }
                const Missing missing = missing_type == "None" ? Missing::None : missing_type == "Zero" ? Missing::Zero : Missing::NaN;
                numeric_split(*node, obj["threshold"].get_double(), obj["default_left"].get_bool(), missing);

                node->left = parse_lgbm_node(obj["left_child"].get_object());
                node->right = parse_lgbm_node(obj["right_child"].get_object());
                return node;
            }

            // The value of a "key:value" token of the objective string (e.g. "sigmoid:0.7").
            double objective_number(const std::vector<std::string_view>& tokens, std::string_view key, double fallback) {
                for (std::string_view t : tokens) {
                    if (t.size() > key.size() && t.substr(0, key.size()) == key && t[key.size()] == ':') {
                        return std::stod(std::string(t.substr(key.size() + 1)));
                    }
                }
                return fallback;
            }

            bool one_of(std::string_view name, std::initializer_list<std::string_view> names) {
                for (std::string_view n : names) if (name == n) return true;
                return false;
            }

            // Every LightGBM objective (as model files name it, and its aliases) with the output
            // transform of booster.predict. An unknown objective is rejected: its transform is not
            // known, so predictions could differ from LightGBM's. Without an objective (custom
            // objectives), LightGBM predicts raw scores; see load_json_document and the text parser.
            void parse_objective(std::string_view objective, ModelInfo& info) {
                std::vector<std::string_view> tokens;
                for (size_t start = 0; start < objective.size();) {
                    size_t end = objective.find(' ', start);
                    if (end == std::string_view::npos) end = objective.size();
                    if (end > start) tokens.push_back(objective.substr(start, end - start));
                    start = end + 1;
                }
                const std::string_view name = tokens.empty() ? std::string_view() : tokens[0];
                info.objective = std::string(name);
                // reg_sqrt=True: trained on sign(y) sqrt|y|, predict() squares back. LightGBM writes
                // "sqrt" only for objectives that apply it (huber and poisson ignore reg_sqrt).
                bool sqrt = false;
                for (size_t i = 1; i < tokens.size(); i++) sqrt |= tokens[i] == "sqrt";

                if (one_of(name, {"regression", "regression_l2", "l2", "mean_squared_error", "mse", "l2_root", "root_mean_squared_error",
                                  "rmse", "regression_l1", "l1", "mean_absolute_error", "mae", "fair", "quantile", "mape",
                                  "mean_absolute_percentage_error"})) {
                    info.transform = sqrt ? transform::SignedSquare : transform::Identity;
                    return;
                }
                if (sqrt) {
                    throw std::runtime_error("Unsupported LightGBM objective '" + std::string(objective) +
                                             "': LightGBM does not write 'sqrt' for this objective.");
                }
                if (one_of(name, {"huber", "lambdarank", "rank_xendcg", "xendcg", "xe_ndcg", "xe_ndcg_mart", "xendcg_mart", "custom",
                                  "none", "null", "na"})) {
                    info.transform = transform::Identity;
                } else if (one_of(name, {"binary", "multiclassova", "multiclass_ova", "ova", "ovr"})) {
                    info.transform = transform::Sigmoid;  // per class for one-vs-all
                    info.sigmoid = objective_number(tokens, "sigmoid", 1.0);
                } else if (one_of(name, {"multiclass", "softmax"})) {
                    info.transform = transform::Softmax;
                } else if (one_of(name, {"cross_entropy", "xentropy"})) {
                    info.transform = transform::Sigmoid;
                } else if (one_of(name, {"cross_entropy_lambda", "xentlambda"})) {
                    info.transform = transform::Log1pExp;
                } else if (one_of(name, {"poisson", "gamma", "tweedie"})) {
                    info.transform = transform::Exp;
                } else {
                    throw std::runtime_error("Unsupported LightGBM objective '" + std::string(name) +
                                             "': lignum does not know how LightGBM transforms its raw scores, so predictions "
                                             "could differ. Please report it at https://github.com/misterekkk/lignum/issues.");
                }
            }

        } // namespace

        void lightgbm_numeric_split(TempBinNode& node, double threshold, bool default_left, int missing_type) {
            numeric_split(node, threshold, default_left, missing_type == 0 ? Missing::None : missing_type == 1 ? Missing::Zero : Missing::NaN);
        }

        void lightgbm_objective(std::string_view objective, ModelInfo& info) {
            parse_objective(objective, info);
        }

        CategoryValue category_value(simdjson::dom::element v) {
            CategoryValue c;
            switch (v.type()) {
                case simdjson::dom::element_type::STRING:
                    c.kind = CategoryValue::Kind::Text;
                    c.text = std::string(v.get_string().value());
                    break;
                case simdjson::dom::element_type::INT64:
                    c.kind = CategoryValue::Kind::Integer;
                    c.integer = v.get_int64().value();
                    break;
                case simdjson::dom::element_type::UINT64:  // above 2^63: kept as a number
                    c.kind = CategoryValue::Kind::Number;
                    c.number = static_cast<double>(v.get_uint64().value());
                    break;
                case simdjson::dom::element_type::DOUBLE:
                    c.kind = CategoryValue::Kind::Number;
                    c.number = v.get_double().value();
                    break;
                case simdjson::dom::element_type::BOOL:
                    c.kind = CategoryValue::Kind::Boolean;
                    c.integer = v.get_bool().value() ? 1 : 0;
                    break;
                default:
                    throw std::runtime_error("Malformed model file: a category must be text, a number or a boolean.");
            }
            return c;
        }

        std::shared_ptr<TrainingCategories> lightgbm_categories(simdjson::dom::element value) {
            if (value.is_null()) return nullptr;
            simdjson::dom::array lists;
            if (value.get_array().get(lists) != simdjson::SUCCESS) throw std::runtime_error("Malformed LightGBM model file: pandas_categorical is not a list.");
            auto categories = std::make_shared<TrainingCategories>();
            categories->by_position = true;        // the k-th list: the k-th category column of the frame
            categories->unseen_is_missing = true;  // set_categories() makes unseen categories NaN
            for (simdjson::dom::element list : lists) {
                simdjson::dom::array values;
                if (list.get_array().get(values) != simdjson::SUCCESS) throw std::runtime_error("Malformed LightGBM model file: pandas_categorical entry is not a list.");
                std::vector<CategoryValue> out;
                for (simdjson::dom::element v : values) out.push_back(category_value(v));
                categories->lists.push_back(std::move(out));
            }
            return categories->lists.empty() ? nullptr : categories;
        }

        namespace {
            std::vector<std::unique_ptr<TempBinNode>> load_json_document(simdjson::dom::element doc, ModelInfo& info);
        }

        std::vector<std::unique_ptr<TempBinNode>> load_lightgbm_json(const std::string& filepath, ModelInfo& info) {
            simdjson::dom::parser parser;
            try {
                return load_json_document(parser.load(filepath), info);
            } catch (const simdjson::simdjson_error& e) {
                // A missing or mistyped field: name the library, not only simdjson's error code.
                throw std::runtime_error("Malformed LightGBM model file " + filepath + ": " + e.what());
            }
        }

        namespace {

        std::vector<std::unique_ptr<TempBinNode>> load_json_document(simdjson::dom::element doc, ModelInfo& info) {
            std::vector<std::unique_ptr<TempBinNode>> forest;

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

            bool average_output = false;
            if (doc["average_output"].get_bool().get(average_output) == simdjson::SUCCESS && average_output) {
                throw std::runtime_error("Malformed LightGBM model file: random forest models (boosting=rf) are not supported yet.");
            }

            auto obj_elem = doc["objective"];
            if (obj_elem.error() == simdjson::SUCCESS) {
                parse_objective(obj_elem.get_string().value(), info);
            }

            simdjson::dom::array names;
            if (doc["feature_names"].get_array().get(names) == simdjson::SUCCESS) {
                for (simdjson::dom::element name : names) info.feature_names.emplace_back(name.get_string().value());
            }
            simdjson::dom::element pandas_categorical;
            if (doc["pandas_categorical"].get(pandas_categorical) == simdjson::SUCCESS) info.categories = lightgbm_categories(pandas_categorical);

            simdjson::dom::array trees;
            if (doc["tree_info"].get_array().get(trees) != simdjson::SUCCESS) {
                throw std::runtime_error("Malformed LightGBM model file: no \"tree_info\" (is it the JSON of LightGBM's dump_model()?).");
            }
            for (simdjson::dom::element tree_obj : trees) {
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

        } // namespace

    } // namespace parsers

} // namespace lignum
