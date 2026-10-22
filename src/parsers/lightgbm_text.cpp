#include "parsers.hpp"
#include "simdjson.h"

#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lignum {

    namespace parsers {

        std::string read_file(const std::string& filepath) {
            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (!file) throw std::runtime_error("Cannot open " + filepath + ".");
            const std::streamoff size = file.tellg();
            if (size < 0) throw std::runtime_error("Cannot read " + filepath + ".");
            std::string content(static_cast<size_t>(size), '\0');
            file.seekg(0);
            if (!file.read(content.data(), size)) throw std::runtime_error("Cannot read " + filepath + ".");
            return content;
        }

        namespace {

            [[noreturn]] void fail(const std::string& what) {
                throw std::runtime_error("Malformed LightGBM model file: " + what + ".");
            }

            // Exact decimal -> double (correctly rounded on every platform, independent of the
            // locale), plus the inf / nan spellings LightGBM may write.
            double parse_double(std::string_view token) {
                if (token.empty()) fail("empty number");
                std::string_view body = token;
                const bool negative = body[0] == '-';
                if (body[0] == '-' || body[0] == '+') body.remove_prefix(1);
                if (body == "inf" || body == "Infinity" || body == "infinity") {
                    return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
                }
                if (body == "nan" || body == "NaN") return std::numeric_limits<double>::quiet_NaN();
                // Strict syntax first: from_chars below assumes a well-formed number.
                size_t i = 0, digits = 0;
                while (i < body.size() && body[i] >= '0' && body[i] <= '9') { i++; digits++; }
                if (i < body.size() && body[i] == '.') {
                    i++;
                    while (i < body.size() && body[i] >= '0' && body[i] <= '9') { i++; digits++; }
                }
                if (digits == 0) fail("invalid number '" + std::string(token) + "'");
                if (i < body.size() && (body[i] == 'e' || body[i] == 'E')) {
                    i++;
                    if (i < body.size() && (body[i] == '+' || body[i] == '-')) i++;
                    const size_t start = i;
                    while (i < body.size() && body[i] >= '0' && body[i] <= '9') i++;
                    if (i == start) fail("invalid number '" + std::string(token) + "'");
                }
                if (i != body.size()) fail("invalid number '" + std::string(token) + "'");
                const double v = simdjson::internal::from_chars(body.data(), body.data() + body.size());
                return negative ? -v : v;
            }

            template <typename T>
            T parse_integer(std::string_view token) {
                T v{};
                const auto r = std::from_chars(token.data(), token.data() + token.size(), v);
                if (r.ec != std::errc() || r.ptr != token.data() + token.size()) fail("invalid integer '" + std::string(token) + "'");
                return v;
            }

            // Lists of decimals go through simdjson as a JSON array: the same correctly rounded
            // result as parse_double, with simdjson's fast path (most numbers need no big-decimal
            // arithmetic). Lists simdjson rejects (inf, nan) fall back to parse_double per token.
            class Doubles {
                public:
                    std::vector<double> parse(std::string_view text) {
                        json_.assign("[");
                        size_t n = 0, start = 0;
                        while (start < text.size()) {
                            while (start < text.size() && text[start] == ' ') start++;
                            if (start >= text.size()) break;
                            size_t end = text.find(' ', start);
                            if (end == std::string_view::npos) end = text.size();
                            if (n++) json_ += ',';
                            json_.append(text.substr(start, end - start));
                            start = end;
                        }
                        json_ += ']';
                        std::vector<double> values;
                        values.reserve(n);
                        simdjson::dom::array array;
                        if (parser_.parse(json_).get_array().get(array) == simdjson::SUCCESS) {
                            bool ok = true;
                            for (simdjson::dom::element v : array) {
                                double d;
                                if (v.get_double().get(d) != simdjson::SUCCESS) { ok = false; break; }
                                values.push_back(d);
                            }
                            if (ok && values.size() == n) return values;
                        }
                        return parse_list_slow(text);
                    }

                private:
                    simdjson::dom::parser parser_;
                    std::string json_;

                    static std::vector<double> parse_list_slow(std::string_view text);
            };

            template <typename T, typename Parse>
            std::vector<T> parse_list(std::string_view text, Parse parse) {
                std::vector<T> values;
                size_t start = 0;
                while (start < text.size()) {
                    while (start < text.size() && text[start] == ' ') start++;
                    if (start >= text.size()) break;
                    size_t end = text.find(' ', start);
                    if (end == std::string_view::npos) end = text.size();
                    values.push_back(parse(text.substr(start, end - start)));
                    start = end;
                }
                return values;
            }

            std::vector<double> Doubles::parse_list_slow(std::string_view text) {
                return parse_list<double>(text, parse_double);
            }

            // key=value lines of one section (header or tree).
            using Section = std::unordered_map<std::string_view, std::string_view>;

            std::string_view field(const Section& section, std::string_view key) {
                const auto it = section.find(key);
                if (it == section.end()) fail("missing '" + std::string(key) + "'");
                return it->second;
            }

            // decision_type bits (LightGBM tree.h): 1 categorical, 2 default left, (>> 2) & 3 missing type.
            constexpr int kCategoricalMask = 1;
            constexpr int kDefaultLeftMask = 2;

            struct TreeArrays {
                std::vector<int32_t> split_feature, left_child, right_child, cat_boundaries;
                std::vector<int> decision_type;
                std::vector<double> threshold, leaf_value;
                std::vector<uint32_t> cat_threshold;
            };

            // Children: >= 0 an internal node, < 0 leaf ~child.
            std::unique_ptr<TempBinNode> build(const TreeArrays& t, int32_t child, int depth) {
                if (depth > 10000) fail("tree too deep");
                auto node = std::make_unique<TempBinNode>();
                if (child < 0) {
                    const size_t leaf = static_cast<size_t>(~child);
                    if (leaf >= t.leaf_value.size()) fail("leaf index out of range");
                    node->is_leaf = true;
                    node->threshold_or_value = t.leaf_value[leaf];
                    return node;
                }
                const size_t i = static_cast<size_t>(child);
                if (i >= t.split_feature.size()) fail("node index out of range");
                node->feature_id = t.split_feature[i];
                if (node->feature_id < 0) fail("negative split_feature");
                const int dt = t.decision_type[i];
                if (dt & kCategoricalMask) {
                    // Same rule as the JSON parser: in the set -> left; NaN and anything else -> right.
                    const double raw = t.threshold[i];
                    if (!(raw >= 0) || raw + 1 >= static_cast<double>(t.cat_boundaries.size())) fail("categorical threshold out of range");
                    const size_t c = static_cast<size_t>(raw);
                    const int32_t begin = t.cat_boundaries[c], end = t.cat_boundaries[c + 1];
                    if (begin < 0 || end < begin || static_cast<size_t>(end) > t.cat_threshold.size()) fail("invalid cat_boundaries");
                    node->categorical = true;
                    node->default_dir = 1;
                    for (int32_t w = begin; w < end; w++) {
                        for (uint32_t bit = 0; bit < 32; bit++) {
                            if ((t.cat_threshold[static_cast<size_t>(w)] >> bit) & 1) node->categories.push_back(static_cast<uint32_t>(w - begin) * 32 + bit);
                        }
                    }
                } else {
                    lightgbm_numeric_split(*node, t.threshold[i], (dt & kDefaultLeftMask) != 0, (dt >> 2) & 3);
                }
                node->left = build(t, t.left_child[i], depth + 1);
                node->right = build(t, t.right_child[i], depth + 1);
                return node;
            }

            std::unique_ptr<TempBinNode> parse_tree(const Section& s, Doubles& doubles) {
                const auto linear = s.find("is_linear");
                if (linear != s.end() && linear->second != "0") fail("linear trees (linear_tree=true) are not supported");
                const int64_t n_leaves = parse_integer<int64_t>(field(s, "num_leaves"));
                if (n_leaves < 1) fail("num_leaves < 1");
                TreeArrays t;
                t.leaf_value = doubles.parse(field(s, "leaf_value"));
                if (static_cast<int64_t>(t.leaf_value.size()) != n_leaves) fail("leaf_value does not match num_leaves");
                if (n_leaves == 1) return build(t, ~0, 0);

                const size_t n_internal = static_cast<size_t>(n_leaves - 1);
                t.split_feature = parse_list<int32_t>(field(s, "split_feature"), parse_integer<int32_t>);
                t.threshold = doubles.parse(field(s, "threshold"));
                t.decision_type = parse_list<int>(field(s, "decision_type"), parse_integer<int>);
                t.left_child = parse_list<int32_t>(field(s, "left_child"), parse_integer<int32_t>);
                t.right_child = parse_list<int32_t>(field(s, "right_child"), parse_integer<int32_t>);
                for (size_t n : {t.split_feature.size(), t.threshold.size(), t.decision_type.size(), t.left_child.size(), t.right_child.size()}) {
                    if (n != n_internal) fail("split arrays do not match num_leaves");
                }
                const auto num_cat = s.find("num_cat");
                if (num_cat != s.end() && parse_integer<int64_t>(num_cat->second) > 0) {
                    t.cat_boundaries = parse_list<int32_t>(field(s, "cat_boundaries"), parse_integer<int32_t>);
                    t.cat_threshold = parse_list<uint32_t>(field(s, "cat_threshold"), parse_integer<uint32_t>);
                }
                return build(t, 0, 0);
            }

        } // namespace

        std::vector<std::unique_ptr<TempBinNode>> load_lightgbm_text(const std::string& filepath, ModelInfo& info) {
            const std::string content = read_file(filepath);
            std::string_view text(content);

            info = ModelInfo{};
            info.precision = Precision::Float64;
            std::vector<std::unique_ptr<TempBinNode>> forest;

            Section header, tree;
            Doubles doubles;
            bool in_tree = false, header_done = false, ended = false;
            auto finish_tree = [&] {
                if (in_tree) forest.push_back(parse_tree(tree, doubles));
                tree.clear();
                in_tree = false;
            };

            size_t pos = 0;
            while (pos < text.size() && !ended) {
                size_t end = text.find('\n', pos);
                if (end == std::string_view::npos) end = text.size();
                std::string_view line = text.substr(pos, end - pos);
                pos = end + 1;
                if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

                if (line.rfind("Tree=", 0) == 0) {
                    finish_tree();
                    header_done = true;
                    in_tree = true;
                    continue;
                }
                if (line == "end of trees") {
                    finish_tree();
                    ended = true;
                    continue;
                }
                if (line.empty()) {
                    if (in_tree) finish_tree();
                    continue;
                }
                const size_t eq = line.find('=');
                if (eq == std::string_view::npos) {
                    if (!header_done && line == "average_output") fail("random forest models (boosting=rf) are not supported yet");
                    continue;  // "tree", section titles
                }
                (in_tree ? tree : header)[line.substr(0, eq)] = line.substr(eq + 1);
            }
            finish_tree();
            if (!ended && forest.empty()) fail("no trees (is this a LightGBM model.txt?)");

            int64_t n_outputs = 1;
            if (auto it = header.find("num_tree_per_iteration"); it != header.end()) n_outputs = parse_integer<int64_t>(it->second);
            else if (auto it2 = header.find("num_class"); it2 != header.end()) n_outputs = parse_integer<int64_t>(it2->second);
            if (n_outputs < 1 || n_outputs > 65535) fail("unsupported number of trees per iteration");
            if (auto it = header.find("objective"); it != header.end()) lightgbm_objective(it->second, info);

            info.base_scores.assign(static_cast<size_t>(n_outputs), 0.0);
            if (n_outputs > 1) {
                for (size_t t = 0; t < forest.size(); t++) info.tree_outputs.push_back(static_cast<uint16_t>(t % static_cast<size_t>(n_outputs)));
            }
            return forest;
        }

    } // namespace parsers

} // namespace lignum
