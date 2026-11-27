#include "catboost.hpp"
#include "catboost_internal.hpp"
#include "parsers/parsers.hpp"
#include "simdjson.h"
#include "validate.hpp"

#include <lignum/lignum.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lignum {

    namespace {

        // Slot of a combination hash in a table; the hashes are already well mixed.
        inline uint64_t slot_of(uint64_t key, uint64_t mask) { return (key ^ (key >> 29)) & mask; }

        // Number of strictly increasing borders below x: the bucket with x > borders[k] <=> bucket >= k + 1.
        inline uint16_t bucket_of(const std::vector<float>& borders, float x) {
            return static_cast<uint16_t>(std::lower_bound(borders.begin(), borders.end(), x) - borders.begin());
        }

        // -----------------------------------------------------------------------------------------
        // JSON model

        [[noreturn]] void malformed(const std::string& what) {
            throw std::runtime_error("Malformed CatBoost model: " + what + ".");
        }

        uint64_t get_u64(simdjson::dom::element e) {
            if (e.is_string()) {
                const std::string_view s = e.get_string();
                uint64_t v = 0;
                const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
                if (ec != std::errc() || end != s.data() + s.size()) malformed("bad hash value");
                return v;
            }
            if (e.is_uint64()) return e.get_uint64();
            const int64_t v = e.get_int64();
            return static_cast<uint64_t>(v);
        }

        double get_number(simdjson::dom::element e) {
            if (e.is_double()) return e.get_double();
            if (e.is_int64()) return static_cast<double>(e.get_int64().value());
            if (e.is_uint64()) return static_cast<double>(e.get_uint64().value());
            malformed("expected a number");
        }

        template <typename T>
        T get_index(simdjson::dom::element e, size_t limit, const char* what) {
            const int64_t v = e.get_int64();
            if (v < 0 || static_cast<uint64_t>(v) >= limit) malformed(std::string(what) + " out of range");
            return static_cast<T>(v);
        }

        std::vector<float> float_list(simdjson::dom::array array, const char* what) {
            std::vector<float> values;
            for (simdjson::dom::element e : array) values.push_back(static_cast<float>(get_number(e)));
            if (!values.empty() && std::isnan(values[0])) malformed(std::string(what) + " contain NaN");
            for (size_t i = 1; i < values.size(); i++) {
                if (!(values[i - 1] < values[i])) malformed(std::string(what) + " are not strictly increasing");
            }
            return values;
        }

        bool one_of(std::string_view name, std::initializer_list<std::string_view> names) {
            for (std::string_view n : names) if (name == n) return true;
            return false;
        }

        // The output transform of what CatBoost's predict gives for the loss: predict_proba for
        // classifiers, predict for regressors and rankers. An unknown loss is rejected: its transform
        // is not known, so predictions could differ from CatBoost's. Without a loss in the file
        // (hand-made models), raw scores.
        uint8_t transform_of(std::string_view loss, uint32_t dim) {
            if (loss.empty()) return transform::Identity;
            if (one_of(loss, {"RMSE", "MAE", "Quantile", "MultiQuantile", "Expectile", "LogLinQuantile", "Lq", "Huber", "MAPE", "LogCosh",
                              "Cox", "SurvivalAft", "MultiRMSE", "MultiRMSEWithMissingValues", "YetiRank", "YetiRankPairwise",
                              "PairLogit", "PairLogitPairwise", "QueryRMSE", "QuerySoftMax", "LambdaMart", "StochasticRank",
                              "StochasticFilter", "GroupQuantile"})) {
                return transform::Identity;
            }
            if (one_of(loss, {"Logloss", "CrossEntropy", "Focal", "MultiClassOneVsAll", "MultiLogloss", "MultiCrossEntropy"})) {
                return transform::Sigmoid;  // per output
            }
            if (loss == "MultiClass") return dim > 1 ? transform::Softmax : transform::Identity;
            if (loss == "Poisson" || loss == "Tweedie") return transform::Exp;
            if (loss == "RMSEWithUncertainty") {
                throw std::runtime_error("CatBoost models with loss RMSEWithUncertainty are not supported yet (predict() gives "
                                         "exp(2 x) for the variance, with CatBoost's own exp).");
            }
            if (loss.substr(0, 17) == "PythonUserDefined") {
                // The file does not say: a regressor's predict gives raw scores, a classifier's
                // predict_proba the sigmoid, and a custom classifier stores no class parameters.
                throw std::runtime_error("CatBoost models trained with a Python objective are not supported: the file does not "
                                         "record whether predict gives raw scores (regressor) or probabilities (classifier).");
            }
            throw std::runtime_error("Unsupported CatBoost loss '" + std::string(loss) +
                                     "': lignum does not know how CatBoost transforms its raw scores, so predictions could differ. "
                                     "Please report it at https://github.com/misterekkk/lignum/issues.");
        }

        enum class CtrKind { Borders, Buckets, Counter, Mean };

        CtrKind ctr_kind(std::string_view type) {
            if (type == "Borders") return CtrKind::Borders;
            if (type == "Buckets") return CtrKind::Buckets;
            if (type == "Counter" || type == "FeatureFreq") return CtrKind::Counter;
            if (type == "BinarizedTargetMeanValue" || type == "FloatTargetMeanValue") return CtrKind::Mean;
            throw std::runtime_error("Unsupported CatBoost CTR type '" + std::string(type) + "'.");
        }

        // Learned statistics of one ctr_data entry: hash -> `stride - 1` numbers per entry.
        struct RawTable {
            std::vector<uint64_t> hashes;
            std::vector<double> values;  // entries x width
            size_t width = 0;
            int64_t denominator = 0;
        };

        RawTable read_table(simdjson::dom::object table) {
            RawTable raw;
            const int64_t stride = table["hash_stride"].get_int64();
            if (stride < 2 || stride > 1 << 16) malformed("bad hash_stride");
            raw.width = static_cast<size_t>(stride - 1);
            raw.denominator = table["counter_denominator"].get_int64();
            if (raw.denominator < 0 || raw.denominator > std::numeric_limits<int>::max()) malformed("bad counter_denominator");
            simdjson::dom::array map = table["hash_map"].get_array();
            if (map.size() % static_cast<size_t>(stride) != 0) malformed("hash_map size is not a multiple of hash_stride");
            size_t k = 0;
            for (simdjson::dom::element e : map) {
                if (k++ % static_cast<size_t>(stride) == 0) raw.hashes.push_back(get_u64(e));
                else {
                    // Counts are ints in CatBoost (and so converted below); mean sums are floats.
                    const double v = get_number(e);
                    if (!(std::fabs(v) <= std::numeric_limits<int>::max())) malformed("CTR statistic out of range");
                    raw.values.push_back(v);
                }
            }
            return raw;
        }

        // CatBoost's TModelCtr::Calc in float arithmetic: (count + prior) / (total + prior_denom),
        // then (ctr + shift) * scale.
        struct CtrFormula {
            float prior_num = 0, prior_den = 1, shift = 0, scale = 1;
            float operator()(float count, float total) const {
                const float ctr = (count + prior_num) / (total + prior_den);
                return (ctr + shift) * scale;
            }
        };

        // The CTR value of table entry `entry` (or of a missing combination when entry < 0), the
        // way TStaticCtrProvider::CalcCtrs reads the statistics for each CTR type.
        float ctr_value(CtrKind kind, const CtrFormula& calc, const RawTable& t, int64_t entry, size_t target_border) {
            if (entry < 0) return kind == CtrKind::Counter ? calc(0.0f, static_cast<float>(static_cast<int>(t.denominator))) : calc(0.0f, 0.0f);
            const double* h = &t.values[static_cast<size_t>(entry) * t.width];
            switch (kind) {
                case CtrKind::Counter:
                    return calc(static_cast<float>(static_cast<int>(h[0])), static_cast<float>(static_cast<int>(t.denominator)));
                case CtrKind::Mean:
                    if (t.width < 2) malformed("mean CTR table needs (sum, count)");
                    return calc(static_cast<float>(h[0]), static_cast<float>(static_cast<int>(h[1])));
                case CtrKind::Buckets: {
                    // CatBoost sums ints; int64 gives the same values without overflow.
                    int64_t good = static_cast<int64_t>(h[target_border]), total = 0;
                    for (size_t c = 0; c < t.width; c++) total += static_cast<int64_t>(h[c]);
                    return calc(static_cast<float>(good), static_cast<float>(total));
                }
                case CtrKind::Borders: {
                    // Binary targets: (class 1, class 0 + class 1); more classes: those above the border.
                    int64_t good = 0, total = 0;
                    for (size_t c = 0; c <= target_border; c++) total += static_cast<int64_t>(h[c]);
                    for (size_t c = target_border + 1; c < t.width; c++) good += static_cast<int64_t>(h[c]);
                    return calc(static_cast<float>(good), static_cast<float>(total + good));
                }
            }
            return 0.0f;
        }

        // -----------------------------------------------------------------------------------------
        // Library-neutral description of a CatBoost model, filled by the JSON and .cbm readers.
        // Features refer to CatBoost's feature indices; a split refers to a binary feature: float
        // borders (features in order), then one-hot values, then CTR borders.

        struct SpecFloat {
            int64_t index = 0;
            uint32_t column = 0;
            bool nan_as_true = false;
            std::vector<float> borders;
            std::string name;  // feature_id; empty for models trained on a matrix
        };
        struct SpecCat {
            int64_t index = 0;
            uint32_t column = 0;
            std::string name;  // feature_id; empty for models trained on a matrix
        };
        struct SpecOneHot {
            int64_t cat = 0;
            std::vector<uint32_t> values;
        };
        struct SpecCtr {
            std::vector<int64_t> cats;                           // hashed in this order
            std::vector<std::pair<int64_t, float>> floats;       // (float feature, border)
            std::vector<std::pair<int64_t, uint32_t>> one_hots;  // (categorical feature, value)
            std::string table;                                   // key into Spec::tables
            CtrKind kind = CtrKind::Borders;
            CtrFormula calc;
            int64_t target_border = 0;
            std::vector<float> borders;
        };
        struct Spec {
            std::vector<SpecFloat> floats;
            std::vector<SpecCat> cats;
            std::vector<SpecOneHot> one_hots;
            std::vector<SpecCtr> ctrs;
            std::map<std::string, RawTable> tables;
            std::vector<uint32_t> tree_depths;
            std::vector<int64_t> splits;  // binary feature of every split, tree after tree
            std::vector<double> leaves;
            uint32_t dim = 1;
            double scale = 1.0;
            std::vector<double> bias;
            std::string loss;
        };

        constexpr size_t kMaxColumns = 1 << 24;
        // Sources (float features, one-hot features, CTRs) are 16-bit indices; the .bin loader
        // accepts at most this many (validate below).
        constexpr uint32_t kMaxSources = 65535;

        void check_borders(const std::vector<float>& borders, const char* what) {
            if (borders.size() > 65534) malformed(std::string("too many ") + what);
            if (!borders.empty() && std::isnan(borders[0])) malformed(std::string(what) + " contain NaN");
            for (size_t i = 1; i < borders.size(); i++) {
                if (!(borders[i - 1] < borders[i])) malformed(std::string(what) + " are not strictly increasing");
            }
        }

        Model build_model(Spec& spec) {
            auto forest = std::make_shared<ObliviousForest>();
            ObliviousForest& f = *forest;
            std::vector<std::pair<uint16_t, uint16_t>> binary;  // binary feature -> (source, threshold)
            std::vector<uint8_t> binary_equal;
            uint32_t n_columns = 0;

            std::unordered_map<int64_t, uint32_t> float_source;  // float feature index -> source
            for (SpecFloat& sf : spec.floats) {
                if (sf.column >= kMaxColumns) malformed("feature column out of range");
                n_columns = std::max(n_columns, sf.column + 1);
                if (sf.borders.empty()) continue;
                check_borders(sf.borders, "float feature borders");
                const uint32_t s = static_cast<uint32_t>(f.floats.size());
                if (s >= kMaxSources) malformed("more than 65535 float, one-hot and CTR features");
                if (!float_source.emplace(sf.index, s).second) malformed("duplicate float feature");
                for (size_t k = 0; k < sf.borders.size(); k++) {
                    binary.emplace_back(static_cast<uint16_t>(s), static_cast<uint16_t>(k + 1));
                    binary_equal.push_back(0);
                }
                ObliviousForest::Float source;
                source.column = sf.column;
                source.nan_as_true = sf.nan_as_true;
                source.borders = std::move(sf.borders);
                f.floats.push_back(std::move(source));
            }
            std::unordered_map<int64_t, uint32_t> cat_index;  // categorical feature index -> cat_columns index
            for (const SpecCat& sc : spec.cats) {
                if (sc.column >= kMaxColumns) malformed("feature column out of range");
                n_columns = std::max(n_columns, sc.column + 1);
                if (!cat_index.emplace(sc.index, static_cast<uint32_t>(f.cat_columns.size())).second) malformed("duplicate categorical feature");
                f.cat_columns.push_back(sc.column);
            }
            const uint32_t n_floats = static_cast<uint32_t>(f.floats.size());
            std::unordered_map<int64_t, uint32_t> one_hot_source;  // categorical feature index -> one-hot index
            for (SpecOneHot& so : spec.one_hots) {
                auto it = cat_index.find(so.cat);
                if (it == cat_index.end()) malformed("one-hot feature of an unknown categorical feature");
                if (so.values.size() > 65534) malformed("too many one-hot values");
                const uint32_t h = static_cast<uint32_t>(f.one_hots.size());
                if (n_floats + h >= kMaxSources) malformed("more than 65535 float, one-hot and CTR features");
                if (!one_hot_source.emplace(so.cat, h).second) malformed("duplicate one-hot feature");
                for (size_t k = 0; k < so.values.size(); k++) {
                    binary.emplace_back(static_cast<uint16_t>(n_floats + h), static_cast<uint16_t>(k + 1));
                    binary_equal.push_back(1);
                }
                ObliviousForest::OneHot one_hot;
                one_hot.cat = it->second;
                one_hot.values = std::move(so.values);
                f.one_hots.push_back(std::move(one_hot));
            }

            // CTRs: projections shared by CTRs over the same combination, tables by key.
            std::map<std::string, uint32_t> projection_ids, table_ids;
            std::vector<const RawTable*> raw_tables;
            const uint32_t first_ctr_source = n_floats + static_cast<uint32_t>(f.one_hots.size());
            for (SpecCtr& sc : spec.ctrs) {
                ObliviousForest::Projection projection;
                std::string key;
                for (int64_t cat : sc.cats) {
                    auto it = cat_index.find(cat);
                    if (it == cat_index.end()) malformed("CTR uses an unknown categorical feature");
                    projection.cats.push_back(it->second);
                    key += "c" + std::to_string(cat) + ";";
                }
                for (const auto& [feature, border] : sc.floats) {
                    auto it = float_source.find(feature);
                    if (it == float_source.end()) malformed("CTR uses an unknown float feature");
                    const auto& borders = f.floats[it->second].borders;
                    auto pos = std::lower_bound(borders.begin(), borders.end(), border);
                    if (pos == borders.end() || *pos != border) malformed("CTR uses a float border the model does not have");
                    const auto k = static_cast<uint16_t>(pos - borders.begin() + 1);
                    projection.floats.emplace_back(it->second, k);
                    key += "f" + std::to_string(it->second) + ":" + std::to_string(k) + ";";
                }
                for (const auto& [feature, value] : sc.one_hots) {
                    auto it = one_hot_source.find(feature);
                    if (it == one_hot_source.end()) malformed("CTR uses a one-hot value the model does not have");
                    const auto& values = f.one_hots[it->second].values;
                    auto pos = std::find(values.begin(), values.end(), value);
                    if (pos == values.end()) malformed("CTR uses a one-hot value the model does not have");
                    const auto k = static_cast<uint16_t>(pos - values.begin() + 1);
                    projection.one_hots.emplace_back(n_floats + it->second, k);
                    key += "o" + std::to_string(it->second) + ":" + std::to_string(k) + ";";
                }
                if (projection.cats.empty()) malformed("CTR without categorical features");
                auto [pit, new_projection] = projection_ids.try_emplace(key, static_cast<uint32_t>(f.projections.size()));
                if (new_projection) f.projections.push_back(std::move(projection));

                auto raw_it = spec.tables.find(sc.table);
                if (raw_it == spec.tables.end()) malformed("no CTR table for a CTR");
                auto [tit, new_table] = table_ids.try_emplace(sc.table, static_cast<uint32_t>(raw_tables.size()));
                if (new_table) {
                    raw_tables.push_back(&raw_it->second);
                    ObliviousForest::Table t;
                    t.projection = pit->second;
                    f.projections[pit->second].tables.push_back(static_cast<uint32_t>(f.tables.size()));
                    f.tables.push_back(std::move(t));
                } else if (f.tables[tit->second].projection != pit->second) {
                    malformed("one CTR table for two combinations");
                }

                const RawTable& raw = *raw_tables[tit->second];
                if (raw.width == 0 || raw.values.size() != raw.hashes.size() * raw.width) malformed("CTR table size");
                if (sc.kind == CtrKind::Mean && raw.width < 2) malformed("mean CTR table needs (sum, count)");
                if (sc.target_border < 0 || static_cast<size_t>(sc.target_border) >= std::max<size_t>(raw.width, 1)) {
                    malformed("target_border_idx out of range");
                }
                const auto target_border = static_cast<size_t>(sc.target_border);
                check_borders(sc.borders, "CTR borders");
                ObliviousForest::Ctr ctr;
                ctr.table = tit->second;
                ctr.entry_buckets.resize(raw.hashes.size() + 1);
                ctr.entry_buckets[0] = bucket_of(sc.borders, ctr_value(sc.kind, sc.calc, raw, -1, target_border));
                for (size_t entry = 0; entry < raw.hashes.size(); entry++) {
                    ctr.entry_buckets[entry + 1] = bucket_of(sc.borders, ctr_value(sc.kind, sc.calc, raw, static_cast<int64_t>(entry), target_border));
                }
                const uint32_t s = first_ctr_source + static_cast<uint32_t>(f.ctrs.size());
                if (s >= kMaxSources) malformed("more than 65535 float, one-hot and CTR features");
                for (size_t k = 0; k < sc.borders.size(); k++) {
                    binary.emplace_back(static_cast<uint16_t>(s), static_cast<uint16_t>(k + 1));
                    binary_equal.push_back(0);
                }
                f.ctrs.push_back(std::move(ctr));
            }

            // Open-addressing index of every table (load factor <= 1/2). CatBoost marks unused
            // buckets of its own hash view with 2^64 - 1; they are not entries.
            for (size_t t = 0; t < f.tables.size(); t++) {
                ObliviousForest::Table& table = f.tables[t];
                const RawTable& raw = *raw_tables[t];
                uint64_t capacity = 2;
                while (capacity < 2 * raw.hashes.size()) capacity <<= 1;
                table.mask = capacity - 1;
                table.keys.assign(capacity, 0);
                table.slots.assign(capacity, 0);
                for (size_t entry = 0; entry < raw.hashes.size(); entry++) {
                    const uint64_t key = raw.hashes[entry];
                    if (key == ~uint64_t{0}) continue;
                    uint64_t slot = slot_of(key, table.mask);
                    while (table.slots[slot] != 0 && table.keys[slot] != key) slot = (slot + 1) & table.mask;
                    if (table.slots[slot] != 0) continue;  // duplicate hash: CatBoost finds the first
                    table.keys[slot] = key;
                    table.slots[slot] = static_cast<uint32_t>(entry + 1);
                }
            }

            if (spec.dim == 0 || spec.dim > 65535 || spec.bias.size() != spec.dim) malformed("bias");
            f.dim = spec.dim;
            f.scale = spec.scale;
            f.bias = spec.bias;

            f.split_begin.push_back(0);
            size_t split = 0, leaf = 0;
            for (uint32_t depth : spec.tree_depths) {
                if (depth > 16) malformed("tree deeper than 16");
                if (split + depth > spec.splits.size()) malformed("tree splits");
                for (uint32_t d = 0; d < depth; d++, split++) {
                    const int64_t index = spec.splits[split];
                    if (index < 0 || static_cast<size_t>(index) >= binary.size()) malformed("split index out of range");
                    f.split_source.push_back(binary[static_cast<size_t>(index)].first);
                    f.split_threshold.push_back(binary[static_cast<size_t>(index)].second);
                    f.split_equal.push_back(binary_equal[static_cast<size_t>(index)]);
                }
                f.split_begin.push_back(static_cast<uint32_t>(f.split_source.size()));
                f.leaf_begin.push_back(leaf);
                leaf += (size_t{1} << depth) * f.dim;
                if (leaf > spec.leaves.size()) malformed("leaf values size does not match the tree depths");
            }
            if (split != spec.splits.size() || leaf != spec.leaves.size()) malformed("tree sizes do not match the splits and leaves");
            f.leaves = std::move(spec.leaves);
            f.n_columns = n_columns;
            catboost_detail::derive_layout(f);

            Model model;
            model.n_outputs = f.dim;
            model.transform = transform_of(spec.loss, f.dim);
            model.n_features = n_columns;
            model.base_scores = f.bias.data();
            model.oblivious = forest;
            model.data = forest;
            model.source = model_source::CatBoost;
            model.objective = spec.loss;
            std::vector<std::string> names(n_columns);
            for (const SpecFloat& sf : spec.floats) names[sf.column] = sf.name;
            for (const SpecCat& sc : spec.cats) names[sc.column] = sc.name;
            model.feature_names = model_feature_names(std::move(names), n_columns, model_source::CatBoost);
            return model;
        }

        // -----------------------------------------------------------------------------------------
        // JSON (save_model(format="json"))

        Spec read_json(simdjson::dom::element doc) {
            Spec spec;
            simdjson::dom::object features = doc["features_info"].get_object();
            for (const char* unsupported : {"text_features", "embedding_features", "estimated_features"}) {
                simdjson::dom::array array;
                if (features[unsupported].get_array().get(array) == simdjson::SUCCESS && array.size() > 0) {
                    throw std::runtime_error(std::string("CatBoost models with ") + unsupported + " are not supported.");
                }
            }
            simdjson::dom::array trees;
            if (doc["oblivious_trees"].get_array().get(trees) != simdjson::SUCCESS) {
                throw std::runtime_error("Only CatBoost models with symmetric (oblivious) trees are supported; "
                                         "this one was trained with grow_policy Depthwise or Lossguide.");
            }
            simdjson::dom::array float_features;
            if (features["float_features"].get_array().get(float_features) == simdjson::SUCCESS) {
                for (simdjson::dom::object ff : float_features) {
                    SpecFloat sf;
                    sf.index = ff["feature_index"].get_int64();
                    sf.column = get_index<uint32_t>(ff["flat_feature_index"], kMaxColumns, "flat_feature_index");
                    simdjson::dom::array borders;
                    if (ff["borders"].get_array().get(borders) == simdjson::SUCCESS) sf.borders = float_list(borders, "float feature borders");
                    std::string_view nan_treatment;  // optional field
                    if (ff["nan_value_treatment"].get_string().get(nan_treatment) != simdjson::SUCCESS) nan_treatment = "AsIs";
                    sf.nan_as_true = nan_treatment == "AsTrue";
                    std::string_view id;
                    if (ff["feature_id"].get_string().get(id) == simdjson::SUCCESS) sf.name = std::string(id);
                    spec.floats.push_back(std::move(sf));
                }
            }
            simdjson::dom::array cat_features;
            if (features["categorical_features"].get_array().get(cat_features) == simdjson::SUCCESS) {
                for (simdjson::dom::object cf : cat_features) {
                    SpecCat sc;
                    sc.index = cf["feature_index"].get_int64();
                    sc.column = get_index<uint32_t>(cf["flat_feature_index"], kMaxColumns, "flat_feature_index");
                    std::string_view id;
                    if (cf["feature_id"].get_string().get(id) == simdjson::SUCCESS) sc.name = std::string(id);
                    spec.cats.push_back(sc);
                    simdjson::dom::array values;
                    if (cf["values"].get_array().get(values) == simdjson::SUCCESS && values.size() > 0) {
                        SpecOneHot so;
                        so.cat = sc.index;
                        for (simdjson::dom::element v : values) so.values.push_back(static_cast<uint32_t>(v.get_int64().value()));
                        spec.one_hots.push_back(std::move(so));
                    }
                }
            }
            simdjson::dom::object ctr_data;
            const bool has_ctr_data = doc["ctr_data"].get_object().get(ctr_data) == simdjson::SUCCESS;
            simdjson::dom::array ctrs;
            if (features["ctrs"].get_array().get(ctrs) == simdjson::SUCCESS) {
                for (simdjson::dom::object cj : ctrs) {
                    SpecCtr sc;
                    for (simdjson::dom::object e : cj["elements"].get_array()) {
                        const std::string_view kind = e["combination_element"].get_string();
                        if (kind == "cat_feature_value") sc.cats.push_back(e["cat_feature_index"].get_int64());
                        else if (kind == "float_feature") sc.floats.emplace_back(e["float_feature_index"].get_int64(), static_cast<float>(get_number(e["border"])));
                        else if (kind == "cat_feature_exact_value") sc.one_hots.emplace_back(e["cat_feature_index"].get_int64(), static_cast<uint32_t>(e["value"].get_int64().value()));
                        else throw std::runtime_error("Unsupported CatBoost CTR element '" + std::string(kind) + "'.");
                    }
                    sc.table = std::string(cj["identifier"].get_string().value());
                    if (!spec.tables.count(sc.table)) {
                        simdjson::dom::object table;
                        if (!has_ctr_data || ctr_data[sc.table].get_object().get(table) != simdjson::SUCCESS) malformed("ctr_data has no table for " + sc.table);
                        spec.tables.emplace(sc.table, read_table(table));
                    }
                    sc.kind = ctr_kind(cj["ctr_type"].get_string());
                    sc.calc.prior_num = static_cast<float>(get_number(cj["prior_numerator"]));
                    sc.calc.prior_den = static_cast<float>(get_number(cj["prior_denomerator"]));
                    sc.calc.shift = static_cast<float>(get_number(cj["shift"]));
                    sc.calc.scale = static_cast<float>(get_number(cj["scale"]));
                    sc.target_border = cj["target_border_idx"].get_int64();
                    sc.borders = float_list(cj["borders"].get_array(), "CTR borders");
                    spec.ctrs.push_back(std::move(sc));
                }
            }
            simdjson::dom::array scale_and_bias = doc["scale_and_bias"].get_array();
            if (scale_and_bias.size() != 2) malformed("scale_and_bias");
            spec.scale = get_number(scale_and_bias.at(0));
            for (simdjson::dom::element b : scale_and_bias.at(1).get_array()) spec.bias.push_back(get_number(b));
            spec.dim = static_cast<uint32_t>(std::min<size_t>(spec.bias.size(), 1 << 20));
            for (simdjson::dom::object tree : trees) {
                // A tree of depth 0 (a single leaf) has "splits": null.
                simdjson::dom::array splits;
                uint32_t depth = 0;
                if (tree["splits"].get_array().get(splits) == simdjson::SUCCESS) {
                    if (splits.size() > 16) malformed("tree deeper than 16");
                    depth = static_cast<uint32_t>(splits.size());
                    for (simdjson::dom::object split : splits) spec.splits.push_back(split["split_index"].get_int64());
                } else if (!tree["splits"].is_null()) {
                    malformed("splits");
                }
                spec.tree_depths.push_back(depth);
                for (simdjson::dom::element v : tree["leaf_values"].get_array()) spec.leaves.push_back(get_number(v));
            }
            simdjson::dom::element loss_type;
            if (doc.at_pointer("/model_info/params/loss_function/type").get(loss_type) == simdjson::SUCCESS) spec.loss = std::string(loss_type.get_string().value());
            return spec;
        }

        // -----------------------------------------------------------------------------------------
        // .cbm: "CBM1", a size, the TModelCore FlatBuffer, then the CTR tables (a count, then a size
        // and a TCtrValueTable FlatBuffer each). Read by hand (no flatbuffers dependency), every
        // offset checked against the buffer.

        class Flat {
            public:
                Flat(const uint8_t* data, size_t size) : data_(data), size_(size) {}

                template <typename T>
                T read(size_t pos) const {
                    if (pos > size_ || size_ - pos < sizeof(T)) malformed("FlatBuffer offset out of range");
                    T v;
                    std::memcpy(&v, data_ + pos, sizeof(T));
                    return v;
                }
                size_t size() const { return size_; }
                const uint8_t* at(size_t pos) const { return data_ + pos; }

            private:
                const uint8_t* data_;
                size_t size_;
        };

        struct FlatVector {
            const Flat* flat = nullptr;
            size_t start = 0;  // first element
            uint32_t length = 0;

            size_t element(size_t k, size_t width) const { return start + k * width; }
        };

        struct FlatTable {
            const Flat* flat = nullptr;
            size_t pos = 0;
            size_t vtable = 0;
            uint16_t vtable_size = 0;

            // Position of field i, or 0 when absent.
            size_t field(int i) const {
                const size_t entry = 4 + 2 * static_cast<size_t>(i);
                if (entry + 2 > vtable_size) return 0;
                const uint16_t offset = flat->read<uint16_t>(vtable + entry);
                return offset ? pos + offset : 0;
            }
            template <typename T>
            T scalar(int i, T fallback) const {
                const size_t p = field(i);
                return p ? flat->read<T>(p) : fallback;
            }
            bool has(int i) const { return field(i) != 0; }
            // The target of the uoffset stored at `at`.
            size_t follow(size_t at) const {
                const uint32_t offset = flat->read<uint32_t>(at);
                if (offset == 0 || at + offset >= flat->size()) malformed("FlatBuffer offset out of range");
                return at + offset;
            }
            FlatVector vector(int i, size_t width) const {
                FlatVector v;
                v.flat = flat;
                const size_t p = field(i);
                if (!p) return v;
                const size_t target = follow(p);
                v.length = flat->read<uint32_t>(target);
                v.start = target + 4;
                if (v.length > (flat->size() - v.start) / std::max<size_t>(width, 1)) malformed("FlatBuffer vector out of range");
                return v;
            }
            std::string_view string(int i) const {
                const FlatVector v = vector(i, 1);
                return v.length ? std::string_view(reinterpret_cast<const char*>(flat->at(v.start)), v.length) : std::string_view();
            }
        };

        FlatTable table_at(const Flat& flat, size_t pos) {
            FlatTable t;
            t.flat = &flat;
            t.pos = pos;
            const int64_t vtable = static_cast<int64_t>(pos) - flat.read<int32_t>(pos);
            if (vtable < 0 || static_cast<uint64_t>(vtable) + 4 > flat.size()) malformed("FlatBuffer vtable out of range");
            t.vtable = static_cast<size_t>(vtable);
            t.vtable_size = flat.read<uint16_t>(t.vtable);
            if (t.vtable_size < 4 || t.vtable + t.vtable_size > flat.size()) malformed("FlatBuffer vtable out of range");
            return t;
        }
        FlatTable root(const Flat& flat) { return table_at(flat, static_cast<size_t>(flat.read<uint32_t>(0))); }
        // Field i of t as a table; `present` is false when it is absent.
        FlatTable subtable(const FlatTable& t, int i, bool* present = nullptr) {
            const size_t p = t.field(i);
            if (present) *present = p != 0;
            if (!p) {
                if (!present) malformed("missing FlatBuffer table");
                return FlatTable{};
            }
            return table_at(*t.flat, t.follow(p));
        }
        FlatTable table_in(const FlatVector& v, size_t k) {
            const size_t at = v.element(k, 4);
            FlatTable dummy;
            dummy.flat = v.flat;
            return table_at(*v.flat, dummy.follow(at));
        }
        template <typename T>
        std::vector<T> scalars(const FlatTable& t, int i) {
            const FlatVector v = t.vector(i, sizeof(T));
            std::vector<T> out(v.length);
            for (uint32_t k = 0; k < v.length; k++) out[k] = v.flat->template read<T>(v.element(k, sizeof(T)));
            return out;
        }

        // TModelCtrBase as a key: the same in the model's CTR features and in its CTR tables.
        std::string ctr_base_key(const FlatTable& base, SpecCtr* ctr) {
            bool has_combination = false;
            const FlatTable combination = subtable(base, 0, &has_combination);
            std::string key;
            if (has_combination) {
                for (int32_t cat : scalars<int32_t>(combination, 0)) {
                    key += "c" + std::to_string(cat) + ";";
                    if (ctr) ctr->cats.push_back(cat);
                }
                const FlatVector floats = combination.vector(1, 8);
                for (uint32_t k = 0; k < floats.length; k++) {
                    const int32_t index = floats.flat->read<int32_t>(floats.element(k, 8));
                    const float border = floats.flat->read<float>(floats.element(k, 8) + 4);
                    uint32_t bits;
                    std::memcpy(&bits, &border, 4);
                    key += "f" + std::to_string(index) + ":" + std::to_string(bits) + ";";
                    if (ctr) ctr->floats.emplace_back(index, border);
                }
                const FlatVector one_hots = combination.vector(2, 8);
                for (uint32_t k = 0; k < one_hots.length; k++) {
                    const int32_t index = one_hots.flat->read<int32_t>(one_hots.element(k, 8));
                    const int32_t value = one_hots.flat->read<int32_t>(one_hots.element(k, 8) + 4);
                    key += "o" + std::to_string(index) + ":" + std::to_string(value) + ";";
                    if (ctr) ctr->one_hots.emplace_back(index, static_cast<uint32_t>(value));
                }
            }
            const int8_t type = base.scalar<int8_t>(1, 0);
            key += "t" + std::to_string(type) + ":" + std::to_string(base.scalar<int32_t>(2, 0));
            if (ctr) {
                switch (type) {
                    case 0: ctr->kind = CtrKind::Borders; break;
                    case 1: ctr->kind = CtrKind::Buckets; break;
                    case 2: case 3: ctr->kind = CtrKind::Mean; break;
                    case 4: case 5: ctr->kind = CtrKind::Counter; break;
                    default: throw std::runtime_error("Unsupported CatBoost CTR type " + std::to_string(type) + ".");
                }
            }
            return key;
        }

        size_t read_size(const Flat& file, size_t& pos) {
            uint64_t size = file.read<uint32_t>(pos);
            pos += 4;
            if (size == 0xffffffffu) {
                size = file.read<uint64_t>(pos);
                pos += 8;
            }
            if (size > file.size() - pos) malformed("section size out of range");
            return static_cast<size_t>(size);
        }

        Spec read_cbm(const std::string& bytes) {
            const Flat file(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
            if (bytes.size() < 8 || bytes.compare(0, 4, "CBM1") != 0) malformed("not a CatBoost .cbm file");
            size_t pos = 4;
            const size_t core_size = read_size(file, pos);
            const Flat core(file.at(pos), core_size);
            pos += core_size;

            Spec spec;
            const FlatTable model = root(core);
            if (model.string(0) != "FlabuffersModel_v1") malformed("unknown .cbm core format");
            const FlatVector info = model.vector(2, 4);
            for (uint32_t k = 0; k < info.length; k++) {
                const FlatTable kv = table_in(info, k);
                if (kv.string(0) != "params") continue;
                // The training parameters, as JSON: only the loss decides the output transform.
                simdjson::dom::parser parser;
                simdjson::dom::element params;
                simdjson::dom::element type;
                const std::string_view text = kv.string(1);
                // An empty value has no data pointer: simdjson would memcpy from nullptr (undefined,
                // and an error under UBSan with glibc, which declares memcpy's pointers nonnull).
                if (text.empty()) continue;
                if (parser.parse(text.data(), text.size()).get(params) == simdjson::SUCCESS &&
                    params.at_pointer("/loss_function/type").get(type) == simdjson::SUCCESS && type.is_string()) {
                    spec.loss = std::string(type.get_string().value());
                }
            }
            bool has_ctr_part = false;
            const FlatVector parts = model.vector(3, 4);
            for (uint32_t k = 0; k < parts.length; k++) {
                const size_t at = parts.element(k, 4);
                FlatTable dummy;
                dummy.flat = &core;
                const size_t target = dummy.follow(at);
                const uint32_t length = core.read<uint32_t>(target);
                if (length > core.size() - target - 4) malformed("FlatBuffer string out of range");
                const std::string_view part(reinterpret_cast<const char*>(core.at(target + 4)), length);
                if (part == "static_provider_v1") has_ctr_part = true;
                else throw std::runtime_error("CatBoost models with " + std::string(part) + " (text or embedding features) are not supported.");
            }

            const FlatTable trees = subtable(model, 1);
            spec.dim = static_cast<uint32_t>(std::clamp<int32_t>(trees.scalar<int32_t>(0, 1), 0, 1 << 20));
            for (int i : {10, 11}) {
                if (trees.vector(i, 4).length > 0) {
                    throw std::runtime_error("Only CatBoost models with symmetric (oblivious) trees are supported; "
                                             "this one was trained with grow_policy Depthwise or Lossguide.");
                }
            }
            for (int i : {12, 13, 18}) {
                if (trees.vector(i, 4).length > 0) throw std::runtime_error("CatBoost models with text, embedding or estimated features are not supported.");
            }
            for (int32_t s : scalars<int32_t>(trees, 1)) spec.splits.push_back(s);
            for (int32_t d : scalars<int32_t>(trees, 2)) {
                if (d < 0 || d > 16) malformed("tree deeper than 16");
                spec.tree_depths.push_back(static_cast<uint32_t>(d));
            }
            const FlatVector cats = trees.vector(4, 4);
            for (uint32_t k = 0; k < cats.length; k++) {
                const FlatTable c = table_in(cats, k);
                SpecCat sc;
                sc.index = c.scalar<int32_t>(0, -1);
                const int32_t column = c.scalar<int32_t>(1, -1);
                if (column < 0) malformed("feature column out of range");
                sc.column = static_cast<uint32_t>(column);
                sc.name = std::string(c.string(2));  // FeatureId
                spec.cats.push_back(sc);
            }
            const FlatVector floats = trees.vector(5, 4);
            for (uint32_t k = 0; k < floats.length; k++) {
                const FlatTable fl = table_in(floats, k);
                SpecFloat sf;
                sf.index = fl.scalar<int32_t>(1, -1);
                const int32_t column = fl.scalar<int32_t>(2, -1);
                if (column < 0) malformed("feature column out of range");
                sf.column = static_cast<uint32_t>(column);
                sf.borders = scalars<float>(fl, 3);
                sf.nan_as_true = fl.scalar<int8_t>(5, 0) == 2;  // ENanValueTreatment::AsTrue
                sf.name = std::string(fl.string(4));  // FeatureId
                spec.floats.push_back(std::move(sf));
            }
            const FlatVector one_hots = trees.vector(6, 4);
            for (uint32_t k = 0; k < one_hots.length; k++) {
                const FlatTable oh = table_in(one_hots, k);
                SpecOneHot so;
                so.cat = oh.scalar<int32_t>(0, -1);
                for (int32_t v : scalars<int32_t>(oh, 1)) so.values.push_back(static_cast<uint32_t>(v));
                spec.one_hots.push_back(std::move(so));
            }
            const FlatVector ctrs = trees.vector(7, 4);
            for (uint32_t k = 0; k < ctrs.length; k++) {
                const FlatTable feature = table_in(ctrs, k);
                const FlatTable ctr = subtable(feature, 0);
                SpecCtr sc;
                sc.table = ctr_base_key(subtable(ctr, 0), &sc);
                sc.target_border = ctr.scalar<int32_t>(1, 0);
                sc.calc.prior_num = ctr.scalar<float>(2, 0.0f);
                sc.calc.prior_den = ctr.scalar<float>(3, 1.0f);
                sc.calc.shift = ctr.scalar<float>(4, 0.0f);
                sc.calc.scale = ctr.scalar<float>(5, 1.0f);
                sc.borders = scalars<float>(feature, 1);
                spec.ctrs.push_back(std::move(sc));
            }
            spec.leaves = scalars<double>(trees, 8);
            spec.scale = trees.scalar<double>(14, 1.0);
            spec.bias = scalars<double>(trees, 16);
            if (spec.bias.empty()) spec.bias.assign(spec.dim, trees.scalar<double>(15, 0.0));

            // CTR tables: TBucket {ui64 hash; ui32 index} (packed) -> statistics in the blob.
            if (has_ctr_part) {
                const size_t n_tables = read_size(file, pos);
                for (size_t t = 0; t < n_tables; t++) {
                    const size_t size = read_size(file, pos);
                    const Flat buffer(file.at(pos), size);
                    pos += size;
                    const FlatTable table = root(buffer);
                    SpecCtr probe;
                    const std::string key = ctr_base_key(subtable(table, 0), &probe);
                    const FlatVector buckets = table.vector(1, 1);
                    const FlatVector blob = table.vector(2, 1);
                    RawTable raw;
                    raw.denominator = table.scalar<int32_t>(3, 0);
                    if (raw.denominator < 0) malformed("bad counter_denominator");
                    const int32_t classes = table.scalar<int32_t>(4, 0);
                    size_t entries = 0;
                    if (probe.kind == CtrKind::Mean) {
                        raw.width = 2;
                        entries = blob.length / 8;
                        for (size_t e = 0; e < entries; e++) {
                            raw.values.push_back(static_cast<double>(buffer.read<float>(blob.start + 8 * e)));
                            raw.values.push_back(static_cast<double>(buffer.read<int32_t>(blob.start + 8 * e + 4)));
                        }
                    } else {
                        raw.width = probe.kind == CtrKind::Counter ? 1 : static_cast<size_t>(classes);
                        if (raw.width == 0 || raw.width > (1 << 16)) malformed("bad TargetClassesCount");
                        entries = blob.length / (4 * raw.width);
                        for (size_t e = 0; e < entries * raw.width; e++) raw.values.push_back(static_cast<double>(buffer.read<int32_t>(blob.start + 4 * e)));
                    }
                    raw.hashes.assign(entries, ~uint64_t{0});
                    for (size_t b = 0; b + 12 <= buckets.length; b += 12) {
                        const uint64_t hash = buffer.read<uint64_t>(buckets.start + b);
                        const uint32_t index = buffer.read<uint32_t>(buckets.start + b + 8);
                        if (hash == ~uint64_t{0}) continue;
                        if (index >= entries) malformed("CTR bucket index out of range");
                        raw.hashes[index] = hash;
                    }
                    spec.tables[key] = std::move(raw);
                }
            }
            return spec;
        }

    } // namespace

    namespace parsers {

        Model load_catboost_json(const std::string& filepath) {
            const simdjson::padded_string padded(read_file(filepath));
            simdjson::dom::parser parser;
            simdjson::dom::element doc;
            if (auto error = parser.parse(padded).get(doc)) {
                throw std::runtime_error("Cannot parse CatBoost JSON " + filepath + ": " + simdjson::error_message(error));
            }
            try {
                Spec spec = read_json(doc);
                return build_model(spec);
            } catch (const simdjson::simdjson_error& e) {
                throw std::runtime_error("Malformed CatBoost model " + filepath + ": " + e.what());
            }
        }

        Model load_catboost_cbm(const std::string& filepath) {
            try {
                Spec spec = read_cbm(read_file(filepath));
                return build_model(spec);
            } catch (const simdjson::simdjson_error& e) {
                throw std::runtime_error("Malformed CatBoost model " + filepath + ": " + e.what());
            }
        }

    } // namespace parsers

} // namespace lignum
