#include "catboost.hpp"
#include "simdjson.h"

#include <lignum/lignum.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace lignum {

    // ---------------------------------------------------------------------------------------------
    // CityHash64 as in CatBoost's util/digest/city.cpp (Google's first release, MIT licence):
    // categories are hashed with CityHash64(utf8) & 0xffffffff. Later CityHash versions give
    // different values (for over 64 bytes even 1.0.3 does).

    namespace {

        constexpr uint64_t k0 = 0xc3a5c85c97cb3127ULL;
        constexpr uint64_t k1 = 0xb492b66fbe98f273ULL;
        constexpr uint64_t k2 = 0x9ae16a3b2f90404fULL;
        constexpr uint64_t k3 = 0xc949d7c7509e6557ULL;

        inline uint64_t fetch64(const char* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
        inline uint32_t fetch32(const char* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
        inline uint64_t rotate(uint64_t v, int s) { return s == 0 ? v : ((v >> s) | (v << (64 - s))); }
        inline uint64_t rotate_by_at_least1(uint64_t v, int s) { return (v >> s) | (v << (64 - s)); }
        inline uint64_t shift_mix(uint64_t v) { return v ^ (v >> 47); }

        inline uint64_t hash_len16(uint64_t u, uint64_t v) {
            constexpr uint64_t kMul = 0x9ddfea08eb382d69ULL;
            uint64_t a = (u ^ v) * kMul;
            a ^= (a >> 47);
            uint64_t b = (v ^ a) * kMul;
            b ^= (b >> 47);
            return b * kMul;
        }

        uint64_t hash_len0to16(const char* s, size_t len) {
            if (len > 8) {
                const uint64_t a = fetch64(s);
                const uint64_t b = fetch64(s + len - 8);
                return hash_len16(a, rotate_by_at_least1(b + len, static_cast<int>(len))) ^ b;
            }
            if (len >= 4) {
                const uint64_t a = fetch32(s);
                return hash_len16(len + (a << 3), fetch32(s + len - 4));
            }
            if (len > 0) {
                const uint8_t a = static_cast<uint8_t>(s[0]);
                const uint8_t b = static_cast<uint8_t>(s[len >> 1]);
                const uint8_t c = static_cast<uint8_t>(s[len - 1]);
                const uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
                const uint32_t z = static_cast<uint32_t>(len) + (static_cast<uint32_t>(c) << 2);
                return shift_mix(y * k2 ^ z * k3) * k2;
            }
            return k2;
        }

        uint64_t hash_len17to32(const char* s, size_t len) {
            const uint64_t a = fetch64(s) * k1;
            const uint64_t b = fetch64(s + 8);
            const uint64_t c = fetch64(s + len - 8) * k2;
            const uint64_t d = fetch64(s + len - 16) * k0;
            return hash_len16(rotate(a - b, 43) + rotate(c, 30) + d, a + rotate(b ^ k3, 20) - c + len);
        }

        std::pair<uint64_t, uint64_t> weak_hash_len32_with_seeds(uint64_t w, uint64_t x, uint64_t y, uint64_t z, uint64_t a, uint64_t b) {
            a += w;
            b = rotate(b + a + z, 21);
            const uint64_t c = a;
            a += x;
            a += y;
            b += rotate(a, 44);
            return {a + z, b + c};
        }

        std::pair<uint64_t, uint64_t> weak_hash_len32_with_seeds(const char* s, uint64_t a, uint64_t b) {
            return weak_hash_len32_with_seeds(fetch64(s), fetch64(s + 8), fetch64(s + 16), fetch64(s + 24), a, b);
        }

        uint64_t hash_len33to64(const char* s, size_t len) {
            uint64_t z = fetch64(s + 24);
            uint64_t a = fetch64(s) + (len + fetch64(s + len - 16)) * k0;
            uint64_t b = rotate(a + z, 52);
            uint64_t c = rotate(a, 37);
            a += fetch64(s + 8);
            c += rotate(a, 7);
            a += fetch64(s + 16);
            const uint64_t vf = a + z;
            const uint64_t vs = b + rotate(a, 31) + c;
            a = fetch64(s + 16) + fetch64(s + len - 32);
            z = fetch64(s + len - 8);
            b = rotate(a + z, 52);
            c = rotate(a, 37);
            a += fetch64(s + len - 24);
            c += rotate(a, 7);
            a += fetch64(s + len - 16);
            const uint64_t wf = a + z;
            const uint64_t ws = b + rotate(a, 31) + c;
            const uint64_t r = shift_mix((vf + ws) * k2 + (wf + vs) * k0);
            return shift_mix(r * k0 + vs) * k2;
        }

    } // namespace

    uint64_t city_hash64(const char* s, size_t len) {
        if (len <= 32) return len <= 16 ? hash_len0to16(s, len) : hash_len17to32(s, len);
        if (len <= 64) return hash_len33to64(s, len);

        // Over 64 bytes: the end first, then 64-byte chunks with 56 bytes of state.
        uint64_t x = fetch64(s);
        uint64_t y = fetch64(s + len - 16) ^ k1;
        uint64_t z = fetch64(s + len - 56) ^ k0;
        auto v = weak_hash_len32_with_seeds(s + len - 64, len, y);
        auto w = weak_hash_len32_with_seeds(s + len - 32, len * k1, k0);
        z += shift_mix(v.second) * k1;
        x = rotate(z + x, 39) * k1;
        y = rotate(y, 33) * k1;
        len = (len - 1) & ~static_cast<size_t>(63);
        do {
            x = rotate(x + y + v.first + fetch64(s + 16), 37) * k1;
            y = rotate(y + v.second + fetch64(s + 48), 42) * k1;
            x ^= w.second;
            y ^= v.first;
            z = rotate(z ^ w.first, 33);
            v = weak_hash_len32_with_seeds(s, v.second * k1, x + w.first);
            w = weak_hash_len32_with_seeds(s + 32, z + w.second, y);
            std::swap(z, x);
            s += 64;
            len -= 64;
        } while (len != 0);
        return hash_len16(hash_len16(v.first, w.first) + shift_mix(y) * k1 + z, hash_len16(v.second, w.second) + x);
    }

    uint32_t catboost_hash(std::string_view category) {
        return static_cast<uint32_t>(city_hash64(category.data(), category.size()));
    }

    namespace {

        // CatBoost's CalcHash: combines a projection's hash with one more value.
        inline uint64_t combine_hash(uint64_t a, uint64_t b) {
            constexpr uint64_t kMagic = 0x4906ba494954cb65ULL;
            return kMagic * (a + kMagic * b);
        }

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
            for (size_t i = 1; i < values.size(); i++) {
                if (!(values[i - 1] < values[i])) malformed(std::string(what) + " are not strictly increasing");
            }
            return values;
        }

        // The output transform of CatBoost's default predict() for the loss.
        uint8_t transform_of(std::string_view loss, uint32_t dim) {
            if (loss == "Logloss" || loss == "CrossEntropy" || loss == "MultiClassOneVsAll") return transform::Sigmoid;
            if (loss == "MultiClass") return dim > 1 ? transform::Softmax : transform::Identity;
            if (loss == "Poisson" || loss == "Tweedie") return transform::Exp;
            return transform::Identity;
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

    } // namespace

    namespace parsers {

        Model load_catboost_json(const std::string& filepath) {
            simdjson::dom::parser parser;
            simdjson::dom::element doc;
            if (auto error = parser.load(filepath).get(doc)) {
                throw std::runtime_error("Cannot parse CatBoost JSON " + filepath + ": " + simdjson::error_message(error));
            }
            try {
                auto forest = std::make_shared<ObliviousForest>();
                ObliviousForest& f = *forest;
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

                // Columns. A split's split_index counts float borders, then one-hot values, then CTR borders.
                constexpr size_t kMaxColumns = 1 << 24;
                std::vector<std::pair<uint16_t, uint16_t>> binary;  // split_index -> (source, threshold)
                std::vector<uint8_t> binary_equal;
                std::unordered_map<int64_t, uint32_t> float_source;  // float feature_index -> source
                std::vector<std::vector<float>> float_borders_by_source;
                uint32_t n_columns = 0;

                simdjson::dom::array float_features;
                if (features["float_features"].get_array().get(float_features) == simdjson::SUCCESS) {
                    for (simdjson::dom::object ff : float_features) {
                        const uint32_t column = get_index<uint32_t>(ff["flat_feature_index"], kMaxColumns, "flat_feature_index");
                        n_columns = std::max(n_columns, column + 1);
                        simdjson::dom::array borders;
                        if (ff["borders"].get_array().get(borders) != simdjson::SUCCESS || borders.size() == 0) continue;
                        ObliviousForest::Float source;
                        source.column = column;
                        source.borders = float_list(borders, "float feature borders");
                        if (source.borders.size() > 65534) malformed("too many borders");
                        std::string_view nan_treatment = "AsIs";
                        (void)ff["nan_value_treatment"].get_string().get(nan_treatment);
                        source.nan_as_true = nan_treatment == "AsTrue";
                        const uint32_t s = static_cast<uint32_t>(f.floats.size());
                        float_source[ff["feature_index"].get_int64()] = s;
                        for (size_t k = 0; k < source.borders.size(); k++) {
                            binary.emplace_back(static_cast<uint16_t>(s), static_cast<uint16_t>(k + 1));
                            binary_equal.push_back(0);
                        }
                        f.floats.push_back(std::move(source));
                    }
                }

                std::unordered_map<int64_t, uint32_t> cat_index;     // categorical feature_index -> cat_columns index
                std::unordered_map<int64_t, uint32_t> one_hot_source;  // categorical feature_index -> one-hot index
                simdjson::dom::array cat_features;
                const bool has_cats = features["categorical_features"].get_array().get(cat_features) == simdjson::SUCCESS;
                if (has_cats) {
                    for (simdjson::dom::object cf : cat_features) {
                        const uint32_t column = get_index<uint32_t>(cf["flat_feature_index"], kMaxColumns, "flat_feature_index");
                        n_columns = std::max(n_columns, column + 1);
                        const int64_t feature = cf["feature_index"].get_int64();
                        const uint32_t c = static_cast<uint32_t>(f.cat_columns.size());
                        cat_index[feature] = c;
                        f.cat_columns.push_back(column);
                        simdjson::dom::array values;
                        if (cf["values"].get_array().get(values) == simdjson::SUCCESS && values.size() > 0) {
                            ObliviousForest::OneHot one_hot;
                            one_hot.cat = c;
                            for (simdjson::dom::element v : values) one_hot.values.push_back(static_cast<uint32_t>(v.get_int64().value()));
                            if (one_hot.values.size() > 65534) malformed("too many one-hot values");
                            one_hot_source[feature] = static_cast<uint32_t>(f.one_hots.size());
                            f.one_hots.push_back(std::move(one_hot));
                        }
                    }
                }
                const uint32_t n_floats = static_cast<uint32_t>(f.floats.size());
                for (uint32_t h = 0; h < f.one_hots.size(); h++) {
                    for (size_t k = 0; k < f.one_hots[h].values.size(); k++) {
                        binary.emplace_back(static_cast<uint16_t>(n_floats + h), static_cast<uint16_t>(k + 1));
                        binary_equal.push_back(1);
                    }
                }

                // CTRs: projections shared by CTRs over the same combination, tables by identifier.
                simdjson::dom::object ctr_data;
                const bool has_ctr_data = doc["ctr_data"].get_object().get(ctr_data) == simdjson::SUCCESS;
                std::map<std::string, uint32_t> projection_ids, table_ids;
                std::vector<RawTable> raw_tables;
                simdjson::dom::array ctrs;
                if (features["ctrs"].get_array().get(ctrs) == simdjson::SUCCESS) {
                    const uint32_t first_ctr_source = n_floats + static_cast<uint32_t>(f.one_hots.size());
                    for (simdjson::dom::object cj : ctrs) {
                        ObliviousForest::Projection projection;
                        std::string projection_key;
                        for (simdjson::dom::object e : cj["elements"].get_array()) {
                            const std::string_view kind = e["combination_element"].get_string();
                            const int64_t feature = kind == "float_feature" ? e["float_feature_index"].get_int64() : e["cat_feature_index"].get_int64();
                            if (kind == "cat_feature_value") {
                                auto it = cat_index.find(feature);
                                if (it == cat_index.end()) malformed("CTR uses an unknown categorical feature");
                                projection.cats.push_back(it->second);
                            } else if (kind == "float_feature") {
                                auto it = float_source.find(feature);
                                if (it == float_source.end()) malformed("CTR uses an unknown float feature");
                                const auto& borders = f.floats[it->second].borders;
                                const float border = static_cast<float>(get_number(e["border"]));
                                auto pos = std::lower_bound(borders.begin(), borders.end(), border);
                                if (pos == borders.end() || *pos != border) malformed("CTR uses a float border the model does not have");
                                projection.floats.emplace_back(it->second, static_cast<uint16_t>(pos - borders.begin() + 1));
                            } else if (kind == "cat_feature_exact_value") {
                                auto it = one_hot_source.find(feature);
                                if (it == one_hot_source.end()) malformed("CTR uses a one-hot value the model does not have");
                                const auto& values = f.one_hots[it->second].values;
                                const uint32_t value = static_cast<uint32_t>(e["value"].get_int64().value());
                                auto pos = std::find(values.begin(), values.end(), value);
                                if (pos == values.end()) malformed("CTR uses a one-hot value the model does not have");
                                projection.one_hots.emplace_back(n_floats + it->second, static_cast<uint16_t>(pos - values.begin() + 1));
                            } else {
                                throw std::runtime_error("Unsupported CatBoost CTR element '" + std::string(kind) + "'.");
                            }
                            projection_key += std::string(kind) + ":" + std::to_string(feature) + ":" + std::string(simdjson::to_string(e)) + ";";
                        }
                        if (projection.cats.empty()) malformed("CTR without categorical features");
                        auto [pit, new_projection] = projection_ids.try_emplace(projection_key, static_cast<uint32_t>(f.projections.size()));
                        if (new_projection) f.projections.push_back(std::move(projection));

                        const std::string identifier(cj["identifier"].get_string().value());
                        auto [tit, new_table] = table_ids.try_emplace(identifier, static_cast<uint32_t>(raw_tables.size()));
                        if (new_table) {
                            simdjson::dom::object table;
                            if (!has_ctr_data || ctr_data[identifier].get_object().get(table) != simdjson::SUCCESS) {
                                malformed("ctr_data has no table for " + identifier);
                            }
                            raw_tables.push_back(read_table(table));
                            ObliviousForest::Table t;
                            t.projection = pit->second;
                            f.projections[pit->second].tables.push_back(static_cast<uint32_t>(f.tables.size()));
                            f.tables.push_back(std::move(t));
                        } else if (f.tables[tit->second].projection != pit->second) {
                            malformed("one CTR table for two combinations");
                        }

                        const CtrKind kind = ctr_kind(cj["ctr_type"].get_string());
                        CtrFormula calc;
                        calc.prior_num = static_cast<float>(get_number(cj["prior_numerator"]));
                        calc.prior_den = static_cast<float>(get_number(cj["prior_denomerator"]));
                        calc.shift = static_cast<float>(get_number(cj["shift"]));
                        calc.scale = static_cast<float>(get_number(cj["scale"]));
                        const RawTable& raw = raw_tables[tit->second];
                        const size_t target_border = get_index<size_t>(cj["target_border_idx"], std::max<size_t>(raw.width, 1), "target_border_idx");
                        const std::vector<float> borders = float_list(cj["borders"].get_array(), "CTR borders");
                        if (borders.size() > 65534) malformed("too many CTR borders");

                        ObliviousForest::Ctr ctr;
                        ctr.table = tit->second;
                        ctr.entry_buckets.resize(raw.hashes.size());
                        for (size_t entry = 0; entry < raw.hashes.size(); entry++) {
                            ctr.entry_buckets[entry] = bucket_of(borders, ctr_value(kind, calc, raw, static_cast<int64_t>(entry), target_border));
                        }
                        ctr.missing_bucket = bucket_of(borders, ctr_value(kind, calc, raw, -1, target_border));
                        const uint32_t s = first_ctr_source + static_cast<uint32_t>(f.ctrs.size());
                        if (s > 65535) malformed("too many features");
                        for (size_t k = 0; k < borders.size(); k++) {
                            binary.emplace_back(static_cast<uint16_t>(s), static_cast<uint16_t>(k + 1));
                            binary_equal.push_back(0);
                        }
                        f.ctrs.push_back(std::move(ctr));
                    }
                }

                // Open-addressing index of every table (load factor <= 1/2). CatBoost marks unused
                // buckets of its own hash view with 2^64 - 1; they are not entries.
                for (size_t t = 0; t < f.tables.size(); t++) {
                    ObliviousForest::Table& table = f.tables[t];
                    const RawTable& raw = raw_tables[t];
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

                // Output dimension and scale/bias.
                simdjson::dom::array scale_and_bias = doc["scale_and_bias"].get_array();
                if (scale_and_bias.size() != 2) malformed("scale_and_bias");
                f.scale = get_number(scale_and_bias.at(0));
                for (simdjson::dom::element b : scale_and_bias.at(1).get_array()) f.bias.push_back(get_number(b));
                if (f.bias.empty() || f.bias.size() > 65535) malformed("bias");
                f.dim = static_cast<uint32_t>(f.bias.size());

                // Trees.
                f.split_begin.push_back(0);
                for (simdjson::dom::object tree : trees) {
                    // A tree of depth 0 (a single leaf) has "splits": null.
                    simdjson::dom::array splits;
                    size_t depth = 0;
                    if (tree["splits"].get_array().get(splits) == simdjson::SUCCESS) {
                        depth = splits.size();
                        if (depth > 16) malformed("tree deeper than 16");
                        for (simdjson::dom::object split : splits) {
                            const size_t index = get_index<size_t>(split["split_index"], binary.size(), "split_index");
                            f.split_source.push_back(binary[index].first);
                            f.split_threshold.push_back(binary[index].second);
                            f.split_equal.push_back(binary_equal[index]);
                        }
                    } else if (!tree["splits"].is_null()) {
                        malformed("splits");
                    }
                    f.split_begin.push_back(static_cast<uint32_t>(f.split_source.size()));
                    f.leaf_begin.push_back(f.leaves.size());
                    simdjson::dom::array leaves = tree["leaf_values"].get_array();
                    if (leaves.size() != (size_t{1} << depth) * f.dim) malformed("leaf_values size does not match the tree depth");
                    for (simdjson::dom::element v : leaves) f.leaves.push_back(get_number(v));
                }
                f.n_columns = n_columns;

                std::string_view loss;
                simdjson::dom::element loss_type;
                if (doc.at_pointer("/model_info/params/loss_function/type").get(loss_type) == simdjson::SUCCESS) loss = loss_type.get_string();

                Model model;
                model.n_outputs = f.dim;
                model.transform = transform_of(loss, f.dim);
                model.n_features = n_columns;
                model.base_scores = f.bias.data();
                model.oblivious = forest;
                model.data = forest;
                return model;
            } catch (const simdjson::simdjson_error& e) {
                throw std::runtime_error("Malformed CatBoost model " + filepath + ": " + e.what());
            }
        }

    } // namespace parsers

    // ---------------------------------------------------------------------------------------------
    // Kernel

    namespace {

        constexpr size_t kBlock = 128;

        // Scratch of one thread: buckets of every source for a block (column-major, so the
        // per-tree loops over rows vectorise), category hashes, table entries, leaf indices.
        // Kept per thread and only grown, so a single row allocates and clears nothing. Arrays
        // are [source][row] with a stride of the block's row count.
        struct Scratch {
            std::vector<uint16_t> buckets;
            std::vector<uint32_t> hashes;
            std::vector<uint32_t> entries;
            std::vector<uint64_t> combination;
            std::vector<uint32_t> index;
            std::vector<double> sums;

            void prepare(const ObliviousForest& f, size_t rows) {
                auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
                grow(buckets, f.n_sources() * rows);
                grow(hashes, f.cat_columns.size() * rows);
                grow(entries, f.tables.size() * rows);
                grow(combination, rows);
                grow(index, rows);
                grow(sums, rows * f.dim);
            }
        };

        Scratch& thread_scratch() {
            thread_local Scratch scratch;
            return scratch;
        }

        void predict_block(const ObliviousForest& f, const double* X, size_t n_features, size_t rows, double* out, Scratch& s) {
            const size_t F = f.floats.size(), H = f.one_hots.size();
            const size_t stride = rows;
            s.prepare(f, rows);
            uint16_t* buckets = s.buckets.data();

            // Phase 1: buckets of every source.
            for (size_t k = 0; k < F; k++) {
                const ObliviousForest::Float& ff = f.floats[k];
                const uint16_t nan_bucket = ff.nan_as_true ? static_cast<uint16_t>(ff.borders.size()) : 0;
                uint16_t* b = buckets + k * stride;
                for (size_t r = 0; r < rows; r++) {
                    const float x = static_cast<float>(X[r * n_features + ff.column]);
                    b[r] = std::isnan(x) ? nan_bucket : bucket_of(ff.borders, x);
                }
            }
            for (size_t c = 0; c < f.cat_columns.size(); c++) {
                uint32_t* h = s.hashes.data() + c * stride;
                const uint32_t column = f.cat_columns[c];
                for (size_t r = 0; r < rows; r++) h[r] = static_cast<uint32_t>(X[r * n_features + column]);
            }
            for (size_t k = 0; k < H; k++) {
                const ObliviousForest::OneHot& oh = f.one_hots[k];
                const uint32_t* h = s.hashes.data() + oh.cat * stride;
                uint16_t* b = buckets + (F + k) * stride;
                for (size_t r = 0; r < rows; r++) {
                    uint16_t v = 0;
                    for (size_t j = 0; j < oh.values.size(); j++) v = oh.values[j] == h[r] ? static_cast<uint16_t>(j + 1) : v;
                    b[r] = v;
                }
            }
            // CTRs: one combination hash per projection, one lookup per table, a bucket per CTR.
            for (size_t p = 0; p < f.projections.size(); p++) {
                const ObliviousForest::Projection& proj = f.projections[p];
                uint64_t* hash = s.combination.data();
                std::fill(hash, hash + rows, uint64_t{0});
                for (uint32_t c : proj.cats) {
                    const uint32_t* h = s.hashes.data() + c * stride;
                    for (size_t r = 0; r < rows; r++) {
                        hash[r] = combine_hash(hash[r], static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(h[r]))));
                    }
                }
                for (const auto& [source, threshold] : proj.floats) {
                    const uint16_t* b = buckets + source * stride;
                    for (size_t r = 0; r < rows; r++) hash[r] = combine_hash(hash[r], b[r] >= threshold ? 1 : 0);
                }
                for (const auto& [source, value] : proj.one_hots) {
                    const uint16_t* b = buckets + source * stride;
                    for (size_t r = 0; r < rows; r++) hash[r] = combine_hash(hash[r], b[r] == value ? 1 : 0);
                }
                for (uint32_t t : proj.tables) {
                    const ObliviousForest::Table& table = f.tables[t];
                    uint32_t* entries = s.entries.data() + t * stride;
                    for (size_t r = 0; r < rows; r++) {
                        uint64_t slot = slot_of(hash[r], table.mask);
                        uint32_t found = 0;
                        while (table.slots[slot] != 0) {
                            if (table.keys[slot] == hash[r]) { found = table.slots[slot]; break; }
                            slot = (slot + 1) & table.mask;
                        }
                        entries[r] = found;
                    }
                }
            }
            for (size_t k = 0; k < f.ctrs.size(); k++) {
                const ObliviousForest::Ctr& ctr = f.ctrs[k];
                const uint32_t* entries = s.entries.data() + ctr.table * stride;
                uint16_t* b = buckets + (F + H + k) * stride;
                for (size_t r = 0; r < rows; r++) b[r] = entries[r] ? ctr.entry_buckets[entries[r] - 1] : ctr.missing_bucket;
            }

            // Phase 2: leaf index per tree from one bit per level, leaves added in model order.
            const size_t dim = f.dim;
            double* sums = s.sums.data();
            std::fill(sums, sums + rows * dim, 0.0);
            uint32_t* index = s.index.data();
            for (size_t t = 0; t < f.n_trees(); t++) {
                std::fill(index, index + rows, 0u);
                for (uint32_t k = f.split_begin[t]; k < f.split_begin[t + 1]; k++) {
                    const uint16_t* b = buckets + f.split_source[k] * stride;
                    const uint16_t threshold = f.split_threshold[k];
                    const uint32_t bit = k - f.split_begin[t];
                    if (f.split_equal[k]) {
                        for (size_t r = 0; r < rows; r++) index[r] |= static_cast<uint32_t>(b[r] == threshold) << bit;
                    } else {
                        for (size_t r = 0; r < rows; r++) index[r] |= static_cast<uint32_t>(b[r] >= threshold) << bit;
                    }
                }
                const double* leaves = f.leaves.data() + f.leaf_begin[t];
                if (dim == 1) {
                    for (size_t r = 0; r < rows; r++) sums[r] += leaves[index[r]];
                } else {
                    for (size_t r = 0; r < rows; r++) {
                        for (size_t d = 0; d < dim; d++) sums[r * dim + d] += leaves[index[r] * dim + d];
                    }
                }
            }
            for (size_t r = 0; r < rows; r++) {
                for (size_t d = 0; d < dim; d++) out[r * dim + d] = f.scale * sums[r * dim + d] + f.bias[d];
            }
        }

        // Categorical columns must hold category hashes; checked before any thread starts.
        void check_categories(const ObliviousForest& f, const double* X, size_t n_samples, size_t n_features) {
            for (uint32_t column : f.cat_columns) {
                for (size_t i = 0; i < n_samples; i++) {
                    const double v = X[i * n_features + column];
                    if (!(v >= 0.0 && v <= 4294967295.0 && v == std::floor(v))) {
                        throw std::invalid_argument("Column " + std::to_string(column) +
                                                    " is categorical: it must hold category hashes (lignum.catboost_hash), got " + std::to_string(v) + ".");
                    }
                }
            }
        }

    } // namespace

    void predict_oblivious(const ObliviousForest& f, const double* X, size_t n_samples, size_t n_features, double* out,
                           int n_threads, Strategy strategy) {
        check_categories(f, X, n_samples, n_features);
        const size_t n_blocks = (n_samples + kBlock - 1) / kBlock;
        const bool serial = strategy == Strategy::Serial || n_threads <= 1 || n_samples < 8;
        if (serial || n_blocks == 1) {
            Scratch& s = thread_scratch();
            for (size_t b = 0; b < n_blocks; b++) {
                const size_t i = b * kBlock;
                predict_block(f, X + i * n_features, n_features, std::min(kBlock, n_samples - i), out + i * f.dim, s);
            }
            return;
        }
        #pragma omp parallel num_threads(n_threads)
        {
            Scratch& s = thread_scratch();
            #pragma omp for schedule(dynamic)
            for (int64_t b = 0; b < static_cast<int64_t>(n_blocks); b++) {
                const size_t i = static_cast<size_t>(b) * kBlock;
                predict_block(f, X + i * n_features, n_features, std::min(kBlock, n_samples - i), out + i * f.dim, s);
            }
        }
    }

} // namespace lignum
