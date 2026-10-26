// CatBoost JSON models: semantics on a hand-made model, CityHash, and corrupted files.
#include "test_util.hpp"
#include "catboost.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

using lignum::test::write_temp;

namespace {

    // CatBoost's CalcHash of the empty projection with one category: the key of a one-feature CTR table.
    std::string table_key(const std::string& category) {
        constexpr uint64_t kMagic = 0x4906ba494954cb65ULL;
        const auto hash = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(lignum::catboost_hash(category))));
        return std::to_string(kMagic * (0 + kMagic * hash));
    }

    double hash_of(const std::string& category) { return static_cast<double>(lignum::catboost_hash(category)); }

    // Column 0: float feature with borders 0.5, 1.5. Column 1: categorical, one-hot value "a" and a
    // Borders CTR learned from (class 0, class 1) counts: "a" (3, 7), "b" (5, 1).
    // Binary features by split_index: 0, 1 float borders; 2 one-hot "a"; 3, 4 CTR borders 0.3, 0.6.
    std::string model_json() {
        const int32_t a = static_cast<int32_t>(lignum::catboost_hash("a"));
        return R"({"ctr_data": {"ctr0": {"counter_denominator": 0, "hash_map": [")" + table_key("a") + R"(", 3, 7, ")" +
               table_key("b") + R"(", 5, 1, "18446744073709551615", 0, 0], "hash_stride": 3}},
          "features_info": {
            "categorical_features": [{"feature_id": "c", "feature_index": 0, "flat_feature_index": 1, "values": [)" + std::to_string(a) + R"(]}],
            "ctrs": [{"borders": [0.3, 0.6], "ctr_type": "Borders", "elements": [{"cat_feature_index": 0, "combination_element": "cat_feature_value"}],
                      "identifier": "ctr0", "prior_denomerator": 1, "prior_numerator": 0.5, "scale": 1, "shift": 0, "target_border_idx": 0}],
            "float_features": [{"borders": [0.5, 1.5], "feature_index": 0, "flat_feature_index": 0, "has_nans": true, "nan_value_treatment": "AsIs"}]},
          "model_info": {"params": {"loss_function": {"type": "Logloss"}}},
          "oblivious_trees": [
            {"leaf_values": [1, 2, 4, 8], "splits": [{"split_index": 1, "split_type": "FloatFeature"}, {"split_index": 4, "split_type": "OnlineCtr"}]},
            {"leaf_values": [16, 32], "splits": [{"split_index": 2, "split_type": "OneHotFeature"}]}],
          "scale_and_bias": [1, [0]]})";
    }

} // namespace

TEST(CatBoost, CityHashMatchesCatBoost) {
    // Hashes CatBoost stored for one-hot values of "u" and "v" in a trained model.
    EXPECT_EQ(lignum::catboost_hash("u"), 1981898051u);
    EXPECT_EQ(lignum::catboost_hash("v"), static_cast<uint32_t>(-1706589142));
}

TEST(CatBoost, HandMadeModel) {
    const auto model = lignum::load(write_temp("catboost.json", model_json()));
    ASSERT_TRUE(model.oblivious);
    EXPECT_EQ(model.n_features, 2u);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // x = 1.0, "a": float bucket 1, CTR (7 + 0.5) / (10 + 1) above both borders, one-hot hit -> 4 + 32.
    // x = 2.0, "b": float bucket 2, CTR 1.5 / 7 below both borders, one-hot miss -> 2 + 16.
    // x = NaN, unseen: bucket 0, CTR of a missing combination 0.5 / 1 (bucket 1) -> 1 + 16.
    // x = 0.5 on a border is not above it: like x = 0.4 -> 4 + 32 for "a".
    // x = +inf is above both borders (and not above the +inf padding of the search): like 2.0.
    // x = -inf is below both borders, like NaN in the Min mode.
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<double> X = {1.0, hash_of("a"), 2.0, hash_of("b"), nan, hash_of("zz"), 0.5, hash_of("a"),
                                   inf, hash_of("b"), -inf, hash_of("zz")};
    EXPECT_EQ(lignum::test::predict_raw(model, X, 2), (std::vector<double>{36, 18, 17, 36, 18, 17}));

    std::vector<double> proba(6);
    model.predict(X.data(), 6, 2, proba.data(), 1);
    EXPECT_EQ(proba[1], 1.0 / (1.0 + std::exp(-18.0)));

    // Categorical columns must hold hashes.
    const std::vector<double> bad = {1.0, 0.5};
    std::vector<double> out(1);
    EXPECT_THROW(model.predict_raw(bad.data(), 1, 2, out.data(), 1), std::invalid_argument);
}

TEST(CatBoost, CorruptedFilesAreRejectedOrSafe) {
    const std::string original = model_json();
    const auto path = (std::filesystem::temp_directory_path() / "lignum_fuzz_catboost.json").string();
    const std::vector<double> X = {1.0, hash_of("a"), 2.0, hash_of("b"), std::nan(""), hash_of("zz"), 0.5, 7.0};
    std::mt19937 rng(4242);
    int loaded = 0;
    for (int iteration = 0; iteration < 3000; iteration++) {
        std::string mutated = original;
        const int flips = 1 + static_cast<int>(rng() % 6);
        for (int k = 0; k < flips; k++) {
            const size_t pos = rng() % mutated.size();
            mutated[pos] = (rng() % 2) ? "0123456789 -.,:[]{}\""[rng() % 21] : static_cast<char>(rng());
        }
        if (rng() % 10 == 0) mutated.resize(rng() % mutated.size());
        std::ofstream(path, std::ios::binary).write(mutated.data(), static_cast<std::streamsize>(mutated.size()));
        try {
            const auto model = lignum::load(path, "catboost");
            if (model.n_features > 2) continue;
            std::vector<double> out(4 * model.output_width(false));
            model.predict(X.data(), 4, 2, out.data(), 1);
            loaded++;
        } catch (const std::exception&) {
            // Rejected: fine.
        }
    }
    EXPECT_GT(loaded, 0);
}

namespace {

    const unsigned char kSmallCbm[] = {
        #include "catboost_small_cbm.inc"
    };

    std::string small_cbm() { return std::string(reinterpret_cast<const char*>(kSmallCbm), sizeof(kSmallCbm)); }

} // namespace

// The fixture was written with (CatBoost 1.2.10, numpy rng seed 0):
//   n = 600; df = DataFrame({"x0": normal(n), "c0": choice(["a","b","c","d"], n), "c1": choice(["u","v"], n)})
//   y = (x0 > 0) ^ c0 in {a, b}
//   CatBoostClassifier(iterations=6, depth=3, random_seed=0, cat_features=["c0","c1"],
//                      max_ctr_complexity=2, border_count=8).fit(df, y).save_model("small.cbm")
// and the expected values are CatBoost's predict(..., prediction_type="RawFormulaVal").
TEST(CatBoost, CbmModel) {
    const auto model = lignum::load(write_temp("catboost_small.cbm", small_cbm()));
    ASSERT_TRUE(model.oblivious);
    EXPECT_EQ(model.n_features, 3u);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> X = {0.5, hash_of("a"), hash_of("u"), -1.2, hash_of("b"), hash_of("v"),
                                   2.0, hash_of("zz"), hash_of("u"), nan, hash_of("d"), hash_of("v")};
    EXPECT_EQ(lignum::test::predict_raw(model, X, 3),
              (std::vector<double>{-0x1.60f24937c833cp-1, 0x1.fdeff1ed31a48p-1, 0x1.88198b1cbd9bcp-1, -0x1.5351ebab1a63cp+0}));
}

// The prefix tree of the projections: every projection once, under the node whose path from the
// root is its category list, and shared prefixes hashed once. Loaded from .cbm and from .bin.
TEST(CatBoost, ProjectionPrefixTree) {
    const auto cbm = lignum::load(write_temp("catboost_small3.cbm", small_cbm()));
    const auto saved = (std::filesystem::temp_directory_path() / "lignum_catboost_prefix.bin").string();
    cbm.save(saved);
    const auto bin = lignum::load(saved);
    for (const auto* model : {&cbm, &bin}) {
        const lignum::ObliviousForest& f = *model->oblivious;
        ASSERT_EQ(f.prefix_projection_begin.size(), f.prefix_cat.size() + 1);
        std::vector<int> seen(f.projections.size(), 0);
        std::vector<uint32_t> path;
        size_t category_hashes = 0;
        for (size_t n = 0; n < f.prefix_cat.size(); n++) {
            ASSERT_LE(f.prefix_level[n], path.size());  // a child of the latest node one level up
            path.resize(f.prefix_level[n]);
            path.push_back(f.prefix_cat[n]);
            for (uint32_t i = f.prefix_projection_begin[n]; i < f.prefix_projection_begin[n + 1]; i++) {
                const uint32_t p = f.prefix_projections[i];
                EXPECT_EQ(f.projections[p].cats, path);
                seen[p]++;
            }
        }
        for (int count : seen) EXPECT_EQ(count, 1);
        for (const auto& p : f.projections) category_hashes += p.cats.size();
        EXPECT_GT(f.projections.size(), 2u);
        EXPECT_LT(f.prefix_cat.size(), category_hashes);  // max_ctr_complexity=2: [c0], [c0, c1] share c0
    }
}

TEST(CatBoost, CorruptedCbmFilesAreRejectedOrSafe) {
    const std::string original = small_cbm();
    const auto path = (std::filesystem::temp_directory_path() / "lignum_fuzz_catboost.cbm").string();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> X = {0.5, hash_of("a"), hash_of("u"), -1.2, hash_of("b"), hash_of("v"),
                                   2.0, hash_of("zz"), hash_of("u"), nan, hash_of("d"), hash_of("v")};
    std::mt19937 rng(9001);
    int loaded = 0;
    for (int iteration = 0; iteration < 3000; iteration++) {
        std::string mutated = original;
        const int flips = 1 + static_cast<int>(rng() % 8);
        for (int k = 0; k < flips; k++) {
            const size_t pos = rng() % mutated.size();
            mutated[pos] = (rng() % 3 == 0) ? static_cast<char>(mutated[pos] ^ (1 << (rng() % 8))) : static_cast<char>(rng());
        }
        if (rng() % 10 == 0) mutated.resize(rng() % mutated.size());
        std::ofstream(path, std::ios::binary).write(mutated.data(), static_cast<std::streamsize>(mutated.size()));
        try {
            const auto model = lignum::load(path, "catboost");
            if (model.n_features > 3) continue;
            std::vector<double> out(4 * model.output_width(false));
            model.predict(X.data(), 4, 3, out.data(), 1);
            loaded++;
        } catch (const std::exception&) {
            // Rejected: fine.
        }
    }
    EXPECT_GT(loaded, 0);
}

TEST(CatBoost, BinaryFormatRoundTripAndCorruption) {
    const auto model = lignum::load(write_temp("catboost_small2.cbm", small_cbm()));
    const auto saved = (std::filesystem::temp_directory_path() / "lignum_catboost_small.bin").string();
    model.save(saved);
    const auto loaded = lignum::load(saved);
    ASSERT_TRUE(loaded.oblivious);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> X = {0.5, hash_of("a"), hash_of("u"), -1.2, hash_of("b"), hash_of("v"),
                                   2.0, hash_of("zz"), hash_of("u"), nan, hash_of("d"), hash_of("v")};
    EXPECT_EQ(lignum::test::predict_raw(loaded, X, 3), lignum::test::predict_raw(model, X, 3));

    std::ifstream in(saved, std::ios::binary);
    const std::string original((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto path = (std::filesystem::temp_directory_path() / "lignum_fuzz_catboost.bin").string();
    std::mt19937 rng(31337);
    int ok = 0;
    for (int iteration = 0; iteration < 3000; iteration++) {
        std::string mutated = original;
        const int flips = 1 + static_cast<int>(rng() % 8);
        for (int k = 0; k < flips; k++) {
            // Mostly the forest arrays (past header and bias), where the indices live.
            const size_t pos = (rng() % 4 == 0) ? rng() % mutated.size() : 128 + rng() % (mutated.size() - 128);
            mutated[pos] = static_cast<char>(rng());
        }
        if (rng() % 10 == 0) mutated.resize(rng() % mutated.size());
        std::ofstream(path, std::ios::binary).write(mutated.data(), static_cast<std::streamsize>(mutated.size()));
        try {
            const auto m = lignum::load(path);
            if (m.n_features > 3) continue;
            std::vector<double> out(4 * m.output_width(false));
            m.predict(X.data(), 4, 3, out.data(), 1);
            ok++;
        } catch (const std::exception&) {
            // Rejected: fine.
        }
    }
    EXPECT_GT(ok, 0);
}
