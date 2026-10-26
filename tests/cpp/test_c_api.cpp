#include "test_util.hpp"

#include <lignum/c_api.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

using lignum::test::write_temp;

namespace {

    const char* kStump = R"({"objective": "regression", "tree_info": [{"tree_structure": {
        "split_feature": 1, "threshold": 0.5, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
        "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}}]})";

} // namespace

TEST(CApi, LoadPredictSaveFree) {
    LignumModelHandle model = nullptr;
    ASSERT_EQ(LignumModelLoad(write_temp("capi.json", kStump).c_str(), "lightgbm", &model), 0) << LignumGetLastError();

    size_t n_features = 0;
    ASSERT_EQ(LignumModelNumFeatures(model, &n_features), 0);
    EXPECT_EQ(n_features, 2u);

    const std::vector<double> X = {0.0, 0.1, 0.0, 0.9, 0.0, 0.5};
    std::vector<double> out(3);
    for (int strategy : {LIGNUM_STRATEGY_AUTO, LIGNUM_STRATEGY_SERIAL, LIGNUM_STRATEGY_ROWS}) {
        ASSERT_EQ(LignumModelPredict(model, X.data(), 3, 2, 1, -1, strategy, out.data()), 0) << LignumGetLastError();
        EXPECT_EQ(out, (std::vector<double>{1.0, 2.0, 1.0}));
    }

    auto bin = (std::filesystem::temp_directory_path() / "lignum_test_capi.bin").string();
    ASSERT_EQ(LignumModelSave(model, bin.c_str()), 0) << LignumGetLastError();
    LignumModelHandle restored = nullptr;
    ASSERT_EQ(LignumModelLoad(bin.c_str(), "auto", &restored), 0) << LignumGetLastError();
    std::vector<double> out2(3);
    ASSERT_EQ(LignumModelPredict(restored, X.data(), 3, 2, 1, 1, LIGNUM_STRATEGY_AUTO, out2.data()), 0);
    EXPECT_EQ(out2, out);

    EXPECT_EQ(LignumModelFree(restored), 0);
    EXPECT_EQ(LignumModelFree(model), 0);
}

TEST(CApi, CatBoostHash) {
    // Same values as CatBoost.CityHashMatchesCatBoost; the length counts, not a terminator.
    EXPECT_EQ(LignumCatBoostHash("u", 1), 1981898051u);
    EXPECT_EQ(LignumCatBoostHash("vw", 1), static_cast<unsigned int>(-1706589142));
    EXPECT_EQ(LignumCatBoostHash(nullptr, 0), LignumCatBoostHash("", 0));
}

TEST(CApi, ReportsErrors) {
    LignumModelHandle model = nullptr;
    EXPECT_EQ(LignumModelLoad("/nonexistent/model.json", "lightgbm", &model), -1);
    EXPECT_NE(std::string(LignumGetLastError()), "");

    ASSERT_EQ(LignumModelLoad(write_temp("capi_err.json", kStump).c_str(), "lightgbm", &model), 0);
    double x[2] = {0.0, 0.0}, out = 0.0;
    EXPECT_EQ(LignumModelPredict(model, x, 1, 1, 0, 1, LIGNUM_STRATEGY_AUTO, &out), -1);  // needs 2 features
    EXPECT_NE(std::string(LignumGetLastError()).find("requires at least 2"), std::string::npos);
    EXPECT_EQ(LignumModelPredict(model, x, 1, 2, 0, 1, 42, &out), -1);
    EXPECT_EQ(LignumModelPredict(nullptr, x, 1, 2, 0, 1, LIGNUM_STRATEGY_AUTO, &out), -1);
    EXPECT_EQ(LignumModelFree(model), 0);
    EXPECT_EQ(LignumModelFree(nullptr), 0);
}

TEST(CApi, CompiledModeAndShippedLibrary) {
    LignumModelHandle model = nullptr;
    ASSERT_EQ(LignumModelLoad(write_temp("capi_cc.json", kStump).c_str(), "lightgbm", &model), 0) << LignumGetLastError();
    const std::vector<double> X = {0.0, 0.1, 0.0, 0.9, 0.0, 0.5, 0.0, NAN};
    std::vector<double> expected(4), out(4);
    ASSERT_EQ(LignumModelPredict(model, X.data(), 4, 2, 1, 1, LIGNUM_STRATEGY_AUTO, expected.data()), 0);

    const auto dir = std::filesystem::temp_directory_path();
    const std::string library = (dir / "lignum_test_capi_lib").string();
    ASSERT_EQ(LignumModelCompile(model, nullptr, nullptr, (dir / "lignum_test_capi_cache").string().c_str(), library.c_str()), 0)
        << LignumGetLastError();
    int compiled = 0;
    ASSERT_EQ(LignumModelUsesCompiled(model, &compiled), 0);
    EXPECT_EQ(compiled, 1);
    ASSERT_EQ(LignumModelPredict(model, X.data(), 4, 2, 1, 1, LIGNUM_STRATEGY_AUTO, out.data()), 0);
    EXPECT_EQ(out, expected);

    LignumModelHandle production = nullptr;
    ASSERT_EQ(LignumModelLoad(write_temp("capi_cc2.json", kStump).c_str(), "lightgbm", &production), 0);
    ASSERT_EQ(LignumModelLoadCompiled(production, library.c_str()), 0) << LignumGetLastError();
    std::fill(out.begin(), out.end(), 0.0);
    ASSERT_EQ(LignumModelPredict(production, X.data(), 4, 2, 1, 1, LIGNUM_STRATEGY_AUTO, out.data()), 0);
    EXPECT_EQ(out, expected);

    ASSERT_EQ(LignumModelDecompile(production), 0);
    ASSERT_EQ(LignumModelUsesCompiled(production, &compiled), 0);
    EXPECT_EQ(compiled, 0);
    EXPECT_EQ(LignumModelLoadCompiled(production, "/nonexistent/lib"), -1);
    EXPECT_NE(std::string(LignumGetLastError()).find("not found"), std::string::npos);
    EXPECT_EQ(LignumModelCompile(nullptr, nullptr, nullptr, nullptr, nullptr), -1);
    EXPECT_EQ(LignumModelFree(production), 0);
    EXPECT_EQ(LignumModelFree(model), 0);
}
