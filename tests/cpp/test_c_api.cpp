#include "test_util.hpp"

#include <lignum/c_api.h>

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
