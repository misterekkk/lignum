#include "test_util.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

using lignum::test::predict_raw;
using lignum::test::write_temp;

namespace {

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    // Single split on feature 0: x <= threshold -> 1.0, otherwise -> 2.0.
    std::string stump(const std::string& threshold, const std::string& missing_type, bool default_left) {
        return R"({"objective": "regression", "tree_info": [{"tree_structure": {
            "split_feature": 0, "threshold": )" + threshold + R"(, "decision_type": "<=",
            "default_left": )" + (default_left ? "true" : "false") + R"(, "missing_type": ")" + missing_type + R"(",
            "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}}]})";
    }

    lignum::Model load_stump(const std::string& name, const std::string& threshold, const std::string& missing_type, bool default_left) {
        return lignum::load(write_temp(name, stump(threshold, missing_type, default_left)), "lightgbm");
    }

} // namespace

TEST(LightGBM, ThresholdIsInclusiveOnTheLeft) {
    auto model = load_stump("lgb_incl.json", "0.5", "NaN", false);
    auto out = predict_raw(model, {0.5, std::nextafter(0.5, 1.0), -3.0, 7.0, 0.5}, 1);
    EXPECT_EQ(out, (std::vector<double>{1.0, 2.0, 1.0, 2.0, 1.0}));
}

TEST(LightGBM, MissingTypeNaNFollowsDefaultDirection) {
    auto right = load_stump("lgb_nan_r.json", "0.5", "NaN", false);
    auto left = load_stump("lgb_nan_l.json", "0.5", "NaN", true);
    EXPECT_EQ(predict_raw(right, {NaN}, 1)[0], 2.0);
    EXPECT_EQ(predict_raw(left, {NaN}, 1)[0], 1.0);
}

TEST(LightGBM, MissingTypeNoneTreatsNaNAsZero) {
    // NaN becomes 0.0 and is compared normally; default_left is irrelevant.
    auto model = load_stump("lgb_none.json", "0.5", "None", false);
    EXPECT_EQ(predict_raw(model, {NaN}, 1)[0], 1.0);
    auto neg = load_stump("lgb_none_neg.json", "-0.5", "None", true);
    EXPECT_EQ(predict_raw(neg, {NaN}, 1)[0], 2.0);
}

TEST(LightGBM, MissingTypeZeroTreatsZeroAndNaNAsMissing) {
    // Normally 0.0 <= 1.0 goes left; as "missing" it must follow default_left = false.
    auto model = load_stump("lgb_zero.json", "1.0", "Zero", false);
    auto out = predict_raw(model, {0.0, NaN, 0.3, 1.5, -0.0}, 1);
    EXPECT_EQ(out, (std::vector<double>{2.0, 2.0, 1.0, 2.0, 2.0}));
}

TEST(LightGBM, TinyInputsAreZeroedLikeLightGBM) {
    // LightGBM drops |x| <= kZeroThreshold (1e-35f) from dense input, so they behave as 0.0.
    const double kZero = static_cast<double>(1e-35f);
    auto model = load_stump("lgb_kzero.json", "-1.0000000180025095e-35", "None", true);
    auto out = predict_raw(model, {-kZero, kZero, -1e-30, 0.0}, 1);
    EXPECT_EQ(out, (std::vector<double>{2.0, 2.0, 1.0, 2.0}));
}

TEST(LightGBM, DeepTreeUsesBothSlotsOfEveryKNode) {
    // Depth-3 tree over features 0 and 1; leaves numbered in-order 1..8.
    const std::string json = R"({"objective": "regression", "tree_info": [{"tree_structure":
      {"split_feature": 0, "threshold": 0.0, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
       "left_child":
        {"split_feature": 1, "threshold": 0.0, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
         "left_child":  {"split_feature": 0, "threshold": -1.0, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
                         "left_child": {"leaf_value": 1}, "right_child": {"leaf_value": 2}},
         "right_child": {"split_feature": 0, "threshold": -2.0, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
                         "left_child": {"leaf_value": 3}, "right_child": {"leaf_value": 4}}},
       "right_child":
        {"split_feature": 1, "threshold": 1.0, "decision_type": "<=", "default_left": false, "missing_type": "NaN",
         "left_child":  {"split_feature": 0, "threshold": 1.0, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
                         "left_child": {"leaf_value": 5}, "right_child": {"leaf_value": 6}},
         "right_child": {"split_feature": 0, "threshold": 2.0, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
                         "left_child": {"leaf_value": 7}, "right_child": {"leaf_value": 8}}}}}]})";
    auto model = lignum::load(write_temp("lgb_deep.json", json), "lightgbm");

    // Nine rows so that both the 4-row interleaved path and the tail path run.
    std::vector<double> X = {
        -1.5, -1.0,   -0.5, -1.0,   -3.0, 0.5,   -1.0, 0.5,
         0.5,  0.0,    1.5,  1.0,    1.5,  2.0,   3.0, 2.0,
         0.5,  NaN,
    };
    auto out = predict_raw(model, X, 2);
    EXPECT_EQ(out, (std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 7}));
}

TEST(LightGBM, SumsTreesInModelOrder) {
    const std::string json = R"({"objective": "regression", "tree_info": [
        {"tree_structure": {"leaf_value": 0.1}},
        {"tree_structure": {"leaf_value": 0.2}},
        {"tree_structure": {"leaf_value": 0.3}}]})";
    auto model = lignum::load(write_temp("lgb_order.json", json), "lightgbm");
    EXPECT_EQ(predict_raw(model, {0.0}, 1)[0], (0.0 + 0.1) + 0.2 + 0.3);
}

TEST(LightGBM, AllStrategiesAgree) {
    const std::string json = R"({"objective": "binary sigmoid:1", "tree_info": [
        {"tree_structure": {"split_feature": 0, "threshold": 0.0, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
          "left_child": {"split_feature": 1, "threshold": 0.5, "decision_type": "<=", "default_left": false, "missing_type": "NaN",
                         "left_child": {"leaf_value": -0.3}, "right_child": {"leaf_value": 0.1}},
          "right_child": {"leaf_value": 0.7}}},
        {"tree_structure": {"split_feature": 1, "threshold": -0.25, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
          "left_child": {"leaf_value": 0.05}, "right_child": {"leaf_value": -0.15}}}]})";
    auto model = lignum::load(write_temp("lgb_strategies.json", json), "lightgbm");

    std::vector<double> X;
    for (int i = 0; i < 37; i++) {
        X.push_back(std::sin(i * 0.7));
        X.push_back(i % 5 == 0 ? NaN : std::cos(i * 1.3));
    }
    const size_t n = X.size() / 2;

    std::vector<double> expected(n);
    model.predict(X.data(), n, 2, expected.data(), 1, lignum::Strategy::Serial);
    for (auto strategy : {lignum::Strategy::Auto, lignum::Strategy::Rows}) {
        for (int n_jobs : {1, 2, 4, -1}) {
            std::vector<double> got(n);
            model.predict(X.data(), n, 2, got.data(), n_jobs, strategy);
            EXPECT_EQ(got, expected);
        }
    }
}

TEST(LightGBM, MulticlassSoftmaxPerRow) {
    // Two iterations of three classes: tree t belongs to class t % 3.
    const std::string json = R"({"objective": "multiclass num_class:3", "num_class": 3, "num_tree_per_iteration": 3, "tree_info": [
        {"tree_structure": {"split_feature": 0, "threshold": 0.0, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
                            "left_child": {"leaf_value": 0.5}, "right_child": {"leaf_value": -0.5}}},
        {"tree_structure": {"leaf_value": 0.25}},
        {"tree_structure": {"leaf_value": -0.25}},
        {"tree_structure": {"leaf_value": 0.125}},
        {"tree_structure": {"leaf_value": 0.0625}},
        {"tree_structure": {"split_feature": 0, "threshold": 1.0, "decision_type": "<=", "default_left": false, "missing_type": "NaN",
                            "left_child": {"leaf_value": 0.0}, "right_child": {"leaf_value": 1.0}}}]})";
    auto model = lignum::load(write_temp("lgb_multi.json", json), "lightgbm");
    ASSERT_EQ(model.n_outputs, 3u);
    ASSERT_EQ(model.output_width(true), 3u);

    const std::vector<double> X = {-1.0, 2.0, NaN};
    std::vector<double> raw(9), prob(9);
    model.predict_raw(X.data(), 3, 1, raw.data(), 1);
    model.predict(X.data(), 3, 1, prob.data(), 1);

    const std::vector<double> expected_raw = {
        0.5 + 0.125, 0.25 + 0.0625, -0.25 + 0.0,
        -0.5 + 0.125, 0.25 + 0.0625, -0.25 + 1.0,
        0.5 + 0.125, 0.25 + 0.0625, -0.25 + 1.0,
    };
    EXPECT_EQ(raw, expected_raw);
    for (size_t i = 0; i < 3; i++) {
        const double* r = &expected_raw[i * 3];
        const double m = std::max({r[0], r[1], r[2]});
        const double e0 = std::exp(r[0] - m), e1 = std::exp(r[1] - m), e2 = std::exp(r[2] - m), sum = e0 + e1 + e2;
        EXPECT_EQ(prob[i * 3 + 0], e0 / sum);
        EXPECT_EQ(prob[i * 3 + 1], e1 / sum);
        EXPECT_EQ(prob[i * 3 + 2], e2 / sum);
    }
}
