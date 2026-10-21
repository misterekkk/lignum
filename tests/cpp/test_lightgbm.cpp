#include "test_util.hpp"

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
