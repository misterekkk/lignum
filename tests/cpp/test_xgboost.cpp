#include "test_util.hpp"

#include <cmath>
#include <limits>
#include <string>

using lignum::test::predict_raw;
using lignum::test::write_temp;

namespace {

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    // Single split on feature 0: float(x) < threshold -> -1.0, otherwise -> 1.0.
    std::string stump(const std::string& threshold, int default_left, const std::string& base_score = "[0E0]") {
        return R"({"learner": {
            "learner_model_param": {"base_score": ")" + base_score + R"(", "num_class": "0", "num_target": "1"},
            "objective": {"name": "reg:squarederror"},
            "gradient_booster": {"model": {"trees": [{
                "left_children": [1, -1, -1], "right_children": [2, -1, -1],
                "split_indices": [0, 0, 0], "split_conditions": [)" + threshold + R"(, -1.0, 1.0],
                "default_left": [)" + std::to_string(default_left) + R"(, 0, 0]}]}}}})";
    }

} // namespace

TEST(XGBoost, ComparesInputsAsFloat32) {
    // XGBoost stores 0.1 as float32 and casts x to float32 before comparing, so every
    // double that rounds to 0.1f is "not less than" the threshold.
    auto model = lignum::load(write_temp("xgb_f32.json", stump("0.1", 0)), "xgboost");
    const double thr = static_cast<double>(0.1f);
    const double prev = static_cast<double>(std::nextafter(0.1f, 0.0f));
    const double mid = (prev + thr) / 2;  // exact in double; rounds to the even neighbour in float
    std::vector<double> X = {
        0.1, 0.0999999985, 0.0999999, thr, 0.2,
        mid, std::nextafter(mid, 0.0), std::nextafter(mid, 1.0), prev,
    };
    std::vector<double> expected;
    for (double x : X) expected.push_back(static_cast<float>(x) < 0.1f ? -1.0 : 1.0);
    EXPECT_EQ(predict_raw(model, X, 1), expected);
}

TEST(XGBoost, MissingFollowsDefaultLeft) {
    auto left = lignum::load(write_temp("xgb_nan_l.json", stump("0.5", 1)), "xgboost");
    auto right = lignum::load(write_temp("xgb_nan_r.json", stump("0.5", 0)), "xgboost");
    EXPECT_EQ(predict_raw(left, {NaN}, 1)[0], -1.0);
    EXPECT_EQ(predict_raw(right, {NaN}, 1)[0], 1.0);
}

TEST(XGBoost, AddsBaseScoreForSquaredError) {
    auto model = lignum::load(write_temp("xgb_base.json", stump("0.5", 0, "[5E-1]")), "xgboost");
    EXPECT_EQ(predict_raw(model, {0.0, 1.0}, 1), (std::vector<double>{-0.5, 1.5}));
}
