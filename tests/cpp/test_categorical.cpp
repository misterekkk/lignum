#include "test_util.hpp"

#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

using lignum::test::predict_raw;
using lignum::test::write_temp;

namespace {

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    constexpr double Inf = std::numeric_limits<double>::infinity();

    // Category in `set` -> 1.0 (left), otherwise -> 2.0 (right).
    lignum::Model categorical_stump(const std::string& name, const std::string& set) {
        const std::string json = R"({"objective": "regression", "tree_info": [{"tree_structure": {
            "split_feature": 0, "threshold": ")" + set + R"(", "decision_type": "==", "default_left": false, "missing_type": "None",
            "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}}]})";
        return lignum::load(write_temp(name, json), "lightgbm");
    }

    // LightGBM 4: NaN and x <= -1 go right; otherwise int(x) (truncation) is the category.
    double expected(double x, const std::vector<int64_t>& set) {
        if (std::isnan(x) || !(x > -1.0 && x < 2147483648.0)) return 2.0;
        const int64_t c = static_cast<int64_t>(x);
        for (int64_t s : set) if (s == c) return 1.0;
        return 2.0;
    }

    const std::vector<double> kProbes = {
        NaN, Inf, -Inf, -1.0, -0.999, -0.5, -1e-36, 0.0, 0.7, 1.0, 4.99, 5.0, 63.0, 63.5, 64.0,
        69.0, 70.0, 71.9, 72.0, 73.0, 99.0, 100.0, 899.0, 900.0, 900.5, 1023.0, 1024.0,
        2147483647.0, 2147483648.0, 4294967296.0, 1e12, -1e12,
    };

    void check(const std::string& name, const std::string& set_text, const std::vector<int64_t>& set) {
        auto model = categorical_stump(name, set_text);
        std::vector<double> want;
        for (double x : kProbes) want.push_back(expected(x, set));
        EXPECT_EQ(predict_raw(model, kProbes, 1), want) << set_text;
    }

} // namespace

TEST(Categorical, MaskEncoding) {
    check("cat_mask.json", "0||1||5||63", {0, 1, 5, 63});
}

TEST(Categorical, RangeEncoding) {
    check("cat_range.json", "70||71||72", {70, 71, 72});
    check("cat_range1.json", "900", {900});
}

TEST(Categorical, BitsetEncoding) {
    check("cat_bitset.json", "3||100||900||1023", {3, 100, 900, 1023});
}

TEST(Categorical, BinaryRoundTripKeepsBitsets) {
    auto model = categorical_stump("cat_io.json", "3||100||900||1023");
    auto bin = (std::filesystem::temp_directory_path() / "lignum_test_cat.bin").string();
    model.save(bin);
    auto restored = lignum::load(bin, "auto");
    EXPECT_EQ(predict_raw(restored, kProbes, 1), predict_raw(model, kProbes, 1));
}
