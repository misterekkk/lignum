#include "test_util.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

using lignum::test::predict_raw;
using lignum::test::write_temp;

TEST(BinaryIO, RoundTripPreservesPredictions) {
    const std::string json = R"({"objective": "binary sigmoid:1", "tree_info": [
        {"tree_structure": {"split_feature": 1, "threshold": 0.25, "decision_type": "<=", "default_left": true,
                            "missing_type": "NaN", "left_child": {"leaf_value": -0.75}, "right_child": {"leaf_value": 0.5}}},
        {"tree_structure": {"leaf_value": 0.125}}]})";
    auto from_json = lignum::load(write_temp("io_src.json", json), "lightgbm");

    auto bin_path = (std::filesystem::temp_directory_path() / "lignum_test_io.bin").string();
    from_json.save(bin_path);
    auto from_bin = lignum::load(bin_path, "auto");

    std::vector<double> X = {0.0, 0.1, 0.0, 0.9, 0.0, 0.25, 0.0, 0.3, 0.0, -1.0};
    EXPECT_EQ(predict_raw(from_json, X, 2), predict_raw(from_bin, X, 2));

    std::vector<double> a(5), b(5);
    from_json.predict(X.data(), 5, 2, a.data(), 1);
    from_bin.predict(X.data(), 5, 2, b.data(), 1);
    EXPECT_EQ(a, b);
}

TEST(BinaryIO, RejectsNonLignumFile) {
    auto path = write_temp("io_garbage.bin", "definitely not a lignum model");
    EXPECT_THROW(lignum::load(path, "auto"), std::invalid_argument);
}
