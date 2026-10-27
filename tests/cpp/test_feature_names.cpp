#include "test_util.hpp"
#include "validate.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using lignum::test::predict_raw;
using lignum::test::write_temp;

namespace {

    // One split on feature 1 of three named features.
    std::string lightgbm_json(const std::string& names) {
        return R"({"objective": "binary sigmoid:1", "feature_names": [)" + names + R"(], "tree_info": [
            {"tree_structure": {"split_feature": 1, "threshold": 0.25, "decision_type": "<=", "default_left": true,
                                "missing_type": "NaN", "left_child": {"leaf_value": -0.75}, "right_child": {"leaf_value": 0.5}}}]})";
    }

    std::string xgboost_json(const std::string& names) {
        return R"({"learner": {"feature_names": [)" + names + R"(],
            "learner_model_param": {"base_score": "[0E0]", "num_class": "0", "num_target": "1"},
            "objective": {"name": "reg:squarederror"},
            "gradient_booster": {"model": {"trees": [{
                "left_children": [1, -1, -1], "right_children": [2, -1, -1],
                "split_indices": [1, 0, 0], "split_conditions": [0.5, -1.0, 1.0], "default_left": [0, 0, 0]}]}}}})";
    }

    const char* kText =
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nobjective=binary sigmoid:1\n"
        "feature_names=age my_income debt\n\n"
        "Tree=0\nnum_leaves=2\nnum_cat=0\nsplit_feature=1\nthreshold=0.25\ndecision_type=2\n"
        "left_child=-1\nright_child=-2\nleaf_value=-0.75 0.5\nis_linear=0\n\nend of trees\n";

    std::string temp_path(const std::string& name) {
        return (std::filesystem::temp_directory_path() / ("lignum_test_" + name)).string();
    }

    using Names = std::vector<std::string>;

} // namespace

TEST(FeatureNames, ReadFromEveryFormat) {
    const Names expected = {"age", "my_income", "debt"};
    auto json = lignum::load(write_temp("names_lgb.json", lightgbm_json(R"("age", "my_income", "debt")")), "lightgbm");
    EXPECT_EQ(json.feature_names, expected);
    EXPECT_EQ(json.source, lignum::model_source::LightGBM);
    auto text = lignum::load(write_temp("names_lgb.txt", kText), "lightgbm");
    EXPECT_EQ(text.feature_names, expected);
    EXPECT_EQ(text.n_features, 2u);  // the trees use features 0 and 1; the names cover all three

    auto xgb = lignum::load(write_temp("names_xgb.json", xgboost_json(R"("a b", "zł/mies", "c")")), "xgboost");
    EXPECT_EQ(xgb.feature_names, (Names{"a b", "zł/mies", "c"}));
    EXPECT_EQ(xgb.source, lignum::model_source::XGBoost);
}

TEST(FeatureNames, DefaultsAndEmptyNamesMeanNone) {
    auto lgb = lignum::load(write_temp("names_default.json", lightgbm_json(R"("Column_0", "Column_1", "Column_2")")), "lightgbm");
    EXPECT_TRUE(lgb.feature_names.empty());
    auto xgb = lignum::load(write_temp("names_none.json", xgboost_json("")), "xgboost");
    EXPECT_TRUE(xgb.feature_names.empty());
    // Column_ names out of order are real names.
    auto swapped = lignum::load(write_temp("names_swapped.json", lightgbm_json(R"("Column_1", "Column_0", "Column_2")")), "lightgbm");
    EXPECT_EQ(swapped.feature_names.size(), 3u);

    EXPECT_TRUE(lignum::model_feature_names({"a", "", "c"}, 3, lignum::model_source::CatBoost).empty());
    EXPECT_THROW(lignum::model_feature_names({"a"}, 2, lignum::model_source::XGBoost), std::runtime_error);
}

TEST(FeatureNames, CheckAcceptsTheModelsOrderOnly) {
    auto model = lignum::load(write_temp("names_check.json", lightgbm_json(R"("age", "my_income", "debt")")), "lightgbm");
    EXPECT_NO_THROW(lignum::check_feature_names(model, {"age", "my_income", "debt"}));
    EXPECT_NO_THROW(lignum::check_feature_names(model, {"age", "my_income", "debt", "extra"}));  // extra columns last: ignored
    EXPECT_NO_THROW(lignum::check_feature_names(model, {"age", "my income", "debt"}));  // LightGBM stores spaces as '_'
    try {
        lignum::check_feature_names(model, {"my_income", "age", "debt"});
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("Column 0 is 'my_income' but the model expects 'age'"), std::string::npos) << e.what();
    }
    try {
        lignum::check_feature_names(model, {"age", "my_income"});
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("2 columns but the model has 3"), std::string::npos) << e.what();
    }
    EXPECT_THROW(lignum::check_feature_names(model, {"id", "age", "my_income", "debt"}), std::invalid_argument);

    // Only LightGBM folds spaces; other libraries keep names as they are.
    auto xgb = lignum::load(write_temp("names_check_xgb.json", xgboost_json(R"("a_b", "c", "d")")), "xgboost");
    EXPECT_THROW(lignum::check_feature_names(xgb, {"a b", "c", "d"}), std::invalid_argument);
    auto unnamed = lignum::load(write_temp("names_check_none.json", xgboost_json("")), "xgboost");
    EXPECT_NO_THROW(lignum::check_feature_names(unnamed, {"anything", "goes"}));
}

TEST(FeatureNames, BinaryFormatKeepsNamesSourceAndWriter) {
    for (bool binned : {true, false}) {
        auto model = lignum::load(write_temp("names_bin.json", xgboost_json(R"("a b", "zł/mies", "c")")), "xgboost", binned);
        const std::string path = temp_path("names.bin");
        model.save(path);
        auto loaded = lignum::load(path, "auto", binned);
        EXPECT_EQ(loaded.feature_names, model.feature_names);
        EXPECT_EQ(loaded.source, lignum::model_source::XGBoost);
        const std::vector<double> X = {0.0, 0.4, 0.0, 0.0, 0.6, 0.0};
        EXPECT_EQ(predict_raw(loaded, X, 3), predict_raw(model, X, 3));

        // Format 0.8, and the version of lignum that wrote it after the 64-byte header.
        std::ifstream in(path, std::ios::binary);
        const std::vector<char> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        ASSERT_GT(bytes.size(), 128u);
        EXPECT_EQ(bytes[6], 0);
        EXPECT_EQ(bytes[7], 8);
        const std::string writer(bytes.data() + 64);
        EXPECT_FALSE(writer.empty());
        EXPECT_NE(writer, "unknown");
    }
    // A model without names, too.
    auto unnamed = lignum::load(write_temp("names_bin_none.json", xgboost_json("")), "xgboost");
    unnamed.save(temp_path("names_none.bin"));
    EXPECT_TRUE(lignum::load(temp_path("names_none.bin"), "auto").feature_names.empty());
}

TEST(FeatureNames, CorruptedNamesAreRejected) {
    auto model = lignum::load(write_temp("names_corrupt.json", lightgbm_json(R"("age", "my_income", "debt")")), "lightgbm");
    const std::string path = temp_path("names_corrupt.bin");
    model.save(path);
    std::ifstream in(path, std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    // InfoHeader at 64: writer[24], names_bytes (u64) at 88, n_names (u32) at 96, source (u8) at 100;
    // the names start at 128 with the length of the first one.
    auto rejected = [&](size_t offset, const std::vector<unsigned char>& patch) {
        std::vector<char> mutated = bytes;
        std::memcpy(mutated.data() + offset, patch.data(), patch.size());
        const std::string bad = temp_path("names_corrupt_mut.bin");
        std::ofstream(bad, std::ios::binary).write(mutated.data(), static_cast<std::streamsize>(mutated.size()));
        try {
            lignum::load(bad, "auto");
            return false;
        } catch (const std::runtime_error&) {
            return true;
        }
    };
    EXPECT_TRUE(rejected(88, {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f}));  // names_bytes past the end
    EXPECT_TRUE(rejected(96, {4, 0, 0, 0}));                                       // more names than bytes hold
    EXPECT_TRUE(rejected(96, {1, 0, 0, 0}));                                       // fewer names than features
    EXPECT_TRUE(rejected(100, {9}));                                               // unknown source
    EXPECT_TRUE(rejected(128, {0xff, 0xff, 0, 0}));                                // a name longer than the section
    EXPECT_FALSE(rejected(132, {'A'}));                                            // a changed character is still a name
}
