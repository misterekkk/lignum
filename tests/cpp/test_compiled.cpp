#include "test_util.hpp"
#include "compiled.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using lignum::test::predict_raw;
using lignum::test::write_temp;

namespace {

    // Two trees; `leaf` and `threshold` vary the model for the fingerprint tests.
    std::string model_json(double leaf, double threshold) {
        return R"({"objective": "binary sigmoid:1", "tree_info": [
            {"tree_structure": {"split_feature": 1, "threshold": )" + std::to_string(threshold) + R"(, "decision_type": "<=",
                                "default_left": true, "missing_type": "NaN",
                                "left_child": {"split_feature": 0, "threshold": -0.5, "decision_type": "<=", "default_left": false,
                                               "missing_type": "NaN", "left_child": {"leaf_value": -0.75},
                                               "right_child": {"leaf_value": )" + std::to_string(leaf) + R"(}},
                                "right_child": {"leaf_value": 0.5}}},
            {"tree_structure": {"split_feature": 0, "threshold": 1.5, "decision_type": "<=", "default_left": false,
                                "missing_type": "NaN", "left_child": {"leaf_value": 0.125}, "right_child": {"leaf_value": -0.25}}}]})";
    }

    lignum::Model load_model(const std::string& name, double leaf = 0.375, double threshold = 0.25) {
        return lignum::load(write_temp(name, model_json(leaf, threshold)), "lightgbm");
    }

    std::string temp_path(const std::string& name) {
        return (std::filesystem::temp_directory_path() / ("lignum_test_" + name)).string();
    }

    lignum::CompileOptions options(const std::string& output = "") {
        lignum::CompileOptions o;
        o.work_dir = temp_path("compiled_cache");
        o.output = output;
        o.jobs = 2;
        o.block_trees = 1;  // two blocks for two trees
        return o;
    }

    const std::vector<double> kRows = {
        0.0, 0.1,   -1.0, 0.25,   NAN, 0.3,   2.0, NAN,   -0.5, -1.0,   1.5, 0.26,
    };

} // namespace

TEST(Compiled, FingerprintIgnoresTheFileFormat) {
    auto from_json = load_model("fp_src.json");
    const std::string bin = temp_path("fp.bin");
    from_json.save(bin);
    auto from_bin = lignum::load(bin, "auto");
    EXPECT_EQ(lignum::compiled_fingerprint(from_json), lignum::compiled_fingerprint(from_bin));
}

TEST(Compiled, FingerprintChangesWithLeavesAndThresholds) {
    const auto base = lignum::compiled_fingerprint(load_model("fp_a.json"));
    EXPECT_NE(base, lignum::compiled_fingerprint(load_model("fp_b.json", 0.376)));
    EXPECT_NE(base, lignum::compiled_fingerprint(load_model("fp_c.json", 0.375, 0.3)));
}

TEST(Compiled, ShippedLibraryGivesTheInterpretersPredictions) {
    auto model = load_model("ship_src.json");
    const std::string bin = temp_path("ship.bin"), library = temp_path("ship.lib_out");
    model.save(bin);
    std::filesystem::remove(library);
    lignum::CompileStats stats;
    model.compiled = lignum::compile_forest(model, options(library), &stats);
    EXPECT_EQ(stats.library, library);
    ASSERT_TRUE(std::filesystem::exists(library));

    auto interpreter = lignum::load(bin, "auto");
    auto production = lignum::load(bin, "auto");
    production.compiled = lignum::load_compiled(production, library);
    EXPECT_EQ(production.compiled->blocks.size(), 2u);
    EXPECT_EQ(predict_raw(production, kRows, 2), predict_raw(interpreter, kRows, 2));
    EXPECT_EQ(predict_raw(model, kRows, 2), predict_raw(interpreter, kRows, 2));
}

TEST(Compiled, RejectsALibraryOfAnotherModel) {
    auto model = load_model("other_src.json");
    const std::string library = temp_path("other.lib_out");
    model.compiled = lignum::compile_forest(model, options(library), nullptr);

    auto different = load_model("other_dst.json", 0.376);
    EXPECT_THROW(lignum::load_compiled(different, library), std::runtime_error);
}

TEST(Compiled, ReportsMissingLibraryAndCompiler) {
    auto model = load_model("missing_src.json");
    EXPECT_THROW(lignum::load_compiled(model, temp_path("does_not_exist.lib_out")), std::runtime_error);

    auto o = options();
    o.compiler = "lignum-no-such-compiler";
    o.work_dir = temp_path("compiled_cache_missing_cc");
    std::filesystem::remove_all(o.work_dir);
    try {
        lignum::compile_forest(model, o, nullptr);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("not found"), std::string::npos) << e.what();
    }
}
