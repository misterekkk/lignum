#include "test_util.hpp"
#include "compiled.hpp"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
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
        o.cache_dir = temp_path("compiled_cache");
        o.library = output;
        o.jobs = 2;
        o.block_trees = 1;  // two blocks for two trees
        return o;
    }

    // A fresh, empty cache folder.
    lignum::CompileOptions cache_options(const std::string& name) {
        auto o = options();
        o.cache_dir = temp_path(name);
        std::filesystem::remove_all(o.cache_dir);
        return o;
    }

    std::vector<std::string> files_in(const std::string& dir) {
        std::vector<std::string> names;
        if (!std::filesystem::exists(dir)) return names;
        for (const auto& entry : std::filesystem::directory_iterator(dir)) names.push_back(entry.path().filename().string());
        return names;
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

    // Below compiled_min_rows the interpreter runs; the results do not change.
    production.compiled_min_rows = 4;
    EXPECT_EQ(predict_raw(production, kRows, 2), predict_raw(interpreter, kRows, 2));
    const std::vector<double> three(kRows.begin(), kRows.begin() + 6);
    EXPECT_EQ(predict_raw(production, three, 2), predict_raw(interpreter, three, 2));
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
    o.cache_dir = temp_path("compiled_cache_missing_cc");
    std::filesystem::remove_all(o.cache_dir);
    try {
        lignum::compile_forest(model, o, nullptr);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("not found"), std::string::npos) << e.what();
    }
}

TEST(Compiled, CacheHoldsOnlyTheLibraryAndReusesIt) {
    auto model = load_model("cache_src.json");
    auto interpreter = load_model("cache_ref.json");
    const auto o = cache_options("cache_reuse");
    lignum::CompileStats first, second;
    model.compiled = lignum::compile_forest(model, o, &first);
    EXPECT_FALSE(first.cached);
    EXPECT_FALSE(first.compiler.empty());
    const auto names = files_in(o.cache_dir);
    ASSERT_EQ(names.size(), 1u);  // no sources, objects or logs left
    EXPECT_EQ(std::filesystem::path(first.library).filename().string(), names[0]);
    EXPECT_NE(names[0].find(lignum::compiled_platform()), std::string::npos);

    auto again = load_model("cache_src2.json");
    again.compiled = lignum::compile_forest(again, o, &second);
    EXPECT_TRUE(second.cached);
    EXPECT_TRUE(second.compiler.empty());
    EXPECT_EQ(second.library, first.library);
    EXPECT_EQ(predict_raw(again, kRows, 2), predict_raw(interpreter, kRows, 2));

    // Other flags give another library.
    auto flags = o;
    flags.flags = "-O1";  // cl takes it too
    lignum::CompileStats third;
    again.compiled = lignum::compile_forest(again, flags, &third);
    EXPECT_FALSE(third.cached);
    EXPECT_NE(third.library, first.library);
}

TEST(Compiled, LibraryInAFolderIsFoundByLoadCompiled) {
    auto model = load_model("folder_src.json");
    auto o = cache_options("folder_cache");
    const std::string folder = temp_path("folder_libs");
    std::filesystem::remove_all(folder);
    o.library = folder + "/";
    lignum::CompileStats stats;
    model.compiled = lignum::compile_forest(model, o, &stats);
    EXPECT_EQ(std::filesystem::path(stats.library), std::filesystem::path(folder) / lignum::compiled_library_name(model));
    EXPECT_FALSE(std::filesystem::exists(o.cache_dir));  // an explicit library bypasses the cache
    EXPECT_EQ(files_in(folder).size(), 1u);

    auto production = load_model("folder_dst.json");
    production.compiled = lignum::load_compiled(production, folder);
    EXPECT_EQ(predict_raw(production, kRows, 2), predict_raw(load_model("folder_ref.json"), kRows, 2));

    // The same folder, now existing, without the separator; another model is not found in it.
    o.library = folder;
    lignum::CompileStats again;
    model.compiled = nullptr;
    production.compiled = nullptr;
    model.compiled = lignum::compile_forest(model, o, &again);
    EXPECT_FALSE(again.cached);
    EXPECT_EQ(again.library, stats.library);
    auto other = load_model("folder_other.json", 0.376);
    try {
        lignum::load_compiled(other, folder);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(lignum::compiled_library_name(other)), std::string::npos) << e.what();
    }
}

TEST(Compiled, ConcurrentCompilesShareOneLibrary) {
    const auto model = load_model("concurrent_src.json");
    const auto o = cache_options("cache_concurrent");
    std::vector<std::shared_ptr<const lignum::CompiledForest>> results(4);
    std::vector<std::string> errors(results.size());
    std::vector<std::thread> threads;
    for (size_t k = 0; k < results.size(); k++) {
        threads.emplace_back([&, k] {
            try {
                results[k] = lignum::compile_forest(model, o, nullptr);
            } catch (const std::exception& e) {
                errors[k] = e.what();
            }
        });
    }
    for (auto& t : threads) t.join();
    for (size_t k = 0; k < results.size(); k++) {
        EXPECT_TRUE(errors[k].empty()) << errors[k];
        EXPECT_NE(results[k], nullptr);
    }
    EXPECT_EQ(files_in(o.cache_dir).size(), 1u);
    auto attached = load_model("concurrent_dst.json");
    attached.compiled = results[0];
    EXPECT_EQ(predict_raw(attached, kRows, 2), predict_raw(load_model("concurrent_ref.json"), kRows, 2));
}

TEST(Compiled, DamagedCacheEntryIsRebuiltAndClearRemovesIt) {
    auto model = load_model("damaged_src.json");
    const auto o = cache_options("cache_damaged");
    lignum::CompileStats first, second;
    model.compiled = lignum::compile_forest(model, o, &first);
    model.compiled = nullptr;  // unload, so the file can be replaced (Windows)
    std::ofstream(first.library, std::ios::binary | std::ios::trunc) << "not a library";

    model.compiled = lignum::compile_forest(model, o, &second);
    EXPECT_FALSE(second.cached);
    EXPECT_EQ(second.library, first.library);
    EXPECT_EQ(predict_raw(model, kRows, 2), predict_raw(load_model("damaged_ref.json"), kRows, 2));

    model.compiled = nullptr;
    EXPECT_EQ(lignum::clear_compiled_cache(o.cache_dir), 1u);
    EXPECT_TRUE(files_in(o.cache_dir).empty());
    EXPECT_EQ(lignum::clear_compiled_cache(temp_path("cache_does_not_exist")), 0u);
}

TEST(Compiled, FailedBuildKeepsOnlyTheLog) {
#if defined(_WIN32)
    GTEST_SKIP() << "the failing flag is for gcc / clang";
#else
    auto model = load_model("failed_src.json");
    auto o = cache_options("cache_failed");
    o.flags = "-include lignum_no_such_header.h";
    try {
        lignum::compile_forest(model, o, nullptr);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("lignum_no_such_header.h"), std::string::npos) << message;  // the compiler's output
        const auto at = message.find("(log: ");
        ASSERT_NE(at, std::string::npos) << message;
        const std::string log = message.substr(at + 6, message.find(')', at) - at - 6);
        EXPECT_TRUE(std::filesystem::exists(log)) << log;
        EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(log).replace_extension()));  // build folder removed
        std::filesystem::remove(log);
    }
    EXPECT_TRUE(files_in(o.cache_dir).empty());
#endif
}
