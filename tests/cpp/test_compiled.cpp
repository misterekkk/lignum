#include "test_util.hpp"
#include "compiled.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
    #include <sys/stat.h>
    #include <unistd.h>
#endif

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

#if !defined(_WIN32)
namespace {

    // Sets (or with nullptr unsets) an environment variable for the scope, then restores it.
    class EnvVar {
        public:
            EnvVar(const char* name, const char* value) : name_(name) {
                if (const char* old = std::getenv(name)) old_ = old, had_ = true;
                if (value) setenv(name, value, 1);
                else unsetenv(name);
            }
            ~EnvVar() {
                if (had_) setenv(name_, old_.c_str(), 1);
                else unsetenv(name_);
            }
        private:
            const char* name_;
            std::string old_;
            bool had_ = false;
    };

} // namespace

TEST(Compiled, FallbackCacheInTheTempFolderMustBePrivate) {
    // Without HOME or LIGNUM_CACHE_DIR the cache goes to the temporary folder, which others can
    // write: a library planted there under the expected name would be loaded and run.
    const auto tmp = std::filesystem::path(temp_path("private_tmp"));
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    const EnvVar no_cache("LIGNUM_CACHE_DIR", nullptr), no_home("HOME", nullptr), no_xdg("XDG_CACHE_HOME", nullptr);
    const EnvVar temp("TMPDIR", tmp.c_str());
    const auto dir = tmp / ("lignum-cache-" + std::to_string(geteuid()));
    ASSERT_EQ(std::filesystem::path(lignum::default_cache_dir()), dir);

    auto o = options();
    o.cache_dir.clear();  // the default cache
    const auto model = load_model("private_src.json");

    // Someone else's folder (here: a symbolic link to another folder) is not used.
    std::filesystem::create_directories(tmp / "elsewhere");
    std::filesystem::create_directory_symlink(tmp / "elsewhere", dir);
    EXPECT_THROW(lignum::compile_forest(model, o, nullptr), std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_empty(tmp / "elsewhere"));
    std::filesystem::remove(dir);

    // A folder of this user is made private and used.
    std::filesystem::create_directories(dir);
    std::filesystem::permissions(dir, std::filesystem::perms::all);
    lignum::CompileStats stats;
    ASSERT_NO_THROW(lignum::compile_forest(model, o, &stats));
    EXPECT_EQ(std::filesystem::path(stats.library).parent_path(), dir);
    struct stat st;
    ASSERT_EQ(stat(dir.c_str(), &st), 0);
    EXPECT_EQ(st.st_mode & 0777, 0700u);
    std::filesystem::remove_all(tmp);
}
#endif

TEST(Compiled, PathsWithSpacesAndShellMetacharactersDoNotExecuteShellCommands) {
    auto model = load_model("shell_src.json");
    const auto base = std::filesystem::path(temp_path("shell_meta $(id) test"));
    std::filesystem::remove_all(base);
    const auto marker = base / "pwned";
    const auto weird_dir = base / ("cache ; touch '" + marker.string() + "' $(touch '" + marker.string() + "')");
    auto o = options();
    o.cache_dir = weird_dir.string();
    model.compiled = lignum::compile_forest(model, o, nullptr);
    EXPECT_FALSE(std::filesystem::exists(marker));
    EXPECT_EQ(predict_raw(model, kRows, 2), predict_raw(load_model("shell_ref.json"), kRows, 2));

    o.cache_dir.clear();
    o.flags = "\"unclosed";
    EXPECT_THROW(lignum::compile_forest(model, o, nullptr), std::invalid_argument);
    model.compiled = nullptr;
    std::filesystem::remove_all(base);
}

TEST(Compiled, HandlesNaNLeavesAndZeroTreeModels) {
    const std::string nan_leaf = R"({"objective": "regression", "tree_info": [
        {"tree_structure": {"split_feature": 0, "threshold": 0.5, "decision_type": "<=", "default_left": true,
                            "missing_type": "NaN", "left_child": {"leaf_value": null}, "right_child": {"leaf_value": 2.0}}}]})";
    // LightGBM JSON leaf_value cannot be null directly, so patch the KNode/Model leaves via save/load or test 0-tree model:
    const std::string zero_trees = R"({"objective": "regression", "tree_info": []})";
    auto empty_model = lignum::load(write_temp("zero_trees.json", zero_trees), "lightgbm");
    empty_model.compiled = lignum::compile_forest(empty_model, cache_options("cache_zero_trees"), nullptr);
    std::vector<double> out(3);
    empty_model.predict_raw(kRows.data(), 3, 2, out.data(), 1);
    EXPECT_EQ(out, (std::vector<double>{0.0, 0.0, 0.0}));
    (void)nan_leaf;

    // Synthesize a 1-tree model with a NaN leaf to test NAN codegen literal.
    auto m = load_model("nan_leaf_src.json");
    auto bin_path = temp_path("nan_leaf.bin");
    m.save(bin_path);
    std::ifstream in(bin_path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    // Overwrite first leaf value with NaN in the .bin file.
    uint64_t n_cat_words = 0, n_nodes = 0, n_offsets = 0, n_leaves = 0, names_bytes = 0;
    uint32_t objective_bytes = 0, categories_bytes = 0;
    std::memcpy(&n_cat_words, bytes.data() + 16, 8);
    std::memcpy(&n_nodes, bytes.data() + 24, 8);
    std::memcpy(&n_offsets, bytes.data() + 32, 8);
    std::memcpy(&n_leaves, bytes.data() + 40, 8);
    std::memcpy(&names_bytes, bytes.data() + 64 + 24, 8);
    std::memcpy(&objective_bytes, bytes.data() + 64 + 40, 4);
    std::memcpy(&categories_bytes, bytes.data() + 64 + 44, 4);
    auto a64 = [](size_t v) { return (v + 63) & ~size_t{63}; };
    const size_t leaves_pos = 128 + a64(names_bytes + objective_bytes + categories_bytes) + a64(n_nodes * 64) + a64(n_offsets * 4);
    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    ASSERT_GT(n_leaves, 0u);
    std::memcpy(bytes.data() + leaves_pos, &nan_val, sizeof(nan_val));
    std::ofstream(bin_path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    auto nan_model = lignum::load(bin_path);
    const auto interp = predict_raw(nan_model, kRows, 2);
    nan_model.compiled = lignum::compile_forest(nan_model, cache_options("cache_nan_leaf"), nullptr);
    const auto comp = predict_raw(nan_model, kRows, 2);
    ASSERT_EQ(interp.size(), comp.size());
    for (size_t i = 0; i < interp.size(); i++) {
        if (std::isnan(interp[i])) EXPECT_TRUE(std::isnan(comp[i]));
        else EXPECT_EQ(interp[i], comp[i]);
    }
}

TEST(Compiled, ConcurrentPredictAndCompileDecompileIsSafe) {
    auto model = load_model("race_src.json");
    const auto lib = temp_path("race.lib_out");
    std::filesystem::remove(lib);
    const auto compiled = lignum::compile_forest(model, options(lib), nullptr);
    const auto expected = predict_raw(model, kRows, 2);

    std::atomic<bool> stop{false};
    std::thread mutator([&] {
        for (int i = 0; i < 2000; i++) {
            lignum::set_compiled(model, (i & 1) ? compiled : nullptr);
        }
        stop.store(true, std::memory_order_release);
    });
    std::vector<double> out(kRows.size() / 2);
    while (!stop.load(std::memory_order_acquire)) {
        model.predict_raw(kRows.data(), out.size(), 2, out.data(), 1);
        EXPECT_EQ(out, expected);
    }
    mutator.join();
    std::filesystem::remove(lib);
}
