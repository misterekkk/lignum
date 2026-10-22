// Corrupted model files must be rejected at load time or predict safely. Meaningful under
// AddressSanitizer (cmake -DLIGNUM_SANITIZE=ON), which turns any out-of-bounds access into a failure.
#include "test_util.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using lignum::test::write_temp;

namespace {

    // Two trees with numeric, zero-as-missing and categorical (mask and bitset) splits.
    const char* kModel = R"({"objective": "binary sigmoid:1", "tree_info": [
      {"tree_structure": {"split_feature": 0, "threshold": "1||5||100||900", "decision_type": "==", "default_left": false, "missing_type": "None",
        "left_child": {"split_feature": 1, "threshold": 0.5, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
                       "left_child": {"split_feature": 2, "threshold": 1.0, "decision_type": "<=", "default_left": false, "missing_type": "Zero",
                                      "left_child": {"leaf_value": 0.1}, "right_child": {"leaf_value": 0.2}},
                       "right_child": {"leaf_value": 0.3}},
        "right_child": {"split_feature": 3, "threshold": "2||3", "decision_type": "==", "default_left": false, "missing_type": "None",
                        "left_child": {"leaf_value": -0.1}, "right_child": {"split_feature": 1, "threshold": -0.25, "decision_type": "<=",
                        "default_left": false, "missing_type": "NaN", "left_child": {"leaf_value": -0.2}, "right_child": {"leaf_value": -0.3}}}}},
      {"tree_structure": {"split_feature": 2, "threshold": 0.0, "decision_type": "<=", "default_left": true, "missing_type": "NaN",
        "left_child": {"leaf_value": 0.05}, "right_child": {"leaf_value": -0.05}}}]})";

    std::vector<char> read_file(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

} // namespace

TEST(Fuzz, CorruptedBinaryFilesAreRejectedOrSafe) {
    const auto dir = std::filesystem::temp_directory_path();
    const auto original = (dir / "lignum_fuzz_original.bin").string();
    lignum::load(write_temp("fuzz.json", kModel), "lightgbm").save(original);
    const std::vector<char> bytes = read_file(original);
    ASSERT_GT(bytes.size(), 64u);

    std::vector<double> X;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> value(-3.0, 1000.0);
    for (int i = 0; i < 16 * 4; i++) X.push_back(i % 7 == 0 ? std::numeric_limits<double>::quiet_NaN() : std::floor(value(rng)));

    const auto corrupted = (dir / "lignum_fuzz_corrupted.bin").string();
    int loaded = 0;
    for (int iteration = 0; iteration < 3000; iteration++) {
        std::vector<char> mutated = bytes;
        const int flips = 1 + static_cast<int>(rng() % 8);
        for (int k = 0; k < flips; k++) {
            // Mostly past the header (where indices live), sometimes the header too.
            const size_t pos = (rng() % 4 == 0) ? rng() % mutated.size() : 64 + rng() % (mutated.size() - 64);
            mutated[pos] = static_cast<char>(rng());
        }
        if (rng() % 16 == 0) mutated.resize(rng() % mutated.size());  // truncation
        std::ofstream(corrupted, std::ios::binary).write(mutated.data(), static_cast<std::streamsize>(mutated.size()));

        for (bool binned : {true, false}) {
            try {
                auto model = lignum::load(corrupted, "auto", binned);
                loaded++;
                if (model.n_features > 4) continue;  // a corrupted feature count: the input would be rejected anyway
                std::vector<double> out(16 * model.output_width(false));
                model.predict(X.data(), 16, 4, out.data(), 1);
                std::vector<double> one(model.output_width(false));
                model.predict(X.data(), 1, 4, one.data(), 1);
            } catch (const std::exception&) {
                // Rejected: fine.
            }
        }
    }
    // Some mutations hit padding or values (not indices), so loading still succeeds sometimes.
    EXPECT_GT(loaded, 0);
}

TEST(Fuzz, XGBoostTreeCyclesAreRejected) {
    const std::string json = R"({"learner": {
        "learner_model_param": {"base_score": "[5E-1]", "num_class": "0", "num_target": "1"},
        "objective": {"name": "reg:squarederror"},
        "gradient_booster": {"model": {"trees": [{
            "left_children": [1, 0, -1], "right_children": [2, 2, -1],
            "split_indices": [0, 0, 0], "split_conditions": [0.5, 0.1, 1.0],
            "default_left": [0, 0, 0]}]}}}})";
    EXPECT_THROW(lignum::load(write_temp("xgb_cycle.json", json), "xgboost"), std::runtime_error);
}

namespace {

    // LightGBM model.txt: a categorical split (categories 1 and 5) above a numeric one.
    const char* kTextModel =
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nobjective=binary sigmoid:1\n\n"
        "Tree=0\nnum_leaves=3\nnum_cat=1\nsplit_feature=0 1\nthreshold=0 0.5\ndecision_type=1 10\n"
        "left_child=1 -1\nright_child=-3 -2\nleaf_value=0.1 0.2 0.3\ncat_boundaries=0 1\ncat_threshold=34\nis_linear=0\n\n"
        "end of trees\n";

    // A small XGBoost model in UBJSON, written marker by marker.
    std::string ubj_model() {
        std::string out;
        auto key = [&](const std::string& k) { out += 'U'; out += static_cast<char>(k.size()); out += k; };
        auto str = [&](const std::string& v) { out += 'S'; key(v); };
        auto ints = [&](std::initializer_list<int> v) {
            out += "[$l#U";
            out += static_cast<char>(v.size());
            for (int x : v) for (int b = 3; b >= 0; b--) out += static_cast<char>((static_cast<uint32_t>(x) >> (8 * b)) & 0xFF);
        };
        auto floats = [&](std::initializer_list<float> v) {
            out += "[$d#U";
            out += static_cast<char>(v.size());
            for (float f : v) {
                uint32_t bits;
                std::memcpy(&bits, &f, 4);
                for (int b = 3; b >= 0; b--) out += static_cast<char>((bits >> (8 * b)) & 0xFF);
            }
        };
        out += '{'; key("learner"); out += '{';
        key("learner_model_param"); out += '{'; key("base_score"); str("[5E-1]"); key("num_class"); str("0"); key("num_target"); str("1"); out += '}';
        key("objective"); out += '{'; key("name"); str("reg:squarederror"); out += '}';
        key("gradient_booster"); out += '{'; key("model"); out += '{'; key("trees"); out += '['; out += '{';
        key("left_children"); ints({1, -1, -1}); key("right_children"); ints({2, -1, -1});
        key("split_indices"); ints({0, 0, 0}); key("split_conditions"); floats({0.5f, 0.1f, 1.0f});
        key("default_left"); ints({0, 0, 0});
        out += "}]}}}}";
        return out;
    }

    void fuzz_file(const std::string& name, const std::string& original, const std::string& format, int iterations) {
        const auto path = (std::filesystem::temp_directory_path() / ("lignum_fuzz_" + name)).string();
        std::vector<double> X(16 * 4, 1.0);
        for (size_t i = 0; i < X.size(); i += 5) X[i] = std::numeric_limits<double>::quiet_NaN();
        std::mt19937 rng(777);
        for (int iteration = 0; iteration < iterations; iteration++) {
            std::string mutated = original;
            const int flips = 1 + static_cast<int>(rng() % 6);
            for (int k = 0; k < flips; k++) {
                const size_t pos = rng() % mutated.size();
                // Text: mostly digits and separators, which keep the structure plausible.
                mutated[pos] = (rng() % 2) ? "0123456789 -=.\n"[rng() % 15] : static_cast<char>(rng());
            }
            if (rng() % 10 == 0) mutated.resize(rng() % mutated.size());
            if (iteration % 50 == 0) {  // pure noise behind the right first bytes
                mutated = original.substr(0, 2);
                for (int i = 0; i < 64; i++) mutated += static_cast<char>(rng());
            }
            std::ofstream(path, std::ios::binary).write(mutated.data(), static_cast<std::streamsize>(mutated.size()));
            for (bool binned : {true, false}) {
                try {
                    auto model = lignum::load(path, format, binned);
                    if (model.n_features > 4) continue;
                    std::vector<double> out(16 * model.output_width(false));
                    model.predict(X.data(), 16, 4, out.data(), 1);
                } catch (const std::exception&) {
                    // Rejected: fine.
                }
            }
        }
    }

} // namespace

TEST(Fuzz, LightGBMTextFilesAreRejectedOrSafe) {
    auto model = lignum::load(write_temp("fuzz_model.txt", kTextModel), "lightgbm");
    const std::vector<double> X = {1.0, 0.25, 5.0, 0.75, 2.0, 0.25, 1.0, NAN};
    std::vector<double> out(4);
    model.predict_raw(X.data(), 4, 2, out.data(), 1);
    EXPECT_EQ(out, (std::vector<double>{0.1, 0.2, 0.3, 0.1}));  // cats 1, 5 left; NaN goes left (default_left)
    fuzz_file("model.txt", kTextModel, "lightgbm", 2000);
}

TEST(Fuzz, UbjsonFilesAreRejectedOrSafe) {
    const std::string ubj = ubj_model();
    auto model = lignum::load(write_temp("fuzz_model.ubj", ubj), "auto");
    const std::vector<double> X = {0.25, 0.75, NAN};
    std::vector<double> out(3);
    model.predict_raw(X.data(), 3, 1, out.data(), 1);
    // XGBoost sums in float32: base_score 0.5 plus the leaf; NaN goes right (default_left = 0).
    EXPECT_EQ(out, (std::vector<double>{static_cast<float>(0.5f + 0.1f), 1.5, 1.5}));
    fuzz_file("model.ubj", ubj, "auto", 2000);
}

TEST(Fuzz, LightGBMTextCyclesAreRejected) {
    // Node 1's left child points back to node 0: must fail, not recurse until the stack overflows.
    const std::string text =
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nobjective=regression\n\n"
        "Tree=0\nnum_leaves=3\nnum_cat=0\nsplit_feature=0 1\nthreshold=0.5 0.5\ndecision_type=2 2\n"
        "left_child=1 0\nright_child=-1 -2\nleaf_value=1 2 3\nis_linear=0\n\nend of trees\n";
    EXPECT_THROW(lignum::load(write_temp("cycle_model.txt", text), "lightgbm"), std::runtime_error);
}
