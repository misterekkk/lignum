// Corrupted model files must be rejected at load time or predict safely. Meaningful under
// AddressSanitizer (cmake -DLIGNUM_SANITIZE=ON), which turns any out-of-bounds access into a failure.
#include "test_util.hpp"

#include <cmath>
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
