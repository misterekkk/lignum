#pragma once

// A hand-made CatBoost JSON model shared by the CatBoost and C API tests.
#include "catboost.hpp"

#include <cstdint>
#include <string>

namespace lignum::test {

    // CatBoost's CalcHash of the empty projection with one category: the key of a one-feature CTR table.
    inline std::string catboost_table_key(const std::string& category) {
        constexpr uint64_t kMagic = 0x4906ba494954cb65ULL;
        const auto hash = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(lignum::catboost_hash(category))));
        return std::to_string(kMagic * (0 + kMagic * hash));
    }

    // Column 0: float feature with borders 0.5, 1.5. Column 1: categorical, one-hot value "a" and a
    // Borders CTR learned from (class 0, class 1) counts: "a" (3, 7), "b" (5, 1).
    // Binary features by split_index: 0, 1 float borders; 2 one-hot "a"; 3, 4 CTR borders 0.3, 0.6.
    inline std::string catboost_model_json() {
        const int32_t a = static_cast<int32_t>(lignum::catboost_hash("a"));
        return R"({"ctr_data": {"ctr0": {"counter_denominator": 0, "hash_map": [")" + catboost_table_key("a") + R"(", 3, 7, ")" +
               catboost_table_key("b") + R"(", 5, 1, "18446744073709551615", 0, 0], "hash_stride": 3}},
          "features_info": {
            "categorical_features": [{"feature_id": "c", "feature_index": 0, "flat_feature_index": 1, "values": [)" + std::to_string(a) + R"(]}],
            "ctrs": [{"borders": [0.3, 0.6], "ctr_type": "Borders", "elements": [{"cat_feature_index": 0, "combination_element": "cat_feature_value"}],
                      "identifier": "ctr0", "prior_denomerator": 1, "prior_numerator": 0.5, "scale": 1, "shift": 0, "target_border_idx": 0}],
            "float_features": [{"borders": [0.5, 1.5], "feature_index": 0, "flat_feature_index": 0, "has_nans": true, "nan_value_treatment": "AsIs"}]},
          "model_info": {"params": {"loss_function": {"type": "Logloss"}}},
          "oblivious_trees": [
            {"leaf_values": [1, 2, 4, 8], "splits": [{"split_index": 1, "split_type": "FloatFeature"}, {"split_index": 4, "split_type": "OnlineCtr"}]},
            {"leaf_values": [16, 32], "splits": [{"split_index": 2, "split_type": "OneHotFeature"}]}],
          "scale_and_bias": [1, [0]]})";
    }

} // namespace lignum::test
