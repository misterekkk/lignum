#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lignum {

    // Library-neutral binary tree produced by the parsers.
    //
    // Parsers translate every library's split semantics into a single rule, so the
    // builder and the kernel never need to know where a model came from:
    //
    //     go right  <=>  isnan(x) ? default_right : (x >= threshold)
    //
    // with two exceptions that cannot be folded into a threshold: LightGBM's
    // missing_type=Zero, where |x| <= kZeroThreshold must also follow the default
    // direction (`zero_is_missing`), and categorical splits (`categorical`).
    struct TempBinNode {
        bool is_leaf = false;
        double threshold_or_value = 0.0;
        int32_t feature_id = -1;
        int32_t default_dir = 0;        // 0 = left, 1 = right
        bool zero_is_missing = false;
        bool categorical = false;           // categorical split on `categories` (see cat_set_goes_right)
        bool cat_set_goes_right = false;    // false: LightGBM (in set -> left); true: XGBoost (in set -> right)
        std::vector<uint32_t> categories;
        std::unique_ptr<TempBinNode> left;
        std::unique_ptr<TempBinNode> right;
    };

    // Arithmetic the source library uses for accumulation and transforms.
    enum class Precision : uint8_t {
        Float64 = 0,  // LightGBM
        Float32 = 1,  // XGBoost
    };

    // Model-wide metadata filled in by the parsers.
    struct ModelInfo {
        std::vector<double> base_scores{0.0};  // one per output (class), in margin space
        std::vector<uint16_t> tree_outputs;     // output of every tree; empty when there is one output
        double sigmoid = 1.0;
        uint8_t transform = 2;
        Precision precision = Precision::Float64;
        std::vector<std::string> feature_names;  // as stored in the file; may be empty or defaults
        std::string objective;                   // the library's name for it ("binary", "reg:squarederror")
    };

} // namespace lignum
