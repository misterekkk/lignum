#pragma once

#include <cstdint>
#include <memory>

namespace lignum {

    // Library-neutral binary tree produced by the parsers.
    //
    // Parsers translate every library's split semantics into a single rule, so the
    // builder and the kernel never need to know where a model came from:
    //
    //     go right  <=>  isnan(x) ? default_right : (x >= threshold)
    //
    // with one exception that cannot be folded into a threshold: LightGBM's
    // missing_type=Zero, where |x| <= kZeroThreshold must also follow the default
    // direction (`zero_is_missing`).
    struct TempBinNode {
        bool is_leaf = false;
        double threshold_or_value = 0.0;
        int32_t feature_id = -1;
        int32_t default_dir = 0;        // 0 = left, 1 = right
        bool zero_is_missing = false;
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
        double base_score = 0.0;
        double sigmoid = 1.0;
        uint8_t transform = 2;
        Precision precision = Precision::Float64;
    };

} // namespace lignum
