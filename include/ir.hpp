#pragma once

#include <memory>

namespace lignum {

    struct TempBinNode {
        bool is_leaf;
        double threshold_or_value;
        int32_t feature_id;
        int32_t default_dir;
        std::unique_ptr<TempBinNode> left;
        std::unique_ptr<TempBinNode> right;

        TempBinNode() : is_leaf(false), threshold_or_value(0.0), feature_id(-1), 
                        default_dir(0), left(nullptr), right(nullptr) {}
    };

} // namespace lignum

