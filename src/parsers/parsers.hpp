#pragma once

#include "ir.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace lignum {

    namespace parsers {

        // LightGBM: dump_model() JSON, or the text format of save_model() (model.txt).
        std::vector<std::unique_ptr<TempBinNode>> load_lightgbm_json(const std::string& filepath, ModelInfo& info);
        std::vector<std::unique_ptr<TempBinNode>> load_lightgbm_text(const std::string& filepath, ModelInfo& info);
        // XGBoost: save_model() as JSON or UBJSON (.ubj).
        std::vector<std::unique_ptr<TempBinNode>> load_xgboost_json(const std::string& filepath, ModelInfo& info);
        std::vector<std::unique_ptr<TempBinNode>> load_xgboost_ubj(const std::string& filepath, ModelInfo& info);

        // Shared by the LightGBM parsers. missing_type: 0 None, 1 Zero, 2 NaN (LightGBM's encoding).
        void lightgbm_numeric_split(TempBinNode& node, double threshold, bool default_left, int missing_type);
        void lightgbm_objective(std::string_view objective, ModelInfo& info);

        std::string ubjson_to_json(std::string_view data);
        std::string read_file(const std::string& filepath);

    } // namespace parsers

} // namespace lignum
