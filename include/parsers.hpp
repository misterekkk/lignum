#pragma once

#include "ir.hpp"

#include <string>
#include <vector>

namespace lignum {

    namespace parsers {

        std::vector<std::unique_ptr<TempBinNode>> load_lightgbm_json(const std::string& filepath);
        std::vector<std::unique_ptr<TempBinNode>> load_xgboost_json(const std::string& filepath, double& base_score);

    } // namespace parsers

} // namespace lignum

