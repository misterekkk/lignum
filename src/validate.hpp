#pragma once

#include <lignum/model.hpp>

namespace lignum {

    // Checks every index stored in a model (KNode and binned form) so that a corrupted or malicious
    // model file cannot make prediction read out of bounds or loop forever. Children must point
    // strictly forward (the builder always lays trees out that way), which guarantees termination.
    // Throws std::runtime_error describing the first problem found.
    void validate_model(const Model& model);

    // The feature names a model keeps: the stored ones, or none when they are absent, partly empty
    // (CatBoost on a matrix) or LightGBM's defaults (Column_0, Column_1, ...). Throws
    // std::runtime_error when there are fewer names than the trees use features.
    std::vector<std::string> model_feature_names(std::vector<std::string> names, uint32_t n_features, uint8_t source);

    // Throws std::invalid_argument when named input columns (Arrow schema) do not start with the
    // model's feature names; LightGBM models compare names with spaces replaced by '_', as LightGBM
    // stores them. No check when the model has no names.
    void check_feature_names(const Model& model, const std::vector<std::string>& columns);

} // namespace lignum
