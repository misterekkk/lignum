#pragma once

#include <lignum/model.hpp>

namespace lignum {

    // Checks every index stored in a model (KNode and binned form) so that a corrupted or malicious
    // model file cannot make prediction read out of bounds or loop forever. Children must point
    // strictly forward (the builder always lays trees out that way), which guarantees termination.
    // Throws std::runtime_error describing the first problem found.
    void validate_model(const Model& model);

} // namespace lignum
