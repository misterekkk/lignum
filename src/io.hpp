#pragma once

#include <lignum/model.hpp>

#include <string>

namespace lignum {

    // binned = false skips the stored binned sections (Model::binned stays null).
    Model load_binary(const std::string& filepath, bool binned = true);

} // namespace lignum
