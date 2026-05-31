#pragma once

#include <string>

#include "model.hpp"

namespace lignum {

    Model load(const std::string& filepath, const std::string& format = "auto");

} // namespace lignum