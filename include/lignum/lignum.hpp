#pragma once

#include <lignum/model.hpp>

#include <string>

namespace lignum {

    Model load(const std::string& filepath, const std::string& format = "auto");

} // namespace lignum