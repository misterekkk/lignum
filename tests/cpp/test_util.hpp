#pragma once

#include <lignum/lignum.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace lignum::test {

    // Writes `content` to a fresh file in the system temp directory and returns its path. Binary
    // mode: on Windows text mode would turn every 0x0A byte of a binary model into \r\n.
    inline std::string write_temp(const std::string& name, const std::string& content) {
        auto path = std::filesystem::temp_directory_path() / ("lignum_test_" + name);
        std::ofstream(path, std::ios::binary) << content;
        return path.string();
    }

    inline std::vector<double> predict_raw(const Model& model, const std::vector<double>& X, size_t n_features) {
        std::vector<double> out(X.size() / n_features);
        model.predict_raw(X.data(), out.size(), n_features, out.data(), 1);
        return out;
    }

} // namespace lignum::test
