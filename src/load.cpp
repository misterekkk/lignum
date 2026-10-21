#include <lignum/lignum.hpp>
#include "builder.hpp"
#include "io.hpp"
#include "binned.hpp"

#include <fstream>
#include <stdexcept>
#include <string_view>
#include <algorithm>
#include <cctype>

namespace lignum {

    namespace {
        
        bool is_lignum_bin(const std::string& filepath) {
            std::ifstream file(filepath, std::ios::binary);
            if (!file) return false;

            char magic[6];
            file.read(magic, 6);
            return (file.gcount() == 6 && std::string_view(magic, 6) == "LIGNUM");
        }

        std::string lower_case(std::string str) {
            std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::tolower(c); });
            return str;
        }

    } // namespace

    namespace {

        Model with_binned(Model model) {
        #if defined(LIGNUM_BINNED) && LIGNUM_BINNED
            model.binned = build_binned(model);
        #endif
            return model;
        }

        Model load_any(const std::string& filepath, const std::string& fmt);

    } // namespace

    Model load(const std::string& filepath, const std::string& format) {
        return with_binned(load_any(filepath, lower_case(format)));
    }

    namespace {

    Model load_any(const std::string& filepath, const std::string& fmt) {

        if (fmt == "auto" || fmt == "lignum" || fmt == "binary" || fmt == "bin") {
            if (is_lignum_bin(filepath)) {
                return load_binary(filepath);
            }
            throw std::invalid_argument("Binary file is not lignum format.");
        }

        if (fmt == "lightgbm" || fmt == "lgbm" || fmt == "lgb" || fmt == "xgboost" || fmt == "xgb") {
            return Builder::load_json(filepath, fmt);
        }
        throw std::invalid_argument("Unknown model format.");
    }

    } // namespace


} // namespace lignum