#include <lignum/lignum.hpp>
#include "builder.hpp"
#include "io.hpp"
#include "binned.hpp"
#include "catboost.hpp"
#include "parsers/parsers.hpp"

#include <fstream>
#include <stdexcept>
#include <string_view>
#include <algorithm>
#include <cctype>

namespace lignum {

    namespace {

        enum class FileKind { Lignum, LightGBMJson, XGBoostJson, CatBoostJson, Ubjson, LightGBMText, Unknown };

        // Recognises a model file by its first bytes, not its name: lignum's magic; JSON (an
        // object whose first key follows '{'), XGBoost's when "learner" appears early, CatBoost's
        // when its first key is one of CatBoost's (they are sorted: "ctr_data", "features_info"); UBJSON
        // (XGBoost .ubj: '{' followed by a length marker or an optimized-container header);
        // LightGBM's text format ("tree" on the first line).
        FileKind sniff(const std::string& filepath) {
            std::ifstream file(filepath, std::ios::binary);
            if (!file) throw std::invalid_argument("Cannot open model file: " + filepath);
            std::string head(4096, '\0');
            file.read(head.data(), static_cast<std::streamsize>(head.size()));
            head.resize(static_cast<size_t>(file.gcount()));
            if (head.rfind("LIGNUM", 0) == 0) return FileKind::Lignum;
            size_t i = head.find_first_not_of(" \t\r\n");
            if (i == std::string::npos) return FileKind::Unknown;
            if (head[i] == '{') {
                const size_t j = head.find_first_not_of(" \t\r\n", i + 1);
                if (j != std::string::npos && (head[j] == '"' || head[j] == '}')) {
                    if (head.compare(j, 10, "\"ctr_data\"") == 0 || head.compare(j, 15, "\"features_info\"") == 0) return FileKind::CatBoostJson;
                    return head.find("\"learner\"") != std::string::npos ? FileKind::XGBoostJson : FileKind::LightGBMJson;
                }
                if (j != std::string::npos && std::string_view("iUIlL$#").find(head[j]) != std::string_view::npos) return FileKind::Ubjson;
                return FileKind::Unknown;
            }
            if (head.compare(i, 4, "tree") == 0) return FileKind::LightGBMText;
            return FileKind::Unknown;
        }

        std::string lower_case(std::string str) {
            std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::tolower(c); });
            return str;
        }

    } // namespace

    namespace {

        // Builds the binned kernel unless it came from the file (or is not wanted).
        Model with_binned(Model model, bool binned) {
            if (!binned || model.oblivious) model.binned = nullptr;
            else if (!model.binned) model.binned = build_binned(model);
            return model;
        }

        Model load_any(const std::string& filepath, const std::string& fmt, bool binned);

    } // namespace

    Model load(const std::string& filepath, const std::string& format, bool binned) {
        return with_binned(load_any(filepath, lower_case(format), binned), binned);
    }

    namespace {

    Model load_any(const std::string& filepath, const std::string& fmt, bool binned) {
        const bool lightgbm = fmt == "lightgbm" || fmt == "lgbm" || fmt == "lgb";
        const bool xgboost = fmt == "xgboost" || fmt == "xgb";
        const bool lignum = fmt == "lignum" || fmt == "binary" || fmt == "bin";
        const bool catboost = fmt == "catboost" || fmt == "cb";
        if (fmt != "auto" && !lightgbm && !xgboost && !lignum && !catboost) {
            throw std::invalid_argument("Unknown model format '" + fmt + "'; expected auto, lignum, lightgbm, xgboost or catboost.");
        }
        const FileKind kind = sniff(filepath);
        // `format` names the library; the file's own bytes say which of its formats it is.
        switch (kind) {
            case FileKind::CatBoostJson:
                if (lightgbm || xgboost || lignum) break;
                return parsers::load_catboost_json(filepath);
            case FileKind::Lignum:
                if (lightgbm || xgboost || catboost) throw std::invalid_argument("This is a lignum binary model; load it with format='auto'.");
                return load_binary(filepath, binned);
            case FileKind::LightGBMText:
                if (xgboost || lignum || catboost) break;
                return Builder::load_model(filepath, ModelSource::LightGBMText);
            case FileKind::Ubjson:
                if (lightgbm || lignum || catboost) break;
                return Builder::load_model(filepath, ModelSource::XGBoostUbjson);
            case FileKind::LightGBMJson:
            case FileKind::XGBoostJson:
                if (lignum || catboost) break;
                if (lightgbm) return Builder::load_model(filepath, ModelSource::LightGBMJson);
                if (xgboost) return Builder::load_model(filepath, ModelSource::XGBoostJson);
                return Builder::load_model(filepath, kind == FileKind::XGBoostJson ? ModelSource::XGBoostJson : ModelSource::LightGBMJson);
            case FileKind::Unknown:
                break;
        }
        throw std::invalid_argument("Unrecognised model file " + filepath + " for format '" + fmt +
                                    "': expected a lignum .bin, LightGBM model.txt or dump_model() JSON, or XGBoost JSON / UBJSON.");
    }

    } // namespace


} // namespace lignum