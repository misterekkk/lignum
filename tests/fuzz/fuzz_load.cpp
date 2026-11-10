// libFuzzer target for one library's model files (LIGNUM_FUZZ_FORMAT: "lignum", "lightgbm",
// "xgboost" or "catboost"). Built with AddressSanitizer and UndefinedBehaviorSanitizer
// (tests/fuzz/CMakeLists.txt), so any out-of-bounds access, use after free or undefined behaviour
// is a crash. Every input must be rejected with an exception, or load and predict safely; and a
// loaded model must also be consistent:
//   - the binned and the direct kernel predict identically (not for .bin files, whose binned
//     sections are stored separately and may be damaged on their own), and
//   - saved as .bin, it loads again and predicts identically.
// Run: tests/fuzz/README.md.
#include <lignum/lignum.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#ifndef LIGNUM_FUZZ_FORMAT
    #error "define LIGNUM_FUZZ_FORMAT"
#endif

namespace {

    constexpr size_t kRows = 9;
    constexpr size_t kMaxFeatures = 64;  // models needing more columns are only loaded

    // Per process: libFuzzer's -fork and -jobs run several processes at once.
    std::string temp_path(const char* suffix) {
        static const std::string base =
            (std::filesystem::temp_directory_path() / ("lignum_fuzz_" + std::to_string(getpid()))).string();
        return base + suffix;
    }

    void write_file(const std::string& path, const uint8_t* data, size_t size) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    [[noreturn]] void fail(const char* what) {
        std::fprintf(stderr, "lignum fuzz: %s\n", what);
        std::abort();
    }

    // Bit-identical, NaN equal to NaN.
    bool same(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (!(a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i]))) || std::signbit(a[i]) != std::signbit(b[i])) return false;
        }
        return true;
    }

    // Two inputs: values that hit every kind of split (NaN, zeros, the zero band, large and
    // negative values), and non-negative integers, which CatBoost accepts as category hashes.
    const std::vector<double>& input(int which, size_t n_features) {
        static std::vector<double> X[2];
        static size_t width = 0;
        if (width != n_features) {
            width = n_features;
            const double values[] = {0.0, -0.0, 1.0, 2.5, -1.0, std::numeric_limits<double>::quiet_NaN(), 1e30, -1e-36, 7.0, 0.5, 63.0};
            for (auto& x : X) x.assign(kRows * n_features, 0.0);
            for (size_t i = 0; i < kRows * n_features; i++) {
                X[0][i] = values[i % (sizeof(values) / sizeof(values[0]))];
                X[1][i] = static_cast<double>((i * 2654435761u) % 1000);
            }
        }
        return X[which];
    }

    // Predictions of every input, transformed and raw, in batches of kRows and of one row; an
    // input the model rejects (e.g. not a category hash) is skipped. Empty when not predicted.
    std::optional<std::vector<double>> predictions(const lignum::Model& m) {
        if (m.n_features > kMaxFeatures) return std::nullopt;
        const size_t nf = std::max<size_t>(m.n_features, 1);
        std::vector<double> all;
        for (int which : {0, 1}) {
            const std::vector<double>& X = input(which, nf);
            for (bool raw : {false, true}) {
                std::vector<double> batch(kRows * m.output_width(raw)), one(m.output_width(raw));
                try {
                    if (raw) {
                        m.predict_raw(X.data(), kRows, nf, batch.data(), 1);
                        m.predict_raw(X.data(), 1, nf, one.data(), 1);
                    } else {
                        m.predict(X.data(), kRows, nf, batch.data(), 1);
                        m.predict(X.data(), 1, nf, one.data(), 1);
                    }
                } catch (const std::exception&) {
                    all.push_back(-12345.0);  // the same rejection is expected from the other kernel
                    continue;
                }
                all.insert(all.end(), batch.begin(), batch.end());
                all.insert(all.end(), one.begin(), one.end());
            }
        }
        return all;
    }

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const std::string path = temp_path(".model");
    write_file(path, data, size);
    const bool binary = std::strcmp(LIGNUM_FUZZ_FORMAT, "lignum") == 0;

    bool loaded[2] = {false, false};
    std::optional<std::vector<double>> by_kernel[2];
    for (bool binned : {true, false}) {
        std::optional<lignum::Model> model;
        try {
            model = lignum::load(path, LIGNUM_FUZZ_FORMAT, binned);
        } catch (const std::exception&) {
            continue;  // rejected: fine
        }
        loaded[binned] = true;
        const auto preds = predictions(*model);
        by_kernel[binned] = preds;

        // A model that loaded must survive the binary format unchanged.
        const std::string bin = temp_path(".bin");
        try {
            model->save(bin);
        } catch (const std::exception&) {
            fail("a loaded model could not be saved");
        }
        try {
            const auto reloaded = lignum::load(bin, "auto", binned);
            if (preds && !same(*preds, *predictions(reloaded))) fail("the saved .bin predicts differently");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "reload error: %s\n", e.what());
            fail("a saved .bin could not be loaded");
        }
    }
    if (!binary && loaded[0] != loaded[1]) fail("only one kernel loaded the file");
    if (!binary && by_kernel[0] && by_kernel[1] && !same(*by_kernel[0], *by_kernel[1])) fail("the binned and direct kernels differ");
    return 0;
}
