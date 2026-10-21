#include <lignum/c_api.h>
#include <lignum/lignum.hpp>

#include <exception>
#include <string>

namespace {

    thread_local std::string last_error;

    template <typename Fn>
    int guarded(Fn&& fn) {
        try {
            fn();
            return 0;
        } catch (const std::exception& e) {
            last_error = e.what();
        } catch (...) {
            last_error = "Unknown error.";
        }
        return -1;
    }

    int fail(const char* message) {
        last_error = message;
        return -1;
    }

    const lignum::Model* as_model(LignumModelHandle handle) {
        return static_cast<const lignum::Model*>(handle);
    }

} // namespace

extern "C" {

const char* LignumGetLastError(void) {
    return last_error.c_str();
}

int LignumModelLoad(const char* filepath, const char* format, LignumModelHandle* out_handle) {
    if (!filepath || !format || !out_handle) return fail("LignumModelLoad: null argument.");
    return guarded([&] { *out_handle = new lignum::Model(lignum::load(filepath, format)); });
}

int LignumModelSave(LignumModelHandle handle, const char* filepath) {
    if (!handle || !filepath) return fail("LignumModelSave: null argument.");
    return guarded([&] { as_model(handle)->save(filepath); });
}

int LignumModelNumFeatures(LignumModelHandle handle, size_t* out_n_features) {
    if (!handle || !out_n_features) return fail("LignumModelNumFeatures: null argument.");
    *out_n_features = as_model(handle)->n_features;
    return 0;
}

int LignumModelPredict(LignumModelHandle handle, const double* X, size_t n_samples, size_t n_features,
                       int raw_score, int n_jobs, int strategy, double* out_preds) {
    if (!handle) return fail("LignumModelPredict: null handle.");
    if (n_samples > 0 && (!X || !out_preds)) return fail("LignumModelPredict: null input or output buffer.");
    if (strategy < LIGNUM_STRATEGY_AUTO || strategy > LIGNUM_STRATEGY_ROWS) return fail("LignumModelPredict: unknown strategy.");

    const auto s = static_cast<lignum::Strategy>(strategy);
    return guarded([&] {
        if (raw_score) as_model(handle)->predict_raw(X, n_samples, n_features, out_preds, n_jobs, s);
        else           as_model(handle)->predict(X, n_samples, n_features, out_preds, n_jobs, s);
    });
}

int LignumModelFree(LignumModelHandle handle) {
    delete static_cast<lignum::Model*>(handle);
    return 0;
}

} // extern "C"
