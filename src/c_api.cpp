#include <lignum/c_api.h>
#include <lignum/lignum.hpp>
#include "compiled.hpp"

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

    lignum::Model* as_model(LignumModelHandle handle) {
        return static_cast<lignum::Model*>(handle);
    }

    std::string or_default(const char* s, const char* fallback) {
        return (s && *s) ? s : fallback;
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

unsigned int LignumCatBoostHash(const char* category, size_t size) {
    return lignum::catboost_hash(std::string_view(category ? category : "", category ? size : 0));
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

int LignumModelNumTrees(LignumModelHandle handle, size_t* out_n_trees) {
    if (!handle || !out_n_trees) return fail("LignumModelNumTrees: null argument.");
    *out_n_trees = as_model(handle)->n_trees();
    return 0;
}

int LignumModelOutputWidth(LignumModelHandle handle, int raw_score, size_t* out_width) {
    if (!handle || !out_width) return fail("LignumModelOutputWidth: null argument.");
    *out_width = as_model(handle)->output_width(raw_score != 0);
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

int LignumModelPredictArrow(LignumModelHandle handle, const struct ArrowSchema* schema, const struct ArrowArray* batch,
                            int raw_score, int n_jobs, int strategy, double* out_preds) {
    if (!handle || !schema || !batch) return fail("LignumModelPredictArrow: null argument.");
    if (batch->length > 0 && !out_preds) return fail("LignumModelPredictArrow: null output buffer.");
    if (strategy < LIGNUM_STRATEGY_AUTO || strategy > LIGNUM_STRATEGY_ROWS) return fail("LignumModelPredictArrow: unknown strategy.");
    return guarded([&] {
        lignum::predict_arrow(*as_model(handle), *schema, *batch, out_preds, raw_score != 0, n_jobs, static_cast<lignum::Strategy>(strategy));
    });
}

int LignumModelCompile(LignumModelHandle handle, const char* compiler, const char* flags, const char* cache_dir, const char* library) {
    if (!handle) return fail("LignumModelCompile: null handle.");
    return guarded([&] {
        lignum::CompileOptions options;
        options.compiler = or_default(compiler, "auto");
        options.flags = or_default(flags, "");
        options.cache_dir = or_default(cache_dir, "");
        options.library = or_default(library, "");
        as_model(handle)->compiled = lignum::compile_forest(*as_model(handle), options, nullptr);
    });
}

int LignumModelLoadCompiled(LignumModelHandle handle, const char* library) {
    if (!handle || !library) return fail("LignumModelLoadCompiled: null argument.");
    return guarded([&] { as_model(handle)->compiled = lignum::load_compiled(*as_model(handle), library); });
}

int LignumModelDecompile(LignumModelHandle handle) {
    if (!handle) return fail("LignumModelDecompile: null handle.");
    as_model(handle)->compiled = nullptr;
    return 0;
}

int LignumModelUsesCompiled(LignumModelHandle handle, int* out_compiled) {
    if (!handle || !out_compiled) return fail("LignumModelUsesCompiled: null argument.");
    *out_compiled = as_model(handle)->compiled != nullptr;
    return 0;
}

int LignumModelFree(LignumModelHandle handle) {
    delete static_cast<lignum::Model*>(handle);
    return 0;
}

} // extern "C"
