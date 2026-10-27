/* C API for lignum. Every function returns 0 on success and -1 on failure; on failure
 * LignumGetLastError() describes the error (thread-local, valid until the next call). */
#ifndef LIGNUM_C_API_H
#define LIGNUM_C_API_H

#include <stddef.h>

#include "arrow.h"

#if defined(_WIN32) && defined(LIGNUM_SHARED)
    #ifdef LIGNUM_EXPORTS
        #define LIGNUM_API __declspec(dllexport)
    #else
        #define LIGNUM_API __declspec(dllimport)
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #define LIGNUM_API __attribute__((visibility("default")))
#else
    #define LIGNUM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* LignumModelHandle;

/* Values match lignum::Strategy. */
#define LIGNUM_STRATEGY_AUTO   0
#define LIGNUM_STRATEGY_SERIAL 1
#define LIGNUM_STRATEGY_ROWS   2

LIGNUM_API const char* LignumGetLastError(void);

/* format: "auto" (lignum binary or detected from the file), "lightgbm", "xgboost", "catboost". */
LIGNUM_API int LignumModelLoad(const char* filepath, const char* format, LignumModelHandle* out_handle);

LIGNUM_API int LignumModelSave(LignumModelHandle handle, const char* filepath);

/* Minimum number of columns X must have. */
LIGNUM_API int LignumModelNumFeatures(LignumModelHandle handle, size_t* out_n_features);

/* Values per row written by LignumModelPredict: the number of classes for multiclass models,
 * except XGBoost multi:softmax (transformed) which writes one class index; otherwise 1. */
LIGNUM_API int LignumModelOutputWidth(LignumModelHandle handle, int raw_score, size_t* out_width);

/* X is row-major float64 [n_samples x n_features]; out_preds holds n_samples x width values
 * (see LignumModelOutputWidth), row-major.
 * n_jobs <= 0 uses all threads; strategy is one of LIGNUM_STRATEGY_*. */
LIGNUM_API int LignumModelPredict(LignumModelHandle handle,
                                  const double* X,
                                  size_t n_samples,
                                  size_t n_features,
                                  int raw_score,
                                  int n_jobs,
                                  int strategy,
                                  double* out_preds);

/* Arrow input (C Data Interface): `batch` is a record batch (struct array) described by
 * `schema`; its first LignumModelNumFeatures columns are the features, by position. Columns may
 * be float64, float32, integers, boolean or null; dictionary columns give their indices (category
 * codes). Nulls are NaN; values convert exactly like a float64 matrix, so predictions are the
 * same. out_preds holds batch->length x width values. Neither structure is released. */
LIGNUM_API int LignumModelPredictArrow(LignumModelHandle handle,
                                       const struct ArrowSchema* schema,
                                       const struct ArrowArray* batch,
                                       int raw_score,
                                       int n_jobs,
                                       int strategy,
                                       double* out_preds);

/* CatBoost's hash of a category: `size` bytes of UTF-8 text; integer categories are hashed as
 * their decimal text ("42"). CatBoost models with categorical features take the hash, as a
 * double, in each categorical column of X. Never fails. */
LIGNUM_API unsigned int LignumCatBoostHash(const char* category, size_t size);

/* Experimental compiled mode: compiles the trees to native code with the system C compiler
 * (see Model.compile in the Python docs). Predictions stay bit-identical.
 * compiler: NULL or "auto" (cc on POSIX; cl, Visual Studio, gcc, clang on Windows);
 * flags: NULL or "" for -O2 / /O2;
 * cache_dir: NULL for the per-user cache (LIGNUM_CACHE_DIR, %LOCALAPPDATA%\lignum\Cache,
 * ~/Library/Caches/lignum or ~/.cache/lignum); compiling the same model again loads the cached library;
 * library: NULL, or where to write the library instead (always built; to ship it next to the model):
 * a file, or a folder (existing, or ending in a separator) that gets lignum_<fingerprint>_<platform>.so/.dll.
 * Needs a model loaded with the binned kernel (the default).
 * Compile, LoadCompiled and Decompile change the handle: call them before sharing it between
 * threads, not concurrently with LignumModelPredict. */
LIGNUM_API int LignumModelCompile(LignumModelHandle handle, const char* compiler, const char* flags,
                                  const char* cache_dir, const char* library);

/* Attaches a library written by LignumModelCompile for this model, without a compiler: a file, or
 * a folder written with LignumModelCompile(..., library = folder). Fails for a
 * library of another model or lignum version. Loading runs the library's code: only load
 * libraries you trust. */
LIGNUM_API int LignumModelLoadCompiled(LignumModelHandle handle, const char* library);

/* Goes back to the interpreter. */
LIGNUM_API int LignumModelDecompile(LignumModelHandle handle);

/* *out_compiled = 1 when predictions use compiled trees, otherwise 0. */
LIGNUM_API int LignumModelUsesCompiled(LignumModelHandle handle, int* out_compiled);

LIGNUM_API int LignumModelFree(LignumModelHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* LIGNUM_C_API_H */
