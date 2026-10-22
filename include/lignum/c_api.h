/* C API for lignum. Every function returns 0 on success and -1 on failure; on failure
 * LignumGetLastError() describes the error (thread-local, valid until the next call). */
#ifndef LIGNUM_C_API_H
#define LIGNUM_C_API_H

#include <stddef.h>

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

/* format: "auto" (lignum binary), "lightgbm", "xgboost". */
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

/* Experimental compiled mode: compiles the trees to native code with the system C compiler
 * (see Model.compile in the Python docs). Predictions stay bit-identical.
 * compiler: NULL or "auto" (cc on POSIX; cl, Visual Studio, gcc, clang on Windows);
 * flags: NULL or "" for -O2 / /O2; work_dir: NULL for the system temp directory (cache);
 * library: NULL, or a path to also write the library to (to ship it next to the model).
 * Needs a model loaded with the binned kernel (the default).
 * Compile, LoadCompiled and Decompile change the handle: call them before sharing it between
 * threads, not concurrently with LignumModelPredict. */
LIGNUM_API int LignumModelCompile(LignumModelHandle handle, const char* compiler, const char* flags,
                                  const char* work_dir, const char* library);

/* Attaches a library written by LignumModelCompile for this model, without a compiler. Fails for a
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
