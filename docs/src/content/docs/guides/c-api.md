---
title: C API
description: Use lignum from C, C++ or any language with a C foreign function interface, for LightGBM, XGBoost and CatBoost models.
---

The C API (`include/lignum/c_api.h`) is built as the shared library `liblignum` by a plain CMake
build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Every function returns 0 on success and -1 on failure; `LignumGetLastError()` then describes the
error (thread-local, valid until the next call).

```c
#include <lignum/c_api.h>
#include <stdio.h>

int main(void) {
    LignumModelHandle model;
    if (LignumModelLoad("model.bin", "auto", &model) != 0) {
        fprintf(stderr, "%s\n", LignumGetLastError());
        return 1;
    }
    size_t n_features, width;
    LignumModelNumFeatures(model, &n_features);
    LignumModelOutputWidth(model, /*raw_score=*/0, &width);

    double row[64] = {0};            /* n_features values, NaN for missing */
    double out[16];                  /* width values per row */
    LignumModelPredict(model, row, /*n_samples=*/1, n_features, /*raw_score=*/0,
                       /*n_jobs=*/1, LIGNUM_STRATEGY_AUTO, out);
    printf("%f\n", out[0]);
    LignumModelFree(model);
    return 0;
}
```

## Functions

| Function | Purpose |
|---|---|
| `LignumModelLoad(path, format, &handle)` | load any [model file](../models/) (`format` as in Python) |
| `LignumModelSave(handle, path)` | write the `.bin` format |
| `LignumModelNumFeatures`, `LignumModelOutputWidth` | input columns, values per output row |
| `LignumModelPredict(handle, X, n, n_features, raw, n_jobs, strategy, out)` | row-major `float64` input |
| `LignumModelPredictArrow(handle, schema, batch, raw, n_jobs, strategy, out)` | an Arrow record batch ([Arrow input](../arrow/)) |
| `LignumCatBoostHash(text, size)` | a category's hash for [CatBoost models](#catboost-models) |
| `LignumModelCompile(handle, compiler, flags, cache_dir, library)` | [compiled mode](../compiled/); `NULL` for defaults (per-user cache); `library` writes it there instead, a file or a folder |
| `LignumModelLoadCompiled(handle, library)` | attach a library written by `LignumModelCompile` |
| `LignumModelDecompile`, `LignumModelUsesCompiled` | back to the interpreter; which one predicts |
| `LignumModelFree(handle)` | release the model |

## CatBoost models

CatBoost models (`.cbm`, JSON, or the `.bin` written by `LignumModelSave`) load and predict through
the same functions as LightGBM and XGBoost models. Numeric columns take their values as usual. Only
categorical columns need one extra step.

### Why categories are hashed

CatBoost does not work with category text internally. When it trains, it turns every category
(`"Private"`, `"ios"`, ...) into a 32-bit hash, and the model's tables (one-hot values and CTR
statistics) are keyed by those hashes, not by the text. `LignumModelPredict` takes a matrix of
`double`, which cannot hold a string, so the caller hashes each category first, with exactly the
function CatBoost uses, and writes the hash into the category's column. A 32-bit integer is exact in
a `double`, so nothing is lost.

`LignumCatBoostHash(text, size)` is that function: CatBoost's hash of `size` bytes of UTF-8 text.
Integer categories are hashed as their decimal text, as CatBoost does, so `42` and `"42"` are the
same category. Python and Arrow callers never see this step: there lignum hashes the categories
itself.

```c
#include <lignum/c_api.h>
#include <stdio.h>
#include <string.h>

/* The category's hash, as a double, for a categorical column. */
static double category(const char* text) {
    return (double)LignumCatBoostHash(text, strlen(text));
}

int main(void) {
    LignumModelHandle model;
    if (LignumModelLoad("model.cbm", "auto", &model) != 0) {
        fprintf(stderr, "%s\n", LignumGetLastError());
        return 1;
    }

    /* Columns as in training: city, device (categorical), age, spend. */
    enum { N_FEATURES = 4, N_ROWS = 2 };
    double X[N_ROWS * N_FEATURES] = {
        category("Krakow"),   category("ios"), 30.0, 10.0,
        category("Atlantis"), category("web"), 50.0, 99.0,   /* an unseen category is fine */
    };
    double out[N_ROWS];   /* one probability per row (LignumModelOutputWidth gives the width) */
    if (LignumModelPredict(model, X, N_ROWS, N_FEATURES, /*raw_score=*/0, /*n_jobs=*/1,
                           LIGNUM_STRATEGY_AUTO, out) != 0) {
        fprintf(stderr, "%s\n", LignumGetLastError());
        return 1;
    }
    printf("%.17g %.17g\n", out[0], out[1]);   /* equal to CatBoost's predict_proba, bit for bit */
    LignumModelFree(model);
    return 0;
}
```

- **Unseen categories** are allowed and get CatBoost's prior, exactly as in CatBoost.
- **Missing values** do not exist for CatBoost categories (CatBoost rejects them in training too):
  hash the text you used for them in training, such as `"NA"`. A NaN, negative or fractional value in
  a categorical column is rejected with an error before anything is predicted.
- **Speed**: hash each distinct value once (for example, once per dictionary entry or cache them in a
  hash map) and reuse it; hashing is the only per-row work outside the model.
- **Arrow**: `LignumModelPredictArrow` takes string, integer and dictionary columns directly and
  hashes them itself, so an Arrow record batch needs no preparation at all.
- **Compiled mode** does not apply to CatBoost models: their symmetric trees are already branch-free,
  and `LignumModelCompile` returns an error saying so.

`LignumModelPredict` can be called from several threads with the same handle. Compile, attach or
decompile before the handle is shared, never concurrently with predictions.

C++ programs can use the C++ API directly: `lignum::load` in `include/lignum/lignum.hpp` and
`Model::predict` in `include/lignum/model.hpp`.
