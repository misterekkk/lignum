---
title: C API
description: Use lignum from C, C++ or any language with a C foreign function interface.
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
| `LignumModelCompile(handle, compiler, flags, work_dir, library)` | [compiled mode](../compiled/); `NULL` for defaults |
| `LignumModelLoadCompiled(handle, library)` | attach a library written by `LignumModelCompile` |
| `LignumModelDecompile`, `LignumModelUsesCompiled` | back to the interpreter; which one predicts |
| `LignumModelFree(handle)` | release the model |

`LignumModelPredict` can be called from several threads with the same handle. Compile, attach or
decompile before the handle is shared, never concurrently with predictions.

C++ programs can use the C++ API directly: `lignum::load` in `include/lignum/lignum.hpp` and
`Model::predict` in `include/lignum/model.hpp`.
