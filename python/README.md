<p align="center">
  <img src="https://raw.githubusercontent.com/misterekkk/lignum/main/docs/assets/logo.svg" width="96" alt="lignum logo">
</p>

# lignum

Fast inference for LightGBM, XGBoost and CatBoost models, with predictions that are
**bit-identical** to the library that trained them.

```bash
pip install pylignum
```

```python
import lignum

model = lignum.load("model.txt")    # LightGBM, XGBoost or CatBoost file, detected from its content
preds = model.predict(X)            # numpy, pandas, pyarrow, polars or a plain list
```

The package is installed as `pylignum` and imported as `lignum`.

## What it does

- **Exact.** Raw scores and probabilities equal the library's `predict` bit for bit: NaN and
  zero-as-missing, categorical splits, XGBoost's float32 arithmetic and CatBoost's CTRs included.
- **Fast to predict.** A cache-friendly interpreter for any batch size, and an optional compiled
  mode that turns LightGBM and XGBoost trees into native code with the system C compiler in a few
  seconds:

  ```python
  model.compile(library="model.so")                       # once, where a compiler is available
  model = lignum.load("model.bin", compiled="model.so")   # production: no compiler needed
  ```

  `model.autotune(X_sample)` measures both on your data and uses each where it is faster.
- **CatBoost with categorical features.** One-hot, CTRs and combinations of categories; categories
  come as strings or integers from pandas, numpy object arrays, lists, pyarrow or polars. 2-3x
  faster than CatBoost on batches and 7-14x on single rows.
- **Fast to start.** `model.save("model.bin")` writes a binary file that loads by memory mapping in
  well under a millisecond.
- **Arrow input without copies.** `predict` reads pyarrow tables and batches, polars data frames and
  any object with the Arrow PyCapsule interface directly; dictionary columns are categories.
- **C API** for services in other languages, with Arrow input as well.

## Supported models

- **LightGBM**: `model.txt` and JSON; numeric and categorical splits, multiclass.
- **XGBoost**: JSON and UBJSON; float32 semantics, categorical splits, multiclass.
- **CatBoost**: `.cbm` and JSON; symmetric trees, one-hot and CTR categories, multiclass, Poisson
  and Tweedie.

Models lignum cannot predict exactly (LightGBM linear trees and random forests, XGBoost
multi-target models, CatBoost non-symmetric trees and text features) are rejected at load time.

## Requirements

Python 3.10 or newer, numpy. Wheels are available for Linux (x86_64, aarch64), macOS 15+ (arm64,
x86_64) and Windows (x64); other platforms build from source with a C++20 compiler. The compiled mode
needs a C compiler at compile time only (cc on Linux and macOS; Visual Studio, MinGW gcc or clang on
Windows).

## Links

- Documentation: https://misterekkk.github.io/lignum/
- Benchmarks: https://misterekkk.github.io/lignum/benchmarks/
- Release notes: https://github.com/misterekkk/lignum/blob/main/CHANGELOG.md
- Source and issues: https://github.com/misterekkk/lignum

Licensed under the Apache License 2.0 (versions up to 0.2.0: MIT).
