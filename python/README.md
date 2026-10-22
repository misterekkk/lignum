<p align="center">
  <img src="https://raw.githubusercontent.com/misterekkk/lignum/dev/v0.2/docs/assets/logo.svg" width="96" alt="lignum logo">
</p>

# lignum

Fast inference for LightGBM and XGBoost models, with predictions that are **bit-identical** to the
library that trained them.

```bash
pip install pylignum
```

```python
import lignum

model = lignum.load("model.txt")    # LightGBM model.txt / JSON, XGBoost .json / .ubj
preds = model.predict(X)            # numpy array, or Arrow data (pyarrow, polars, ...)
```

The package is installed as `pylignum` and imported as `lignum`.

## What it does

- **Exact.** Raw scores and probabilities equal `booster.predict(...)` bit for bit, including NaN,
  zero-as-missing and categorical splits, for LightGBM and XGBoost (XGBoost's float32 arithmetic
  included).
- **Fast to start.** `model.save("model.bin")` writes a binary file that loads by memory mapping in
  well under a millisecond.
- **Fast to predict.** A cache-friendly interpreter for any batch size, and an optional compiled
  mode that turns the trees into native code with the system C compiler in a few seconds:

  ```python
  model.compile(library="model.so")                       # once, where a compiler is available
  model = lignum.load("model.bin", compiled="model.so")   # production: no compiler needed
  ```

  `model.autotune(X_sample)` measures both on your data and uses each where it is faster.
- **Arrow input without copies.** `predict` reads pyarrow tables and batches, polars data frames and
  any object with the Arrow PyCapsule interface directly; dictionary columns are categories.
- **C API** for services in other languages (`include/lignum/c_api.h`), with Arrow input as well.

## Requirements

Python 3.10 or newer, numpy. Wheels are available for Linux (x86_64, aarch64), macOS 15+ (arm64,
x86_64) and Windows (x64); other platforms build from source with a C++20 compiler. The compiled mode
needs a C compiler at compile time only (cc on Linux and macOS; Visual Studio, MinGW gcc or clang on
Windows).

## Links

- Source and issues: https://github.com/misterekkk/lignum
- Documentation: https://misterekkk.github.io/lignum/
