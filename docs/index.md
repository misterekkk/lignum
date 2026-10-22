# lignum

Ultra-fast inference library for decision-tree based models (LightGBM, XGBoost) in Python. 

The `lignum` package bypasses the bottlenecks of standard parsers by providing a native C++ engine designed for maximum CPU cache efficiency and instant file loading via memory mapping (`mmap`).

## Key Features

* **Zero-Overhead Architecture:** The custom binary format (`.bin`) is loaded via `mmap`, allowing massive models to be initialized in milliseconds while sharing RAM across multiple processes.
* **Direct JSON Inference:** You can load and predict directly from XGBoost/LightGBM JSON dumps without any mandatory conversion steps.
* **Cache Efficiency:** The package is optimized for C-contiguous (row-major) NumPy arrays, ensuring maximum performance for the CPU cache during inference.
* **Multi-threading:** Native OpenMP support (`n_jobs`) allows predictions to scale seamlessly across all available CPU cores.

## Installation

You can install the package directly from the source using `pip`. 

```bash
# Clone the repository
git clone [https://github.com/misterekkk/lignum.git](https://github.com/misterekkk/lignum.git)
cd lignum

# Install the Python package directly from the directory
pip install .

```

## Quick Start

### Scenario A: Direct Prediction from JSON

You can run predictions straight from a model exported by LightGBM or XGBoost. This is ideal for testing and standard data science workflows.

```python
import numpy as np
import lignum

# 1. Load a trained model from a JSON file
model = lignum.load("model_lightgbm.json", format="lightgbm")

# 2. Prepare test data
# For optimal performance, ensure the array is C-contiguous float64
X = np.random.randn(1000, 50).astype(np.float64)
X = np.ascontiguousarray(X)

# 3. Predict using all available CPU cores
preds = model.predict(X, raw_score=False, n_jobs=-1)

```

### Scenario B: Production mode via Binary Format

For production environments (e.g., serverless functions, REST APIs) where cold-start time and RAM usage matter, save the model to Lignum's binary format.

```python
import numpy as np
import lignum

# 1. Load your JSON model and convert it to a binary file
model = lignum.load("model_lightgbm.json", format="lightgbm")
model.save("lignum_model.bin")

# 2. Load via mmap (instantaneous, shares RAM across processes)
fast_model = lignum.load("lignum_model.bin", format="binary")

# 3. Predict
# (Assuming X is defined as above)
preds = fast_model.predict(X, n_jobs=-1)

```
### Scenario C: Compiled mode (experimental)

For the highest throughput, compile the trees to native code with the system C compiler. Predictions stay
bit-identical to the source library; on real data the compiled trees are typically 2-4x faster than the
interpreter. Compiling 500 trees takes a few seconds, so compile once on a machine with a compiler and ship
the library next to the binary model:

```python
import lignum

# Build machine (needs gcc or clang)
model = lignum.load("model_lightgbm.json", format="lightgbm")
model.save("lignum_model.bin")
model.compile(library="lignum_model.so")

# Production (no compiler needed): attaching the library takes about 2 ms
model = lignum.load("lignum_model.bin", compiled="lignum_model.so")
preds = model.predict(X)
```

- A library records the model it was generated from and the lignum version of the generated code; loading it
  for another model or version raises `RuntimeError`, so stale libraries cannot give wrong predictions.
- Loading a library runs its native code: only load libraries you built or trust, as with pickle files.
- `model.compile(fallback=True)` keeps the interpreter with a warning when no compiler is available.
- Both modes stay available: `model.decompile()` returns to the interpreter, `model.uses_compiled` tells which
  one predicts. The interpreter can be faster on data whose splits are hard to predict, mostly for small batches:
  `model.autotune(X_sample)` (or `compile(autotune=X_sample)`) times both on your data and sets
  `model.compiled_min_rows`, the batch size from which the compiled trees are used.
- On Windows, `compiler="auto"` uses `cl` (Visual Studio with C++ tools, found automatically), MinGW `gcc` or
  `clang`; on Linux and macOS it uses `cc`.

### Scenario D: Arrow data (pyarrow, polars, streams)

`predict` accepts Arrow data through the PyCapsule interface, without copying the table into a numpy matrix and
without requiring pyarrow itself: `pyarrow.RecordBatch`, `Table` and `RecordBatchReader`, polars `DataFrame`, and
any object with `__arrow_c_array__` / `__arrow_c_stream__`. Columns are used by position (the first
`model.n_features` columns); float64, float32, integer, boolean and null columns are supported, nulls are NaN and
dictionary (categorical) columns give their category codes. Predictions equal those for the same float64 matrix.

```python
import pyarrow as pa

table = pa.Table.from_pandas(df)        # categorical pandas columns become dictionary columns
preds = model.predict(table)            # one result for all batches of the table
for batch in reader:                    # e.g. batches decoded from a Kafka topic
    preds = model.predict(batch)
```

C and C++ programs use `LignumModelPredictArrow` (C API) or `lignum::predict_arrow` with the structures of the
Arrow C Data Interface (`include/lignum/arrow.h`).
