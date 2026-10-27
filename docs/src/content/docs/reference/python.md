---
title: Python API
description: Every public function and method of the lignum package.
---

Generated from the docstrings of lignum 0.3.1 by `docs/scripts/api_reference.py`.

## `lignum.load`

```python
lignum.load(filepath: str, format: str = 'auto', binned: bool = True, compiled: str | os.PathLike | None = None) -> Model
```

Load a trained model: LightGBM or XGBoost files, or a lignum `.bin`.

**Parameters**

- `filepath` *(str)* Path to the model file on disk.
- `format` *({'auto', 'lignum', 'lightgbm', 'xgboost', 'catboost'}, default='auto')* Library of the model. The file's own bytes tell which of its formats it is, so 'auto' loads a lignum `.bin`, a LightGBM `model.txt` (`save_model`) or `dump_model()` JSON, an XGBoost `.json` or `.ubj` (`save_model`) and a CatBoost JSON (`save_model(..., format="json")`). Naming the library only adds a check.
- `binned` *(bool, default=True)* Prepare the binned kernel: inputs are mapped once per row to integer bin codes and categorical decisions are precomputed, so the tree walk compares small integers. Results are identical either way; False keeps only the direct traversal (lower memory use).
- `compiled` *(path, optional)* Library written by `Model.compile(library=...)` for this model, or the folder it was written to (the file for this model and platform is picked); predictions then use the compiled trees without compiling again. Needs `binned=True`. A library for another model or lignum version raises `RuntimeError`. Only load libraries you trust.

**Returns**

- `model` *(lignum.Model)* The loaded model, ready for immediate inference.

**Raises**

- `FileNotFoundError` If the specified `filepath` does not exist.

**Examples**

```python
import lignum

model = lignum.load("xgboost_model.json", format="xgboost")
```

## `lignum.catboost_hash`

```python
lignum.catboost_hash(category) -> int
```

CatBoost's hash of a category: a string, or an integer (hashed as its decimal text, like
CatBoost). `Model.predict` hashes categorical columns itself; this is the value the kernel
sees, and what C and C++ callers put in the categorical columns of a float64 matrix.

## `lignum.cache_dir`

```python
lignum.cache_dir() -> str
```

Folder where `Model.compile()` keeps compiled libraries: `LIGNUM_CACHE_DIR` if set, otherwise
`~/.cache/lignum` (Linux, or `$XDG_CACHE_HOME/lignum`), `~/Library/Caches/lignum` (macOS) or
`%LOCALAPPDATA%\lignum\Cache` (Windows).

## `lignum.clear_cache`

```python
lignum.clear_cache() -> int
```

Remove the compiled libraries from `cache_dir()` and return how many were removed. Models that
already use one keep working; on Windows a library loaded by a running process stays.
Libraries written with `compile(library=...)` are not in the cache and are not touched.

## `lignum.Model`

A loaded LightGBM, XGBoost or CatBoost model.

Create it with `lignum.load()`. Predictions use the interpreter, or the compiled trees after
`compile()` / `load(..., compiled=...)`; both give the same, bit-identical results.

### `Model.autotune`

```python
Model.autotune(X: np.ndarray, n_jobs: int = -1, budget: float = 1.0) -> dict
```

Measure the interpreter and the compiled trees on a sample and use each where it is faster.

Which one wins depends on the model, the data and the CPU: compiled code is usually faster,
but on data whose splits are hard to predict the branch-free interpreter can win, mostly
for small batches. Batches of 1, 4, 16, ... rows up to the sample size are timed with
both (about `budget` seconds in total), and `compiled_min_rows` is set to the smallest
size from which the compiled trees win at every larger size (`sys.maxsize`: never).
Predictions are identical either way.

**Parameters**

- `X` *(array-like, shape (n_samples, n_features))* Representative rows (a few thousand are enough).
- `n_jobs` *(int, default=-1)* Threads, as in `predict`.
- `budget` *(float, default=1.0)* Approximate time to spend measuring, in seconds.

**Returns**

- `dict` `compiled_min_rows` and `timings`: {rows: (interpreter_us, compiled_us)}, median per call.

### `Model.categorical_features`

*Property.* Columns that hold categories (CatBoost models with categorical features), in order.

### `Model.compile`

```python
Model.compile(compiler: str = 'auto', flags: str = '', *, library: str | os.PathLike | None = None, jobs: int = 0, block_trees: int = 16, fallback: bool = False, autotune: np.ndarray | None = None) -> dict
```

Experimental: compile the trees to native code with the system C compiler.

Each tree becomes C code with constant thresholds; leaves are read from per-block
tables. Trees are grouped in blocks of `block_trees`, and a block is walked for a whole
chunk of rows before the next one; binning of categorical features, threads and
transforms stay in lignum and every row sums its trees in model order, so predictions
are unchanged (bit-identical).

Without `library`, the library is kept in a per-user cache (`lignum.cache_dir()`) and
compiling the same model again only loads it; `lignum.clear_cache()` empties the cache.
The code is built in a temporary folder that is always removed. Where the library is:
`model.compiled_library`, and `library` in the returned dict.

**Parameters**

- `compiler` *(str, default='auto')* C compiler. `'auto'` uses `cc` on Linux and macOS; on Windows `cl` if it is on PATH, otherwise the newest Visual Studio with C++ tools (found with vswhere), then `gcc` (MinGW) and `clang`. MSVC-style compilers (`cl`, `clang-cl`) are driven with their own options.
- `flags` *(str, default='')* Optimisation flags; empty means `-O2` (gcc, clang) or `/O2` (MSVC).
- `library` *(path, optional)* Write the library here instead of the cache, e.g. next to the `.bin` model, so production can attach it with `lignum.load(model_path, compiled=library)` without a compiler. A file path, or a folder (an existing one, or a path ending in `/`) that gets `lignum_<model fingerprint>_<platform>.so` (`.dll` on Windows); one folder can hold the libraries of several models and platforms. Always compiled anew.
- `jobs` *(int, default=0)* Source files compiled in parallel; 0 means one per CPU core.
- `block_trees` *(int, default=16)* Trees per generated function; larger blocks compile slower and run about the same.
- `fallback` *(bool, default=False)* When the compiler is missing or fails, warn and keep the interpreter instead of raising. The returned dict then has `compiled=False` and the `error`.
- `autotune` *(array, optional)* Sample of the data to predict on; after compiling, `autotune(sample)` picks the batch size from which the compiled trees are used (result under `autotune`).

**Returns**

- `dict` `compiled`, timings (`codegen_ms`, `compile_ms`, `load_ms`), `source_bytes`, `cached`, the `library` path and the `compiler` used (empty for a cached library).

**Examples**

```python
model = lignum.load("model.json", format="lightgbm")
model.save("model.bin")
model.compile(library="model.so")      # build machine: needs a C compiler

model = lignum.load("model.bin", compiled="model.so")  # production: no compiler
```

### `Model.compiled_library`

*Property.* Path of the compiled library the model uses (in the cache, or where it was written or
loaded from), or None when it uses the interpreter.

### `Model.compiled_min_rows`

*Property.* Batches with fewer rows use the interpreter even when compiled (set by `autotune`).

### `Model.decompile`

```python
Model.decompile() -> None
```

Go back to the interpreter.

### `Model.load_compiled`

```python
Model.load_compiled(library: str | os.PathLike) -> None
```

Attach a library built earlier by `compile(library=...)` for this model: the file, or
the folder it was written to (the file for this model and platform is picked).

The library records which model it was generated from and the lignum ABI; a library
for another model or lignum version raises `RuntimeError`. Loading runs the library's
native code, so only load libraries you trust (like a pickle).

### `Model.n_features`

*Property.* Minimum number of columns the input must have.

### `Model.n_outputs`

*Property.* Number of model outputs: the number of classes for multiclass models, otherwise 1.

### `Model.predict`

```python
Model.predict(X: np.ndarray, raw_score: bool = False, n_jobs: int = -1, strategy: str = 'auto') -> np.ndarray
```

Predict values for the given input data.

For optimal performance, ensure `X` is a C-contiguous float64 array.
Otherwise, an internal copy will be triggered, which may degrade performance.

**Parameters**

- `X` *(array-like or Arrow data, shape (n_samples, n_features))* The input samples to predict: anything numpy converts to a float64 matrix, or Arrow data through the PyCapsule interface (`pyarrow.RecordBatch` / `Table` / `RecordBatchReader`, polars `DataFrame`, ...). Arrow columns are used by position; float64, float32, integer, boolean and null columns are read without copying the table, nulls are NaN and dictionary (categorical) columns give their category codes. Predictions equal those for the same float64 matrix. pandas DataFrames are converted with numpy; pass `pyarrow.Table.from_pandas(df)` for categorical columns. A 1-D array or a flat list is one row. CatBoost models predict pandas frames of 50,000 rows or more through pyarrow when it is installed (faster, same results). CatBoost models with categorical features take their categories as strings or integers: a pandas DataFrame, a numpy object array, or a list (one row) or list of rows; see `categorical_features` and `catboost_hash`.
- `raw_score` *(bool, default=False)* Whether to output raw scores (leaf values) instead of transformed predictions.
- `n_jobs` *(int, default=-1)* Number of OpenMP threads to use. `-1` means using all available logical cores.
- `strategy` *({'auto', 'serial', 'rows'}, default='auto')* How the work is spread across threads. All strategies return identical results; only speed differs. `'serial'` runs on the calling thread (lowest latency for a few rows), `'rows'` splits rows between threads. `'auto'` uses `'serial'` below 8 rows or with one thread, `'rows'` otherwise.

**Returns**

- `preds` *(ndarray, shape (n_samples,) or (n_samples, n_classes))* The predicted values. Multiclass models return one column per class (probabilities, or margins with `raw_score=True`), except XGBoost `multi:softmax`, which returns the predicted class index.

**Examples**

```python
import numpy as np
import lignum

model = lignum.load("model.bin")
X = np.ascontiguousarray(np.array([[1.2, 3.4], [5.6, 7.8]], dtype=np.float64))
model.predict(X, n_jobs=2)
# array([0.45, 0.89])
```

### `Model.save`

```python
Model.save(filepath: str) -> None
```

Save the current model to the Lignum binary format.

This format is optimized for fast, memory-mapped (mmap) loading in 
subsequent sessions.

**Parameters**

- `filepath` *(str)* Destination file path for the compiled `.bin` model.

**Examples**

```python
model.save("optimized_model.bin")
```

### `Model.uses_binned`

*Property.* Whether batches use the binned kernel (False when disabled or unsupported by the model).

### `Model.uses_compiled`

*Property.* Whether predictions use compiled trees.
