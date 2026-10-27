"""
lignum: bit-exact, fast inference for LightGBM, XGBoost and CatBoost models.

`lignum.load(path)` reads LightGBM (`model.txt`, JSON), XGBoost (JSON, UBJSON) and CatBoost (JSON)
models or lignum's own `.bin` file; `Model.predict` takes numpy arrays, pandas DataFrames or Arrow
data. Predictions equal the library's `predict` bit for bit, with the interpreter or with trees
compiled to native code (`Model.compile`).
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import warnings

import numpy as np

from . import _core

__all__ = ["Model", "load", "catboost_hash", "cache_dir", "clear_cache"]
__version__ = "0.3.1"


def cache_dir() -> str:
    """
    Folder where `Model.compile()` keeps compiled libraries: `LIGNUM_CACHE_DIR` if set, otherwise
    `~/.cache/lignum` (Linux, or `$XDG_CACHE_HOME/lignum`), `~/Library/Caches/lignum` (macOS) or
    `%LOCALAPPDATA%\\lignum\\Cache` (Windows).
    """
    return _core.cache_dir()


def clear_cache() -> int:
    """
    Remove the compiled libraries from `cache_dir()` and return how many were removed. Models that
    already use one keep working; on Windows a library loaded by a running process stays.
    Libraries written with `compile(library=...)` are not in the cache and are not touched.
    """
    return _core.clear_cache()


def _is_pandas(X) -> bool:
    return type(X).__module__.split(".")[0] == "pandas"


def catboost_hash(category) -> int:
    """
    CatBoost's hash of a category: a string, or an integer (hashed as its decimal text, like
    CatBoost). `Model.predict` hashes categorical columns itself; this is the value the kernel
    sees, and what C and C++ callers put in the categorical columns of a float64 matrix.
    """
    return _core.catboost_hash(category)


# From this many rows, CatBoost models predict pandas frames through Arrow (parallel hashing and
# binning); below, pyarrow's fixed cost per frame (~0.3-1 ms) is more than it saves.
_ARROW_MIN_ROWS = 50_000


def _arrow_table(frame):
    """A pandas DataFrame as a pyarrow Table, or None (no pyarrow, or columns Arrow cannot convert).
    CatBoost models predict large frames through Arrow: columns are hashed and binned per block
    in parallel, instead of a float64 matrix built column by column in Python first."""
    try:
        import pyarrow as pa
    except ImportError:
        return None
    try:
        return pa.Table.from_pandas(frame, preserve_index=False)
    except (pa.ArrowException, TypeError, ValueError):
        return None


def _category_hashes(column) -> np.ndarray:
    """Hashes of one categorical column (a 1-D array) as float64."""
    return _core._catboost_hashes(np.asarray(column, dtype=object))


class Model:
    """
    A loaded LightGBM, XGBoost or CatBoost model.

    Create it with `lignum.load()`. Predictions use the interpreter, or the compiled trees after
    `compile()` / `load(..., compiled=...)`; both give the same, bit-identical results.
    """

    def __init__(self, _c_model, validate_features: bool = True):
        self._c_model = _c_model
        self._categorical = frozenset(_c_model.categorical_features)
        self._category_cache = {}
        self._feature_names = list(_c_model.feature_names)
        self._names_index = None     # pandas Index of the feature names, made on first use
        self._checked_columns = None  # the last columns object that passed the check
        self.validate_features = validate_features

    @property
    def feature_names(self) -> list[str] | None:
        """Names of the columns the model was trained on, in order, or None when the model has no
        names (trained on a matrix). `X[model.feature_names]` puts a DataFrame's columns in the
        model's order."""
        return list(self._feature_names) if self._feature_names else None

    def _check_columns(self, columns) -> None:
        """Raise ValueError when a DataFrame's columns do not start with the model's feature names.
        Frames without names (e.g. built from a numpy array: integer column labels) are used by
        position, like numpy arrays."""
        if columns is self._checked_columns or columns.dtype.kind not in "OU":
            return
        if self._names_index is None:
            import pandas as pd

            self._names_index = pd.Index(self._feature_names)
        n = len(self._names_index)
        if not (columns if len(columns) == n else columns[:n]).equals(self._names_index):
            # Mismatch, or a LightGBM name with spaces (stored with '_'): C++ decides and explains.
            self._c_model.check_feature_names([str(c) for c in columns])
        self._checked_columns = columns

    def _catboost_frame(self, X) -> np.ndarray:
        """float64 matrix of a pandas DataFrame, column by column: categorical columns as the
        hashes of their categories (cached per categories object) indexed by their codes."""
        out = np.empty(X.shape, dtype=np.float64)
        iter_arrays = getattr(X, "_iter_column_arrays", None)  # pandas internals: ~1 µs per column
        arrays = iter_arrays() if iter_arrays is not None else (X.iloc[:, j].array for j in range(X.shape[1]))
        for j, a in enumerate(arrays):
            if j not in self._categorical:
                out[:, j] = a if isinstance(a, np.ndarray) else a.to_numpy(dtype=np.float64, na_value=np.nan)
            elif hasattr(a, "codes") and hasattr(a, "categories"):  # pandas Categorical
                codes = np.asarray(a.codes)
                if (codes < 0).any():
                    raise ValueError(f"Categorical column {X.columns[j]!r} has missing values; CatBoost needs a category "
                                     "(e.g. the string 'nan') in every row.")
                out[:, j] = self._category_table(a.categories)[codes]
            else:
                out[:, j] = _core._catboost_hashes(a if isinstance(a, np.ndarray) and a.dtype == object else np.asarray(a, dtype=object))
        return out

    def _category_table(self, categories) -> np.ndarray:
        """Hashes of a pandas categories Index. Kept while the Index lives (frames sliced from one
        frame share it), so small frames do not hash every category again."""
        cache = self._category_cache
        entry = cache.get(id(categories))
        if entry is not None and entry[0] is categories:
            return entry[1]
        if len(cache) >= 256:
            cache.clear()
        hashes = _core._catboost_hashes(categories.to_numpy(dtype=object))
        cache[id(categories)] = (categories, hashes)
        return hashes

    def _catboost_matrix(self, X) -> np.ndarray:
        """float64 matrix with category hashes in the categorical columns (CatBoost models)."""
        categorical = self._c_model.categorical_features
        if isinstance(X, (list, tuple)):
            return _core._catboost_rows(X, categorical)
        if _is_pandas(X):
            if X.ndim == 1:  # a Series is one row
                return _core._catboost_rows(X.tolist(), categorical)
            return self._catboost_frame(X)
        if isinstance(X, np.ndarray) and X.dtype.kind in "fc" and self._categorical:
            raise ValueError("This CatBoost model has categorical features; pass them as strings or integers "
                             "(an object array, a list or a pandas DataFrame), not floats.")
        A = np.asarray(X, dtype=object if not (isinstance(X, np.ndarray) and X.dtype.kind in "iu") else X.dtype)
        if A.ndim == 1:  # one row
            A = A.reshape(1, -1)
        if A.ndim != 2:
            raise ValueError("Input must be 2-D (rows x features), or one row.")
        out = np.empty(A.shape, dtype=np.float64)
        for j in range(A.shape[1]):
            out[:, j] = _category_hashes(A[:, j]) if j in self._categorical else A[:, j].astype(np.float64)
        return out

    def predict(self, X: np.ndarray, raw_score: bool = False, n_jobs: int = -1, strategy: str = "auto",
                validate_features: bool | None = None) -> np.ndarray:
        """
        Predict values for the given input data.

        For optimal performance, ensure `X` is a C-contiguous float64 array.
        Otherwise, an internal copy will be triggered, which may degrade performance.

        Parameters
        ----------
        X : array-like or Arrow data, shape (n_samples, n_features)
            The input samples to predict: anything numpy converts to a float64 matrix, or Arrow
            data through the PyCapsule interface (`pyarrow.RecordBatch` / `Table` /
            `RecordBatchReader`, polars `DataFrame`, ...). Arrow columns are used by position;
            float64, float32, integer, boolean and null columns are read without copying the
            table, nulls are NaN and dictionary (categorical) columns give their category codes.
            Predictions equal those for the same float64 matrix. pandas DataFrames are converted
            with numpy; pass `pyarrow.Table.from_pandas(df)` for categorical columns. A 1-D
            array or a flat list is one row. CatBoost models predict pandas frames of 50,000 rows
            or more through pyarrow when it is installed (faster, same results). CatBoost models with categorical features take their categories as strings or
            integers: a pandas DataFrame, a numpy object array, or a list (one row) or list of
            rows; see `categorical_features` and `catboost_hash`.
        raw_score : bool, default=False
            Whether to output raw scores (leaf values) instead of transformed predictions.
        n_jobs : int, default=-1
            Number of OpenMP threads to use. `-1` means using all available logical cores.
        strategy : {'auto', 'serial', 'rows'}, default='auto'
            How the work is spread across threads. All strategies return identical results;
            only speed differs. `'serial'` runs on the calling thread (lowest latency for a
            few rows), `'rows'` splits rows between threads. `'auto'` uses `'serial'` below
            8 rows or with one thread, `'rows'` otherwise.
        validate_features : bool, optional
            Check that the column names of a pandas DataFrame or Arrow data start with
            `feature_names`, in the same order, and raise `ValueError` otherwise (a reordered or
            missing column would silently give other predictions). Default: the model's
            `validate_features` (True unless set with `load(..., validate_features=False)`).
            numpy arrays have no names and are always used by position, as are frames without
            column names and models without feature names. The check costs about 1-5 µs per call.

        Returns
        -------
        preds : ndarray, shape (n_samples,) or (n_samples, n_classes)
            The predicted values. Multiclass models return one column per class
            (probabilities, or margins with `raw_score=True`), except XGBoost
            `multi:softmax`, which returns the predicted class index.

        Examples
        --------
        ```python
        import numpy as np
        import lignum

        model = lignum.load("model.bin")
        X = np.ascontiguousarray(np.array([[1.2, 3.4], [5.6, 7.8]], dtype=np.float64))
        model.predict(X, n_jobs=2)
        # array([0.45, 0.89])
        ```
        """
        validate = (self.validate_features if validate_features is None else validate_features) and bool(self._feature_names)
        if not isinstance(X, np.ndarray) and not _is_pandas(X):
            # Arrow PyCapsule interface: pyarrow, polars, duckdb, nanoarrow... (no pyarrow needed here).
            if hasattr(X, "__arrow_c_array__"):
                schema, array = X.__arrow_c_array__()
                return self._c_model.predict_arrow(schema, array, raw_score, n_jobs, strategy, validate)
            if hasattr(X, "__arrow_c_stream__"):
                return self._c_model.predict_arrow_stream(X.__arrow_c_stream__(), raw_score, n_jobs, strategy, validate)
        if validate and _is_pandas(X) and X.ndim == 2:
            self._check_columns(X.columns)
        if self._categorical:
            if _is_pandas(X) and X.ndim == 2 and len(X) >= _ARROW_MIN_ROWS:
                table = _arrow_table(X)
                if table is not None:
                    try:
                        return self._c_model.predict_arrow_stream(table.__arrow_c_stream__(), raw_score, n_jobs, strategy)
                    except ValueError:
                        pass  # e.g. a column Arrow typed differently: the pandas path decides
            return self._c_model.predict(self._catboost_matrix(X), raw_score, n_jobs, strategy)
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        if X_arr.ndim == 1:  # one row
            X_arr = X_arr.reshape(1, -1)
        return self._c_model.predict(X_arr, raw_score, n_jobs, strategy)

    @property
    def n_outputs(self) -> int:
        """Number of model outputs: the number of classes for multiclass models, otherwise 1."""
        return self._c_model.n_outputs

    def compile(self, compiler: str = "auto", flags: str = "", *, library: str | os.PathLike | None = None,
                jobs: int = 0, block_trees: int = 16, fallback: bool = False,
                autotune: np.ndarray | None = None) -> dict:
        """
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

        Parameters
        ----------
        compiler : str, default='auto'
            C compiler. `'auto'` uses `cc` on Linux and macOS; on Windows `cl` if it is on PATH,
            otherwise the newest Visual Studio with C++ tools (found with vswhere), then `gcc`
            (MinGW) and `clang`. MSVC-style compilers (`cl`, `clang-cl`) are driven with their
            own options.
        flags : str, default=''
            Optimisation flags; empty means `-O2` (gcc, clang) or `/O2` (MSVC).
        library : path, optional
            Write the library here instead of the cache, e.g. next to the `.bin` model, so
            production can attach it with `lignum.load(model_path, compiled=library)` without a
            compiler. A file path, or a folder (an existing one, or a path ending in `/`) that
            gets `lignum_<model fingerprint>_<platform>.so` (`.dll` on Windows); one folder can
            hold the libraries of several models and platforms. Always compiled anew.
        jobs : int, default=0
            Source files compiled in parallel; 0 means one per CPU core.
        block_trees : int, default=16
            Trees per generated function; larger blocks compile slower and run about the same.
        fallback : bool, default=False
            When the compiler is missing or fails, warn and keep the interpreter instead of
            raising. The returned dict then has `compiled=False` and the `error`.
        autotune : array, optional
            Sample of the data to predict on; after compiling, `autotune(sample)` picks the
            batch size from which the compiled trees are used (result under `autotune`).

        Returns
        -------
        dict
            `compiled`, timings (`codegen_ms`, `compile_ms`, `load_ms`), `source_bytes`,
            `cached`, the `library` path and the `compiler` used (empty for a cached library).

        Examples
        --------
        ```python
        model = lignum.load("model.json", format="lightgbm")
        model.save("model.bin")
        model.compile(library="model.so")      # build machine: needs a C compiler

        model = lignum.load("model.bin", compiled="model.so")  # production: no compiler
        ```
        """
        try:
            stats = dict(self._c_model.compile(compiler, flags, jobs, block_trees,
                                               "" if library is None else os.fspath(library)))
        except RuntimeError as error:
            if not fallback:
                raise
            warnings.warn(f"lignum: compiled mode unavailable, using the interpreter ({error})", RuntimeWarning, stacklevel=2)
            return {"compiled": False, "error": str(error)}
        stats["compiled"] = True
        if autotune is not None:
            stats["autotune"] = self.autotune(autotune)
        return stats

    def autotune(self, X: np.ndarray, n_jobs: int = -1, budget: float = 1.0) -> dict:
        """
        Measure the interpreter and the compiled trees on a sample and use each where it is faster.

        Which one wins depends on the model, the data and the CPU: compiled code is usually faster,
        but on data whose splits are hard to predict the branch-free interpreter can win, mostly
        for small batches. Batches of 1, 4, 16, ... rows up to the sample size are timed with
        both (about `budget` seconds in total), and `compiled_min_rows` is set to the smallest
        size from which the compiled trees win at every larger size (`sys.maxsize`: never).
        Predictions are identical either way.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Representative rows (a few thousand are enough).
        n_jobs : int, default=-1
            Threads, as in `predict`.
        budget : float, default=1.0
            Approximate time to spend measuring, in seconds.

        Returns
        -------
        dict
            `compiled_min_rows` and `timings`: {rows: (interpreter_us, compiled_us)}, median per call.
        """
        if not self.uses_compiled:
            raise RuntimeError("autotune needs compiled trees: call compile() or load(..., compiled=...) first.")
        X = np.ascontiguousarray(X, dtype=np.float64)
        sizes = []
        n = 1
        while n < len(X):
            sizes.append(n)
            n *= 4
        sizes.append(len(X))
        per_measurement = budget / (2 * len(sizes))

        def median_us(batch, min_rows):
            self._c_model.compiled_min_rows = min_rows
            self.predict(batch, n_jobs=n_jobs)  # warm up
            times = []
            deadline = time.perf_counter() + per_measurement
            while len(times) < 3 or (time.perf_counter() < deadline and len(times) < 10000):
                t0 = time.perf_counter()
                self.predict(batch, n_jobs=n_jobs)
                times.append(time.perf_counter() - t0)
            return statistics.median(times) * 1e6

        timings = {}
        try:
            for rows in sizes:
                batch = X[:rows]
                timings[rows] = (median_us(batch, sys.maxsize), median_us(batch, 0))
        finally:
            self._c_model.compiled_min_rows = 0
        threshold = sys.maxsize
        for rows in reversed(sizes):
            interpreter, compiled = timings[rows]
            if compiled > interpreter:
                break
            threshold = rows
        self._c_model.compiled_min_rows = threshold
        return {"compiled_min_rows": threshold, "timings": timings}

    @property
    def compiled_min_rows(self) -> int:
        """Batches with fewer rows use the interpreter even when compiled (set by `autotune`)."""
        return self._c_model.compiled_min_rows

    @compiled_min_rows.setter
    def compiled_min_rows(self, rows: int) -> None:
        self._c_model.compiled_min_rows = rows

    def load_compiled(self, library: str | os.PathLike) -> None:
        """
        Attach a library built earlier by `compile(library=...)` for this model: the file, or
        the folder it was written to (the file for this model and platform is picked).

        The library records which model it was generated from and the lignum ABI; a library
        for another model or lignum version raises `RuntimeError`. Loading runs the library's
        native code, so only load libraries you trust (like a pickle).
        """
        self._c_model.load_compiled(os.fspath(library))

    def decompile(self) -> None:
        """Go back to the interpreter."""
        self._c_model.decompile()

    @property
    def uses_compiled(self) -> bool:
        """Whether predictions use compiled trees."""
        return self._c_model.uses_compiled

    @property
    def compiled_library(self) -> str | None:
        """Path of the compiled library the model uses (in the cache, or where it was written or
        loaded from), or None when it uses the interpreter."""
        return self._c_model.compiled_library

    @property
    def uses_binned(self) -> bool:
        """Whether batches use the binned kernel (False when disabled or unsupported by the model)."""
        return self._c_model.uses_binned

    @property
    def categorical_features(self) -> list[int]:
        """Columns that hold categories (CatBoost models with categorical features), in order."""
        return list(self._c_model.categorical_features)

    @property
    def n_features(self) -> int:
        """Minimum number of columns the input must have."""
        return self._c_model.n_features

    def save(self, filepath: str) -> None:
        """
        Save the current model to the Lignum binary format.
        
        This format is optimized for fast, memory-mapped (mmap) loading in 
        subsequent sessions.

        Parameters
        ----------
        filepath : str
            Destination file path for the compiled `.bin` model.

        Examples
        --------
        ```python
        model.save("optimized_model.bin")
        ```
        """
        self._c_model.save(str(filepath))


def load(filepath: str, format: str = "auto", binned: bool = True,
         compiled: str | os.PathLike | None = None, validate_features: bool = True) -> Model:
    """
    Load a trained model: LightGBM or XGBoost files, or a lignum `.bin`.

    Parameters
    ----------
    filepath : str
        Path to the model file on disk.
    format : {'auto', 'lignum', 'lightgbm', 'xgboost', 'catboost'}, default='auto'
        Library of the model. The file's own bytes tell which of its formats it is, so 'auto'
        loads a lignum `.bin`, a LightGBM `model.txt` (`save_model`) or `dump_model()` JSON,
        an XGBoost `.json` or `.ubj` (`save_model`) and a CatBoost JSON
        (`save_model(..., format="json")`). Naming the library only adds a check.
    binned : bool, default=True
        Prepare the binned kernel: inputs are mapped once per row to integer bin codes and
        categorical decisions are precomputed, so the tree walk compares small integers. Results are identical either way; False keeps
        only the direct traversal (lower memory use).
    compiled : path, optional
        Library written by `Model.compile(library=...)` for this model, or the folder it was
        written to (the file for this model and platform is picked); predictions then use the
        compiled trees without compiling again. Needs `binned=True`. A library for another
        model or lignum version raises `RuntimeError`. Only load libraries you trust.
    validate_features : bool, default=True
        Default of `Model.predict(..., validate_features=...)`: check the column names of pandas
        and Arrow inputs against the model's feature names. Also settable later as
        `model.validate_features`.

    Returns
    -------
    model : lignum.Model
        The loaded model, ready for immediate inference.

    Raises
    ------
    FileNotFoundError
        If the specified `filepath` does not exist.

    Examples
    --------
    ```python
    import lignum

    model = lignum.load("xgboost_model.json", format="xgboost")
    ```
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    model = Model(_core.load(str(filepath), format, binned), validate_features)
    if compiled is not None:
        model.load_compiled(compiled)
    return model