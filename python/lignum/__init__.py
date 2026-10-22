"""
Lignum: A fast decision tree inference library.

This module provides the core interface for loading and running predictions
with compiled Lignum models. It bypasses standard parsers by using a native 
C++ engine for direct JSON inference and mmap-based binary loading.
"""

import os
import statistics
import sys
import time
import warnings

import numpy as np

from . import _core

__all__ = ["Model", "load"]
__version__ = "0.1.0"


class Model:
    """
    Compiled Lignum decision tree model.
    
    This class wraps the underlying C++ model and provides a high-performance 
    interface for inference. Instances of this class should generally be created 
    using the `lignum.load()` function rather than instantiating directly.
    """

    def __init__(self, _c_model):
        self._c_model = _c_model

    def predict(self, X: np.ndarray, raw_score: bool = False, n_jobs: int = -1, strategy: str = "auto") -> np.ndarray:
        """
        Predict values for the given input data.

        For optimal performance, ensure `X` is a C-contiguous float64 array.
        Otherwise, an internal copy will be triggered, which may degrade performance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples to predict.
        raw_score : bool, default=False
            Whether to output raw scores (leaf values) instead of transformed predictions.
        n_jobs : int, default=-1
            Number of OpenMP threads to use. `-1` means using all available logical cores.
        strategy : {'auto', 'serial', 'rows'}, default='auto'
            How the work is spread across threads. All strategies return identical results;
            only speed differs. `'serial'` runs on the calling thread (lowest latency for a
            few rows), `'rows'` splits rows between threads. `'auto'` uses `'serial'` below
            8 rows or with one thread, `'rows'` otherwise.

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
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        return self._c_model.predict(X_arr, raw_score, n_jobs, strategy)

    @property
    def n_outputs(self) -> int:
        """Number of model outputs: the number of classes for multiclass models, otherwise 1."""
        return self._c_model.n_outputs

    def compile(self, compiler: str = "auto", flags: str = "", work_dir: str = "", jobs: int = 0,
                block_trees: int = 16, library: str | os.PathLike | None = None, fallback: bool = False,
                autotune: np.ndarray | None = None) -> dict:
        """
        Experimental: compile the trees to native code with the system C compiler.

        Each tree becomes C code with constant thresholds; leaves are read from per-block
        tables. Trees are grouped in blocks of `block_trees`, and a block is walked for a whole
        chunk of rows before the next one; binning of categorical features, threads and
        transforms stay in lignum and every row sums its trees in model order, so predictions
        are unchanged (bit-identical). Libraries are cached by content in `work_dir` (default:
        a `lignum_compiled` folder in the temp directory).

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
            Also write the compiled library here, e.g. next to the `.bin` model, so production
            can attach it with `lignum.load(model_path, compiled=library)` without a compiler.
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
            stats = dict(self._c_model.compile(compiler, flags, str(work_dir), jobs, block_trees,
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
        Attach a library built earlier by `compile(library=...)` for this model.

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
    def uses_binned(self) -> bool:
        """Whether batches use the binned kernel (False when disabled or unsupported by the model)."""
        return self._c_model.uses_binned

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
         compiled: str | os.PathLike | None = None) -> Model:
    """
    Load a trained model from a JSON or Lignum binary file.

    Parameters
    ----------
    filepath : str
        Path to the model file on disk.
    format : {'auto', 'lignum', 'binary', 'lightgbm', 'xgboost'}, default='auto'
        Expected format of the model file. If set to 'auto', the function will 
        attempt to infer the format from the file contents.
    binned : bool, default=True
        Prepare the binned kernel: inputs are mapped once per row to integer bin codes and
        categorical decisions are precomputed, so the tree walk compares small integers. Results are identical either way; False keeps
        only the direct traversal (lower memory use).
    compiled : path, optional
        Library written by `Model.compile(library=...)` for this model; predictions then use
        the compiled trees without compiling again. Needs `binned=True`. A library for another
        model or lignum version raises `RuntimeError`. Only load libraries you trust.

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

    model = Model(_core.load(str(filepath), format, binned))
    if compiled is not None:
        model.load_compiled(compiled)
    return model