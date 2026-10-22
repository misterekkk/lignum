"""
Lignum: A fast decision tree inference library.

This module provides the core interface for loading and running predictions
with compiled Lignum models. It bypasses standard parsers by using a native 
C++ engine for direct JSON inference and mmap-based binary loading.
"""

import os
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

    def compile(self, compiler: str = "cc", flags: str = "-O2", work_dir: str = "", jobs: int = 0,
                block_trees: int = 16) -> dict:
        """
        Experimental: compile the trees to native code with the system C compiler.

        Each tree becomes C code with constant thresholds and leaf values. Trees are grouped in
        blocks of `block_trees`, and a block is walked for a whole chunk of rows before the next
        one; binning and threads stay in lignum and every row sums its trees in model order, so
        predictions are unchanged (bit-identical). Libraries are cached
        by content in `work_dir` (default: a `lignum_compiled` folder in the temp directory).
        Linux and macOS only.

        Returns timings: codegen_ms, compile_ms, load_ms, source_bytes, cached, library.
        """
        return dict(self._c_model.compile(compiler, flags, str(work_dir), jobs, block_trees))

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


def load(filepath: str, format: str = "auto", binned: bool = True) -> Model:
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

    c_model = _core.load(str(filepath), format, binned)
    return Model(c_model)