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

    def predict(self, X: np.ndarray, raw_score: bool = False, n_jobs: int = -1) -> np.ndarray:
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

        Returns
        -------
        preds : ndarray, shape (n_samples,)
            The predicted values.

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
        return self._c_model.predict(X_arr, raw_score, n_jobs)

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


def load(filepath: str, format: str = "auto") -> Model:
    """
    Load a trained model from a JSON or Lignum binary file.

    Parameters
    ----------
    filepath : str
        Path to the model file on disk.
    format : {'auto', 'lignum', 'binary', 'lightgbm', 'xgboost'}, default='auto'
        Expected format of the model file. If set to 'auto', the function will 
        attempt to infer the format from the file contents.

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

    c_model = _core.load(str(filepath), format)
    return Model(c_model)