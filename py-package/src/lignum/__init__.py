import os
import numpy as np

from . import _core

__all__ = ["Model", "load"]
__version__ = "0.1.0"


class Model:
    """
    Compiled Lignum decision tree model.
    """

    def __init__(self, _c_model):
        self._c_model = _c_model

    def predict(self, X: np.ndarray, raw_score: bool = False, n_jobs: int = -1) -> np.ndarray:
        """
        Predict values for the given input data.

        For optimal performance, ensure X is a C-contiguous float64 array.
        Otherwise, an internal copy will be triggered.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        raw_score : bool, deafault=False
            Whether to predict raw scores.
        n_jobs : int, default=-1
            Number of OpenMP threads to run. -1 means using all available cores.

        Returns
        -------
        preds : ndarray, shape (n_samples,)
            The predicted values (raw leaves sum).
        """
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        return self._c_model.predict(X_arr, raw_score, n_jobs)

    def save(self, filepath: str) -> None:
        """
        Save the model to Lignum binary format for fast mmap-based loading.

        Parameters
        ----------
        filepath : str
            Destination path for the .bin file.
        """
        self._c_model.save(str(filepath))


def load(filepath: str, format: str = "auto") -> Model:
    """
    Load a trained model from a JSON or Lignum binary file.

    Parameters
    ----------
    filepath : str
        Path to the model file.
    format : {'auto', 'lignum', 'binary', 'lightgbm', 'xgboost'}, default='auto'
        Format of the model file. If 'auto' then .bin file is required.

    Returns
    -------
    model : lignum.Model
        The loaded model ready for inference.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    c_model = _core.load(str(filepath), format)
    return Model(c_model)