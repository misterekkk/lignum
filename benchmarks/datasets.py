"""Real datasets for parity tests and benchmarks (downloaded once into benchmarks/.cache/datasets).

Each loader returns a `Dataset`:
  frame        pandas DataFrame with categorical columns as `category` dtype (what LightGBM /
               XGBoost are trained on),
  X            float64 matrix for lignum: numeric columns as is, categorical columns as category
               codes with NaN for missing values,
  y            target,
  task         "regression" | "binary" | "multiclass",
  categorical  indices of categorical columns.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path(__file__).resolve().parent / ".cache" / "datasets"
RAW = HOME / "raw"

UCI = {
    "adult": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "bank": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "diabetes": "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
}


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    task: str
    categorical: list[int]
    n_classes: int = 1


def _download(name: str, timeout: float = 120.0) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    url = UCI[name]
    path = RAW / url.rsplit("/", 1)[-1]
    if not path.exists():
        with urllib.request.urlopen(url, timeout=timeout) as response:
            path.write_bytes(response.read())
    return path


def _zip_member(path: Path, suffix: str) -> bytes:
    """Reads a file ending with `suffix` from a zip, looking into nested zips too."""
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(suffix) and "__MACOSX" not in name:
                return z.read(name)
        for name in z.namelist():
            if name.endswith(".zip"):
                try:
                    with zipfile.ZipFile(io.BytesIO(z.read(name))) as inner:
                        for inner_name in inner.namelist():
                            if inner_name.endswith(suffix) and "__MACOSX" not in inner_name:
                                return inner.read(inner_name)
                except zipfile.BadZipFile:
                    continue
    raise FileNotFoundError(f"{suffix} not found in {path}")


def _finish(name: str, frame: pd.DataFrame, y: np.ndarray, task: str, n_classes: int = 1) -> Dataset:
    categorical = [i for i, c in enumerate(frame.columns) if isinstance(frame[c].dtype, pd.CategoricalDtype)]
    cols = []
    for c in frame.columns:
        if isinstance(frame[c].dtype, pd.CategoricalDtype):
            codes = frame[c].cat.codes.to_numpy(dtype=np.float64)
            codes[codes < 0] = np.nan
            cols.append(codes)
        else:
            cols.append(frame[c].to_numpy(dtype=np.float64))
    X = np.ascontiguousarray(np.column_stack(cols))
    return Dataset(name, frame, X, np.asarray(y), task, categorical, n_classes)


def _text_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if pd.api.types.is_object_dtype(frame[c]) or pd.api.types.is_string_dtype(frame[c])]


def _categorize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for c in columns:
        frame[c] = frame[c].astype("category")
    return frame


def california() -> Dataset:
    from sklearn.datasets import fetch_california_housing

    d = fetch_california_housing(data_home=str(HOME), as_frame=True)
    return _finish("california", d.data, d.target.to_numpy(), "regression")


def covtype(n_rows: int | None = None) -> Dataset:
    from sklearn.datasets import fetch_covtype

    d = fetch_covtype(data_home=str(HOME), as_frame=True)
    frame, y = d.data, d.target.to_numpy() - 1
    if n_rows:
        frame, y = frame.iloc[:n_rows], y[:n_rows]
    return _finish("covtype", frame.reset_index(drop=True), y, "multiclass", 7)


def adult() -> Dataset:
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
               "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    frame = pd.read_csv(_download("adult"), header=None, names=columns, skipinitialspace=True, na_values="?")
    y = (frame.pop("income") == ">50K").to_numpy(dtype=float)
    cats = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
    return _finish("adult", _categorize(frame, cats), y, "binary")


def bank() -> Dataset:
    frame = pd.read_csv(io.BytesIO(_zip_member(_download("bank"), "bank-full.csv")), sep=";")
    y = (frame.pop("y") == "yes").to_numpy(dtype=float)
    return _finish("bank", _categorize(frame, _text_columns(frame)), y, "binary")


def diabetes() -> Dataset:
    """100k hospital encounters; ICD diagnosis codes give categories with hundreds of values."""
    frame = pd.read_csv(io.BytesIO(_zip_member(_download("diabetes"), "diabetic_data.csv")), na_values="?", low_memory=False)
    y = (frame.pop("readmitted") == "<30").to_numpy(dtype=float)
    frame = frame.drop(columns=["encounter_id", "patient_nbr"])
    cats = _text_columns(frame) + ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    return _finish("diabetes", _categorize(frame, cats), y, "binary")


def catboost_frame(d: Dataset) -> pd.DataFrame:
    """The frame as CatBoost takes it: categorical columns as string categories, missing values as
    the category "NA" (CatBoost rejects NaN in categorical features)."""
    frame = d.frame.copy()
    for j in d.categorical:
        c = frame.columns[j]
        frame[c] = frame[c].astype(object).where(frame[c].notna(), "NA").astype(str).astype("category")
    return frame


def synthetic(n_rows: int = 300_000) -> Dataset:
    """bench.py's synthetic data (make_classification, 20 features, 5% NaN), binary: the same
    data the LightGBM / XGBoost numbers use."""
    import bench

    X, y = bench.make_data(n_rows, 20)
    return _finish("synthetic", pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])]), y, "binary")


def synthetic_cat(n_rows: int = 300_000) -> Dataset:
    """bench.py's categorical data (lightgbm-cat / xgboost-cat): 5 categorical columns of
    cardinality 10 to 5000 (Zipf-skewed) and 15 numeric ones, 5% NaN, binary."""
    import bench

    X, y = bench.make_categorical_data(n_rows)
    n_cat = len(bench.CAT_CARDINALITIES)
    columns = {}
    for j in range(X.shape[1]):
        if j < n_cat:
            codes = pd.Series(X[:, j])
            columns[f"c{j}"] = codes.map(lambda v: f"v{int(v)}", na_action="ignore").astype("category")
        else:
            columns[f"f{j}"] = X[:, j]
    return _finish("synthetic-cat", pd.DataFrame(columns), y, "binary")


LOADERS = {"california": california, "covtype": covtype, "adult": adult, "bank": bank, "diabetes": diabetes,
           "synthetic": synthetic, "synthetic-cat": synthetic_cat}
