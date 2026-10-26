"""Bit-exact parity on real datasets (UCI / scikit-learn), LightGBM, XGBoost and CatBoost.

Datasets are downloaded once into benchmarks/.cache/datasets; tests are skipped when that fails.
Categorical columns are pandas categories for the libraries and category codes for lignum
(CatBoost: string categories for both).
"""

import functools
import json
import sys
import urllib.error
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

import lignum as lg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import datasets  # noqa: E402

N_ROUNDS = 80
MAX_ROWS = 60_000


@functools.cache
def load(name):
    try:
        if name == "covtype":
            return datasets.covtype(n_rows=MAX_ROWS)
        return datasets.LOADERS[name]()
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as e:  # network or archive problems only
        pytest.skip(f"{name} unavailable: {type(e).__name__}: {e}")


def lgb_params(d):
    params = {"num_leaves": 63, "learning_rate": 0.1, "verbose": -1, "seed": 7, "max_cat_to_onehot": 4,
              "max_cat_threshold": 64, "min_data_per_group": 20}
    if d.task == "regression":
        params["objective"] = "regression"
    elif d.task == "binary":
        params["objective"] = "binary"
    else:
        params.update(objective="multiclass", num_class=d.n_classes)
    return params


def xgb_params(d):
    params = {"max_depth": 6, "learning_rate": 0.1, "tree_method": "hist", "seed": 7, "max_cat_to_onehot": 4}
    if d.task == "regression":
        params["objective"] = "reg:squarederror"
    elif d.task == "binary":
        params["objective"] = "binary:logistic"
    else:
        params.update(objective="multi:softprob", num_class=d.n_classes)
    return params


def rows(d):
    n = min(len(d.y), MAX_ROWS)
    return d.frame.iloc[:n], d.X[:n], d.y[:n]


@pytest.mark.parametrize("name", ["california", "covtype", "adult", "bank", "diabetes"])
def test_lightgbm_real_data(tmp_path, name):
    d = load(name)
    frame, X, y = rows(d)
    split = int(len(y) * 0.8)
    booster = lgb.train(lgb_params(d), lgb.Dataset(frame.iloc[:split], y[:split]), num_boost_round=N_ROUNDS)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(booster.dump_model()))
    model = lg.load(str(path), format="lightgbm")

    # All rows: training rows hit split thresholds exactly, held-out rows contain unseen values.
    np.testing.assert_array_equal(model.predict(X, raw_score=True), booster.predict(frame, raw_score=True))
    np.testing.assert_array_equal(model.predict(X), booster.predict(frame))
    for i in range(0, 200, 7):
        np.testing.assert_array_equal(model.predict(X[i:i + 1]), booster.predict(frame.iloc[i:i + 1]))


@pytest.mark.parametrize("name", ["california", "covtype", "adult", "bank", "diabetes"])
def test_xgboost_real_data(tmp_path, name):
    d = load(name)
    frame, X, y = rows(d)
    split = int(len(y) * 0.8)
    train = xgb.DMatrix(frame.iloc[:split], y[:split], enable_categorical=True)
    booster = xgb.train(xgb_params(d), train, num_boost_round=N_ROUNDS)
    path = tmp_path / "model.json"
    booster.save_model(str(path))
    model = lg.load(str(path), format="xgboost")

    full = xgb.DMatrix(frame, enable_categorical=True)
    np.testing.assert_array_equal(model.predict(X, raw_score=True), booster.predict(full, output_margin=True))
    np.testing.assert_array_equal(model.predict(X), booster.predict(full))
    for i in range(0, 200, 7):
        np.testing.assert_array_equal(model.predict(X[i:i + 1]), booster.predict(xgb.DMatrix(frame.iloc[i:i + 1], enable_categorical=True)))


@pytest.mark.parametrize("name", ["california", "covtype", "adult", "bank", "diabetes"])
def test_catboost_real_data(tmp_path, kernel, name):
    if kernel is not True:
        pytest.skip("CatBoost models have a single kernel")
    cb = pytest.importorskip("catboost")
    d = load(name)
    n = min(len(d.y), MAX_ROWS)
    frame, y = datasets.catboost_frame(d).iloc[:n], d.y[:n]
    cats = [frame.columns[j] for j in d.categorical]
    split = int(n * 0.8)
    params = {"iterations": N_ROUNDS, "depth": 6, "random_seed": 7, "verbose": False, "thread_count": 4}
    if d.task == "multiclass":
        params["loss_function"] = "MultiClass"
    else:
        params["loss_function"] = "Logloss" if d.task == "binary" else "RMSE"
    booster = cb.CatBoost(params)
    booster.fit(cb.Pool(frame.iloc[:split], y[:split], cat_features=cats))
    booster.save_model(str(tmp_path / "model.cbm"))
    model = lg.load(str(tmp_path / "model.cbm"))
    assert model.categorical_features == list(d.categorical)

    # Training rows hit the float borders and the CTR tables, held-out rows bring unseen combinations.
    pool = cb.Pool(frame, cat_features=cats)
    raw = booster.predict(pool, prediction_type="RawFormulaVal")
    np.testing.assert_array_equal(model.predict(frame, raw_score=True), raw)
    expected = booster.predict(pool, prediction_type="Probability" if d.task != "regression" else "RawFormulaVal")
    if d.task == "binary":
        expected = expected[:, 1]
    np.testing.assert_array_equal(model.predict(frame), expected)
    # Object columns (strings) and Arrow (dictionary columns) give the same hashes as categories.
    np.testing.assert_array_equal(model.predict(frame.astype({c: object for c in cats}), raw_score=True), raw)
    pa = pytest.importorskip("pyarrow")
    np.testing.assert_array_equal(model.predict(pa.Table.from_pandas(frame, preserve_index=False), raw_score=True), raw)
    for i in range(0, 200, 7):
        np.testing.assert_array_equal(model.predict(frame.iloc[i:i + 1], raw_score=True), raw[i:i + 1])
