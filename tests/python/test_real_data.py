"""Bit-exact parity on real datasets (UCI / scikit-learn), LightGBM and XGBoost.

Datasets are downloaded once into benchmarks/.cache/datasets; tests are skipped when that fails.
Categorical columns are pandas categories for the libraries and category codes for lignum.
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
