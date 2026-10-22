"""Bit-exact parity for XGBoost categorical splits.

XGBoost semantics differ from LightGBM: the category is int(float32(x)); categories in the stored
set go RIGHT, invalid ones (negative or too large) go LEFT, and NaN follows default_left.
Inputs are category codes in a float matrix (what XGBoost's DMatrix holds for a categorical column).
"""

import json

import numpy as np
import pytest
import xgboost as xgb

import lignum as lg

N_ROWS = 6000


def make_data(rng, cardinalities, n_numeric=3):
    cols, signal = [], np.zeros(N_ROWS)
    for k in cardinalities:
        c = rng.integers(0, k, N_ROWS)
        signal += rng.normal(size=k)[c]
        cols.append(c.astype(float))
    for _ in range(n_numeric):
        x = rng.normal(size=N_ROWS)
        signal += 0.5 * x
        cols.append(x)
    X = np.column_stack(cols)
    X[rng.random(X.shape) < 0.03] = np.nan
    return X, signal


def feature_types(n_cat, n_total):
    return ["c"] * n_cat + ["q"] * (n_total - n_cat)


def dmatrix(X, n_cat, label=None):
    return xgb.DMatrix(X, label=label, feature_types=feature_types(n_cat, X.shape[1]), enable_categorical=True)


def stress(X, n_cat, cardinalities, rng):
    special = np.array([np.nan, -0.5, -1.0, -3.0, -1e9, 0.3, 1.5, 2.9999999999, 2.99999, 63.0, 64.0,
                        65.5, 999.0, 16777215.0, 16777216.0, 3e9, 1e30])
    X_s = X.copy()
    for j in range(n_cat):
        mask = rng.random(len(X_s)) < 0.15
        X_s[mask, j] = rng.choice(special, size=int(mask.sum()))
        unseen = rng.random(len(X_s)) < 0.05
        X_s[unseen, j] = cardinalities[j] + rng.integers(0, 5, int(unseen.sum()))
    return X_s


def to_lignum(booster, tmp_path):
    path = tmp_path / "model.json"
    booster.save_model(str(path))
    return lg.load(str(path), format="xgboost")


def assert_exact(model, booster, X, n_cat):
    d = dmatrix(X, n_cat)
    for strategy in ["serial", "rows"]:
        np.testing.assert_array_equal(model.predict(X, raw_score=True, strategy=strategy), booster.predict(d, output_margin=True))
        np.testing.assert_array_equal(model.predict(X, strategy=strategy), booster.predict(d))
    for i in range(30):
        np.testing.assert_array_equal(model.predict(X[i:i + 1]), booster.predict(dmatrix(X[i:i + 1], n_cat)))


def has_categorical_splits(path):
    trees = json.load(open(path))["learner"]["gradient_booster"]["model"]["trees"]
    return any(t.get("categories_nodes") for t in trees)


@pytest.mark.parametrize("cardinality", [2, 5, 63, 64, 200, 1000])
@pytest.mark.parametrize("max_cat_to_onehot", [1, 64])
def test_single_categorical_feature(tmp_path, cardinality, max_cat_to_onehot):
    rng = np.random.default_rng(cardinality + max_cat_to_onehot)
    X, signal = make_data(rng, [cardinality], n_numeric=1)
    params = {"objective": "reg:squarederror", "max_depth": 4, "tree_method": "hist", "max_cat_to_onehot": max_cat_to_onehot}
    booster = xgb.train(params, dmatrix(X, 1, signal), num_boost_round=30)
    model = to_lignum(booster, tmp_path)
    assert has_categorical_splits(tmp_path / "model.json")
    assert_exact(model, booster, stress(X, 1, [cardinality], rng), 1)


@pytest.mark.parametrize("objective", ["binary:logistic", "reg:squarederror"])
def test_mixed_numeric_and_categorical(tmp_path, objective):
    rng = np.random.default_rng(11)
    cards = [3, 30, 300]
    X, signal = make_data(rng, cards)
    y = (signal > np.median(signal)).astype(float) if objective.startswith("binary") else signal
    booster = xgb.train({"objective": objective, "max_depth": 6, "tree_method": "hist", "max_cat_to_onehot": 4},
                        dmatrix(X, len(cards), y), num_boost_round=60)
    assert_exact(to_lignum(booster, tmp_path), booster, stress(X, len(cards), cards, rng), len(cards))


def test_multiclass_with_categorical(tmp_path):
    rng = np.random.default_rng(12)
    cards = [10, 150]
    X, signal = make_data(rng, cards)
    y = np.digitize(signal, np.quantile(signal, [0.25, 0.5, 0.75]))
    booster = xgb.train({"objective": "multi:softprob", "num_class": 4, "max_depth": 4, "tree_method": "hist"},
                        dmatrix(X, len(cards), y), num_boost_round=15)
    assert_exact(to_lignum(booster, tmp_path), booster, stress(X, len(cards), cards, rng), len(cards))


def test_binary_roundtrip(tmp_path):
    rng = np.random.default_rng(13)
    cards = [5, 500]
    X, signal = make_data(rng, cards)
    booster = xgb.train({"objective": "reg:squarederror", "max_depth": 5, "tree_method": "hist"}, dmatrix(X, len(cards), signal), num_boost_round=20)
    model = to_lignum(booster, tmp_path)
    model.save(str(tmp_path / "model.bin"))
    restored = lg.load(str(tmp_path / "model.bin"))
    X_s = stress(X, len(cards), cards, rng)
    np.testing.assert_array_equal(restored.predict(X_s), booster.predict(dmatrix(X_s, len(cards))))
