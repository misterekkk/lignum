"""Bit-exact parity with LightGBM and XGBoost on data that stresses split semantics.

Inputs include NaNs, exact zeros, values in LightGBM's zero band (|x| <= 1e-35) and values
placed exactly on every split threshold, which is where precision bugs show up.
"""

import json

import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

import lignum as lg

N_ROWS = 4000
N_FEATURES = 8


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N_ROWS, N_FEATURES))
    y = 2 * X[:, 0] + np.sin(X[:, 1]) + 0.1 * rng.normal(size=N_ROWS)
    X_nan = X.copy()
    X_nan[rng.random(X.shape) < 0.1] = np.nan
    X_zero = np.where(rng.random(X.shape) < 0.1, 0.0, X_nan)
    return {"X": X, "X_nan": X_nan, "X_zero": X_zero, "y": y, "rng": rng}


def stress_inputs(X, thresholds, rng):
    """Rows hitting every threshold exactly, plus zeros and the LightGBM zero band."""
    X_hit = X.copy()
    for k, (feature, value) in enumerate(thresholds):
        X_hit[k % len(X_hit), feature] = value
    special = np.array([0.0, -0.0, 1e-35, -1e-35, float(np.float32(1e-35)), -float(np.float32(1e-35)), 1e-30, np.nan])
    mask = rng.random(X.shape) < 0.05
    X_hit[mask] = rng.choice(special, size=int(mask.sum()))
    return X_hit


def lgb_thresholds(booster):
    out = []

    def walk(node):
        if "split_feature" in node:
            out.append((node["split_feature"], node["threshold"]))
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in booster.dump_model()["tree_info"]:
        walk(tree["tree_structure"])
    return out


def xgb_thresholds(path):
    model = json.load(open(path))
    out = []
    for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
        for i, left in enumerate(tree["left_children"]):
            if left != -1:
                out.append((tree["split_indices"][i], tree["split_conditions"][i]))
    return out


def lgb_to_lignum(booster, tmp_path):
    path = tmp_path / "lgb.json"
    path.write_text(json.dumps(booster.dump_model()))
    return lg.load(str(path), format="lightgbm")


LGB_CASES = [
    ("regression", {}),
    ("binary", {}),
    ("binary", {"sigmoid": 0.7}),
    ("poisson", {}),
    ("gamma", {}),
    ("tweedie", {}),
    ("cross_entropy", {}),
    ("cross_entropy_lambda", {}),
    ("regression", {"use_missing": False}),
    ("regression", {"zero_as_missing": True}),
    ("binary", {"zero_as_missing": True}),
]


def lgb_target(objective, y):
    if objective in ("binary", "cross_entropy", "cross_entropy_lambda"):
        return (y > 0).astype(float)
    if objective in ("poisson", "gamma", "tweedie"):
        return np.abs(y) + 0.1
    return y


@pytest.mark.parametrize("train_on", ["X", "X_nan", "X_zero"])
@pytest.mark.parametrize("objective, extra", LGB_CASES)
def test_lightgbm_bit_exact(tmp_path, data, objective, extra, train_on):
    X_train = data[train_on]
    params = {"objective": objective, "num_leaves": 31, "verbose": -1, **extra}
    booster = lgb.train(params, lgb.Dataset(X_train, lgb_target(objective, data["y"])), num_boost_round=60)
    model = lgb_to_lignum(booster, tmp_path)

    X_test = stress_inputs(X_train, lgb_thresholds(booster), data["rng"])
    np.testing.assert_array_equal(model.predict(X_test, raw_score=True), booster.predict(X_test, raw_score=True))
    np.testing.assert_array_equal(model.predict(X_test), booster.predict(X_test))


XGB_CASES = [
    ("reg:squarederror", {}),
    ("reg:squarederror", {"base_score": 0.0}),
    ("binary:logistic", {}),
    ("binary:logistic", {"base_score": 0.3}),
    ("binary:logitraw", {}),
    ("count:poisson", {}),
    ("reg:gamma", {}),
    ("reg:tweedie", {}),
]


def xgb_target(objective, y):
    if objective.startswith("binary"):
        return (y > 0).astype(float)
    if objective in ("count:poisson", "reg:gamma", "reg:tweedie"):
        return np.abs(y) + 0.1
    return y


@pytest.mark.parametrize("train_on", ["X", "X_nan"])
@pytest.mark.parametrize("objective, extra", XGB_CASES)
def test_xgboost_bit_exact(tmp_path, data, objective, extra, train_on):
    X_train = data[train_on]
    params = {"objective": objective, "max_depth": 5, **extra}
    booster = xgb.train(params, xgb.DMatrix(X_train, xgb_target(objective, data["y"])), num_boost_round=60)
    path = tmp_path / "xgb.json"
    booster.save_model(str(path))
    model = lg.load(str(path), format="xgboost")

    X_test = stress_inputs(X_train, xgb_thresholds(path), data["rng"])
    dtest = xgb.DMatrix(X_test)
    np.testing.assert_array_equal(model.predict(X_test, raw_score=True), booster.predict(dtest, output_margin=True))
    np.testing.assert_array_equal(model.predict(X_test), booster.predict(dtest))


@pytest.mark.parametrize("fmt", ["lightgbm", "xgboost"])
def test_binary_roundtrip_keeps_semantics(tmp_path, data, fmt):
    if fmt == "lightgbm":
        booster = lgb.train({"objective": "binary", "zero_as_missing": True, "verbose": -1},
                            lgb.Dataset(data["X_zero"], (data["y"] > 0).astype(float)), num_boost_round=30)
        model = lgb_to_lignum(booster, tmp_path)
        expected = booster.predict(data["X_zero"])
    else:
        booster = xgb.train({"objective": "binary:logistic", "max_depth": 4},
                            xgb.DMatrix(data["X_nan"], (data["y"] > 0).astype(float)), num_boost_round=30)
        path = tmp_path / "xgb.json"
        booster.save_model(str(path))
        model = lg.load(str(path), format="xgboost")
        expected = booster.predict(xgb.DMatrix(data["X_zero"]))

    bin_path = tmp_path / "model.bin"
    model.save(str(bin_path))
    restored = lg.load(str(bin_path))
    np.testing.assert_array_equal(restored.predict(data["X_zero"]), expected)


def test_rejects_too_few_features(tmp_path, data):
    booster = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(data["X"], data["y"]), num_boost_round=5)
    model = lgb_to_lignum(booster, tmp_path)
    with pytest.raises(ValueError, match="requires at least"):
        model.predict(data["X"][:, :1])


@pytest.mark.parametrize("n_rows", [0, 1, 3, 4, 5, 257])
def test_small_batches(tmp_path, data, n_rows):
    booster = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(data["X"], data["y"]), num_boost_round=20)
    model = lgb_to_lignum(booster, tmp_path)
    X = data["X_nan"][:n_rows]
    expected = booster.predict(X, raw_score=True) if n_rows else np.empty(0)
    np.testing.assert_array_equal(model.predict(X, raw_score=True), expected)


@pytest.mark.parametrize("fmt", ["lightgbm", "xgboost"])
@pytest.mark.parametrize("strategy", ["auto", "serial", "rows"])
@pytest.mark.parametrize("n_jobs", [1, 3, -1])
def test_strategies_are_bit_identical(tmp_path, data, fmt, strategy, n_jobs):
    y = (data["y"] > 0).astype(float)
    if fmt == "lightgbm":
        booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(data["X_nan"], y), num_boost_round=40)
        model = lgb_to_lignum(booster, tmp_path)
        reference = lambda X: booster.predict(X)
    else:
        booster = xgb.train({"objective": "binary:logistic", "max_depth": 5}, xgb.DMatrix(data["X_nan"], y), num_boost_round=40)
        path = tmp_path / "xgb.json"
        booster.save_model(str(path))
        model = lg.load(str(path), format="xgboost")
        reference = lambda X: booster.predict(xgb.DMatrix(X))

    for n_rows in [1, 3, 7, 8, 13, 100, 1000, N_ROWS]:
        X = data["X_nan"][:n_rows]
        np.testing.assert_array_equal(model.predict(X, n_jobs=n_jobs, strategy=strategy), reference(X))


def test_rejects_unknown_strategy(tmp_path, data):
    booster = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(data["X"], data["y"]), num_boost_round=2)
    model = lgb_to_lignum(booster, tmp_path)
    with pytest.raises(ValueError, match="strategy must be one of"):
        model.predict(data["X"][:5], strategy="fastest")
