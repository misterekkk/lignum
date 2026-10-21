"""Bit-exact parity for LightGBM categorical splits.

Covers every encoding lignum uses for a category set (64-bit mask, contiguous range, bitset),
mixed numeric/categorical models, multiclass, and the inputs where categorical semantics are
subtle: unseen, negative, fractional, NaN, infinite and huge values.
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

import lignum as lg

N_ROWS = 6000
CARDINALITIES = [2, 5, 63, 64, 200, 1000]

CAT_PARAMS = {
    "max_cat_to_onehot": 1,       # force many-vs-many splits
    "min_data_per_group": 5,
    "cat_smooth": 1.0,
    "cat_l2": 1.0,
    "max_cat_threshold": 512,     # allow large category sets (bitset encoding)
    "min_data_in_leaf": 5,
    "verbose": -1,
}


def make_data(rng, cardinalities, n_numeric=3):
    cols, signal = [], np.zeros(N_ROWS)
    for k in cardinalities:
        c = rng.integers(0, k, N_ROWS)
        effect = rng.normal(size=k)
        signal += effect[c]
        cols.append(c.astype(float))
    for _ in range(n_numeric):
        x = rng.normal(size=N_ROWS)
        signal += 0.5 * x
        cols.append(x)
    X = np.column_stack(cols)
    X[rng.random(X.shape) < 0.03] = np.nan
    return X, signal


def stress(X, n_cat, cardinalities, rng):
    """Training rows plus rows with unusual values in the categorical columns."""
    special = np.array([np.nan, -0.5, -0.999, -1.0, -3.0, 0.3, 1.7, 2.5, 63.0, 64.0, 65.0, 199.9,
                        1000.0, 1e6, 2.0 ** 31, 2.0 ** 32, 1e12, -1e12, np.inf, -np.inf, 1e-36, -1e-36])
    X_s = X.copy()
    for j in range(n_cat):
        mask = rng.random(len(X_s)) < 0.15
        X_s[mask, j] = rng.choice(special, size=int(mask.sum()))
        unseen = rng.random(len(X_s)) < 0.05
        X_s[unseen, j] = cardinalities[j] + rng.integers(0, 5, int(unseen.sum()))
    return X_s


def to_lignum(booster, tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(booster.dump_model()))
    return lg.load(str(path), format="lightgbm")


def assert_exact(model, booster, X):
    for strategy in ["serial", "rows"]:
        np.testing.assert_array_equal(model.predict(X, raw_score=True, strategy=strategy), booster.predict(X, raw_score=True))
        np.testing.assert_array_equal(model.predict(X, strategy=strategy), booster.predict(X))
    for i in range(0, 50):
        np.testing.assert_array_equal(model.predict(X[i:i + 1]), booster.predict(X[i:i + 1]))


def has_categorical_splits(booster):
    return '"decision_type": "=="' in json.dumps(booster.dump_model())


@pytest.mark.parametrize("cardinality", CARDINALITIES)
def test_single_categorical_feature(tmp_path, cardinality):
    rng = np.random.default_rng(cardinality)
    X, signal = make_data(rng, [cardinality], n_numeric=1)
    booster = lgb.train({"objective": "regression", **CAT_PARAMS}, lgb.Dataset(X, signal, categorical_feature=[0]), num_boost_round=40)
    assert has_categorical_splits(booster)
    assert_exact(to_lignum(booster, tmp_path), booster, stress(X, 1, [cardinality], rng))


@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_mixed_numeric_and_categorical(tmp_path, objective):
    rng = np.random.default_rng(1)
    X, signal = make_data(rng, CARDINALITIES)
    y = (signal > np.median(signal)).astype(float) if objective == "binary" else signal
    n_cat = len(CARDINALITIES)
    booster = lgb.train({"objective": objective, "num_leaves": 31, **CAT_PARAMS},
                        lgb.Dataset(X, y, categorical_feature=list(range(n_cat))), num_boost_round=80)
    assert_exact(to_lignum(booster, tmp_path), booster, stress(X, n_cat, CARDINALITIES, rng))


def test_default_one_hot_splits(tmp_path):
    """With LightGBM defaults, low-cardinality features use one-vs-rest '==' splits."""
    rng = np.random.default_rng(2)
    X, signal = make_data(rng, [3, 4, 30])
    booster = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(X, signal, categorical_feature=[0, 1, 2]), num_boost_round=40)
    assert has_categorical_splits(booster)
    assert_exact(to_lignum(booster, tmp_path), booster, stress(X, 3, [3, 4, 30], rng))


def test_multiclass_with_categorical(tmp_path):
    rng = np.random.default_rng(3)
    X, signal = make_data(rng, [10, 200])
    y = np.digitize(signal, np.quantile(signal, [0.25, 0.5, 0.75]))
    booster = lgb.train({"objective": "multiclass", "num_class": 4, **CAT_PARAMS},
                        lgb.Dataset(X, y, categorical_feature=[0, 1]), num_boost_round=20)
    assert_exact(to_lignum(booster, tmp_path), booster, stress(X, 2, [10, 200], rng))


def test_zero_as_missing_with_categorical(tmp_path):
    rng = np.random.default_rng(4)
    X, signal = make_data(rng, [8, 120])
    X[rng.random(X.shape) < 0.1] = 0.0
    booster = lgb.train({"objective": "regression", "zero_as_missing": True, **CAT_PARAMS},
                        lgb.Dataset(X, signal, categorical_feature=[0, 1]), num_boost_round=40)
    assert_exact(to_lignum(booster, tmp_path), booster, stress(X, 2, [8, 120], rng))


def test_binary_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    X, signal = make_data(rng, CARDINALITIES)
    booster = lgb.train({"objective": "regression", **CAT_PARAMS},
                        lgb.Dataset(X, signal, categorical_feature=list(range(len(CARDINALITIES)))), num_boost_round=30)
    model = to_lignum(booster, tmp_path)
    model.save(str(tmp_path / "model.bin"))
    restored = lg.load(str(tmp_path / "model.bin"))
    X_s = stress(X, len(CARDINALITIES), CARDINALITIES, rng)
    np.testing.assert_array_equal(restored.predict(X_s), booster.predict(X_s))


def test_pandas_categorical_codes(tmp_path):
    """Models trained on pandas categoricals: pass the category codes (NaN -> code -1)."""
    rng = np.random.default_rng(6)
    colors = np.array(["red", "green", "blue", "cyan", "magenta", "yellow", "black"])
    df = pd.DataFrame({
        "color": pd.Categorical(rng.choice(colors, N_ROWS), categories=colors),
        "x": rng.normal(size=N_ROWS),
    })
    df.loc[rng.random(N_ROWS) < 0.05, "color"] = np.nan
    y = (df["color"].isin(["red", "blue", "black"]).astype(float) + 0.3 * df["x"]).to_numpy()
    booster = lgb.train({"objective": "regression", **CAT_PARAMS}, lgb.Dataset(df, y), num_boost_round=30)
    model = to_lignum(booster, tmp_path)

    X = np.column_stack([df["color"].cat.codes.to_numpy(dtype=float), df["x"].to_numpy()])
    np.testing.assert_array_equal(model.predict(X, raw_score=True), booster.predict(df, raw_score=True))


def test_range_encoding_high_category_ids(tmp_path):
    """One-hot splits on categories >= 64 give single-category sets, encoded as ranges."""
    rng = np.random.default_rng(7)
    X, signal = make_data(rng, [4])
    X[:, 0] += 100  # categories 100..103 (NaN stays NaN)
    booster = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(X, signal, categorical_feature=[0]), num_boost_round=30)
    sets = [sorted(int(v) for v in t.split("||")) for t in _categorical_thresholds(booster)]
    assert any(s[0] >= 64 and s[-1] - s[0] + 1 == len(s) for s in sets)  # at least some range encodings
    X_s = stress(X, 1, [104], rng)
    X_s[rng.random(len(X_s)) < 0.1, 0] = rng.choice([99.0, 99.5, 100.0, 100.9, 103.0, 103.99, 104.0], size=1)
    assert_exact(to_lignum(booster, tmp_path), booster, X_s)


def _categorical_thresholds(booster):
    out = []

    def walk(node):
        if "split_feature" in node:
            if node["decision_type"] == "==":
                out.append(node["threshold"])
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in booster.dump_model()["tree_info"]:
        walk(tree["tree_structure"])
    return out
