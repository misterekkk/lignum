"""Feature names: read from every library, kept in .bin, checked for named inputs (pandas, Arrow)."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import lignum as lg

cb = pytest.importorskip("catboost")
pa = pytest.importorskip("pyarrow")
try:
    import polars as pl
except ImportError:  # optional: not in the test extras
    pl = None

NAMES = ["age", "my income", "zł/mies", "debt"]


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(800, 4)), columns=NAMES)
    X.iloc[::17, 1] = np.nan
    X.iloc[::23, 2] = 0.0
    y = ((X["age"] > 0.1) ^ (X["zł/mies"].fillna(0) > 0)).astype(int)
    return X, y


def lightgbm_text(tmp_path, X, y):
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=15)
    booster.save_model(str(tmp_path / "m.txt"))
    return tmp_path / "m.txt", booster.predict


def lightgbm_json(tmp_path, X, y):
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=15)
    (tmp_path / "m.json").write_text(json.dumps(booster.dump_model()))
    return tmp_path / "m.json", booster.predict


def xgboost_json(tmp_path, X, y):
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 4}, xgb.DMatrix(X, y), num_boost_round=15)
    booster.save_model(str(tmp_path / "m.ubj"))
    return tmp_path / "m.ubj", lambda d: booster.predict(xgb.DMatrix(d))


def catboost_cbm(tmp_path, X, y):
    model = cb.CatBoostClassifier(iterations=15, depth=4, verbose=0, allow_writing_files=False).fit(X, y)
    model.save_model(str(tmp_path / "m.cbm"))
    return tmp_path / "m.cbm", lambda d: model.predict_proba(d)[:, 1]


MAKERS = [lightgbm_text, lightgbm_json, xgboost_json, catboost_cbm]
IDS = ["lightgbm-text", "lightgbm-json", "xgboost", "catboost"]


@pytest.mark.parametrize("make", MAKERS, ids=IDS)
def test_names_are_read_and_kept_in_bin(tmp_path, frame, make):
    X, y = frame
    path, reference = make(tmp_path, X, y)
    model = lg.load(str(path))
    expected = ["age", "my_income", "zł/mies", "debt"] if "lightgbm" in make.__name__ else NAMES
    assert model.feature_names == expected
    np.testing.assert_array_equal(model.predict(X), reference(X))  # LightGBM: "my income" matches "my_income"

    model.save(str(tmp_path / "m.bin"))
    from_bin = lg.load(str(tmp_path / "m.bin"))
    assert from_bin.feature_names == expected
    np.testing.assert_array_equal(from_bin.predict(X), reference(X))


@pytest.mark.parametrize("make", MAKERS, ids=IDS)
def test_reordered_columns_raise_for_every_input(tmp_path, frame, make):
    X, y = frame
    path, reference = make(tmp_path, X, y)
    model = lg.load(str(path))
    swapped = X[["my income", "age", "zł/mies", "debt"]]
    inputs = [swapped, pa.Table.from_pandas(swapped), pa.RecordBatch.from_pandas(swapped)]
    if pl is not None:
        inputs.append(pl.from_pandas(swapped))
    for data in inputs:
        with pytest.raises(ValueError, match="Column 0 is 'my income' but the model expects 'age'"):
            model.predict(data)
    # The fix the message suggests (for pandas), and the same predictions as the library.
    np.testing.assert_array_equal(model.predict(swapped[X.columns]), reference(X))
    # Missing and leading columns are errors too; extra columns after the model's are ignored.
    with pytest.raises(ValueError, match="3 columns but the model has 4"):
        model.predict(X.iloc[:, :3])
    with pytest.raises(ValueError, match="Column 0 is 'id'"):
        model.predict(X.assign(id=1)[["id"] + list(X.columns)])
    np.testing.assert_array_equal(model.predict(X.assign(extra=1.0)), reference(X))
    np.testing.assert_array_equal(model.predict(pa.Table.from_pandas(X.assign(extra=1.0))), reference(X))


@pytest.mark.parametrize("make", MAKERS, ids=IDS)
def test_validate_features_false_uses_columns_by_position(tmp_path, frame, make):
    X, y = frame
    path, _ = make(tmp_path, X, y)
    swapped = X[["my income", "age", "zł/mies", "debt"]]
    positional = lg.load(str(path)).predict(swapped.to_numpy())
    model = lg.load(str(path))
    np.testing.assert_array_equal(model.predict(swapped, validate_features=False), positional)
    np.testing.assert_array_equal(model.predict(pa.Table.from_pandas(swapped), validate_features=False), positional)

    off = lg.load(str(path), validate_features=False)
    assert off.validate_features is False
    np.testing.assert_array_equal(off.predict(swapped), positional)
    with pytest.raises(ValueError):
        off.predict(swapped, validate_features=True)
    off.validate_features = True
    with pytest.raises(ValueError):
        off.predict(swapped)


def test_unnamed_inputs_and_models_are_not_checked(tmp_path, frame):
    X, y = frame
    path, reference = lightgbm_text(tmp_path, X, y)
    model = lg.load(str(path))
    # numpy and frames without names (integer column labels) are used by position.
    np.testing.assert_array_equal(model.predict(X.to_numpy()), reference(X))
    np.testing.assert_array_equal(model.predict(pd.DataFrame(X.to_numpy())), reference(X))

    # Trained on a matrix: no names to check (LightGBM's Column_0... and XGBoost's empty list).
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X.to_numpy(), y), num_boost_round=5)
    booster.save_model(str(tmp_path / "plain.txt"))
    plain = lg.load(str(tmp_path / "plain.txt"))
    assert plain.feature_names is None
    np.testing.assert_array_equal(plain.predict(X[X.columns[::-1]]), booster.predict(X[X.columns[::-1]].to_numpy()))
    xb = xgb.train({"objective": "binary:logistic"}, xgb.DMatrix(X.to_numpy(), y), num_boost_round=5)
    xb.save_model(str(tmp_path / "plain.json"))
    assert lg.load(str(tmp_path / "plain.json")).feature_names is None


def test_checked_columns_are_remembered(tmp_path, frame):
    X, y = frame
    path, reference = xgboost_json(tmp_path, X, y)
    model = lg.load(str(path))
    for _ in range(3):  # the same frame again: the remembered columns object skips the comparison
        np.testing.assert_array_equal(model.predict(X), reference(X))
    swapped = X[["my income", "age", "zł/mies", "debt"]]
    with pytest.raises(ValueError):
        model.predict(swapped)
    np.testing.assert_array_equal(model.predict(X[:5]), reference(X[:5]))
