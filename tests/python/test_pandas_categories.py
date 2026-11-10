"""pandas `category` columns: LightGBM (pandas_categorical) and XGBoost 3 (cats) store the categories
of training, and their predict(df) maps a DataFrame's categories onto them. lignum must give the
same predictions bit for bit whatever order or subset of categories the input has, from pandas and
from Arrow, and raise where the library raises (XGBoost: unseen categories; both: other categorical
columns than in training).
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import lignum

pa = pytest.importorskip("pyarrow")

N = 3000
STRINGS = list("abcde")
INTEGERS = [10, 20, 30, 40]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    s = rng.choice(STRINGS, size=N)
    k = rng.choice(INTEGERS, size=N)
    x = rng.normal(size=N)
    y = (np.isin(s, ["a", "d"]) ^ (x > 0) ^ (k == 30)).astype(int)
    frame = pd.DataFrame({"x": x, "s": pd.Categorical(s, categories=STRINGS), "k": pd.Categorical(k, categories=INTEGERS)})
    return {"frame": frame, "y": y, "s": s, "k": k, "x": x}


def input_frames(data):
    """Frames whose categories differ from training's in every way the libraries handle."""
    rows = slice(0, 60)
    s, k, x = data["s"][rows], data["k"][rows], data["x"][rows]

    def frame(s_cats, k_cats, s_values=s, k_values=k):
        return pd.DataFrame({"x": x, "s": pd.Categorical(s_values, categories=s_cats), "k": pd.Categorical(k_values, categories=k_cats)})

    nan_s = s.astype(object).copy()
    nan_s[[2, 7]] = np.nan
    nan_k = k.astype(float)
    nan_k[[4]] = np.nan
    return {
        "training order": frame(STRINGS, INTEGERS),
        "reversed": frame(STRINGS[::-1], INTEGERS[::-1]),
        "from the batch": pd.DataFrame({"x": x, "s": pd.Series(s).astype("category"), "k": pd.Series(k).astype("category")}),
        "one row": frame(STRINGS[::-1], INTEGERS).iloc[3:4],
        "missing values": frame(STRINGS[::-1], INTEGERS, nan_s, nan_k),
    }


def unseen_frame(data):
    s = data["s"][:60].astype(object).copy()
    s[[1, 9]] = "z"
    return pd.DataFrame({"x": data["x"][:60], "s": pd.Categorical(s, categories=STRINGS + ["z"]),
                         "k": pd.Categorical(data["k"][:60], categories=INTEGERS)})


def save_lightgbm(booster, tmp_path, fmt):
    path = tmp_path / f"model.{fmt}"
    if fmt == "txt":
        booster.save_model(str(path))
    else:
        path.write_text(json.dumps(booster.dump_model()))
    return path


def round_trips(path, tmp_path):
    """The model from its file and from the .bin written from it (format 0.9 keeps the categories)."""
    model = lignum.load(str(path))
    model.save(str(tmp_path / "model.bin"))
    return {"file": model, "bin": lignum.load(str(tmp_path / "model.bin"))}


@pytest.mark.parametrize("fmt", ["txt", "json"])
def test_lightgbm_maps_categories_like_lightgbm(tmp_path, data, fmt):
    booster = lgb.train({"objective": "binary", "verbose": -1, "min_data_per_group": 1, "cat_smooth": 1},
                        lgb.Dataset(data["frame"], data["y"]), num_boost_round=20)
    frames = input_frames(data)
    frames["unseen values"] = unseen_frame(data)  # LightGBM: missing
    for source, model in round_trips(save_lightgbm(booster, tmp_path, fmt), tmp_path).items():
        assert model._c_model.training_categories["lists"] == [STRINGS, INTEGERS]
        for name, frame in frames.items():
            expected = booster.predict(frame)
            np.testing.assert_array_equal(model.predict(frame), expected, err_msg=f"{source}, {name}, pandas")
            table = pa.Table.from_pandas(frame, preserve_index=False)
            np.testing.assert_array_equal(model.predict(table), expected, err_msg=f"{source}, {name}, arrow")


@pytest.mark.parametrize("fmt", ["json", "ubj"])
def test_xgboost_maps_categories_like_xgboost(tmp_path, data, fmt):
    booster = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "max_depth": 3},
                        xgb.DMatrix(data["frame"], data["y"], enable_categorical=True), num_boost_round=20)
    path = tmp_path / f"model.{fmt}"
    booster.save_model(str(path))
    if lignum.load(str(path))._c_model.training_categories is None:
        pytest.skip("this XGBoost does not store the categories of training (XGBoost < 3.1)")
    for source, model in round_trips(path, tmp_path).items():
        for name, frame in input_frames(data).items():
            expected = booster.predict(xgb.DMatrix(frame, enable_categorical=True))
            np.testing.assert_array_equal(model.predict(frame), expected, err_msg=f"{source}, {name}, pandas")
            table = pa.Table.from_pandas(frame, preserve_index=False)
            np.testing.assert_array_equal(model.predict(table), expected, err_msg=f"{source}, {name}, arrow")
        # XGBoost rejects a category it was not trained with, even one no row uses.
        frame = unseen_frame(data)
        with pytest.raises(xgb.core.XGBoostError):
            booster.predict(xgb.DMatrix(frame, enable_categorical=True))
        for X in (frame, pa.Table.from_pandas(frame, preserve_index=False), frame.iloc[[0]]):
            with pytest.raises(ValueError, match="not one of the categories"):
                model.predict(X)


def test_other_categorical_columns_than_in_training_are_rejected(tmp_path, data):
    lgb_booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(data["frame"], data["y"]), num_boost_round=5)
    xgb_booster = xgb.train({"tree_method": "hist", "max_depth": 2}, xgb.DMatrix(data["frame"], data["y"], enable_categorical=True), 3)
    lgb_path = save_lightgbm(lgb_booster, tmp_path, "txt")
    xgb_path = tmp_path / "model.json"
    xgb_booster.save_model(str(xgb_path))
    extra = data["frame"].head(5).copy()
    extra["x"] = pd.Categorical(["p", "q", "p", "q", "p"])  # a numeric column of training as a category
    with pytest.raises(ValueError, match="do not match"):
        lgb_booster.predict(extra)
    with pytest.raises(ValueError, match="pandas category columns"):
        lignum.load(str(lgb_path)).predict(extra)
    if lignum.load(str(xgb_path))._c_model.training_categories is not None:
        with pytest.raises(ValueError, match="numeric"):
            lignum.load(str(xgb_path)).predict(extra)


def test_codes_without_categories_of_training(tmp_path, data):
    # A model trained on the codes of a numpy matrix stores no categories: like LightGBM, a category
    # column of a DataFrame gives its own codes (not its values).
    codes = data["frame"]["k"].cat.codes.to_numpy(dtype=float)
    booster = lgb.train({"objective": "binary", "verbose": -1, "min_data_per_group": 1, "cat_smooth": 1},
                        lgb.Dataset(np.column_stack([data["x"], codes]), data["y"], categorical_feature=[1]), num_boost_round=10)
    path = save_lightgbm(booster, tmp_path, "txt")
    model = lignum.load(str(path))
    assert model._c_model.training_categories is None
    frame = pd.DataFrame({"x": data["x"][:50], "k": pd.Categorical(data["k"][:50], categories=INTEGERS)})
    np.testing.assert_array_equal(model.predict(frame, validate_features=False), booster.predict(frame))


def test_polars_categorical(tmp_path, data):
    pl = pytest.importorskip("polars")
    booster = lgb.train({"objective": "binary", "verbose": -1, "min_data_per_group": 1, "cat_smooth": 1},
                        lgb.Dataset(data["frame"][["x", "s"]], data["y"]), num_boost_round=10)
    model = lignum.load(str(save_lightgbm(booster, tmp_path, "txt")))
    s = data["s"][:80]
    frame = pl.DataFrame({"x": data["x"][:80], "s": pl.Series(s[::-1]).cast(pl.Categorical)[::-1]})  # categories in another order
    expected = booster.predict(pd.DataFrame({"x": data["x"][:80], "s": pd.Categorical(s, categories=STRINGS)}))
    np.testing.assert_array_equal(model.predict(frame), expected)
