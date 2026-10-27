"""Model inspection: source library, objective, tree and leaf counts, info(), repr, kept in .bin."""

import json

import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

import lignum as lg

cb = pytest.importorskip("catboost")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 5))
    X[::13, 2] = np.nan
    y = (X[:, 0] + 0.5 * np.nan_to_num(X[:, 2]) > 0).astype(int)
    classes = np.digitize(X[:, 1], [-0.5, 0.5])
    return X, y, classes


def lightgbm_leaves(booster):
    return sum(tree["num_leaves"] for tree in booster.dump_model()["tree_info"])


def xgboost_leaves(booster):
    return sum(dump.count("leaf=") for dump in booster.get_dump())


def test_lightgbm(tmp_path, data, kernel):
    X, y, classes = data
    binary = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=12)
    multi = lgb.train({"objective": "multiclass", "num_class": 3, "verbose": -1}, lgb.Dataset(X, classes), num_boost_round=7)
    for booster, objective, transform in [(binary, "binary", "sigmoid"), (multi, "multiclass", "softmax")]:
        booster.save_model(str(tmp_path / "m.txt"))
        (tmp_path / "m.json").write_text(json.dumps(booster.dump_model()))
        for path in ("m.txt", "m.json"):
            model = lg.load(str(tmp_path / path))
            assert model.source == "lightgbm"
            assert model.objective == objective
            assert model.n_trees == booster.num_trees()
            info = model.info()
            assert info["transform"] == transform
            assert info["n_leaves"] == lightgbm_leaves(booster)
            assert info["n_outputs"] == booster.num_model_per_iteration()
            assert info["kernel"] == {True: "binned", False: "direct", "compiled": "compiled"}[kernel]
            assert (info["compiled_library"] is not None) == (kernel == "compiled")
            assert info["written_by"] is None
    assert multi.num_trees() == 21


def test_xgboost(tmp_path, data):
    X, y, classes = data
    cases = [({"objective": "binary:logistic"}, y, "binary:logistic", "sigmoid"),
             ({"objective": "multi:softprob", "num_class": 3}, classes, "multi:softprob", "softmax"),
             ({"objective": "reg:squarederror"}, X[:, 0], "reg:squarederror", "identity")]
    for params, target, objective, transform in cases:
        booster = xgb.train({**params, "max_depth": 3}, xgb.DMatrix(X, target), num_boost_round=6)
        for name in ("m.json", "m.ubj"):
            booster.save_model(str(tmp_path / name))
            model = lg.load(str(tmp_path / name))
            assert (model.source, model.objective) == ("xgboost", objective)
            assert model.n_trees == len(booster.get_dump())
            assert model.info()["n_leaves"] == xgboost_leaves(booster)
            assert model.info()["transform"] == transform


def test_catboost(tmp_path, data):
    X, y, _ = data
    clf = cb.CatBoostClassifier(iterations=9, depth=4, verbose=0, allow_writing_files=False).fit(X, y)
    clf.save_model(str(tmp_path / "m.cbm"))
    model = lg.load(str(tmp_path / "m.cbm"))
    info = model.info()
    assert (info["source"], info["objective"], info["transform"]) == ("catboost", "Logloss", "sigmoid")
    assert info["n_trees"] == clf.tree_count_ == 9
    assert info["n_leaves"] == 9 * 2 ** 4
    assert info["kernel"] == "catboost"


def test_bin_keeps_everything_and_records_the_writer(tmp_path, data):
    X, y, _ = data
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=12)
    booster.save_model(str(tmp_path / "m.txt"))
    model = lg.load(str(tmp_path / "m.txt"))
    model.save(str(tmp_path / "m.bin"))
    loaded = lg.load(str(tmp_path / "m.bin"))
    before, after = model.info(), loaded.info()
    assert after.pop("written_by") == lg.__version__
    before.pop("written_by")
    before.pop("compiled_library"), after.pop("compiled_library")  # compiled mode: each model has its own
    assert after == before


def test_repr(tmp_path, data):
    X, y, _ = data
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=12)
    booster.save_model(str(tmp_path / "m.txt"))
    assert repr(lg.load(str(tmp_path / "m.txt"))) == "<lignum.Model lightgbm binary, 12 trees, 5 features>"
