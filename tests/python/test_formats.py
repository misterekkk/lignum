"""Model file formats: LightGBM model.txt and XGBoost .ubj give the same model as the JSON files.

Every case is checked bit-exact against the library on inputs that hit every threshold, NaNs,
zeros and LightGBM's zero band (helpers from test_parity / test_categorical).
"""

import json

import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

import lignum as lg
import test_categorical_xgboost as xgb_cat
from test_categorical import CAT_PARAMS, make_data, stress
from test_parity import LGB_CASES, XGB_CASES, data, lgb_target, lgb_thresholds, stress_inputs, xgb_target, xgb_thresholds  # noqa: F401


def lightgbm_files(booster, tmp_path):
    txt, js = tmp_path / "model.txt", tmp_path / "model.json"
    booster.save_model(str(txt))
    js.write_text(json.dumps(booster.dump_model()))
    return txt, js


def assert_same(model, booster_predict, X):
    np.testing.assert_array_equal(model.predict(X, raw_score=True), booster_predict(X, True))
    np.testing.assert_array_equal(model.predict(X), booster_predict(X, False))


@pytest.mark.parametrize("train_on", ["X_nan", "X_zero"])
@pytest.mark.parametrize("objective, extra", LGB_CASES)
def test_lightgbm_text_matches_json_and_library(tmp_path, data, objective, extra, train_on):
    X_train = data[train_on]
    params = {"objective": objective, "num_leaves": 31, "verbose": -1, **extra}
    booster = lgb.train(params, lgb.Dataset(X_train, lgb_target(objective, data["y"])), num_boost_round=40)
    txt, js = lightgbm_files(booster, tmp_path)
    X = stress_inputs(X_train, lgb_thresholds(booster), data["rng"])
    predict = lambda X, raw: booster.predict(X, raw_score=raw)
    for path, fmt in ((txt, "lightgbm"), (txt, "auto"), (js, "auto")):
        assert_same(lg.load(str(path), format=fmt), predict, X)


def test_lightgbm_text_multiclass_categories_and_stumps(tmp_path):
    rng = np.random.default_rng(1)
    cards = [2, 5, 63, 200, 1000]
    X, y = make_data(rng, cards)
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    booster = lgb.train({"objective": "multiclass", "num_class": 3, **CAT_PARAMS},
                        lgb.Dataset(X, labels, categorical_feature=list(range(len(cards)))), num_boost_round=15)
    txt, _ = lightgbm_files(booster, tmp_path)
    assert "cat_threshold=" in txt.read_text()
    X_test = stress(X, len(cards), cards, rng)
    assert_same(lg.load(str(txt)), lambda X, raw: booster.predict(X, raw_score=raw), X_test)

    # Trees with a single leaf (nothing to split on after the first rounds).
    stump = lgb.train({"objective": "regression", "verbose": -1, "min_data_in_leaf": 10_000}, lgb.Dataset(X[:500], y[:500]),
                      num_boost_round=3)
    txt2 = tmp_path / "stump.txt"
    stump.save_model(str(txt2))
    assert "num_leaves=1" in txt2.read_text()
    assert_same(lg.load(str(txt2)), lambda X, raw: stump.predict(X, raw_score=raw), X_test)


def xgboost_files(booster, tmp_path):
    ubj, js = tmp_path / "model.ubj", tmp_path / "model.json"
    booster.save_model(str(ubj))
    booster.save_model(str(js))
    assert ubj.read_bytes()[:1] == b"{" and ubj.read_bytes()[1:2] != b'"'
    return ubj, js


@pytest.mark.parametrize("objective, extra", XGB_CASES)
def test_xgboost_ubj_matches_json_and_library(tmp_path, data, objective, extra):
    params = {"objective": objective, "max_depth": 5, **extra}
    booster = xgb.train(params, xgb.DMatrix(data["X_nan"], xgb_target(objective, data["y"])), num_boost_round=40)
    ubj, js = xgboost_files(booster, tmp_path)
    X = stress_inputs(data["X_nan"], xgb_thresholds(js), data["rng"])
    predict = lambda X, raw: booster.predict(xgb.DMatrix(X), output_margin=raw)
    for path, fmt in ((ubj, "xgboost"), (ubj, "auto"), (js, "auto")):
        assert_same(lg.load(str(path), format=fmt), predict, X)


def test_xgboost_ubj_multiclass_and_categories(tmp_path):
    rng = np.random.default_rng(2)
    cards = [10, 150, 1000]
    X, signal = xgb_cat.make_data(rng, cards)
    labels = np.digitize(signal, np.quantile(signal, [0.33, 0.66]))
    booster = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 4, "tree_method": "hist",
                         "max_cat_to_onehot": 1}, xgb_cat.dmatrix(X, len(cards), labels), num_boost_round=10)
    ubj, js = xgboost_files(booster, tmp_path)
    X_test = xgb_cat.stress(X, len(cards), cards, rng)
    predict = lambda X, raw: booster.predict(xgb_cat.dmatrix(X, len(cards)), output_margin=raw)
    for path in (ubj, js):
        assert_same(lg.load(str(path)), predict, X_test)


def test_auto_detection_and_errors(tmp_path, data):
    booster = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(data["X"], data["y"]), num_boost_round=5)
    txt, js = lightgbm_files(booster, tmp_path)
    model = lg.load(str(txt))
    model.save(str(tmp_path / "model.bin"))
    np.testing.assert_array_equal(lg.load(str(tmp_path / "model.bin")).predict(data["X"]), booster.predict(data["X"]))

    with pytest.raises(ValueError, match="Unrecognised model file"):
        lg.load(str(txt), format="xgboost")
    with pytest.raises(ValueError, match="lignum binary"):
        lg.load(str(tmp_path / "model.bin"), format="lightgbm")
    garbage = tmp_path / "garbage.txt"
    garbage.write_text("not a model\n")
    with pytest.raises(ValueError, match="Unrecognised model file"):
        lg.load(str(garbage))
    broken = tmp_path / "broken.txt"
    broken.write_text(txt.read_text().replace("leaf_value=", "leaf_value=1 2 x", 1))
    with pytest.raises(RuntimeError, match="Malformed LightGBM model file"):
        lg.load(str(broken))
    linear = tmp_path / "linear.txt"
    linear.write_text(txt.read_text().replace("is_linear=0", "is_linear=1", 1))
    with pytest.raises(RuntimeError, match="linear trees"):
        lg.load(str(linear))
    truncated = tmp_path / "truncated.ubj"
    xgb_booster = xgb.train({"max_depth": 2}, xgb.DMatrix(data["X"], data["y"]), num_boost_round=2)
    xgb_booster.save_model(str(truncated))
    truncated.write_bytes(truncated.read_bytes()[:200])
    with pytest.raises(RuntimeError, match="Malformed UBJSON"):
        lg.load(str(truncated))


@pytest.mark.parametrize("params, message", [({"linear_tree": True}, "linear trees"),
                                             ({"boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1}, "random forest")])
def test_unsupported_lightgbm_models_are_rejected(tmp_path, data, params, message):
    # Predicting them with plain trees would be silently wrong, so both formats refuse them.
    booster = lgb.train({"objective": "regression", "verbose": -1, **params}, lgb.Dataset(data["X"], data["y"]), num_boost_round=5)
    txt, js = lightgbm_files(booster, tmp_path)
    for path in (txt, js):
        with pytest.raises(RuntimeError, match=message):
            lg.load(str(path))


@pytest.mark.filterwarnings("ignore::UserWarning")  # booster=dart is deprecated in XGBoost 3.x
def test_dart_without_dropped_trees_matches_both_libraries(tmp_path, data):
    # LightGBM bakes the DART weights into the saved leaves; XGBoost keeps one weight per tree,
    # all 1 when DART dropped nothing (rate_drop=0, the default).
    X, y = data["X"], data["y"]
    booster = lgb.train({"objective": "regression", "boosting": "dart", "drop_rate": 0.3, "verbose": -1},
                        lgb.Dataset(X, y), num_boost_round=15)
    for path in lightgbm_files(booster, tmp_path):
        assert_same(lg.load(str(path)), lambda d, raw: booster.predict(d, raw_score=raw), X)
    xb = xgb.train({"objective": "binary:logistic", "booster": "dart"}, xgb.DMatrix(X, y > np.median(y)), num_boost_round=10)
    for name in ("model.json", "model.ubj"):
        xb.save_model(str(tmp_path / name))
        assert_same(lg.load(str(tmp_path / name)), lambda d, raw: xb.predict(xgb.DMatrix(d), output_margin=raw), X)


@pytest.mark.filterwarnings("ignore::UserWarning")  # dart and gblinear are deprecated in XGBoost 3.x
@pytest.mark.parametrize("params, message", [({"booster": "dart", "rate_drop": 0.3, "skip_drop": 0.0}, "DART"),
                                             ({"booster": "gblinear"}, "gblinear")])
def test_unsupported_xgboost_boosters_are_rejected(tmp_path, data, params, message):
    xb = xgb.train({"objective": "binary:logistic", **params}, xgb.DMatrix(data["X"], data["y"] > np.median(data["y"])),
                   num_boost_round=10)
    for name in ("model.json", "model.ubj"):
        xb.save_model(str(tmp_path / name))
        with pytest.raises(RuntimeError, match=message):
            lg.load(str(tmp_path / name))
