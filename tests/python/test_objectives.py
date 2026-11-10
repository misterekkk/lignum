"""Every objective lignum accepts, trained for real: predict() equals the library's own prediction
bit for bit (predict / predict_proba of the model), and unknown objectives are rejected at load time
instead of predicted with the wrong output transform.

The lists in the parsers (src/parsers/lightgbm.cpp, src/parsers/xgboost.cpp, src/catboost.cpp) and
here go together: an objective added there gets a case here.
"""

import json

import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

import lignum

N = 600


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(N, 4))
    X[rng.random(X.shape) < 0.05] = np.nan
    lin = np.nan_to_num(X[:, 0]) * 1.5 + np.nan_to_num(X[:, 1]) ** 2
    y = lin + rng.normal(size=N) * 0.3
    X_test = np.vstack([X, rng.normal(size=(300, 4)) * 3, np.zeros((1, 4)), np.full((1, 4), np.nan)])
    return {
        "X": X, "X_test": X_test, "rng": rng,
        "reg": y, "pos": np.abs(y) + 0.5, "bin": (lin > np.median(lin)).astype(int),
        "cls": np.digitize(lin, np.quantile(lin, [0.33, 0.66])), "rank": np.clip(np.round(lin - lin.min()), 0, 4).astype(int),
        "groups": np.full(N // 20, 20),
    }


# ---- LightGBM -------------------------------------------------------------------------------------

LGB_CASES = [
    # (objective, target, extra parameters, expected transform)
    ("regression", "reg", {}, "identity"),
    ("regression", "reg", {"reg_sqrt": True}, "signed_square"),
    ("regression_l1", "reg", {"reg_sqrt": True}, "signed_square"),
    ("fair", "reg", {}, "identity"),
    ("fair", "reg", {"reg_sqrt": True}, "signed_square"),
    ("quantile", "reg", {"alpha": 0.8}, "identity"),
    ("quantile", "reg", {"alpha": 0.8, "reg_sqrt": True}, "signed_square"),
    ("mape", "pos", {}, "identity"),
    ("mape", "pos", {"reg_sqrt": True}, "signed_square"),
    ("huber", "reg", {"reg_sqrt": True}, "identity"),  # huber ignores reg_sqrt
    ("poisson", "pos", {"reg_sqrt": True}, "exp"),     # so does poisson
    ("gamma", "pos", {}, "exp"),
    ("tweedie", "pos", {"tweedie_variance_power": 1.3}, "exp"),
    ("binary", "bin", {"sigmoid": 0.6}, "sigmoid"),
    ("multiclass", "cls", {"num_class": 3}, "softmax"),
    ("multiclassova", "cls", {"num_class": 3, "sigmoid": 0.5}, "sigmoid"),
    ("cross_entropy", "bin", {}, "sigmoid"),
    ("cross_entropy_lambda", "bin", {}, "log1p_exp"),
    ("lambdarank", "rank", {}, "identity"),
    ("rank_xendcg", "rank", {}, "identity"),
    ("custom", "reg", {}, "identity"),  # a Python objective: the file has no objective
]


@pytest.mark.parametrize("fmt", ["txt", "json"])
@pytest.mark.parametrize("objective, target, extra, expected", LGB_CASES, ids=[f"{c[0]}-{c[2]}" for c in LGB_CASES])
def test_lightgbm_objective(tmp_path, data, objective, target, extra, expected, fmt):
    params = {"objective": objective, "verbose": -1, "num_leaves": 7, "min_data_in_leaf": 5, **extra}
    if objective == "custom":
        params["objective"] = lambda preds, ds: (preds - ds.get_label(), np.ones_like(preds))
    group = data["groups"] if objective in ("lambdarank", "rank_xendcg") else None
    booster = lgb.train(params, lgb.Dataset(data["X"], data[target], group=group), num_boost_round=15)
    path = tmp_path / f"model.{fmt}"
    if fmt == "txt":
        booster.save_model(str(path))
    else:
        path.write_text(json.dumps(booster.dump_model()))
    model = lignum.load(str(path))
    assert model.info()["transform"] == expected
    X = data["X_test"]
    np.testing.assert_array_equal(model.predict(X, raw_score=True), booster.predict(X, raw_score=True))
    np.testing.assert_array_equal(model.predict(X), booster.predict(X))
    model.save(str(tmp_path / "model.bin"))  # the transform survives the binary format
    np.testing.assert_array_equal(lignum.load(str(tmp_path / "model.bin")).predict(X), booster.predict(X))


def test_lightgbm_unknown_objectives_are_rejected(tmp_path, data):
    booster = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(data["X"], data["reg"]), num_boost_round=3)
    text = booster.model_to_string()
    for objective, message in [("made_up", "Unsupported LightGBM objective 'made_up'"),
                               ("huber sqrt", "does not write 'sqrt'"), ("poisson sqrt", "does not write 'sqrt'")]:
        path = tmp_path / "model.txt"
        path.write_text(text.replace("objective=regression", f"objective={objective}"))
        with pytest.raises(RuntimeError, match=message):
            lignum.load(str(path))
        dump = booster.dump_model()
        dump["objective"] = objective
        (tmp_path / "model.json").write_text(json.dumps(dump))
        with pytest.raises(RuntimeError, match=message):
            lignum.load(str(tmp_path / "model.json"))


# ---- XGBoost --------------------------------------------------------------------------------------

XGB_CASES = [
    ("reg:squarederror", "reg", {}, "identity"),
    ("reg:squaredlogerror", "pos", {}, "identity"),
    ("reg:pseudohubererror", "reg", {}, "identity"),
    ("reg:absoluteerror", "reg", {}, "identity"),
    ("reg:quantileerror", "reg", {"quantile_alpha": 0.7}, "identity"),
    ("reg:expectileerror", "reg", {"expectile_alpha": 0.7}, "identity"),
    ("reg:logistic", "bin", {}, "sigmoid"),
    ("binary:logistic", "bin", {"base_score": 0.3}, "sigmoid"),
    ("binary:logitraw", "bin", {}, "identity"),
    ("binary:hinge", "bin", {}, "hinge"),
    ("binary:hinge", "bin", {"base_score": 0.3}, "hinge"),
    ("count:poisson", "pos", {}, "exp"),
    ("reg:gamma", "pos", {}, "exp"),
    ("reg:tweedie", "pos", {}, "exp"),
    ("survival:cox", "cox", {}, "exp"),
    ("survival:aft", "aft", {}, "exp"),
    ("multi:softmax", "cls", {"num_class": 3}, "argmax"),
    ("multi:softprob", "cls", {"num_class": 3}, "softmax"),
    ("rank:ndcg", "rank", {}, "identity"),
    ("rank:map", "rank", {}, "identity"),
    ("rank:pairwise", "rank", {}, "identity"),
]


def xgb_matrix(data, objective, target):
    X, rng = data["X"], np.random.default_rng(2)
    if target == "cox":
        return xgb.DMatrix(X, np.where(rng.random(N) < 0.3, -1, 1) * data["pos"])
    if target == "aft":
        d = xgb.DMatrix(X)
        d.set_float_info("label_lower_bound", data["pos"])
        d.set_float_info("label_upper_bound", np.where(rng.random(N) < 0.3, np.inf, data["pos"]))
        return d
    y = (data["rank"] > 2).astype(int) if objective == "rank:map" else data[target]
    d = xgb.DMatrix(X, y)
    if objective.startswith("rank:"):
        d.set_group(data["groups"])
    return d


@pytest.mark.parametrize("fmt", ["json", "ubj"])
@pytest.mark.parametrize("objective, target, extra, expected", XGB_CASES, ids=[f"{c[0]}-{c[2]}" for c in XGB_CASES])
def test_xgboost_objective(tmp_path, data, objective, target, extra, expected, fmt):
    booster = xgb.train({"objective": objective, "max_depth": 3, **extra}, xgb_matrix(data, objective, target), num_boost_round=15)
    path = tmp_path / f"model.{fmt}"
    booster.save_model(str(path))
    model = lignum.load(str(path))
    assert model.info()["transform"] == expected
    d = xgb.DMatrix(data["X_test"])
    np.testing.assert_array_equal(model.predict(data["X_test"], raw_score=True), booster.predict(d, output_margin=True))
    np.testing.assert_array_equal(model.predict(data["X_test"]), booster.predict(d))
    model.save(str(tmp_path / "model.bin"))  # the transform survives the binary format
    np.testing.assert_array_equal(lignum.load(str(tmp_path / "model.bin")).predict(data["X_test"]), booster.predict(d))


def test_xgboost_unknown_objectives_are_rejected(tmp_path, data):
    booster = xgb.train({"objective": "reg:squarederror", "max_depth": 2}, xgb.DMatrix(data["X"], data["reg"]), num_boost_round=3)
    path = tmp_path / "model.json"
    booster.save_model(str(path))
    doc = json.loads(path.read_text())
    doc["learner"]["objective"]["name"] = "reg:made_up"
    path.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="Unsupported XGBoost objective 'reg:made_up'"):
        lignum.load(str(path))


# ---- CatBoost (one kernel: these run once) --------------------------------------------------------

cb = pytest.importorskip("catboost")


@pytest.fixture
def catboost_only_once(kernel):
    if kernel is not True:
        pytest.skip("CatBoost models have a single kernel")


CB_KW = {"iterations": 15, "depth": 3, "verbose": False, "allow_writing_files": False, "random_seed": 0, "thread_count": 2}


class PythonRMSE:
    def calc_ders_range(self, approxes, targets, weights):
        return [(t - a, -1.0) for a, t in zip(approxes, targets)]


class PythonLogloss:
    def calc_ders_range(self, approxes, targets, weights):
        p = 1 / (1 + np.exp(-np.asarray(approxes)))
        return list(zip(np.asarray(targets) - p, -p * (1 - p)))


CB_REGRESSION = [("RMSE", "reg"), ("MAE", "reg"), ("Quantile:alpha=0.7", "reg"), ("MultiQuantile:alpha=0.2,0.8", "reg"),
                 ("Expectile:alpha=0.7", "reg"), ("LogLinQuantile:alpha=0.6", "pos"), ("Lq:q=1.5", "reg"), ("Huber:delta=1.0", "reg"),
                 ("MAPE", "pos"), ("LogCosh", "reg"), ("Poisson", "pos"), ("Tweedie:variance_power=1.5", "pos"), ("Cox", "cox"),
                 ("SurvivalAft", "aft"), ("MultiRMSE", "multi"),
                 ("MultiRMSEWithMissingValues", "multi_nan")]
CB_CLASSIFICATION = [("Logloss", "bin"), ("CrossEntropy", "bin"), ("Focal:focal_alpha=0.5;focal_gamma=2", "bin"),
                     ("MultiClass", "cls"), ("MultiClassOneVsAll", "cls"), ("MultiLogloss", "labels"), ("MultiCrossEntropy", "labels")]
CB_RANKING = ["YetiRank", "YetiRankPairwise", "PairLogit", "PairLogitPairwise", "QueryRMSE", "QuerySoftMax", "LambdaMart",
              "StochasticRank:metric=NDCG", "StochasticFilter", "GroupQuantile:alpha=0.7"]


def cb_target(data, target):
    rng = np.random.default_rng(3)
    if target == "cox":
        return np.where(rng.random(N) < 0.3, -1, 1) * data["pos"]
    if target == "aft":
        return np.column_stack([data["pos"], np.where(rng.random(N) < 0.3, -1, data["pos"])])
    if target == "multi":
        return np.column_stack([data["reg"], -data["reg"]])
    if target == "multi_nan":
        y = np.column_stack([data["reg"], -data["reg"]])
        y[rng.random(y.shape) < 0.1] = np.nan  # missing targets
        return y
    if target == "labels":
        return np.column_stack([data["bin"], np.nan_to_num(data["X"][:, 2]) > 0]).astype(int)
    return data[target]


def cb_params(loss):
    """loss_function (a Python object for Python* losses, which also need an eval_metric) and CB_KW."""
    custom = {"PythonRMSE": (PythonRMSE(), "RMSE"), "PythonLogloss": (PythonLogloss(), "Logloss")}
    if loss in custom:
        objective, metric = custom[loss]
        return {"loss_function": objective, "eval_metric": metric, **CB_KW}
    return {"loss_function": loss, **CB_KW}


@pytest.mark.usefixtures("catboost_only_once")
@pytest.mark.parametrize("fmt", ["cbm", "json"])
@pytest.mark.parametrize("loss, target", CB_REGRESSION)
def test_catboost_regression_loss(tmp_path, data, loss, target, fmt):
    if fmt == "json" and loss.startswith("Multi"):
        pytest.skip("CatBoost's JSON format has no multi-output models")
    m = cb.CatBoostRegressor(**cb_params(loss)).fit(data["X"], cb_target(data, target))
    path = tmp_path / f"model.{fmt}"
    m.save_model(str(path), format=fmt)
    model = lignum.load(str(path))
    np.testing.assert_array_equal(model.predict(data["X_test"]), m.predict(data["X_test"]))


@pytest.mark.usefixtures("catboost_only_once")
@pytest.mark.parametrize("loss, target", CB_CLASSIFICATION)
def test_catboost_classification_loss(tmp_path, data, loss, target):
    m = cb.CatBoostClassifier(**cb_params(loss)).fit(data["X"], cb_target(data, target))
    m.save_model(str(tmp_path / "model.cbm"))
    model = lignum.load(str(tmp_path / "model.cbm"))
    proba = m.predict_proba(data["X_test"])
    expected = proba[:, 1] if model.n_outputs == 1 else proba  # binary: the probability of class 1
    np.testing.assert_array_equal(model.predict(data["X_test"]), expected)


@pytest.mark.usefixtures("catboost_only_once")
@pytest.mark.parametrize("loss", CB_RANKING)
def test_catboost_ranking_loss(tmp_path, data, loss):
    pool = cb.Pool(data["X"], data["rank"], group_id=np.repeat(np.arange(N // 20), 20))
    m = cb.CatBoostRanker(loss_function=loss, **CB_KW).fit(pool)
    m.save_model(str(tmp_path / "model.cbm"))
    np.testing.assert_array_equal(lignum.load(str(tmp_path / "model.cbm")).predict(data["X_test"]), m.predict(data["X_test"]))


@pytest.mark.usefixtures("catboost_only_once")
def test_catboost_unknown_losses_are_rejected(tmp_path, data):
    m = cb.CatBoostRegressor(loss_function="RMSEWithUncertainty", **CB_KW).fit(data["X"], data["reg"])
    m.save_model(str(tmp_path / "model.cbm"))
    with pytest.raises(RuntimeError, match="RMSEWithUncertainty are not supported"):
        lignum.load(str(tmp_path / "model.cbm"))

    # A Python objective: the file does not say whether predict gives raw scores or probabilities
    # (a custom classifier stores no class parameters), so both are rejected.
    for model_cls, loss, y in [(cb.CatBoostRegressor, "PythonRMSE", data["reg"]), (cb.CatBoostClassifier, "PythonLogloss", data["bin"])]:
        model_cls(**cb_params(loss)).fit(data["X"], y).save_model(str(tmp_path / "custom.cbm"))
        with pytest.raises(RuntimeError, match="Python objective"):
            lignum.load(str(tmp_path / "custom.cbm"))

    m = cb.CatBoostRegressor(**CB_KW).fit(data["X"], data["reg"])
    m.save_model(str(tmp_path / "model.json"), format="json")
    doc = json.loads((tmp_path / "model.json").read_text())
    doc["model_info"]["params"]["loss_function"]["type"] = "MadeUp"
    (tmp_path / "model.json").write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="Unsupported CatBoost loss 'MadeUp'"):
        lignum.load(str(tmp_path / "model.json"))
