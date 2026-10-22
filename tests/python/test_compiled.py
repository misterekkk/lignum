"""Compiled mode as shipped to production: library next to the model, checks, fallback."""

import json
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

import lignum as lg

@pytest.fixture(autouse=True)
def binned_only(kernel):
    # These tests compile on their own; once per test is enough.
    if kernel is not True:
        pytest.skip("runs with the binned kernel only")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 6))
    X[rng.random(X.shape) < 0.1] = np.nan
    X[:, 5] = rng.integers(0, 40, size=len(X))  # categorical codes
    X[rng.random(len(X)) < 0.05, 5] = np.nan
    y = (np.nan_to_num(X[:, 0]) + (X[:, 5] % 3 == 0) > 0.3).astype(int) + (np.nan_to_num(X[:, 1]) > 1)
    return X, y


def lightgbm_model(tmp_path, X, y):
    booster = lgb.train({"objective": "multiclass", "num_class": 3, "verbose": -1, "max_cat_to_onehot": 4},
                        lgb.Dataset(X, y, categorical_feature=[5]), num_boost_round=12)
    path = tmp_path / "lgb.json"
    path.write_text(json.dumps(booster.dump_model()))
    return path, "lightgbm", booster.predict


def xgboost_model(tmp_path, X, y):
    types = ["q"] * 5 + ["c"]
    dtrain = xgb.DMatrix(X, y, feature_types=types, enable_categorical=True)
    booster = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 4, "tree_method": "hist"},
                        dtrain, num_boost_round=12)
    path = tmp_path / "xgb.json"
    booster.save_model(str(path))
    return path, "xgboost", lambda X: booster.predict(xgb.DMatrix(X, feature_types=types, enable_categorical=True))


@pytest.mark.parametrize("make", [lightgbm_model, xgboost_model], ids=["lightgbm", "xgboost"])
def test_shipped_library_is_bit_identical(tmp_path, data, make):
    X, y = data
    source, fmt, reference = make(tmp_path, X, y)
    build = lg.load(str(source), format=fmt)
    build.save(str(tmp_path / "model.bin"))
    stats = build.compile(work_dir=str(tmp_path / "cache"), library=tmp_path / "model.so", block_trees=5)
    assert stats["compiled"] and stats["library"] == str(tmp_path / "model.so")

    # Production: the cache is gone, only the model and its library are shipped.
    for f in (tmp_path / "cache").iterdir():
        f.unlink()
    model = lg.load(str(tmp_path / "model.bin"), compiled=tmp_path / "model.so")
    assert model.uses_compiled
    np.testing.assert_array_equal(model.predict(X), reference(X))
    np.testing.assert_array_equal(model.predict(X[:1]), reference(X[:1]))
    np.testing.assert_array_equal(model.predict(X, n_jobs=1), reference(X))


def test_rejects_library_of_another_model(tmp_path, data):
    X, y = data
    source, fmt, _ = lightgbm_model(tmp_path, X, y)
    model = lg.load(str(source), format=fmt)
    model.compile(work_dir=str(tmp_path / "cache"), library=tmp_path / "a.so")

    other = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, (y > 0).astype(int)), num_boost_round=5)
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other.dump_model()))
    with pytest.raises(RuntimeError, match="different model"):
        lg.load(str(other_path), format="lightgbm", compiled=tmp_path / "a.so")
    with pytest.raises(RuntimeError, match="not found"):
        lg.load(str(source), format=fmt, compiled=tmp_path / "missing.so")


def test_fallback_keeps_the_interpreter(tmp_path, data):
    X, y = data
    source, fmt, reference = lightgbm_model(tmp_path, X, y)
    model = lg.load(str(source), format=fmt)
    with pytest.raises(RuntimeError, match="not found"):
        model.compile(compiler="lignum-no-such-compiler", work_dir=str(tmp_path / "cache"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats = model.compile(compiler="lignum-no-such-compiler", work_dir=str(tmp_path / "cache"), fallback=True)
    assert stats["compiled"] is False and "not found" in stats["error"]
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert not model.uses_compiled
    np.testing.assert_array_equal(model.predict(X), reference(X))


def test_compiled_needs_binned_kernel(tmp_path, data):
    X, y = data
    source, fmt, _ = lightgbm_model(tmp_path, X, y)
    lg.load(str(source), format=fmt).compile(work_dir=str(tmp_path / "cache"), library=tmp_path / "m.so")
    with pytest.raises(RuntimeError, match="binned"):
        lg.load(str(source), format=fmt, binned=False, compiled=tmp_path / "m.so")


def test_compiled_min_rows_switches_paths_exactly(tmp_path, data):
    X, y = data
    source, fmt, reference = lightgbm_model(tmp_path, X, y)
    model = lg.load(str(source), format=fmt)
    model.compile(work_dir=str(tmp_path / "cache"), block_trees=4)
    model.compiled_min_rows = 50
    for rows in (1, 7, 49, 50, 51, 1000):
        np.testing.assert_array_equal(model.predict(X[:rows]), reference(X[:rows]))
        np.testing.assert_array_equal(model.predict(X[:rows], n_jobs=1), reference(X[:rows]))


def test_autotune_picks_a_threshold_and_stays_exact(tmp_path, data):
    X, y = data
    source, fmt, reference = xgboost_model(tmp_path, X, y)
    model = lg.load(str(source), format=fmt)
    with pytest.raises(RuntimeError, match="compile"):
        model.autotune(X[:256])
    stats = model.compile(work_dir=str(tmp_path / "cache"), autotune=X[:1024])
    result = stats["autotune"]
    assert sorted(result["timings"]) == [1, 4, 16, 64, 256, 1024]
    assert all(t > 0 for pair in result["timings"].values() for t in pair)
    assert model.compiled_min_rows == result["compiled_min_rows"]
    assert result["compiled_min_rows"] == sys.maxsize or result["compiled_min_rows"] in result["timings"]
    for rows in (1, 16, 3000):
        np.testing.assert_array_equal(model.predict(X[:rows]), reference(X[:rows]))
