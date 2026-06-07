import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
import lignum as lg
import json

OBJECTIVES = [
    ("binary", "classifier", False),
    ("regression", "regressor", False),
    ("poisson", "regressor", True),
    ("gamma", "regressor", True)
]

@pytest.fixture
def base_data():
    np.random.seed(42)
    X = np.random.randint(0, 100, size=(100, 5)).astype(np.float64)
    y_reg = np.random.rand(100).astype(np.float32)
    y_clf = np.random.randint(0, 2, 100).astype(np.float32)
    return X, y_reg, y_clf


@pytest.mark.parametrize("objective, model_type, positive_y", OBJECTIVES)
def test_lightgbm_objectives(tmp_path, base_data, objective, model_type, positive_y):
    X, y_reg, y_clf = base_data
    
    if model_type == "classifier":
        y = y_clf
        model = lgb.LGBMClassifier(n_estimators=10, max_depth=3, objective=objective, random_state=42)
    else:
        y = y_reg + 0.1 if positive_y else y_reg
        model = lgb.LGBMRegressor(n_estimators=10, max_depth=3, objective=objective, random_state=42)

    model_path = str(tmp_path / f"lgb_{objective}.json")
    model.fit(X, y)
    
    model_dict = model.booster_.dump_model()
    with open(model_path, "w") as f:
        json.dump(model_dict, f)

    lg_model = lg.load(model_path, format="lightgbm")

    expected_raw = model.predict(X, raw_score=True)
    actual_raw = lg_model.predict(X, raw_score=True)
    np.testing.assert_allclose(actual_raw, expected_raw, rtol=1e-5, atol=1e-5)

    if model_type == "classifier":
        expected_out = model.predict_proba(X)[:, 1]
    else:
        expected_out = model.predict(X)
        
    actual_out = lg_model.predict(X, raw_score=False)
    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-5, atol=1e-5)

def test_lightgbm_rejects_multiclass(tmp_path, base_data):
    X, _, _ = base_data
    y_multi = np.random.randint(0, 3, 100).astype(np.float32)
    
    model = lgb.LGBMClassifier(n_estimators=2, max_depth=2, objective="multiclass")
    model.fit(X, y_multi)
    
    model_path = str(tmp_path / "lgb_multi.json")
    model_dict = model.booster_.dump_model()
    with open(model_path, "w") as f:
        json.dump(model_dict, f)
    
    with pytest.raises(RuntimeError, match="Multiclass models are not supported yet"):
        lg.load(model_path, format="lightgbm")


def test_lightgbm_rejects_string_categorical(tmp_path):
    cats = ["A", "B", "C", "D", "E", "F"] * 100
    df = pd.DataFrame({
        "cat": pd.Categorical(cats)
    })
    
    y = np.array([1 if c in ["A", "C", "E"] else 0 for c in cats]).astype(np.float32)
    
    model = lgb.LGBMClassifier(
        n_estimators=1,
        max_depth=2,
        max_cat_to_onehot=1,
        min_data_in_leaf=1,
        min_gain_to_split=0.0,
        random_state=42
    )
    model.fit(df, y)
    
    model_path = str(tmp_path / "lgb_cat.json")
    model_dict = model.booster_.dump_model()
    with open(model_path, "w") as f:
        json.dump(model_dict, f)
    
    with pytest.raises(RuntimeError, match="Native string-based categorical splits .* are not supported yet"):
        lg.load(model_path, format="lightgbm")