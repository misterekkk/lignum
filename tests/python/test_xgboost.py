import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
import lignum as lg

OBJECTIVES = [
    ("binary:logistic", "classifier", False),
    ("reg:squarederror", "regressor", False),
    ("count:poisson", "regressor", True),
    ("reg:gamma", "regressor", True)
]

@pytest.fixture
def base_data():
    np.random.seed(42)
    X = np.random.randint(0, 100, size=(100, 5)).astype(np.float64)
    y_reg = np.random.rand(100).astype(np.float32)
    y_clf = np.random.randint(0, 2, 100).astype(np.float32)
    
    return X, y_reg, y_clf

@pytest.mark.parametrize("objective, model_type, positive_y", OBJECTIVES)
def test_xgboost_objectives(tmp_path, base_data, objective, model_type, positive_y):
    X, y_reg, y_clf = base_data
    
    if model_type == "classifier":
        y = y_clf
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, objective=objective, random_state=42)
    else:
        y = y_reg + 0.1 if positive_y else y_reg
        model = xgb.XGBRegressor(n_estimators=10, max_depth=3, objective=objective, random_state=42)

    model_path = str(tmp_path / f"xgb_{objective.replace(':', '_')}.json")
    model.fit(X, y)
    model.save_model(model_path)

    lg_model = lg.load(model_path, format="xgboost")

    expected_raw = model.predict(X, output_margin=True)
    actual_raw = lg_model.predict(X, raw_score=True)
    np.testing.assert_allclose(actual_raw, expected_raw, rtol=1e-4, atol=1e-4)

    if model_type == "classifier":
        expected_out = model.predict_proba(X)[:, 1]
    else:
        expected_out = model.predict(X)
        
    actual_out = lg_model.predict(X, raw_score=False)
    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)

def test_xgboost_rejects_multiclass(tmp_path, base_data):
    X, _, _ = base_data
    y_multi = np.random.randint(0, 3, 100).astype(np.float32)
    
    model = xgb.XGBClassifier(n_estimators=2, max_depth=2, objective="multi:softprob")
    model.fit(X, y_multi)
    
    model_path = str(tmp_path / "xgb_multi.json")
    model.save_model(model_path)
    
    with pytest.raises(RuntimeError, match="not supported"):
        lg.load(model_path, format="xgboost")

def test_xgboost_rejects_multitarget(tmp_path, base_data):
    X, _, _ = base_data
    y_multitarget = np.random.rand(100, 3).astype(np.float32)
    
    model = xgb.XGBRegressor(n_estimators=2, max_depth=2)
    model.fit(X, y_multitarget)
    
    model_path = str(tmp_path / "xgb_multitarget.json")
    model.save_model(model_path)
    
    with pytest.raises(RuntimeError, match="not supported"):
        lg.load(model_path, format="xgboost")

def test_xgboost_rejects_categorical(tmp_path):
    df = pd.DataFrame({
        "cat": pd.Categorical(["A", "B", "A", "B", "C", "A"] * 20)
    })
    y = (df["cat"] == "A").astype(np.float32)
    
    model = xgb.XGBRegressor(n_estimators=2, max_depth=2, enable_categorical=True)
    model.fit(df, y)
    
    model_path = str(tmp_path / "xgb_cat.json")
    model.save_model(model_path)
    
    with pytest.raises(RuntimeError, match="not supported"):
        lg.load(model_path, format="xgboost")