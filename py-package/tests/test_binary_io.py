import pytest
import numpy as np
import lightgbm as lgb
import lignum as lg
import json

def test_binary_serialization_pipeline(tmp_path):
    np.random.seed(42)
    X = np.random.randint(0, 100, size=(100, 5)).astype(np.float64)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    
    lgb_model = lgb.LGBMClassifier(n_estimators=10, max_depth=3, objective="binary", random_state=42)
    lgb_model.fit(X, y)
    
    json_path = str(tmp_path / "model_transfer.json")
    bin_path = str(tmp_path / "model_transfer.bin")
    
    model_dict = lgb_model.booster_.dump_model()
    with open(json_path, "w") as f:
        json.dump(model_dict, f)
        
    lg_model_json = lg.load(json_path, format="lightgbm")
    
    lg_model_json.save(bin_path)
    lg_model_bin = lg.load(bin_path, format="binary")
    
    expected_raw = lgb_model.predict(X, raw_score=True)
    actual_json_raw = lg_model_json.predict(X, raw_score=True)
    actual_bin_raw = lg_model_bin.predict(X, raw_score=True)
    
    expected_prob = lgb_model.predict_proba(X)[:, 1]
    actual_json_prob = lg_model_json.predict(X, raw_score=False)
    actual_bin_prob = lg_model_bin.predict(X, raw_score=False)
    
    np.testing.assert_allclose(actual_json_raw, expected_raw, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_json_prob, expected_prob, rtol=1e-5, atol=1e-5)
    
    np.testing.assert_allclose(actual_bin_raw, actual_json_raw, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(actual_bin_prob, actual_json_prob, rtol=1e-7, atol=1e-7)