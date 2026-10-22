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
    
    np.testing.assert_array_equal(actual_json_raw, expected_raw)
    np.testing.assert_array_equal(actual_json_prob, expected_prob)
    
    np.testing.assert_array_equal(actual_bin_raw, actual_json_raw)
    np.testing.assert_array_equal(actual_bin_prob, actual_json_prob)

@pytest.mark.parametrize("save_binned", [True, False])
@pytest.mark.parametrize("load_binned", [True, False])
def test_binned_sections_round_trip(tmp_path, save_binned, load_binned):
    """The binned kernel is stored in the file; files without it get it built at load."""
    import lignum

    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 6))
    X[rng.random(X.shape) < 0.05] = np.nan
    cat = rng.integers(0, 40, size=2000).astype(float)
    X = np.column_stack([cat, X])
    y = (X[:, 1] + (cat % 3 == 0) > 0).astype(float)
    booster = lgb.train({"objective": "binary", "verbose": -1, "max_cat_to_onehot": 1, "min_data_per_group": 5},
                        lgb.Dataset(X, y, categorical_feature=[0]), num_boost_round=20)
    json_path = tmp_path / "model.json"
    json_path.write_text(json.dumps(booster.dump_model()))

    source = lignum.load(str(json_path), format="lightgbm", binned=save_binned)
    bin_path = tmp_path / "model.bin"
    source.save(str(bin_path))
    restored = lignum.load(str(bin_path), binned=load_binned)

    assert restored.uses_binned == load_binned
    np.testing.assert_array_equal(restored.predict(X), booster.predict(X))
