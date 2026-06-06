import os
import numpy as np
import lignum as lg

DIR = os.path.dirname(__file__)

def test_xgboost_predictions():
    X = np.loadtxt(os.path.join(DIR, "test_X.csv"), delimiter=",")
    expected = np.loadtxt(os.path.join(DIR, "test_preds_xgb.csv"), delimiter=",")
    
    model = lg.load(os.path.join(DIR, "dummy_model_xgb.json"), format="xgboost")
    preds = model.predict(X, n_jobs=1)
    
    np.testing.assert_allclose(preds, expected, rtol=1e-5, atol=1e-5)

def test_lightgbm_predictions():
    X = np.loadtxt(os.path.join(DIR, "test_X.csv"), delimiter=",")
    expected = np.loadtxt(os.path.join(DIR, "test_preds_lgb.csv"), delimiter=",")
    
    model = lg.load(os.path.join(DIR, "dummy_model_lgb.json"), format="lightgbm")
    preds = model.predict(X, n_jobs=1)
    
    np.testing.assert_allclose(preds, expected, rtol=1e-5, atol=1e-5)