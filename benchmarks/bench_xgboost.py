import numpy as np
import xgboost as xgb
import time
import lignum as lg
import os
from sklearn.datasets import make_classification
MAX_CORES = os.cpu_count()

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(DIR, "bench_model_xgb.json")

def run_benchmark():
    n_samples = 1_000_000
    n_features = 20
    num_trees = 1000

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=15, random_state=42)
    X = np.round(X, 4)
    np.random.seed(42)
    nan_mask = np.random.rand(*X.shape) < 0.05
    X[nan_mask] = np.nan

    dtrain = xgb.DMatrix(X, label=y)
    params = {'max_depth': 6, 'eta': 0.1, 'objective': 'binary:logistic', 'tree_method': 'hist'}
    bst = xgb.train(params, dtrain, num_boost_round=num_trees)
    bst.save_model(MODEL_JSON)

    lignum_model = lg.load(MODEL_JSON, format="xgboost")

    bst.set_param({'nthread': MAX_CORES})

    X_warmup = X[:1000]
    _ = bst.predict(xgb.DMatrix(X_warmup))
    _ = lignum_model.predict(X_warmup, n_jobs=MAX_CORES)
    
    t0 = time.perf_counter()
    xgb_preds = bst.predict(dtrain)
    xgb_time = time.perf_counter() - t0
    print(f"[XGBoost] Time: {xgb_time:.4f} s")

    t0 = time.perf_counter()
    lignum_preds = lignum_model.predict(X, n_jobs=MAX_CORES)
    lignum_time = time.perf_counter() - t0
    print(f"[Lignum] Time: {lignum_time:.4f} s")

    diffs = np.abs(xgb_preds - lignum_preds)
    bad_count = np.sum(diffs > 1e-4)
    print(f"Errors: {bad_count} / {n_samples}")

if __name__ == "__main__":
    run_benchmark()