import numpy as np
import time
import lignum as lg
import treelite
import treelite_runtime
import os
from sklearn.datasets import make_classification
MAX_CORES = os.cpu_count()

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TXT = os.path.join(DIR, "bench_model_lgb.txt")
MODEL_JSON = os.path.join(DIR, "bench_model_lgb.json")
LIB_PATH = os.path.join(DIR, "treelite_model.dylib")

def run_benchmark():
    n_samples = 1_000_000
    n_features = 20

    X, _ = make_classification(n_samples=n_samples, n_features=n_features, n_informative=15, random_state=42)
    X = np.round(X, 4)
    np.random.seed(42)
    nan_mask = np.random.rand(*X.shape) < 0.05
    X[nan_mask] = np.nan

    t_comp_start = time.perf_counter()
    tl_model = treelite.Model.load(MODEL_TXT, model_format='lightgbm')
    tl_model.export_lib(toolchain='clang', libpath=LIB_PATH, params={'parallel_comp': MAX_CORES}, verbose=False)
    tl_predictor = treelite_runtime.Predictor(LIB_PATH, nthread=MAX_CORES)
    comp_time = time.perf_counter() - t_comp_start
    print(f"Treelite compilation time: {comp_time:.2f} s")

    t_load_start = time.perf_counter()
    lignum_model = lg.load(MODEL_JSON, format="lightgbm")
    load_time = time.perf_counter() - t_load_start
    print(f"Lignum loading time: {load_time:.4f} s")

    X_warmup = X[:1000]
    _ = tl_predictor.predict(treelite_runtime.DMatrix(X_warmup))
    _ = lignum_model.predict(X_warmup, n_jobs=MAX_CORES)
    
    dmat = treelite_runtime.DMatrix(X)
    t0 = time.perf_counter()
    tl_preds = tl_predictor.predict(dmat, pred_margin=True)
    tl_time = time.perf_counter() - t0
    print(f"[Treelite] Time: {tl_time:.4f} s")

    t0 = time.perf_counter()
    lignum_preds = lignum_model.predict(X, n_jobs=MAX_CORES)
    lignum_time = time.perf_counter() - t0
    print(f"[Lignum] Time: {lignum_time:.4f} s")

    diffs = np.abs(tl_preds - lignum_preds)
    bad_count = np.sum(diffs > 1e-4)
    print(f"Errors: {bad_count} / {n_samples}")
    
    if os.path.exists(LIB_PATH):
        os.remove(LIB_PATH)

if __name__ == "__main__":
    run_benchmark()