"""Head-to-head CPU inference benchmark: lignum vs native libraries and other inference engines.

Usage:
    python benchmarks/bench.py                 # full run (1M rows, 1000 trees)
    python benchmarks/bench.py --quick         # smoke run (100k rows, 200 trees)
    python benchmarks/bench.py --only lightgbm --engines lignum,tl2cgen

For every model and engine it reports:
  * preparation time (load / compile / convert),
  * throughput on a large batch with all threads and with one thread,
  * latency for 1 and 100 rows,
  * agreement with the native library: rows that are not bit-identical and the max abs difference,
    measured both on the benchmark data and on training rows (which hit split thresholds exactly).

Generated data and models are cached in benchmarks/.cache; results are written to benchmarks/results/.
Real datasets: --only lightgbm-real-diabetes,lightgbm-real-covtype,... (see benchmarks/datasets.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
RESULTS = HERE / "results"
N_THREADS = os.cpu_count() or 1


# ---------------------------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------------------------

@dataclass
class Workload:
    name: str               # "lightgbm" | "xgboost" | "lightgbm-multi" | "xgboost-multi"
    X: np.ndarray           # benchmark data (C-contiguous float64, 5% NaN)
    X_train: np.ndarray     # rows the model was trained on (hit split thresholds exactly)
    X_holdout: np.ndarray   # rows not used for training
    model_json: Path        # JSON dump for lignum
    model_native: Path      # native text/json file for the other engines
    booster: object
    n_classes: int = 1      # > 1: multiclass softmax, predictions are (n, n_classes)

    @property
    def family(self) -> str:
        return self.name.split("-")[0]

    def columns(self, out) -> np.ndarray:
        """Normalises an engine's output to (n,) for binary or (n, n_classes) for multiclass."""
        out = np.asarray(out, dtype=np.float64)
        n = out.shape[0]
        return out.reshape(n, -1)[:, -1] if self.n_classes == 1 else out.reshape(n, self.n_classes)


def make_data(n_rows: int, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=n_rows, n_features=n_features, n_informative=int(n_features * 0.75),
                               random_state=42)
    rng = np.random.default_rng(42)
    X[rng.random(X.shape) < 0.05] = np.nan
    return np.ascontiguousarray(X), y


CAT_CARDINALITIES = [10, 50, 200, 1000, 5000]


def make_categorical_data(n_rows: int, n_numeric: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Categorical columns (as float codes) first, then numeric ones; 5% NaN everywhere."""
    rng = np.random.default_rng(11)
    cols, signal = [], np.zeros(n_rows)
    for k in CAT_CARDINALITIES:
        codes = np.minimum(rng.zipf(1.3, n_rows) - 1, k - 1)  # skewed, like real categories
        effect = rng.normal(size=k)
        signal += effect[codes]
        cols.append(codes.astype(float))
    for _ in range(n_numeric):
        x = rng.normal(size=n_rows)
        signal += 0.3 * x
        cols.append(x)
    X = np.column_stack(cols)
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (signal + rng.normal(scale=0.5, size=n_rows) > np.median(signal)).astype(float)
    return X, y


def build_workloads(args) -> list[Workload]:
    import lightgbm as lgb
    import xgboost as xgb

    key = f"r{args.rows}_f{args.features}_t{args.trees}_d{args.depth}" + (f"_l{args.num_leaves}" if args.num_leaves else "")
    cache = CACHE / key
    cache.mkdir(parents=True, exist_ok=True)

    data_path = cache / "data.npz"
    if not data_path.exists():
        X, y = make_data(args.rows, args.features)
        np.savez(data_path, X=X, y=y)
    data = np.load(data_path)
    X, y = data["X"], data["y"]
    n_train = min(args.train_rows, len(X) // 2)
    X_train, y_train = np.ascontiguousarray(X[:n_train]), y[:n_train]
    X_holdout = np.ascontiguousarray(X[n_train:n_train + 100_000])

    workloads = []
    if "lightgbm" in args.only:
        txt, js = cache / "lgb.txt", cache / "lgb.json"
        if not txt.exists():
            params = {"objective": "binary", "max_depth": args.depth, "num_leaves": args.num_leaves or 2 ** args.depth - 1,
                      "learning_rate": 0.1, "verbose": -1, "seed": 42}
            booster = lgb.train(params, lgb.Dataset(X_train, y_train), num_boost_round=args.trees)
            booster.save_model(str(txt))
            js.write_text(json.dumps(booster.dump_model()))
        booster = lgb.Booster(model_file=str(txt))
        workloads.append(Workload("lightgbm", X, X_train, X_holdout, js, txt, booster))

    n_classes = 10
    y_multi = None
    if any(name.endswith("-multi") for name in args.only):
        # Every feature contributes (like the binary task), so no handful of features dominates the splits.
        weights = np.random.default_rng(7).normal(size=X_train.shape[1])
        Z = np.nan_to_num(X_train)
        score = Z @ weights + np.sin(Z[:, 0] * Z[:, 1]) + 0.5 * np.abs(Z[:, 2])
        y_multi = np.digitize(score, np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]))

    if "lightgbm-multi" in args.only:
        txt, js = cache / "lgb_multi.txt", cache / "lgb_multi.json"
        if not txt.exists():
            params = {"objective": "multiclass", "num_class": n_classes, "max_depth": args.depth,
                      "num_leaves": 2 ** args.depth - 1, "learning_rate": 0.1, "verbose": -1, "seed": 42}
            booster = lgb.train(params, lgb.Dataset(X_train, y_multi), num_boost_round=args.trees // n_classes)
            booster.save_model(str(txt))
            js.write_text(json.dumps(booster.dump_model()))
        booster = lgb.Booster(model_file=str(txt))
        workloads.append(Workload("lightgbm-multi", X, X_train, X_holdout, js, txt, booster, n_classes))

    if "xgboost-multi" in args.only:
        js = cache / "xgb_multi.json"
        if not js.exists():
            params = {"objective": "multi:softprob", "num_class": n_classes, "max_depth": args.depth,
                      "learning_rate": 0.1, "tree_method": "hist", "seed": 42}
            booster = xgb.train(params, xgb.DMatrix(X_train, y_multi), num_boost_round=args.trees // n_classes)
            booster.save_model(str(js))
        booster = xgb.Booster(model_file=str(js))
        workloads.append(Workload("xgboost-multi", X, X_train, X_holdout, js, js, booster, n_classes))

    if "lightgbm-cat" in args.only:
        txt, js, cat_data = cache / "lgb_cat.txt", cache / "lgb_cat.json", cache / "data_cat.npz"
        if not cat_data.exists():
            np.savez(cat_data, **dict(zip(["X", "y"], make_categorical_data(args.rows))))
        Xc, yc = (np.load(cat_data)[k] for k in ["X", "y"])
        Xc_train = np.ascontiguousarray(Xc[:n_train])
        if not txt.exists():
            params = {"objective": "binary", "max_depth": args.depth, "num_leaves": 2 ** args.depth - 1,
                      "learning_rate": 0.1, "verbose": -1, "seed": 42, "max_cat_to_onehot": 4,
                      "max_cat_threshold": 256, "cat_smooth": 10, "min_data_per_group": 50}
            booster = lgb.train(params, lgb.Dataset(Xc_train, yc[:n_train], categorical_feature=list(range(len(CAT_CARDINALITIES)))),
                                num_boost_round=args.trees)
            booster.save_model(str(txt))
            js.write_text(json.dumps(booster.dump_model()))
        booster = lgb.Booster(model_file=str(txt))
        workloads.append(Workload("lightgbm-cat", np.ascontiguousarray(Xc), Xc_train,
                                  np.ascontiguousarray(Xc[n_train:n_train + 100_000]), js, txt, booster))

    # Real datasets (benchmarks/datasets.py): lightgbm-real-<name>, LightGBM trained on 80% of the rows.
    for name in [n for n in args.only if n.startswith("lightgbm-real-")]:
        sys.path.insert(0, str(HERE))
        import datasets

        ds = name[len("lightgbm-real-"):]
        d = datasets.LOADERS[ds]()
        real_cache = CACHE / "real"
        real_cache.mkdir(parents=True, exist_ok=True)
        txt, js = real_cache / f"{ds}_t{args.trees}.txt", real_cache / f"{ds}_t{args.trees}.json"
        split = int(len(d.y) * 0.8)
        if not txt.exists():
            params = {"num_leaves": 63, "learning_rate": 0.1, "verbose": -1, "seed": 7, "max_cat_to_onehot": 4,
                      "max_cat_threshold": 64, "min_data_per_group": 20}
            if d.task == "multiclass":
                params.update(objective="multiclass", num_class=d.n_classes)
            else:
                params["objective"] = "binary" if d.task == "binary" else "regression"
            rounds = max(1, args.trees // d.n_classes)
            booster = lgb.train(params, lgb.Dataset(d.frame.iloc[:split], d.y[:split]), num_boost_round=rounds)
            booster.save_model(str(txt))
            js.write_text(json.dumps(booster.dump_model()))
        booster = lgb.Booster(model_file=str(txt))
        workloads.append(Workload(name, d.X, np.ascontiguousarray(d.X[:min(split, 100_000)]),
                                  np.ascontiguousarray(d.X[split:split + 100_000]), js, txt, booster,
                                  d.n_classes if d.task == "multiclass" else 1))

    if "xgboost-cat" in args.only:
        js, cat_data = cache / "xgb_cat.json", cache / "data_cat.npz"
        if not cat_data.exists():
            np.savez(cat_data, **dict(zip(["X", "y"], make_categorical_data(args.rows))))
        Xc, yc = (np.load(cat_data)[k] for k in ["X", "y"])
        n_cat = len(CAT_CARDINALITIES)
        types = ["c"] * n_cat + ["q"] * (Xc.shape[1] - n_cat)
        Xc_train = np.ascontiguousarray(Xc[:n_train])
        if not js.exists():
            params = {"objective": "binary:logistic", "max_depth": args.depth, "learning_rate": 0.1,
                      "tree_method": "hist", "seed": 42, "max_cat_to_onehot": 4, "max_cat_threshold": 256}
            dtrain = xgb.DMatrix(Xc_train, yc[:n_train], feature_types=types, enable_categorical=True)
            xgb.train(params, dtrain, num_boost_round=args.trees).save_model(str(js))
        booster = xgb.Booster(model_file=str(js))
        workloads.append(Workload("xgboost-cat", np.ascontiguousarray(Xc), Xc_train,
                                  np.ascontiguousarray(Xc[n_train:n_train + 100_000]), js, js, booster))

    if "xgboost" in args.only:
        js = cache / "xgb.json"
        if not js.exists():
            params = {"objective": "binary:logistic", "max_depth": args.depth, "learning_rate": 0.1,
                      "tree_method": "hist", "seed": 42}
            booster = xgb.train(params, xgb.DMatrix(X_train, y_train), num_boost_round=args.trees)
            booster.save_model(str(js))
        booster = xgb.Booster(model_file=str(js))
        workloads.append(Workload("xgboost", X, X_train, X_holdout, js, js, booster))

    return workloads


# ---------------------------------------------------------------------------------------------
# Engines: each returns predict(X, nthread) -> 1-D float64 probabilities
# ---------------------------------------------------------------------------------------------

Predict = Callable[[np.ndarray, int], np.ndarray]


def engine_native(w: Workload, tmp: Path) -> Predict:
    if w.family == "lightgbm":
        return lambda X, n: w.booster.predict(X, num_threads=n)

    boosters = {}

    def predict(X, n):
        if n not in boosters:
            b = w.booster.copy()
            b.set_param({"nthread": n})
            boosters[n] = b
        return boosters[n].inplace_predict(X).astype(np.float64)

    return predict


def engine_lignum(w: Workload, tmp: Path) -> Predict:
    import lignum

    model = lignum.load(str(w.model_json), format=w.family)
    return lambda X, n: model.predict(X, n_jobs=n)


def engine_lignum_bin(w: Workload, tmp: Path) -> Predict:
    import lignum

    path = tmp / f"{w.name}.bin"
    if not path.exists():
        lignum.load(str(w.model_json), format=w.family).save(str(path))
    model = lignum.load(str(path))
    return lambda X, n: model.predict(X, n_jobs=n)


def _treelite_model(w: Workload):
    import treelite

    if w.family == "lightgbm":
        return treelite.frontend.load_lightgbm_model(str(w.model_native))
    return treelite.frontend.load_xgboost_model(str(w.model_native))


def _tl2cgen(w: Workload, tmp: Path, params: dict, tag: str) -> Predict:
    import tl2cgen

    model = _treelite_model(w)
    if params.pop("annotate", False):
        annotation = tmp / f"{w.name}_{tag}_annotation.json"
        tl2cgen.annotate_branch(model, tl2cgen.DMatrix(w.X_train), path=str(annotation), nthread=N_THREADS)
        params["annotate_in"] = str(annotation)
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    libpath = tmp / f"{w.name}_{tag}{ext}"
    tl2cgen.export_lib(model, toolchain="clang" if sys.platform == "darwin" else "gcc", libpath=str(libpath),
                       params={"parallel_comp": N_THREADS, **params}, options=["-O3"], nthread=N_THREADS)
    predictors = {}

    def predict(X, n):
        if n not in predictors:
            predictors[n] = tl2cgen.Predictor(str(libpath), nthread=n)
        return w.columns(predictors[n].predict(tl2cgen.DMatrix(X)))

    return predict


def engine_tl2cgen(w, tmp):
    return _tl2cgen(w, tmp, {}, "default")


def engine_tl2cgen_tuned(w, tmp):
    return _tl2cgen(w, tmp, {"quantize": 1, "annotate": True}, "tuned")


def engine_gtil(w: Workload, tmp: Path) -> Predict:
    import treelite

    model = _treelite_model(w)
    return lambda X, n: w.columns(treelite.gtil.predict(model, X, nthread=n))


def engine_lleaves(w: Workload, tmp: Path) -> Predict:
    if w.family != "lightgbm":
        raise NotSupported("lleaves only supports LightGBM")
    import lleaves

    model = lleaves.Model(model_file=str(w.model_native))
    model.compile(cache=str(tmp / f"{w.name}_lleaves.o"))
    return lambda X, n: model.predict(X, n_jobs=n)


def engine_onnxruntime(w: Workload, tmp: Path) -> Predict:
    import onnxmltools
    import onnxruntime as ort
    from onnxmltools.convert.common.data_types import FloatTensorType

    n_features = w.X.shape[1]
    initial_types = [("input", FloatTensorType([None, n_features]))]
    if w.family == "lightgbm":
        onx = onnxmltools.convert_lightgbm(w.booster, initial_types=initial_types, zipmap=False, target_opset=15)
    else:
        import xgboost as xgb

        clf = xgb.XGBClassifier()
        clf.load_model(str(w.model_native))
        onx = onnxmltools.convert_xgboost(clf, initial_types=initial_types, target_opset=15)
    payload = onx.SerializeToString()
    sessions = {}

    def predict(X, n):
        if n not in sessions:
            so = ort.SessionOptions()
            so.intra_op_num_threads = n
            so.inter_op_num_threads = 1
            so.log_severity_level = 3  # the converted classifier declares a fixed label shape; ORT warns every call
            sessions[n] = ort.InferenceSession(payload, so, providers=["CPUExecutionProvider"])
        out = sessions[n].run(None, {"input": X.astype(np.float32)})
        return w.columns(out[1])

    return predict


class NotSupported(Exception):
    pass


ENGINES: dict[str, Callable[[Workload, Path], Predict]] = {
    "native": engine_native,
    "lignum": engine_lignum,
    "lignum-bin": engine_lignum_bin,
    "tl2cgen": engine_tl2cgen,
    "tl2cgen-tuned": engine_tl2cgen_tuned,
    "gtil": engine_gtil,
    "lleaves": engine_lleaves,
    "onnxruntime": engine_onnxruntime,
}


# ---------------------------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------------------------

@dataclass
class Result:
    model: str
    engine: str
    prep_s: float | None = None
    batch_best_s: float | None = None
    batch_median_s: float | None = None
    batch1_best_s: float | None = None
    lat1_us: float | None = None
    lat1_st_us: float | None = None
    lat1_st_p99_us: float | None = None
    lat100_us: float | None = None
    lat100_st_us: float | None = None
    mismatch_bench: int | None = None
    maxdiff_bench: float | None = None
    mismatch_train: int | None = None
    maxdiff_train: float | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def time_batch(predict: Predict, X: np.ndarray, n: int, repeats: int) -> tuple[float, float]:
    predict(X[:1000], n)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict(X, n)
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times)


def time_latency(predict: Predict, X: np.ndarray, n: int, budget_s: float, with_p99: bool = False):
    """Median latency in µs (and p99 when with_p99)."""
    predict(X, n)
    times = []
    deadline = time.perf_counter() + budget_s
    while len(times) < 50 or (time.perf_counter() < deadline and len(times) < 20000):
        t0 = time.perf_counter()
        predict(X, n)
        times.append(time.perf_counter() - t0)
    median = statistics.median(times) * 1e6
    if not with_p99:
        return median
    return median, float(np.percentile(times, 99)) * 1e6


def compare(got: np.ndarray, ref: np.ndarray) -> tuple[int, float]:
    """Rows where any output differs from the reference, and the max abs difference."""
    got = np.asarray(got, dtype=np.float64).reshape(len(ref), -1)
    ref = np.asarray(ref, dtype=np.float64).reshape(len(ref), -1)
    return int(np.sum(np.any(got != ref, axis=1))), float(np.max(np.abs(got - ref)))


def run_engine(w: Workload, engine: str, args, tmp: Path, reference: dict) -> Result:
    r = Result(w.name, engine)
    try:
        t0 = time.perf_counter()
        predict = ENGINES[engine](w, tmp)
        r.prep_s = time.perf_counter() - t0

        r.mismatch_bench, r.maxdiff_bench = compare(predict(w.X_holdout, N_THREADS), reference["bench"])
        r.mismatch_train, r.maxdiff_train = compare(predict(w.X_train[:100_000], N_THREADS), reference["train"])

        r.batch_best_s, r.batch_median_s = time_batch(predict, w.X, N_THREADS, args.repeats)
        X_st = w.X[: max(1, len(w.X) // 5)]
        best_st, _ = time_batch(predict, X_st, 1, max(1, args.repeats // 2))
        r.batch1_best_s = best_st * (len(w.X) / len(X_st))  # normalised to the full batch size

        X1, X100 = np.ascontiguousarray(w.X[:1]), np.ascontiguousarray(w.X[:100])
        r.lat1_us = time_latency(predict, X1, N_THREADS, args.latency_budget)
        r.lat1_st_us, r.lat1_st_p99_us = time_latency(predict, X1, 1, args.latency_budget, with_p99=True)
        r.lat100_us = time_latency(predict, X100, N_THREADS, args.latency_budget)
        r.lat100_st_us = time_latency(predict, X100, 1, args.latency_budget)
    except NotSupported as e:
        r.error = f"n/a ({e})"
    except Exception as e:  # report and continue with the other engines
        r.error = f"{type(e).__name__}: {e}".splitlines()[0][:160]
        if args.verbose:
            traceback.print_exc()
    return r


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------

def fmt(v, kind):
    if v is None:
        return "—"
    if kind == "s":
        return f"{v:.3f}"
    if kind == "us":
        return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"
    if kind == "diff":
        return "0" if v == 0 else f"{v:.1e}"
    return str(v)


def environment() -> dict:
    import importlib.metadata as md

    def version(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "include", "python"], cwd=HERE,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "unknown", False
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": commit + ("-dirty" if dirty else ""),
        "cpu": cpu,
        "threads": N_THREADS,
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ["lignum", "lightgbm", "xgboost", "treelite", "tl2cgen", "lleaves", "onnxruntime", "onnxmltools"]},
    }


def render(results: list[Result], env: dict, args) -> str:
    lines = [
        f"# Benchmark {env['date']}",
        "",
        f"- lignum commit `{env['commit']}`, {env['cpu']}, {env['threads']} threads, Python {env['python']}",
        f"- {args.rows:,} rows × {args.features} features (5% NaN), {args.trees} trees, max depth {args.depth}"
        + (f", num_leaves {args.num_leaves}" if args.num_leaves else "") + "; "
        f"binary classification, and 10-class softmax ({args.trees // 10} iterations × 10 trees) for `*-multi`; "
        f"`lightgbm-cat`: {len(CAT_CARDINALITIES)} categorical features (cardinality {', '.join(map(str, CAT_CARDINALITIES))}) + 15 numeric",
        "- versions: " + ", ".join(f"{k} {v}" for k, v in env["versions"].items() if v),
        "",
        "Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.",
        "Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).",
        "",
    ]
    for model in dict.fromkeys(r.model for r in results):
        lines += [
            f"## {model}",
            "",
            "| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in (r for r in results if r.model == model):
            if r.error:
                lines.append(f"| {r.engine} | {r.error} |" + " |" * 12)
                continue
            maxdiff = max(r.maxdiff_bench, r.maxdiff_train)
            lines.append(
                f"| {r.engine} | {fmt(r.prep_s, 's')} | {fmt(r.batch_best_s, 's')} | {fmt(r.batch_median_s, 's')} | "
                f"{fmt(r.batch1_best_s, 's')} | {fmt(r.lat1_us, 'us')} | {fmt(r.lat1_st_us, 'us')} | {fmt(r.lat100_us, 'us')} | "
                f"{fmt(r.lat100_st_us, 'us')} | {r.mismatch_bench:,} | {r.mismatch_train:,} | {fmt(maxdiff, 'diff')} | {fmt(r.lat1_st_p99_us, 'us')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6, help="max depth (-1: unlimited, LightGBM leaf-wise)")
    parser.add_argument("--num-leaves", type=int, default=None, help="LightGBM num_leaves (default 2**depth - 1)")
    parser.add_argument("--train-rows", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--latency-budget", type=float, default=2.0, help="seconds per latency measurement")
    parser.add_argument("--only", default="lightgbm,xgboost,lightgbm-multi,xgboost-multi,lightgbm-cat,xgboost-cat")
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--quick", action="store_true", help="100k rows, 200 trees, fewer repeats")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.rows, args.trees, args.repeats, args.latency_budget = 100_000, 200, 3, 0.5
    args.only = args.only.split(",")
    engines = args.engines.split(",")

    env = environment()
    results = []
    with tempfile.TemporaryDirectory(prefix="lignum_bench_") as tmpdir:
        tmp = Path(tmpdir)
        for w in build_workloads(args):
            print(f"== {w.name}: computing native reference", flush=True)
            native = engine_native(w, tmp)
            reference = {"bench": native(w.X_holdout, N_THREADS), "train": native(w.X_train[:100_000], N_THREADS)}
            for engine in engines:
                print(f"   {engine} ...", end=" ", flush=True)
                t0 = time.perf_counter()
                r = run_engine(w, engine, args, tmp, reference)
                print(r.error or f"done in {time.perf_counter() - t0:.0f}s", flush=True)
                results.append(r)

    report = render(results, env, args)
    print("\n" + report)
    RESULTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}.md").write_text(report)
    (RESULTS / f"{stamp}.json").write_text(json.dumps({"env": env, "args": vars(args), "results": [r.__dict__ for r in results]}, indent=2))
    print(f"\nSaved to {RESULTS / (stamp + '.md')}")


if __name__ == "__main__":
    main()
