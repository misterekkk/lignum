"""The benchmark behind the published tables: one set of models, one method, every platform.

Usage:
    python benchmarks/suite.py                                   # everything (about 1-2 hours)
    python benchmarks/suite.py --libraries lightgbm --datasets california,covtype
    python benchmarks/suite.py --quick                           # smoke run: small batches, few calls

Models (trained once, cached in benchmarks/.cache/suite; copy that folder to other machines so that
every platform predicts with the same files, their SHA-256 is in the report):
  - 500 boosting rounds, learning rate 0.1, seed 7, 80% of each data set (a multiclass round is one
    tree per class in LightGBM / XGBoost and one tree with a leaf vector in CatBoost);
  - LightGBM: num_leaves 63, max_depth -1 (leaf-wise growth);
  - XGBoost: tree_method hist, grow_policy lossguide, max_leaves 63, max_depth 0 (the same leaf-wise
    growth; the pairing LightGBM's own docs/Experiments.rst uses);
  - CatBoost: symmetric trees of depth 6 = 64 leaves (lignum reads symmetric trees only). Every row
    makes exactly 6 comparisons per tree; the report gives the measured average path length of the
    other models.

Measurements, for every engine on one thread and on all threads:
  - batches of 1, 10 and 100 rows: p50 / p99 / p99.9 of single calls (--calls, at least 10,000),
    each call on different rows (1,000 distinct batches in rotation), garbage collector off;
  - batches of 10k, 100k and 1M rows: best and median of repeated calls on different windows of the
    data (data sets with fewer rows are tiled), rows per second from the median;
  - preparation: loading the model, plus compiling it for the engines that compile, from a cold cache;
  - exactness: rows not bit-identical to the library's own predict, on held-out rows and on training
    rows (which hit split thresholds exactly).

Inputs (--sources), the same rows in both:
  - matrix: LightGBM and XGBoost engines get the float64 matrix (categories as codes, what
    datasets.py builds); CatBoost and lignum on CatBoost models get the pandas DataFrame with string
    categories, as CatBoost users call it;
  - arrow: a pyarrow Table (categories as dictionary columns), end to end: lignum reads it directly,
    LightGBM and XGBoost read it when it has no dictionary columns and go through to_pandas()
    otherwise, CatBoost and lleaves always go through to_pandas(). tl2cgen takes matrices only.
Results: benchmarks/results/<stamp>-suite.{md,json}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "suite"
RESULTS = HERE / "results"
N_THREADS = os.cpu_count() or 1

sys.path.insert(0, str(HERE))
import bench  # noqa: E402  (environment())
import datasets  # noqa: E402

LIBRARIES = ["lightgbm", "xgboost", "catboost"]
DATASETS = ["california", "covtype", "adult", "bank", "diabetes", "synthetic", "synthetic-cat"]
ENGINES = {
    "lightgbm": ["native", "lignum", "lignum-compiled", "lleaves", "tl2cgen"],
    "xgboost": ["native", "lignum", "lignum-compiled", "tl2cgen"],
    "catboost": ["native", "lignum"],
}
SMALL = [1, 10, 100]
LARGE = [10_000, 100_000, 1_000_000]
DISTINCT = 1000  # distinct small batches in rotation


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------

@dataclass
class Model:
    library: str
    dataset: str
    task: str
    n_classes: int
    path: Path
    split: int
    n_trees: int = 0
    avg_path: float | None = None
    train_s: float | None = None
    sha256: str = ""


def train(library: str, d: datasets.Dataset, rounds: int) -> Model:
    ext = {"lightgbm": "txt", "xgboost": "json", "catboost": "cbm"}[library]
    path = CACHE / f"{library}_{d.name}_r{rounds}.{ext}"
    split = int(len(d.y) * 0.8)
    m = Model(library, d.name, d.task, d.n_classes if d.task == "multiclass" else 1, path, split)
    if path.exists():
        return m
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"   training {library} on {d.name} ({rounds} rounds) ...", flush=True)
    t0 = time.perf_counter()
    if library == "lightgbm":
        import lightgbm as lgb

        params = {"num_leaves": 63, "max_depth": -1, "learning_rate": 0.1, "seed": 7, "verbose": -1,
                  "max_cat_to_onehot": 4, "max_cat_threshold": 64, "min_data_per_group": 20}
        if d.task == "multiclass":
            params.update(objective="multiclass", num_class=d.n_classes)
        else:
            params["objective"] = "binary" if d.task == "binary" else "regression"
        lgb.train(params, lgb.Dataset(d.frame.iloc[:split], d.y[:split]), num_boost_round=rounds).save_model(str(path))
    elif library == "xgboost":
        import xgboost as xgb

        params = {"tree_method": "hist", "grow_policy": "lossguide", "max_leaves": 63, "max_depth": 0,
                  "learning_rate": 0.1, "seed": 7, "max_cat_to_onehot": 4}
        if d.task == "multiclass":
            params.update(objective="multi:softprob", num_class=d.n_classes)
        else:
            params["objective"] = "binary:logistic" if d.task == "binary" else "reg:squarederror"
        dtrain = xgb.DMatrix(d.frame.iloc[:split], d.y[:split], enable_categorical=True)
        xgb.train(params, dtrain, num_boost_round=rounds).save_model(str(path))
    else:
        import catboost as cb

        frame = datasets.catboost_frame(d)
        loss = {"binary": "Logloss", "regression": "RMSE", "multiclass": "MultiClass"}[d.task]
        model = cb.CatBoost({"iterations": rounds, "depth": 6, "learning_rate": 0.1, "loss_function": loss,
                             "random_seed": 7, "verbose": False, "thread_count": N_THREADS})
        cats = [frame.columns[j] for j in d.categorical]
        model.fit(cb.Pool(frame.iloc[:split], d.y[:split], cat_features=cats))
        model.save_model(str(path))
    m.train_s = time.perf_counter() - t0
    return m


def _lightgbm_depths(tree: dict) -> dict[int, int]:
    depths, stack = {}, [(tree["tree_structure"], 0)]
    while stack:
        node, depth = stack.pop()
        if "leaf_index" in node:
            depths[node["leaf_index"]] = depth
        elif "left_child" in node:
            stack += [(node["left_child"], depth + 1), (node["right_child"], depth + 1)]
        else:  # a tree with a single leaf
            depths[0] = 0
    return depths


def _xgboost_depths(tree: dict) -> dict[int, int]:
    depths, stack = {}, [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if "children" in node:
            stack += [(child, depth + 1) for child in node["children"]]
        else:
            depths[node["nodeid"]] = depth
    return depths


def describe(m: Model, X: np.ndarray) -> None:
    """Tree count and the average number of comparisons per tree on X (held-out rows)."""
    m.sha256 = hashlib.sha256(m.path.read_bytes()).hexdigest()
    if m.library == "lightgbm":
        import lightgbm as lgb

        b = lgb.Booster(model_file=str(m.path))
        trees = b.dump_model()["tree_info"]
        leaves = b.predict(X, pred_leaf=True).reshape(len(X), -1)
        depths = [_lightgbm_depths(t) for t in trees]
    elif m.library == "xgboost":
        import xgboost as xgb

        b = xgb.Booster(model_file=str(m.path))
        trees = [json.loads(t) for t in b.get_dump(dump_format="json")]
        dm = xgb.DMatrix(X, feature_names=b.feature_names, feature_types=b.feature_types, enable_categorical=True)
        leaves = b.predict(dm,
                           pred_leaf=True).astype(np.int64).reshape(len(X), -1)
        depths = [_xgboost_depths(t) for t in trees]
    else:
        import catboost as cb

        b = cb.CatBoost()
        b.load_model(str(m.path))
        m.n_trees, m.avg_path = b.tree_count_, 6.0  # symmetric trees: depth 6 on every path
        return
    m.n_trees = len(trees)
    m.avg_path = float(np.mean([[depths[t][leaf] for t, leaf in enumerate(row)] for row in leaves]))


# ---------------------------------------------------------------------------------------------
# Engines: build(model, tmp) -> predict(data, n_threads), outputs as the library's predict
# ---------------------------------------------------------------------------------------------

Predict = Callable[[Any, int], np.ndarray]


class NotSupported(Exception):
    pass


def engine_native(m: Model, tmp: Path) -> Predict:
    if m.library == "lightgbm":
        import lightgbm as lgb

        b = lgb.Booster(model_file=str(m.path))
        return lambda X, n: b.predict(X, num_threads=n)
    if m.library == "xgboost":
        import xgboost as xgb

        boosters = {}

        def predict(X, n):  # one booster per thread count; no DMatrix per call
            if n not in boosters:
                boosters[n] = xgb.Booster(model_file=str(m.path))
                boosters[n].set_param({"nthread": n})
            return boosters[n].inplace_predict(X)

        return predict
    import catboost as cb

    b = cb.CatBoost()
    b.load_model(str(m.path))
    kind = "RawFormulaVal" if m.task == "regression" else "Probability"
    return lambda frame, n: b.predict(frame, prediction_type=kind, thread_count=n)


def engine_lignum(m: Model, tmp: Path) -> Predict:
    import lignum

    model = lignum.load(str(m.path))
    return lambda X, n: model.predict(X, n_jobs=n)


def engine_lignum_compiled(m: Model, tmp: Path) -> Predict:
    import lignum

    model = lignum.load(str(m.path))
    model.compile(library=tmp / f"{m.library}_{m.dataset}_lignum.lib")
    model.compiled_min_rows = 1  # the compiled trees for every batch size, not autotune's choice
    return lambda X, n: model.predict(X, n_jobs=n)


def engine_lleaves(m: Model, tmp: Path) -> Predict:
    try:
        import lleaves
    except ImportError as e:
        raise NotSupported("lleaves is not installed on this platform") from e

    model = lleaves.Model(model_file=str(m.path))
    model.compile(cache=str(tmp / f"{m.dataset}_lleaves.o"))
    return lambda X, n: model.predict(X, n_jobs=n)


def engine_tl2cgen(m: Model, tmp: Path) -> Predict:
    try:
        import tl2cgen
        import treelite
    except ImportError as e:
        raise NotSupported("tl2cgen is not installed on this platform") from e

    if m.library == "lightgbm":
        model = treelite.frontend.load_lightgbm_model(str(m.path))
    else:
        try:
            model = treelite.frontend.load_xgboost_model(str(m.path))
        except treelite.TreeliteError as e:
            if "category_container" in str(e):
                raise NotSupported("Treelite does not load XGBoost models trained on pandas categories") from e
            raise
    libpath = tmp / f"{m.library}_{m.dataset}_tl2cgen{'.dylib' if sys.platform == 'darwin' else '.so'}"
    tl2cgen.export_lib(model, toolchain="clang" if sys.platform == "darwin" else "gcc", libpath=str(libpath),
                       params={"parallel_comp": N_THREADS}, options=["-O3"], nthread=N_THREADS)
    predictors = {}

    def predict(X, n):
        if n not in predictors:
            predictors[n] = tl2cgen.Predictor(str(libpath), nthread=n)
        out = predictors[n].predict(tl2cgen.DMatrix(X))
        return out.reshape(len(X), -1)[:, -1] if m.n_classes == 1 else out.reshape(len(X), m.n_classes)

    return predict


BUILDERS = {"native": engine_native, "lignum": engine_lignum, "lignum-compiled": engine_lignum_compiled,
            "lleaves": engine_lleaves, "tl2cgen": engine_tl2cgen}


# ---------------------------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------------------------

@dataclass
class Result:
    library: str
    dataset: str
    engine: str
    source: str = "matrix"
    prep_s: float | None = None
    mismatch_holdout: int | None = None
    mismatch_train: int | None = None
    max_abs_diff: float | None = None
    small: dict = field(default_factory=dict)  # "<threads>/<rows>": {p50_us, p99_us, p999_us, calls}
    large: dict = field(default_factory=dict)  # "<threads>/<rows>": {best_s, median_s, rows_per_s, repeats}
    error: str | None = None


def as_rows(out, n: int, n_classes: int) -> np.ndarray:
    out = np.asarray(out, dtype=np.float64).reshape(n, -1)
    return out[:, -1] if n_classes == 1 else out  # binary probabilities: the positive class


def compare(got, ref, n_classes: int) -> tuple[int, float]:
    got, ref = as_rows(got, len(ref), n_classes), as_rows(ref, len(ref), n_classes)
    diff = got != ref
    return int(np.sum(diff.reshape(len(ref), -1).any(axis=1))), float(np.max(np.abs(got - ref)))


def latency(predict: Predict, pool: list, n: int, calls: int, max_s: float) -> dict:
    """Percentiles of single calls, rotating through the pool of distinct inputs."""
    for i in range(min(len(pool), 1000)):
        predict(pool[i], n)
    times = np.empty(calls, dtype=np.int64)
    k, count, deadline = len(pool), 0, time.perf_counter() + max_s
    gc.disable()
    try:
        while count < calls and (count < 10_000 or time.perf_counter() < deadline):
            batch = pool[count % k]
            t0 = time.perf_counter_ns()
            predict(batch, n)
            times[count] = time.perf_counter_ns() - t0
            count += 1
    finally:
        gc.enable()
    t = times[:count] / 1000
    return {"p50_us": float(np.percentile(t, 50)), "p99_us": float(np.percentile(t, 99)),
            "p999_us": float(np.percentile(t, 99.9)), "calls": count}


def throughput(predict: Predict, windows: list, n: int, rows: int, max_repeats: int, max_s: float) -> dict:
    """Best and median of repeated calls, each on the next window of the data (at least 3)."""
    first = windows[0]
    predict(first.slice(0, 1000) if hasattr(first, "slice") else first[:1000], n)
    times, deadline = [], time.perf_counter() + max_s
    while len(times) < max_repeats and (len(times) < 3 or time.perf_counter() < deadline):
        data = windows[len(times) % len(windows)]
        t0 = time.perf_counter()
        predict(data, n)
        times.append(time.perf_counter() - t0)
    median = statistics.median(times)
    return {"best_s": min(times), "median_s": median, "rows_per_s": rows / median, "repeats": len(times)}


@dataclass
class Inputs:
    holdout: Any
    train: Any
    small: dict[int, list]
    large: dict[int, list]


def make_inputs(m: Model, d: datasets.Dataset, source: str, args) -> Inputs:
    """Inputs as the engines take them. source "matrix": the float64 matrix for LightGBM / XGBoost,
    CatBoost's DataFrame for CatBoost; "arrow": a pyarrow Table of the same rows (categories as
    dictionary columns)."""
    import pandas as pd

    frame = datasets.catboost_frame(d) if m.library == "catboost" else d.frame
    if source == "arrow":
        import pyarrow as pa

        data = pa.Table.from_pandas(frame, preserve_index=False).combine_chunks()
        take = lambda src, s, size: src.slice(s, size)
        tile = lambda reps: pa.concat_tables([data] * reps).combine_chunks()
    elif m.library == "catboost":
        data = frame
        take = lambda src, s, size: src.iloc[s:s + size].reset_index(drop=True)
        tile = lambda reps: pd.concat([data] * reps, ignore_index=True)
    else:
        data = d.X
        take = lambda src, s, size: np.ascontiguousarray(src[s:s + size])
        tile = lambda reps: np.concatenate([data] * reps)
    n = len(data)

    rng = np.random.default_rng(0)
    small = {b: [take(data, int(s), b) for s in rng.integers(0, n - b, DISTINCT)] for b in args.small}
    largest = max(args.large) if args.large else 0
    tiled = tile(-(-largest // n)) if largest > n else data
    total = len(tiled)
    large = {}
    for b in args.large:
        # Five windows up to 100k rows; one for 1M (caches do not matter there, memory does).
        starts = [0] if b >= total or b > 100_000 else [int(s) for s in rng.integers(0, total - b, 5)]
        large[b] = [take(tiled, s, b) for s in starts]
    holdout = take(data, m.split, min(n - m.split, 100_000))
    train_rows = take(data, 0, min(m.split, 100_000))
    return Inputs(holdout, train_rows, small, large)


def from_arrow(m: Model, engine: str, predict: Predict) -> Predict:
    """An engine fed with Arrow tables, converting as its users would have to: lignum reads them
    directly; LightGBM and XGBoost read tables without dictionary columns and go through pandas
    otherwise; CatBoost and lleaves take pandas only."""
    import pyarrow.types as pa_types

    if engine.startswith("lignum"):
        return predict
    if engine in ("native",) and m.library in ("lightgbm", "xgboost"):
        def native(t, n):
            if any(pa_types.is_dictionary(f.type) for f in t.schema):
                t = t.to_pandas()
            return predict(t, n)
        return native
    return lambda t, n: predict(t.to_pandas(), n)


ARROW_ENGINES = {
    "lightgbm": ["native", "lignum", "lignum-compiled", "lleaves"],
    "xgboost": ["native", "lignum", "lignum-compiled"],
    "catboost": ["native", "lignum"],
}


def run_engine(m: Model, engine: str, source: str, inputs: Inputs, reference: dict, args, tmp: Path) -> Result:
    r = Result(m.library, m.dataset, engine, source)
    try:
        t0 = time.perf_counter()
        predict = BUILDERS[engine](m, tmp)
        r.prep_s = time.perf_counter() - t0
        if source == "arrow":
            predict = from_arrow(m, engine, predict)
        r.mismatch_holdout, d1 = compare(predict(inputs.holdout, N_THREADS), reference["holdout"], m.n_classes)
        r.mismatch_train, d2 = compare(predict(inputs.train, N_THREADS), reference["train"], m.n_classes)
        r.max_abs_diff = max(d1, d2)
        for threads in sorted({1, N_THREADS}):
            for b, pool in inputs.small.items():
                r.small[f"{threads}/{b}"] = latency(predict, pool, threads, args.calls, args.max_seconds)
            for b, windows in inputs.large.items():
                repeats = {10_000: 50, 100_000: 10}.get(b, 5)
                r.large[f"{threads}/{b}"] = throughput(predict, windows, threads, b, repeats, args.max_seconds)
    except NotSupported as e:
        r.error = f"n/a ({e})"
    except Exception as e:  # report and continue with the other engines
        r.error = f"{type(e).__name__}: {e}".splitlines()[0][:200]
        if args.verbose:
            traceback.print_exc()
    return r


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def us(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def secs(v: float) -> str:
    return f"{v:.4f}" if v < 0.1 else f"{v:.3f}"


def render(models: list[Model], results: list[Result], env: dict, args) -> str:
    lines = [f"# Benchmark suite {env['date']}", "",
             f"- lignum commit `{env['commit']}`, {env['cpu']}, {env['threads']} threads, Python {env['python']}",
             "- versions: " + ", ".join(f"{k} {v}" for k, v in env["versions"].items() if v),
             f"- {args.rounds} rounds; LightGBM num_leaves 63; XGBoost lossguide max_leaves 63; CatBoost depth 6",
             "- latency: one call, rows differ on every call (p50 / p99 / p99.9, µs); "
             "throughput: median of repeats on different windows (s); prep: load + compile, cold",
             "- exact: rows not bit-identical to the library's predict (held-out / training rows)",
             "- input: `matrix` = float64 matrix (LightGBM, XGBoost) or DataFrame (CatBoost); `arrow` = pyarrow "
             "Table, converted the way each engine needs (to_pandas for dictionary columns, CatBoost and lleaves)", ""]
    for lib in args.libraries:
        for name in args.datasets:
            m = next((x for x in models if x.library == lib and x.dataset == name), None)
            if m is None or not any(r.library == lib and r.dataset == name for r in results):
                continue
            lines += [f"## {lib} / {name}", "",
                      f"{m.n_trees} trees, average path {m.avg_path:.2f} comparisons per tree, "
                      f"model sha256 `{m.sha256[:12]}`", ""]
            for source, threads in [(s, t) for s in args.sources for t in sorted({1, N_THREADS})]:
                rows = [r for r in results if r.library == lib and r.dataset == name and r.source == source]
                if not rows:
                    continue
                head = [f"{b} row{'s' if b > 1 else ''}" for b in args.small] + [f"{b:,} rows" for b in args.large]
                lines += [f"{source} input, {'one thread' if threads == 1 else f'{threads} threads'}:", "",
                          "| engine | prep [s] | exact | " + " | ".join(head) + " |",
                          "|---|---:|---:|" + "---:|" * len(head)]
                for r in rows:
                    if r.error:
                        lines.append(f"| {r.engine} | {r.error} |" + " |" * (len(head) + 1))
                        continue
                    exact = "yes" if r.mismatch_holdout == 0 and r.mismatch_train == 0 else \
                        f"no ({r.mismatch_holdout:,} / {r.mismatch_train:,}, {r.max_abs_diff:.1e})"
                    cells = []
                    for b in args.small:
                        s = r.small[f"{threads}/{b}"]
                        cells.append(f"{us(s['p50_us'])} / {us(s['p99_us'])} / {us(s['p999_us'])}")
                    for b in args.large:
                        cells.append(secs(r.large[f"{threads}/{b}"]["median_s"]))
                    lines.append(f"| {r.engine} | {r.prep_s:.2f} | {exact} | " + " | ".join(cells) + " |")
                lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--libraries", default=",".join(LIBRARIES))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--engines", default=None, help="comma-separated subset (default: all for each library)")
    parser.add_argument("--sources", default="matrix,arrow", help="inputs: matrix, arrow")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--calls", type=int, default=100_000, help="calls per small-batch measurement")
    parser.add_argument("--max-seconds", type=float, default=20.0,
                        help="time budget per measurement (small batches still make 10,000 calls, large ones 3)")
    parser.add_argument("--quick", action="store_true", help="1, 100 and 10k rows, 10,000 calls")
    parser.add_argument("--train-only", action="store_true", help="train the missing models and stop")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.libraries = args.libraries.split(",")
    args.datasets = args.datasets.split(",")
    args.sources = args.sources.split(",")
    args.small, args.large = SMALL, LARGE
    if args.quick:
        args.small, args.large, args.calls, args.max_seconds = [1, 100], [10_000], 10_000, 2.0
    wanted = set(args.engines.split(",")) if args.engines else None

    env = bench.environment()
    env["versions"]["catboost"] = _version("catboost")
    models, results = [], []
    with tempfile.TemporaryDirectory(prefix="lignum_suite_") as tmpdir:
        tmp = Path(tmpdir)
        for name in args.datasets:
            d = datasets.LOADERS[name]()
            for lib in args.libraries:
                m = train(lib, d, args.rounds)
                if args.train_only:
                    continue
                describe(m, np.ascontiguousarray(d.X[m.split:m.split + 10_000]))
                models.append(m)
                reference = None
                for source in args.sources:
                    print(f"== {lib} / {name} / {source} input", flush=True)
                    inputs = make_inputs(m, d, source, args)
                    if reference is None:  # the same rows in every source: one reference
                        native = engine_native(m, tmp)
                        matrix = inputs if source == "matrix" else make_inputs(m, d, "matrix", args)
                        reference = {"holdout": native(matrix.holdout, N_THREADS), "train": native(matrix.train, N_THREADS)}
                        del matrix
                    for engine in (ENGINES if source == "matrix" else ARROW_ENGINES)[lib]:
                        if wanted and engine not in wanted:
                            continue
                        print(f"   {engine} ...", end=" ", flush=True)
                        t0 = time.perf_counter()
                        r = run_engine(m, engine, source, inputs, reference, args, tmp)
                        print(r.error or f"done in {time.perf_counter() - t0:.0f}s", flush=True)
                        results.append(r)
                    del inputs
                    gc.collect()

    if args.train_only:
        print(f"Models in {CACHE}")
        return
    report = render(models, results, env, args)
    print("\n" + report)
    RESULTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}-suite.md").write_text(report)
    payload = {"env": env, "args": {k: v for k, v in vars(args).items()},
               "models": [{**m.__dict__, "path": m.path.name} for m in models], "results": [r.__dict__ for r in results]}
    (RESULTS / f"{stamp}-suite.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {RESULTS / (stamp + '-suite.md')}")


def _version(pkg: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
