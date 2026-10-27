"""CatBoost models on real and synthetic data: lignum against CatBoost, from every input.

Usage:
    python benchmarks/bench_catboost.py                          # adult, bank, diabetes, california, covtype,
                                                                 # synthetic, synthetic-cat (bench.py's data)
    python benchmarks/bench_catboost.py --datasets adult --quick
    python benchmarks/bench_catboost.py --no-capi                # Python only

For each data set (benchmarks/datasets.py) it trains CatBoost (500 trees of depth 6, categorical
columns as cat_features, missing categories as "NA") on 80% of the rows, and measures raw scores:

  Python    CatBoost from a pandas DataFrame, a prebuilt Pool (its build time is reported apart),
            polars (Categorical and String columns), and pyarrow (CatBoost does not take Arrow: `to_pandas()` is part of the time);
            lignum from pandas, pyarrow, polars and a float64 matrix of category hashes.
            Batch with all threads and with one thread, 1 row (from a list and from a DataFrame),
            100 rows.
  C API     CatBoost's libcatboostmodel (downloaded once from its GitHub release) against lignum's
            C API, from category strings and from precomputed hashes (benchmarks/catboost_capi.cpp).

The rows are repeated up to --rows so that batches are long enough to time. Every engine is checked
against CatBoost's predictions on all rows (mismatch = rows not bit-identical).
Results go to benchmarks/results/<stamp>-catboost.{md,json}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / ".cache"
RESULTS = HERE / "results"
N_THREADS = os.cpu_count() or 1
CATBOOST_RELEASE = "https://github.com/catboost/catboost/releases/download/v{v}/{name}"

sys.path.insert(0, str(HERE))
import bench  # noqa: E402  (environment() and fmt())
import datasets  # noqa: E402


@dataclass
class Row:
    dataset: str
    front: str
    engine: str
    prep_s: float | None = None
    batch_best_s: float | None = None
    batch_median_s: float | None = None
    batch1_s: float | None = None
    lat1_list_us: float | None = None
    lat1_frame_us: float | None = None
    lat100_us: float | None = None
    mismatch: int | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def train(name: str, d: datasets.Dataset, frame: pd.DataFrame, trees: int) -> Path:
    import catboost as cb

    path = CACHE / "real" / f"catboost_{name}_t{trees}.cbm"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    split = int(len(frame) * 0.8)
    loss = {"binary": "Logloss", "regression": "RMSE", "multiclass": "MultiClass"}[d.task]
    model = cb.CatBoost({"iterations": trees, "depth": 6, "loss_function": loss, "random_seed": 7, "verbose": False,
                         "thread_count": N_THREADS})
    cats = [frame.columns[j] for j in d.categorical]
    model.fit(cb.Pool(frame.iloc[:split], d.y[:split], cat_features=cats))
    model.save_model(str(path))
    return path


def tile(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    if len(frame) >= rows:
        return frame.iloc[:rows].reset_index(drop=True)
    reps = -(-rows // len(frame))
    return pd.concat([frame] * reps, ignore_index=True).iloc[:rows]


def time_batch(fn, repeats: int) -> tuple[float, float]:
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times)


def time_latency(fn, budget_s: float) -> float:
    """Median µs per call; fn(i) cycles through the first 200 rows."""
    for i in range(20):
        fn(i)
    times, i = [], 0
    deadline = time.perf_counter() + budget_s
    while len(times) < 50 or (time.perf_counter() < deadline and len(times) < 20000):
        t0 = time.perf_counter()
        fn(i % 200)
        times.append(time.perf_counter() - t0)
        i += 1
    return statistics.median(times) * 1e6


def mismatch(got, ref) -> int:
    got = np.asarray(got, dtype=np.float64).reshape(len(ref), -1)
    return int(np.sum(np.any(got != ref.reshape(len(ref), -1), axis=1)))


def python_fronts(name, model_path, frame, cats, ref, args) -> list[Row]:
    """Every input format of both engines: batches, and single rows for pandas."""
    import catboost as cb
    import polars as pl
    import pyarrow as pa

    import lignum

    booster = cb.CatBoost()
    booster.load_model(str(model_path))
    model = lignum.load(str(model_path))
    model.validate_features = False  # benchmarks measure prediction alone: no column-name check
    raw = "RawFormulaVal"

    def cb_predict(data, n):
        return booster.predict(data, prediction_type=raw, thread_count=n)

    table = pa.Table.from_pandas(frame, preserve_index=False)
    polars_frame = pl.from_arrow(table)
    polars_strings = polars_frame.with_columns(pl.col(cats).cast(pl.String))
    rows_list = frame.iloc[:200].astype(object).values.tolist()
    small = [frame.iloc[i:i + 1] for i in range(200)]
    hundred = frame.iloc[:100]

    fronts = []
    t0 = time.perf_counter()
    pool = cb.Pool(frame, cat_features=cats)
    pool_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    hashed = model._catboost_matrix(frame)
    hashed_s = time.perf_counter() - t0

    fronts.append(("catboost", "pandas", None, cb_predict, frame))
    fronts.append(("catboost", "Pool (prebuilt)", pool_s, cb_predict, pool))
    if cats and int(pl.__version__.split(".")[0]) >= 2:
        # CatBoost 1.2 reads polars Categorical columns through get_categories, removed in polars
        # 2.0: an error on Linux and macOS, an access violation inside CatBoost on Windows
        # (catboost/core.py Pool._init; it killed the Windows CI job at random). Not called.
        def cb_polars_categorical(data, n):
            raise RuntimeError("not supported: CatBoost cannot read polars 2.x Categorical columns (see String columns)")
        fronts.append(("catboost", "polars", None, cb_polars_categorical, polars_frame))
    else:
        fronts.append(("catboost", "polars", None, cb_predict, polars_frame))
    fronts.append(("catboost", "polars (String columns)", None, cb_predict, polars_strings))
    fronts.append(("catboost", "pyarrow -> to_pandas()", None, lambda t, n: cb_predict(t.to_pandas(), n), table))
    fronts.append(("lignum", "pandas", None, lambda X, n: model.predict(X, raw_score=True, n_jobs=n), frame))
    fronts.append(("lignum", "pyarrow", None, lambda X, n: model.predict(X, raw_score=True, n_jobs=n), table))
    fronts.append(("lignum", "polars", None, lambda X, n: model.predict(X, raw_score=True, n_jobs=n), polars_frame))
    # Python rejects float matrices for models with categories (codes would pass as hashes): the
    # prebuilt hashes go to the native model directly, as from C.
    fronts.append(("lignum", "float64 hashes (prebuilt)", hashed_s, lambda X, n: model._c_model.predict(X, True, n, "auto"), hashed))

    results = []
    for engine, front, prep, predict, data in fronts:
        r = Row(name, front, engine, prep_s=prep)
        print(f"   {engine} / {front} ...", end=" ", flush=True)
        try:
            r.mismatch = mismatch(predict(data, N_THREADS), ref)
            r.batch_best_s, r.batch_median_s = time_batch(lambda: predict(data, N_THREADS), args.repeats)
            fifth = slice(0, max(1, len(frame) // 5))
            part = data[fifth] if not isinstance(data, cb.Pool) else data.slice(list(range(fifth.stop)))
            best1, _ = time_batch(lambda: predict(part, 1), max(1, args.repeats // 2))
            r.batch1_s = best1 * len(frame) / fifth.stop
            if front == "pandas":
                predict_one = (lambda x: cb_predict(x, 1)) if engine == "catboost" else (lambda x: model.predict(x, raw_score=True, n_jobs=1))
                r.lat1_list_us = time_latency(lambda i: predict_one(rows_list[i]), args.latency_budget)
                r.lat1_frame_us = time_latency(lambda i: predict_one(small[i]), args.latency_budget)
                r.lat100_us = time_latency(lambda i: predict_one(hundred), args.latency_budget)
                one = np.asarray([predict_one(rows_list[i]) for i in range(200)]).reshape(200, -1)
                r.mismatch += mismatch(one, ref[:200])
            print("done", flush=True)
        except Exception as e:  # report and continue with the other fronts
            r.error = f"{type(e).__name__}: {e}".splitlines()[0][:160]
            print(r.error, flush=True)
        results.append(r)
    return results


# ---------------------------------------------------------------------------------------------
# C API
# ---------------------------------------------------------------------------------------------

def catboost_library() -> Path:
    import catboost

    v = catboost.__version__
    name = {"darwin": f"libcatboostmodel-darwin-universal2-{v}.dylib",
            "linux": f"libcatboostmodel-linux-{platform.machine().replace('arm64', 'aarch64')}-{v}.so"}.get(sys.platform)
    if name is None:
        raise RuntimeError(f"no libcatboostmodel for {sys.platform}")
    path = CACHE / "catboost" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"   downloading {name}", flush=True)
        with urllib.request.urlopen(CATBOOST_RELEASE.format(v=v, name=name), timeout=300) as response:
            path.write_bytes(response.read())
    return path


def build_capi(cmake_args: list[str]) -> Path:
    """liblignum (C API, Release, plus cmake_args) and benchmarks/catboost_capi.cpp, in
    benchmarks/.cache/capi; one build directory per set of arguments."""
    build = CACHE / "capi"
    joined = " ".join(cmake_args)
    tag = "".join(ch if ch.isalnum() else "_" for ch in joined)[:40] + "-" + hashlib.sha1(joined.encode()).hexdigest()[:8]
    libdir = build / ("lignum" + (f"-{tag}" if joined else ""))
    subprocess.run(["cmake", "-S", str(ROOT), "-B", str(libdir), "-DCMAKE_BUILD_TYPE=Release", *cmake_args], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", str(libdir), "--target", "lignum_c", "--parallel"], check=True,
                   stdout=subprocess.DEVNULL)
    exe = libdir / "catboost_capi"
    subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++20", "-I", str(ROOT / "include"),
                    str(HERE / "catboost_capi.cpp"), "-o", str(exe), f"-L{libdir}", "-llignum",
                    f"-Wl,-rpath,{libdir}", "-ldl", "-pthread"], check=True)
    return exe


def write_capi_data(path: Path, frame: pd.DataFrame, cats: list[int], ref: np.ndarray, dim: int):
    float_cols = [j for j in range(frame.shape[1]) if j not in cats]
    floats = np.ascontiguousarray(frame.iloc[:, float_cols].to_numpy(dtype=np.float64, na_value=np.nan))
    text = frame.iloc[:, cats].astype(str).to_numpy().ravel() if cats else np.array([], dtype=object)
    encoded = [s.encode() for s in text]
    offsets = np.zeros(len(encoded) + 1, dtype=np.uint64)
    np.cumsum([len(b) for b in encoded], out=offsets[1:])
    with open(path, "wb") as f:
        f.write(b"LCB1")
        f.write(np.array([len(frame)], dtype=np.uint64).tobytes())
        f.write(np.array([frame.shape[1], len(float_cols), len(cats), dim], dtype=np.uint32).tobytes())
        f.write(np.array(float_cols + cats, dtype=np.uint32).tobytes())
        f.write(floats.tobytes())
        f.write(offsets.tobytes())
        f.write(b"".join(encoded))
        f.write(np.ascontiguousarray(ref, dtype=np.float64).tobytes())


def capi_rows(name, exe, lib, model_path, frame, cats, ref, dim, args) -> list[Row]:
    data = CACHE / "capi" / f"{name}_{len(frame)}.lcb"
    write_capi_data(data, frame, cats, ref, dim)
    print("   C API ...", end=" ", flush=True)
    proc = subprocess.run([str(exe), str(model_path), str(data), str(lib), str(N_THREADS), str(args.repeats)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr.strip(), flush=True)
        return [Row(name, "C API", "—", error=proc.stderr.strip()[:160])]
    print("done", flush=True)
    rows = []
    for line in proc.stdout.splitlines():
        c = json.loads(line)
        engine, front = c["case"].split(" ", 1)
        r = Row(name, "C API: " + front, engine, mismatch=c["mismatch"])
        if "best_s" in c:
            r.batch_best_s = c["best_s"]
        else:
            r.lat1_list_us = c["median_us"]
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------------------------

def render(results: list[Row], env: dict, args, shapes: dict) -> str:
    fmt = bench.fmt
    versions = dict(env["versions"])
    try:
        import catboost
        import polars
        import pyarrow
        versions.update(catboost=catboost.__version__, polars=polars.__version__, pyarrow=pyarrow.__version__)
    except ImportError:
        pass
    lines = [
        f"# CatBoost benchmark {env['date']}" + (f" ({args.label})" if args.label else ""),
        "",
        f"- lignum commit `{env['commit']}`, {env['cpu']}, {env['threads']} threads, Python {env['python']}",
        f"- CatBoost models: {args.trees} trees of depth 6 trained on 80% of each data set; rows repeated up to {args.rows:,}",
        "- versions: " + ", ".join(f"{k} {v}" for k, v in versions.items() if v and k not in ("treelite", "tl2cgen", "lleaves", "onnxruntime", "onnxmltools")),
        "",
        "Raw scores. Batch = all rows, all threads (best / median); 1 thread is normalised to all rows. "
        "Prep = building the prebuilt input (Pool, hash matrix) once. Latency = median per call, one thread. "
        "Mismatch = rows not bit-identical to CatBoost's Python predictions (batch and single rows).",
        "",
    ]
    for ds in dict.fromkeys(r.dataset for r in results):
        lines += [f"## {ds} ({shapes[ds]})", "",
                  "| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in (r for r in results if r.dataset == ds and not r.front.startswith("C API")):
            if r.error:
                lines.append(f"| {r.engine} | {r.front} | {r.error} |" + " |" * 7)
                continue
            lines.append(f"| {r.engine} | {r.front} | {fmt(r.prep_s, 's')} | {fmt(r.batch_best_s, 's')} | {fmt(r.batch_median_s, 's')} | "
                         f"{fmt(r.batch1_s, 's')} | {fmt(r.lat1_list_us, 'us')} | {fmt(r.lat1_frame_us, 'us')} | {fmt(r.lat100_us, 'us')} | {r.mismatch:,} |")
        capi = [r for r in results if r.dataset == ds and r.front.startswith("C API")]
        if capi:
            lines += ["", "C API (CatBoost's C API has no thread setting: \"threads\" splits the rows over std::threads; "
                      "lignum hashed uses its own threads):", "",
                      "| engine | case | batch best [s] | median per call [µs] | mismatch |", "|---|---|---:|---:|---:|"]
            for r in capi:
                if r.error:
                    lines.append(f"| — | {r.error} | | | |")
                    continue
                lines.append(f"| {r.engine} | {r.front[len('C API: '):]} | {fmt(r.batch_best_s, 's')} | {fmt(r.lat1_list_us, 'us')} | {r.mismatch:,} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="adult,bank,diabetes,california,covtype,synthetic,synthetic-cat")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--latency-budget", type=float, default=1.0)
    parser.add_argument("--no-capi", action="store_true")
    parser.add_argument("--cmake-args", default="", help="extra CMake arguments for the C API build, e.g. --cmake-args=-DLIGNUM_NATIVE=ON")
    parser.add_argument("--label", default="", help="build name for the report title")
    parser.add_argument("--quick", action="store_true", help="100k rows, 3 repeats")
    args = parser.parse_args()
    if args.quick:
        args.rows, args.repeats, args.latency_budget = 100_000, 3, 0.3

    import catboost as cb

    exe = lib = None
    if not args.no_capi:
        lib = catboost_library()
        exe = build_capi(args.cmake_args.split())

    env = bench.environment()
    results, shapes = [], {}
    for name in args.datasets.split(","):
        d = datasets.LOADERS[name]()
        frame = datasets.catboost_frame(d)
        model_path = train(name, d, frame, args.trees)
        data = tile(frame, args.rows)
        cats = [frame.columns[j] for j in d.categorical]
        shapes[name] = f"{len(frame):,} rows, {frame.shape[1] - len(cats)} numeric + {len(cats)} categorical columns, {d.task}"
        print(f"== {name}: {shapes[name]}", flush=True)
        booster = cb.CatBoost()
        booster.load_model(str(model_path))
        ref = np.asarray(booster.predict(cb.Pool(data, cat_features=cats), prediction_type="RawFormulaVal"), dtype=np.float64)
        dim = 1 if ref.ndim == 1 else ref.shape[1]
        results += python_fronts(name, model_path, data, cats, ref, args)
        if exe:
            results += capi_rows(name, exe, lib, model_path, data, list(d.categorical), ref, dim, args)

    report = render(results, env, args, shapes)
    print("\n" + report)
    RESULTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}-catboost.md").write_text(report)
    (RESULTS / f"{stamp}-catboost.json").write_text(json.dumps(
        {"env": env, "args": vars(args), "results": [r.__dict__ for r in results]}, indent=2))
    print(f"\nSaved to {RESULTS / (stamp + '-catboost.md')}")


if __name__ == "__main__":
    main()
