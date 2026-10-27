"""Arrow input end to end: from a pyarrow Table to predictions, conversions included.

lignum reads Arrow directly (PyCapsule interface); lleaves takes numpy or pandas only, so an Arrow
user converts first (table.to_pandas(), which keeps categorical columns); LightGBM 4.x predicts
pyarrow Tables itself. Tables are built from each data set's DataFrame, so categorical columns are
dictionary columns, as a user would have them.

Usage:
    python benchmarks/bench_arrow.py                           # all real data sets, 500 trees
    python benchmarks/bench_arrow.py --only covtype,diabetes --engines lignum,lignum-compiled,lleaves

Models come from benchmarks/.cache/real (bench.py --only lightgbm-real-<name> trains them).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import datasets  # noqa: E402

ENGINES = ["lignum", "lignum-compiled", "lleaves", "lleaves-numpy", "lightgbm"]


def timed(f, repeats):
    f()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times)


def engines_for(name, txt, js, tmp):
    import lightgbm as lgb
    import lignum
    import pyarrow.types as pa_types

    out = {}
    model = lignum.load(str(js), format="lightgbm")
    model.validate_features = False  # benchmarks measure prediction alone: no column-name check
    out["lignum"] = (lambda t, n, m=model: m.predict(t, n_jobs=n), 0.0)
    compiled = lignum.load(str(js), format="lightgbm")
    compiled.validate_features = False
    t0 = time.perf_counter()
    compiled.compile(library=tmp / "compiled" / f"{name}_lignum.lib")
    out["lignum-compiled"] = (lambda t, n, m=compiled: m.predict(t, n_jobs=n), time.perf_counter() - t0)
    try:
        import lleaves

        ll = lleaves.Model(model_file=str(txt))
        t0 = time.perf_counter()
        ll.compile(cache=str(tmp / f"{name}_lleaves.o"))
        prep = time.perf_counter() - t0
        out["lleaves"] = (lambda t, n: ll.predict(t.to_pandas(), n_jobs=n), prep)
        out["lleaves-numpy"] = ("numpy", ll, prep)  # timed on a ready matrix, for reference
    except ImportError:
        pass
    booster = lgb.Booster(model_file=str(txt))
    # LightGBM reads Arrow tables itself, except dictionary (categorical) columns: those go through pandas.
    def native(t, n):
        if any(pa_types.is_dictionary(f.type) for f in t.schema):
            return booster.predict(t.to_pandas(), num_threads=n)
        return booster.predict(t, num_threads=n)
    out["lightgbm"] = (native, 0.0)
    return out, booster


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="california,covtype,adult,bank,diabetes")
    p.add_argument("--engines", default=",".join(ENGINES))
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    import pyarrow as pa

    threads = os.cpu_count() or 1
    wanted = args.engines.split(",")
    lines = [f"# Arrow input benchmark {dt.datetime.now():%Y-%m-%dT%H:%M:%S}", "",
             f"- {platform.machine()}, {threads} threads, Python {platform.python_version()}, {args.trees} trees",
             "- Time from a pyarrow Table to predictions, conversions included (best of repeats);",
             "  `lleaves` converts with `table.to_pandas()`, `lleaves-numpy` starts from a ready float64 matrix;",
             "  `lightgbm` predicts the Table itself (through pandas when it has dictionary columns).",
             "- exact: bit-identical to LightGBM's predict on the DataFrame.",
             "- 100 rows: median per call on a 100-row RecordBatch, one thread.", ""]
    for name in args.only.split(","):
        d = datasets.LOADERS[name]()
        txt = HERE / ".cache" / "real" / f"{name}_t{args.trees}.txt"
        js = HERE / ".cache" / "real" / f"{name}_t{args.trees}.json"
        if not txt.exists():
            print(f"skip {name}: no model at {txt} (run bench.py --only lightgbm-real-{name})")
            continue
        table = pa.Table.from_pandas(d.frame, preserve_index=False)
        small = table.slice(0, 100).combine_chunks().to_batches()[0]
        with tempfile.TemporaryDirectory() as tmp:
            engines, booster = engines_for(name, txt, js, Path(tmp))
            reference = booster.predict(d.frame)
            lines += [f"## {name} ({table.num_rows:,} rows, {table.num_columns} columns)", "",
                      "| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |",
                      "|---|---:|---:|---:|---:|---:|"]
            for engine in wanted:
                if engine not in engines:
                    continue
                spec = engines[engine]
                if spec[0] == "numpy":
                    ll = spec[1]
                    from lleaves.data_processing import data_to_ndarray
                    matrix = data_to_ndarray(d.frame, ll._pandas_categorical)
                    run = lambda t, n, m=matrix: ll.predict(m, n_jobs=n)
                    run_small = lambda n, m=np.ascontiguousarray(matrix[:100]): ll.predict(m, n_jobs=n)
                    prep = spec[2]
                else:
                    run, prep = spec
                    run_small = lambda n, r=run: r(small, n)
                out = np.asarray(run(table, threads)).reshape(reference.shape)
                exact = "yes" if np.array_equal(out, reference) else f"no ({int((out != reference).any(axis=-1).sum() if out.ndim > 1 else (out != reference).sum())} rows)"
                batch = timed(lambda: run(table, threads), args.repeats)[0]
                single = timed(lambda: run(table, 1), max(2, args.repeats // 2))[0]
                latency = timed(lambda: run_small(1), 300)[1] * 1e6
                row = f"| {engine} | {prep:.2f} | {batch:.3f} | {single:.3f} | {latency:,.0f} | {exact} |"
                print(f"{name:11s} {row}", flush=True)
                lines.append(row)
            lines.append("")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{dt.datetime.now():%Y%m%d-%H%M%S}-arrow.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
