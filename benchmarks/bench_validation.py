"""Cost of the column-name check (`validate_features`) per predict call.

Each call is timed with the check off and on, alternately; the overhead is the median of the
paired differences, so the cost of building the input and drift of the CPU cancel out. pandas
inputs are new DataFrames with names parsed from JSON on every call (as in a web service, where
the columns object is never the one checked before); "same frame" reuses one DataFrame, whose
columns object the model remembers. Arrow inputs are pyarrow Tables (checked in C++, once per call).

    python benchmarks/bench_validation.py [--trees 500] [--out benchmarks/results/<machine>/<date>-validation.md]
"""

import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

import lignum

ROWS = [1, 100, 10_000]
COLUMNS = [20, 200]


def paired(off, on, budget):
    """Median time of off and on, and the median of on - off, in microseconds."""
    off(), on()
    t_off, t_on, diff = [], [], []
    deadline = time.perf_counter() + budget
    while len(diff) < 30 or (time.perf_counter() < deadline and len(diff) < 20000):
        t0 = time.perf_counter()
        off()
        t1 = time.perf_counter()
        on()
        t2 = time.perf_counter()
        t_off.append(t1 - t0)
        t_on.append(t2 - t1)
        diff.append((t2 - t1) - (t1 - t0))
    return tuple(statistics.median(v) * 1e6 for v in (t_off, t_on, diff))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--budget", type=float, default=1.0, help="seconds per measurement")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    lines = [f"# Column-name check overhead ({platform.machine()}, {platform.system()}, lignum {lignum.__version__})", "",
             f"LightGBM, {args.trees} trees, 1 thread. Median us per predict call; overhead = median of paired (on - off).", "",
             "| columns | rows | input | check off | check on | overhead |", "|---:|---:|---|---:|---:|---:|"]
    with tempfile.TemporaryDirectory() as tmp:
        for n_columns in COLUMNS:
            names = [f"feature_{i}" for i in range(n_columns)]
            X = rng.normal(size=(20_000, n_columns))
            booster = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 63},
                                lgb.Dataset(pd.DataFrame(X, columns=names), (X[:, 0] + X[:, 1] > 0).astype(int)),
                                num_boost_round=args.trees)
            path = Path(tmp) / f"m{n_columns}.txt"
            booster.save_model(str(path))
            model = lignum.load(str(path))
            assert model.feature_names == names
            wire = json.dumps(names)
            for rows in ROWS:
                data = rng.normal(size=(rows, n_columns))
                fresh = lambda: pd.DataFrame(data, columns=json.loads(wire))
                same = pd.DataFrame(data, columns=names)
                table = pa.Table.from_pandas(same)
                cases = [
                    ("pandas, new frame", lambda: model.predict(fresh(), n_jobs=1, validate_features=False),
                     lambda: model.predict(fresh(), n_jobs=1, validate_features=True)),
                    ("pandas, same frame", lambda: model.predict(same, n_jobs=1, validate_features=False),
                     lambda: model.predict(same, n_jobs=1, validate_features=True)),
                    ("pyarrow Table", lambda: model.predict(table, n_jobs=1, validate_features=False),
                     lambda: model.predict(table, n_jobs=1, validate_features=True)),
                ]
                for label, off, on in cases:
                    t_off, t_on, overhead = paired(off, on, args.budget)
                    lines.append(f"| {n_columns} | {rows:,} | {label} | {t_off:,.1f} | {t_on:,.1f} | {overhead:+.1f} |")
                    print(lines[-1], flush=True)
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    else:
        print()
        print(text)


if __name__ == "__main__":
    main()
