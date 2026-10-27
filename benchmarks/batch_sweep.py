"""Latency/throughput of lignum across batch sizes, thread counts and strategies.

Uses the cached models from bench.py (run `python benchmarks/bench.py --quick` or the full run first).

    python benchmarks/batch_sweep.py [--cache benchmarks/.cache/<key>] [--strategies auto,rows,trees,serial]
"""

import argparse
import inspect
import os
import statistics
import time
from pathlib import Path

import numpy as np

import lignum

HERE = Path(__file__).resolve().parent
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]


def median_time(fn, budget=0.4, min_runs=20, max_runs=3000):
    fn()
    times = []
    deadline = time.perf_counter() + budget
    while len(times) < min_runs or (time.perf_counter() < deadline and len(times) < max_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=None)
    parser.add_argument("--model", default="lgb.json")
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--threads", default=f"1,{os.cpu_count()}")
    args = parser.parse_args()

    cache = Path(args.cache) if args.cache else max((HERE / ".cache").iterdir(), key=lambda p: p.stat().st_size if p.is_dir() else 0)
    if not args.cache:
        cache = sorted((HERE / ".cache").iterdir(), key=lambda p: (p / "data.npz").stat().st_size)[-1]
    X = np.load(cache / "data.npz")["X"]
    fmt = "lightgbm" if args.model.startswith("lgb") else "xgboost"
    model = lignum.load(str(cache / args.model), format=fmt)
    model.validate_features = False  # benchmarks measure prediction alone: no column-name check

    has_strategy = "strategy" in inspect.signature(model.predict).parameters
    strategies = args.strategies.split(",") if args.strategies else (["auto", "rows", "trees", "serial"] if has_strategy else [None])
    threads = [int(t) for t in args.threads.split(",")]

    print(f"model {cache.name}/{args.model}; µs per call (median); per-row µs in brackets")
    header = "rows".rjust(7) + "".join(f"{(s or 'default') + '/' + str(t):>20}" for s in strategies for t in threads)
    print(header)
    for n in SIZES:
        Xn = np.ascontiguousarray(X[:n])
        cells = []
        for s in strategies:
            for t in threads:
                kwargs = {"n_jobs": t}
                if s:
                    kwargs["strategy"] = s
                us = median_time(lambda: model.predict(Xn, **kwargs)) * 1e6
                cells.append(f"{us:>11,.1f} [{us / n:6.2f}]")
        print(f"{n:>7}" + "".join(f"{c:>20}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
