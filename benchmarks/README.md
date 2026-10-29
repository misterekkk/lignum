# Benchmarks

`suite.py` is the benchmark behind the published tables: the same models (500 rounds; LightGBM
num_leaves 63, XGBoost lossguide max_leaves 63, CatBoost depth 6) on five real and two synthetic
data sets, run the same way on every platform. For LightGBM, XGBoost and CatBoost models it
compares lignum (interpreter and compiled) with the library itself, lleaves and TL2cgen where they
apply, from a float64 matrix (a DataFrame for CatBoost) and from a pyarrow Table:

- batches of 1, 10 and 100 rows: p50 / p99 / p99.9 of single calls, different rows on every call;
- batches of 10k, 100k and 1M rows: best and median of repeats, on one thread and on all threads;
- preparation time (load and compile, cold cache) and bit-exactness against the library's predict.

The first run trains the models into `.cache/suite` (`--train-only` stops there). Copy that folder
to the other machines so that every platform measures the same files; the report lists each
model's SHA-256. Reports go to `results/<stamp>-suite.{md,json}`.

```bash
make venv bench-deps py-build
.venv/bin/python benchmarks/suite.py                 # everything, about 1-2 hours
.venv/bin/python benchmarks/suite.py --quick --datasets california   # smoke run
```

## Running on a cloud server

`cloud.sh` runs the suite on a fresh Ubuntu 24.04 server (e.g. Hetzner CCX with dedicated vCPUs, or
CAX for ARM): it installs the toolchain and the other engines (TL2cgen has no ARM wheel and may be
reported as not installed there), builds lignum, checks that the data sets and models are present,
and packs the report into `~/lignum-bench-<host>-<arch>-<date>.tar.gz`.

```bash
git clone https://github.com/misterekkk/lignum.git && cd lignum
# from your machine: the data sets (figshare refuses most cloud servers) and the trained models
scp -r benchmarks/.cache/datasets benchmarks/.cache/suite root@<server-ip>:~/lignum/benchmarks/.cache/
tmux new -s bench 'bash benchmarks/cloud.sh'   # survives a dropped SSH connection
```

Detach with Ctrl-b d and reattach with `tmux attach -t bench`; the output is also in
`~/lignum-bench.log`.

For stable numbers: plug the laptop in, close other programs, and on Linux set the performance
governor (`sudo cpupower frequency-set -g performance`).

## Experiments

`bench.py` is the older, configurable comparison (synthetic shapes, ONNX Runtime, Treelite GTIL,
host-tuned builds) used by the CI benchmark workflows and `run_matrix.sh` (shallow, deep, large and
wide models); its latency also rotates through different rows. `bench_catboost.py` compares CatBoost
input formats and the C APIs (`catboost_capi.cpp`), `batch_sweep.py` latency against batch size and
strategy, `datasets.py` loads the real data sets.
