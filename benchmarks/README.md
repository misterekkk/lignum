# Benchmarks

`bench.py` compares lignum with native LightGBM/XGBoost, TL2cgen, Treelite GTIL, lleaves and ONNX
Runtime: preparation time, large-batch throughput (all threads, one thread), latency for 1 and 100
rows (median, and p99 for one row on one thread) and bit-exactness. Reports go to `results/`.

## Running on a cloud server

`cloud.sh` does everything on a fresh Ubuntu 24.04 server (e.g. Hetzner CCX with dedicated vCPUs, or
CAX for ARM): installs the toolchain, builds lignum, runs `bench.py` (interpreter, compiled, lleaves,
native; real and synthetic models), `bench_arrow.py`, the host-tuned build and `autotune`, and packs
the reports into `~/lignum-bench-<host>-<date>.tar.gz`. About 30-40 minutes.

```bash
curl -LsSf https://raw.githubusercontent.com/misterekkk/lignum/dev/v0.2/benchmarks/cloud.sh | bash
```

## Running on a dedicated machine

Linux (or WSL2 on Windows; lleaves and TL2cgen are not supported natively on Windows):

```bash
git clone https://github.com/misterekkk/lignum.git && cd lignum && git checkout dev/v0.2
sudo apt install -y build-essential cmake            # compiler + CMake
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv (Python environments)
make venv bench-deps                                 # .venv with lignum's test deps and the other engines
make py-build                                        # lignum itself

# Main comparison: synthetic + real data, against lleaves (the strongest engine) and TL2cgen.
.venv/bin/python benchmarks/bench.py --engines lignum,lleaves,tl2cgen-tuned \
    --only lightgbm,lightgbm-cat,lightgbm-real-covtype,lightgbm-real-diabetes,lightgbm-real-adult

# Host-tuned lignum (lleaves always compiles for the host CPU):
CMAKE_ARGS="-DLIGNUM_NATIVE=ON" make py-build
.venv/bin/python benchmarks/bench.py --engines lignum \
    --only lightgbm,lightgbm-cat,lightgbm-real-covtype,lightgbm-real-diabetes,lightgbm-real-adult
```

For stable numbers: plug the laptop in, close other programs, and on Linux set the performance
governor (`sudo cpupower frequency-set -g performance`). The first run downloads the datasets and
trains the models (a few minutes); later runs reuse `benchmarks/.cache`.

Other tools: `batch_sweep.py` (latency vs batch size and strategy), `run_matrix.sh` (model shapes:
shallow, deep, large, wide), `datasets.py` (real datasets).
