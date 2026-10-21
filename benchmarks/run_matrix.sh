#!/usr/bin/env bash
# Model-shape matrix: lignum vs the strongest engines on shapes beyond the default benchmark.
# Usage: benchmarks/run_matrix.sh [python]   (results land in benchmarks/results/)
set -euo pipefail
PY=${1:-.venv/bin/python}
ENGINES=native,lignum,lleaves,tl2cgen-tuned
cd "$(dirname "$0")/.."
run() { echo "=== $*"; "$PY" benchmarks/bench.py --engines "$ENGINES" "$@" 2>&1 | grep -vE "Treelite version|serializer.cc|split.cc" | grep -E "^#|^- |^\||Saved"; }
run --only lightgbm,xgboost --depth 3                                   # shallow
run --only lightgbm --depth -1 --num-leaves 255 --trees 500              # deep, leaf-wise, unbalanced
run --only xgboost --depth 10 --trees 500                                # deep, depth-wise
run --only lightgbm,xgboost --trees 5000                                 # large model
run --only lightgbm,xgboost --features 500 --rows 200000 --train-rows 100000   # wide data
