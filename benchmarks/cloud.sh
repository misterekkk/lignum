#!/usr/bin/env bash
# One-shot benchmark run on a fresh Ubuntu 24.04 cloud server (e.g. Hetzner CCX / CAX):
# installs the toolchain, builds lignum, runs the comparisons and packs the reports.
#
#   curl -LsSf https://raw.githubusercontent.com/misterekkk/lignum/dev/v0.2/benchmarks/cloud.sh | bash
#   # or, from a clone: bash benchmarks/cloud.sh
#
# Result: ~/lignum-bench-<host>-<date>.tar.gz with the Markdown reports, CPU info and the commit.
# About 30-40 minutes on 4-8 cores (dataset downloads, training, lleaves compiling its models).
# Delete the server afterwards: a stopped server is still billed.
set -euo pipefail

BRANCH="${BRANCH:-dev/v0.2}"
TREES="${TREES:-500}"
REAL="lightgbm-real-california,lightgbm-real-covtype,lightgbm-real-adult,lightgbm-real-bank,lightgbm-real-diabetes"

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git curl linux-tools-common >/dev/null
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d lignum ] && [ ! -f benchmarks/bench.py ]; then
    git clone -q https://github.com/misterekkk/lignum.git
    cd lignum
elif [ -d lignum ]; then
    cd lignum
fi
git fetch -q origin "$BRANCH" && git checkout -q "$BRANCH" && git pull -q --ff-only
COMMIT=$(git rev-parse --short HEAD)

# Stable clocks where the kernel allows it (often not on cloud VMs; harmless if it fails).
sudo cpupower frequency-set -g performance >/dev/null 2>&1 || true

make venv >/dev/null
uv pip install -q --python .venv/bin/python ".[test]" nanobind scikit-build-core
make bench-deps >/dev/null
make py-build >/dev/null
PY=.venv/bin/python

OUT="$HOME/lignum-bench-$(hostname)-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"
{ lscpu; echo; nproc; echo; free -h; echo; echo "lignum commit $COMMIT"; } > "$OUT/machine.txt"
cat "$OUT/machine.txt" | head -20

start=$(date +%s)
# 1. Real data sets and the synthetic models: interpreter, compiled, lleaves, native LightGBM.
$PY benchmarks/bench.py --engines lignum,lignum-compiled,lleaves,native \
    --only "lightgbm,lightgbm-cat,$REAL" --trees "$TREES" --repeats 5
# 2. Arrow input end to end (lignum vs lleaves via pandas vs LightGBM).
$PY benchmarks/bench_arrow.py --trees "$TREES"
# 3. Host-tuned build, for comparison with lleaves (which always targets the host CPU).
CMAKE_ARGS="-DLIGNUM_NATIVE=ON" make py-build >/dev/null
$PY benchmarks/bench.py --engines lignum,lignum-compiled --only "lightgbm,$REAL" --trees "$TREES" --repeats 5
CMAKE_ARGS="" make py-build >/dev/null
# 4. Which path autotune picks per model on this CPU.
$PY - <<'EOF' | tee "$OUT/autotune.txt"
import sys
sys.path.insert(0, "benchmarks")
import numpy as np, lignum, datasets
for name in ["california", "covtype", "adult", "diabetes"]:
    d = datasets.LOADERS[name]()
    m = lignum.load(f"benchmarks/.cache/real/{name}_t500.txt")
    r = m.compile(autotune=d.X[:4096])["autotune"]
    print(name, "compiled_min_rows", r["compiled_min_rows"], {k: (round(a), round(b)) for k, (a, b) in r["timings"].items()})
# The synthetic model bench.py trained above (1M rows, 20 features, independent: hard to predict).
import glob
cache = sorted(glob.glob("benchmarks/.cache/r1000000_f20_t*_d6"))[-1]
X = np.load(f"{cache}/data.npz")["X"][:4096]
m = lignum.load(f"{cache}/lgb.txt")
r = m.compile(autotune=X)["autotune"]
print("synthetic", "compiled_min_rows", r["compiled_min_rows"], {k: (round(a), round(b)) for k, (a, b) in r["timings"].items()})
EOF
echo "total $(( ($(date +%s) - start) / 60 )) min" >> "$OUT/machine.txt"

cp benchmarks/results/2*.md "$OUT"/ 2>/dev/null || true
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "Reports: $OUT.tar.gz"
echo "Copy them home with:  scp root@<server-ip>:$OUT.tar.gz ."
