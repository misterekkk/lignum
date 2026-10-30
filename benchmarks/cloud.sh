#!/usr/bin/env bash
# One-shot benchmark run on a fresh Ubuntu 24.04 cloud server (e.g. Hetzner CCX / CAX):
# installs the toolchain, builds lignum, runs benchmarks/suite.py and packs the report.
#
#   git clone https://github.com/misterekkk/lignum.git && cd lignum
#   # copy the data sets and models (below), then, inside tmux so that a dropped SSH connection
#   # does not stop the run (detach: Ctrl-b d, reattach: tmux attach -t bench):
#   tmux new -s bench 'bash benchmarks/cloud.sh'
#
# The output also goes to ~/lignum-bench.log (tail -f ~/lignum-bench.log to follow it).
#
# Every platform must predict with the same model files, so the server does not train: copy the data
# sets and the models trained on your machine (benchmarks/suite.py trains them on its first run)
# before running this script:
#   scp -r benchmarks/.cache/datasets benchmarks/.cache/suite root@<server-ip>:<clone>/benchmarks/.cache/
# (scikit-learn's source for California and Covertype, figshare, also refuses most cloud servers.)
#
# Result: ~/lignum-bench-<host>-<arch>-<date>.tar.gz with the reports, CPU info and the commit.
# About 2 hours on 4-8 cores. Delete the server afterwards: a stopped server is still billed.
# Works on x86-64 (CCX, AMD EPYC) and aarch64 (CAX, Ampere Altra); run it once per server.
set -euo pipefail

# Everything runs inside main, so a `curl | bash` pipe cut short cannot run half a script.
main() {

exec > >(tee -a "$HOME/lignum-bench.log") 2>&1
BRANCH="${BRANCH:-main}"
ROUNDS="${ROUNDS:-500}"

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git curl tmux linux-tools-common >/dev/null
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
# uv's own Python 3.12 (with headers for the extension), whatever the image ships or lacks.
export UV_PYTHON_PREFERENCE=only-managed

if [ ! -d lignum ] && [ ! -f benchmarks/suite.py ]; then
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
uv pip install -q --python .venv/bin/python lleaves "llvmlite<0.45"
# tl2cgen has no aarch64 wheel: try the source build, and report it as not installed if that fails.
uv pip install -q --python .venv/bin/python treelite tl2cgen \
    || echo "tl2cgen could not be installed: the report marks it n/a on this platform"
make py-build >/dev/null
PY=.venv/bin/python

# Data sets and models first, so a missing file stops the run now and not after an hour.
missing=$($PY - "$ROUNDS" <<'EOF'
import sys
sys.path.insert(0, "benchmarks")
import datasets, suite
rounds = int(sys.argv[1])
ext = {"lightgbm": "txt", "xgboost": "json", "catboost": "cbm"}
for name, load in datasets.LOADERS.items():
    try:
        load()
    except Exception as e:
        print(f"data set {name}: {type(e).__name__}")
for lib in suite.LIBRARIES:
    for name in suite.DATASETS:
        if not (suite.CACHE / f"{lib}_{name}_r{rounds}.{ext[lib]}").exists():
            print(f"model {lib}_{name}_r{rounds}")
EOF
)
if [ -n "$missing" ]; then
    echo "$missing"
    echo
    echo "Copy the data sets and models from the machine that trained them, then run again:"
    echo "  scp -r benchmarks/.cache/datasets benchmarks/.cache/suite root@<server-ip>:$PWD/benchmarks/.cache/"
    exit 1
fi

OUT="$HOME/lignum-bench-$(hostname)-$(uname -m)-$(date +%Y%m%d-%H%M)"
mkdir -p "$OUT"
touch "$OUT/.start"   # reports newer than this are the ones from this run
{
    lscpu; echo; nproc; echo; free -h; echo
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'n/a (VM)')"
    gcc --version | head -1
    # The defaults are measured on purpose: OMP_PROC_BIND pins the main thread at libgomp load, so
    # thread pools started from Python (lleaves) would share one core; OMP_WAIT_POLICY=active
    # leaves idle OpenMP threads spinning next to the other engines.
    env | grep -E '^(OMP|GOMP|KMP)_' || echo "no OMP_* / GOMP_* / KMP_* variables"
    echo "lignum commit $COMMIT"
} > "$OUT/machine.txt"
head -20 "$OUT/machine.txt"

start=$(date +%s)
$PY benchmarks/suite.py --rounds "$ROUNDS"
echo "total $(( ($(date +%s) - start) / 60 )) min" >> "$OUT/machine.txt"

find benchmarks/results -maxdepth 1 -type f -newer "$OUT/.start" -exec cp {} "$OUT"/ \;
rm "$OUT/.start"
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "Reports: $OUT.tar.gz"
echo "Copy them home with:  scp root@<server-ip>:$OUT.tar.gz ."
}

main "$@"
