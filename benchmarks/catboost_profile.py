"""Work per row of lignum and CatBoost on CatBoost models, counted by cachegrind (Linux, valgrind).

Usage (after bench_catboost.py has trained the models and written the C API data):
    python benchmarks/catboost_profile.py --datasets california,adult,covtype --rows 20000

For each data set it runs benchmarks/catboost_capi.cpp once per engine and case (one thread) under
`valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes`, subtracts the program's own setup
(the `none` case) and prints per row: instructions, memory reads and writes, simulated L1 / last
level data misses and branch mispredictions. Then cg_annotate's source lines for lignum's batch
(built with -g), and the stack accesses (spills) in lignum's hot functions from objdump.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_catboost  # noqa: E402

# lignum's CatBoost kernel functions (not the LightGBM / XGBoost kernels' predict_rows).
KERNEL = re.compile(r"::(predict_block|phase2|tree_index\w*|row_trees\w*|predict_row)(<[^(]*>)?\(")
CASES = ["none", "catboost-batch", "lignum-batch", "catboost-row", "lignum-row"]
EVENTS = {"I refs": "instructions", "D refs": "reads+writes", "D1  misses": "L1d misses", "LLd misses": "LL misses",
          "Mispredicts": "mispredicts"}


def number(text: str) -> int:
    return int(text.replace(",", ""))


def cachegrind(exe: Path, args: list[str], out: Path) -> dict:
    """Totals of one run: instructions, reads, writes, L1d / LLd misses, branch mispredictions."""
    proc = subprocess.run(["valgrind", "--tool=cachegrind", "--cache-sim=yes", "--branch-sim=yes",
                           f"--cachegrind-out-file={out}", str(exe), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    totals = {}
    for line in proc.stderr.splitlines():
        line = re.sub(r"^==\d+== ", "", line)
        m = re.match(r"(I refs|D refs|D1  misses|LLd misses|Mispredicts):\s+([\d,]+)(?:\s+\(\s*([\d,]+) rd\s+\+\s+([\d,]+) wr\))?", line)
        if m:
            totals[m.group(1)] = number(m.group(2))
            if m.group(3):
                totals[m.group(1) + " rd"] = number(m.group(3))
                totals[m.group(1) + " wr"] = number(m.group(4))
    return totals


def stack_accesses(library: Path) -> list[tuple[str, int, int]]:
    """(function, instructions, instructions touching the stack pointer) for lignum's CatBoost kernel."""
    text = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", str(library)], capture_output=True, text=True).stdout
    result, name, count, stack = [], None, 0, 0
    for line in text.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            if name and KERNEL.search(name):
                result.append((name, count, stack))
            name, count, stack = m.group(1), 0, 0
        elif line.startswith(" ") and "\t" in line:
            count += 1
            if re.search(r"%rsp|\[sp|, sp\b", line):
                stack += 1
    if name and KERNEL.search(name):
        result.append((name, count, stack))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="california,adult,covtype")
    parser.add_argument("--rows", type=int, default=20000, help="rows of the data written by bench_catboost.py --rows")
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()

    exe = bench_catboost.build_capi(["-DCMAKE_CXX_FLAGS=-g"])
    library = bench_catboost.catboost_library()
    work = bench_catboost.CACHE / "profile"
    work.mkdir(parents=True, exist_ok=True)
    print(f"# CatBoost models under cachegrind, per row ({args.rows:,} rows, one thread)\n")
    print("Simulated caches and branch predictor (cachegrind), lignum built with -O3 -g. Setup (`none`) subtracted.\n")
    for name in args.datasets.split(","):
        model = bench_catboost.CACHE / "real" / f"catboost_{name}_t{args.trees}.cbm"
        data = bench_catboost.CACHE / "capi" / f"{name}_{args.rows}.lcb"
        runs = {case: cachegrind(exe, [str(model), str(data), str(library), "1", "1", case], work / f"{name}_{case}.out") for case in CASES}
        base = runs["none"]
        print(f"## {name}\n")
        print("| case | instructions | reads | writes | L1d misses | LL misses | mispredicts |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for case in CASES[1:]:
            r = runs[case]
            per = lambda key: (r.get(key, 0) - base.get(key, 0)) / args.rows
            print(f"| {case} | {per('I refs'):,.0f} | {per('D refs rd'):,.0f} | {per('D refs wr'):,.0f} | {per('D1  misses'):,.1f} | "
                  f"{per('LLd misses'):,.2f} | {per('Mispredicts'):,.1f} |")
        print()
        if name == args.datasets.split(",")[0]:
            for case in ["lignum-batch", "lignum-row"]:
                annotated = subprocess.run(["cg_annotate", "--threshold=0.5", str(work / f"{name}_{case}.out")],
                                           capture_output=True, text=True).stdout
                lines = [l for l in annotated.splitlines() if l.strip()]
                print(f"### cg_annotate, {case} ({name}), lines over 0.5% of instructions\n\n```")
                print("\n".join(lines[:220]))
                print("```\n")
    lib = next(Path(exe).parent.glob("liblignum.*"))
    print("## Stack accesses in lignum's CatBoost functions (objdump)\n")
    print("| function | instructions | touching the stack |")
    print("|---|---:|---:|")
    for fn, count, stack in stack_accesses(lib):
        print(f"| `{fn[:110]}` | {count} | {stack} |")


if __name__ == "__main__":
    main()
