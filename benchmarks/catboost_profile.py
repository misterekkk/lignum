"""Work per row of lignum and CatBoost on CatBoost models, counted by cachegrind (Linux, valgrind),
for several builds of lignum's CatBoost kernel on the same machine.

Usage:
    python benchmarks/catboost_profile.py --datasets california,adult,covtype
    python benchmarks/catboost_profile.py --variants "search u16:-DLIGNUM_CATBOOST_FLOATS=0 -DLIGNUM_CATBOOST_U8=0"

For each data set (CatBoost trained as in bench_catboost.py) and each variant (a name and C++
flags for the C API build), benchmarks/catboost_capi.cpp runs one engine and case at a time on one
thread under `valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes`; the program's own
setup (the `none` case) is subtracted. Per row: instructions, memory reads and writes, simulated
L1 / last-level data misses and branch mispredictions. Then the time of the same builds without
valgrind (best of 5, larger data), the hottest lines of src/catboost.cpp (cg_annotate) and the
stack accesses in the kernel's functions (objdump).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_catboost  # noqa: E402
import datasets  # noqa: E402

# lignum's CatBoost kernel functions (not the LightGBM / XGBoost kernels' predict_rows).
KERNEL = re.compile(r"::(predict_block\w*|phase2|tree_index\w*|float_buckets|row_trees\w*|predict_row)(<[^(]*>)?\(")
CASES = ["none", "catboost-batch", "lignum-batch", "catboost-row", "lignum-row"]
# Default build (float buckets per feature, u8, groups of 4 trees) and one change at a time.
# Default build and one change at a time; -m flags only where the CPU has the instructions.
VARIANTS = ["default:", "range index:-DLIGNUM_CATBOOST_GE=0", "prefetch:-DLIGNUM_CATBOOST_PREFETCH=1",
            "block 256:-DLIGNUM_CATBOOST_BLOCK=256", "avx2:-mavx2", "avx2 block 256:-mavx2 -DLIGNUM_CATBOOST_BLOCK=256"]
# The functions whose assembly is printed (built with -DLIGNUM_CATBOOST_ASM, so they are not inlined).
ASM_FUNCTIONS = [r"tree_index_ge<6, unsigned char, unsigned char>", r"tree_index<6, unsigned char, unsigned char>",
                 r"add_leaves<4, 1, unsigned char>", r"float_buckets<unsigned char>"]


def cpu_supports(flags: str) -> bool:
    """Whether this machine can run code built with these -m flags (Linux /proc/cpuinfo)."""
    wanted = [f[2:] for f in flags.split() if f.startswith("-m")]
    if not wanted:
        return True
    try:
        cpu = open("/proc/cpuinfo").read().split()
    except OSError:
        return False
    return all(w in cpu for w in wanted)


def loops(body: list[str]) -> list[list[str]]:
    """Instruction ranges closed by a backward jump (loops) in objdump lines `addr: insn`."""
    addrs = []
    for line in body:
        m = re.match(r"\s*([0-9a-f]+):\s*(.*)$", line)
        if m:
            addrs.append((int(m.group(1), 16), m.group(2).strip()))
    found = []
    for i, (addr, insn) in enumerate(addrs):
        m = re.search(r"\b(?:j[a-z]+|b(?:\.[a-z]+)?|cbn?z|tbn?z)\s+(?:\S+,\s*)*(?:0x)?([0-9a-f]+)\b", insn)
        if m:
            target = int(m.group(1), 16)
            if target <= addr:
                start = next((k for k, (a, _) in enumerate(addrs) if a == target), None)
                if start is not None:
                    found.append([f"{a:x}: {t}" for a, t in addrs[start:i + 1]])
    return found


def assembly(library: Path) -> list[str]:
    """The loops of ASM_FUNCTIONS: instruction count, vector instructions, stack accesses, body."""
    text = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", str(library)], capture_output=True, text=True).stdout
    out = []
    for wanted in ASM_FUNCTIONS:
        m = re.search(r"^[0-9a-f]+ <([^>]*" + re.escape(wanted) + r"[^>]*)>:\n((?:.+\n)+?)\n", text, re.M)
        if not m:
            out.append(f"### `{wanted}`: not found\n")
            continue
        out.append(f"### `{m.group(1)[:140]}`\n")
        for body in sorted(loops(m.group(2).splitlines()), key=len)[:3]:  # innermost (shortest) loops
            vec = sum(bool(re.search(r"%[xyz]mm|\bv[0-9]+\b|\bq[0-9]+\b", l)) for l in body)
            stack = sum(bool(re.search(r"%rsp|\[sp", l)) for l in body)
            out.append(f"loop of {len(body)} instructions, {vec} vector, {stack} stack:\n\n```\n" + "\n".join(body[:70]) + "\n```\n")
    return out


def ensure_data(name: str, rows: int, trees: int) -> tuple[Path, Path]:
    """The trained model and C API data of `rows` rows (repeated), with CatBoost's raw scores."""
    import catboost as cb

    d = datasets.LOADERS[name]()
    frame = datasets.catboost_frame(d)
    model = bench_catboost.train(name, d, frame, trees)
    path = bench_catboost.CACHE / "capi" / f"{name}_{rows}.lcb"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bench_catboost.tile(frame, rows)
        cats = [frame.columns[j] for j in d.categorical]
        booster = cb.CatBoost()
        booster.load_model(str(model))
        ref = np.asarray(booster.predict(cb.Pool(data, cat_features=cats), prediction_type="RawFormulaVal"), dtype=np.float64)
        bench_catboost.write_capi_data(path, data, list(d.categorical), ref, 1 if ref.ndim == 1 else ref.shape[1])
    return model, path


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
            totals[m.group(1)] = int(m.group(2).replace(",", ""))
            if m.group(3):
                totals[m.group(1) + " rd"] = int(m.group(3).replace(",", ""))
                totals[m.group(1) + " wr"] = int(m.group(4).replace(",", ""))
    return totals


def per_row(run: dict, base: dict, rows: int) -> str:
    v = lambda key: (run.get(key, 0) - base.get(key, 0)) / rows
    return (f"{v('I refs'):,.0f} | {v('D refs rd'):,.0f} | {v('D refs wr'):,.0f} | {v('D1  misses'):,.1f} | "
            f"{v('LLd misses'):,.2f} | {v('Mispredicts'):,.1f}")


def hot_lines(out: Path, rows: int, top: int = 12) -> list[str]:
    """The lines of src/catboost.cpp with the most instructions: per row Ir, Dr, Dw, Bcm."""
    text = subprocess.run(["cg_annotate", "--threshold=0", "--auto=yes", str(out)], capture_output=True, text=True).stdout
    m = re.search(r"-- Annotated source file: \S*src/catboost\.cpp\n", text)
    part = text[m.end():].split("-- Annotated source file:", 1)[0] if m else ""
    token = r"(\.|[\d,]+(?:\s+\(\s*[\d.]+%\))?)"
    found = []
    for line in part.splitlines():
        m = re.match(r"^\s*" + r"\s+".join([token] * 13) + r"\s(.*)$", line)
        if not m:
            continue
        v = [0 if g == "." else int(g.split()[0].replace(",", "")) for g in m.groups()[:13]]
        if v[0]:
            found.append((v[0], v[3], v[6], v[10], m.group(14).strip()))
    found.sort(reverse=True)
    return [f"| {ir / rows:,.0f} | {dr / rows:,.0f} | {dw / rows:,.0f} | {bcm / rows:,.2f} | `{src[:90]}` |"
            for ir, dr, dw, bcm, src in found[:top]]


def stack_accesses(library: Path) -> list[tuple[str, int, int]]:
    """(function, instructions, instructions touching the stack pointer) of the kernel."""
    text = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", str(library)], capture_output=True, text=True).stdout
    result, name, count, stack = [], None, 0, 0
    for line in text.splitlines() + ["0 <end>:"]:
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            if name and KERNEL.search(name):
                result.append((name, count, stack))
            name, count, stack = m.group(1), 0, 0
        elif line.startswith(" ") and "\t" in line:
            count += 1
            stack += bool(re.search(r"%rsp|\[sp|, sp\b", line))
    return result


# valgrind cannot run AVX-512 code: such variants are timed and counted with perf only.
NO_VALGRIND = re.compile(r"-mavx512")
PERF_EVENTS = "cycles,instructions,branch-misses,L1-dcache-load-misses"


def perf_stat(exe: Path, args: list[str]) -> dict | None:
    """Hardware counters of one run (Linux perf), or None where perf or the counters are missing."""
    try:
        proc = subprocess.run(["perf", "stat", "-x", ",", "-e", PERF_EVENTS, str(exe), *args], capture_output=True, text=True)
    except OSError:
        return None
    counts = {}
    for line in proc.stderr.splitlines():
        parts = line.split(",")
        if len(parts) > 2 and parts[0].strip().isdigit():
            counts[parts[2].split(":")[0]] = int(parts[0])
    return counts if proc.returncode == 0 and "cycles" in counts and "instructions" in counts else None


def perf_row(run: dict | None, base: dict | None, rows: int) -> str:
    if not run or not base:
        return "n/a | n/a | n/a | n/a | n/a"
    v = lambda key: (run.get(key, 0) - base.get(key, 0)) / rows
    ipc = v("instructions") / v("cycles") if v("cycles") > 0 else 0.0
    return f"{v('cycles'):,.0f} | {v('instructions'):,.0f} | {ipc:.2f} | {v('branch-misses'):,.2f} | {v('L1-dcache-load-misses'):,.1f}"


def cpu_name() -> str:
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        return "unknown CPU"


def timings(exe: Path, model: Path, data: Path, library: Path) -> dict:
    """catboost_capi's own timings (best of 5): batch on one thread, one row, 100 rows."""
    proc = subprocess.run([str(exe), str(model), str(data), str(library), "1", "5"], capture_output=True, text=True, check=True)
    return {c["case"]: c.get("best_s", c.get("median_us")) for c in map(json.loads, proc.stdout.splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="california,adult,covtype")
    parser.add_argument("--rows", type=int, default=20000, help="rows under cachegrind")
    parser.add_argument("--time-rows", type=int, default=200000, help="rows for the timings")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--variants", default=";".join(VARIANTS), help="name:flags;name:flags (C++ flags of each build)")
    args = parser.parse_args()

    variants = [v.split(":", 1) for v in args.variants.split(";")]
    skipped = [name for name, flags in variants if not cpu_supports(flags)]
    builds = {name: bench_catboost.build_capi([f"-DCMAKE_CXX_FLAGS=-g {flags}".strip()]) for name, flags in variants
              if name not in skipped}
    asm_build = bench_catboost.build_capi(["-DCMAKE_CXX_FLAGS=-g -DLIGNUM_CATBOOST_ASM"])
    library = bench_catboost.catboost_library()
    work = bench_catboost.CACHE / "profile"
    work.mkdir(parents=True, exist_ok=True)
    print(f"# CatBoost kernel variants under cachegrind ({args.rows:,} rows, one thread)\n")
    print(f"CPU: {cpu_name()}.\n")
    print("Per row; simulated caches and branch predictor; setup (`none`) subtracted. lignum's C API built "
          "-O3 -g with each variant's flags. Time: best of 5 without valgrind, "
          f"{args.time_rows:,} rows, one thread.\n")
    for name in args.datasets.split(","):
        model, data = ensure_data(name, args.rows, args.trees)
        _, time_data = ensure_data(name, args.time_rows, args.trees)
        a = [str(model), str(data), str(library), "1", "1"]
        first = next(iter(builds.values()))
        base = cachegrind(first, a + ["none"], work / f"{name}_none.out")
        cb_batch = cachegrind(first, a + ["catboost-batch"], work / f"{name}_cb_batch.out")
        cb_row = cachegrind(first, a + ["catboost-row"], work / f"{name}_cb_row.out")
        cb_time = timings(first, model, time_data, library)
        print(f"## {name}\n")
        print("| engine | batch: instructions | reads | writes | L1d misses | LL misses | mispredicts | "
              "1 row: instructions | reads | writes | L1d misses | LL misses | mispredicts | 1 thread [s] | 1 row [µs] |")
        print("|---|" + "---:|" * 14)
        print(f"| CatBoost | {per_row(cb_batch, base, args.rows)} | {per_row(cb_row, base, args.rows)} | "
              f"{cb_time['catboost strings, 1 thread']:.3f} | {cb_time['catboost 1 row (CalcModelPredictionSingle)']:.1f} |")
        hot = {}
        flags = dict(variants)
        for vname, exe in builds.items():
            t = timings(exe, model, time_data, library)
            times = f"{t['lignum strings, 1 thread']:.3f} | {t['lignum 1 row']:.1f}"
            if NO_VALGRIND.search(flags[vname]):
                print(f"| lignum {vname} | " + " | ".join(["—"] * 12) + f" | {times} |")
                continue
            batch = cachegrind(exe, a + ["lignum-batch"], work / f"{name}_{vname.replace(' ', '_')}_batch.out")
            row = cachegrind(exe, a + ["lignum-row"], work / f"{name}_{vname.replace(' ', '_')}_row.out")
            print(f"| lignum {vname} | {per_row(batch, base, args.rows)} | {per_row(row, base, args.rows)} | {times} |")
            hot[vname] = hot_lines(work / f"{name}_{vname.replace(' ', '_')}_batch.out", args.rows)
        print()
        # Real CPU counters for a batch (perf stat, time data, setup subtracted), where available.
        pa = [str(model), str(time_data), str(library), "1", "1"]
        p_base = perf_stat(first, pa + ["none"])
        print(f"perf stat, batch of {args.time_rows:,} rows, per row (real CPU; n/a: no hardware counters here):\n")
        print("| engine | cycles | instructions | IPC | branch misses | L1d load misses |\n|---|---:|---:|---:|---:|---:|")
        print(f"| CatBoost | {perf_row(perf_stat(first, pa + ['catboost-batch']), p_base, args.time_rows)} |")
        for vname, exe in builds.items():
            print(f"| lignum {vname} | {perf_row(perf_stat(exe, pa + ['lignum-batch']), p_base, args.time_rows)} |")
        print()
        for vname, lines in hot.items():
            print(f"<details><summary>{name}, lignum {vname}: hottest lines of a batch, per row</summary>\n")
            print("| instructions | reads | writes | mispredicts | line |\n|---:|---:|---:|---:|---|")
            print("\n".join(lines) + "\n\n</details>\n")
    if skipped:
        print(f"Not run (this CPU lacks the instructions): {', '.join(skipped)}.\n")
    print("## Assembly of the hot loops (default build, functions kept out of line)\n")
    print("\n".join(assembly(next(Path(asm_build).parent.glob("liblignum.*")))))
    print("## Stack accesses in the kernel's functions (objdump)\n")
    for vname, exe in builds.items():
        lib = next(Path(exe).parent.glob("liblignum.*"))
        rows = stack_accesses(lib)
        total = sum(c for _, c, _ in rows), sum(s for _, _, s in rows)
        print(f"- {vname}: {total[0]:,} instructions in {len(rows)} functions, {total[1]:,} touching the stack")


if __name__ == "__main__":
    main()
