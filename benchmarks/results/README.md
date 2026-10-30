# Benchmark reports

Raw reports of `benchmarks/suite.py`, the benchmark behind the published tables, one folder per
machine: the Markdown report, the same numbers in JSON (every batch size, percentile and repeat)
and the machine description. File names: `<date>-suite-<commit>[-<engines>]`.

| Folder | Machine |
|---|---|
| `m4/` | MacBook Pro, Apple M4 (4 performance + 6 efficiency cores, 16 GB), macOS 26, Apple clang 17 |
| `hetzner-ccx33/` | Hetzner CCX33, 8 dedicated vCPU = 4 cores with SMT, AMD EPYC-Milan, Ubuntu 24.04, gcc 13.3 (`cloud.sh`) |

On the M4, `2026-10-30-suite-41dcee1-compiled` repeats the compiled engine after the fix that made
`compile()` run the C compiler in parallel on macOS: its preparation times replace those of the
full run; its prediction times equal the full run's within noise. The Linux server never had that
problem.

New runs land in this folder itself and stay local (git-ignored) until moved into one of the above.
Older reports (other models, the same input row on every call) are in the git history.
