# Benchmark reports

Raw reports as the scripts write them (`bench.py`, `bench_arrow.py`, `cloud.sh`), one folder per
machine. File names: `<date>-<commit or experiment>[-arrow|-native|-autotune|-machine]`.

| Folder | Machine |
|---|---|
| `m4/` | MacBook Pro, Apple M4 (10 cores) |
| `github/x86_64/`, `github/aarch64/` | GitHub runners, 4 vCPU (`bench-lleaves.yml`): AMD EPYC 7763 / 9V45, Neoverse-N2 |
| `hetzner/x86_64/` | Hetzner CCX, 4 dedicated vCPU, AMD EPYC-Milan (`cloud.sh`) |
| `hetzner/aarch64/` | Hetzner CAX, 4 shared vCPU, Ampere Altra / Neoverse-N1 (`cloud.sh`) |

`-native` is lignum built with `LIGNUM_NATIVE=ON`; `-machine.txt` holds `lscpu`, memory, compiler
and the commit. New runs land in this folder itself and stay local until moved into one of the above.
