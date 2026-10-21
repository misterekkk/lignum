# Tasks

Ordered roadmap. Each stage ends with: all tests green, bit-exact parity, benchmark vs the previous stage.

## Stage 0 — Cleanup and correctness (done, branch `dev/v0.2`)

- [x] Archive the categorical-splits attempt (`dev/new-features`), tag `v0.1-with-R`
- [x] Drop the R package; new layout (`include/lignum`, `src`, `python`, `tests/{cpp,python}`)
- [x] Single root `CMakeLists.txt` (C++20, static `lignum` lib, Python module, GoogleTest) and root `pyproject.toml`
- [x] Bit-exact parity fixes:
  - [x] LightGBM `missing_type=None` (NaN → 0) and `missing_type=Zero`
  - [x] LightGBM input zeroing of `|x| <= 1e-35`
  - [x] LightGBM `sigmoid:` parameter, `cross_entropy_lambda` transform
  - [x] XGBoost float32 thresholds, accumulation, base margin and transforms
  - [x] Sum trees in model order (no per-tile partial sums)
- [x] Validate `n_features`; binary format 0.2 with size validation
- [x] XGBoost parser: O(n) instead of O(n²) array access
- [x] Parity test suite with `assert_array_equal`
- [x] C API (`include/lignum/c_api.h`): load/save/predict with strategy/num_features, thread-local errors; shared `liblignum` exports only the C API
- [ ] Categorical benchmark data (Stage 3; scripts on `dev/new-features`)
- [x] One benchmark suite (`benchmarks/bench.py`): native, TL2cgen (default and tuned), GTIL, lleaves, ONNX Runtime; throughput, latency, prep time, bit-exactness. Baseline: `benchmarks/results/2026-10-21-m4-baseline.md`

Benchmark at the end of Stage 0 vs `main` (M4, 1M rows × 20 features, 1000 trees, depth 6):

| Model | main | dev/v0.2 | Note |
|---|---|---|---|
| LightGBM binary | 0.73 s | 0.73 s | — |
| LightGBM `zero_as_missing` | 0.73 s (wrong results) | 0.88 s | extra check per split; see Stage 1 finding |
| XGBoost binary:logistic | 0.78 s | 0.80 s | float32 accumulation, required for parity |
| LightGBM, 100 rows | 343 µs | ~360 µs | superseded by Stage 1b: 161 µs |

## Stage 1 — Tree kinds and outputs

- [x] `tree_kinds[]` in `Model` and the binary format (0.3); `tree_output[]` moves to Stage 2 where it is used
- [x] Kernel dispatches per tree, in model order; only trees with `ZeroIsMissing` splits use that path
- [x] No result changes, no regression (1M rows 0.715 / 0.724 s)

Finding: with `zero_as_missing=True` LightGBM marks every split `missing_type=Zero`; 36% of splits
need the zero check and every tree has some, so per-tree dispatch barely helps those models
(0.88 → ~0.83 s vs 0.71 s for a normal model). A branchless check was slower; accepted as the cost
of that option.

## Stage 1b — Small-batch latency (from the baseline benchmark)

Large batches: lignum is the fastest engine measured (0.78 s vs lleaves 0.85 s, tuned TL2cgen 1.29 s).
Small batches are the weak spot:

- 1 row with all threads costs ~29 µs vs ~10 µs with one thread: the OpenMP region dominates.
- 100 rows fit in one `N_BLOCK` (256 on Apple), so they run on a single thread: 516 µs vs 235 µs (tuned TL2cgen) and 314 µs (lleaves).
- Per-row single-thread cost at 1 row (10.4 µs) trails lleaves (8.3 µs).

- [x] `Strategy` {Auto, Serial, Rows} in C++ and `strategy=` in Python; all bit-identical (tested)
- [x] Serial path for tiny inputs; `Auto` uses it below 8 rows (`benchmarks/batch_sweep.py`)
- [x] Row blocks sized from `n_samples / n_threads` (capped at `N_BLOCK`)
- [x] Tree-parallel path (per-tree leaf buffer, rows sum in order): measured, lost everywhere, removed
- [x] Hot traversal moved to force-inlined free functions over local pointers (single thread back to 4.9 µs/row, large batch 0.78 → 0.73 s)
- [x] Single-row cost: Python overhead is < 1 µs; traversal was L2-latency bound with one dependent chain. Leftover rows now interleave four trees of the same row: 9.7 → 8.0 µs in C++ (lleaves 8.3 µs, ONNX Runtime 8.8 µs)
- [x] `LIGNUM_NATIVE=ON` on the M4: no measurable change (0.726 vs 0.728 s; Apple clang already targets the M family). Re-check on x86 (AVX2 vs baseline x86-64-v2)

Result (`benchmarks/results/2026-10-21-m4-stage1b.md`): 1M rows 0.73 s (LightGBM) / 0.72 s (XGBoost),
1 row 10.4 µs (was 29), 100 rows 161 µs (was 516; tuned TL2cgen 207–235, lleaves 314).

## Stage 2 — Multiclass

- [x] `tree_outputs[]` (class per tree) and `base_scores[]` in `Model` and the binary format (0.4)
- [x] LightGBM (`t % num_tree_per_iteration`) and XGBoost (`tree_info`, vector `base_score`)
- [x] `n_samples × K` output; softmax (LightGBM double; XGBoost float max/exp, double sum, float division — verified), per-class sigmoid (`multiclassova`, `sigmoid:`), XGBoost `multi:softmax` = argmax of margins
- [x] Python returns 2-D for K > 1 (`Model.n_outputs`), C API `LignumModelOutputWidth`
- [x] Parity tests: K = 3 and 10, both strategies, single rows, binary round trip
- [x] Benchmark (`benchmarks/results/2026-10-21-m4-stage2-multi.md`): 10 classes × 100 iterations, 1M rows: lignum 0.77 s (LightGBM) / 0.75 s (XGBoost) vs lleaves 0.95 s, tuned TL2cgen 2.2–2.3 s; only lignum is bit-exact on both
- [ ] Multi-output overhead: 0.77 s vs 0.71 s for a binary model of the same size (read-modify-write of `out[i*K + c]` per leaf); try per-block accumulators
- [ ] Note: when a few features dominate the splits, compiled branchy code (lleaves) gets faster (0.66 s vs our 0.75 s, see `*-multi-4features.md`)

## Stage 3 — LightGBM categorical splits (exact semantics)

- [ ] IR: categorical node kind + category list
- [ ] Lowering: contiguous set → `CAT_RANGE` (`(uint32)(int(x) - a) <= b - a`), span < 64 → `CAT_MASK`, otherwise `CAT_BITSET`
- [ ] Opt-in `assume_known_categories` for float lowering
- [ ] `CAT` kernel kind; numeric-only trees keep the numeric kernel
- [ ] Parity tests: cardinality 2/5/63/64/200, unseen/negative/NaN/non-integer categories, mixed models
- [ ] Benchmark vs Treelite on a categorical model

## Stage 4 — XGBoost categorical splits

- [ ] `categories_nodes/segments/sizes`; in XGBoost a category in the set goes **right**

## Stage 5 — Zero-copy input streaming (inference only)

- [ ] Arrow C Data / C Stream Interface input (no libarrow dependency)
- [ ] Column-major accessor in the kernel; benchmark vs row-major
- [ ] float32 inputs, dictionary columns as categories, validity bitmaps
- [ ] Re-check latency with Arrow input

## Stage 6 — PyPI

- [ ] cibuildwheel: manylinux_2_28 x86_64/aarch64, macOS arm64/x86_64, Windows x64
- [ ] nanobind stable ABI (abi3) for Python ≥ 3.12 + per-version wheels for 3.9–3.11
- [ ] Bundle OpenMP runtimes (auditwheel / delocate / vcomp)
- [ ] Trusted publishing via TestPyPI first; check that the `lignum` name is available
