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

- [x] Semantics verified against LightGBM 4.7: NaN always right (even when 0 is in the set and for any `missing_type`), x <= -1 right, otherwise `int(x)` (truncation) in set → left
- [x] IR: categorical nodes with their category list; per-slot `kinds[]` in `KNode` (was padding), `tree_kind::Categorical`
- [x] Encodings: `CatMask` (all categories < 64), `CatRange` (contiguous), `CatBitset` (side array `cat_bitsets`, word 0 reserved as a zero target); binary format 0.5
- [x] Parity tests: cardinality 2/5/63/64/200/1000, all three encodings (range via one-hot splits on ids ≥ 64), unseen/negative/fractional/NaN/±inf/huge values, mixed, multiclass, zero_as_missing, binary round trip, pandas category codes
- [x] Benchmark (`benchmarks/results/2026-10-22-m4-stage3-categorical.md`, 5 categorical + 15 numeric features): lignum 2.95 s vs tuned TL2cgen 5.36 s, native 8.41 s, lleaves 2.09 s; only lignum and TL2cgen are bit-exact
- [ ] Opt-in `assume_known_categories` (float lowering) — only if a real model needs it

Kernel experiments on the categorical model (100k rows, 200 trees; same data without categories: 0.014 s):

| variant | 10 threads | 1 thread |
|---|---:|---:|
| branchy `cat_dir` (early returns + switch) | 0.053 s | 0.387 s |
| everything branch-free, also numeric slots | 0.061 s | — |
| **branch on slot kind + branch-free `cat_dir` (kept)** | **0.049 s** | **0.335 s** |
| branch per node (flag in `kinds[3]`) | 0.052 s | 0.333 s |
| 8 interleaved rows instead of 4 | 0.056 s | 0.347 s |
| numeric model forced through the mixed path | 0.015 s | 0.101 s |

The cost is the longer dependency chain of a categorical comparison (float→int, dependent bitset
load from L2), not dispatch. Next idea: keep a 64-bit mask for categories < 64 inline and touch
`cat_bitsets` only for larger ids (cheap when category ids are frequency-ordered, which is common).

## Stage 4 — XGBoost categorical splits

- [x] Semantics verified against XGBoost 3.4: `categories` lists the categories that go **right**; the category is `int(float32(x))`; negative or ≥ 2^24 goes **left**; NaN follows `default_left`
- [x] Parser reads `split_type`, `categories`, `categories_nodes/segments/sizes`; same slot encodings as LightGBM, `dir_bits::XGBoostCategory` selects the semantics
- [x] Parity tests (16): cardinality 2..1000, one-hot and partition splits, mixed, multiclass, round trip, values such as 2.9999999999 (→ 3 in float32), 16777216, negatives
- [ ] Benchmark vs TL2cgen / native (`xgboost-cat` workload)

## Model-shape matrix (`benchmarks/run_matrix.sh`, M4, 1M rows unless noted)

| shape | lignum | lleaves | tuned TL2cgen | native |
|---|---:|---:|---:|---:|
| depth 3, LightGBM | 0.52 s | **0.33 s** | **0.33 s** | 2.52 s |
| depth 3, XGBoost | **0.47 s** | — | 0.54 s | 0.68 s |
| leaf-wise 255 leaves, 500 trees | **1.12 s** | 1.19 s | 1.53 s | 8.21 s |
| XGBoost depth 10, 500 trees | **0.72 s** | — | 1.76 s | 1.44 s |
| 5000 trees, LightGBM / XGBoost | **3.62 / 3.67 s** | 4.52 s | 9.05 / 12.2 s | 38.2 / 7.48 s |
| 500 features (200k rows) | **0.20 / 0.20 s** | 0.40 s | 0.60 / 0.66 s | 1.01 / 0.35 s |

GitHub runners (`benchmarks/results/ci/`, 300k rows, 500 trees, 4 threads, generic lignum build):
on x86-64 lleaves beats lignum on large LightGBM batches (0.45 s vs 0.72 s; categorical 1.36 vs 2.22 s),
on aarch64 Linux too (0.33 vs 0.44 s). lignum still wins vs TL2cgen (1.5×), native libraries, small
batches (100 rows: 268 µs vs 403 TL2cgen / 668 lleaves) and is the only bit-exact engine.

### Branchy traversal (CI `bench-lleaves.yml`, 300k rows, 500 trees, 4 threads)

The default step evaluates all three comparisons of a KNode branch-free. A branchy step (branch on the
root comparison, evaluate only the taken child) does two comparisons and lets the CPU speculate:

| | default | branchy everywhere | hybrid (kept) |
|---|---:|---:|---:|
| x86-64 numeric batch | **0.76–0.84 s** | 1.12 s | 0.83 s |
| x86-64 categorical batch | 2.71–2.76 s | 2.23 s | **2.28 s** |
| x86-64 one row (numeric / categorical) | 7.6 / 18.7 µs | 4.8 / 10.1 µs | **6.2 / 13.2 µs** |
| aarch64 numeric batch | **0.44 s** | 0.60 s | 0.44 s |
| aarch64 categorical batch | 1.59 s | 1.39 s | **1.37 s** |
| M4 one row (numeric / categorical) | 7.9 / 17.6 µs | — | **6.6 / 13.2 µs** |

Kept: branchy for leftover rows everywhere, branchy in categorical trees except on Apple (M4 is ~20%
slower there). Rejected on M4: prefetching a node's children (+30%).

Flags do not explain the gap to lleaves: `LIGNUM_NATIVE=ON` gives ~3% on x86-64 and nothing on aarch64.

Weak spots:
- [x] Exact input binning: per-feature 16-bit codes once per row, integer comparisons, 3 tree levels per 64-byte node (`src/binned.*`); default for batches of 8+ rows, stored in the binary format (0.6)
- [x] Categorical decisions precomputed per row (category -> decision bits table per feature); a categorical slot reads one bit from L1
- [x] Zero-as-missing in the binned kernel (zero band = one bin, branch-free override)
- [x] Validate every index in loaded model files (children strictly forward, leaves, features, categorical tables): 0.15-0.64 ms per load; XGBoost JSON child order checked (no cycles); fuzz test under ASan/UBSan in CI
- [x] Binned kernel for every batch size: with four trees interleaved per leftover row it beats the KNode walk from one row on (M4 one row: numeric 6.9 -> 6.6 us, categorical 14.4 -> 8.3 us)

### Binned kernel results (CI `bench-lleaves.yml`, 300k rows, 500 trees, 4 threads)

| | KNode kernel | binned | lleaves |
|---|---:|---:|---:|
| x86-64 numeric batch / 1 thread | 0.83 / 2.43 s | 0.68 / 1.58 s | 0.56 / 1.81 s |
| x86-64 categorical batch / 1 thread | 2.29 / 7.18 s | **1.22 / 3.68 s** | 2.03 / 4.97 s |
| aarch64 numeric batch / 1 thread | 0.43 / 1.69 s | 0.34 / 1.33 s | 0.33 / 1.33 s |
| aarch64 categorical batch / 1 thread | 1.37 / 5.42 s | **0.71 / 2.82 s** | 0.87 / 3.44 s |

Kept trying on the way: categorical sets in a side table (-18..-22% on servers, +10% on M4) before
precomputed decision bits (-47%); evaluating all 7 slots speculatively (2x slower).
- [ ] Large LightGBM batches on x86-64 / Linux aarch64 vs lleaves (host-tuned): measure `LIGNUM_NATIVE=ON` on the same runner, then profile the kernel on x86 (register pressure of 4-way interleave with 16 GPRs, N_BLOCK/T_BLOCK)
- [ ] Shallow trees: a depth-3 tree costs 73% of a depth-6 one (per-tree overhead: root, loop, leaf load, sum; the last KNode level is half empty). Ideas: single-level node for the last odd level, fixed-depth unrolled traversal for complete trees, several trees per loop iteration
- [ ] Single row on large models (5000 trees, ~6.5 MB): 39 µs vs 23–25 µs (TL2cgen) — memory-latency bound

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
