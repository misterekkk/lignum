# Architecture

## Pipeline

```
model file ──parser──▶ TempBinNode forest (IR) ──Builder──▶ Model ──predict──▶ scores
                       src/ir.hpp                 src/builder.cpp   src/model.cpp
                                                      │
                                        Model::save ──┼──▶ .bin ──mmap──▶ Model (zero-copy)
                                                      │                   src/io.cpp
                                     compile_forest ──┴──▶ C ──cc──▶ shared library (src/compiled.cpp)
```

- **Parsers** (`src/parsers/`) read LightGBM `model.txt` (`lightgbm_text.cpp`) and `dump_model()`
  JSON, XGBoost JSON and UBJSON (`ubjson.cpp` rewrites it as JSON) into a library-neutral binary
  tree. `src/load.cpp` recognises the format from the file's first bytes. All library-specific
  semantics are resolved here (see "Split semantics").
- **Builder** packs two tree levels into one 64-byte `KNode` and stores leaves in a separate array.
- **Model** is a view: raw pointers to nodes/offsets/leaves plus a `shared_ptr<Data>` owning the
  memory, which is either `VectorData` (built from a model file) or `MmapData` (binary file).

## KNode

```
struct alignas(64) KNode {          // one cache line = two levels of a binary tree
    double   thresholds[4];         // slot 0: root split, slot 1: left child, slot 2: right child, [3] unused
    int32_t  children[4];           // four grandchildren; negative = leaf index (-k-1)
    uint16_t features[4];
    uint8_t  dirs[4];               // bit0: missing goes right, bit1: zero is missing (LightGBM)
    uint8_t  kinds[4];              // slot_kind: Numeric, CatMask, CatRange, CatBitset
};
```

If a child of the root is a leaf, both of its grandchild pointers reference that leaf and the slot's
comparison is ignored.

A categorical slot keeps its category set in the 64 bits of `thresholds[slot]`: a mask when every
category is below 64, `[first, last - first]` when the set is contiguous, otherwise
`[offset, words]` into `Model::cat_bitsets` (word 0 is always zero, so branch-free lookups of
non-bitset slots read a valid address).

## Kernel (src/model.cpp)

- `Kernel<Acc, ZeroMissing>` predicts a block of rows: trees in tiles of `T_BLOCK` so a tile of
  trees stays in cache while the block walks it. Strategies only decide how blocks are scheduled:
  - `Serial`: blocks of `N_BLOCK` on the calling thread, no OpenMP region.
  - `Rows`: OpenMP over blocks sized `n_samples / (2 * threads)`, rounded to 4, capped at `N_BLOCK`.
  - `Auto`: `Serial` below 8 rows or with one thread, `Rows` otherwise (thresholds from
    `benchmarks/batch_sweep.py`).
- Four rows are interleaved to hide memory latency; leftover rows (fewer than four, e.g. a single
  row) interleave four trees instead. Leaves are still added in tree order.
- Per KNode the three comparisons are evaluated speculatively and the child is selected with
  `children[(r << 1) | (r ? c : l)]`, which removes the dependency between the two levels.
- Template parameters: accumulator type (`double` for LightGBM, `float` for XGBoost) and `Mixed`
  (the model has trees that are not plain numeric). With `Mixed`, the traversal is specialised per
  tree from `tree_kinds[]` (bits `ZeroMissing`, `Categorical`); otherwise there is no per-tree branch.
- In categorical trees each slot branches on its kind (structural, predictable) and the categorical
  test itself is branch-free (`cat_dir`), because NaN/range/category branches depend on the data.
- Two step variants: branch-free (all three comparisons, default for numeric trees in 4-row walks) and
  branchy (branch on the root comparison, evaluate only the taken child). Branchy is used for leftover
  rows (fewer than four, e.g. one row) everywhere and for 4-row walks of categorical trees except on
  Apple (`LIGNUM_BRANCHY_TAIL`, `LIGNUM_BRANCHY_CAT`); choices measured on x86-64, aarch64 and M4. The traversal (`step`, `walk4`, `walk1`) is force-inlined and works
  on local copies of the model pointers; going through `Model&` members cost ~7-13%.
- Leaves are added to a running sum loaded from `out_preds[i]`, in model order, so floating point
  results are identical to the source library.
- Multiclass (`Multi` template parameter): row i's K outputs live in `out[i*K .. i*K+K)`; each tree
  adds its leaf to `out[i*K + tree_outputs[t]]`, still in model order. Softmax runs per row after the
  last tile. XGBoost `multi:softmax` (argmax) is computed from raw margins in a temporary buffer,
  because narrowing K columns to one in place would race between threads.

## Binned kernel (src/binned.*, default)

Built from the KNode form at load time (or mapped from the binary file):

- Every used feature becomes a *dense feature* with a 16-bit code per row, computed once per row
  block before the tree walk:
  - numeric: `bin(x)` = number of the feature's split thresholds `t` with `t <= x` (branch-free
    binary search over sorted unique thresholds), so `x >= t_k` ⇔ `bin(x) >= k + 1` exactly;
  - categorical: the integer category under LightGBM or XGBoost rules; NaN and invalid values get
    codes that are never in a set;
  - NaN: `0xFFFF`.
- `BNode` (64 bytes) holds **three** tree levels: 7 slots (16-bit threshold or decision-bit index,
  16-bit dense feature) and 8 subtrees. The walk compares only along the path (3 dependent integer
  comparisons per node); evaluating all 7 speculatively was 2x slower.
- NaN without a branch: `0xFFFF` is above every threshold (goes right); slots whose NaN goes left
  add 1 to the code, which wraps `0xFFFF` to 0.
- Categorical splits are decided before the walk: per categorical feature a table maps each
  category to the decision bits of all of that feature's slots in the model; each row copies one
  table row per feature into its decision buffer, and a categorical slot reads one bit from L1.
- LightGBM zero-as-missing: both edges of the zero band are added as cuts, so the band is one bin;
  flagged slots override the comparison for that bin (branch-free).
- Rows outside the 4-row groups (including a single row) walk four trees at once.
- Results are identical to the KNode kernel, which remains for `load(..., binned=False)` and models
  the binned form cannot represent; every Python test runs with both kernels
  (`tests/python/conftest.py`).

## Split semantics

Every split is normalised to: **go right ⇔ `isnan(x) ? default_right : x >= threshold`**.

| Library | Native rule | Translation in the parser |
|---|---|---|
| LightGBM | right iff `z(x) > t`, where dense input with `\|x\| ≤ 1e-35f` becomes 0 | `threshold = nextafter(t, +inf)`; for `t` in the zero band the threshold moves to the band edge |
| LightGBM `missing_type=NaN` | NaN → `default_left` | direct |
| LightGBM `missing_type=None` | NaN → 0.0, compared normally | `default_right = (0 >= threshold)` |
| LightGBM `missing_type=Zero` | NaN → 0.0, and 0 is missing → `default_left` | direct; nodes where this differs from the natural direction of 0 get `ZeroIsMissing` |
| XGBoost | left iff `float(x) < t` (t is float32), NaN → `default_left` | threshold = smallest double that rounds to ≥ `t` in float32 |
| LightGBM categorical (`==`) | NaN → right; x ≤ -1 → right; else `int(x)` in set → left | categorical slot (see KNode) |
| XGBoost categorical (`split_type` 1) | NaN → `default_left`; `c = int(float32(x))`; c < 0 or c ≥ 2^24 → left; c in `categories` → right | categorical slot with `dir_bits::XGBoostCategory` |

XGBoost also accumulates and transforms in float32, and converts `base_score` to margin space with
float32 `log`/logit (`binary:logitraw` uses `base_score` as is; multiclass models store a vector of
per-class base scores already in margin space). XGBoost's softmax takes the max and `expf` in float,
sums in double and divides by `float(sum)`; LightGBM's softmax is all double.

Verified bit-exact by `tests/python/test_parity.py` on inputs with NaNs, zeros, values in the zero
band and values placed exactly on every split threshold.

## Binary format (version 0.6)

64-byte header (`magic "LIGNUM"`, version, transform, flags {float32, special trees}, sigmoid,
n_features, n_outputs, n_cat_words, section sizes), followed by 64-byte aligned sections: nodes, tree offsets,
leaves, tree kinds, base scores, tree outputs (multiclass only), categorical bitsets and, when the
`Binned` flag is set, a `BinnedFileHeader` followed by the binned kernel's sections (nodes, roots,
tree kinds, dense features, cuts, zero bins, categorical decision tables).
Loading validates the version and the file size, maps the file read-only and then checks every
stored index (`src/validate.cpp`): children must point strictly forward (so a walk always ends),
leaves, features, categorical bitsets and decision tables must be in range. This takes 0.15-0.64 ms
for the benchmark models. `tests/cpp/test_fuzz.cpp` corrupts files at random and runs under
ASan/UBSan in CI. Older versions are rejected; re-export from JSON.

## Compiled mode (src/compiled.*)

`compile_forest` generates C from the `BinnedForest`: `lignum_bin` (numeric binning specialised per
feature: constant cut tables, unrolled binary search, linear count for <= 8 cuts) and one
`lignum_block_<b>` per block of 16 trees. A block walks a chunk of rows (~2 MB of codes, decision bits
and sums, `LIGNUM_COMPILED_CHUNK_KB`) before the next block runs, adding leaves (from a per-block
table) in model order. The `LIGNUM_COMPILED_LOCALS` most used features of a block are read once per
row into locals (int16 codes: NaN = -1, so the NaN-left comparison needs no wrap), kept in registers
by an empty asm; leaf pairs become selects. Categorical decision bits, threads and transforms stay in
the kernel (`BinnedKernel` receives the compiled forest; `Model::compiled_min_rows` sends small
batches to the interpreter). Sources are compiled in parallel (gcc/clang, or one `cl /LD /MP`
command), linked, cached by source hash and loaded with dlopen / LoadLibrary. The library exports
`lignum_abi` and `lignum_fingerprint` (FNV-1a of trees, cuts, leaves, outputs, precision), checked by
`load_compiled`.

## Arrow input (src/arrow.cpp, src/columns.hpp)

`predict_arrow` takes a record batch through the Arrow C Data Interface (`include/lignum/arrow.h`).
With the binned kernel, an `ArrowColumns` `ColumnSource` lets `BinnedKernel` bin each feature column
of a chunk straight from the buffers (typed conversion into a small contiguous buffer, then one
binning loop); the direct kernel gets row-major blocks converted in L1-sized tiles. Values convert
exactly like numpy's `astype(float64)`; nulls (column or batch) are NaN; dictionary columns give
their indices.

## APIs

- C++: `include/lignum/lignum.hpp` (`lignum::load`, `predict_arrow`, `predict_arrow_stream`) and
  `model.hpp` (`Model::predict`, `Strategy`).
- C: `include/lignum/c_api.h`, built as the shared `liblignum` (`LIGNUM_BUILD_C_LIBRARY`, on by
  default outside Python builds): loading, prediction (row-major and Arrow), compiled mode. Only
  `Lignum*` symbols are exported; errors are reported through return codes and a thread-local
  `LignumGetLastError()`.
- Python: `python/bindings.cpp` (nanobind, abi3 for Python >= 3.12) + `python/lignum/__init__.py`,
  published as `pylignum`.

## Known limitations

No multi-target / vector-leaf models, no LightGBM linear trees or random forests (rejected at load
time).
