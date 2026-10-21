# Architecture

## Pipeline

```
JSON model ──parser──▶ TempBinNode forest (IR) ──Builder──▶ Model ──predict──▶ scores
(LightGBM / XGBoost)   src/ir.hpp                src/builder.cpp   src/model.cpp
                                                     │
                                       Model::save ──┴──▶ .bin ──mmap──▶ Model (zero-copy)
                                                         src/io.cpp
```

- **Parsers** (`src/parsers/`) read the library's JSON with simdjson and emit a library-neutral binary
  tree. All library-specific semantics are resolved here (see "Split semantics").
- **Builder** packs two tree levels into one 64-byte `KNode` and stores leaves in a separate array.
- **Model** is a view: raw pointers to nodes/offsets/leaves plus a `shared_ptr<Data>` owning the
  memory, which is either `VectorData` (built from JSON) or `MmapData` (binary file).

## KNode

```
struct alignas(64) KNode {          // one cache line = two levels of a binary tree
    double   thresholds[4];         // slot 0: root split, slot 1: left child, slot 2: right child, [3] unused
    int32_t  children[4];           // four grandchildren; negative = leaf index (-k-1)
    uint16_t features[4];
    uint8_t  dirs[4];               // bit0: missing goes right, bit1: zero is missing (LightGBM)
    uint8_t  padding[4];
};
```

If a child of the root is a leaf, both of its grandchild pointers reference that leaf and the slot's
comparison is ignored.

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
- Template parameters: accumulator type (`double` for LightGBM, `float` for XGBoost) and whether
  any split uses zero-as-missing. The traversal (`step`, `walk4`, `walk1`) is force-inlined and works
  on local copies of the model pointers; going through `Model&` members cost ~7-13%.
- Leaves are added to a running sum loaded from `out_preds[i]`, in model order, so floating point
  results are identical to the source library.

## Split semantics

Every split is normalised to: **go right ⇔ `isnan(x) ? default_right : x >= threshold`**.

| Library | Native rule | Translation in the parser |
|---|---|---|
| LightGBM | right iff `z(x) > t`, where dense input with `\|x\| ≤ 1e-35f` becomes 0 | `threshold = nextafter(t, +inf)`; for `t` in the zero band the threshold moves to the band edge |
| LightGBM `missing_type=NaN` | NaN → `default_left` | direct |
| LightGBM `missing_type=None` | NaN → 0.0, compared normally | `default_right = (0 >= threshold)` |
| LightGBM `missing_type=Zero` | NaN → 0.0, and 0 is missing → `default_left` | direct; nodes where this differs from the natural direction of 0 get `ZeroIsMissing` |
| XGBoost | left iff `float(x) < t` (t is float32), NaN → `default_left` | threshold = smallest double that rounds to ≥ `t` in float32 |

XGBoost also accumulates and transforms in float32, and converts `base_score` to margin space with
float32 `log`/logit (`binary:logitraw` uses `base_score` as is).

Verified bit-exact by `tests/python/test_parity.py` on inputs with NaNs, zeros, values in the zero
band and values placed exactly on every split threshold.

## Binary format (version 0.2)

64-byte header (`magic "LIGNUM"`, version, transform, flags {float32, zero-missing}, base_score,
sigmoid, n_features, section sizes), followed by 64-byte aligned sections: nodes, tree offsets, leaves.
Loading validates the version and the file size, then maps the file read-only. Older versions are
rejected; re-export from JSON.

## Known limitations

See [TASKS.md](TASKS.md): no categorical splits, no multiclass, only row-major float64 input.
