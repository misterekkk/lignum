---
title: How it works
description: From a model file to predictions - parsers, the two interpreter kernels and the compiled trees.
---

```
model file ──parser──▶ binary trees (IR) ──builder──▶ Model ──predict──▶ scores
                                                        │
                           model.save ──▶ .bin ──mmap──▶ Model (no copy)
                           model.compile ──▶ C code ──cc──▶ shared library
```

## Parsers

Each source format (LightGBM text and JSON, XGBoost JSON and UBJSON) is read into the same
library-neutral binary trees. All library-specific semantics are resolved here (see
[Bit-exact predictions](../exactness/)), so nothing downstream knows where a model came from. JSON is
parsed with [simdjson](https://github.com/simdjson/simdjson).

## Interpreter: two kernels

**Direct kernel.** Two tree levels per 64-byte node (`KNode`, one cache line), leaves in a separate
array. Four rows are walked at once to hide memory latency, and the three comparisons of a node are
evaluated without branches.

**Binned kernel (default).** Before the walk, every row is mapped to 16-bit codes: a numeric value
becomes the number of the feature's split thresholds below or at it, so `x >= t_k` is exactly
`code >= k + 1`. NaN gets the largest code. Categorical decisions are precomputed per row from a table
(category to decision bits). A 64-byte node (`BNode`) then holds **three** tree levels, compared as
small integers. It is faster than the direct kernel for almost every model.

Trees are processed in tiles that stay in cache while a block of rows walks them, and leaves are
always added in model order.

## Compiled trees

`model.compile()` generates C from the binned form: each tree becomes nested `if`s on constant
thresholds, with the comparisons written as integer immediates and the leaves in per-block tables.
Trees are grouped into blocks of 16; a block walks a whole chunk of rows (about 2 MB of codes and
sums) before the next block starts, which keeps its code and branch history hot. The most used
features of a block are read once per row into registers. Binning is generated too, specialised per
feature. The source is split into files compiled in parallel, linked into a shared library and
loaded with `dlopen` / `LoadLibrary`.

The library records a fingerprint of the model and the version of the generated code, so it can be
shipped next to the `.bin` file and attached without recompiling.

## Binary format

A 64-byte header followed by 64-byte aligned sections: nodes, tree offsets, leaves, tree kinds, base
scores, class of every tree, category sets, and the binned kernel's sections. Loading maps the file
read-only and validates every stored index (children must point forward, so every walk ends), which
takes a fraction of a millisecond. Corrupted files are tested by fuzzing under AddressSanitizer.

## Arrow input

Arrow data comes in through the Arrow C Data Interface. The binned kernel reads each feature column
of a chunk once, converts it exactly as numpy's `astype(float64)` would, and bins it directly; the
table is never copied into a matrix.
