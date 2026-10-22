---
title: Predicting
description: Inputs, outputs, threads and the two interpreter kernels.
---

```python
preds = model.predict(X)                    # transformed (probabilities for classifiers)
margins = model.predict(X, raw_score=True)  # raw scores
```

## Input

- **numpy**: a C-contiguous `float64` array of shape `(n_rows, n_columns)` is read in place; anything
  else numpy can convert (other dtypes, Fortran order, lists) is copied to that form first.
- **Arrow**: pyarrow tables and batches, polars data frames and other Arrow data are read without
  copying the table. See [Arrow input](../arrow/).
- **pandas**: data frames go through numpy. For categorical columns, pass
  `pyarrow.Table.from_pandas(df)` instead, which keeps them as category codes.

NaN is a missing value, with the same direction rules as the library (including LightGBM's
`zero_as_missing` and `use_missing=False`).

## Output

| Model | `predict(X)` | `predict(X, raw_score=True)` |
|---|---|---|
| regression, binary | `(n_rows,)` | `(n_rows,)` |
| multiclass, K classes | `(n_rows, K)` probabilities | `(n_rows, K)` margins |
| XGBoost `multi:softmax` | `(n_rows,)` class index | `(n_rows, K)` margins |

## Threads

```python
model.predict(X, n_jobs=-1)   # all cores (default)
model.predict(X, n_jobs=1)    # the calling thread only
```

`strategy` decides how rows are spread over threads: `"serial"` runs on the calling thread,
`"rows"` splits the rows between threads, and `"auto"` (the default) uses `serial` below 8 rows or
with one thread. Every strategy gives the same result; only the time differs.

`predict` releases the GIL, and a model can be used from several threads at once. `compile`,
`load_compiled` and `decompile` change the model: call them before sharing it.

## Two interpreter kernels

By default batches use the **binned kernel**: each row is mapped once to 16-bit bin codes and its
categorical decisions are precomputed, so the tree walk compares small integers, three tree levels
per cache line. `lignum.load(path, binned=False)` keeps only the direct traversal, which uses less
memory. Both are exact; the binned kernel is faster for almost every model.

For the largest batches, [compiled mode](../compiled/) turns the trees into native code.

## Latency

A single row of a 500-tree model takes about 5-13 µs with the interpreter and 2-9 µs with compiled trees in the [benchmarks](../../benchmarks/). Keep the model
loaded, pass one contiguous `float64` row, and leave `strategy="auto"`: small batches stay on the
calling thread, without waking the thread pool.
