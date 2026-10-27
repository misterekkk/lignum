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

## Column names

A model trained on named columns (a pandas DataFrame, or `feature_names` in XGBoost) knows them:
`model.feature_names`. When the input has column names too (a pandas DataFrame, pyarrow, polars or
other Arrow data), `predict` checks that its columns start with those names, in the same order, and
raises `ValueError` otherwise:

```python
model.feature_names                  # ['age', 'income', 'debt']
model.predict(df)                    # df columns: age, income, debt (+ any extra columns after them)
model.predict(df[["income", "age", "debt"]])
# ValueError: Column 0 is 'income' but the model expects 'age' there; ...
model.predict(df[model.feature_names])   # selects the model's columns, in its order
```

A reordered, missing or extra leading column would otherwise give other predictions without any
error, because the trees read columns by position. lignum does not reorder columns for you: a
mismatch usually means the data pipeline changed, and the error makes that visible.

- **Not checked**: numpy arrays (they have no names) and data frames without column names (e.g.
  built from a numpy array) are always used by position, as are models trained on a matrix, which
  have no names (`model.feature_names` is `None`; LightGBM's `Column_0, Column_1, ...` count as no
  names). The [C API](../c-api/) does not check names either.
- **LightGBM** stores spaces in names as `_`; a column `my income` matches the model's `my_income`.
- **Turn it off** for one call with `model.predict(X, validate_features=False)`, or for a model with
  `lignum.load(path, validate_features=False)` or `model.validate_features = False`. The columns are
  then used by position, as in lignum 0.3.

### Cost

The check runs once per call, not per row. Apple M4, LightGBM with 500 trees, one thread, median
over thousands of calls of the difference between calls with and without the check
([`benchmarks/bench_validation.py`](https://github.com/misterekkk/lignum/blob/main/benchmarks/bench_validation.py)):

| Input | 20 columns, 1 row | 20 columns, 100 rows | 200 columns, 1 row | 200 columns, 100 rows |
|---|---:|---:|---:|---:|
| pandas, a new DataFrame per call | +2.8-3.3 µs | +3.2-3.9 µs | +3.7-3.9 µs | +4.2-4.5 µs |
| pandas, the same DataFrame again | +0.2 µs | +0.2-0.4 µs | +0.2 µs | +0.0-0.2 µs |
| pyarrow Table | +0.0-0.1 µs | +0.0 µs | +0.7 µs | +0.2-0.4 µs |
| numpy | 0 | 0 | 0 | 0 |

For comparison, the same calls take 13-90 µs for one row and 0.45-0.66 ms for 100 rows (most of it
pandas' own cost for small frames); from 10,000 rows on the difference is below the noise of the
measurement. Where every microsecond counts, pass a numpy array: it skips both pandas and the
check. All other timings in these docs and in the [benchmarks](../../benchmarks/) are measured with
`validate_features=False`.

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
