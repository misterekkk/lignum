---
title: Arrow input
description: Predict on pyarrow, polars or any Arrow data without copying it into a matrix.
---

`predict` accepts Arrow data through the
[PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html):
`pyarrow.RecordBatch`, `Table` and `RecordBatchReader`, polars `DataFrame`, and any object with
`__arrow_c_array__` or `__arrow_c_stream__`. lignum itself does not depend on pyarrow.

```python
import pyarrow as pa

table = pa.Table.from_pandas(df)    # categorical pandas columns become dictionary columns
preds = model.predict(table)        # one result for all batches of the table

for batch in reader:                # e.g. a stream of record batches
    preds = model.predict(batch)
```

## Columns

The first `model.n_features` columns are the features, by position, exactly like the columns of a
matrix; further columns are ignored, whatever their type. When the model has feature names, the
schema's column names must start with them, in the same order (`ValueError` otherwise; see
[column names](../predicting/#column-names)); `table.select(model.feature_names)` puts a table's
columns in the model's order.

| Arrow type | Value |
|---|---|
| float64, float32 | the value (float32 widened exactly) |
| int8 ... int64, uint8 ... uint64 | the value as a double, as numpy's `astype(float64)` gives it |
| boolean | 0 or 1 |
| dictionary (categorical) | the index, i.e. the category code |
| null | NaN |
| a null entry in any column | NaN |

Other types (strings, decimals, dates, float16) raise `ValueError` naming the column. Predictions are
the same as for the equivalent `float64` matrix, bit for bit.

## Performance

The binned kernel bins each column straight from the Arrow buffers, so the table is never copied
into a row-major matrix. On an Apple M4 (500 trees, 10 threads):

| Data set | numpy matrix | Arrow table | Arrow table → numpy → predict |
|---|---:|---:|---:|
| Covertype, compiled | 62 ms | 67 ms | 256 ms |
| Covertype, interpreter | 253 ms | 255 ms | — |
| Diabetes, compiled | 43 ms | 43 ms | — |

For a single small batch, pyarrow's own export costs about 0.2 µs per column; for single-row
latency from Python, a numpy row is the faster input.

## C and C++

Services that hold Arrow data in C, C++, Rust or Go pass it through the Arrow C Data Interface,
without Python: `LignumModelPredictArrow` in the [C API](../c-api/), or `lignum::predict_arrow` and
`lignum::predict_arrow_stream` in C++. The structures are in `include/lignum/arrow.h`, copied from
the Arrow specification; libarrow is not needed.
