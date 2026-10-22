---
title: Model files
description: Which LightGBM and XGBoost files lignum reads, and the binary format for production.
---

`lignum.load(path)` recognises a file by its contents, not by its name:

| Library | File | Written by |
|---|---|---|
| LightGBM | `model.txt` (text format) | `booster.save_model("model.txt")` |
| LightGBM | JSON | `json.dump(booster.dump_model(), f)` |
| XGBoost | JSON | `booster.save_model("model.json")` |
| XGBoost | UBJSON | `booster.save_model("model.ubj")` |
| lignum | `.bin` | `model.save("model.bin")` |

Every source gives the same model and the same, bit-identical predictions. `format="lightgbm"` or
`format="xgboost"` only adds a check that the file comes from that library.

```python
model = lignum.load("model.txt")                    # LightGBM
model = lignum.load("model.ubj", format="xgboost")  # XGBoost, checked
```

## The binary format

`model.save("model.bin")` writes lignum's own format: the trees in the layout the kernels read, the
binned form, leaves, base scores and the metadata, in 64-byte aligned sections. Loading maps the file
read-only and checks every stored index, so a damaged file is rejected instead of crashing.

| Model (500 trees) | From the library's file | From `.bin` |
|---|---:|---:|
| LightGBM, Covertype | 7.2 ms (`model.txt`) | 0.2 ms |
| LightGBM, 5000 trees | 67 ms (`model.txt`) | 0.7 ms |
| XGBoost, 3000 trees | 36 ms (JSON), 63 ms (UBJSON) | — |

Measured on an Apple M4; parsing only, without building the binned kernel. Convert once and load the
`.bin` file in production. A `.bin` file from an older lignum version is rejected: re-export it from
the original model.

## What a model must look like

- **Features** are columns by position: feature `j` of the model is column `j` of the input. A model
  needs at least `model.n_features` columns; extra columns are ignored.
- **Categorical features** take the integer category codes the library was trained on. For pandas
  `category` columns these are `series.cat.codes` (NaN for missing), which is also what a dictionary
  column in [Arrow input](../arrow/) gives.
- **Multiclass** models predict one column per class (`model.n_outputs`).

## Not supported

These are rejected with an error rather than predicted wrongly:

- LightGBM linear trees (`linear_tree=True`) and random forests (`boosting="rf"`).
- XGBoost multi-target models (`num_target > 1`).
