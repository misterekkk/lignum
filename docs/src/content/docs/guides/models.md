---
title: Model files
description: Which LightGBM, XGBoost and CatBoost files lignum reads, and the binary format for production.
---

`lignum.load(path)` recognises a file by its contents, not by its name:

| Library | File | Written by |
|---|---|---|
| LightGBM | `model.txt` (text format) | `booster.save_model("model.txt")` |
| LightGBM | JSON | `json.dump(booster.dump_model(), f)` |
| XGBoost | JSON | `booster.save_model("model.json")` |
| XGBoost | UBJSON | `booster.save_model("model.ubj")` |
| CatBoost | `.cbm` | `model.save_model("model.cbm")` |
| CatBoost | JSON | `model.save_model("model.json", format="json")` (not for multiclass) |
| lignum | `.bin` | `model.save("model.bin")` |

Every source gives the same model and the same, bit-identical predictions. `format="lightgbm"`,
`format="xgboost"` or `format="catboost"` only adds a check that the file comes from that library.
CatBoost models have their own page: [CatBoost](../catboost/).

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
| CatBoost, Adult | 0.76 ms (`.cbm`) | 0.15 ms |
| CatBoost, Diabetes 130-US (large CTR tables) | 12.8 ms (`.cbm`) | 2.3 ms |

Measured on an Apple M4; parsing only, without building the binned kernel. Convert once and load the
`.bin` file in production, and keep the original model file: a `.bin` file from an older lignum
version is rejected, and is written again from the original. Format 0.8 (lignum 0.4) adds the
feature names, the library the model came from and the lignum version that wrote the file; files
written by lignum 0.3 (format 0.6 and 0.7) must be saved again.

## What a model must look like

- **Features** are columns by position: feature `j` of the model is column `j` of the input. A model
  needs at least `model.n_features` columns; extra columns after them are ignored. Named inputs
  (pandas, Arrow) are checked against the model's `feature_names` first, see
  [column names](../predicting/#column-names).
- **Categorical features** take the integer category codes the library was trained on. For pandas
  `category` columns these are `series.cat.codes` (NaN for missing), which is also what a dictionary
  column in [Arrow input](../arrow/) gives. CatBoost models take the category values themselves
  (strings or integers), see [CatBoost](../catboost/#categorical-input).
- **Multiclass** models predict one column per class (`model.n_outputs`).

## Not supported

These are rejected with an error rather than predicted wrongly:

- LightGBM linear trees (`linear_tree=True`) and random forests (`boosting="rf"`).
- XGBoost multi-target models (`num_target > 1`).
- CatBoost `Depthwise` and `Lossguide` trees, and text or embedding features.
