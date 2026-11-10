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

## Inspecting a model

Before serving a file, check that it is the model you expect:

```python
>>> model = lignum.load("california.bin")  # LightGBM, 500 trees, saved by lignum 0.4.0
>>> model
<lignum.Model lightgbm regression, 500 trees, 8 features>
>>> model.info()
{'source': 'lightgbm', 'objective': 'regression', 'transform': 'identity', 'n_trees': 500,
 'n_leaves': 15500, 'n_features': 8, 'n_outputs': 1,
 'feature_names': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'],
 'categorical_features': [], 'kernel': 'binned', 'compiled_library': None, 'written_by': '0.4.0'}
```

`n_trees` counts every tree, so a multiclass model has one per class and iteration (LightGBM's
`num_trees()`, XGBoost's `len(booster.get_dump())`, CatBoost's `tree_count_`). `objective`
is the library's own name for it; `transform` is what lignum applies to the raw scores
(`predict(raw_score=True)` skips it). `kernel` says which code predicts: `binned` or `direct`
(the interpreter), `compiled` (after `compile()`) or `catboost`. `written_by` is the lignum version
that wrote a `.bin` file, and `None` for the libraries' own files.

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
version is rejected, and is written again from the original. Format 0.9 (lignum 0.4.2) adds the
categories of training of pandas `category` columns; lignum reads 0.8 files too. Format 0.8 (lignum 0.4) adds the
feature names, the library the model came from, its objective and the lignum version that wrote the
file (`model.info()["written_by"]`); files written by lignum 0.3 (format 0.6 and 0.7) must be saved
again.

The file is memory-mapped while a model uses it. Replace it with a new file (write it under another
name, then rename it) rather than overwriting it in place, which would change or cut off the data of
running models.

## What a model must look like

- **Features** are columns by position: feature `j` of the model is column `j` of the input. A model
  needs at least `model.n_features` columns; extra columns after them are ignored. Named inputs
  (pandas, Arrow) are checked against the model's `feature_names` first, see
  [column names](../predicting/#column-names).
- **Categorical features** take the integer category codes the library was trained on. For pandas
  `category` columns these are `series.cat.codes` (NaN for missing), which is also what a dictionary
  column in [Arrow input](../arrow/) gives. CatBoost models take the category values themselves
  (strings or integers), see [CatBoost](../catboost/#categorical-input).

### pandas `category` columns

A model trained on a DataFrame with `category` columns stores their categories (LightGBM's
`pandas_categorical`, XGBoost 3's `cats`), and the library's `predict(df)` maps a DataFrame's own
categories onto them: a category gets the code it had in training, whatever the order or subset of
categories of the input (for instance categories made from the current batch with
`astype("category")`). lignum does the same, for pandas and for Arrow dictionary columns (pyarrow,
polars), with the libraries' rules:

| | LightGBM | XGBoost |
|---|---|---|
| which column a list belongs to | the k-th `category` column of the input | its column in training |
| a category not in training | missing (NaN) | `ValueError` |
| other `category` columns than in training | `ValueError` | `ValueError` |

Columns of numbers are taken as codes, as they are: the codes of training must be in them. The
categories are part of the `.bin` file.
- **Multiclass** models predict one column per class (`model.n_outputs`).

## Not supported

These are rejected with an error rather than predicted wrongly:

- LightGBM linear trees (`linear_tree=True`) and random forests (`boosting="rf"`).
- XGBoost multi-target models (`num_target > 1`), linear boosters (`booster="gblinear"`) and DART
  models that dropped trees (`rate_drop > 0`). DART models that kept every tree are supported, and
  so are all LightGBM DART models.
- CatBoost `Depthwise` and `Lossguide` trees, and text or embedding features.
- Objectives lignum does not know, so that it cannot apply the library's output transform. Every
  objective in the documentation of LightGBM 4, XGBoost 3 and CatBoost is known and tested except
  CatBoost's `RMSEWithUncertainty`, `QueryCrossEntropy` (trained on GPU only, not tested yet) and
  Python objectives (a CatBoost file does not record whether such a model predicts raw scores or
  probabilities). LightGBM and XGBoost models trained with a Python objective predict raw scores, as
  the libraries do.

## Objectives

`predict` gives what the library's own prediction gives: `booster.predict` for LightGBM and
XGBoost, `predict_proba` for CatBoost classifiers and `predict` for CatBoost regressors and rankers.
`model.info()["transform"]` names the transform from raw scores:

| Transform | Objectives |
|---|---|
| `identity` | regression, ranking and quantile objectives (LightGBM `regression`, `huber`, `lambdarank`...; XGBoost `reg:squarederror`, `rank:ndcg`...; CatBoost `RMSE`, `Quantile`, `YetiRank`...), XGBoost `binary:logitraw` |
| `sigmoid` | LightGBM `binary`, `multiclassova`, `cross_entropy`; XGBoost `binary:logistic`, `reg:logistic`; CatBoost `Logloss`, `CrossEntropy`, `Focal`, `MultiClassOneVsAll`, `MultiLogloss`, `MultiCrossEntropy` |
| `softmax` | LightGBM `multiclass`, XGBoost `multi:softprob`, CatBoost `MultiClass` |
| `exp` | `poisson`, `gamma`, `tweedie` (all three libraries), XGBoost `survival:cox`, `survival:aft` |
| `log1p_exp` | LightGBM `cross_entropy_lambda` |
| `argmax` | XGBoost `multi:softmax` (the class index) |
| `signed_square` | LightGBM with `reg_sqrt=True` (`regression`, `regression_l1`, `fair`, `quantile`, `mape`) |
| `hinge` | XGBoost `binary:hinge` (0 or 1) |

`predict(X, raw_score=True)` skips the transform.

## Early stopping

lignum predicts with every tree in the file, like the library's `Booster.predict`:

- **LightGBM** `save_model` writes the trees up to the best iteration by default, so the file and
  `predict` agree.
- **XGBoost** keeps every tree in the file. `Booster.predict` uses all of them, but the scikit-learn
  wrappers (`XGBClassifier.predict_proba`, `XGBRegressor.predict`) stop at `best_iteration`. To
  serve what the wrapper predicts, save the trees up to it:

```python
clf.get_booster()[: clf.best_iteration + 1].save_model("model.json")
lignum.load("model.json").predict(X)   # == clf.predict_proba(X)[:, 1], bit for bit
```

## Limits

- At most 65,536 input columns for LightGBM and XGBoost models, and 65,535 float, one-hot and CTR
  features for CatBoost models.
- Trees at most 1,024 levels deep, and category sets of a model at most about 256 MB (categories
  below 2^31).
- The binned kernel needs fewer than 65,534 thresholds per feature and categories below 65,535;
  other models use the direct kernel, with the same results (`model.uses_binned`).
