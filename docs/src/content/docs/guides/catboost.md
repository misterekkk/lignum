---
title: CatBoost
description: CatBoost models with categorical features and CTRs, bit-exact, from pandas, numpy or lists.
---

lignum predicts CatBoost models exactly like CatBoost: the same raw scores and probabilities, bit for
bit. This covers categorical features, one-hot encoding, and CTRs (CatBoost's target statistics),
including combinations of several categories.

```python
model.save_model("model.cbm")   # CatBoost (or format="json")

import lignum
lm = lignum.load("model.cbm")   # .cbm or JSON; lm.save("model.bin") for lignum's own format
lm.predict(df)                  # pandas DataFrame with the training columns, in order
lm.predict(table)               # pyarrow Table / polars DataFrame: strings and categoricals read in place
lm.predict(["a", 1.5, 7])       # one row as a list
```

## Supported models

- **Trees**: CatBoost's default symmetric trees (`grow_policy="SymmetricTree"`), any depth up to 16.
- **Features**:
  - float features, with NaN in `Min` or `Max` mode;
  - categorical features, as one-hot or CTRs (`Borders`, `Buckets`, `Counter`,
    `BinarizedTargetMeanValue`), including combinations of categories with each other and with
    binarised float features (`max_ctr_complexity`).
- **Losses**: for the classifier, `predict` gives the probability of class 1 (`Logloss`,
  `CrossEntropy`), or one probability per class for `MultiClass` (softmax) and
  `MultiClassOneVsAll` (sigmoid per class). For regression it gives what CatBoost's `predict` gives: `exp` of the raw score
  for `Poisson` and `Tweedie`, the raw score otherwise.
- **Files**: CatBoost's `.cbm` (`save_model("model.cbm")`, read directly, without the flatbuffers
  library) and JSON (`format="json"`, which cannot hold multiclass models). `model.save("model.bin")`
  writes lignum's format, which loads in about 0.2 ms.

Not yet supported:

- `Depthwise` and `Lossguide` trees;
- text and embedding features.

`compile()` is not needed: symmetric trees have no branches, so the interpreter is already the fast
path. Unsupported models are rejected at load time with a message that says why.

## Categorical input

Categorical columns hold the same values as in training:

- **Types**: strings, or integers, which CatBoost hashes as their decimal text (so `7` and `"7"`
  are the same category).
- **Containers**:
  - pandas columns: object, string or category dtype;
  - Arrow: `string`, `large_string` and `string_view` columns, integer columns, and dictionaries
    of strings or integers (pandas and polars categoricals). Dictionary values are hashed once
    per batch;
  - numpy object arrays;
  - lists and tuples.
- **Rejected, as in CatBoost**: floats, NaN, `None` or Arrow nulls in a categorical column. Turn missing values
  into a string such as `"nan"` before training and predicting.
- **Categories the model never saw** are allowed and get CatBoost's prior, as in CatBoost.

`model.categorical_features` lists the categorical columns. Underneath, a categorical column holds
the category's hash as a float64: `lignum.catboost_hash("red")` gives it. In Python, `predict`
hashes the categories itself and rejects a float matrix for a model with categorical features, since
category codes would look like valid hashes. C and C++ callers pass the hashes in a float64 matrix:
`LignumCatBoostHash` in the [C API](../c-api/), `lignum::catboost_hash` in C++.

## Comparing with CatBoost

Compare against the trained model, or the same model saved as `.cbm`. CatBoost's own JSON loader
rounds some leaf values, so a CatBoost model loaded back from JSON can differ in the last bits from
the model that wrote the file. lignum reads the JSON exactly, so it matches the original.

## Speed

500 trees of depth 6 on public data sets, CatBoost 1.2.10 / lignum, from Python with a pandas
DataFrame (categories as strings), a different row on every call:

| | x86-64 (EPYC-Milan) | Apple M4 |
|---|---:|---:|
| Adult, 1M rows, one thread | 2.24 / **1.25 s** | 1.55 / **0.613 s** |
| Diabetes 130-US, 1M rows, one thread | 7.42 / **4.20 s** | 3.66 / **1.55 s** |
| Covertype, 1M rows, one thread | 1.76 / **0.593 s** | 1.19 / **0.437 s** |
| California housing, 1M rows, one thread | 0.344 / **0.292 s** | 0.415 / **0.159 s** |
| One row, Adult (DataFrame) | 582 / **55.1 µs** | 245 / **27.5 µs** |
| One row, Diabetes 130-US (DataFrame) | 5,761 / **237 µs** | 2,417 / **102 µs** |
| One row, Adult (pyarrow Table) | 5,567 / **18.2 µs** | 755 / **9.1 µs** |
| One row, Diabetes 130-US (pyarrow Table) | 31,577 / **98.4 µs** | 4,759 / **32.8 µs** |

For single rows most of CatBoost's time is converting its input; lignum reads Arrow tables directly
(CatBoost needs `to_pandas()` first). On purely numeric models with all threads on x86, CatBoost
comes within 5-11%. Every data set, all threads and the methodology:
[Benchmarks](../../benchmarks/#catboost-models).

Loading, on the M4: Adult takes 0.15 ms from lignum's `.bin`, 0.76 ms from `.cbm` (CatBoost: 0.33
ms). Models with large CTR tables load more slowly, because lignum copies the tables into its own
hash layout: Diabetes 130-US (395,000 table entries) takes 2.3 ms from `.bin` and 13 ms from `.cbm`
(CatBoost: 1.7 ms). Convert once with `save("model.bin")` and load the `.bin` in production.

Where the time goes: lignum turns each row's features into small integer buckets once per block of
256 rows (floats by counting borders, categories by hashing them and looking up their CTR tables),
then evaluates the trees on bytes, 16 or 32 rows at a time with SIMD instructions. On x86-64 the
kernel is built twice and the AVX2 build is chosen at run time on CPUs that have it.
