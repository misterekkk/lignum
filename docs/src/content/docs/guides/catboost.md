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
the category's hash as a float64: `lignum.catboost_hash("red")` gives it. A float64 matrix built
that way goes straight to the kernel. This is also how C and C++ callers pass categories:
`lignum::catboost_hash` in C++.

## Comparing with CatBoost

Compare against the trained model, or the same model saved as `.cbm`. CatBoost's own JSON loader
rounds some leaf values, so a CatBoost model loaded back from JSON can differ in the last bits from
the model that wrote the file. lignum reads the JSON exactly, so it matches the original.

## Speed

A model with 500 trees of depth 6, 2 float and 3 categorical features, on an Apple M4 (10 cores):

| | CatBoost 1.2.10 | lignum |
|---|---:|---:|
| 500k rows from a DataFrame, all threads | 0.188 s | **0.102 s** |
| 500k rows, one thread | 0.669 s | **0.445 s** |
| One row from a list | 49.7 µs | **3.7 µs** |
| One row from a DataFrame | 95.9 µs | **22 µs** |
| Loading the model | 0.8 ms (`.cbm`) | **0.2 ms** (`.bin`), 0.9 ms (`.cbm`) |
| 500k rows from Arrow (pyarrow or polars) | 0.114 s (polars) | **0.077 s** |

On large batches most of the gain is the conversion: lignum hashes categories in C++ about three
times faster than building a CatBoost `Pool`. From an already built `Pool`, CatBoost takes 0.109 s.
