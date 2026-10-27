---
title: Getting started
description: Install lignum, load a model and check that its predictions equal the library's.
---

## Install

```bash
pip install pylignum
```

The package is installed as `pylignum` and imported as `lignum`. It needs Python 3.10 or newer and
numpy. Wheels exist for Linux (x86_64, aarch64), macOS 15+ (arm64, x86_64) and Windows (x64); on
other platforms pip builds it from source, which needs a C++20 compiler and CMake 3.18+.

## Load a model and predict

lignum reads the files LightGBM, XGBoost and CatBoost write. There is nothing to convert first:

```python
import lightgbm as lgb
import numpy as np
import lignum

X = np.random.default_rng(0).normal(size=(10_000, 20))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=200)
booster.save_model("model.txt")

model = lignum.load("model.txt")
preds = model.predict(X)

assert np.array_equal(preds, booster.predict(X))   # equal, not merely close
```

`predict` returns probabilities for classifiers (`raw_score=True` gives margins), one column per class
for multiclass models. See [Predicting](../predicting/) for threads, input types and output shapes.

## CatBoost models with categories

CatBoost models take the same DataFrame CatBoost does, categories as strings included:

```python
from catboost import CatBoostClassifier

clf = CatBoostClassifier(iterations=200, verbose=False).fit(df, y, cat_features=["city", "device"])
clf.save_model("model.cbm")

model = lignum.load("model.cbm")
assert np.array_equal(model.predict(df), clf.predict_proba(df)[:, 1])
```

[CatBoost](../catboost/) lists the supported models and the accepted input types.

## Save a binary model for production

```python
model.save("model.bin")
model = lignum.load("model.bin")   # memory-mapped: well under a millisecond
```

The `.bin` file holds everything lignum needs, including the binned form of the trees, and is checked
when it is loaded. Processes that map the same file share its memory.

## Go faster with compiled trees

For large batches, compile the trees to native code once (it takes a few seconds and needs a C
compiler) and ship the library next to the model:

```python
model.compile(library="model.so")                       # build machine
model = lignum.load("model.bin", compiled="model.so")   # production: no compiler needed
```

[Compiled mode](../compiled/) explains when it pays off and how `autotune` picks the faster path for
your data.
