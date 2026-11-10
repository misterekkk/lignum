"""Seed corpora for the libFuzzer targets: small models of every format and feature.

    python tests/fuzz/make_corpus.py <out_dir>

writes <out_dir>/{lightgbm,xgboost,catboost,lignum}/: LightGBM model.txt and dump_model() JSON,
XGBoost JSON and UBJSON, CatBoost .cbm and JSON (numeric, categorical, missing values, zero as
missing, multiclass, CTRs), and lignum's .bin of each. Needs lightgbm, xgboost, catboost, pandas
and lignum (`pip install ".[test]"`). Models are small, so the fuzzer runs many inputs per second.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

import lignum

warnings.filterwarnings("ignore")


def main(out: Path) -> None:
    import catboost as cb
    import lightgbm as lgb
    import pandas as pd
    import xgboost as xgb

    for d in ("lightgbm", "xgboost", "catboost", "lignum"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    X[rng.random(X.shape) < 0.1] = np.nan
    X[rng.random(X.shape) < 0.1] = 0.0
    cat = rng.integers(0, 70, size=n).astype(float)  # bitsets: categories above 63
    y = np.nan_to_num(X[:, 0]) + (cat % 3 == 0)
    yb = (y > np.median(y)).astype(int)
    y3 = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    Xc = np.column_stack([X, cat])

    def lightgbm(name, params, data, label, rounds=3, categorical="auto"):
        b = lgb.train({"verbose": -1, "num_leaves": 5, "min_data_in_leaf": 5, "min_data_per_group": 5, **params},
                      lgb.Dataset(data, label, categorical_feature=categorical), rounds)
        b.save_model(str(out / "lightgbm" / f"{name}.txt"))
        (out / "lightgbm" / f"{name}.json").write_text(json.dumps(b.dump_model()))

    lightgbm("regression", {"objective": "regression"}, X, y)
    lightgbm("zero_missing", {"objective": "binary", "zero_as_missing": True}, X, yb)
    lightgbm("no_missing", {"objective": "regression", "use_missing": False, "reg_sqrt": True}, X, y)
    lightgbm("categorical", {"objective": "binary"}, Xc, yb, categorical=[3])
    lightgbm("multiclass", {"objective": "multiclass", "num_class": 3}, Xc, y3, rounds=2, categorical=[3])

    def xgboost(name, params, dmatrix, rounds=3):
        b = xgb.train({"max_depth": 2, **params}, dmatrix, rounds)
        b.save_model(str(out / "xgboost" / f"{name}.json"))
        b.save_model(str(out / "xgboost" / f"{name}.ubj"))

    xgboost("regression", {"objective": "reg:squarederror"}, xgb.DMatrix(X, y))
    xgboost("logistic", {"objective": "binary:logistic"}, xgb.DMatrix(X, yb))
    xgboost("hinge", {"objective": "binary:hinge"}, xgb.DMatrix(X, yb))
    xgboost("multiclass", {"objective": "multi:softprob", "num_class": 3}, xgb.DMatrix(X, y3), rounds=2)
    frame = pd.DataFrame({"a": X[:, 0], "c": pd.Categorical(cat.astype(int))})
    xgboost("categorical", {"objective": "binary:logistic", "tree_method": "hist"}, xgb.DMatrix(frame, yb, enable_categorical=True))

    kw = {"iterations": 3, "depth": 2, "verbose": False, "allow_writing_files": False, "random_seed": 0}
    df = pd.DataFrame({"a": np.nan_to_num(X[:, 0]), "b": X[:, 1], "c": [f"v{int(c) % 7}" for c in cat],
                       "d": [f"w{int(c) % 3}" for c in cat]})
    models = {
        "numeric": cb.CatBoostRegressor(**kw).fit(np.nan_to_num(X), y),
        "nan_max": cb.CatBoostRegressor(nan_mode="Max", **kw).fit(X, y),
        "ctr": cb.CatBoostClassifier(cat_features=["c", "d"], one_hot_max_size=2, max_ctr_complexity=2, **kw).fit(df, yb),
        "one_hot": cb.CatBoostClassifier(cat_features=["d"], one_hot_max_size=10, **kw).fit(df[["a", "d"]], yb),
        "multiclass": cb.CatBoostClassifier(loss_function="MultiClass", **kw).fit(np.nan_to_num(X), y3),
        "poisson": cb.CatBoostRegressor(loss_function="Poisson", **kw).fit(np.nan_to_num(X), np.abs(y) + 1),
    }
    for name, m in models.items():
        m.save_model(str(out / "catboost" / f"{name}.cbm"))
        if name != "multiclass":  # CatBoost's JSON has no multiclass models
            m.save_model(str(out / "catboost" / f"{name}.json"), format="json")

    for d in ("lightgbm", "xgboost", "catboost"):
        for path in sorted((out / d).iterdir()):
            lignum.load(str(path)).save(str(out / "lignum" / f"{d}_{path.stem}_{path.suffix[1:]}.bin"))
    for d in ("lightgbm", "xgboost", "catboost", "lignum"):
        print(d, len(list((out / d).iterdir())), "files")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
