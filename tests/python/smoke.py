"""Smoke test for built wheels: imports, loads a model, predicts (interpreter, threads, .bin).

Needs numpy only, so it runs on every Python a wheel is built for.
"""

import json
import os
import tempfile

import numpy as np

import lignum

MODEL = {"objective": "regression", "tree_info": [{"tree_structure": {
    "split_feature": 1, "threshold": 0.5, "decision_type": "<=", "default_left": True, "missing_type": "NaN",
    "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}}]}

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model.json")
    with open(path, "w") as f:
        json.dump(MODEL, f)
    model = lignum.load(path)
    X = np.array([[0.0, 0.1], [0.0, 0.9], [0.0, np.nan]] * 2000)
    expected = np.array([1.0, 2.0, 1.0] * 2000)
    for n_jobs in (1, -1):
        assert np.array_equal(model.predict(X, n_jobs=n_jobs), expected)
    model.save(os.path.join(tmp, "model.bin"))
    assert np.array_equal(lignum.load(os.path.join(tmp, "model.bin")).predict(X), expected)
print(f"lignum {lignum.__version__} OK")
