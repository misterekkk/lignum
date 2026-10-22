"""Arrow input (PyCapsule interface): bit-identical to the float64 matrix path and the libraries."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import lignum as lg

pa = pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 4000
    X = rng.normal(size=(n, 6)) * 4
    X[:, 2] = rng.integers(-3, 4, size=n)       # small integers (exactly on thresholds)
    X[:, 3] = rng.integers(0, 2, size=n)        # 0/1
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 2]) > 0).astype(int)
    return X, y


def lgb_model(tmp_path, X, y, **kwargs):
    booster = lgb.train({"objective": "binary", "verbose": -1, **kwargs}, lgb.Dataset(X, y), num_boost_round=30)
    path = tmp_path / "lgb.json"
    path.write_text(json.dumps(booster.dump_model()))
    return booster, lg.load(str(path), format="lightgbm")


def table_of(X, types):
    """Columns of X as Arrow arrays of the given types; NaN becomes null (except in float columns)."""
    columns = []
    for j, t in enumerate(types):
        col = X[:, j]
        mask = np.isnan(col)
        if pa.types.is_floating(t):
            columns.append(pa.array(col, type=pa.float64()).cast(t))
        else:
            columns.append(pa.array(np.where(mask, 0, col).astype(t.to_pandas_dtype()), type=t, mask=mask))
    return pa.table(columns, names=[f"f{j}" for j in range(len(types))])


def as_matrix(table):
    """What the float64 path sees for the same data: numpy's astype(float64), nulls as NaN."""
    cols = [np.asarray(c.to_numpy(zero_copy_only=False), dtype=np.float64) for c in table.columns]
    return np.ascontiguousarray(np.column_stack(cols))


def test_float64_table_matches_matrix_and_lightgbm(tmp_path, data):
    X, y = data
    booster, model = lgb_model(tmp_path, X, y)
    table = table_of(X, [pa.float64()] * 6)
    expected = booster.predict(X)
    np.testing.assert_array_equal(model.predict(table), expected)                  # Table: stream path
    np.testing.assert_array_equal(model.predict(table.to_batches()[0]), expected)  # RecordBatch: array path
    np.testing.assert_array_equal(model.predict(table, raw_score=True), booster.predict(X, raw_score=True))


def test_mixed_column_types_with_nulls(tmp_path, data):
    X, y = data
    _, model = lgb_model(tmp_path, X, y)
    types = [pa.float32(), pa.float64(), pa.int8(), pa.bool_(), pa.int64(), pa.uint16()]
    Xi = X.copy()
    Xi[:, 4] = np.round(Xi[:, 4] * 1000)
    Xi[:, 5] = np.abs(np.round(Xi[:, 5] * 100))
    table = table_of(Xi, types)
    matrix = as_matrix(table)
    np.testing.assert_array_equal(model.predict(table), model.predict(matrix))
    for n_jobs in (1, 4):
        np.testing.assert_array_equal(model.predict(table, n_jobs=n_jobs), model.predict(matrix, n_jobs=n_jobs))


def test_null_column_and_extra_columns(tmp_path, data):
    X, y = data
    _, model = lgb_model(tmp_path, X, y)
    table = table_of(X, [pa.float64()] * 6)
    table = table.set_column(1, "f1", pa.nulls(len(X)))
    table = table.append_column("unused", pa.array(["text"] * len(X)))  # beyond n_features: ignored
    matrix = X.copy()
    matrix[:, 1] = np.nan
    np.testing.assert_array_equal(model.predict(table), model.predict(matrix))


def test_slices_chunks_and_streams(tmp_path, data):
    X, y = data
    booster, model = lgb_model(tmp_path, X, y)
    table = table_of(X, [pa.float64()] * 6)
    batch = table.to_batches()[0]
    np.testing.assert_array_equal(model.predict(batch.slice(1234, 777)), booster.predict(X[1234:2011]))

    chunked = pa.Table.from_batches([batch.slice(0, 1000), batch.slice(1000, 1), batch.slice(1001)])
    assert chunked.column(0).num_chunks == 3
    np.testing.assert_array_equal(model.predict(chunked), booster.predict(X))

    reader = pa.RecordBatchReader.from_batches(batch.schema, iter([batch.slice(0, 10), batch.slice(10)]))
    np.testing.assert_array_equal(model.predict(reader), booster.predict(X))
    np.testing.assert_array_equal(model.predict(batch.slice(0, 0)), np.empty(0))


def test_large_batch_spans_several_chunks(tmp_path, data):
    X, y = data
    booster, model = lgb_model(tmp_path, X, y)
    big = np.tile(X, (40, 1))  # 160k rows: three conversion chunks
    np.testing.assert_array_equal(model.predict(table_of(big, [pa.float64()] * 6)), booster.predict(big))


def test_dictionary_columns_are_categories(tmp_path, data):
    X, y = data
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(X[:, :4], columns=["a", "b", "c", "d"])
    frame["cat"] = pd.Categorical(rng.choice(["x", "y", "z", "w", None], size=len(X)))
    target = (np.nan_to_num(X[:, 0]) + (frame["cat"] == "y") * 2 > 0.5).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1, "max_cat_to_onehot": 2, "min_data_per_group": 5},
                        lgb.Dataset(frame, target, categorical_feature=["cat"]), num_boost_round=30)
    path = tmp_path / "cat.json"
    path.write_text(json.dumps(booster.dump_model()))
    model = lg.load(str(path), format="lightgbm")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    assert pa.types.is_dictionary(table.schema.field("cat").type)
    np.testing.assert_array_equal(model.predict(table), booster.predict(frame))


def test_xgboost_multiclass_from_arrow(tmp_path, data):
    X, y = data
    labels = (np.nan_to_num(X[:, 0]) > 1).astype(int) + (np.nan_to_num(X[:, 1]) > 0)
    booster = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 4}, xgb.DMatrix(X, labels), num_boost_round=10)
    path = tmp_path / "xgb.json"
    booster.save_model(str(path))
    model = lg.load(str(path), format="xgboost")
    out = model.predict(table_of(X, [pa.float32()] * 6))
    X32 = X.astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(out, booster.predict(xgb.DMatrix(X32)))
    assert out.shape == (len(X), 3)


def test_rejects_bad_arrow_input(tmp_path, data):
    X, y = data
    _, model = lgb_model(tmp_path, X, y)
    table = table_of(X, [pa.float64()] * 6)
    with pytest.raises(ValueError, match="Unsupported Arrow type 'u'.*column 2 \\('f2'\\)"):
        model.predict(table.set_column(2, "f2", pa.array(["a"] * len(X))))
    with pytest.raises(ValueError, match="requires at least 6"):
        model.predict(table.select([0, 1, 2]))
