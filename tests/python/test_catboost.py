"""Bit-exact parity with CatBoost (symmetric trees, categorical features, CTRs), from JSON and .cbm.

References are the trained models themselves (the same as `.cbm`): CatBoost's own JSON loader
rounds leaf values, so a model loaded back from JSON into CatBoost predicts slightly differently.
Inputs include unseen categories, the empty category, NaN in float features and values exactly on
the float borders. CatBoost models have one kernel, so these tests run once (not per kernel).
"""

import json
import string

import numpy as np
import pandas as pd
import pytest

import lignum

cb = pytest.importorskip("catboost")

N_ROWS = 4000


@pytest.fixture(autouse=True)
def _one_kernel(kernel):
    if kernel is not True:
        pytest.skip("CatBoost models have a single kernel")


def make_frame(rng, n=N_ROWS, n_float=3, cards=(2, 8, 60)):
    data = {f"x{i}": rng.normal(size=n) for i in range(n_float)}
    for j, k in enumerate(cards):
        data[f"c{j}"] = rng.choice([f"v{j}_{i}" for i in range(k)] + [""], n)
    df = pd.DataFrame(data)
    df.loc[rng.random(n) < 0.05, "x1"] = np.nan
    signal = df.x0 + sum(df[f"c{j}"].str[-1].isin(list("137")) * (j + 1) * 0.5 for j in range(len(cards)))
    return df, signal.to_numpy()


def cat_names(df):
    return [c for c in df.columns if c.startswith("c")]


@pytest.fixture(params=["json", "cbm", "bin"])
def model_format(request):
    return request.param


def probe_rows(df, model_json):
    """Rows that exercise unseen categories, NaN and values exactly on float borders."""
    t = df.iloc[:1500].copy()
    cats = cat_names(df)
    t.loc[t.index[:100], cats[-1]] = "never seen"
    t.loc[t.index[100:150], cats[0]] = "unseen too"
    t.loc[t.index[150:200], "x0"] = np.nan
    for f in (model_json or {}).get("features_info", {}).get("float_features", []):
        borders = f.get("borders") or []
        if borders:
            col = df.columns[f["flat_feature_index"]]
            t.loc[t.index[200:200 + len(borders)], col] = np.asarray(borders[:len(t) - 200], dtype=np.float32)
    return t


def train(model_cls, df, y, tmp_path, fmt="json", **params):
    """Trains, saves as JSON, .cbm and lignum's .bin; returns the model, the path in `fmt` and the parsed JSON."""
    params = {"iterations": 40, "depth": 5, "verbose": 0, "allow_writing_files": False, "random_seed": 0, "thread_count": 2, **params}
    m = model_cls(cat_features=cat_names(df), **params)
    m.fit(df, y)
    m.save_model(str(tmp_path / "model.cbm"))
    try:
        m.save_model(str(tmp_path / "model.json"), format="json")
        j = json.loads((tmp_path / "model.json").read_text())
    except cb.CatBoostError:  # multiclass models have no JSON form
        j = None
    lignum.load(tmp_path / "model.cbm").save(str(tmp_path / "model.bin"))
    return m, tmp_path / f"model.{fmt}", j


@pytest.mark.parametrize("complexity", [1, 2, 4])
@pytest.mark.parametrize("one_hot", [2, 10])
@pytest.mark.parametrize("depth", [1, 4, 7])
def test_binary(tmp_path, complexity, one_hot, depth, model_format):
    rng = np.random.default_rng(complexity * 100 + one_hot * 10 + depth)
    df, signal = make_frame(rng)
    y = (signal + rng.normal(size=len(df)) > 0.5).astype(int)
    m, path, j = train(cb.CatBoostClassifier, df, y, tmp_path, model_format, depth=depth, max_ctr_complexity=complexity, one_hot_max_size=one_hot)
    t = probe_rows(df, j)
    model = lignum.load(path)
    assert model.is_catboost if hasattr(model, "is_catboost") else True
    np.testing.assert_array_equal(model.predict(t, raw_score=True), m.predict(t, prediction_type="RawFormulaVal"))
    np.testing.assert_array_equal(model.predict(t), m.predict_proba(t)[:, 1])


@pytest.mark.parametrize("loss", ["RMSE", "Poisson", "MAE"])
@pytest.mark.parametrize("ctrs", [["Borders"], ["Buckets"], ["Counter", "BinarizedTargetMeanValue"]])
def test_regression(tmp_path, loss, ctrs, model_format):
    rng = np.random.default_rng(len(loss) + len(ctrs))
    df, signal = make_frame(rng)
    y = np.exp(signal / 3) if loss == "Poisson" else signal * 10 + 3
    if loss == "Poisson":
        y = np.round(y)
    m, path, j = train(cb.CatBoostRegressor, df, y, tmp_path, model_format, loss_function=loss, simple_ctr=ctrs, combinations_ctr=ctrs)
    if loss == "RMSE":
        assert j["scale_and_bias"][1][0] != 0  # boost_from_average: a bias to add
    t = probe_rows(df, j)
    model = lignum.load(path)
    np.testing.assert_array_equal(model.predict(t, raw_score=True), m.predict(t, prediction_type="RawFormulaVal"))
    np.testing.assert_array_equal(model.predict(t), m.predict(t))


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_nan_modes_and_many_borders(tmp_path, nan_mode, model_format):
    rng = np.random.default_rng(7)
    df, _ = make_frame(rng, cards=(5,))
    df.loc[rng.random(len(df)) < 0.2, "x1"] = np.nan
    y = np.sin(6 * df.x0) + df.x0 + np.nan_to_num(df.x1, nan=2.0) + (df.c0 == "v0_1")
    # More than 255 borders in use needs 16-bit buckets.
    m, path, j = train(cb.CatBoostRegressor, df, y, tmp_path, model_format, nan_mode=nan_mode, border_count=1024, iterations=400, depth=6)
    assert max(len(f.get("borders") or []) for f in j["features_info"]["float_features"]) > 255
    t = probe_rows(df, j)
    np.testing.assert_array_equal(lignum.load(path).predict(t, raw_score=True), m.predict(t, prediction_type="RawFormulaVal"))


def test_input_forms_and_threads(tmp_path):
    rng = np.random.default_rng(3)
    df, signal = make_frame(rng)
    y = (signal > 0.5).astype(int)
    m, path, j = train(cb.CatBoostClassifier, df, y, tmp_path)
    model = lignum.load(path)
    t = probe_rows(df, j)
    want = m.predict_proba(t)[:, 1]

    np.testing.assert_array_equal(model.predict(t), want)
    as_category = t.astype({c: "category" for c in cat_names(t)})
    np.testing.assert_array_equal(model.predict(as_category), want)
    objects = t.to_numpy(dtype=object)
    np.testing.assert_array_equal(model.predict(objects), want)
    np.testing.assert_array_equal(model.predict(objects[0].tolist()), want[:1])  # one row as a list
    np.testing.assert_array_equal(model.predict(t.iloc[0]), want[:1])          # one row as a Series
    for n_jobs in [1, 3]:
        for strategy in ["auto", "serial", "rows"]:
            np.testing.assert_array_equal(model.predict(t, n_jobs=n_jobs, strategy=strategy), want)
    for i in range(20):
        np.testing.assert_array_equal(model.predict(t.iloc[i:i + 1]), want[i:i + 1])

    # Hand-made float64 matrix with hashes in the categorical columns.
    X = t.copy()
    for c in cat_names(t):
        X[c] = [lignum.catboost_hash(v) for v in t[c]]
    np.testing.assert_array_equal(model._c_model.predict(X.to_numpy(np.float64), False, -1, "auto"), want)
    assert model.categorical_features == [df.columns.get_loc(c) for c in cat_names(df)] if hasattr(model, "categorical_features") else True


def test_integer_categories(tmp_path):
    rng = np.random.default_rng(11)
    n = N_ROWS
    df = pd.DataFrame({"x0": rng.normal(size=n), "c0": rng.integers(-5, 40, n), "c1": rng.integers(0, 3, n)})
    y = ((df.c0 % 7 == 0) ^ (df.x0 > 0)).astype(int)
    m = cb.CatBoostClassifier(iterations=30, depth=4, verbose=0, allow_writing_files=False, cat_features=["c0", "c1"], one_hot_max_size=3)
    m.fit(df, y)
    path = tmp_path / "model.json"
    m.save_model(str(path), format="json")
    model = lignum.load(path)
    want = m.predict_proba(df)[:, 1]
    np.testing.assert_array_equal(model.predict(df), want)
    np.testing.assert_array_equal(model.predict(df.to_numpy(dtype=object)), want)  # integers hashed as text


def calc_hash(a, b):
    """CatBoost's CalcHash on uint64."""
    magic, mask = 0x4906BA494954CB65, (1 << 64) - 1
    return (magic * ((a + magic * b) & mask)) & mask


def test_hash_matches_catboost(tmp_path):
    """The CTR table of one categorical feature holds CalcHash(0, hash) of every training category:
    CatBoost's own hashes, for every CityHash length branch."""
    rng = np.random.default_rng(5)
    alphabet = list(string.ascii_letters + string.digits + " _-")
    values = ["", "a", "ab", "abc", "zażółć gęślą jaźń", "日本語"]
    values += ["".join(rng.choice(alphabet, size=L)) for L in list(range(4, 70)) + [100, 127, 128, 129, 200, 1000]]
    n = len(values) * 30
    df = pd.DataFrame({"x0": rng.normal(size=n), "c0": np.repeat(values, 30)})
    y = np.repeat(rng.integers(0, 2, len(values)), 30)  # the category decides: the CTR is used
    m = cb.CatBoostClassifier(iterations=10, depth=2, verbose=0, allow_writing_files=False, cat_features=["c0"], one_hot_max_size=1,
                              max_ctr_complexity=1, simple_ctr=["Borders"])
    m.fit(df, y)
    path = tmp_path / "model.json"
    m.save_model(str(path), format="json")
    (table,) = json.loads(path.read_text())["ctr_data"].values()
    stored = {int(h) for h in table["hash_map"][::table["hash_stride"]]} - {(1 << 64) - 1}
    ours = {calc_hash(0, int(np.int64(np.int32(np.uint32(lignum.catboost_hash(v))))) & ((1 << 64) - 1)) for v in values}
    assert ours == stored
    assert lignum.catboost_hash(42) == lignum.catboost_hash("42")


def test_errors(tmp_path):
    rng = np.random.default_rng(2)
    df, signal = make_frame(rng, cards=(4,))
    y = (signal > 0).astype(int)
    m, path, _ = train(cb.CatBoostClassifier, df, y, tmp_path)
    model = lignum.load(path)
    with pytest.raises(ValueError, match="strings or integers"):
        bad = df.astype({"c0": object})
        bad.loc[0, "c0"] = 1.5
        model.predict(bad)
    with pytest.raises(ValueError, match="missing values"):
        model.predict(df.astype({"c0": "category"}).assign(c0=lambda d: d.c0.cat.add_categories(["z"]).where(d.index > 0)))
    with pytest.raises(ValueError, match="not floats"):
        model.predict(np.zeros((2, df.shape[1])))
    with pytest.raises(RuntimeError, match="no compiled mode"):
        model.compile()

    lossguide = cb.CatBoostClassifier(iterations=5, depth=3, verbose=0, allow_writing_files=False, grow_policy="Lossguide", cat_features=["c0"])
    lossguide.fit(df, y)
    lossguide.save_model(str(tmp_path / "lossguide.json"), format="json")
    with pytest.raises(RuntimeError, match="symmetric"):
        lignum.load(tmp_path / "lossguide.json")


def test_single_leaf_trees(tmp_path):
    """A tree of depth 0 is written with "splits": null; it adds its one leaf to every row."""
    rng = np.random.default_rng(13)
    df, signal = make_frame(rng)
    m, path, j = train(cb.CatBoostClassifier, df, (signal > 0.5).astype(int), tmp_path)
    j["oblivious_trees"].append({"leaf_values": [0.25], "leaf_weights": [1], "splits": None})
    path.write_text(json.dumps(j))
    t = probe_rows(df, j)
    np.testing.assert_array_equal(lignum.load(path).predict(t, raw_score=True), m.predict(t, prediction_type="RawFormulaVal") + 0.25)


def test_arrow_input(tmp_path):
    pa = pytest.importorskip("pyarrow")
    rng = np.random.default_rng(17)
    n = N_ROWS
    df = pd.DataFrame({
        "x0": rng.normal(size=n),
        "c0": rng.choice(["a", "bb", "x" * 40, "y" * 100, ""], n),  # long strings leave the inline views
        "c1": rng.integers(-3, 30, n),
        "c2": rng.choice(["u", "v"], n),
    })
    y = ((df.x0 > 0) ^ (df.c0 == "a") ^ (df.c1 % 3 == 0)).astype(int)
    m = cb.CatBoostClassifier(iterations=40, depth=5, verbose=0, allow_writing_files=False, cat_features=["c0", "c1", "c2"], thread_count=2)
    m.fit(df, y)
    path = tmp_path / "model.json"
    m.save_model(str(path), format="json")
    model = lignum.load(path)
    want = m.predict_proba(df)[:, 1]

    table = pa.Table.from_pandas(df, preserve_index=False)  # large_string, int64
    np.testing.assert_array_equal(model.predict(table), want)
    narrow = table.cast(pa.schema([("x0", pa.float64()), ("c0", pa.string()), ("c1", pa.int32()), ("c2", pa.string())]))
    np.testing.assert_array_equal(model.predict(narrow), want)
    dictionaries = pa.Table.from_pandas(df.astype({"c0": "category", "c1": "category", "c2": "category"}), preserve_index=False)
    np.testing.assert_array_equal(model.predict(dictionaries), want)
    np.testing.assert_array_equal(model.predict(table.slice(7, 1)), want[7:8])
    np.testing.assert_array_equal(model.predict(table.slice(100, 500)), want[100:600])
    np.testing.assert_array_equal(model.predict(table.to_reader(max_chunksize=333)), want)
    with pytest.raises(ValueError, match="has nulls"):
        model.predict(pa.table({"x0": [1.0], "c0": pa.array([None], pa.string()), "c1": [1], "c2": ["u"]}))
    with pytest.raises(ValueError, match="strings, integers or a dictionary"):
        model.predict(pa.table({"x0": [1.0], "c0": [1.5], "c1": [1], "c2": ["u"]}))

    pl = pytest.importorskip("polars")
    frame = pl.from_pandas(df)  # string_view columns
    np.testing.assert_array_equal(model.predict(frame), want)
    categorical = frame.with_columns(pl.col("c0").cast(pl.Categorical), pl.col("c2").cast(pl.Categorical))
    np.testing.assert_array_equal(model.predict(categorical), want)


@pytest.mark.parametrize("loss", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("complexity", [1, 3])
@pytest.mark.parametrize("fmt", ["cbm", "bin"])
def test_multiclass(tmp_path, loss, complexity, fmt):
    """Only .cbm can hold multiclass models; CTRs over more than two classes count by target border."""
    rng = np.random.default_rng(complexity + len(loss))
    df, signal = make_frame(rng)
    y = np.digitize(signal + rng.normal(size=len(df)) * 0.5, [0.3, 1.0, 1.8])  # 4 classes
    m, path, _ = train(cb.CatBoostClassifier, df, y, tmp_path, fmt, loss_function=loss, max_ctr_complexity=complexity)
    t = probe_rows(df, None)
    model = lignum.load(path)
    assert model.n_outputs == 4
    np.testing.assert_array_equal(model.predict(t, raw_score=True), m.predict(t, prediction_type="RawFormulaVal"))
    np.testing.assert_array_equal(model.predict(t), m.predict_proba(t))
    for i in range(5):
        np.testing.assert_array_equal(model.predict(t.iloc[i:i + 1]), m.predict_proba(t.iloc[i:i + 1]))


@pytest.mark.parametrize("depth, n_classes", [(12, 1), (16, 1), (3, 12)])
def test_deep_trees_and_many_classes(tmp_path, depth, n_classes):
    """The kernel specialises depths 1-10 and 1-10 outputs; deeper trees and more classes take the
    general loops."""
    rng = np.random.default_rng(depth + n_classes)
    df, signal = make_frame(rng)
    if n_classes == 1:
        m, path, _ = train(cb.CatBoostRegressor, df, signal + rng.normal(size=len(df)), tmp_path, "cbm", depth=depth, iterations=15)
        want = m.predict(df)
    else:
        y = np.digitize(signal + rng.normal(size=len(df)) * 0.3, np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1]))
        m, path, _ = train(cb.CatBoostClassifier, df, y, tmp_path, "cbm", depth=depth, loss_function="MultiClass")
        want = m.predict(df, prediction_type="RawFormulaVal")
    model = lignum.load(path)
    assert model.n_outputs == n_classes
    np.testing.assert_array_equal(model.predict(df, raw_score=True), want)
    for i in range(5):
        np.testing.assert_array_equal(model.predict(df.iloc[i:i + 1], raw_score=True), want[i:i + 1])


def test_cbm_matches_json_and_loads_fast(tmp_path):
    rng = np.random.default_rng(23)
    df, signal = make_frame(rng)
    m, path, j = train(cb.CatBoostClassifier, df, (signal > 0.5).astype(int), tmp_path, "cbm", iterations=200, max_ctr_complexity=4)
    t = probe_rows(df, j)
    from_cbm, from_json = lignum.load(path), lignum.load(tmp_path / "model.json")
    np.testing.assert_array_equal(from_cbm.predict(t), from_json.predict(t))
    np.testing.assert_array_equal(from_cbm.predict(t), m.predict_proba(t)[:, 1])
    assert from_cbm.categorical_features == from_json.categorical_features
    with pytest.raises(ValueError, match="CatBoost JSON / .cbm"):
        lignum.load(path, format="lightgbm")
