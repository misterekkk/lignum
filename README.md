<p align="center"><img src="docs/assets/logo.svg" width="112" alt="lignum logo"></p>

<h1 align="center">lignum</h1>

<p align="center">
  Fast inference for LightGBM, XGBoost and CatBoost models,<br>
  with predictions <b>bit-identical</b> to the library that trained them.
</p>

<p align="center">
  <a href="https://pypi.org/project/pylignum/"><img src="https://img.shields.io/pypi/v/pylignum" alt="PyPI"></a>
  <a href="https://pypi.org/project/pylignum/"><img src="https://img.shields.io/pypi/pyversions/pylignum" alt="Python versions"></a>
  <a href="https://github.com/misterekkk/lignum/actions/workflows/tests.yml"><img src="https://github.com/misterekkk/lignum/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/misterekkk/lignum/actions/workflows/wheels.yml"><img src="https://github.com/misterekkk/lignum/actions/workflows/wheels.yml/badge.svg" alt="Wheels"></a>
  <a href="https://misterekkk.github.io/lignum/"><img src="https://github.com/misterekkk/lignum/actions/workflows/docs.yml/badge.svg" alt="Documentation"></a>
  <br>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License: Apache 2.0"></a>
  <a href="https://pypi.org/project/pylignum/#files"><img src="https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Platforms: Linux, macOS, Windows"></a>
  <a href="https://pepy.tech/projects/pylignum"><img src="https://img.shields.io/pypi/dm/pylignum" alt="Downloads per month"></a>
  <a href="https://misterekkk.github.io/lignum/benchmarks/"><img src="https://img.shields.io/badge/predictions-bit--identical-0094a0" alt="Predictions: bit-identical"></a>
</p>

```bash
pip install pylignum
```

```python
import lignum

model = lignum.load("model.txt")    # LightGBM, XGBoost or CatBoost file, detected from its content
preds = model.predict(X)            # numpy, pandas, pyarrow, polars or a plain list
                                    # == booster.predict(X), bit for bit
```

Installed as `pylignum`, imported as `lignum`. Wheels for Linux (x86_64, aarch64), macOS 15+
(arm64, x86_64) and Windows (x64); Python 3.10+.

## Why lignum

- **Exact.** Raw scores and probabilities equal the library's `predict` bit for bit: NaN and
  zero-as-missing, categorical splits, XGBoost's float32 arithmetic and CatBoost's CTRs included.
  Named columns (pandas, Arrow) are checked against the model's feature names, so a reordered
  DataFrame raises instead of predicting wrong. What the data scientist saw in the notebook is what
  production returns.
- **Fast.** A binned interpreter for any batch size, and a compiled mode that turns LightGBM and
  XGBoost trees into native code in a few seconds for 500 trees (`model.compile()`);
  `model.autotune(X)` picks the faster one per batch size. One row of a 500-tree model takes
  10-30 µs on one x86 core, 6-21 µs on an Apple M4.
- **CatBoost with categories.** Categorical features, one-hot, CTRs and their combinations, from
  strings in pandas, Arrow or lists: 1.2-3x faster than CatBoost on batches, 7-27x on single rows.
- **Instant start.** `model.save("model.bin")` writes a file that loads by memory mapping in under
  a millisecond.
- **Arrow and C.** `predict` reads pyarrow, polars and any Arrow PyCapsule object without copies; a
  C API (`include/lignum/c_api.h`) serves other languages.
- **Small.** One C++20 library with no dependencies beyond the vendored simdjson and OpenMP.

## Benchmarks

The same models everywhere (500 rounds, 63 leaves; CatBoost depth 6), a different row on every call,
AMD EPYC-Milan (Hetzner CCX33, 8 vCPU). **LightGBM** models, 1,000,000 rows on all threads and one
row on one thread (median / p99):

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;lignum | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;lleaves | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TL2cgen | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LightGBM |
|---|---:|---:|---:|---:|
| California, 1M rows | **0.32 s** | 0.48 s | 1.96 s | 3.56 s |
| Covertype, 1M rows | **1.57 s** | 2.20 s | 34.5 s | 37.4 s |
| Diabetes, 1M rows | **0.75 s** | 4.23 s | 31.9 s | 9.33 s |
| California, 1 row (µs) | **11.0 / 23.7** | 24.6 / 56.4 | 49.2 / 87.8 | 57.7 / 92.9 |
| Diabetes, 1 row (µs) | **13.3 / 34.2** | 77.6 / 118 | 280 / 365 | 82.5 / 121 |
| Covertype, compile | **22 s** | 503 s | 70 s | — |
| Bit-identical | yes | no | yes | — |

Covertype has 7 classes (3,500 trees), Diabetes 130-US categorical features with hundreds of
values. lignum: the compiled mode for a million rows; for one row the faster of its two modes (the
interpreter on California, the compiled trees on Diabetes), which `model.autotune(X)` picks for you.
The interpreter needs no compiling and loads Covertype in 0.1 s. lleaves differs from LightGBM on
55-100% of the rows.

**XGBoost** models: 4-8x faster than XGBoost on a million rows and 5-15x on single rows (TL2cgen
cannot load XGBoost 3 models trained on pandas categories, and is not bit-identical on the others).
**CatBoost** models, from a pandas DataFrame: 1.2-3x faster on a million rows on one thread
(2.3-2.7x on an Apple M4) and 8-27x on single rows (216x with high-cardinality categories).

lignum is not the fastest everywhere: on an Apple M4 lleaves predicts one Covertype row in 38 µs
against lignum's 43 µs, and on x86 the compiled mode's p99 on single rows is worse than the interpreter's.
[All results](https://misterekkk.github.io/lignum/benchmarks/): every data set, Apple M4, Arrow
input, compile times, the methodology and how to reproduce them.

## Supported models

| Library | Files | Notes |
|---|---|---|
| LightGBM | `model.txt`, JSON | numeric and categorical splits, multiclass; not linear trees or random forests |
| XGBoost | JSON, UBJSON | float32 semantics, categorical splits, multiclass, DART; not multi-target models, gblinear or DART with dropped trees |
| CatBoost | `.cbm`, JSON | symmetric trees, one-hot and CTR categories, multiclass, Poisson / Tweedie; not Depthwise / Lossguide trees or text features |

Unsupported models are rejected at load time with a message that says why, never predicted wrongly.

## Documentation

**[misterekkk.github.io/lignum](https://misterekkk.github.io/lignum/)**: getting started, model
formats, CatBoost, the compiled mode, Arrow input, the C API, how exactness is guaranteed, and the
Python reference. Release notes are in [CHANGELOG.md](CHANGELOG.md).

## Used by

lignum is young. If you use it, in production, research or a side project, please add yourself to
[ADOPTERS.md](ADOPTERS.md) or tell us in an [issue](https://github.com/misterekkk/lignum/issues):
it helps others decide and shapes what comes next.

## Contributing

Bug reports, models that lignum predicts differently from their library, and benchmark results from
your hardware are all welcome. [CONTRIBUTING.md](CONTRIBUTING.md) explains the setup and the rules
every change keeps (bit-exactness, measured performance, tests on three platforms);
[ARCHITECTURE.md](ARCHITECTURE.md) describes the engine. Security problems: see
[SECURITY.md](SECURITY.md).

```bash
make venv && make py-build && make test   # C++ (GoogleTest) + Python parity tests
```

## Acknowledgments

lignum parses JSON models and numbers with [simdjson](https://github.com/simdjson/simdjson).

## License

[Apache License 2.0](LICENSE). Versions up to 0.2.0 were released under the MIT License. If you use
lignum in research, [CITATION.cff](CITATION.cff) has the citation.
