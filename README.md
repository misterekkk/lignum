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
  XGBoost trees into native code in a few seconds (`model.compile()`); `model.autotune(X)` picks
  the faster one per batch size. Single rows take a few microseconds.
- **CatBoost with categories.** Categorical features, one-hot, CTRs and their combinations, from
  strings in pandas, Arrow or lists: 2-3x faster than CatBoost on batches, 7-14x on single rows.
- **Instant start.** `model.save("model.bin")` writes a file that loads by memory mapping in under
  a millisecond.
- **Arrow and C.** `predict` reads pyarrow, polars and any Arrow PyCapsule object without copies; a
  C API (`include/lignum/c_api.h`) serves other languages.
- **Small.** One C++20 library with no dependencies beyond the vendored simdjson and OpenMP.

## Benchmarks

**LightGBM**, 500 trees on public data sets, AMD EPYC-Milan (4 dedicated vCPU). The whole data set
on all threads, and one row on one thread:

| | lignum compiled | lleaves | LightGBM |
|---|---:|---:|---:|
| California housing (20k rows) | **0.014 s** | 0.021 s | 0.140 s |
| Covertype (581k rows, 7 classes) | **0.265 s** | 0.269 s | 2.401 s |
| Diabetes 130-US (102k rows, categorical) | **0.151 s** | 0.803 s | 1.830 s |
| One row, one thread (Covertype) | **2.9 µs** | 7.8 µs | 32.0 µs |
| Bit-identical to LightGBM | yes | no (81-100% of rows differ) | — |

**CatBoost**, 500 trees of depth 6, CatBoost 1.2.10 / lignum through the C API, one thread,
300k rows:

| | x86-64 (AMD EPYC 9V45) | ARM (Neoverse-N2) |
|---|---:|---:|
| Adult (8 categorical columns) | 0.424 / **0.213 s** | 0.941 / **0.339 s** |
| Diabetes 130-US (categorical, large CTR tables) | 1.624 / **0.735 s** | 2.926 / **1.209 s** |
| Covertype (numeric, 7 classes) | 0.287 / **0.119 s** | 0.818 / **0.265 s** |
| California housing (numeric) | 0.071 / **0.061 s** | 0.318 / **0.088 s** |
| One row, Adult | 20.1 / **2.3 µs** | 50.7 / **4.9 µs** |

lignum is not the fastest everywhere: lleaves wins Covertype on one thread by about 7%, and on x86
CatBoost comes within 10-15% of lignum on some purely numeric models (California above).
[All results](https://misterekkk.github.io/lignum/benchmarks/): more data sets, Apple M4, Windows,
Arrow input, and how to reproduce them.

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
