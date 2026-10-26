<p align="center"><img src="docs/assets/logo.svg" width="112" alt="lignum logo"></p>

<h1 align="center">lignum</h1>

<p align="center">
  Fast inference for LightGBM, XGBoost and CatBoost models, with predictions <b>bit-identical</b> to the library that trained them.
</p>

<p align="center">
  <a href="https://pypi.org/project/pylignum/"><img src="https://img.shields.io/pypi/v/pylignum" alt="PyPI"></a>
  <a href="https://github.com/misterekkk/lignum/actions/workflows/tests.yml"><img src="https://github.com/misterekkk/lignum/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://misterekkk.github.io/lignum/"><img src="https://github.com/misterekkk/lignum/actions/workflows/docs.yml/badge.svg" alt="Documentation"></a>
</p>

```bash
pip install pylignum
```

```python
import lignum

model = lignum.load("model.txt")    # LightGBM model.txt / JSON, XGBoost .json / .ubj, CatBoost .json
preds = model.predict(X)            # numpy array, or Arrow data (pyarrow, polars, ...)
```

Installed as `pylignum`, imported as `lignum`. Wheels for Linux (x86_64, aarch64), macOS 15+ and
Windows; Python 3.10+.

## Why lignum

- **Exact.** Raw scores and probabilities equal `booster.predict(...)` bit for bit: NaN,
  zero-as-missing, categorical splits and XGBoost's float32 arithmetic included.
- **Fast.** An interpreter for any batch size, and a compiled mode that turns the trees into native
  code in a few seconds (`model.compile()`); `model.autotune(X)` picks the faster one per batch size.
- **Instant start.** `model.save("model.bin")` writes a binary file that loads by memory mapping in
  under a millisecond.
- **CatBoost included.** Categorical features, CTRs and their combinations, bit-exact; one row from a
  list in ~7 µs, CatBoost itself takes ~50 µs ([details](https://misterekkk.github.io/lignum/guides/catboost/)).
- **Arrow and C.** `predict` reads pyarrow, polars and any Arrow PyCapsule object without copies;
  a C API (`include/lignum/c_api.h`) serves other languages.

## Benchmarks

LightGBM models with 500 trees on public data sets, AMD EPYC-Milan (4 dedicated vCPU). Seconds for
the whole data set on all threads, and one row on one thread.

| | lignum compiled | lleaves | LightGBM |
|---|---:|---:|---:|
| California housing (20k rows) | **0.014 s** | 0.021 s | 0.140 s |
| Covertype (581k rows, 7 classes) | 0.265 s | 0.269 s | 2.401 s |
| Diabetes 130-US (102k rows, categorical) | **0.151 s** | 0.803 s | 1.830 s |
| One row, one thread (Covertype) | **2.9 µs** | 7.8 µs | 32.0 µs |
| Bit-identical to LightGBM | yes | no | — |

lleaves is the fastest other engine for LightGBM; it differs from LightGBM on 81-100% of the rows.
Covertype is a tie on all threads (within noise), and on one thread lleaves is faster (0.650 vs
0.698 s).
More data sets, ARM, Apple M4, Arrow input and the cases where lignum is not the fastest:
[benchmarks](https://misterekkk.github.io/lignum/benchmarks/).

## Documentation

**[misterekkk.github.io/lignum](https://misterekkk.github.io/lignum/)**: getting started, model
formats, the compiled mode, Arrow input, the C API and the Python reference.

## Development

```bash
make venv       # .venv with Python 3.12 via uv and the test dependencies
make py-build   # editable install
make test       # C++ (GoogleTest) + Python parity tests against LightGBM and XGBoost
```

macOS needs `brew install libomp`. The engine is C++20 (`src/`, public headers in
`include/lignum/`) with nanobind bindings in `python/`; [ARCHITECTURE.md](ARCHITECTURE.md) describes
the design and [CLAUDE.md](CLAUDE.md) the rules for changes.

## Acknowledgments

lignum parses JSON models and numbers with [simdjson](https://github.com/simdjson/simdjson).

## License

[MIT](LICENSE).
