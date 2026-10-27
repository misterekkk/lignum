# Contributing to lignum

Thanks for your interest. Bug reports, models that lignum predicts differently from the library
that trained them, and benchmark results from your hardware are all very welcome as
[issues](https://github.com/misterekkk/lignum/issues). Security problems go through the private
channel described in [SECURITY.md](SECURITY.md).

## Setting up

```bash
make venv       # .venv with Python 3.12 via uv and the test dependencies
make py-build   # editable install (run again after every C++ change)
make test       # C++ (GoogleTest) + Python parity tests
```

macOS needs `brew install libomp`. [ARCHITECTURE.md](ARCHITECTURE.md) describes the engine.

## The rules every change keeps

1. **Predictions are bit-identical to the source library.** Tests compare with
   `np.testing.assert_array_equal`, never `allclose`. Trees are summed in model order; no
   `-ffast-math` or FMA contraction. Library-specific semantics (thresholds, missing values,
   XGBoost's float32) belong in the parsers, not in the kernels.
2. **No performance regressions.** A kernel change comes with a before / after benchmark against
   the previous commit (best and median for 1M rows, median latency for 100 rows), and the numbers
   go into the commit message. Kernel decisions are measured on x86-64 and arm64.
3. **Tests with every feature.** GoogleTest in `tests/cpp` for logic, pytest parity tests in
   `tests/python` against real LightGBM / XGBoost / CatBoost models, with NaN, zeros and values
   exactly on thresholds. Every Python test runs in three kernels (binned, KNode walk, compiled);
   all three must stay exact. Parser changes also get fuzz tests in `tests/cpp/test_fuzz.cpp`.
4. **Portable.** Linux, macOS and Windows (MSVC) are all tested in CI; no POSIX-only code without a
   Windows path. C++20, no new dependencies beyond the vendored simdjson and OpenMP.

Code, comments and commit messages are in English. Commit messages say what changed and why, with
measurements for performance work.

## Pull requests

Open an issue first for larger changes, so we can agree on the approach. Keep pull requests
focused, make sure `make test` passes, and describe how you tested the change.

## License

lignum is licensed under the [Apache License 2.0](LICENSE). By submitting a contribution you agree
that it is licensed under the same terms (section 5 of the license).

## Using lignum?

If you use lignum in a project or in production, we would love to hear about it: open a pull
request adding yourself to [ADOPTERS.md](ADOPTERS.md).
