# Lignum — rules for working in this repo

Lignum is a production inference engine for LightGBM/XGBoost models (C++20 core, Python bindings).
Read [ARCHITECTURE.md](ARCHITECTURE.md) before touching the kernel or parsers; the roadmap lives in [TASKS.md](TASKS.md).

## Non-negotiable rules

1. **Predictions are bit-identical to the source library.** Raw scores and transformed outputs must
   equal `booster.predict(...)` exactly (`np.testing.assert_array_equal`, never `allclose`).
   - Sum trees in model order, continuing from `out_preds[i]`; never pre-sum a block of trees.
   - No `-ffast-math`, `-ffp-contract=fast`, `/fp:fast`.
   - Library-specific semantics (thresholds, missing values, float32) are resolved in the parser,
     not in the kernel. XGBoost uses float32 for comparisons, accumulation and transforms.
2. **No performance regressions.** Any kernel change needs a before/after benchmark against the
   previous commit (see "Benchmarking"). Put the numbers in the commit message.
3. **No hand-written SIMD** unless a benchmark shows a clear win on both x86-64 and arm64.
4. `KNode` is exactly 64 bytes (`static_assert`). Every union member must be naturally aligned per slot.
5. The core is a library: no Kafka/network code, no dependencies beyond vendored simdjson and OpenMP.
6. C++20. Code, comments, commit messages and repo docs in English.
7. Every feature ships with tests: GoogleTest in `tests/cpp` for logic, pytest parity tests in
   `tests/python` against real LightGBM/XGBoost models (include NaN, zeros, values exactly on thresholds).
   There are two kernels (binned by default, KNode walk for `binned=False` / unsupported models); every
   Python test runs with both (`tests/python/conftest.py`), and kernel changes must keep both exact.
8. Distributed builds never use `-march=native`/`-mcpu=native` (`LIGNUM_NATIVE` is for local benchmarks only).
9. Kernel ideas are measured on x86-64 and aarch64 before they are kept: put compile flags in
   `EXPERIMENT_CXX_FLAGS` of `.github/workflows/bench-lleaves.yml` and push; M4 alone has misled us
   (categorical side table: faster on servers, slower on M4).

## Layout

```
include/lignum/   public API (lignum.hpp, model.hpp)
src/              engine: parsers/, ir.hpp, builder, model.cpp (kernel), io.cpp (binary format)
python/           nanobind bindings (bindings.cpp) + lignum package
tests/cpp         GoogleTest      tests/python   pytest parity tests
benchmarks/       benchmark scripts (generated models/data are gitignored)
extern/simdjson   vendored
```

## Commands

```bash
make venv        # .venv with Python 3.12 via uv (lightgbm, xgboost, pytest, nanobind, ...)
make py-build    # editable install of the package into .venv
make test        # cpp-test + py-test
make cpp-test    # cmake -DLIGNUM_BUILD_TESTS=ON, build, ctest
```

macOS needs `brew install libomp`; CMake finds Homebrew's keg-only libomp automatically.

## Benchmarking

Compare against the previous state, not against memory:
1. `git worktree add <scratch>/base <commit>` and install it into a separate venv.
2. Run the same script on the same saved models and data with both venvs, at least twice.
3. Report best and median for 1M rows and the median latency for 100 rows.
