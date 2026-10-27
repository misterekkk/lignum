# Changelog

All notable changes to lignum (`pip install pylignum`). The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may change the API).

## [Unreleased]

## [0.3.0] — 2026-10-27

### Added

- **CatBoost models**, bit-identical to CatBoost: symmetric trees of any depth up to 16, float
  features (NaN as `Min` / `Max`), categorical features as one-hot or CTRs (`Borders`, `Buckets`,
  `Counter`, `BinarizedTargetMeanValue`) including combinations of categories with each other and
  with float features; binary, regression, `Poisson`, `Tweedie`, `MultiClass` and
  `MultiClassOneVsAll`. Reads `.cbm` (without the flatbuffers library) and JSON.
- Categorical input as strings or integers from pandas (object, string and category columns),
  numpy object arrays, lists, pyarrow and polars (string and dictionary columns). Large DataFrames
  go through pyarrow when it is installed.
- `lignum.catboost_hash(value)`, `lignum::catboost_hash` in C++ and `LignumCatBoostHash` in the
  C API, for callers that pass category hashes in a float64 matrix.
- The CatBoost kernel is built twice on x86-64 (generic and AVX2) and picks the AVX2 build by cpuid
  at run time; `LIGNUM_ISA=generic` forces the generic one. Linux, macOS and Windows (MSVC).
- Speed, against CatBoost 1.2.10's C API on one thread (500 trees of depth 6): 1.9-2.2x faster on
  models with categories, 1.1-2.4x on numeric models on x86-64 (AMD EPYC 9V45), 2.2-3.6x on ARM
  (Neoverse-N2) and 2.3-2.8x on an Apple M4; one row 7-14x faster with categories. Details on the
  [benchmarks page](https://misterekkk.github.io/lignum/benchmarks/).
- Security policy (`SECURITY.md`), contributing guide, citation file and this changelog.

### Changed

- **License: Apache License 2.0** (was MIT). Releases up to 0.2.0 remain available under MIT.
- The `.bin` format is now version 0.7 (CatBoost models). `.bin` files written by 0.2.0 are
  rejected with a clear error: export them again from the original model.
- A 1-D array or flat list is one row for every model.

### Fixed

- `predict` accepts read-only arrays (pandas 3.1 returns them from `np.asarray(DataFrame)`).

## [0.2.0] — 2026-10-23

First release on PyPI (`pip install pylignum`, `import lignum`).

### Added

- Bit-exact LightGBM and XGBoost semantics: thresholds, NaN and zero-as-missing, categorical
  splits, multiclass models and XGBoost's float32 arithmetic, checked against the real libraries
  in every test.
- Two engines: a binned interpreter for any batch size, and a compiled mode (`model.compile()`)
  that turns the trees into native code with the system C compiler in a few seconds. A compiled
  library ships next to the model and loads in about 2 ms without a compiler;
  `model.autotune(X)` picks the interpreter or the compiled code per batch size.
- Formats: LightGBM `model.txt` and JSON, XGBoost JSON and UBJSON, detected from the file content.
  `model.save("model.bin")` writes a binary file that loads by memory mapping in under a
  millisecond.
- Arrow input without copies: pyarrow tables and batches, polars data frames and any object with
  the Arrow PyCapsule interface; dictionary columns are categories.
- C API for other languages, with Arrow input and the compiled mode.
- Wheels for Linux x86_64 and aarch64 (manylinux_2_28), macOS 15+ (arm64, x86_64) and Windows x64;
  Python 3.10, 3.11 and abi3 for 3.12+.

### Not supported

- LightGBM linear trees and random forests, XGBoost multi-target / vector-leaf models: rejected at
  load time.

## [0.1.0] — 2026-06-10

Initial release of the package (GitHub only).

[Unreleased]: https://github.com/misterekkk/lignum/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/misterekkk/lignum/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/misterekkk/lignum/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/misterekkk/lignum/releases/tag/v0.1.0
