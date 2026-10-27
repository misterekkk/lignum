---
title: Release notes
description: What changed in each lignum version.
---

<!-- Generated from CHANGELOG.md by docs/scripts/release_notes.py: edit CHANGELOG.md. -->

Versions follow [Semantic Versioning](https://semver.org/); before 1.0, minor versions may
change the API. Install a specific version with `pip install pylignum==<version>`.

## [0.4.0] — 2026-10-27

### Changed

- **Breaking: column names are checked.** When a model was trained on named columns and the input
  has column names (pandas DataFrame, pyarrow, polars or other Arrow data), `predict` checks that
  the columns start with the model's `feature_names`, in the same order, and raises `ValueError`
  otherwise. Before, a reordered or missing column silently gave other predictions. numpy arrays,
  frames without column names, models without names and the C API are unchanged (by position).
  `predict(..., validate_features=False)`, `load(..., validate_features=False)` or
  `model.validate_features = False` turn the check off; it costs 0-5 µs per call (Apple M4).
- **Breaking: `.bin` format 0.8.** It adds the feature names, the library the model was trained
  with, its objective and the lignum version that wrote the file. Files written by lignum 0.3 are rejected: save
  them again from the original model.

- **Compiled-library files** (`Model.compile()`): without `library=`, libraries go to a per-user
  cache (`~/.cache/lignum`, `~/Library/Caches/lignum`, `%LOCALAPPDATA%\lignum\Cache`, or
  `LIGNUM_CACHE_DIR`) instead of `lignum_compiled` in the temp directory, and the code is built in a
  temporary folder that is always removed: no `.c`, `.o` or log files are left behind (a failed
  build keeps only its log, named in the error together with the compiler's last lines).
  `library=` writes the library exactly there, always compiling it and bypassing the cache.
- **Breaking:** `compile()` has no `work_dir` any more (set `LIGNUM_CACHE_DIR` to move the cache),
  and its options after `flags` are keyword-only. In the C API the fourth argument of
  `LignumModelCompile` is now the cache folder (`NULL`: the per-user cache).

### Added

- Model inspection: `Model.n_trees`, `Model.source` (`"lightgbm"`, `"xgboost"` or `"catboost"`),
  `Model.objective` (the library's name, e.g. `"binary"`, `"multi:softprob"`, `"Logloss"`) and
  `Model.info()`, a dict that adds the output transform, the number of leaves, the kernel in use,
  the compiled library and, for `.bin` files, the lignum version that wrote them. `repr(model)`
  shows the source, objective, trees and features. C API: `LignumModelNumTrees`.
- `Model.feature_names`: the names of the training columns (LightGBM, XGBoost and CatBoost files,
  kept in `.bin`), or `None` for models trained on a matrix.
- `lignum.cache_dir()`, `lignum.clear_cache()` and `Model.compiled_library` (the path of the
  library in use).
- `compile(library=folder)` writes `lignum_<model fingerprint>_<platform>.so` (`.dll` on Windows)
  into the folder, and `load(..., compiled=folder)` picks the file for the model and platform, so one
  folder can hold the libraries of several models and platforms.

### Fixed

- Processes compiling the same model at once no longer write the same files: each builds in its own
  folder and the library is moved into place with an atomic rename.
- The cache key includes the CPU architecture: x86-64 (e.g. under Rosetta) and arm64 processes no
  longer load each other's libraries. The key is also stable across compilers (FNV-1a instead of
  `std::hash`), and a cached library that fails to load is rebuilt.
- Errors when loading a library built for another platform name the platform that is needed.
- XGBoost models that were predicted wrongly without an error are now rejected: linear boosters
  (`booster="gblinear"`, which loaded as an empty forest predicting `base_score`) and DART models
  that dropped trees (`rate_drop > 0`: XGBoost weights each tree, lignum ignored the weights). DART
  files from XGBoost before 3.4 (trees nested under `gbtree`) used to load as an empty forest too;
  they now predict correctly when no tree was dropped. LightGBM DART models were and are exact.

The old `lignum_compiled` folder in the temp directory is no longer used and can be deleted.

## [0.3.1] — 2026-10-27

### Fixed

- **License notices**: `NOTICE` now carries the notices of the third-party code lignum derives
  from, as their licenses require for source and binary copies: CityHash (Copyright (c) 2011
  Google, Inc., MIT License; CatBoost's category hash) and fmath by herumi (BSD 3-Clause; the
  `exp` CatBoost uses on x86). The 0.3.0 wheels lacked them. No change to predictions or the API.
- Two GCC warnings (`-Wall -Wextra`): an unchecked simdjson result for an optional field of CatBoost
  JSON models, and an unused variable on ARM.
- The CatBoost benchmark no longer asks CatBoost to read polars 2.x Categorical columns, which it
  does not support (on Windows CatBoost could crash on them); lignum was not affected.

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

[Unreleased]: https://github.com/misterekkk/lignum/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/misterekkk/lignum/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/misterekkk/lignum/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/misterekkk/lignum/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/misterekkk/lignum/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/misterekkk/lignum/releases/tag/v0.1.0
