# Fuzzing

`fuzz_load.cpp` is a [libFuzzer](https://llvm.org/docs/LibFuzzer.html) target built once per
library (`fuzz_lignum` for `.bin` files, `fuzz_lightgbm`, `fuzz_xgboost`, `fuzz_catboost`), with
AddressSanitizer and UndefinedBehaviorSanitizer. libFuzzer mutates model files guided by the code
they reach; every input must be rejected or load and predict safely, a loaded model must predict
the same with the binned and the direct kernel, and the same after a round trip through `.bin`.
`tests/cpp/test_fuzz.cpp` (random mutations, part of the GoogleTest suite) stays as a quick check;
`.github/workflows/fuzz.yml` runs these targets on every push and every night.

```bash
# Clang with libFuzzer: on Linux the distribution's clang (+ libclang-rt-dev), on macOS Homebrew's
# LLVM (Apple Clang has no libFuzzer).
cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DLIGNUM_SANITIZE=ON "-DLIGNUM_SANITIZERS=address,undefined,fuzzer-no-link" -DLIGNUM_BUILD_FUZZERS=ON \
      -DLIGNUM_BUILD_C_LIBRARY=OFF
cmake --build build-fuzz --parallel
python tests/fuzz/make_corpus.py seeds           # small models of every format (needs ".[test]")

mkdir -p corpus/xgboost && cp seeds/xgboost/* corpus/xgboost/
ASAN_OPTIONS=detect_leaks=0 ./build-fuzz/tests/fuzz/fuzz_xgboost corpus/xgboost -max_total_time=600 -timeout=10 -rss_limit_mb=2048
```

A crash writes the input to `crash-<hash>` (CI: the `fuzz-crashes-<target>` artifact). Reproduce
it by passing the file instead of the corpus folder, e.g. `./build-fuzz/tests/fuzz/fuzz_xgboost
crash-<hash>`, then add a regression test with the smallest input that still fails to
`tests/cpp` (libFuzzer's `-minimize_crash=1` shrinks it).

Found so far: a use after free when one XGBoost `base_score` is copied to every class
(`tests/cpp/test_xgboost.cpp`, `OneBaseScoreForEveryClass`).
