# libFuzzer targets, one per library (tests/fuzz/README.md). Needs Clang with libFuzzer (not Apple
# Clang: e.g. Homebrew's LLVM on macOS); the library itself is built with LIGNUM_SANITIZE and
# fuzzer-no-link (see the top-level CMakeLists.txt), so libFuzzer sees its coverage.
foreach(format lignum lightgbm xgboost catboost)
    add_executable(fuzz_${format} fuzz_load.cpp)
    target_compile_definitions(fuzz_${format} PRIVATE LIGNUM_FUZZ_FORMAT="${format}")
    target_compile_options(fuzz_${format} PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz_${format} PRIVATE -fsanitize=fuzzer)
    target_link_libraries(fuzz_${format} PRIVATE lignum)
endforeach()
