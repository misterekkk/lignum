# Security policy

lignum reads model files (LightGBM, XGBoost and CatBoost formats, and its own `.bin`) and, in the
compiled mode, loads native libraries. A model file can come from somewhere you do not fully
control, so the parsers are written to reject malformed input: every format is fuzzed in CI
(`tests/cpp/test_fuzz.cpp`, also under AddressSanitizer and UndefinedBehaviorSanitizer) and must fail
with an error, never crash, including on Windows' 1 MB stack.

## Supported versions

Fixes go into the latest release on PyPI (`pylignum`). Older versions do not get backports.

## Reporting a vulnerability

Please report security problems privately through GitHub:
**[Report a vulnerability](https://github.com/misterekkk/lignum/security/advisories/new)**
(the repository's Security tab). Do not open a public issue for them.

Useful details: the lignum version, the platform, and if possible a model file or a short script
that reproduces the problem. You should get an answer within a week. Once a fix is released, the
advisory is published with credit to the reporter, unless you prefer otherwise.

## Scope

In scope: crashes, memory errors or hangs caused by a model file (including `.bin`) or by input
data passed to `predict`, and wrong predictions caused by malformed input that was accepted.

Out of scope: compiled-mode libraries (the output of `compile()`, paths passed as `compiled=`) are
native code by design. Load them only from places you trust, as you would any shared library.
