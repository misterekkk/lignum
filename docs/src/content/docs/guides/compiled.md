---
title: Compiled mode
description: Turn the trees into native code, ship the library with the model, let autotune choose.
---

`model.compile()` turns the trees into C code, builds it with the system C compiler and loads the
result. Predictions stay bit-identical; on real data sets the compiled trees are typically 2-4x
faster than the interpreter, and faster than [lleaves](https://github.com/siboehm/lleaves) on most of
them (see [Benchmarks](../../benchmarks/)).

```python
model = lignum.load("model.txt")
stats = model.compile()      # a few seconds for 500 trees
preds = model.predict(X)     # now uses the compiled trees
```

## Where the library goes

Without `library=`, the library is kept in a per-user cache, so compiling the same model again (with
the same compiler and flags) only loads it. The code is built in a temporary folder that is always
removed; a failed build keeps only its log, whose path and last lines are in the error. Nothing else
is left behind.

```python
stats = model.compile()
stats["library"]                 # where it is, e.g. ~/.cache/lignum/lignum_3f..._linux-x86_64.so
model.compiled_library           # the same, while the model uses it
lignum.cache_dir()               # the cache folder
lignum.clear_cache()             # remove the cached libraries; returns how many
```

| Platform | Cache folder |
|---|---|
| Linux | `$XDG_CACHE_HOME/lignum`, or `~/.cache/lignum` |
| macOS | `~/Library/Caches/lignum` |
| Windows | `%LOCALAPPDATA%\lignum\Cache` |
| any | `LIGNUM_CACHE_DIR`, when set |

Libraries are keyed by the generated code, the compiler, the flags and the platform, so x86-64 and
arm64 builds (e.g. under Rosetta) never mix. Processes that compile the same model at once (e.g.
several workers starting together) are safe: each builds in its own folder and the library is moved
into place atomically.

## Ship the library, not the compiler

Compile once on a machine with a compiler and ship the library next to the binary model:

```python
# build machine
model = lignum.load("model.txt")
model.save("model.bin")
model.compile(library="model.so")

# production: attaching the library takes about 2 ms
model = lignum.load("model.bin", compiled="model.so")
```

`library=` writes the library exactly there and always compiles it; the cache is not used. It can
also be a folder (an existing one, or a path ending in `/`): the file is then named after the model
and the platform, `lignum_<fingerprint>_<platform>.so` (`.dll` on Windows), so one folder can hold the
libraries of several models, built on Linux, macOS and Windows. `load(..., compiled=folder)` picks
the file for this model and platform:

```python
model.compile(library="libs/")                     # on each platform you deploy to
model = lignum.load("model.bin", compiled="libs/")
```

The library records the fingerprint of the model it was compiled from and the version of lignum's
generated code. Loading it for another model or with an incompatible lignum raises `RuntimeError`,
so a stale library cannot give wrong predictions. Build it for the platform it runs on (Linux,
macOS or Windows, same CPU architecture); the code is not tuned for the build machine's CPU, so it
runs on any CPU of that architecture.

:::caution
Loading a compiled library runs its native code. Only load libraries you built or trust, as with
pickle files.
:::

## When there is no compiler

`compile()` raises `RuntimeError` naming the compiler it looked for. With `fallback=True` it warns
and keeps the interpreter instead:

```python
stats = model.compile(fallback=True)
if not stats["compiled"]:
    print("interpreter:", stats["error"])
```

`compiler="auto"` (the default) uses `cc` on Linux and macOS. On Windows it tries `cl` on PATH, then
the newest Visual Studio with C++ tools (found with vswhere), then MinGW `gcc`, then `clang`. Any
other compiler can be named explicitly, e.g. `compiler="clang-cl"` or a full path.

## Interpreter or compiled: autotune

Compiled trees are usually faster, but not always: on data whose splits are hard to predict, the
branch-free interpreter can win, mostly for small batches and on some CPUs. `autotune` measures both
on your data and uses each where it is faster:

```python
result = model.autotune(X_sample)          # about one second
result["compiled_min_rows"]                # batches from this size use the compiled trees
model.compile(library="model.so", autotune=X_sample)   # or both at once
```

The choice is stored in `model.compiled_min_rows`, which can also be set by hand; it does not
change any prediction. `model.decompile()` returns to the interpreter.

## Options

All options after `flags` are keyword-only.

| Option | Default | Meaning |
|---|---|---|
| `compiler` | `"auto"` | C compiler (see above) |
| `flags` | `""` | optimisation flags; empty means `-O2` or `/O2` |
| `jobs` | all cores | source files compiled in parallel |
| `block_trees` | 16 | trees per generated function; larger blocks compile slower and run about the same |
| `library` | none | write the library here (a file or a folder) instead of the cache |
| `fallback` | `False` | keep the interpreter when compiling fails |
| `autotune` | none | sample to run `autotune` on after compiling |
