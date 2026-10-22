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

Compiled libraries are cached by content (in the temp directory, or `work_dir`), so compiling the
same model again is instant.

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

| Option | Default | Meaning |
|---|---|---|
| `compiler` | `"auto"` | C compiler (see above) |
| `flags` | `""` | optimisation flags; empty means `-O2` or `/O2` |
| `work_dir` | temp dir | cache of compiled libraries |
| `jobs` | all cores | source files compiled in parallel |
| `block_trees` | 16 | trees per generated function; larger blocks compile slower and run about the same |
| `library` | none | also write the library to this path |
| `fallback` | `False` | keep the interpreter when compiling fails |
| `autotune` | none | sample to run `autotune` on after compiling |
