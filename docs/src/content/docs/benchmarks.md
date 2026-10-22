---
title: Benchmarks
description: lignum against lleaves and the libraries themselves, on real data sets, x86-64 and ARM.
---

All numbers are for LightGBM models with 500 trees (`num_leaves=63`) trained on five public data
sets, predicting the whole data set. lignum is bit-exact on every one of them; lleaves, the fastest
other engine for LightGBM, is not (it differs from LightGBM on 81-100% of the rows). Lower is
better; the fastest engine of each row is in bold.

## Apple M4 (10 cores)

Milliseconds, best of 2 x 7 runs.

| Data set | rows | lignum interpreter | lignum compiled | lleaves |
|---|---:|---:|---:|---:|
| **1 thread** | | | | |
| California housing | 20,640 | 85.3 | **21.3** | 35.8 |
| Covertype (7 classes) | 581,012 | 1,679 | 410.2 | **388.0** |
| Adult | 32,561 | 226 | **45.0** | 55.7 |
| Bank marketing | 45,211 | 227 | **49.4** | 61.6 |
| Diabetes 130-US | 101,766 | 976 | **303.6** | 1,501 |
| **10 threads** | | | | |
| California housing | | 12.4 | **3.4** | 6.3 |
| Covertype | | 254 | **57.7** | 70.8 |
| Adult | | 32 | **7.2** | 9.8 |
| Bank marketing | | 33 | **8.3** | 10.5 |
| Diabetes 130-US | | 137 | **43.1** | 272 |

Compiling the trees took 6-12 s; lleaves took 13-134 s.

## x86-64 and ARM (GitHub runners, 4 vCPU)

Seconds, all threads / one thread. These are shared cloud machines: compare engines within a column,
not across runs.

| Data set | x86 interpreter | x86 compiled | x86 lleaves | ARM interpreter | ARM compiled | ARM lleaves |
|---|---:|---:|---:|---:|---:|---:|
| California | 0.056 / 0.153 | **0.017 / 0.044** | 0.029 / 0.069 | 0.033 / 0.135 | **0.008 / 0.033** | 0.015 / 0.057 |
| Covertype | 1.372 / 3.286 | **0.326 / 0.795** | 0.367 / **0.795** | 0.736 / 2.776 | **0.173 / 0.659** | 0.205 / 0.719 |
| Adult | 0.129 / 0.399 | **0.031 / 0.098** | 0.048 / 0.151 | 0.079 / 0.315 | **0.016 / 0.067** | 0.025 / 0.100 |
| Bank | 0.146 / 0.410 | **0.043 / 0.086** | 0.054 / 0.108 | 0.088 / 0.331 | **0.022 / 0.068** | 0.027 / 0.088 |
| Diabetes | 0.556 / 1.584 | **0.180 / 0.555** | 1.210 / 2.680 | 0.315 / 1.225 | **0.096 / 0.373** | 0.628 / 2.472 |

x86: AMD EPYC 7763; ARM: Neoverse-N2. A single row took 4.6-5.3 µs compiled on x86 (lleaves
13-50 µs) and 3.6-3.8 µs on ARM (lleaves 10-54 µs).

## Arrow input end to end

From a pyarrow Table to predictions on the M4, conversions included: lignum reads the table
directly; lleaves needs `table.to_pandas()`; LightGBM predicts the table itself (through pandas when
it has categorical columns). Seconds, all threads / one thread, and microseconds for 100 rows.

| Data set | lignum compiled | lleaves | LightGBM |
|---|---:|---:|---:|
| California | **0.004 / 0.022 / 74** | 0.008 / 0.037 / 211 | 0.070 / 0.449 / 2,069 |
| Covertype | **0.079** / 0.439 / **80** | 0.119 / **0.436** / 336 | 1.411 / 8.286 / 1,292 |
| Adult | **0.007 / 0.045 / 78** | 0.014 / 0.061 / 1,580 | 0.105 / 0.789 / 4,038 |
| Bank | **0.010 / 0.050 / 67** | 0.018 / 0.067 / 1,470 | 0.138 / 0.740 / 2,821 |
| Diabetes | **0.058 / 0.308 / 190** | 0.294 / 1.559 / 7,859 | 0.422 / 2.944 / 9,407 |

## Where lignum is not the fastest

- **Covertype on one thread**: lleaves is about 5% faster on the M4 and equal on x86. Covertype has 54
  features and 7 classes; binning the input costs time that lleaves does not spend.
- **Unpredictable data on x86**: on a synthetic model with independent random features, the
  interpreter beat the compiled trees on one x86 CPU (EPYC 9V45) for one thread and small batches.
  [`autotune`](../guides/compiled/#interpreter-or-compiled-autotune) measures this on your data.
- **Single rows from Python with Arrow**: pyarrow's export costs about 0.2 µs per column per call;
  pass a numpy row for the lowest latency.

## Reproduce

```bash
python benchmarks/bench.py --engines lignum,lignum-compiled,lleaves \
    --only lightgbm-real-covtype,lightgbm-real-diabetes --trees 500
python benchmarks/bench_arrow.py
```

The scripts train and cache the models, check every engine against the library and write Markdown
reports to `benchmarks/results/`. `.github/workflows/bench-lleaves.yml` runs them on x86-64 and ARM.
