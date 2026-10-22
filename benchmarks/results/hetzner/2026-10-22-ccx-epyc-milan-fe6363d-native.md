# Benchmark 2026-10-22T21:05:36

- lignum commit `fe6363d`, AMD EPYC-Milan Processor (x86_64), 4 threads, Python 3.12.3
- 1,000,000 rows × 20 features (5% NaN), 500 trees, max depth 6; binary classification, and 10-class softmax (50 iterations × 10 trees) for `*-multi`; `lightgbm-cat`: 5 categorical features (cardinality 10, 50, 200, 1000, 5000) + 15 numeric
- versions: lightgbm 4.7.0, xgboost 3.4.1, lleaves 1.3.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## lightgbm

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.042 | 2.094 | 2.095 | 4.884 | 7.2 | 7.2 | 261 | 490 | 0 | 0 | 0 | 13.4 |
| lignum-compiled | 6.883 | 1.126 | 1.129 | 3.568 | 3.7 | 3.6 | 238 | 423 | 0 | 0 | 0 | 9.4 |

## lightgbm-real-california

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.039 | 0.052 | 0.052 | 0.139 | 9.9 | 9.9 | 306 | 610 | 0 | 0 | 0 | 19.3 |
| lignum-compiled | 6.921 | 0.014 | 0.014 | 0.039 | 3.0 | 3.0 | 134 | 220 | 0 | 0 | 0 | 3.6 |

## lightgbm-real-covtype

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.040 | 1.229 | 1.231 | 2.958 | 8.9 | 8.9 | 260 | 482 | 0 | 0 | 0 | 18.4 |
| lignum-compiled | 5.502 | 0.260 | 0.261 | 0.687 | 2.7 | 2.7 | 98.9 | 149 | 0 | 0 | 0 | 3.2 |

## lightgbm-real-adult

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.034 | 0.113 | 0.113 | 0.358 | 10.0 | 10.0 | 443 | 1,101 | 0 | 0 | 0 | 19.2 |
| lignum-compiled | 6.588 | 0.026 | 0.026 | 0.084 | 2.9 | 2.9 | 174 | 311 | 0 | 0 | 0 | 3.3 |

## lightgbm-real-bank

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.035 | 0.128 | 0.128 | 0.365 | 9.3 | 9.3 | 292 | 747 | 0 | 0 | 0 | 18.4 |
| lignum-compiled | 6.257 | 0.035 | 0.036 | 0.076 | 2.8 | 2.8 | 121 | 177 | 0 | 0 | 0 | 3.3 |

## lightgbm-real-diabetes

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.055 | 0.485 | 0.485 | 1.417 | 11.8 | 11.8 | 621 | 1,367 | 0 | 0 | 0 | 21.2 |
| lignum-compiled | 6.160 | 0.149 | 0.150 | 0.478 | 3.2 | 3.2 | 248 | 494 | 0 | 0 | 0 | 3.7 |
