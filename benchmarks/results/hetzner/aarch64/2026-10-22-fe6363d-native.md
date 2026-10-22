# Benchmark 2026-10-22T22:14:20

- lignum commit `fe6363d`, Neoverse-N1 (aarch64), 4 threads, Python 3.12.3
- 1,000,000 rows × 20 features (5% NaN), 500 trees, max depth 6; binary classification, and 10-class softmax (50 iterations × 10 trees) for `*-multi`; `lightgbm-cat`: 5 categorical features (cardinality 10, 50, 200, 1000, 5000) + 15 numeric
- versions: lightgbm 4.7.0, xgboost 3.4.1, lleaves 1.3.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## lightgbm

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.049 | 1.938 | 1.944 | 7.529 | 10.7 | 10.7 | 277 | 783 | 0 | 0 | 0 | 21.7 |
| lignum-compiled | 11.294 | 1.031 | 1.034 | 3.943 | 8.8 | 8.8 | 244 | 488 | 0 | 0 | 0 | 21.7 |

## lightgbm-real-california

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.052 | 0.052 | 0.052 | 0.205 | 14.0 | 14.0 | 353 | 1,022 | 0 | 0 | 0 | 30.8 |
| lignum-compiled | 12.187 | 0.012 | 0.012 | 0.046 | 7.4 | 7.4 | 151 | 303 | 0 | 0 | 0 | 19.4 |

## lightgbm-real-covtype

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.049 | 1.219 | 1.224 | 4.505 | 12.6 | 12.6 | 276 | 793 | 0 | 0 | 0 | 23.2 |
| lignum-compiled | 10.016 | 0.247 | 0.247 | 0.942 | 7.0 | 7.0 | 79.1 | 193 | 0 | 0 | 0 | 16.6 |

## lightgbm-real-adult

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.048 | 0.126 | 0.127 | 0.474 | 14.1 | 13.9 | 563 | 1,506 | 0 | 0 | 0 | 30.3 |
| lignum-compiled | 11.351 | 0.024 | 0.025 | 0.097 | 7.2 | 7.2 | 201 | 358 | 0 | 0 | 0 | 33.6 |

## lightgbm-real-bank

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.047 | 0.135 | 0.137 | 0.494 | 13.2 | 13.1 | 378 | 1,089 | 0 | 0 | 0 | 27.1 |
| lignum-compiled | 10.564 | 0.033 | 0.033 | 0.101 | 7.0 | 7.0 | 97.5 | 244 | 0 | 0 | 0 | 22.4 |

## lightgbm-real-diabetes

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.084 | 0.523 | 0.527 | 1.979 | 17.3 | 17.3 | 757 | 1,930 | 0 | 0 | 0 | 43.1 |
| lignum-compiled | 9.664 | 0.145 | 0.147 | 0.585 | 6.9 | 6.9 | 280 | 529 | 0 | 0 | 0 | 18.5 |
