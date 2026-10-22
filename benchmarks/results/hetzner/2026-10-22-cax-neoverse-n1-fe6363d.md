# Benchmark 2026-10-22T21:24:45

- lignum commit `fe6363d`, Neoverse-N1 (aarch64), 4 threads, Python 3.12.3
- 1,000,000 rows × 20 features (5% NaN), 500 trees, max depth 6; binary classification, and 10-class softmax (50 iterations × 10 trees) for `*-multi`; `lightgbm-cat`: 5 categorical features (cardinality 10, 50, 200, 1000, 5000) + 15 numeric
- versions: lightgbm 4.7.0, xgboost 3.4.1, lleaves 1.3.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## lightgbm

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.045 | 1.942 | 1.949 | 7.673 | 11.1 | 11.1 | 264 | 780 | 0 | 0 | 0 | 24.4 |
| lignum-compiled | 11.316 | 1.034 | 1.037 | 3.946 | 8.9 | 8.8 | 271 | 481 | 0 | 0 | 0 | 25.8 |
| lleaves | 36.644 | 1.395 | 1.400 | 5.564 | 354 | 17.7 | 1,284 | 698 | 54,764 | 53,960 | 2.2e-15 | 72.1 |
| native | 0.000 | 19.828 | 20.761 | 54.918 | 165 | 159 | 2,248 | 5,946 | 0 | 0 | 0 | 268 |

## lightgbm-cat

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.079 | 3.684 | 3.685 | 14.316 | 13.0 | 12.9 | 511 | 1,500 | 0 | 0 | 0 | 25.6 |
| lignum-compiled | 10.349 | 1.191 | 1.204 | 4.789 | 7.2 | 7.2 | 237 | 492 | 0 | 0 | 0 | 20.8 |
| lleaves | 167.464 | 3.719 | 3.737 | 14.797 | 882 | 47.1 | 2,463 | 1,751 | 62,093 | 61,641 | 8.9e-16 | 166 |
| native | 0.000 | 33.036 | 33.525 | 89.050 | 194 | 189 | 3,679 | 9,034 | 0 | 0 | 0 | 327 |

## lightgbm-real-california

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.061 | 0.053 | 0.053 | 0.209 | 14.4 | 14.4 | 356 | 1,046 | 0 | 0 | 0 | 34.1 |
| lignum-compiled | 12.048 | 0.012 | 0.012 | 0.045 | 7.5 | 7.5 | 140 | 289 | 0 | 0 | 0 | 25.5 |
| lleaves | 49.552 | 0.020 | 0.022 | 0.075 | 405 | 25.2 | 1,179 | 561 | 3,821 | 15,284 | 1.6e-14 | 117 |
| native | 0.000 | 0.362 | 0.371 | 1.037 | 159 | 151 | 1,983 | 5,067 | 0 | 0 | 0 | 245 |

## lightgbm-real-covtype

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.065 | 1.256 | 1.257 | 4.551 | 13.0 | 13.0 | 271 | 810 | 0 | 0 | 0 | 28.7 |
| lignum-compiled | 10.462 | 0.245 | 0.245 | 0.938 | 7.1 | 7.1 | 75.1 | 188 | 0 | 0 | 0 | 21.9 |
| lleaves | 50.059 | 0.298 | 0.299 | 1.073 | 322 | 15.3 | 935 | 208 | 100,000 | 100,000 | 1.4e-14 | 67.9 |
| native | 0.000 | 6.602 | 6.732 | 16.213 | 136 | 130 | 1,382 | 3,076 | 0 | 0 | 0 | 229 |

## lightgbm-real-adult

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.070 | 0.127 | 0.129 | 0.485 | 14.4 | 14.5 | 566 | 1,515 | 0 | 0 | 0 | 40.0 |
| lignum-compiled | 10.964 | 0.024 | 0.024 | 0.097 | 7.4 | 7.4 | 175 | 361 | 0 | 0 | 0 | 22.5 |
| lleaves | 56.922 | 0.036 | 0.038 | 0.137 | 365 | 17.0 | 1,192 | 591 | 5,257 | 21,104 | 1.6e-15 | 70.8 |
| native | 0.000 | 0.707 | 0.744 | 1.918 | 175 | 168 | 2,611 | 6,181 | 0 | 0 | 0 | 297 |

## lightgbm-real-bank

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.075 | 0.138 | 0.139 | 0.506 | 13.5 | 13.5 | 377 | 1,113 | 0 | 0 | 0 | 35.7 |
| lignum-compiled | 10.368 | 0.033 | 0.034 | 0.101 | 7.1 | 7.1 | 109 | 249 | 0 | 0 | 0 | 19.3 |
| lleaves | 54.781 | 0.037 | 0.039 | 0.115 | 362 | 16.8 | 1,042 | 330 | 7,962 | 32,854 | 7.8e-16 | 78.9 |
| native | 0.000 | 0.981 | 1.028 | 2.513 | 171 | 163 | 2,145 | 5,462 | 0 | 0 | 0 | 274 |

## lightgbm-real-diabetes

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.124 | 0.522 | 0.525 | 1.969 | 17.8 | 17.8 | 781 | 1,982 | 0 | 0 | 0 | 44.4 |
| lignum-compiled | 9.177 | 0.143 | 0.145 | 0.590 | 7.0 | 7.0 | 270 | 542 | 0 | 0 | 0 | 21.9 |
| lleaves | 478.998 | 0.834 | 0.838 | 3.171 | 1,718 | 125 | 5,661 | 4,257 | 18,623 | 72,947 | 1.6e-15 | 247 |
| native | 0.000 | 4.406 | 4.442 | 13.893 | 233 | 212 | 4,777 | 12,775 | 0 | 0 | 0 | 317 |
