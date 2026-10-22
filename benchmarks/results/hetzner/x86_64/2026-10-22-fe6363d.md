# Benchmark 2026-10-22T20:33:59

- lignum commit `fe6363d`, AMD EPYC-Milan Processor (x86_64), 4 threads, Python 3.12.3
- 1,000,000 rows × 20 features (5% NaN), 500 trees, max depth 6; binary classification, and 10-class softmax (50 iterations × 10 trees) for `*-multi`; `lightgbm-cat`: 5 categorical features (cardinality 10, 50, 200, 1000, 5000) + 15 numeric
- versions: lightgbm 4.7.0, xgboost 3.4.1, lleaves 1.3.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## lightgbm

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.038 | 2.157 | 2.168 | 5.032 | 7.7 | 7.7 | 267 | 507 | 0 | 0 | 0 | 19.3 |
| lignum-compiled | 8.762 | 1.226 | 1.245 | 3.899 | 4.0 | 4.0 | 255 | 459 | 0 | 0 | 0 | 10.2 |
| lleaves | 34.037 | 1.614 | 1.637 | 5.576 | 147 | 9.6 | 861 | 535 | 54,892 | 54,156 | 2.3e-15 | 24.4 |
| native | 0.000 | 10.623 | 10.687 | 34.531 | 63.1 | 42.1 | 1,062 | 3,474 | 0 | 0 | 0 | 74.0 |

## lightgbm-cat

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.062 | 3.549 | 3.549 | 11.226 | 8.3 | 8.3 | 460 | 1,105 | 0 | 0 | 0 | 15.6 |
| lignum-compiled | 6.838 | 1.360 | 1.368 | 4.761 | 3.3 | 3.3 | 218 | 437 | 0 | 0 | 0 | 6.2 |
| lleaves | 132.045 | 4.576 | 4.585 | 12.854 | 426 | 14.6 | 2,107 | 1,506 | 61,992 | 61,689 | 8.9e-16 | 39.5 |
| native | 0.000 | 14.039 | 14.056 | 50.851 | 64.7 | 46.1 | 1,439 | 5,077 | 0 | 0 | 0 | 83.4 |

## lightgbm-real-california

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.045 | 0.050 | 0.050 | 0.135 | 9.6 | 9.6 | 288 | 575 | 0 | 0 | 0 | 19.2 |
| lignum-compiled | 7.792 | 0.014 | 0.014 | 0.039 | 2.9 | 2.9 | 136 | 217 | 0 | 0 | 0 | 3.5 |
| lleaves | 43.370 | 0.021 | 0.022 | 0.058 | 163 | 9.8 | 784 | 359 | 3,821 | 15,284 | 1.6e-14 | 30.0 |
| native | 0.000 | 0.140 | 0.141 | 0.414 | 58.1 | 38.8 | 721 | 2,232 | 0 | 0 | 0 | 69.8 |

## lightgbm-real-covtype

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.041 | 1.187 | 1.188 | 2.819 | 8.3 | 8.3 | 251 | 455 | 0 | 0 | 0 | 15.2 |
| lignum-compiled | 5.855 | 0.265 | 0.266 | 0.698 | 2.9 | 2.9 | 101 | 151 | 0 | 0 | 0 | 3.6 |
| lleaves | 63.035 | 0.269 | 0.269 | 0.650 | 139 | 7.8 | 692 | 147 | 100,000 | 100,000 | 1.4e-14 | 19.0 |
| native | 0.000 | 2.401 | 2.417 | 5.941 | 48.4 | 32.0 | 409 | 955 | 0 | 0 | 0 | 67.0 |

## lightgbm-real-adult

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.061 | 0.111 | 0.111 | 0.353 | 9.3 | 9.3 | 430 | 1,085 | 0 | 0 | 0 | 18.6 |
| lignum-compiled | 6.798 | 0.026 | 0.026 | 0.085 | 3.2 | 3.2 | 170 | 312 | 0 | 0 | 0 | 3.8 |
| lleaves | 49.587 | 0.037 | 0.038 | 0.124 | 144 | 8.7 | 807 | 486 | 5,256 | 21,122 | 1.4e-15 | 21.8 |
| native | 0.000 | 0.294 | 0.299 | 0.942 | 58.5 | 38.3 | 956 | 2,979 | 0 | 0 | 0 | 69.5 |

## lightgbm-real-bank

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.038 | 0.125 | 0.126 | 0.357 | 8.8 | 8.8 | 277 | 736 | 0 | 0 | 0 | 17.0 |
| lignum-compiled | 7.261 | 0.035 | 0.036 | 0.077 | 3.0 | 3.0 | 119 | 176 | 0 | 0 | 0 | 3.8 |
| lleaves | 51.804 | 0.042 | 0.043 | 0.091 | 140 | 7.8 | 722 | 202 | 7,967 | 32,906 | 7.8e-16 | 20.9 |
| native | 0.000 | 0.343 | 0.345 | 0.832 | 57.0 | 36.9 | 696 | 1,611 | 0 | 0 | 0 | 65.4 |

## lightgbm-real-diabetes

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff | 1 row, 1 thr p99 [µs] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum | 0.082 | 0.484 | 0.486 | 1.397 | 11.5 | 11.5 | 617 | 1,355 | 0 | 0 | 0 | 21.4 |
| lignum-compiled | 6.927 | 0.151 | 0.152 | 0.499 | 3.1 | 3.1 | 241 | 503 | 0 | 0 | 0 | 10.2 |
| lleaves | 269.131 | 0.803 | 0.805 | 2.072 | 929 | 20.1 | 3,209 | 2,319 | 18,623 | 72,947 | 1.6e-15 | 55.7 |
| native | 0.000 | 1.830 | 1.831 | 5.917 | 73.5 | 50.6 | 1,848 | 5,847 | 0 | 0 | 0 | 84.4 |
