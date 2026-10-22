# Benchmark 2026-10-21T22:51:05

- lignum commit `c4e9965`, Apple M4, 10 threads, Python 3.12.15
- 1,000,000 rows × 20 features (5% NaN), 1000 trees, max depth 6; binary classification, and 10-class softmax (100 iterations × 10 trees) for `*-multi`
- versions: lignum 0.2.0.dev0, lightgbm 4.7.0, xgboost 3.4.1, treelite 4.7.2, tl2cgen 1.0.0, lleaves 1.3.0, onnxruntime 1.31.0, onnxmltools 1.16.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## lightgbm

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 5.472 | 5.576 | 32.918 | 51.2 | 41.2 | 669 | 3,292 | 0 | 0 | 0 |
| lignum | 0.033 | 0.713 | 0.721 | 4.915 | 8.6 | 8.6 | 161 | 499 | 0 | 0 | 0 |
| lignum-bin | 0.025 | 0.721 | 0.725 | 4.904 | 8.5 | 8.6 | 161 | 497 | 0 | 0 | 0 |
| tl2cgen | 6.004 | 2.201 | 2.219 | 15.492 | 14.8 | 14.0 | 289 | 1,231 | 0 | 0 | 0 |
| tl2cgen-tuned | 4.604 | 1.271 | 1.291 | 8.659 | 12.6 | 11.9 | 174 | 662 | 0 | 0 | 0 |
| gtil | 0.074 | 11.359 | 11.420 | 67.539 | 153 | 80.8 | 1,438 | 7,046 | 0 | 0 | 0 |
| lleaves | 27.550 | 0.841 | 0.858 | 5.905 | 37.6 | 8.3 | 310 | 345 | 53,721 | 52,488 | 4.1e-15 |
| onnxruntime | 1.098 | 4.503 | 4.840 | 21.993 | 10.2 | 15.9 | 326 | 2,070 | 100,000 | 100,000 | 6.9e-04 |

## lightgbm-multi

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 4.993 | 5.003 | 30.845 | 49.3 | 38.8 | 587 | 2,974 | 0 | 0 | 0 |
| lignum | 0.030 | 0.746 | 0.770 | 5.141 | 9.6 | 9.5 | 163 | 519 | 0 | 0 | 0 |
| lignum-bin | 0.027 | 0.752 | 0.764 | 5.137 | 9.5 | 9.7 | 163 | 518 | 0 | 0 | 0 |
| tl2cgen | 7.320 | 2.461 | 2.486 | 17.148 | 18.5 | 13.8 | 327 | 1,441 | 0 | 0 | 0 |
| tl2cgen-tuned | 5.321 | 1.804 | 1.813 | 12.742 | 13.5 | 13.0 | 248 | 1,023 | 0 | 0 | 0 |
| gtil | 0.077 | 11.789 | 11.968 | 68.335 | 155 | 77.5 | 1,521 | 7,144 | 100,000 | 100,000 | 6.0e-08 |
| lleaves | 25.450 | 0.655 | 0.677 | 4.350 | 38.8 | 9.0 | 327 | 381 | 100,000 | 100,000 | 1.4e-15 |
| onnxruntime | 1.128 | 8.659 | 8.745 | 23.921 | 11.4 | 20.7 | 480 | 2,410 | 100,000 | 100,000 | 7.7e-04 |

## xgboost-multi

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 1.377 | 1.409 | 7.998 | 148 | 75.9 | 651 | 956 | 0 | 0 | 0 |
| lignum | 0.021 | 0.769 | 0.778 | 5.159 | 9.2 | 9.3 | 163 | 521 | 0 | 0 | 0 |
| lignum-bin | 0.021 | 0.764 | 0.791 | 5.163 | 9.0 | 9.0 | 164 | 520 | 0 | 0 | 0 |
| tl2cgen | 4.171 | 3.755 | 3.783 | 24.372 | 19.5 | 19.5 | 525 | 2,310 | 96,237 | 96,050 | 3.0e-07 |
| tl2cgen-tuned | 5.558 | 1.706 | 1.719 | 11.630 | 13.5 | 12.1 | 232 | 927 | 96,237 | 96,051 | 6.8e-05 |
| gtil | 0.037 | 11.279 | 11.343 | 65.263 | 143 | 77.7 | 1,396 | 6,788 | 0 | 0 | 0 |
| lleaves | n/a (lleaves only supports LightGBM) | | | | | | | | | | | |
| onnxruntime | 0.421 | 8.654 | 8.835 | 22.641 | 18.2 | 18.0 | 433 | 2,179 | 99,995 | 99,999 | 1.0e-06 |

## xgboost

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 1.446 | 1.467 | 8.893 | 121 | 73.6 | 645 | 997 | 0 | 0 | 0 |
| lignum | 0.019 | 0.712 | 0.725 | 4.936 | 8.6 | 8.6 | 160 | 501 | 0 | 0 | 0 |
| lignum-bin | 0.018 | 0.718 | 0.726 | 4.942 | 8.6 | 8.6 | 160 | 501 | 0 | 0 | 0 |
| tl2cgen | 3.006 | 2.803 | 2.856 | 16.852 | 19.2 | 18.7 | 442 | 1,568 | 35,050 | 34,334 | 3.9e-07 |
| tl2cgen-tuned | 4.857 | 1.537 | 1.559 | 10.411 | 13.8 | 12.1 | 207 | 889 | 35,050 | 34,335 | 4.4e-06 |
| gtil | 0.033 | 11.265 | 12.099 | 65.454 | 149 | 73.0 | 1,442 | 6,760 | 35,561 | 35,138 | 3.0e-07 |
| lleaves | n/a (lleaves only supports LightGBM) | | | | | | | | | | | |
| onnxruntime | 0.361 | 4.242 | 4.522 | 20.458 | 8.7 | 16.4 | 302 | 1,886 | 57,026 | 56,061 | 1.7e-06 |
