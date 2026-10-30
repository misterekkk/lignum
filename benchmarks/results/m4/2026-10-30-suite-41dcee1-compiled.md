# Benchmark suite 2026-10-30T08:48:15

- lignum commit `41dcee1`, Apple M4, 10 threads, Python 3.12.15
- versions: lightgbm 4.7.0, xgboost 3.4.1, treelite 4.7.2, tl2cgen 1.0.0, lleaves 1.3.0, onnxruntime 1.31.0, onnxmltools 1.16.0, catboost 1.2.10
- 500 rounds; LightGBM num_leaves 63; XGBoost lossguide max_leaves 63; CatBoost depth 6
- latency: one call, rows differ on every call (p50 / p99 / p99.9, µs); throughput: median of repeats on different windows (s); prep: load + compile, cold
- exact: rows not bit-identical to the library's predict (held-out / training rows)
- input: `matrix` = float64 matrix (LightGBM, XGBoost) or DataFrame (CatBoost); `arrow` = pyarrow Table, converted the way each engine needs (to_pandas for dictionary columns, CatBoost and lleaves)

## lightgbm / california

500 trees, average path 5.56 comparisons per tree, model sha256 `dd35e5214a16`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.69 | yes | 6.3 / 13.5 / 18.8 | 19.1 / 24.0 / 25.2 | 112 / 140 / 216 | 0.0095 | 0.0921 | 0.927 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.69 | yes | 6.3 / 7.6 / 8.1 | 28.4 / 44.7 / 53.5 | 75.9 / 98.6 / 104 | 0.0017 | 0.0175 | 0.156 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.99 | yes | 9.5 / 14.8 / 19.6 | 22.6 / 28.4 / 29.9 | 123 / 149 / 152 | 0.0103 | 0.0981 | 0.974 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.99 | yes | 9.2 / 10.6 / 11.0 | 30.5 / 48.0 / 51.9 | 76.5 / 97.5 / 140 | 0.0018 | 0.0179 | 0.158 |

## lightgbm / covtype

3500 trees, average path 5.92 comparisons per tree, model sha256 `004b5744d66c`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 12.71 | yes | 45.6 / 63.5 / 67.7 | 102 / 188 / 217 | 591 / 1,056 / 1,173 | 0.0628 | 0.468 | 5.162 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 12.71 | yes | 42.7 / 62.4 / 216 | 82.0 / 130 / 141 | 261 / 376 / 468 | 0.0115 | 0.0777 | 0.768 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 13.15 | yes | 57.3 / 75.3 / 79.6 | 117 / 204 / 235 | 617 / 1,098 / 1,230 | 0.0636 | 0.465 | 5.190 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 13.15 | yes | 56.2 / 74.4 / 78.4 | 95.0 / 145 / 153 | 284 / 400 / 486 | 0.0112 | 0.0767 | 0.768 |

## lightgbm / adult

500 trees, average path 4.97 comparisons per tree, model sha256 `5631a9e6bdaf`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.96 | yes | 7.3 / 10.5 / 13.2 | 22.8 / 26.0 / 27.3 | 154 / 165 / 173 | 0.0138 | 0.137 | 1.373 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.96 | yes | 7.1 / 8.3 / 9.0 | 28.2 / 43.2 / 47.3 | 82.7 / 97.3 / 103 | 0.0024 | 0.0233 | 0.205 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.70 | yes | 12.7 / 15.1 / 17.3 | 28.5 / 32.0 / 33.7 | 160 / 172 / 178 | 0.0139 | 0.138 | 1.380 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.70 | yes | 12.6 / 14.1 / 15.6 | 33.1 / 51.0 / 55.0 | 89.0 / 105 / 113 | 0.0025 | 0.0234 | 0.207 |

## lightgbm / bank

500 trees, average path 4.50 comparisons per tree, model sha256 `ccd3a479527c`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.82 | yes | 7.2 / 10.3 / 12.4 | 22.3 / 28.3 / 29.6 | 128 / 166 / 179 | 0.0103 | 0.107 | 1.075 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.82 | yes | 7.0 / 8.1 / 8.9 | 28.2 / 43.3 / 47.1 | 77.1 / 98.8 / 105 | 0.0018 | 0.0184 | 0.173 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.81 | yes | 13.2 / 15.5 / 17.7 | 28.2 / 34.3 / 35.7 | 136 / 174 / 189 | 0.0104 | 0.108 | 1.082 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.81 | yes | 13.0 / 14.3 / 15.9 | 33.4 / 51.4 / 55.8 | 84.3 / 106 / 114 | 0.0018 | 0.0182 | 0.172 |

## lightgbm / diabetes

500 trees, average path 7.24 comparisons per tree, model sha256 `7cdfbbd121dd`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.82 | yes | 7.4 / 8.5 / 9.5 | 34.7 / 37.4 / 38.5 | 300 / 314 / 320 | 0.0297 | 0.297 | 2.972 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.82 | yes | 7.4 / 8.5 / 8.7 | 31.4 / 45.3 / 53.3 | 122 / 132 / 172 | 0.0068 | 0.0570 | 0.452 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.31 | yes | 25.8 / 33.7 / 85.9 | 53.3 / 57.0 / 106 | 319 / 341 / 558 | 0.0294 | 0.294 | 2.938 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.31 | yes | 25.1 / 26.6 / 29.9 | 52.4 / 60.2 / 68.0 | 145 / 165 / 266 | 0.0047 | 0.0459 | 0.444 |

## lightgbm / synthetic

500 trees, average path 4.36 comparisons per tree, model sha256 `e8d808ec6ae3`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.24 | yes | 6.7 / 8.0 / 9.0 | 22.7 / 25.3 / 26.0 | 163 / 322 / 782 | 0.0145 | 0.150 | 1.432 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.24 | yes | 6.7 / 15.6 / 53.7 | 28.2 / 49.7 / 102 | 81.8 / 109 / 141 | 0.0025 | 0.0256 | 0.235 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.75 | yes | 11.6 / 13.2 / 14.7 | 28.3 / 31.3 / 32.8 | 171 / 180 / 189 | 0.0148 | 0.143 | 1.426 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.75 | yes | 11.4 / 12.9 / 14.2 | 32.1 / 49.9 / 54.2 | 90.1 / 105 / 112 | 0.0025 | 0.0264 | 0.241 |

## lightgbm / synthetic-cat

500 trees, average path 9.82 comparisons per tree, model sha256 `4d440320fee5`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.76 | yes | 9.7 / 10.6 / 12.0 | 43.3 / 45.8 / 47.8 | 379 / 390 / 396 | 0.0370 | 0.370 | 3.703 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.76 | yes | 9.5 / 10.4 / 11.9 | 34.8 / 51.0 / 56.8 | 139 / 149 / 153 | 0.0059 | 0.0551 | 0.537 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.02 | yes | 15.9 / 17.0 / 19.0 | 50.8 / 53.6 / 55.9 | 390 / 403 / 422 | 0.0375 | 0.375 | 3.754 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.02 | yes | 15.9 / 17.0 / 19.0 | 41.7 / 45.2 / 55.9 | 149 / 158 / 162 | 0.0059 | 0.0539 | 0.529 |

## xgboost / california

500 trees, average path 3.69 comparisons per tree, model sha256 `2e1e5f92d013`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.93 | yes | 5.4 / 12.1 / 16.9 | 17.4 / 21.9 / 33.4 | 98.8 / 118 / 169 | 0.0083 | 0.0801 | 0.792 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.93 | yes | 5.2 / 6.3 / 6.8 | 28.4 / 44.0 / 48.0 | 63.7 / 77.9 / 82.9 | 0.0014 | 0.0138 | 0.124 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.95 | yes | 8.1 / 13.4 / 18.5 | 20.5 / 24.6 / 26.7 | 103 / 120 / 124 | 0.0085 | 0.0812 | 0.805 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.95 | yes | 8.0 / 9.1 / 9.6 | 28.6 / 46.5 / 50.5 | 67.3 / 84.3 / 110 | 0.0014 | 0.0140 | 0.129 |

## xgboost / covtype

3500 trees, average path 4.09 comparisons per tree, model sha256 `2a8e167914f6`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 11.53 | yes | 41.2 / 55.7 / 60.0 | 97.4 / 178 / 203 | 554 / 1,077 / 1,138 | 0.0584 | 0.445 | 4.927 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 11.53 | yes | 40.4 / 56.8 / 93.0 | 77.5 / 118 / 133 | 248 / 423 / 634 | 0.0104 | 0.0748 | 0.732 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 11.27 | yes | 54.2 / 71.6 / 109 | 113 / 197 / 226 | 579 / 1,107 / 1,188 | 0.0586 | 0.447 | 4.945 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 11.27 | yes | 53.2 / 67.9 / 72.2 | 91.6 / 132 / 147 | 263 / 368 / 447 | 0.0105 | 0.0758 | 0.734 |

## xgboost / adult

500 trees, average path 6.10 comparisons per tree, model sha256 `a9cf063657e7`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.81 | yes | 6.4 / 9.1 / 10.8 | 24.0 / 27.9 / 29.3 | 179 / 191 / 196 | 0.0182 | 0.183 | 1.831 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.81 | yes | 6.2 / 7.5 / 8.1 | 28.3 / 43.5 / 48.0 | 90.7 / 104 / 110 | 0.0030 | 0.0305 | 0.295 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.84 | yes | 11.9 / 14.1 / 16.3 | 29.7 / 33.6 / 35.1 | 185 / 198 / 203 | 0.0183 | 0.184 | 1.838 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.84 | yes | 11.9 / 13.5 / 14.9 | 33.8 / 51.8 / 56.5 | 97.5 / 111 / 119 | 0.0030 | 0.0305 | 0.296 |

## xgboost / bank

500 trees, average path 5.22 comparisons per tree, model sha256 `db4a69cc292f`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.87 | yes | 6.3 / 8.6 / 10.1 | 24.0 / 29.4 / 31.2 | 156 / 188 / 195 | 0.0141 | 0.141 | 1.408 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.87 | yes | 6.4 / 7.5 / 8.2 | 28.5 / 43.1 / 47.0 | 87.5 / 103 / 109 | 0.0023 | 0.0238 | 0.230 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.03 | yes | 12.5 / 16.5 / 42.5 | 30.2 / 35.8 / 37.9 | 164 / 206 / 371 | 0.0141 | 0.145 | 1.414 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.03 | yes | 12.4 / 13.7 / 14.4 | 33.9 / 60.8 / 126 | 93.2 / 112 / 132 | 0.0024 | 0.0238 | 0.230 |

## xgboost / diabetes

500 trees, average path 4.80 comparisons per tree, model sha256 `913c8c2068d9`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.63 | yes | 5.9 / 8.3 / 9.8 | 25.0 / 31.0 / 51.6 | 185 / 199 / 236 | 0.0181 | 0.180 | 1.796 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.63 | yes | 5.9 / 7.2 / 7.6 | 28.3 / 43.1 / 47.3 | 93.2 / 109 / 128 | 0.0041 | 0.0323 | 0.335 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.47 | yes | 23.9 / 29.5 / 45.9 | 43.2 / 48.7 / 63.0 | 204 / 216 / 220 | 0.0180 | 0.178 | 1.776 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.47 | yes | 23.6 / 25.2 / 25.9 | 48.7 / 60.1 / 65.6 | 113 / 126 / 133 | 0.0033 | 0.0294 | 0.293 |

## xgboost / synthetic

500 trees, average path 4.63 comparisons per tree, model sha256 `41cfb5e58c71`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.69 | yes | 6.3 / 7.6 / 8.4 | 23.5 / 26.2 / 27.6 | 168 / 178 / 196 | 0.0151 | 0.147 | 1.464 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.69 | yes | 6.5 / 7.7 / 8.5 | 28.9 / 43.8 / 48.0 | 87.3 / 100 / 106 | 0.0026 | 0.0270 | 0.245 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.59 | yes | 11.4 / 12.8 / 14.2 | 29.0 / 31.9 / 33.5 | 178 / 187 / 193 | 0.0155 | 0.150 | 1.497 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 1.59 | yes | 11.3 / 12.7 / 14.0 | 32.7 / 50.1 / 54.9 | 95.8 / 108 / 117 | 0.0026 | 0.0276 | 0.251 |

## xgboost / synthetic-cat

500 trees, average path 7.36 comparisons per tree, model sha256 `d022b8a3af11`

matrix input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.87 | yes | 9.8 / 10.9 / 12.2 | 43.6 / 46.2 / 48.7 | 366 / 380 / 394 | 0.0363 | 0.363 | 3.636 |

matrix input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.87 | yes | 9.6 / 10.6 / 12.0 | 35.5 / 50.3 / 56.4 | 141 / 150 / 156 | 0.0056 | 0.0530 | 0.523 |

arrow input, one thread:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.73 | yes | 16.0 / 17.2 / 19.0 | 51.0 / 54.0 / 56.4 | 378 / 391 / 400 | 0.0365 | 0.365 | 3.651 |

arrow input, 10 threads:

| engine | prep [s] | exact | 1 row | 10 rows | 100 rows | 10,000 rows | 100,000 rows | 1,000,000 rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lignum-compiled | 2.73 | yes | 15.8 / 17.1 / 18.9 | 42.6 / 46.2 / 56.7 | 150 / 161 / 224 | 0.0057 | 0.0548 | 0.543 |
