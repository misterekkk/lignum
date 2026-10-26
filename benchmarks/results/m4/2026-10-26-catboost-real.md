# CatBoost benchmark 2026-10-26T14:00:11

- lignum commit `0f4d5eb`, Apple M4, 10 threads, Python 3.12.15
- CatBoost models: 500 trees of depth 6 trained on 80% of each data set; rows repeated up to 500,000
- versions: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, polars 2.1.0, pyarrow 26.0.0

Raw scores. Batch = all rows, all threads (best / median); 1 thread is normalised to all rows. Prep = building the prebuilt input (Pool, hash matrix) once. Latency = median per call, one thread. Mismatch = rows not bit-identical to CatBoost's Python predictions (batch and single rows).

## adult (32,561 rows, 6 numeric + 8 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.142 | 0.152 | 0.774 | 54.7 | 256 | 384 | 0 |
| catboost | Pool (prebuilt) | 0.006 | 0.138 | 0.139 | 0.752 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.146 | 0.151 | 0.793 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.157 | 0.163 | 0.783 | — | — | — | 0 |
| lignum | pandas | — | 0.109 | 0.114 | 0.488 | 4.4 | 78.6 | 448 | 0 |
| lignum | pyarrow | — | 0.088 | 0.113 | 0.502 | — | — | — | 0 |
| lignum | polars | — | 0.106 | 0.114 | 0.494 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.025 | 0.090 | 0.096 | 0.472 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.806 | — | 0 |
| catboost | hashed, 1 thread | 0.771 | — | 0 |
| lignum | strings, 1 thread | 0.484 | — | 0 |
| lignum | hashed, 1 thread | 0.469 | — | 0 |
| catboost | strings, threads | 0.144 | — | 0 |
| catboost | hashed, threads | 0.139 | — | 0 |
| lignum | strings, threads | 0.089 | — | 0 |
| lignum | hashed, threads | 0.086 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 22.9 | 0 |
| lignum | 1 row | — | 2.8 | 0 |
| catboost | 100 rows | — | 156 | 0 |
| lignum | 100 rows | — | 89.5 | 0 |

## bank (45,211 rows, 7 numeric + 9 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.110 | 0.118 | 0.588 | 52.4 | 214 | 318 | 0 |
| catboost | Pool (prebuilt) | 0.005 | 0.103 | 0.104 | 0.563 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.110 | 0.113 | 0.585 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.108 | 0.112 | 0.594 | — | — | — | 0 |
| lignum | pandas | — | 0.120 | 0.121 | 0.404 | 4.1 | 82.1 | 489 | 0 |
| lignum | pyarrow | — | 0.073 | 0.076 | 0.397 | — | — | — | 0 |
| lignum | polars | — | 0.087 | 0.089 | 0.410 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.046 | 0.083 | 0.083 | 0.390 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.607 | — | 0 |
| catboost | hashed, 1 thread | 0.573 | — | 0 |
| lignum | strings, 1 thread | 0.414 | — | 0 |
| lignum | hashed, 1 thread | 0.398 | — | 0 |
| catboost | strings, threads | 0.115 | — | 0 |
| catboost | hashed, threads | 0.109 | — | 0 |
| lignum | strings, threads | 0.078 | — | 0 |
| lignum | hashed, threads | 0.082 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 17.3 | 0 |
| lignum | 1 row | — | 2.4 | 0 |
| catboost | 100 rows | — | 121 | 0 |
| lignum | 100 rows | — | 74.5 | 0 |

## diabetes (101,766 rows, 8 numeric + 39 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.396 | 0.408 | 2.118 | 108 | 2,533 | 2,855 | 0 |
| catboost | Pool (prebuilt) | 0.022 | 0.372 | 0.378 | 1.912 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.401 | 0.407 | 2.111 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.400 | 0.421 | 2.020 | — | — | — | 0 |
| lignum | pandas | — | 0.445 | 0.446 | 1.215 | 7.8 | 373 | 1,455 | 0 |
| lignum | pyarrow | — | 0.252 | 0.259 | 1.161 | — | — | — | 0 |
| lignum | polars | — | 0.316 | 0.320 | 1.211 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.174 | 0.301 | 0.310 | 1.161 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 2.041 | — | 0 |
| catboost | hashed, 1 thread | 1.914 | — | 0 |
| lignum | strings, 1 thread | 1.223 | — | 0 |
| lignum | hashed, 1 thread | 1.163 | — | 0 |
| catboost | strings, threads | 0.415 | — | 0 |
| catboost | hashed, threads | 0.370 | — | 0 |
| lignum | strings, threads | 0.257 | — | 0 |
| lignum | hashed, threads | 0.295 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 73.3 | 0 |
| lignum | 1 row | — | 5.8 | 0 |
| catboost | 100 rows | — | 405 | 0 |
| lignum | 100 rows | — | 222 | 0 |

## california (20,640 rows, 8 numeric + 0 categorical columns, regression)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.042 | 0.044 | 0.212 | 87.6 | 156 | 199 | 0 |
| catboost | Pool (prebuilt) | 0.000 | 0.040 | 0.041 | 0.208 | — | — | — | 0 |
| catboost | polars | — | 0.039 | 0.042 | 0.209 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.043 | 0.046 | 0.209 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.045 | 0.047 | 0.210 | — | — | — | 0 |
| lignum | pandas | ValueError: Input array must be 2D. | | | | | | | |
| lignum | pyarrow | — | 0.043 | 0.044 | 0.232 | — | — | — | 0 |
| lignum | polars | — | 0.043 | 0.044 | 0.231 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.007 | 0.040 | 0.040 | 0.238 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.204 | — | 0 |
| catboost | hashed, 1 thread | 0.204 | — | 0 |
| lignum | strings, 1 thread | 0.228 | — | 0 |
| lignum | hashed, 1 thread | 0.228 | — | 0 |
| catboost | strings, threads | 0.041 | — | 0 |
| catboost | hashed, threads | 0.040 | — | 0 |
| lignum | strings, threads | 0.044 | — | 0 |
| lignum | hashed, threads | 0.040 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 1.6 | 0 |
| lignum | 1 row | — | 1.8 | 0 |
| catboost | 100 rows | — | 42.0 | 0 |
| lignum | 100 rows | — | 43.0 | 0 |

## covtype (581,012 rows, 54 numeric + 0 categorical columns, multiclass)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.123 | 0.141 | 0.581 | 46.1 | 480 | 608 | 0 |
| catboost | Pool (prebuilt) | 0.001 | 0.122 | 0.126 | 0.577 | — | — | — | 0 |
| catboost | polars | — | 0.120 | 0.125 | 0.577 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.122 | 0.124 | 0.582 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.130 | 0.133 | 0.587 | — | — | — | 0 |
| lignum | pandas | ValueError: Input array must be 2D. | | | | | | | |
| lignum | pyarrow | — | 0.140 | 0.142 | 0.701 | — | — | — | 0 |
| lignum | polars | — | 0.137 | 0.141 | 0.703 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.183 | 0.126 | 0.127 | 0.695 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.547 | — | 0 |
| catboost | hashed, 1 thread | 0.543 | — | 0 |
| lignum | strings, 1 thread | 0.698 | — | 0 |
| lignum | hashed, 1 thread | 0.697 | — | 0 |
| catboost | strings, threads | 0.106 | — | 0 |
| catboost | hashed, threads | 0.110 | — | 0 |
| lignum | strings, threads | 0.132 | — | 0 |
| lignum | hashed, threads | 0.124 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 2.8 | 0 |
| lignum | 1 row | — | 2.9 | 0 |
| catboost | 100 rows | — | 109 | 0 |
| lignum | 100 rows | — | 134 | 0 |
