# CatBoost benchmark 2026-10-26T15:26:43

- lignum commit `4b6de64`, Apple M4, 10 threads, Python 3.12.15
- CatBoost models: 500 trees of depth 6 trained on 80% of each data set; rows repeated up to 500,000
- versions: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, polars 2.1.0, pyarrow 26.0.0

Raw scores. Batch = all rows, all threads (best / median); 1 thread is normalised to all rows. Prep = building the prebuilt input (Pool, hash matrix) once. Latency = median per call, one thread. Mismatch = rows not bit-identical to CatBoost's Python predictions (batch and single rows).

## adult (32,561 rows, 6 numeric + 8 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.142 | 0.145 | 0.783 | 55.4 | 252 | 379 | 0 |
| catboost | Pool (prebuilt) | 0.005 | 0.135 | 0.138 | 0.752 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.141 | 0.149 | 0.792 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.144 | 0.147 | 0.791 | — | — | — | 0 |
| lignum | pandas | — | 0.065 | 0.067 | 0.355 | 4.4 | 26.0 | 84.0 | 0 |
| lignum | pyarrow | — | 0.064 | 0.065 | 0.346 | — | — | — | 0 |
| lignum | polars | — | 0.076 | 0.077 | 0.361 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.020 | 0.064 | 0.065 | 0.341 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.809 | — | 0 |
| catboost | hashed, 1 thread | 0.768 | — | 0 |
| lignum | strings, 1 thread | 0.358 | — | 0 |
| lignum | hashed, 1 thread | 0.341 | — | 0 |
| catboost | strings, threads | 0.143 | — | 0 |
| catboost | hashed, threads | 0.141 | — | 0 |
| lignum | strings, threads | 0.066 | — | 0 |
| lignum | hashed, threads | 0.065 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 22.7 | 0 |
| lignum | 1 row | — | 2.8 | 0 |
| catboost | 100 rows | — | 155 | 0 |
| lignum | 100 rows | — | 64.8 | 0 |

## bank (45,211 rows, 7 numeric + 9 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.108 | 0.125 | 0.587 | 51.8 | 210 | 313 | 0 |
| catboost | Pool (prebuilt) | 0.005 | 0.102 | 0.120 | 0.557 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.114 | 0.136 | 0.586 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.111 | 0.115 | 0.591 | — | — | — | 0 |
| lignum | pandas | — | 0.053 | 0.054 | 0.277 | 4.2 | 28.0 | 76.3 | 0 |
| lignum | pyarrow | — | 0.050 | 0.054 | 0.275 | — | — | — | 0 |
| lignum | polars | — | 0.064 | 0.065 | 0.287 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.040 | 0.061 | 0.061 | 0.265 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.607 | — | 0 |
| catboost | hashed, 1 thread | 0.572 | — | 0 |
| lignum | strings, 1 thread | 0.297 | — | 0 |
| lignum | hashed, 1 thread | 0.281 | — | 0 |
| catboost | strings, threads | 0.118 | — | 0 |
| catboost | hashed, threads | 0.104 | — | 0 |
| lignum | strings, threads | 0.054 | — | 0 |
| lignum | hashed, threads | 0.061 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 17.5 | 0 |
| lignum | 1 row | — | 2.4 | 0 |
| catboost | 100 rows | — | 121 | 0 |
| lignum | 100 rows | — | 52.0 | 0 |

## diabetes (101,766 rows, 8 numeric + 39 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.389 | 0.408 | 2.018 | 106 | 2,590 | 2,905 | 0 |
| catboost | Pool (prebuilt) | 0.020 | 0.390 | 0.396 | 1.957 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.419 | 0.430 | 2.040 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.403 | 0.416 | 2.007 | — | — | — | 0 |
| lignum | pandas | — | 0.231 | 0.237 | 1.021 | 8.2 | 96.7 | 262 | 0 |
| lignum | pyarrow | — | 0.227 | 0.241 | 0.978 | — | — | — | 0 |
| lignum | polars | — | 0.290 | 0.299 | 1.070 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.158 | 0.273 | 0.274 | 0.976 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 2.054 | — | 0 |
| catboost | hashed, 1 thread | 1.953 | — | 0 |
| lignum | strings, 1 thread | 1.051 | — | 0 |
| lignum | hashed, 1 thread | 0.990 | — | 0 |
| catboost | strings, threads | 0.420 | — | 0 |
| catboost | hashed, threads | 0.369 | — | 0 |
| lignum | strings, threads | 0.242 | — | 0 |
| lignum | hashed, threads | 0.284 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 74.3 | 0 |
| lignum | 1 row | — | 6.0 | 0 |
| catboost | 100 rows | — | 403 | 0 |
| lignum | 100 rows | — | 186 | 0 |

## california (20,640 rows, 8 numeric + 0 categorical columns, regression)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.041 | 0.043 | 0.212 | 90.8 | 158 | 198 | 0 |
| catboost | Pool (prebuilt) | 0.000 | 0.040 | 0.044 | 0.208 | — | — | — | 0 |
| catboost | polars | — | 0.043 | 0.044 | 0.211 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.048 | 0.050 | 0.212 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.043 | 0.047 | 0.210 | — | — | — | 0 |
| lignum | pandas | — | 0.021 | 0.022 | 0.115 | 2.7 | 18.2 | 40.7 | 0 |
| lignum | pyarrow | — | 0.021 | 0.022 | 0.113 | — | — | — | 0 |
| lignum | polars | — | 0.021 | 0.022 | 0.110 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.007 | 0.021 | 0.021 | 0.114 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.203 | — | 0 |
| catboost | hashed, 1 thread | 0.203 | — | 0 |
| lignum | strings, 1 thread | 0.107 | — | 0 |
| lignum | hashed, 1 thread | 0.107 | — | 0 |
| catboost | strings, threads | 0.038 | — | 0 |
| catboost | hashed, threads | 0.041 | — | 0 |
| lignum | strings, threads | 0.026 | — | 0 |
| lignum | hashed, threads | 0.020 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 1.6 | 0 |
| lignum | 1 row | — | 1.7 | 0 |
| catboost | 100 rows | — | 42.0 | 0 |
| lignum | 100 rows | — | 23.1 | 0 |

## covtype (581,012 rows, 54 numeric + 0 categorical columns, multiclass)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.119 | 0.128 | 0.592 | 46.4 | 487 | 588 | 0 |
| catboost | Pool (prebuilt) | 0.001 | 0.120 | 0.129 | 0.584 | — | — | — | 0 |
| catboost | polars | — | 0.128 | 0.140 | 0.580 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.126 | 0.137 | 0.583 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.129 | 0.135 | 0.594 | — | — | — | 0 |
| lignum | pandas | — | 0.074 | 0.075 | 0.331 | 5.2 | 21.0 | 81.2 | 0 |
| lignum | pyarrow | — | 0.072 | 0.076 | 0.330 | — | — | — | 0 |
| lignum | polars | — | 0.070 | 0.073 | 0.338 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.195 | 0.062 | 0.062 | 0.316 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.546 | — | 0 |
| catboost | hashed, 1 thread | 0.542 | — | 0 |
| lignum | strings, 1 thread | 0.343 | — | 0 |
| lignum | hashed, 1 thread | 0.314 | — | 0 |
| catboost | strings, threads | 0.105 | — | 0 |
| catboost | hashed, threads | 0.110 | — | 0 |
| lignum | strings, threads | 0.062 | — | 0 |
| lignum | hashed, threads | 0.062 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 2.8 | 0 |
| lignum | 1 row | — | 2.9 | 0 |
| catboost | 100 rows | — | 109 | 0 |
| lignum | 100 rows | — | 61.6 | 0 |
