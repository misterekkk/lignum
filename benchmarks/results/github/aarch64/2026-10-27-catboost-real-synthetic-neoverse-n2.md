# CatBoost benchmark 2026-10-27T01:12:46 (default build (x86: AVX2 kernel chosen at run time))

- lignum commit `bf66e1c`, Neoverse-N2 (aarch64), 4 threads, Python 3.12.15
- CatBoost models: 500 trees of depth 6 trained on 80% of each data set; rows repeated up to 300,000
- versions: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, polars 2.1.0, pyarrow 26.0.0

Raw scores. Batch = all rows, all threads (best / median); 1 thread is normalised to all rows. Prep = building the prebuilt input (Pool, hash matrix) once. Latency = median per call, one thread. Mismatch = rows not bit-identical to CatBoost's Python predictions (batch and single rows).

## adult (32,561 rows, 6 numeric + 8 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.261 | 0.264 | 0.951 | 215 | 915 | 1,214 | 0 |
| catboost | Pool (prebuilt) | 0.008 | 0.244 | 0.245 | 0.935 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.259 | 0.264 | 0.966 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.259 | 0.271 | 0.967 | — | — | — | 0 |
| lignum | pandas | — | 0.101 | 0.102 | 0.354 | 9.4 | 72.4 | 201 | 0 |
| lignum | pyarrow | — | 0.095 | 0.096 | 0.331 | — | — | — | 0 |
| lignum | polars | — | 0.115 | 0.116 | 0.352 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.017 | 0.094 | 0.095 | 0.324 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.941 | — | 0 |
| catboost | hashed, 1 thread | 0.902 | — | 0 |
| lignum | strings, 1 thread | 0.339 | — | 0 |
| lignum | hashed, 1 thread | 0.327 | — | 0 |
| catboost | strings, threads | 0.249 | — | 0 |
| catboost | hashed, threads | 0.234 | — | 0 |
| lignum | strings, threads | 0.090 | — | 0 |
| lignum | hashed, threads | 0.094 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 50.7 | 0 |
| lignum | 1 row | — | 4.9 | 0 |
| catboost | 100 rows | — | 324 | 0 |
| lignum | 100 rows | — | 117 | 0 |

## bank (45,211 rows, 7 numeric + 9 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.202 | 0.203 | 0.743 | 213 | 799 | 1,041 | 0 |
| catboost | Pool (prebuilt) | 0.007 | 0.183 | 0.184 | 0.717 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.197 | 0.204 | 0.750 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.196 | 0.200 | 0.758 | — | — | — | 0 |
| lignum | pandas | — | 0.080 | 0.080 | 0.288 | 7.2 | 73.3 | 176 | 0 |
| lignum | pyarrow | — | 0.075 | 0.076 | 0.266 | — | — | — | 0 |
| lignum | polars | — | 0.096 | 0.096 | 0.289 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.018 | 0.076 | 0.076 | 0.259 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.739 | — | 0 |
| catboost | hashed, 1 thread | 0.698 | — | 0 |
| lignum | strings, 1 thread | 0.263 | — | 0 |
| lignum | hashed, 1 thread | 0.256 | — | 0 |
| catboost | strings, threads | 0.195 | — | 0 |
| catboost | hashed, threads | 0.179 | — | 0 |
| lignum | strings, threads | 0.069 | — | 0 |
| lignum | hashed, threads | 0.074 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 30.7 | 0 |
| lignum | 1 row | — | 3.6 | 0 |
| catboost | 100 rows | — | 253 | 0 |
| lignum | 100 rows | — | 89.2 | 0 |

## diabetes (101,766 rows, 8 numeric + 39 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.865 | 0.871 | 3.278 | 497 | 8,125 | 9,215 | 0 |
| catboost | Pool (prebuilt) | 0.035 | 0.837 | 0.850 | 3.190 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.848 | 0.877 | 3.267 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.879 | 0.897 | 3.341 | — | — | — | 0 |
| lignum | pandas | — | 0.394 | 0.395 | 1.337 | 45.9 | 310 | 803 | 0 |
| lignum | pyarrow | — | 0.370 | 0.374 | 1.249 | — | — | — | 0 |
| lignum | polars | — | 0.451 | 0.468 | 1.313 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.086 | 0.368 | 0.379 | 1.175 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 2.926 | — | 0 |
| catboost | hashed, 1 thread | 2.647 | — | 0 |
| lignum | strings, 1 thread | 1.209 | — | 0 |
| lignum | hashed, 1 thread | 1.146 | — | 0 |
| catboost | strings, threads | 0.755 | — | 0 |
| catboost | hashed, threads | 0.712 | — | 0 |
| lignum | strings, threads | 0.348 | — | 0 |
| lignum | hashed, threads | 0.378 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 237 | 0 |
| lignum | 1 row | — | 33.1 | 0 |
| catboost | 100 rows | — | 1,040 | 0 |
| lignum | 100 rows | — | 517 | 0 |

## california (20,640 rows, 8 numeric + 0 categorical columns, regression)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.088 | 0.101 | 0.323 | 281 | 569 | 681 | 0 |
| catboost | Pool (prebuilt) | 0.001 | 0.081 | 0.101 | 0.320 | — | — | — | 0 |
| catboost | polars | — | 0.098 | 0.099 | 0.324 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.081 | 0.100 | 0.324 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.083 | 0.083 | 0.328 | — | — | — | 0 |
| lignum | pandas | — | 0.025 | 0.025 | 0.092 | 3.7 | 39.5 | 87.6 | 0 |
| lignum | pyarrow | — | 0.025 | 0.025 | 0.092 | — | — | — | 0 |
| lignum | polars | — | 0.025 | 0.025 | 0.092 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.005 | 0.022 | 0.022 | 0.089 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.318 | — | 0 |
| catboost | hashed, 1 thread | 0.313 | — | 0 |
| lignum | strings, 1 thread | 0.088 | — | 0 |
| lignum | hashed, 1 thread | 0.087 | — | 0 |
| catboost | strings, threads | 0.079 | — | 0 |
| catboost | hashed, threads | 0.079 | — | 0 |
| lignum | strings, threads | 0.022 | — | 0 |
| lignum | hashed, threads | 0.022 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 3.1 | 0 |
| lignum | 1 row | — | 1.2 | 0 |
| catboost | 100 rows | — | 107 | 0 |
| lignum | 100 rows | — | 31.6 | 0 |

## covtype (581,012 rows, 54 numeric + 0 categorical columns, multiclass)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.346 | 0.359 | 0.933 | 174 | 1,369 | 1,807 | 0 |
| catboost | Pool (prebuilt) | 0.070 | 0.242 | 0.247 | 0.856 | — | — | — | 0 |
| catboost | polars | — | 0.262 | 0.265 | 0.871 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.249 | 0.257 | 0.888 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.254 | 0.256 | 0.887 | — | — | — | 0 |
| lignum | pandas | — | 0.072 | 0.072 | 0.265 | 7.0 | 47.2 | 194 | 0 |
| lignum | pyarrow | — | 0.085 | 0.086 | 0.287 | — | — | — | 0 |
| lignum | polars | — | 0.084 | 0.085 | 0.285 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.087 | 0.070 | 0.071 | 0.264 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.818 | — | 0 |
| catboost | hashed, 1 thread | 0.812 | — | 0 |
| lignum | strings, 1 thread | 0.265 | — | 0 |
| lignum | hashed, 1 thread | 0.265 | — | 0 |
| catboost | strings, threads | 0.227 | — | 0 |
| catboost | hashed, threads | 0.229 | — | 0 |
| lignum | strings, threads | 0.072 | — | 0 |
| lignum | hashed, threads | 0.071 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 5.8 | 0 |
| lignum | 1 row | — | 2.5 | 0 |
| catboost | 100 rows | — | 279 | 0 |
| lignum | 100 rows | — | 104 | 0 |

## synthetic (300,000 rows, 20 numeric + 0 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.096 | 0.103 | 0.379 | 129 | 667 | 795 | 0 |
| catboost | Pool (prebuilt) | 0.001 | 0.094 | 0.095 | 0.375 | — | — | — | 0 |
| catboost | polars | — | 0.105 | 0.119 | 0.384 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.112 | 0.117 | 0.384 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.098 | 0.099 | 0.385 | — | — | — | 0 |
| lignum | pandas | — | 0.033 | 0.033 | 0.117 | 4.4 | 41.9 | 89.7 | 0 |
| lignum | pyarrow | — | 0.030 | 0.030 | 0.113 | — | — | — | 0 |
| lignum | polars | — | 0.030 | 0.030 | 0.115 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.016 | 0.026 | 0.026 | 0.105 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.357 | — | 0 |
| catboost | hashed, 1 thread | 0.353 | — | 0 |
| lignum | strings, 1 thread | 0.104 | — | 0 |
| lignum | hashed, 1 thread | 0.104 | — | 0 |
| catboost | strings, threads | 0.089 | — | 0 |
| catboost | hashed, threads | 0.089 | — | 0 |
| lignum | strings, threads | 0.026 | — | 0 |
| lignum | hashed, threads | 0.026 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 3.3 | 0 |
| lignum | 1 row | — | 1.6 | 0 |
| catboost | 100 rows | — | 121 | 0 |
| lignum | 100 rows | — | 36.5 | 0 |

## synthetic-cat (300,000 rows, 15 numeric + 5 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.416 | 0.435 | 1.622 | 313 | 15,435 | 16,034 | 0 |
| catboost | Pool (prebuilt) | 0.020 | 0.395 | 0.401 | 1.539 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.409 | 0.416 | 1.490 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.409 | 0.412 | 1.643 | — | — | — | 0 |
| lignum | pandas | — | 0.207 | 0.211 | 0.678 | 17.3 | 77.1 | 370 | 0 |
| lignum | pyarrow | — | 0.204 | 0.208 | 0.643 | — | — | — | 0 |
| lignum | polars | — | 0.213 | 0.218 | 0.651 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.021 | 0.187 | 0.192 | 0.640 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 1.356 | — | 0 |
| catboost | hashed, 1 thread | 1.340 | — | 0 |
| lignum | strings, 1 thread | 0.604 | — | 0 |
| lignum | hashed, 1 thread | 0.576 | — | 0 |
| catboost | strings, threads | 0.374 | — | 0 |
| catboost | hashed, threads | 0.360 | — | 0 |
| lignum | strings, threads | 0.167 | — | 0 |
| lignum | hashed, threads | 0.171 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 80.1 | 0 |
| lignum | 1 row | — | 10.6 | 0 |
| catboost | 100 rows | — | 481 | 0 |
| lignum | 100 rows | — | 250 | 0 |
