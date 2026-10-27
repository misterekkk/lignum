# CatBoost benchmark 2026-10-27T01:58:17 (M4, d78b4bc)

- lignum commit `bf66e1c`, Apple M4, 10 threads, Python 3.12.15
- CatBoost models: 500 trees of depth 6 trained on 80% of each data set; rows repeated up to 500,000
- versions: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, polars 2.1.0, pyarrow 26.0.0

Raw scores. Batch = all rows, all threads (best / median); 1 thread is normalised to all rows. Prep = building the prebuilt input (Pool, hash matrix) once. Latency = median per call, one thread. Mismatch = rows not bit-identical to CatBoost's Python predictions (batch and single rows).

## adult (32,561 rows, 6 numeric + 8 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.145 | 0.151 | 0.781 | 55.1 | 246 | 376 | 0 |
| catboost | Pool (prebuilt) | 0.006 | 0.137 | 0.143 | 0.748 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.147 | 0.152 | 0.790 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.141 | 0.149 | 0.783 | — | — | — | 0 |
| lignum | pandas | — | 0.056 | 0.058 | 0.301 | 3.7 | 25.3 | 76.4 | 0 |
| lignum | pyarrow | — | 0.054 | 0.055 | 0.296 | — | — | — | 0 |
| lignum | polars | — | 0.065 | 0.067 | 0.307 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.022 | 0.055 | 0.055 | 0.288 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.805 | — | 0 |
| catboost | hashed, 1 thread | 0.766 | — | 0 |
| lignum | strings, 1 thread | 0.307 | — | 0 |
| lignum | hashed, 1 thread | 0.291 | — | 0 |
| catboost | strings, threads | 0.140 | — | 0 |
| catboost | hashed, threads | 0.137 | — | 0 |
| lignum | strings, threads | 0.056 | — | 0 |
| lignum | hashed, threads | 0.055 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 23.0 | 0 |
| lignum | 1 row | — | 2.1 | 0 |
| catboost | 100 rows | — | 155 | 0 |
| lignum | 100 rows | — | 59.7 | 0 |

## bank (45,211 rows, 7 numeric + 9 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.108 | 0.114 | 0.583 | 51.2 | 211 | 309 | 0 |
| catboost | Pool (prebuilt) | 0.005 | 0.104 | 0.107 | 0.559 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.112 | 0.116 | 0.584 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.107 | 0.110 | 0.586 | — | — | — | 0 |
| lignum | pandas | — | 0.046 | 0.046 | 0.237 | 3.3 | 27.4 | 70.2 | 0 |
| lignum | pyarrow | — | 0.042 | 0.043 | 0.232 | — | — | — | 0 |
| lignum | polars | — | 0.057 | 0.057 | 0.244 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.040 | 0.053 | 0.053 | 0.224 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.605 | — | 0 |
| catboost | hashed, 1 thread | 0.571 | — | 0 |
| lignum | strings, 1 thread | 0.252 | — | 0 |
| lignum | hashed, 1 thread | 0.237 | — | 0 |
| catboost | strings, threads | 0.110 | — | 0 |
| catboost | hashed, threads | 0.104 | — | 0 |
| lignum | strings, threads | 0.048 | — | 0 |
| lignum | hashed, threads | 0.053 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 17.0 | 0 |
| lignum | 1 row | — | 1.7 | 0 |
| catboost | 100 rows | — | 121 | 0 |
| lignum | 100 rows | — | 47.4 | 0 |

## diabetes (101,766 rows, 8 numeric + 39 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.389 | 0.398 | 2.008 | 105 | 2,496 | 2,782 | 0 |
| catboost | Pool (prebuilt) | 0.020 | 0.371 | 0.377 | 1.891 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.401 | 0.414 | 1.999 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.399 | 0.402 | 2.003 | — | — | — | 0 |
| lignum | pandas | — | 0.198 | 0.200 | 0.842 | 7.4 | 94.9 | 245 | 0 |
| lignum | pyarrow | — | 0.186 | 0.188 | 0.814 | — | — | — | 0 |
| lignum | polars | — | 0.248 | 0.256 | 0.888 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.153 | 0.233 | 0.234 | 0.825 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 2.030 | — | 0 |
| catboost | hashed, 1 thread | 1.897 | — | 0 |
| lignum | strings, 1 thread | 0.891 | — | 0 |
| lignum | hashed, 1 thread | 0.830 | — | 0 |
| catboost | strings, threads | 0.416 | — | 0 |
| catboost | hashed, threads | 0.372 | — | 0 |
| lignum | strings, threads | 0.196 | — | 0 |
| lignum | hashed, threads | 0.233 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 73.4 | 0 |
| lignum | 1 row | — | 5.2 | 0 |
| catboost | 100 rows | — | 403 | 0 |
| lignum | 100 rows | — | 169 | 0 |

## california (20,640 rows, 8 numeric + 0 categorical columns, regression)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.041 | 0.041 | 0.207 | 88.8 | 156 | 201 | 0 |
| catboost | Pool (prebuilt) | 0.000 | 0.040 | 0.041 | 0.207 | — | — | — | 0 |
| catboost | polars | — | 0.041 | 0.044 | 0.207 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.041 | 0.041 | 0.208 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.042 | 0.047 | 0.209 | — | — | — | 0 |
| lignum | pandas | — | 0.015 | 0.015 | 0.079 | 1.6 | 17.8 | 34.9 | 0 |
| lignum | pyarrow | — | 0.015 | 0.015 | 0.079 | — | — | — | 0 |
| lignum | polars | — | 0.015 | 0.015 | 0.079 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.007 | 0.013 | 0.014 | 0.077 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.203 | — | 0 |
| catboost | hashed, 1 thread | 0.203 | — | 0 |
| lignum | strings, 1 thread | 0.078 | — | 0 |
| lignum | hashed, 1 thread | 0.077 | — | 0 |
| catboost | strings, threads | 0.040 | — | 0 |
| catboost | hashed, threads | 0.039 | — | 0 |
| lignum | strings, threads | 0.016 | — | 0 |
| lignum | hashed, threads | 0.013 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 1.5 | 0 |
| lignum | 1 row | — | 0.6 | 0 |
| catboost | 100 rows | — | 42.0 | 0 |
| lignum | 100 rows | — | 17.5 | 0 |

## covtype (581,012 rows, 54 numeric + 0 categorical columns, multiclass)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.116 | 0.121 | 0.578 | 45.8 | 473 | 602 | 0 |
| catboost | Pool (prebuilt) | 0.001 | 0.114 | 0.117 | 0.576 | — | — | — | 0 |
| catboost | polars | — | 0.117 | 0.120 | 0.583 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.118 | 0.121 | 0.579 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.125 | 0.139 | 0.586 | — | — | — | 0 |
| lignum | pandas | — | 0.049 | 0.049 | 0.203 | 2.8 | 18.4 | 59.0 | 0 |
| lignum | pyarrow | — | 0.043 | 0.043 | 0.203 | — | — | — | 0 |
| lignum | polars | — | 0.043 | 0.043 | 0.204 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.156 | 0.035 | 0.036 | 0.188 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.545 | — | 0 |
| catboost | hashed, 1 thread | 0.545 | — | 0 |
| lignum | strings, 1 thread | 0.195 | — | 0 |
| lignum | hashed, 1 thread | 0.191 | — | 0 |
| catboost | strings, threads | 0.102 | — | 0 |
| catboost | hashed, threads | 0.105 | — | 0 |
| lignum | strings, threads | 0.042 | — | 0 |
| lignum | hashed, threads | 0.036 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 2.8 | 0 |
| lignum | 1 row | — | 1.0 | 0 |
| catboost | 100 rows | — | 109 | 0 |
| lignum | 100 rows | — | 39.6 | 0 |

## synthetic (300,000 rows, 20 numeric + 0 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.049 | 0.052 | 0.235 | 36.5 | 198 | 243 | 0 |
| catboost | Pool (prebuilt) | 0.000 | 0.047 | 0.049 | 0.235 | — | — | — | 0 |
| catboost | polars | — | 0.049 | 0.053 | 0.240 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.048 | 0.050 | 0.240 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.050 | 0.055 | 0.238 | — | — | — | 0 |
| lignum | pandas | — | 0.020 | 0.020 | 0.095 | 2.0 | 17.4 | 37.9 | 0 |
| lignum | pyarrow | — | 0.019 | 0.020 | 0.098 | — | — | — | 0 |
| lignum | polars | — | 0.019 | 0.020 | 0.099 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.038 | 0.017 | 0.018 | 0.098 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.230 | — | 0 |
| catboost | hashed, 1 thread | 0.226 | — | 0 |
| lignum | strings, 1 thread | 0.092 | — | 0 |
| lignum | hashed, 1 thread | 0.091 | — | 0 |
| catboost | strings, threads | 0.043 | — | 0 |
| catboost | hashed, threads | 0.044 | — | 0 |
| lignum | strings, threads | 0.019 | — | 0 |
| lignum | hashed, threads | 0.017 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 1.5 | 0 |
| lignum | 1 row | — | 0.8 | 0 |
| catboost | 100 rows | — | 46.3 | 0 |
| lignum | 100 rows | — | 20.2 | 0 |

## synthetic-cat (300,000 rows, 15 numeric + 5 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.174 | 0.176 | 0.851 | 63.3 | 4,948 | 5,008 | 0 |
| catboost | Pool (prebuilt) | 0.009 | 0.162 | 0.163 | 0.807 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.172 | 0.174 | 0.827 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.171 | 0.187 | 0.871 | — | — | — | 0 |
| lignum | pandas | — | 0.080 | 0.081 | 0.375 | 4.0 | 23.1 | 86.4 | 0 |
| lignum | pyarrow | — | 0.073 | 0.075 | 0.347 | — | — | — | 0 |
| lignum | polars | — | 0.087 | 0.090 | 0.357 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.043 | 0.074 | 0.075 | 0.345 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.832 | — | 0 |
| catboost | hashed, 1 thread | 0.789 | — | 0 |
| lignum | strings, 1 thread | 0.354 | — | 0 |
| lignum | hashed, 1 thread | 0.349 | — | 0 |
| catboost | strings, threads | 0.196 | — | 0 |
| catboost | hashed, threads | 0.177 | — | 0 |
| lignum | strings, threads | 0.077 | — | 0 |
| lignum | hashed, threads | 0.074 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 25.2 | 0 |
| lignum | 1 row | — | 2.3 | 0 |
| catboost | 100 rows | — | 165 | 0 |
| lignum | 100 rows | — | 72.8 | 0 |
