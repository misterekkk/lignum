# CatBoost benchmark 2026-10-27T01:09:15 (default build (x86: AVX2 kernel chosen at run time))

- lignum commit `bf66e1c`, AMD EPYC 9V45 96-Core Processor (x86_64), 4 threads, Python 3.12.15
- CatBoost models: 500 trees of depth 6 trained on 80% of each data set; rows repeated up to 300,000
- versions: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, polars 2.1.0, pyarrow 26.0.0

Raw scores. Batch = all rows, all threads (best / median); 1 thread is normalised to all rows. Prep = building the prebuilt input (Pool, hash matrix) once. Latency = median per call, one thread. Mismatch = rows not bit-identical to CatBoost's Python predictions (batch and single rows).

## adult (32,561 rows, 6 numeric + 8 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.169 | 0.170 | 0.424 | 70.4 | 418 | 535 | 0 |
| catboost | Pool (prebuilt) | 0.006 | 0.162 | 0.164 | 0.406 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.175 | 0.177 | 0.439 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.166 | 0.169 | 0.410 | — | — | — | 0 |
| lignum | pandas | — | 0.090 | 0.092 | 0.207 | 4.2 | 36.7 | 97.3 | 0 |
| lignum | pyarrow | — | 0.087 | 0.095 | 0.195 | — | — | — | 0 |
| lignum | polars | — | 0.094 | 0.097 | 0.204 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.011 | 0.082 | 0.082 | 0.190 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.424 | — | 0 |
| catboost | hashed, 1 thread | 0.396 | — | 0 |
| lignum | strings, 1 thread | 0.213 | — | 0 |
| lignum | hashed, 1 thread | 0.202 | — | 0 |
| catboost | strings, threads | 0.181 | — | 0 |
| catboost | hashed, threads | 0.166 | — | 0 |
| lignum | strings, threads | 0.085 | — | 0 |
| lignum | hashed, threads | 0.085 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 20.1 | 0 |
| lignum | 1 row | — | 2.3 | 0 |
| catboost | 100 rows | — | 145 | 0 |
| lignum | 100 rows | — | 74.1 | 0 |

## bank (45,211 rows, 7 numeric + 9 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.143 | 0.145 | 0.348 | 70.5 | 318 | 433 | 0 |
| catboost | Pool (prebuilt) | 0.006 | 0.139 | 0.140 | 0.350 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.156 | 0.159 | 0.367 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.148 | 0.159 | 0.361 | — | — | — | 0 |
| lignum | pandas | — | 0.088 | 0.091 | 0.193 | 4.4 | 42.0 | 103 | 0 |
| lignum | pyarrow | — | 0.079 | 0.079 | 0.183 | — | — | — | 0 |
| lignum | polars | — | 0.091 | 0.091 | 0.192 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.013 | 0.085 | 0.085 | 0.178 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.378 | — | 0 |
| catboost | hashed, 1 thread | 0.349 | — | 0 |
| lignum | strings, 1 thread | 0.198 | — | 0 |
| lignum | hashed, 1 thread | 0.188 | — | 0 |
| catboost | strings, threads | 0.161 | — | 0 |
| catboost | hashed, threads | 0.146 | — | 0 |
| lignum | strings, threads | 0.078 | — | 0 |
| lignum | hashed, threads | 0.083 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 20.1 | 0 |
| lignum | 1 row | — | 2.2 | 0 |
| catboost | 100 rows | — | 135 | 0 |
| lignum | 100 rows | — | 68.4 | 0 |

## diabetes (101,766 rows, 8 numeric + 39 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.650 | 0.660 | 1.686 | 155 | 4,784 | 5,234 | 0 |
| catboost | Pool (prebuilt) | 0.025 | 0.616 | 0.618 | 1.465 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.666 | 0.675 | 1.680 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.648 | 0.658 | 1.693 | — | — | — | 0 |
| lignum | pandas | — | 0.323 | 0.345 | 0.683 | 9.7 | 149 | 368 | 0 |
| lignum | pyarrow | — | 0.298 | 0.300 | 0.652 | — | — | — | 0 |
| lignum | polars | — | 0.339 | 0.341 | 0.693 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.043 | 0.303 | 0.304 | 0.679 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 1.624 | — | 0 |
| catboost | hashed, 1 thread | 1.477 | — | 0 |
| lignum | strings, 1 thread | 0.735 | — | 0 |
| lignum | hashed, 1 thread | 0.691 | — | 0 |
| catboost | strings, threads | 0.659 | — | 0 |
| catboost | hashed, threads | 0.606 | — | 0 |
| lignum | strings, threads | 0.292 | — | 0 |
| lignum | hashed, threads | 0.309 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 88.3 | 0 |
| lignum | 1 row | — | 7.1 | 0 |
| catboost | 100 rows | — | 544 | 0 |
| lignum | 100 rows | — | 253 | 0 |

## california (20,640 rows, 8 numeric + 0 categorical columns, regression)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.034 | 0.034 | 0.075 | 112 | 211 | 242 | 0 |
| catboost | Pool (prebuilt) | 0.000 | 0.034 | 0.035 | 0.075 | — | — | — | 0 |
| catboost | polars | — | 0.034 | 0.035 | 0.076 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.035 | 0.035 | 0.076 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.037 | 0.037 | 0.079 | — | — | — | 0 |
| lignum | pandas | — | 0.028 | 0.028 | 0.063 | 2.6 | 21.8 | 46.4 | 0 |
| lignum | pyarrow | — | 0.028 | 0.028 | 0.062 | — | — | — | 0 |
| lignum | polars | — | 0.028 | 0.029 | 0.062 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.002 | 0.026 | 0.027 | 0.061 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.071 | — | 0 |
| catboost | hashed, 1 thread | 0.073 | — | 0 |
| lignum | strings, 1 thread | 0.061 | — | 0 |
| lignum | hashed, 1 thread | 0.061 | — | 0 |
| catboost | strings, threads | 0.030 | — | 0 |
| catboost | hashed, threads | 0.030 | — | 0 |
| lignum | strings, threads | 0.027 | — | 0 |
| lignum | hashed, threads | 0.027 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 2.0 | 0 |
| lignum | 1 row | — | 0.9 | 0 |
| catboost | 100 rows | — | 27.8 | 0 |
| lignum | 100 rows | — | 24.8 | 0 |

## covtype (581,012 rows, 54 numeric + 0 categorical columns, multiclass)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.211 | 0.218 | 0.355 | 64.1 | 646 | 800 | 0 |
| catboost | Pool (prebuilt) | 0.038 | 0.148 | 0.149 | 0.319 | — | — | — | 0 |
| catboost | polars | — | 0.151 | 0.155 | 0.317 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.146 | 0.151 | 0.311 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.159 | 0.159 | 0.340 | — | — | — | 0 |
| lignum | pandas | — | 0.055 | 0.055 | 0.116 | 3.8 | 22.0 | 65.2 | 0 |
| lignum | pyarrow | — | 0.063 | 0.063 | 0.128 | — | — | — | 0 |
| lignum | polars | — | 0.061 | 0.062 | 0.123 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.054 | 0.052 | 0.053 | 0.112 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.287 | — | 0 |
| catboost | hashed, 1 thread | 0.289 | — | 0 |
| lignum | strings, 1 thread | 0.119 | — | 0 |
| lignum | hashed, 1 thread | 0.120 | — | 0 |
| catboost | strings, threads | 0.126 | — | 0 |
| catboost | hashed, threads | 0.131 | — | 0 |
| lignum | strings, threads | 0.055 | — | 0 |
| lignum | hashed, threads | 0.054 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 3.1 | 0 |
| lignum | 1 row | — | 1.4 | 0 |
| catboost | 100 rows | — | 96.8 | 0 |
| lignum | 100 rows | — | 42.4 | 0 |

## synthetic (300,000 rows, 20 numeric + 0 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.041 | 0.042 | 0.093 | 48.9 | 263 | 294 | 0 |
| catboost | Pool (prebuilt) | 0.001 | 0.040 | 0.041 | 0.092 | — | — | — | 0 |
| catboost | polars | — | 0.042 | 0.043 | 0.096 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.041 | 0.042 | 0.098 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.044 | 0.047 | 0.101 | — | — | — | 0 |
| lignum | pandas | — | 0.037 | 0.037 | 0.074 | 2.9 | 21.0 | 48.9 | 0 |
| lignum | pyarrow | — | 0.035 | 0.037 | 0.078 | — | — | — | 0 |
| lignum | polars | — | 0.035 | 0.037 | 0.079 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.009 | 0.031 | 0.032 | 0.072 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.082 | — | 0 |
| catboost | hashed, 1 thread | 0.085 | — | 0 |
| lignum | strings, 1 thread | 0.074 | — | 0 |
| lignum | hashed, 1 thread | 0.071 | — | 0 |
| catboost | strings, threads | 0.035 | — | 0 |
| catboost | hashed, threads | 0.035 | — | 0 |
| lignum | strings, threads | 0.031 | — | 0 |
| lignum | hashed, threads | 0.032 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 2.1 | 0 |
| lignum | 1 row | — | 1.2 | 0 |
| catboost | 100 rows | — | 31.5 | 0 |
| lignum | 100 rows | — | 28.7 | 0 |

## synthetic-cat (300,000 rows, 15 numeric + 5 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.251 | 0.263 | 0.636 | 90.5 | 10,140 | 10,207 | 0 |
| catboost | Pool (prebuilt) | 0.015 | 0.230 | 0.233 | 0.549 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.243 | 0.249 | 0.584 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.256 | 0.266 | 0.639 | — | — | — | 0 |
| lignum | pandas | — | 0.145 | 0.147 | 0.315 | 5.3 | 33.4 | 140 | 0 |
| lignum | pyarrow | — | 0.136 | 0.139 | 0.304 | — | — | — | 0 |
| lignum | polars | — | 0.143 | 0.148 | 0.316 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.011 | 0.129 | 0.130 | 0.299 | — | — | — | 0 |

C API (CatBoost's C API has no thread setting: "threads" splits the rows over std::threads; lignum hashed uses its own threads):

| engine | case | batch best [s] | median per call [µs] | mismatch |
|---|---|---:|---:|---:|
| catboost | strings, 1 thread | 0.596 | — | 0 |
| catboost | hashed, 1 thread | 0.561 | — | 0 |
| lignum | strings, 1 thread | 0.311 | — | 0 |
| lignum | hashed, 1 thread | 0.313 | — | 0 |
| catboost | strings, threads | 0.253 | — | 0 |
| catboost | hashed, threads | 0.245 | — | 0 |
| lignum | strings, threads | 0.132 | — | 0 |
| lignum | hashed, threads | 0.132 | — | 0 |
| catboost | 1 row (CalcModelPredictionSingle) | — | 33.9 | 0 |
| lignum | 1 row | — | 3.7 | 0 |
| catboost | 100 rows | — | 214 | 0 |
| lignum | 100 rows | — | 118 | 0 |
