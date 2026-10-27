# CatBoost benchmark 2026-10-27T00:59:30 (MSVC, generic kernel)

- lignum commit `d78b4bc`, AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD, 4 threads, Python 3.12.10
- CatBoost models: 500 trees of depth 6 trained on 80% of each data set; rows repeated up to 200,000
- versions: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, polars 2.1.0, pyarrow 26.0.0

Raw scores. Batch = all rows, all threads (best / median); 1 thread is normalised to all rows. Prep = building the prebuilt input (Pool, hash matrix) once. Latency = median per call, one thread. Mismatch = rows not bit-identical to CatBoost's Python predictions (batch and single rows).

## california (20,640 rows, 8 numeric + 0 categorical columns, regression)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.027 | 0.027 | 0.057 | 267 | 374 | 410 | 0 |
| catboost | Pool (prebuilt) | 0.000 | 0.026 | 0.027 | 0.054 | — | — | — | 0 |
| catboost | polars | — | 0.028 | 0.028 | 0.060 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.027 | 0.028 | 0.077 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.030 | 0.031 | 0.062 | — | — | — | 0 |
| lignum | pandas | — | 0.021 | 0.021 | 0.052 | 3.8 | 19.1 | 41.2 | 0 |
| lignum | pyarrow | — | 0.018 | 0.018 | 0.048 | — | — | — | 0 |
| lignum | polars | — | 0.018 | 0.018 | 0.048 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.003 | 0.017 | 0.017 | 0.045 | — | — | — | 0 |

## adult (32,561 rows, 6 numeric + 8 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.148 | 0.153 | 0.350 | 153 | 529 | 646 | 0 |
| catboost | Pool (prebuilt) | 0.005 | 0.142 | 0.142 | 0.335 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.159 | 0.159 | 0.371 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.149 | 0.166 | 0.350 | — | — | — | 0 |
| lignum | pandas | — | 0.061 | 0.062 | 0.173 | 4.6 | 41.2 | 104 | 0 |
| lignum | pyarrow | — | 0.057 | 0.057 | 0.173 | — | — | — | 0 |
| lignum | polars | — | 0.066 | 0.066 | 0.181 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.011 | 0.055 | 0.055 | 0.159 | — | — | — | 0 |

## covtype (581,012 rows, 54 numeric + 0 categorical columns, multiclass)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.174 | 0.177 | 0.312 | 150 | 713 | 882 | 0 |
| catboost | Pool (prebuilt) | 0.028 | 0.144 | 0.151 | 0.288 | — | — | — | 0 |
| catboost | polars | — | 0.148 | 0.152 | 0.329 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.146 | 0.149 | 0.317 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.162 | 0.163 | 0.309 | — | — | — | 0 |
| lignum | pandas | — | 0.047 | 0.047 | 0.134 | 4.3 | 20.2 | 73.9 | 0 |
| lignum | pyarrow | — | 0.057 | 0.057 | 0.143 | — | — | — | 0 |
| lignum | polars | — | 0.058 | 0.058 | 0.143 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.051 | 0.047 | 0.047 | 0.134 | — | — | — | 0 |
