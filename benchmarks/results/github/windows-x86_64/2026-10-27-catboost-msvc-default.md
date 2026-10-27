# CatBoost benchmark 2026-10-27T01:01:14 (MSVC, default kernel (AVX2 where the CPU has it))

- lignum commit `d78b4bc`, AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD, 4 threads, Python 3.12.10
- CatBoost models: 500 trees of depth 6 trained on 80% of each data set; rows repeated up to 200,000
- versions: lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, polars 2.1.0, pyarrow 26.0.0

Raw scores. Batch = all rows, all threads (best / median); 1 thread is normalised to all rows. Prep = building the prebuilt input (Pool, hash matrix) once. Latency = median per call, one thread. Mismatch = rows not bit-identical to CatBoost's Python predictions (batch and single rows).

## california (20,640 rows, 8 numeric + 0 categorical columns, regression)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.026 | 0.027 | 0.055 | 263 | 371 | 399 | 0 |
| catboost | Pool (prebuilt) | 0.000 | 0.026 | 0.026 | 0.054 | — | — | — | 0 |
| catboost | polars | — | 0.027 | 0.027 | 0.059 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.026 | 0.027 | 0.059 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.029 | 0.029 | 0.060 | — | — | — | 0 |
| lignum | pandas | — | 0.018 | 0.019 | 0.043 | 3.9 | 19.8 | 40.1 | 0 |
| lignum | pyarrow | — | 0.015 | 0.017 | 0.045 | — | — | — | 0 |
| lignum | polars | — | 0.016 | 0.018 | 0.046 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.003 | 0.014 | 0.014 | 0.037 | — | — | — | 0 |

## adult (32,561 rows, 6 numeric + 8 categorical columns, binary)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.142 | 0.142 | 0.336 | 146 | 524 | 654 | 0 |
| catboost | Pool (prebuilt) | 0.005 | 0.136 | 0.137 | 0.326 | — | — | — | 0 |
| catboost | polars | AttributeRemovedError: `get_categories` was removed in version 2.0; to get the distinct values present in a Categorical column, use `Expr.unique()`. For the fix | | | | | | | |
| catboost | polars (String columns) | — | 0.152 | 0.153 | 0.348 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.145 | 0.146 | 0.342 | — | — | — | 0 |
| lignum | pandas | — | 0.060 | 0.062 | 0.162 | 4.7 | 38.6 | 105 | 0 |
| lignum | pyarrow | — | 0.055 | 0.057 | 0.162 | — | — | — | 0 |
| lignum | polars | — | 0.066 | 0.067 | 0.168 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.011 | 0.054 | 0.055 | 0.161 | — | — | — | 0 |

## covtype (581,012 rows, 54 numeric + 0 categorical columns, multiclass)

| engine | input | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row, list [µs] | 1 row, DataFrame [µs] | 100 rows [µs] | mismatch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catboost | pandas | — | 0.176 | 0.176 | 0.326 | 137 | 719 | 906 | 0 |
| catboost | Pool (prebuilt) | 0.029 | 0.143 | 0.146 | 0.283 | — | — | — | 0 |
| catboost | polars | — | 0.147 | 0.151 | 0.308 | — | — | — | 0 |
| catboost | polars (String columns) | — | 0.148 | 0.150 | 0.321 | — | — | — | 0 |
| catboost | pyarrow -> to_pandas() | — | 0.164 | 0.168 | 0.324 | — | — | — | 0 |
| lignum | pandas | — | 0.049 | 0.050 | 0.135 | 4.3 | 20.6 | 73.8 | 0 |
| lignum | pyarrow | — | 0.057 | 0.057 | 0.139 | — | — | — | 0 |
| lignum | polars | — | 0.060 | 0.062 | 0.145 | — | — | — | 0 |
| lignum | float64 hashes (prebuilt) | 0.049 | 0.049 | 0.049 | 0.132 | — | — | — | 0 |
