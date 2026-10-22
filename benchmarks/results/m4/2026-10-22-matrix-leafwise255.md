# Benchmark 2026-10-22T00:04:55

- lignum commit `acedf42`, Apple M4, 10 threads, Python 3.12.15
- 1,000,000 rows × 20 features (5% NaN), 500 trees, max depth -1, num_leaves 255; binary classification, and 10-class softmax (50 iterations × 10 trees) for `*-multi`; `lightgbm-cat`: 5 categorical features (cardinality 10, 50, 200, 1000, 5000) + 15 numeric
- versions: lignum 0.2.0.dev0, lightgbm 4.7.0, xgboost 3.4.1, treelite 4.7.2, tl2cgen 1.0.0, lleaves 1.3.0, onnxruntime 1.31.0, onnxmltools 1.16.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## lightgbm

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 8.206 | 8.452 | 65.352 | 52.5 | 43.1 | 833 | 5,928 | 0 | 0 | 0 |
| lignum | 0.072 | 1.122 | 1.154 | 8.222 | 8.9 | 8.9 | 174 | 610 | 0 | 0 | 0 |
| lleaves | 187.381 | 1.190 | 1.201 | 8.706 | 36.9 | 7.1 | 367 | 529 | 47,631 | 45,877 | 8.9e-16 |
| tl2cgen-tuned | 7.977 | 1.534 | 1.560 | 10.836 | 12.2 | 11.5 | 206 | 821 | 0 | 0 | 0 |
