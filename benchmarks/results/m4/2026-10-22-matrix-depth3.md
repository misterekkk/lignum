# Benchmark 2026-10-22T00:03:42

- lignum commit `d764ae7`, Apple M4, 10 threads, Python 3.12.15
- 1,000,000 rows × 20 features (5% NaN), 1000 trees, max depth 3; binary classification, and 10-class softmax (100 iterations × 10 trees) for `*-multi`; `lightgbm-cat`: 5 categorical features (cardinality 10, 50, 200, 1000, 5000) + 15 numeric
- versions: lignum 0.2.0.dev0, lightgbm 4.7.0, xgboost 3.4.1, treelite 4.7.2, tl2cgen 1.0.0, lleaves 1.3.0, onnxruntime 1.31.0, onnxmltools 1.16.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## lightgbm

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 2.520 | 2.688 | 15.693 | 34.7 | 20.6 | 373 | 1,433 | 0 | 0 | 0 |
| lignum | 0.010 | 0.518 | 0.531 | 3.668 | 3.7 | 3.7 | 86.3 | 233 | 0 | 0 | 0 |
| lleaves | 1.550 | 0.333 | 0.335 | 1.932 | 40.7 | 6.4 | 277 | 128 | 61,282 | 61,107 | 1.7e-15 |
| tl2cgen-tuned | 1.252 | 0.327 | 0.334 | 2.119 | 10.6 | 9.7 | 42.2 | 105 | 0 | 0 | 0 |

## xgboost

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 0.675 | 0.683 | 4.149 | 120 | 63.3 | 370 | 454 | 0 | 0 | 0 |
| lignum | 0.004 | 0.467 | 0.469 | 3.038 | 5.3 | 5.7 | 98.0 | 307 | 0 | 0 | 0 |
| lleaves | n/a (lleaves only supports LightGBM) | | | | | | | | | | | |
| tl2cgen-tuned | 1.278 | 0.544 | 0.547 | 3.454 | 21.4 | 11.5 | 150 | 280 | 39,032 | 38,698 | 1.5e-04 |
