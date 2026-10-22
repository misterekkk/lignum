# Benchmark 2026-10-22T00:10:32

- lignum commit `f95fd12`, Apple M4, 10 threads, Python 3.12.15
- 1,000,000 rows × 20 features (5% NaN), 500 trees, max depth 10; binary classification, and 10-class softmax (50 iterations × 10 trees) for `*-multi`; `lightgbm-cat`: 5 categorical features (cardinality 10, 50, 200, 1000, 5000) + 15 numeric
- versions: lignum 0.2.0.dev0, lightgbm 4.7.0, xgboost 3.4.1, treelite 4.7.2, tl2cgen 1.0.0, lleaves 1.3.0, onnxruntime 1.31.0, onnxmltools 1.16.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## xgboost

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 1.440 | 1.492 | 9.382 | 105 | 59.8 | 690 | 1,415 | 0 | 0 | 0 |
| lignum | 0.046 | 0.715 | 0.721 | 4.926 | 7.9 | 7.9 | 165 | 482 | 0 | 0 | 0 |
| lleaves | n/a (lleaves only supports LightGBM) | | | | | | | | | | | |
| tl2cgen-tuned | 14.587 | 1.764 | 1.770 | 11.710 | 12.8 | 12.1 | 236 | 934 | 29,833 | 28,233 | 7.5e-05 |
