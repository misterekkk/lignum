# Benchmark 2026-10-21T23:43:22

- lignum commit `2c8d131`, Apple M4, 10 threads, Python 3.12.15
- 1,000,000 rows × 20 features (5% NaN), 1000 trees, max depth 6; binary classification, and 10-class softmax (100 iterations × 10 trees) for `*-multi`; `lightgbm-cat`: 5 categorical features (cardinality 10, 50, 200, 1000, 5000) + 15 numeric
- versions: lignum 0.2.0.dev0, lightgbm 4.7.0, xgboost 3.4.1, treelite 4.7.2, tl2cgen 1.0.0, lleaves 1.3.0, onnxruntime 1.31.0, onnxmltools 1.16.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## lightgbm-cat

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 8.410 | 8.494 | 58.865 | 41.3 | 30.0 | 957 | 5,547 | 0 | 0 | 0 |
| lignum | 0.049 | 2.953 | 2.973 | 19.943 | 16.3 | 15.0 | 390 | 2,012 | 0 | 0 | 0 |
| lignum-bin | 0.048 | 2.940 | 2.949 | 19.929 | 16.2 | 16.2 | 389 | 2,015 | 0 | 0 | 0 |
| tl2cgen | 13.031 | 6.696 | 6.774 | 40.189 | 20.4 | 18.8 | 1,086 | 3,903 | 0 | 0 | 0 |
| tl2cgen-tuned | 15.247 | 5.358 | 5.452 | 36.054 | 19.7 | 18.8 | 775 | 3,511 | 0 | 0 | 0 |
| gtil | 0.100 | 43.261 | 44.021 | 253.973 | 263 | 197 | 5,429 | 25,074 | 0 | 0 | 0 |
| lleaves | 67.841 | 2.092 | 2.106 | 12.403 | 123 | 19.1 | 796 | 1,035 | 67,974 | 66,823 | 2.0e-15 |
| onnxruntime | 9.020 | 45.094 | 45.983 | 275.495 | 65.6 | 326 | 4,467 | 28,353 | 100,000 | 100,000 | 8.9e-04 |

## Notes

- Model: 1000 trees, 36,897 numeric / 6,019 mask / 12,149 bitset splits; bitsets total ~1.4 MB.
- Same data and tree shape without categorical features (100k rows, 200 trees): 0.014 s vs 0.049 s
  with them. Experiments put the overhead in the categorical comparison
  itself: float→int conversion plus a dependent bitset load (mostly L2) lengthen every step that
  hits a categorical slot. lleaves embeds bitsets as constants and is 1.4× faster on this model.
