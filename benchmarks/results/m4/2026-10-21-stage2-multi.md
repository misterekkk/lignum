# Benchmark 2026-10-21T23:09:37

- lignum commit `c4e9965`, Apple M4, 10 threads, Python 3.12.15
- 1,000,000 rows × 20 features (5% NaN), 1000 trees, max depth 6; binary classification, and 10-class softmax (100 iterations × 10 trees) for `*-multi`
- versions: lignum 0.2.0.dev0, lightgbm 4.7.0, xgboost 3.4.1, treelite 4.7.2, tl2cgen 1.0.0, lleaves 1.3.0, onnxruntime 1.31.0, onnxmltools 1.16.0

Batch = full data set, all threads (best / median of repeats); 1 thread is normalised to the full data set.
Latency = median per call. Mismatch = rows not bit-identical to the native library on held-out rows / training rows (training rows hit split thresholds exactly).

## lightgbm-multi

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 4.927 | 4.965 | 31.091 | 44.7 | 35.1 | 632 | 2,910 | 0 | 0 | 0 |
| lignum | 0.033 | 0.773 | 0.810 | 5.180 | 9.0 | 9.0 | 169 | 521 | 0 | 0 | 0 |
| lignum-bin | 0.027 | 0.766 | 0.808 | 5.139 | 9.5 | 9.5 | 162 | 518 | 0 | 0 | 0 |
| tl2cgen | 7.706 | 2.737 | 2.899 | 20.078 | 25.4 | 14.7 | 456 | 1,759 | 0 | 0 | 0 |
| tl2cgen-tuned | 6.359 | 2.262 | 2.404 | 15.729 | 13.8 | 12.7 | 297 | 1,407 | 0 | 0 | 0 |
| gtil | 0.077 | 13.135 | 13.186 | 67.901 | 154 | 80.8 | 1,630 | 7,002 | 100,000 | 100,000 | 6.0e-08 |
| lleaves | 27.445 | 0.950 | 0.973 | 6.863 | 38.5 | 8.4 | 363 | 536 | 100,000 | 100,000 | 1.4e-15 |
| onnxruntime | 1.167 | 8.911 | 9.139 | 25.472 | 20.0 | 21.2 | 488 | 2,474 | 100,000 | 100,000 | 5.1e-04 |

## xgboost-multi

| engine | prep [s] | batch best [s] | batch median [s] | 1 thread [s] | 1 row [µs] | 1 row, 1 thr [µs] | 100 rows [µs] | 100 rows, 1 thr [µs] | mismatch bench | mismatch train | max abs diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| native | 0.000 | 1.415 | 1.432 | 8.451 | 147 | 73.2 | 643 | 950 | 0 | 0 | 0 |
| lignum | 0.021 | 0.754 | 0.764 | 5.148 | 9.6 | 9.6 | 162 | 520 | 0 | 0 | 0 |
| lignum-bin | 0.021 | 0.756 | 0.763 | 5.134 | 9.0 | 9.0 | 163 | 519 | 0 | 0 | 0 |
| tl2cgen | 4.316 | 3.979 | 4.152 | 27.826 | 22.2 | 21.1 | 740 | 2,691 | 94,341 | 93,995 | 2.4e-07 |
| tl2cgen-tuned | 7.878 | 2.169 | 2.208 | 13.846 | 24.5 | 13.0 | 280 | 1,388 | 94,341 | 93,995 | 1.0e-02 |
| gtil | 0.042 | 18.464 | 19.110 | 71.960 | 144 | 81.3 | 2,117 | 7,238 | 0 | 0 | 0 |
| lleaves | n/a (lleaves only supports LightGBM) | | | | | | | | | | | |
| onnxruntime | 0.498 | 20.242 | 21.281 | 25.192 | 19.8 | 20.9 | 854 | 2,455 | 99,998 | 99,999 | 7.2e-07 |

## Notes

- Multiclass labels depend on all 20 features; top-4 features take 50% of the LightGBM splits.
- `2026-10-21-m4-stage2-multi-4features.md` is the same benchmark with labels built from 4 features
  (86% of splits on 4 features): there lleaves wins LightGBM multiclass (0.655 s vs 0.746 s) because
  its compiled branches become highly predictable. lignum's branch-free traversal stays at
  0.75–0.77 s in both cases; lleaves moves between 0.66 and 0.95 s.
- `tl2cgen-tuned` (quantize=1) changes paths on XGBoost multiclass (max diff 1.0e-2).
