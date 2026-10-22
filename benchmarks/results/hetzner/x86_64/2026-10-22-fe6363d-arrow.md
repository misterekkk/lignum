# Arrow input benchmark 2026-10-22T20:53:55

- x86_64, 4 threads, Python 3.12.3, 500 trees
- Time from a pyarrow Table to predictions, conversions included (best of repeats);
  `lleaves` converts with `table.to_pandas()`, `lleaves-numpy` starts from a ready float64 matrix;
  `lightgbm` predicts the Table itself (through pandas when it has dictionary columns).
- exact: bit-identical to LightGBM's predict on the DataFrame.
- 100 rows: median per call on a 100-row RecordBatch, one thread.

## california (20,640 rows, 8 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 0.049 | 0.134 | 599 | yes |
| lignum-compiled | 6.79 | 0.013 | 0.036 | 240 | yes |
| lleaves | 41.53 | 0.024 | 0.056 | 756 | no (19105 rows) |
| lleaves-numpy | 41.53 | 0.023 | 0.056 | 360 | no (19105 rows) |
| lightgbm | 0.00 | 0.168 | 0.440 | 2,484 | yes |

## covtype (581,012 rows, 54 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 1.190 | 2.920 | 477 | yes |
| lignum-compiled | 5.46 | 0.252 | 0.592 | 187 | yes |
| lleaves | 39.51 | 0.365 | 0.771 | 1,152 | no (581012 rows) |
| lleaves-numpy | 39.51 | 0.299 | 0.712 | 150 | no (581012 rows) |
| lightgbm | 0.00 | 4.291 | 7.768 | 1,545 | yes |

## adult (32,561 rows, 14 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 0.111 | 0.351 | 1,093 | yes |
| lignum-compiled | 6.53 | 0.026 | 0.082 | 333 | yes |
| lleaves | 47.97 | 0.046 | 0.122 | 10,387 | no (26378 rows) |
| lleaves-numpy | 47.97 | 0.040 | 0.117 | 489 | no (26378 rows) |
| lightgbm | 0.00 | 0.482 | 1.417 | 8,808 | yes |

## bank (45,211 rows, 16 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 0.125 | 0.373 | 753 | yes |
| lignum-compiled | 6.27 | 0.035 | 0.084 | 199 | yes |
| lleaves | 50.48 | 0.048 | 0.110 | 9,416 | no (40873 rows) |
| lleaves-numpy | 50.48 | 0.045 | 0.104 | 211 | no (40873 rows) |
| lightgbm | 0.00 | 0.580 | 1.389 | 6,916 | yes |

## diabetes (101,766 rows, 47 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 0.481 | 1.404 | 1,399 | yes |
| lignum-compiled | 6.05 | 0.149 | 0.482 | 559 | yes |
| lleaves | 241.68 | 0.864 | 2.111 | 16,819 | no (91570 rows) |
| lleaves-numpy | 241.68 | 0.831 | 2.081 | 2,287 | no (91570 rows) |
| lightgbm | 0.00 | 2.500 | 7.775 | 21,850 | yes |

