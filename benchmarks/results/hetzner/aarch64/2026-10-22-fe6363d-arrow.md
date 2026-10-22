# Arrow input benchmark 2026-10-22T21:53:47

- aarch64, 4 threads, Python 3.12.3, 500 trees
- Time from a pyarrow Table to predictions, conversions included (best of repeats);
  `lleaves` converts with `table.to_pandas()`, `lleaves-numpy` starts from a ready float64 matrix;
  `lightgbm` predicts the Table itself (through pandas when it has dictionary columns).
- exact: bit-identical to LightGBM's predict on the DataFrame.
- 100 rows: median per call on a 100-row RecordBatch, one thread.

## california (20,640 rows, 8 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 0.054 | 0.203 | 1,111 | yes |
| lignum-compiled | 12.22 | 0.014 | 0.043 | 340 | yes |
| lleaves | 47.94 | 0.022 | 0.074 | 1,383 | no (19105 rows) |
| lleaves-numpy | 47.94 | 0.022 | 0.073 | 565 | no (19105 rows) |
| lightgbm | 0.00 | 0.421 | 1.161 | 6,425 | yes |

## covtype (581,012 rows, 54 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 1.250 | 4.715 | 899 | yes |
| lignum-compiled | 10.72 | 0.242 | 0.936 | 293 | yes |
| lleaves | 49.66 | 0.412 | 1.215 | 1,574 | no (581012 rows) |
| lleaves-numpy | 49.66 | 0.293 | 1.097 | 211 | no (581012 rows) |
| lightgbm | 0.00 | 10.120 | 26.197 | 4,881 | yes |

## adult (32,561 rows, 14 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 0.129 | 0.476 | 1,596 | yes |
| lignum-compiled | 11.12 | 0.024 | 0.096 | 422 | yes |
| lleaves | 56.40 | 0.047 | 0.137 | 8,238 | no (26361 rows) |
| lleaves-numpy | 56.40 | 0.036 | 0.127 | 599 | no (26361 rows) |
| lightgbm | 0.00 | 1.121 | 3.148 | 18,738 | yes |

## bank (45,211 rows, 16 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 0.136 | 0.515 | 1,170 | yes |
| lignum-compiled | 10.78 | 0.033 | 0.109 | 292 | yes |
| lleaves | 55.60 | 0.047 | 0.137 | 7,934 | no (40816 rows) |
| lleaves-numpy | 55.60 | 0.037 | 0.126 | 331 | no (40816 rows) |
| lightgbm | 0.00 | 1.424 | 3.862 | 16,823 | yes |

## diabetes (101,766 rows, 47 columns)

| engine | prep [s] | all threads [s] | 1 thread [s] | 100 rows [µs] | exact |
|---|---:|---:|---:|---:|---:|
| lignum | 0.00 | 0.525 | 1.985 | 2,144 | yes |
| lignum-compiled | 9.08 | 0.143 | 0.568 | 678 | yes |
| lleaves | 478.75 | 0.892 | 3.263 | 31,453 | no (91570 rows) |
| lleaves-numpy | 478.75 | 0.855 | 3.217 | 3,643 | no (91570 rows) |
| lightgbm | 0.00 | 6.006 | 18.452 | 47,751 | yes |

