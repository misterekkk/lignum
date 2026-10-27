# Column-name check overhead (arm64, Darwin, lignum 0.3.1)

LightGBM, 500 trees, 1 thread. Median us per predict call; overhead = median of paired (on - off).

| columns | rows | input | check off | check on | overhead |
|---:|---:|---|---:|---:|---:|
| 20 | 1 | pandas, new frame | 70.4 | 73.6 | +3.3 |
| 20 | 1 | pandas, same frame | 26.0 | 26.3 | +0.2 |
| 20 | 1 | pyarrow Table | 13.5 | 13.5 | +0.0 |
| 20 | 100 | pandas, new frame | 631.7 | 636.3 | +3.9 |
| 20 | 100 | pandas, same frame | 573.7 | 574.2 | +0.4 |
| 20 | 100 | pyarrow Table | 551.5 | 551.5 | +0.0 |
| 20 | 10,000 | pandas, new frame | 75,718.2 | 75,743.6 | +5.4 |
| 20 | 10,000 | pandas, same frame | 75,449.4 | 75,439.2 | -31.7 |
| 20 | 10,000 | pyarrow Table | 75,338.1 | 75,259.1 | -12.5 |
| 200 | 1 | pandas, new frame | 87.0 | 91.5 | +3.9 |
| 200 | 1 | pandas, same frame | 25.5 | 25.7 | +0.2 |
| 200 | 1 | pyarrow Table | 47.1 | 47.8 | +0.7 |
| 200 | 100 | pandas, new frame | 566.2 | 571.6 | +4.5 |
| 200 | 100 | pandas, same frame | 485.6 | 485.6 | +0.0 |
| 200 | 100 | pyarrow Table | 481.9 | 482.5 | +0.4 |
| 200 | 10,000 | pandas, new frame | 69,037.2 | 69,065.4 | +23.6 |
| 200 | 10,000 | pandas, same frame | 67,497.9 | 67,487.3 | -2.9 |
| 200 | 10,000 | pyarrow Table | 66,430.8 | 66,451.6 | -4.5 |
