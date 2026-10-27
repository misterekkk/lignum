# Column-name check overhead (arm64, Darwin, lignum 0.3.1)

LightGBM, 500 trees, 1 thread. Median us per predict call; overhead = median of paired (on - off).

| columns | rows | input | check off | check on | overhead |
|---:|---:|---|---:|---:|---:|
| 20 | 1 | pandas, new frame | 70.6 | 73.9 | +2.8 |
| 20 | 1 | pandas, same frame | 25.2 | 25.4 | +0.2 |
| 20 | 1 | pyarrow Table | 13.7 | 13.8 | +0.1 |
| 20 | 100 | pandas, new frame | 657.7 | 658.3 | +3.2 |
| 20 | 100 | pandas, same frame | 572.0 | 572.2 | +0.2 |
| 20 | 100 | pyarrow Table | 570.0 | 570.0 | +0.0 |
| 20 | 10,000 | pandas, new frame | 75,764.6 | 75,780.1 | +6.6 |
| 20 | 10,000 | pandas, same frame | 75,479.9 | 75,494.6 | +12.2 |
| 20 | 10,000 | pyarrow Table | 75,318.9 | 75,351.4 | +1.9 |
| 200 | 1 | pandas, new frame | 86.5 | 90.8 | +3.7 |
| 200 | 1 | pandas, same frame | 24.7 | 25.0 | +0.2 |
| 200 | 1 | pyarrow Table | 47.2 | 47.8 | +0.7 |
| 200 | 100 | pandas, new frame | 585.3 | 590.0 | +4.2 |
| 200 | 100 | pandas, same frame | 531.1 | 531.8 | +0.2 |
| 200 | 100 | pyarrow Table | 524.5 | 524.9 | +0.2 |
| 200 | 10,000 | pandas, new frame | 69,654.8 | 70,195.2 | -13.9 |
| 200 | 10,000 | pandas, same frame | 67,688.7 | 67,650.4 | -16.0 |
| 200 | 10,000 | pyarrow Table | 66,401.1 | 66,452.0 | -5.6 |
