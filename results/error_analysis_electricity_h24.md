# Error analysis: electricity, horizon 24

Forecast averaged over the seeds of each model. Shares are shares of the model's own total squared error, which is what RSE is built from.

## 1. Overall

| Metric | lstnet | lstnet_revin | lstnet_ar168 |
|---|---|---|---|
| RSE | 0.0994 | 0.1024 | 0.0965 |
| CORR | 0.9102 | 0.9195 | 0.9270 |

## 2. The 10 largest series against the rest

| Squared error | lstnet | lstnet_revin | lstnet_ar168 |
|---|---|---|---|
| 10 largest series | 4.523e+12 (98.2 %) | 4.814e+12 (98.4 %) | 4.271e+12 (98.3 %) |
| all other series | 8.376e+10 (1.8 %) | 7.653e+10 (1.6 %) | 7.195e+10 (1.7 %) |

The 10 largest series hold 54.7 % of the total test load or output.

## 2b. The single largest series (#313)

It holds 26.5 % of the total test load or output.

| | lstnet | lstnet_revin | lstnet_ar168 |
|---|---|---|---|
| Share of squared error from #313 | 85.5 % | 87.9 % | 86.7 % |
| Test RSE without #313 (seed-averaged forecast) | 0.1022 | 0.0962 | 0.0951 |
| Test RSE without #313 (mean ± std over seeds) | 0.1042 ± 0.0040 | 0.0985 ± 0.0011 | 0.0980 ± 0.0015 |
| Paired t-test against lstnet without #313 (raw p) | – | 0.0030 | 0.0002 |
| Series with smaller squared error than lstnet | – | 299 of 321 | 305 of 321 |

## 3. Series whose level changes by more than 20 % from training to test

| Squared error | lstnet | lstnet_revin | lstnet_ar168 |
|---|---|---|---|
| changed (24 series) | 9.344e+09 (0.2 %) | 7.179e+09 (0.1 %) | 6.798e+09 (0.2 %) |
| unchanged (297 series) | 4.597e+12 (99.8 %) | 4.883e+12 (99.9 %) | 4.336e+12 (99.8 %) |

## 4. Time of day (4-hour blocks)

| Hours | lstnet | lstnet_revin | lstnet_ar168 |
|---|---|---|---|
| 00–04 | 3.3 % | 3.7 % | 3.3 % |
| 04–08 | 1.5 % | 1.4 % | 1.3 % |
| 08–12 | 20.8 % | 20.1 % | 20.1 % |
| 12–16 | 28.5 % | 28.2 % | 29.1 % |
| 16–20 | 27.3 % | 27.7 % | 27.4 % |
| 20–24 | 18.6 % | 19.0 % | 18.8 % |

## 6. Worst cases of lstnet

* The worst 1 % of test steps hold 17.4 % of the squared error.
* The 5 worst test steps (file step, position in the day): 22072 (16), 22073 (17), 23249 (17), 22074 (18), 24613 (13).
* Correlation per series: median 0.943; the 5 worst series: #2 (-0.014), #317 (0.213), #299 (0.412), #113 (0.439), #121 (0.510).
