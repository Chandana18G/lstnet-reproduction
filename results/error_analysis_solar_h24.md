# Error analysis: solar, horizon 24

Forecast averaged over the seeds of each model. Shares are shares of the model's own total squared error, which is what RSE is built from.

## 1. Overall

| Metric | lstnet | lstnet_revin | lstnet_noar | lstnet_skip144 |
|---|---|---|---|---|
| RSE | 0.4444 | 0.4817 | 0.4328 | 0.4405 |
| CORR | 0.8961 | 0.8694 | 0.9009 | 0.8944 |

## 2. The 10 largest series against the rest

| Squared error | lstnet | lstnet_revin | lstnet_noar | lstnet_skip144 |
|---|---|---|---|---|
| 10 largest series | 7.474e+06 (33.0 %) | 8.022e+06 (30.2 %) | 7.082e+06 (33.0 %) | 7.082e+06 (31.8 %) |
| all other series | 1.517e+07 (67.0 %) | 1.857e+07 (69.8 %) | 1.439e+07 (67.0 %) | 1.516e+07 (68.2 %) |

The 10 largest series hold 16.4 % of the total test load or output.

## 2b. The single largest series (#5)

It holds 2.0 % of the total test load or output.

| | lstnet | lstnet_revin | lstnet_noar | lstnet_skip144 |
|---|---|---|---|---|
| Share of squared error from #5 | 4.9 % | 4.7 % | 4.9 % | 4.9 % |
| Test RSE without #5 (seed-averaged forecast) | 0.4454 | 0.4830 | 0.4336 | 0.4413 |
| Test RSE without #5 (mean ± std over seeds) | 0.4766 ± 0.0252 | 0.5068 ± 0.0135 | 0.4538 ± 0.0103 | 0.4560 ± 0.0108 |
| Paired t-test against lstnet without #5 (raw p) | – | 0.0211 | 0.0077 | 0.0706 |
| Series with smaller squared error than lstnet | – | 10 of 137 | 136 of 137 | 62 of 137 |

## 3. Series whose level changes by more than 20 % from training to test

| Squared error | lstnet | lstnet_revin | lstnet_noar | lstnet_skip144 |
|---|---|---|---|---|
| changed (123 series) | 1.892e+07 (83.5 %) | 2.227e+07 (83.7 %) | 1.784e+07 (83.1 %) | 1.858e+07 (83.5 %) |
| unchanged (14 series) | 3.727e+06 (16.5 %) | 4.326e+06 (16.3 %) | 3.634e+06 (16.9 %) | 3.668e+06 (16.5 %) |

## 4. Time of day (4-hour blocks)

| Hours | lstnet | lstnet_revin | lstnet_noar | lstnet_skip144 |
|---|---|---|---|---|
| 00–04 | 0.0 % | 0.1 % | 0.1 % | 0.0 % |
| 04–08 | 4.2 % | 3.7 % | 4.0 % | 3.3 % |
| 08–12 | 57.7 % | 61.2 % | 58.4 % | 59.7 % |
| 12–16 | 36.8 % | 32.6 % | 36.4 % | 36.0 % |
| 16–20 | 1.2 % | 2.4 % | 1.1 % | 0.9 % |
| 20–24 | 0.1 % | 0.1 % | 0.0 % | 0.0 % |

## 5. Steps where all series are 0 (56.2 % of test steps)

| | lstnet | lstnet_revin | lstnet_noar | lstnet_skip144 |
|---|---|---|---|---|
| Share of squared error | 0.6 % | 0.7 % | 0.5 % | 0.4 % |

## 6. Worst cases of lstnet

* The worst 1 % of test steps hold 13.1 % of the squared error.
* The 5 worst test steps (file step, position in the day): 47880 (72), 47879 (71), 47881 (73), 47878 (70), 47882 (74).
* Correlation per series: median 0.897; the 5 worst series: #135 (0.864), #73 (0.865), #68 (0.875), #41 (0.877), #2 (0.878).
