# Results over seeds: solar, horizon 24

| Model | Seeds | Test RSE (mean ± std) | Test CORR (mean ± std) |
|---|---|---|---|
| lstnet | 10 | 0.4753 ± 0.0249 | 0.8835 ± 0.0098 |
| lstnet_revin | 10 | 0.5055 ± 0.0137 | 0.8578 ± 0.0060 |
| lstnet_noar | 10 | 0.4529 ± 0.0104 | 0.8922 ± 0.0051 |
| dlinear | 10 | 0.5740 ± 0.0011 | 0.8167 ± 0.0009 |

## Each model vs LSTNet

Negative difference / DM statistic = the model is better (smaller error) than LSTNet.
p-values after Holm correction; the raw t-test and Wilcoxon p-values are in brackets.

| Model | Paired seeds | Mean RSE difference | Paired t-test p | Wilcoxon p | DM statistic | DM p |
|---|---|---|---|---|---|---|
| lstnet_revin | 10 | +0.0302 | 0.0414 (0.0207) | (0.0273) | +2.12 | 0.0337 |
| lstnet_noar | 10 | -0.0224 | 0.0224 (0.0075) | (0.0059) | -2.49 | 0.0259 |
| dlinear | 10 | +0.0987 | 0.0000 (0.0000) | (0.0020) | +4.70 | 0.0000 |
| AR (baselines.py) | 0 | – | – | – | +5.81 | 0.0000 |

*Produced by `src/stats.py` in the Colab notebook (cell 6, `DATA = "solar"`, run by Dinesh), from
the 40 Solar runs in his Google Drive. Copied here from the Colab output (29.10.2026).*
