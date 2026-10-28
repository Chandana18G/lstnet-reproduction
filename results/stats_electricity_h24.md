# Results over seeds: electricity, horizon 24

| Model | Seeds | Test RSE (mean ± std) | Test CORR (mean ± std) |
|---|---|---|---|
| lstnet | 10 | 0.1008 ± 0.0018 | 0.9070 ± 0.0062 |
| lstnet_revin | 10 | 0.1060 ± 0.0014 | 0.9172 ± 0.0010 |
| lstnet_noar | 10 | 0.1476 ± 0.0096 | 0.8617 ± 0.0043 |
| dlinear | 10 | 0.1035 ± 0.0003 | 0.9052 ± 0.0003 |

## Each model vs LSTNet

Negative difference / DM statistic = the model is better (smaller error) than LSTNet.
p-values after Holm correction; the raw t-test and Wilcoxon p-values are in brackets.

| Model | Paired seeds | Mean RSE difference | Paired t-test p | Wilcoxon p | DM statistic | DM p |
|---|---|---|---|---|---|---|
| lstnet_revin | 10 | +0.0052 | 0.0001 (0.0000) | (0.0020) | +1.39 | 0.3290 |
| lstnet_noar | 10 | +0.0468 | 0.0000 (0.0000) | (0.0020) | +9.37 | 0.0000 |
| dlinear | 10 | +0.0027 | 0.0017 (0.0008) | (0.0059) | +1.85 | 0.1936 |
| AR (baselines.py) | 0 | – | – | – | -0.88 | 0.3773 |

*Produced by `src/stats.py` in `notebooks/03_train_on_colab.ipynb` (cell 6) on 28.10.2026, from the
40 runs in Google Drive (`lstnet-checkpoints`). Copied here from the Colab output.*
