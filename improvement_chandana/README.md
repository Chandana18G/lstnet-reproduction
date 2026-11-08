# Improvement: Chandana Gurusiddappa (Electricity)

The Electricity part of the project. The data has 321 customers and one value per hour, so horizon 24 means one day ahead.
Full write-up: [`report.md`](../report.md), sections 2–6 (the Electricity parts).

## What I changed: one AR model per customer over the whole week (improvement 2)

LSTNet adds a linear AR part to the output of its neural network. In the paper this part looks at
the last 24 hours and uses the same weights for all customers. I give **every customer its own
weights over the whole 168-hour window**. Nothing else changes: same data, split, training and seeds.

**Hypothesis:** the AR part matters most on Electricity (without it RSE gets 46 % worse), and our
AR baseline with one model per customer and a long window already reaches RSE 0.0981, better than
LSTNet's 0.1008. So I expected a per-customer 168-hour AR part to lower the test RSE at horizon 24
by about 2–4 % (0.1008 → about 0.098).

The change is about 10 lines in `src/model.py` (`ar_per_series=True, ar_window=168`). In
`src/train.py` it is the model `lstnet_ar168`.

## How to run it

```bash
bash improvement_chandana/run_per_customer_ar.sh     # 10 seeds; use a GPU (about 7 min per seed on a Colab T4)
python src/stats.py --data electricity --horizon 24 --pred-dir checkpoints
python src/error_analysis.py --data electricity --horizon 24 --pred-dir checkpoints
```

On Colab: `notebooks/03_train_on_colab.ipynb`, section 5b.

## Result (test set, 10 seeds, mean ± std)

| Model | Test RSE | Test CORR | Parameters |
|---|---|---|---|
| LSTNet-Skip (reproduction) | 0.1008 ± 0.0018 | 0.9070 ± 0.0062 | 325,871 |
| **One AR model per customer, 168 h** | **0.0988 ± 0.0012** | **0.9246 ± 0.0019** | 380,095 |
| *Paper: LSTNet-Skip* | *0.1007* | *0.9119* | – |

RSE is 2 % lower, at the low end of what I expected, and the model is better on 8 of 10 seeds. The
difference is **significant** (paired t-test Holm p = 0.026, Wilcoxon p = 0.020, Diebold–Mariano
Holm p = 0.022). It is about as good as our stand-alone AR baseline (0.0981), so on Electricity the
deep part still adds little on top of a good linear model. I had the idea after seeing test results
of other models, so the gain may be slightly optimistic (report, section 6).

I also tested RevIN (improvement 1): RSE 0.1060 ± 0.0014, worse than LSTNet, but CORR is better
(0.9172). The error analysis shows that one customer (#313) makes 85.5 % of LSTNet's squared error,
and RevIN is better on 299 of 321 customers. Without #313 its RSE is better than LSTNet's. I found
this after seeing the results, so it needs testing again.

## My files

Checkpoints (`.pt`) and test predictions (`.npy`) of all Electricity runs: [Electricity, Google Drive](https://drive.google.com/drive/folders/1qYeRl8aC5b4PAMzLaBK3ZAr2Sv9-4tPO?usp=sharing).

| File | What it is |
|---|---|
| [`notebooks/01_explore_electricity.ipynb`](../notebooks/01_explore_electricity.ipynb) | Data exploration of Electricity |
| [`notebooks/02_first_network.ipynb`](../notebooks/02_first_network.ipynb) | The first network: one neuron |
| [`notebooks/03_train_on_colab.ipynb`](../notebooks/03_train_on_colab.ipynb) | All Electricity training on Colab (section 5b = this improvement) |
| [`run_per_customer_ar.sh`](run_per_customer_ar.sh) | The improvement's 10 runs, without Colab |
| [`results/runs/electricity_h24_*.json`](../results/runs) | Result of every Electricity run (50 runs) |
| [`results/stats_electricity_h24.md`](../results/stats_electricity_h24.md) | Statistical tests on Electricity |
| [`results/ar_checks_electricity.md`](../results/ar_checks_electricity.md) | The four checks of our AR baseline |
| [`results/error_analysis_electricity_h24.md`](../results/error_analysis_electricity_h24.md) | Where the Electricity errors are |
| [`results/experiments.md`](../results/experiments.md) | Electricity section of the experiment table |
| `results/figures/electricity_*.png`, `first_network_weights.png`, `error_by_time_electricity.png`, `paired_runs_electricity.png` | Electricity figures |
