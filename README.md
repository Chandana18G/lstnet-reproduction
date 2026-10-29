# LSTNet reproduction: Electricity and Solar-Energy forecasting

Course project for *Deep Learning* (M.Sc. Applied Data Science and AI, SRH University Munich,
winter semester 2026/27).

We reproduce one result of **LSTNet** (Lai, Chang, Yang, Liu: *Modeling Long- and Short-Term
Temporal Patterns with Deep Neural Networks*, SIGIR 2018,
[arXiv 1703.07015](https://arxiv.org/abs/1703.07015)) and test an improvement with a proper
statistical comparison.

**Question:** the paper argues that LSTNet needs its linear autoregressive part because neural
networks cannot follow changes in scale. Can a simple linear model match LSTNet, and does
handling scale shift directly (RevIN) make the deep part worth its complexity?

| Dataset | Owner | Size | Target (Table 2, horizon 24, LSTNet-Skip) |
|---|---|---|---|
| Electricity | Chandana Gurusiddappa | 26,304 hours × 321 customers | RSE 0.1007, CORR 0.9119 |
| Solar-Energy | Dinesh | 52,560 × 10 min × 137 PV plants | RSE 0.4643, CORR 0.8870 |

## Who does what

| | Chandana Gurusiddappa | Dinesh |
|---|---|---|
| Dataset | Electricity | Solar-Energy |
| Reproduction | LSTNet-Skip on Electricity, horizon 24 | LSTNet-Skip on Solar-Energy, horizon 24 |
| Own improvement | RevIN against level shifts, compared with linear models (AR, DLinear) | Skip length of one day (p = 144) with a longer window; RevIN for the winter test period |
| Shared code | data loading, metrics, baselines, model, training (both review and can explain all of it) | |

## Results (test set, horizon 24, mean ± std over 10 seeds)

| Model | Electricity RSE | Electricity CORR | Solar RSE | Solar CORR |
|---|---|---|---|---|
| *Paper: LSTNet-Skip* | *0.1007* | *0.9119* | *0.4643* | *0.8870* |
| **Ours: LSTNet-Skip (reproduction)** | **0.1008 ± 0.0018** | 0.9070 ± 0.0062 | **0.4753 ± 0.0249** | 0.8835 ± 0.0098 |
| + RevIN (improvement) | 0.1060 ± 0.0014 | 0.9172 ± 0.0010 | 0.5055 ± 0.0137 | 0.8578 ± 0.0060 |
| without AR part (paper's ablation) | 0.1476 ± 0.0096 | 0.8617 ± 0.0043 | 0.4529 ± 0.0104 | 0.8922 ± 0.0051 |
| DLinear | 0.1035 ± 0.0003 | 0.9052 ± 0.0003 | 0.5740 ± 0.0011 | 0.8167 ± 0.0009 |
| AR, one linear model per series | 0.0981 | 0.9231 | 0.5247 | 0.8462 |

**In short:** both reproductions are close to the paper. On Electricity a simple linear model is as
good as LSTNet; on Solar LSTNet clearly beats the linear models. RevIN did not lower RSE on either
dataset. Statistical tests: [`results/stats_electricity_h24.md`](results/stats_electricity_h24.md),
[`results/stats_solar_h24.md`](results/stats_solar_h24.md); full write-up: [`report.md`](report.md).

## Status

- [x] Data downloaded and explored (Electricity and Solar notebooks 01)
- [x] Baselines: copy-the-past rules, one-neuron model (3 seeds), AR per series ([experiment table](results/experiments.md))
- [x] LSTNet-Skip implemented, tested against the authors' code, reproduced on both datasets (10 seeds)
- [x] Improvement (RevIN), DLinear and the "without AR" ablation, 10 seeds each, with statistical tests
- [x] Checks of the AR baseline ([`results/ar_checks_electricity.md`](results/ar_checks_electricity.md))
- [ ] Error analysis (report section 6), final report and AI-use section

## Getting started

```bash
pip install -r requirements.txt
bash scripts/download_data.sh      # both datasets into data/ (not committed); add 'electricity' or 'solar' for one
jupyter notebook notebooks/
pytest tests/                       # checks windows, split and metrics on toy arrays
python src/baselines.py --data electricity --horizon 24   # copy-the-past rules + AR (~8 min on CPU)
python src/neuron_baseline.py --data electricity --horizon 24 --seeds 0 1 2   # first neural baseline
python src/train.py --data electricity --horizon 24 --model lstnet --seed 0   # or lstnet_revin, lstnet_noar, dlinear (use a GPU)
python src/stats.py --data electricity --horizon 24 --pred-dir checkpoints     # table over seeds + statistical tests
```

Data source: [laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data).

## Repository layout

```
scripts/download_data.sh   download both datasets into data/
notebooks/                 data exploration, first network
src/data.py                loading, scaling, split, windows
src/metrics.py             RSE, CORR, per-window loss
src/baselines.py           copy-the-past rules and AR (one model per series)
src/neuron_baseline.py     first neural baseline (one neuron), several seeds
src/model.py               LSTNet-Skip
src/train.py               training and evaluation, resumable
src/stats.py               results over seeds and statistical tests
report/                    presentation and report outlines
tests/                     small hand-checkable tests
results/experiments.md     experiment table
results/figures/           plots used in the report
AI_USAGE.md                log of AI assistance, as required by the course
```
