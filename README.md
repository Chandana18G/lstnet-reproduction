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
| Solar-Energy | teammate | 52,560 × 10 min × 137 PV plants | RSE 0.4643, CORR 0.8870 |

## Status

- [x] Data downloaded and explored (`notebooks/01_explore_electricity.ipynb`)
- [x] First network: one shared neuron, test RSE 0.1032 / CORR 0.8698, one seed (`notebooks/02_first_network.ipynb`)
- [x] Data windows, metrics (with tests), AR baseline: test RSE 0.0981 / CORR 0.9231, already better than the paper's LSTNet-Skip ([experiment table](results/experiments.md))
- [x] LSTNet-Skip implemented and tested against the authors' code; training script with resumable checkpoints
- [ ] LSTNet-Skip reproduced (full run on Colab: `notebooks/03_train_on_colab.ipynb`)
- [ ] Improvement (RevIN), linear comparison (DLinear), ablation
- [ ] Multi-seed runs and statistical tests
- [ ] Report

## Getting started

```bash
pip install -r requirements.txt
bash scripts/download_data.sh      # both datasets into data/ (not committed); add 'electricity' or 'solar' for one
jupyter notebook notebooks/
pytest tests/                       # checks windows, split and metrics on toy arrays
python src/baselines.py --data electricity --horizon 24   # copy-the-past rules + AR (~8 min on CPU)
python src/neuron_baseline.py --data electricity --horizon 24 --seeds 0 1 2   # first neural baseline
python src/train.py --data electricity --horizon 24 --skip 24 --seed 0   # LSTNet (use a GPU)
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
tests/                     small hand-checkable tests
results/experiments.md     experiment table
results/figures/           plots used in the report
AI_USAGE.md                log of AI assistance, as required by the course
```
