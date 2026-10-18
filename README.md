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
- [ ] Data windows, metrics, AR baseline
- [ ] LSTNet-Skip reproduced
- [ ] Improvement (RevIN), linear comparison (DLinear), ablation
- [ ] Multi-seed runs and statistical tests
- [ ] Report

## Getting started

```bash
pip install -r requirements.txt
bash scripts/download_data.sh      # downloads into data/ (not committed)
jupyter notebook notebooks/01_explore_electricity.ipynb
```

Data source: [laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data).

## Repository layout

```
scripts/download_data.sh   download both datasets into data/
notebooks/                 data exploration
src/                       data windows, model, metrics, training (in progress)
results/figures/           plots used in the report
AI_USAGE.md                log of AI assistance, as required by the course
```
