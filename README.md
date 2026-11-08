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
| Solar-Energy | Dineshkumar Swaminathan | 52,560 × 10 min × 137 PV plants | RSE 0.4643, CORR 0.8870 |

## Who does what

| | Chandana Gurusiddappa | Dineshkumar Swaminathan |
|---|---|---|
| Dataset | Electricity | Solar-Energy |
| Exploration | `notebooks/01_explore_electricity.ipynb`, `02_first_network.ipynb` | `notebooks/01_explore_solar.ipynb` |
| Reproduction | LSTNet-Skip on Electricity, horizon 24 (`notebooks/03_train_on_colab.ipynb`) | LSTNet-Skip on Solar-Energy, horizon 24 (`notebooks/03_train_on_colab_solar.ipynb`) |
| Own improvement | 1. RevIN against level shifts; 2. one AR model per customer over 168 h | 1. RevIN for the winter test period; 2. a one-day skip (p = 144) with a 3-day window |
| Improvement folder | [`improvement_chandana/`](improvement_chandana) | [`improvement_dinesh/`](improvement_dinesh) |
| Comparisons, statistics, error analysis | Electricity | Solar-Energy |
| Shared code (`src/`, `tests/`) | wrote it | added the one-day skip to `stats.py` and `error_analysis.py` |
| Report | Electricity parts, shared sections | Solar-Energy parts |

Both of us have reviewed all the shared code and can explain it.

## Results (test set, horizon 24, mean ± std over 10 seeds)

| Model | Electricity RSE | Electricity CORR | Solar RSE | Solar CORR |
|---|---|---|---|---|
| *Paper: LSTNet-Skip* | *0.1007* | *0.9119* | *0.4643* | *0.8870* |
| **Ours: LSTNet-Skip (reproduction)** | **0.1008 ± 0.0018** | 0.9070 ± 0.0062 | **0.4753 ± 0.0249** | 0.8835 ± 0.0098 |
| + RevIN (improvement 1) | 0.1060 ± 0.0014 | 0.9172 ± 0.0010 | 0.5055 ± 0.0137 | 0.8578 ± 0.0060 |
| + per-series AR over 168 h (improvement 2) | **0.0988 ± 0.0012** | **0.9246 ± 0.0019** | – | – |
| one-day skip, p = 144, q = 438 (Solar improvement 2) | – | – | 0.4551 ± 0.0107 | 0.8881 ± 0.0057 |
| without AR part (paper's ablation) | 0.1476 ± 0.0096 | 0.8617 ± 0.0043 | 0.4529 ± 0.0104 | 0.8922 ± 0.0051 |
| DLinear | 0.1035 ± 0.0003 | 0.9052 ± 0.0003 | 0.5740 ± 0.0011 | 0.8167 ± 0.0009 |
| AR, one linear model per series | 0.0981 | 0.9231 | 0.5247 | 0.8462 |

**In short:** both reproductions are close to the paper. On Electricity a simple linear model is as
good as LSTNet; on Solar LSTNet clearly beats the linear models. RevIN did not lower RSE on either
dataset. On Solar a one-day skip lowered the mean RSE by 4 %, but not significantly (p = 0.15). Giving LSTNet one AR model per customer over the whole week improved Electricity by 2 %
(significant over seeds and over test hours, p ≈ 0.02). One customer (#313) makes 85 % of the
squared error on Electricity, and without it RevIN is better than LSTNet too (see report, section 6). Statistical tests: [`results/stats_electricity_h24.md`](results/stats_electricity_h24.md),
[`results/stats_solar_h24.md`](results/stats_solar_h24.md); full write-up: [`report.md`](report.md).

## Results and checkpoints

- Test result of every run: [`results/runs/`](results/runs) (one `.json` per model and seed).
- Best checkpoints (`.pt`) and test predictions (`.npy`) are too big for GitHub:
  [Electricity, Google Drive](https://drive.google.com/drive/folders/1qYeRl8aC5b4PAMzLaBK3ZAr2Sv9-4tPO?usp=sharing).

## Status

- [x] Data downloaded and explored (Electricity and Solar notebooks 01)
- [x] Baselines: copy-the-past rules, one-neuron model (3 seeds), AR per series ([experiment table](results/experiments.md))
- [x] LSTNet-Skip implemented, tested against the authors' code, reproduced on both datasets (10 seeds)
- [x] Improvement 1 (RevIN), DLinear and the "without AR" ablation, 10 seeds each, with statistical tests
- [x] Improvement 2 on Electricity (per-customer AR over 168 h), 10 seeds, with statistical tests
- [x] Checks of the AR baseline ([`results/ar_checks_electricity.md`](results/ar_checks_electricity.md))
- [x] Error analysis on Electricity (report section 6)
- [x] Improvement 2 on Solar-Energy (one-day skip), 10 seeds, with statistical tests
- [x] Error analysis on Solar-Energy, final report and AI-use section

## Getting started

```bash
git clone https://github.com/Chandana18G/lstnet-reproduction.git
cd lstnet-reproduction
pip install -r requirements.txt
bash scripts/download_data.sh      # both datasets into data/ (not committed); add 'electricity' or 'solar' for one
jupyter notebook notebooks/
pytest tests/                       # checks windows, split and metrics on toy arrays
python src/baselines.py --data electricity --horizon 24   # copy-the-past rules + AR (~8 min on CPU)
python src/neuron_baseline.py --data electricity --horizon 24 --seeds 0 1 2   # first neural baseline
python src/train.py --data electricity --horizon 24 --model lstnet --seed 0   # or lstnet_revin, lstnet_ar168, lstnet_noar, dlinear (use a GPU)
python src/stats.py --data electricity --horizon 24 --pred-dir checkpoints     # table over seeds + statistical tests
python src/error_analysis.py --data electricity --horizon 24 --pred-dir checkpoints   # where the errors are
```

Data source: [laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data).
On Windows, run the download script in Git Bash. Or download
[electricity.txt.gz](https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz)
and [solar_AL.txt.gz](https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz)
by hand and put them, still zipped, into a folder `data/`.

Training was done on Google Colab: `notebooks/03_train_on_colab.ipynb` (Electricity) and
`notebooks/03_train_on_colab_solar.ipynb` (Solar-Energy).

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
src/error_analysis.py      where the test errors are (report section 6)
src/check_ar.py            the four checks of our AR baseline
src/save_split.py          writes the fixed split to results/splits.json
report.md                  the report
improvement_chandana/      Chandana's improvement (Electricity)
improvement_dinesh/        Dinesh's improvement (Solar-Energy)
tests/                     small hand-checkable tests
results/experiments.md     experiment table
results/figures/           plots used in the report
AI_USAGE.md                log of AI assistance, as required by the course
```
