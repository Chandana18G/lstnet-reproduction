# Improvement: Dineshkumar Swaminathan (Solar-Energy)

The Solar-Energy part of the project. The data has 137 PV plants and one value every 10 minutes, so one day is 144 steps and horizon 24 means 4 hours ahead.
Full write-up: [`report.md`](../report.md), sections 2–6 (the "Solar-Energy (Dinesh)" parts).

## What I changed: a one-day skip (improvement 2)

LSTNet's skip-GRU links each moment with the moment `p` steps earlier. The paper uses p = 24 and
tried only 2–64. On Solar, 24 steps are only 4 hours, so the model never compares a moment with
the same time yesterday. I set **p = 144 (one day)**. A skip sequence has `(q − 6) // p` steps, so
with the paper's window q = 168 it would have only 1 step. I therefore use **q = 438** (3 days + 6),
which gives 3 steps per sequence: the same time today, yesterday and the day before.

**Hypothesis:** Solar repeats itself every day (median autocorrelation 0.85 at 144 steps, only
0.32 at 24 steps), and our AR baseline got much better once its window reached one day back. So I
expected a one-day skip to lower the test RSE at horizon 24 by about 2–5 % (0.4753 → about 0.455).

No new model code was needed: the skip and the window are already options of `src/model.py`
and `src/train.py`.

## How to run it

```bash
bash improvement_dinesh/run_day_skip.sh     # 10 seeds; use a GPU (about 20 min per seed on a Colab T4)
python src/stats.py --data solar --horizon 24 --pred-dir checkpoints
python src/error_analysis.py --data solar --horizon 24 --pred-dir checkpoints
```

On Colab: `notebooks/03_train_on_colab_solar.ipynb`, section 5b.

## Result (test set, 10 seeds, mean ± std)

| Model | Test RSE | Test CORR | Parameters |
|---|---|---|---|
| LSTNet-Skip (reproduction, p = 24, q = 168) | 0.4753 ± 0.0249 | 0.8835 ± 0.0098 | 193,002 |
| **One-day skip (p = 144, q = 438)** | **0.4551 ± 0.0107** | **0.8881 ± 0.0057** | 357,402 |
| *Paper: LSTNet-Skip* | *0.4643* | *0.8870* | – |

The mean RSE is 4 % lower and the spread over seeds is less than half, but the model is better
on only 5 of 10 seeds. The difference is **not significant** (paired t-test Holm p = 0.147,
Wilcoxon p = 0.23, Diebold–Mariano p = 0.57). The model is also bigger and sees a longer window,
so the small gain may not come from the skip itself.

I also tested RevIN on Solar (improvement 1, the same code as on Electricity): RSE 0.5055 ± 0.0137,
worse than LSTNet.

## My files

| File | What it is |
|---|---|
| [`notebooks/01_explore_solar.ipynb`](../notebooks/01_explore_solar.ipynb) | Data exploration of Solar |
| [`notebooks/03_train_on_colab_solar.ipynb`](../notebooks/03_train_on_colab_solar.ipynb) | All Solar training on Colab (section 5b = this improvement) |
| [`run_day_skip.sh`](run_day_skip.sh) | The improvement's 10 runs, without Colab |
| [`results/runs/solar_h24_*.json`](../results/runs) | Result of every Solar run (50 runs) |
| [`results/stats_solar_h24.md`](../results/stats_solar_h24.md) | Statistical tests on Solar |
| [`results/error_analysis_solar_h24.md`](../results/error_analysis_solar_h24.md) | Where the Solar errors are |
| [`results/experiments.md`](../results/experiments.md) | Solar section of the experiment table |
| `results/figures/solar_*.png`, `error_by_time_solar.png` | Solar figures |

In the shared code I added the run names of the one-day skip to `src/stats.py` and
`src/error_analysis.py`.
