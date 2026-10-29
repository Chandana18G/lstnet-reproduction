# Experiment table: Electricity, horizon 24

Test set = last 20% of the hours (5,261 targets × 321 customers). RSE lower is better, CORR
higher is better; both computed in kWh. Customer #182 is constant in the test period and is
left out of CORR (as in the authors' code).

Models with a random start are run with 3 seeds (0, 1, 2) and shown as mean ± std. Naive rules
and AR (least squares) have no randomness, so every seed gives the same number.

| Model | Setting | Seeds | Trainable numbers | Valid RSE | Valid CORR | Test RSE | Test CORR | Source |
|---|---|---|---|---|---|---|---|---|
| Copy: same time last week | lag 168 | none needed | 0 | 0.1463 | 0.8775 | 0.1874 | 0.8779 | `src/baselines.py` |
| Copy: same time yesterday | lag 24 | none needed | 0 | 0.1104 | 0.8543 | 0.1048 | 0.8667 | `src/baselines.py` |
| **First baseline: one shared neuron** | q = 24, L1 loss, 60 epochs | 0, 1, 2 | 25 | 0.1051 ± 0.0002 | – | 0.1043 ± 0.0003 | 0.8702 ± 0.0001 | `src/neuron_baseline.py` |
| **AR, one model per customer** | L = 256 chosen on valid from 2⁰…2⁹, least squares | none needed | 82,497 | **0.0978** | **0.9176** | **0.0981** | **0.9231** | `src/baselines.py` |
| **LSTNet-Skip (reproduction)** | authors' settings, 100 epochs | 0–9 | 325,871 | – | – | **0.1008 ± 0.0018** | 0.9070 ± 0.0062 | `src/train.py`, Colab |
| LSTNet + RevIN (improvement) | same + RevIN | 0–9 | 326,513 | – | – | 0.1060 ± 0.0014 | 0.9172 ± 0.0010 | `src/train.py`, Colab |
| LSTNet without AR (ablation) | same, no AR part | 0–9 | 325,846 | – | – | 0.1476 ± 0.0096 | 0.8617 ± 0.0043 | `src/train.py`, Colab |
| DLinear (comparison) | window 168, moving average 25 | 0–9 | 338 | – | – | 0.1035 ± 0.0003 | 0.9052 ± 0.0003 | `src/train.py`, Colab |
| *Paper: AR* | – | – | – | – | – | *0.1054* | *0.8595* | Table 2 |
| *Paper: LSTNet-Skip* | – | – | – | – | – | *0.1007* | *0.9119* | Table 2 |

Statistical tests of the neural models: `results/stats_electricity_h24.md`.

## Notes

* **LSTNet-Skip is reproduced:** 0.1008 ± 0.0018 over 10 seeds vs 0.1007 in the paper.
* RevIN does **not** improve RSE (worse by 0.0052, paired t-test p = 0.0001), but improves CORR.
* Without its AR part LSTNet is much worse (0.1476), as the paper's ablation says.
* Our AR beats the paper's AR **and** the paper's LSTNet-Skip on both metrics. Checks done:
  an independent loop implementation gives the same predictions (difference < 1e-11);
  no validation or test target is used for fitting; the max-scaling cannot leak into a
  per-series least-squares fit (it multiplies each series by a constant).
* Likely reason the paper's AR is worse: a longer lag. Validation RSE falls from 0.1094 (L = 1)
  to 0.0978 (L = 256): with L ≥ 168 the model can see the same hour last week and earlier.
  The paper does not report which L its AR used.
* AR is deterministic (least squares, no random start), so it needs no seeds.
* This makes the project question sharper: our LSTNet reproduction must beat **0.0981**, not
  only the paper's 0.1007, for the deep part to earn its complexity.

Validation RSE of AR by lag order L:

| L | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| valid RSE | 0.1094 | 0.1089 | 0.1088 | 0.1087 | 0.1083 | 0.1033 | 0.1019 | 0.1000 | **0.0978** | 0.0983 |

# Experiment table: Solar-Energy, horizon 24

Test set = last 20% of the steps (10,512 targets × 137 plants, late October to December).
One step = 10 minutes, so horizon 24 = 4 hours ahead and one day = 144 steps. RSE lower is
better, CORR higher is better; both computed in MW. No plant is constant in the test period,
so none is left out of CORR.

| Model | Setting | Seeds | Trainable numbers | Valid RSE | Valid CORR | Test RSE | Test CORR | Source |
|---|---|---|---|---|---|---|---|---|
| Copy: same time last week | lag 1008 | none needed | 0 | 0.5513 | 0.8441 | 0.7445 | 0.7124 | `src/baselines.py` |
| Copy: same time yesterday | lag 144 | none needed | 0 | 0.5099 | 0.8648 | 0.6113 | 0.8030 | `src/baselines.py` |
| **First baseline: one shared neuron** | q = 24, L1 loss, 60 epochs | 0, 1, 2 | 25 | 0.8689 ± 0.0038 | – | 0.8909 ± 0.0035 | 0.4729 ± 0.0059 | `src/neuron_baseline.py` |
| **AR, one model per plant** | L = 512 chosen on valid from 2⁰…2⁹, least squares | none needed | 70,281 | **0.4064** | **0.9099** | **0.5247** | **0.8462** | `src/baselines.py` |
| **LSTNet-Skip (reproduction)** | `solar.sh` settings (skip-GRU 10), 100 epochs | 0–9 | – | – | – | **0.4753 ± 0.0249** | 0.8835 ± 0.0098 | `src/train.py`, Colab |
| LSTNet + RevIN (improvement) | same + RevIN | 0–9 | – | – | – | 0.5055 ± 0.0137 | 0.8578 ± 0.0060 | `src/train.py`, Colab |
| LSTNet without AR (ablation) | same, no AR part | 0–9 | – | – | – | 0.4529 ± 0.0104 | 0.8922 ± 0.0051 | `src/train.py`, Colab |
| DLinear (comparison) | window 168, moving average 25 | 0–9 | 338 | – | – | 0.5740 ± 0.0011 | 0.8167 ± 0.0009 | `src/train.py`, Colab |
| *Paper: AR* | – | – | – | – | – | *0.8699* | *0.5314* | Table 2 |
| *Paper: LSTNet-Skip* | – | – | – | – | – | *0.4643* | *0.8870* | Table 2 |

Statistical tests of the neural models: `results/stats_solar_h24.md`.

## Notes

* **LSTNet-Skip is reproduced:** 0.4753 ± 0.0249 over 10 seeds vs 0.4643 in the paper (within half a std).
* RevIN makes RSE worse (+0.0302, p = 0.04) and CORR worse; LSTNet without AR is slightly better (p = 0.02); LSTNet clearly beats AR and DLinear.
* Our AR is far better than the paper's AR (0.5247 vs 0.8699) but, unlike on Electricity,
  worse than the paper's LSTNet-Skip (0.4643). So on Solar the deep part may really help.
* Why our AR is so much better than the paper's: the lag. Validation RSE falls sharply between
  L = 64 (0.6133) and L = 128 (0.4428). From L = 121 on, the window reaches 144 steps back
  (L + h − 1 ≥ 144), i.e. the same time yesterday (median autocorrelation 0.85 at one day,
  only 0.32 at our horizon of 24 steps). The paper does not report which L its AR used.
* L = 512 is the largest value in the grid and validation RSE was still falling, so a longer
  lag might be slightly better. We keep the paper's grid.
* Validation RSE (0.41) is much better than test RSE (0.52) for every model. The test period is
  late autumn and early winter: every plant produces less (median 0.71 of its training level,
  see `notebooks/01_explore_solar.ipynb`) and days are shorter and cloudier. This is the
  distribution shift that RevIN is meant to handle.
* The one shared neuron sees only q = 24 steps (4 hours), so it cannot see the daily cycle; it
  is close to the paper's AR. On Electricity the same q = 24 is a full day, which is why the
  neuron did well there.

Validation RSE of AR by lag order L:

| L | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| valid RSE | 0.9221 | 0.8903 | 0.8601 | 0.8111 | 0.7738 | 0.7463 | 0.6133 | 0.4428 | 0.4265 | **0.4064** |
