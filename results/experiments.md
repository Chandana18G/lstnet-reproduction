# Experiment table: Electricity, horizon 24

Test set = last 20% of the hours (5,261 targets × 321 customers). RSE lower is better, CORR
higher is better; both computed in kWh. Customer #182 is constant in the test period and is
left out of CORR (as in the authors' code).

| Model | Setting | Trainable numbers | Valid RSE | Valid CORR | Test RSE | Test CORR | Source |
|---|---|---|---|---|---|---|---|
| Naive: same hour last week | lag 168 | 0 | 0.1463 | 0.8775 | 0.1874 | 0.8779 | `src/baselines.py` |
| Naive: same hour yesterday | lag 24 | 0 | 0.1104 | 0.8543 | 0.1048 | 0.8667 | `src/baselines.py` |
| One shared neuron | q = 24, L1 loss, 60 epochs, 1 seed | 25 | 0.1063 | 0.8578 | 0.1032 | 0.8698 | `notebooks/02_first_network.ipynb` |
| **AR, one model per customer** | L = 256 chosen on valid from 2⁰…2⁹, least squares | 82,497 | **0.0978** | **0.9176** | **0.0981** | **0.9231** | `src/baselines.py` |
| *Paper: AR* | – | – | – | – | *0.1054* | *0.8595* | Table 2 |
| *Paper: LSTNet-Skip* | – | – | – | – | *0.1007* | *0.9119* | Table 2 |

## Notes

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
