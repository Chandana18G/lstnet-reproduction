# Mid-course presentation, Thu 29.10 (4 minutes)

Asked for: our baseline against the paper, and the planned improvement. 5 slides, about
45 seconds each. Numbers in [brackets] are filled in after the Colab runs.

---

## Slide 1 · The paper and our question (Chandana)

- LSTNet (Lai et al., SIGIR 2018): CNN + GRU + skip-GRU + a linear AR part, for multivariate
  forecasting. We reproduce **one number**: Table 2, horizon 24.
- Electricity (Chandana): 321 customers, hourly, 3 years. Solar-Energy (Dinesh): 137 PV
  plants, every 10 minutes, 1 year.
- Task: last 168 steps of all series in → all series at one point 24 steps later out.
- **Our question:** does the deep part earn its complexity over simple linear models?

## Slide 2 · The data (both)

- Split by time, 60/20/20, as the authors; test set used once, at the end.
- Electricity: strong daily (autocorrelation 0.94 at 24 h) and weekly (0.91) cycle → explains
  skip p = 24 and window 168. One customer is 0 for the whole test period → left out of CORR.
- Solar: 55% night zeros; test = late Oct–Dec, about 30% less output than in training.
- Figure: `results/figures/electricity_autocorrelation.png`.

## Slide 3 · Baseline vs paper (Chandana, Electricity)

| Model (test, horizon 24) | RSE ↓ | CORR ↑ |
|---|---|---|
| Copy: same time yesterday | 0.1048 | 0.8667 |
| One neuron (3 seeds) | 0.1043 ± 0.0003 | 0.8702 |
| AR, one model per customer (ours) | **0.0981** | **0.9231** |
| Paper: AR | 0.1054 | 0.8595 |
| Paper: LSTNet-Skip | 0.1007 | 0.9119 |
| **Our LSTNet-Skip** | [ ] | [ ] |

- Our LSTNet matches the authors' code exactly (a test compares the forward passes).
- Surprise: our simple AR already beats the published LSTNet. Checked with a second
  implementation.

## Slide 4 · Solar baseline vs paper (Dinesh)

- Paper LSTNet-Skip, horizon 24: RSE 0.4643, CORR 0.8870; paper AR: 0.8699.
- Our baselines: [ ].

## Slide 5 · Planned improvement and evaluation (both)

- **Improvement: RevIN** (Kim et al., ICLR 2022): normalise each input window, de-normalise the
  prediction. Why: the paper itself says LSTNet needs its AR part because neural networks can't
  follow changes in level; RevIN removes the level directly. Solar's test period has a strong
  level shift (winter), Electricity a small one.
- Comparisons: DLinear (a strong simple linear model), LSTNet without AR (ablation).
- Evaluation: 10 seeds per model, paired t-test / Wilcoxon over seeds, Diebold–Mariano test on
  the test hours, Holm correction.
