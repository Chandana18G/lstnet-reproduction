# Reproducing LSTNet on Electricity and Solar-Energy, and testing whether the deep part earns its complexity

**Deep Learning · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**
**Author(s):** Chandana Gurusiddappa (Electricity), Dineshkumar Swaminathan (Solar-Energy) · **Repository:** https://github.com/Chandana18G/lstnet-reproduction · **Date:** 08.11.2026

---

## Abstract

We forecast all series of a multivariate time series one value 24 steps ahead and reproduce LSTNet-Skip (Lai et al., SIGIR 2018) on Electricity (321 customers, hourly) and Solar-Energy (137 PV plants, every 10 minutes). With the authors' settings and 10 seeds we reach test RSE 0.1008 ± 0.0018 on Electricity (paper 0.1007) and 0.4753 ± 0.0249 on Solar-Energy (paper 0.4643). Our first improvement, RevIN, made RSE worse on both datasets (+5 % and +6 %). Our second, one AR model per customer over the whole 168-hour window, lowers Electricity RSE by 2 % to 0.0988 ± 0.0012 (paired t-test p = 0.026, Diebold–Mariano p = 0.022, Holm-corrected). Comparisons with linear models show that the deep part does not clearly earn its complexity on Electricity, but does on Solar-Energy.

## 1 · The paper and the task (research understanding, part of 15 %)

- **Paper:** Guokun Lai, Wei-Cheng Chang, Yiming Yang, Hanxiao Liu. *Modeling Long- and
  Short-Term Temporal Patterns with Deep Neural Networks.* SIGIR 2018 (41st International ACM
  SIGIR Conference on Research and Development in Information Retrieval), Ann Arbor, USA.
  https://doi.org/10.1145/3209978.3210006 · free full text: https://arxiv.org/abs/1703.07015 ·
  authors' code: https://github.com/laiguokun/LSTNet
- **Task:** multivariate time-series forecasting. The input is the last q = 168 time steps of all
  n series at once (an n × 168 table); the output is the value of all n series at **one** future
  time step, h = 24 steps after the last input step (regression, not a sequence of 24 values).
  - *Electricity* (Chandana): 168 hours of electricity use of 321 customers → the use of all 321
    customers 24 hours later.
  - *Solar-Energy* (Dinesh): 168 ten-minute steps (28 hours) of power output of 137 PV plants →
    the output of all 137 plants 24 steps (4 hours) later.
- **Their method in our own words (LSTNet):** real series such as electricity use repeat
  themselves every day and every week (long-term pattern), but also depend on the last few hours
  (short-term pattern). LSTNet combines five parts:
  1. **Convolution:** 100 filters, each 6 time steps long and as wide as *all* series, slide
     along time and pick up short local patterns across all series at once (ReLU, no pooling).
  2. **GRU (recurrent network):** reads the convolution output step by step and keeps a memory;
     its last hidden state summarises the whole window.
  3. **Skip-GRU:** a second GRU that only looks at every p-th step (p = 24 for hourly data), i.e.
     the same hour on consecutive days. This lets it learn the daily pattern directly instead of
     remembering it over 24 steps.
  4. **Dense layer:** combines the GRU and skip-GRU states into one prediction per series.
  5. **Autoregressive (AR) "highway":** a plain linear model on the last 24 values of each series
     (the same weights for every series), added to the neural prediction. The paper argues that
     the neural part alone does not react to changes in scale, and the linear part does
     (Sec. 4.6).

  Training (Sec. 3.7–3.8 and the authors' code): absolute (L1) or squared loss, chosen on the
  validation set; Adam optimiser; dropout 0.2; the model with the best validation error is
  kept. Data are split 60 % / 20 % / 20 % in time order (Sec. 4.3). Hyperparameters were tuned by
  grid search on the validation set (Sec. 4.4); the authors' run scripts give the final values.
  They report two variants (LSTNet-Skip and LSTNet-Attn); we reproduce **LSTNet-Skip**.
- **Their main result:** LSTNet beats all baselines on the three datasets with repeating
  patterns, by up to 22 % RSE over a plain GRU at horizon 24 (Sec. 4.5). We reproduce one column
  of **Table 2** (page 101): **horizon 24**, LSTNet-Skip, plus the AR baseline from the same table.
  Metrics (Sec. 4.2): RSE, root relative squared error (lower is better), and CORR, the
  correlation between prediction and truth averaged over series (higher is better).

  | Dataset (horizon 24) | LSTNet-Skip RSE | LSTNet-Skip CORR | AR RSE | AR CORR |
  |---|---|---|---|---|
  | Electricity | 0.1007 | 0.9119 | 0.1054 | 0.8595 |
  | Solar-Energy | 0.4643 | 0.8870 | 0.8699 | 0.5314 |

  *Source: Lai et al. 2018, Table 2.*

## 2 · The data (start this week)

### Electricity (Chandana)

| | |
|---|---|
| Source and version | Authors' repository `laiguokun/multivariate-time-series-data`, file `electricity/electricity.txt.gz` (last commit 7f402f1, 09.04.2017), downloaded 18.10.2026 with `scripts/download_data.sh`. Raw data: UCI *ElectricityLoadDiagrams20112014*; the authors removed 2011, kept 321 clients and converted to hourly values (their README; paper Sec. 4.3). |
| Licence | The authors' data repository has **no licence file**, so we name the original source: UCI *ElectricityLoadDiagrams20112014*, licence **CC BY 4.0** (use, share and adapt, also commercially, with attribution). The authors' code (github.com/laiguokun/LSTNet), which we read for the settings, is under the **MIT licence**. |
| Size | 26,304 hourly time steps (2012–2014) × 321 customers, no missing values. One sample = 168 hours × 321 customers (input) → 321 values (target). |
| Classes or targets | Regression, no classes: the target is the consumption (kWh) of every customer 24 hours after the input window. Instead of class balance: customer means range from 11 to 200,529 kWh (factor ~19,000); 1.1 % of values are 0; customer #182 is 0 for the whole test period (left out of CORR). Strong daily and weekly cycle (median autocorrelation 0.94 at 24 h, 0.91 at 168 h). |
| Splits | By time, 60 / 20 / 20 as in the paper: training hours 0–15,781 (~15,600 samples), validation 15,782–21,042 (5,261), test 21,043–26,303 (5,261). Why: forecasting predicts the future from the past; neighbouring hours are almost identical, so a random split would leak information; and it is the paper's split, so the numbers are comparable. |
| Preprocessing | No cleaning needed. Each customer is divided by its largest absolute value (the authors' `normalize=2`); predictions are multiplied back to kWh for the metrics. As in the authors' code, this maximum is taken over the whole series, not over the training set only (see leakage checks). |
| Leakage checks | Split strictly by time; a test checks that the newest input step is exactly 24 h before the target and that no target is in two splits. Scaling on all data: 53 of 321 customers reach their maximum in the test period, but the median ratio of whole-series to training-only maximum is 1.00; we keep the authors' scaling for comparability and name it as a limitation. Best epoch and AR lag order are chosen on validation; the test set is used once. No duplicates: each hour appears once. |
| Shared with Biomedical Image Analysis? | No. |

![Electricity: autocorrelation of six customers](results/figures/electricity_autocorrelation.png)
*Figure 1. Electricity: autocorrelation of six customers (training part); peaks every 24 hours and at one week.*

![Electricity: change in level from training to test](results/figures/electricity_level_shift.png)
*Figure 2. Electricity: change of each customer's mean from training to test (log2 ratio, 0 = no change). Median 0.99; 24 customers change by more than 20 %.*

### Solar-Energy (Dinesh)

| | |
|---|---|
| Source and version | Authors' repository `laiguokun/multivariate-time-series-data`, file `solar-energy/solar_AL.txt.gz` (last commit 7f402f1, 09.04.2017), downloaded 22.10.2026 with `scripts/download_data.sh`. Raw data: NREL *Solar Power Data for Integration Studies* (paper Sec. 4.3): **synthetic** PV output of simulated plants for the weather of 2006, published as 5-minute data per US state; the authors used the Alabama plants at 10 minutes. NREL is now called the National Laboratory of the Rockies (NLR); the page moved to nlr.gov (checked 07.11.2026). |
| Licence | The NLR data terms (www.nlr.gov/disclaimer, checked 07.11.2026) allow anyone to use and copy the data free of charge, provided the notice is kept and DOE/NLR is credited in any publication; the data come "as is", with no warranty. NLR also says the data should not be used for site-specific project work. The authors' copy has no licence file of its own; we credit DOE/NREL (now NLR) and the authors. |
| Size | 52,560 time steps (every 10 minutes, one year) × 137 plants, no missing values. One sample = 168 steps (28 hours) × 137 plants (input) → 137 values (target). |
| Classes or targets | Regression, no classes: the target is the output (MW) of every plant 24 steps (4 hours) after the input window. Instead of class balance: plant means range from 1.6 to 19.1 MW (factor ~12); 55.1 % of all values are 0 (night; at 49.5 % of steps all plants are 0); no constant plant. Strong daily cycle (median autocorrelation 0.85 at 144 steps = one day), weak at the horizon (0.32 at 24 steps). |
| Splits | By time, 60 / 20 / 20 as in the paper: training days 0–219 (January to early August, ~31,300 samples), validation days 219–292 (10,512), test days 292–365 (late October to December, 10,512). Why: as for Electricity; in addition, the split is a season split, and only a split by time shows whether the model handles a season it has not seen. |
| Preprocessing | No cleaning; night zeros are kept, as in the paper. Each plant is divided by its largest absolute value (the authors' scaling); predictions are multiplied back for the metrics. |
| Leakage checks | Split strictly by time (same tests as above). Scaling on all data: no plant reaches its maximum in the test period, median ratio of whole-series to training-only maximum 1.00, so the scaling leaks nothing here. Choices on validation only; test used once. No duplicates. Note the shift: every plant produces less in the test period (median test/train ratio 0.71). |
| Shared with Biomedical Image Analysis? | No. |

![Solar: one week of example plants](results/figures/solar_one_week.png)
*Figure 3. Solar-Energy: one week of output of example plants: zero at night, one peak per day.*

![Solar: change in level from training to test](results/figures/solar_level_shift.png)
*Figure 4. Solar-Energy: every plant produces less in the test period (winter); median ratio 0.71.*

## 3 · Reproducing the baseline (part of 15 %, and 25 % implementation)

**The recipe**, read from the paper and the authors' code (github.com/laiguokun/LSTNet: `main.py`,
`models/LSTNet.py`, `utils.py` and the run scripts `ele.sh`, `solar.sh`). Our implementation:
`src/data.py`, `src/model.py`, `src/train.py`.

| | The paper says | Where (page, table, code) | What we use |
|---|---|---|---|
| Data: version, split | Electricity: 321 clients, hourly, 2012–2014. Solar-Energy: 137 PV plants, 10 min, 2006. Split 60 / 20 / 20 in chronological order. | Sec. 4.3 (p. 99), Table 1 (p. 100); `utils.py` | The authors' published files (section 2); the same 60 / 20 / 20 split by time, decided by the target time step. |
| Model: layers, sizes, pretrained? | LSTNet-Skip: convolution → GRU + skip-GRU → dense, plus a linear AR component. Sizes chosen by grid search: window q from 2⁰…2⁹, CNN/GRU hidden {50, 100, 200}, skip-GRU hidden {20, 50, 100}, skip p = 24 (Electricity), tuned 2…64 (Solar). Not pretrained. CNN filter width, AR window: **not stated**. | Sec. 3.2–3.6 (p. 97–98), Sec. 4.4 (p. 100) | The authors' run scripts: window 168, CNN 100 filters of width 6, GRU 100, skip 24, skip-GRU 5 (Electricity) / 10 (Solar), AR window 24, dropout 0.2, no output activation. 325,871 parameters for Electricity. A test shows our forward pass equals the authors' code (`tests/test_model.py`). Not pretrained. |
| Training: optimiser, learning rate, schedule | Adam. Learning rate and schedule **not stated**. Dropout 0.1 or 0.2. Loss: L1 or squared, chosen on validation. | Sec. 3.7–3.8 (p. 98–99), Sec. 4.4 (p. 100); code: `main.py` | Adam, learning rate 0.001, no schedule, gradient clipping to norm 10, dropout 0.2, L1 loss (the code's default) **summed and computed in original units** (prediction × scale vs target × scale), as in `main.py`. |
| Training: epochs, batch size | **Not stated.** | code: `main.py` defaults | 100 epochs, batch size 128; we keep the weights of the epoch with the best validation RSE (as `main.py`). |
| Augmentation and preprocessing | No augmentation. Normalisation **not stated** in the paper. | code: `utils.py` (`normalize=2`) | No augmentation. Each series divided by its maximum absolute value over the whole series (as the code); see section 2 for the leak this causes and its size. |
| Metric, and how it was averaged | RSE: squared error pooled over all series and test steps, relative to the spread of the test targets. CORR: Pearson correlation per series, averaged over series. | Sec. 4.2, Eq. 10–11 (p. 99); code: `evaluate()` in `main.py` | The same, in original units (`src/metrics.py`). Series with zero variance in the test targets are left out of CORR, as in the code (Electricity: 1 customer; Solar: none). |

- **Setup:** Python and PyTorch (laptop: Python 3.14, PyTorch 2.12 on CPU; Colab: the PyTorch version preinstalled on Colab in October–November 2026, not recorded, on an NVIDIA T4 GPU), NumPy, SciPy. Electricity, horizon 24, uses the settings of the authors' `ele.sh`; Solar uses `solar.sh` but with horizon 24 (see below). 10 seeds per model (0–9); same seed = same initial weights and batch order.
- **What the paper did not say,** and what we chose instead:
  - learning rate, epochs, batch size, gradient clipping, CNN filter width, AR window, normalisation → the authors' code defaults (0.001, 100, 128, 10, 6, 24, divide by maximum);
  - the final hyperparameters of the grid search → the authors' run scripts; we do **not** repeat the grid search;
  - how many runs the reported number is based on → not stated (probably one run); we report mean ± standard deviation over 10 seeds;
  - the training loss is summed in original units in the code (not mentioned in the paper) → we do the same.
- **Where paper and code disagree:**
  - the paper uses ReLU in the GRU's candidate state (Sec. 3.3, p. 98); the code uses the standard GRU (tanh) → we follow the code;
  - skip sequence length (q − ω) / p = 6.75 in the code (float) → we use the whole number 6 (the last 144 CNN steps);
  - `solar.sh` does not set the horizon, so it runs at horizon 12, but Table 2 reports horizon 24 → we use 24;
  - the paper's grid includes an AR regularisation coefficient {0.1, 1, 10}; the code has none → no regularisation.

| Test set, horizon 24 | Paper (Table 2) | Ours (mean ± std, 10 seeds) |
|---|---|---|
| Electricity, RSE | 0.1007 | **0.1008 ± 0.0018** |
| Electricity, CORR | 0.9119 | 0.9070 ± 0.0062 |
| Solar-Energy, RSE | 0.4643 | **0.4753 ± 0.0249** |
| Solar-Energy, CORR | 0.8870 | 0.8835 ± 0.0098 |

**Electricity: reproduced.** Our mean test RSE over 10 seeds (0.1008) is within 0.0001 of the
paper's 0.1007, well inside one standard deviation (0.0018). CORR is 0.0049 lower than published
(0.9070 vs 0.9119); the paper's number lies within one standard deviation of ours (0.0062). The
remaining small CORR gap can come from details we could not match exactly: the paper reports
one run (seed and number of runs not stated), the code uses tanh in the GRU where the paper
writes ReLU, and the skip length is rounded (6 instead of 6.75 steps). Results:
`results/stats_electricity_h24.md`; training: `notebooks/03_train_on_colab.ipynb`.

**Checks of our AR baseline (Electricity)** (professor's feedback; `src/check_ar.py`,
`results/ar_checks_electricity.md`). Our AR (one linear model per customer, least squares) reaches
test RSE 0.0981, the paper's AR 0.1054. We checked four possible reasons:

1. *RSE as in Eq. 10:* our formula and the authors' code formula give the same value on the same
   predictions (0.098090 both): original units, pooled over all customers and test hours,
   divided by the spread of the whole test set.
2. *Off-by-one in the horizon:* a unit test checks that the newest input is exactly h steps before
   the target. Our AR is better than the paper's at **every** horizon (h = 3: 0.0818 vs 0.0995;
   h = 6: 0.0913 vs 0.1035; h = 12: 0.0957 vs 0.1050; h = 24: 0.0981 vs 0.1054), so the gap is
   not a shifted horizon.
3. *Scaling on the training part only:* gives exactly the same test RSE (0.098090); a least-squares
   fit per customer does not change when the customer is multiplied by a constant.
4. *The window:* this explains the gap. With short windows (L = 1 to 16 hours) our AR reaches test
   RSE 0.1035–0.1042 at horizon 24, within 1–2 % of the paper's 0.1054. Only windows of 128 hours
   or more, which see the same hour on several previous days, bring it down to 0.098. We choose
   L on validation from the paper's grid 1 … 512 (L = 256); the paper does not report which window
   its AR used.

So the gap is explained by the window length, not by a bug: a well-tuned linear model with a long
window is stronger than the AR baseline in the paper.

**Solar-Energy: reproduced.** Our mean test RSE over 10 seeds (0.4753) is 2 % above the paper's
0.4643, within half a standard deviation (0.0249); CORR is almost the same (0.8835 vs 0.8870). The
spread over seeds is much larger than on Electricity (std 0.0249 vs 0.0018), so a single run, as in
the paper, can land anywhere between about 0.45 and 0.50. Settings: the authors' `solar.sh`
(skip-GRU size 10) at horizon 24. Results: `results/stats_solar_h24.md`.

## 4 · Your improvement (20 %)

**Improvement:** add RevIN (Kim et al., ICLR 2022) to LSTNet-Skip. Each input window is
normalised to mean 0 and std 1, and the prediction is moved back afterwards. Everything else
stays the same as in the reproduction.

- **Electricity (Chandana):** **Because** 24 of 321 customers change their level by more than
  20 % between training and test, **we expect** RevIN **to improve** the test RSE at horizon 24
  **by about** 1–3 % (0.101 → 0.099, mean over 10 seeds).
- **Solar-Energy (Dinesh):** **Because** every plant produces about 29 % less in the winter test
  period, **we expect** RevIN **to improve** the test RSE at horizon 24 **by about** 3–5 %
  (0.464 → 0.445, mean over 10 seeds).

**Improvement 2 (Electricity, after the mid-course feedback):** RevIN did not help, so we change
the part that matters most on Electricity, the AR highway. In the paper it uses the last 24 h
with the same weights for all customers; we give every customer its own weights over the whole
168-h window (`lstnet_ar168`). Everything else stays the same.

- **Electricity (Chandana):** **Because** our AR baseline (one linear model per customer, long
  window) already reaches RSE 0.0981, better than LSTNet's 0.1008, **we expect** a per-customer
  168-h AR highway **to improve** the test RSE at horizon 24 **by about** 2–4 %
  (0.1008 → about 0.098, mean over 10 seeds).

**Improvement 2 (Solar-Energy, after the mid-course feedback):** RevIN did not help on Solar
either, so we change the part of LSTNet that is meant to catch periodic patterns, the skip-GRU.
The paper uses a skip of p = 24 steps, which on Solar is only 4 hours; the paper tuned p only
between 2 and 64, so it never compares a moment with the same moment on the day before. We set
p = 144 (one day). A skip sequence has `(q − 6) // p` steps, so with the paper's window q = 168 it
would hold only one step; we therefore use q = 438 (3 days + 6) to get 3 steps per sequence: the
same time today, yesterday and the day before (`lstnet_skip144`). Everything else stays the same.

- **Solar-Energy (Dinesh):** **Because** Solar repeats itself every day (median autocorrelation
  0.85 at 144 steps and still 0.82 at 288 and 432 steps, but only 0.32 at 24 steps) and our AR
  baseline became much better as soon as its window reached one day back, **we expect** a one-day
  skip with a 3-day window **to improve** the test RSE at horizon 24 **by about** 2–5 %
  (0.4753 → about 0.455, mean over 10 seeds).

Comparisons (not improvements): AR (one linear model per series), DLinear (Zeng et al., AAAI
2023) and the paper's "LSTNet without AR" ablation.

## 5 · Experiments (part of 20 % and 15 %)

- **Design:** one change at a time; the same data splits, the same budget; at least three seeds per setting.

**Electricity, horizon 24, test set** (each neural model: 10 seeds, the same splits, settings and
budget of 100 epochs; the best epoch is chosen on validation):

| Setting | Test RSE mean | ± spread (std) | Test CORR mean ± std | seeds |
|---|---|---|---|---|
| baseline: LSTNet-Skip (reproduction) | 0.1008 | 0.0018 | 0.9070 ± 0.0062 | 10 |
| + RevIN (improvement 1) | 0.1060 | 0.0014 | 0.9172 ± 0.0010 | 10 |
| + per-customer AR over 168 h (improvement 2) | **0.0988** | 0.0012 | **0.9246 ± 0.0019** | 10 |
| − AR component (paper's ablation LSTw/oAR) | 0.1476 | 0.0096 | 0.8617 ± 0.0043 | 10 |
| DLinear (comparison) | 0.1035 | 0.0003 | 0.9052 ± 0.0003 | 10 |
| AR, one model per customer (comparison, least squares) | 0.0981 | – (deterministic) | 0.9231 | – |
| *Paper: LSTNet-Skip* | *0.1007* | – | *0.9119* | – |

- **Statistical comparison** (`src/stats.py`; Holm-corrected over all comparisons with LSTNet):
  - *Over seeds:* paired t-test and Wilcoxon signed-rank test, seed k against seed k (same seed =
    same initial weights and batch order).
  - *Over test hours:* Diebold–Mariano test on the squared error per hour (summed over customers),
    using the forecast averaged over the 10 seeds; Newey–West variance with lag h − 1 = 23 and the
    Harvey–Leybourne–Newbold correction.

  | Model vs LSTNet | Mean RSE difference | Paired t-test p (Holm) | Wilcoxon p (raw) | DM statistic | DM p (Holm) |
  |---|---|---|---|---|---|
  | + RevIN | +0.0052 (worse) | 0.0001 | 0.0020 | +1.39 | 0.33 |
  | + per-customer AR 168 h | −0.0020 (better) | 0.026 | 0.020 | −2.78 | 0.022 |
  | − AR component | +0.0468 (worse) | < 0.0001 | 0.0020 | +9.37 | < 0.0001 |
  | DLinear | +0.0027 (worse) | 0.0025 | 0.0059 | +1.85 | 0.19 |
  | AR (one model per customer) | −0.0027 (better) | – (no seeds) | – | −0.88 | 0.38 |

  **What this shows for Electricity:**
  - **Improvement 1 (RevIN): the hypothesis is not confirmed.** RevIN makes RSE **worse** by 5 % (0.1060 vs 0.1008),
    consistently over all 10 seeds (paired t-test p = 0.0001). It improves CORR (0.9172 vs
    0.9070) and makes the runs more stable (std 0.0010 vs 0.0062). Over the test hours, the
    difference is not significant (DM p = 0.33). Our reading: RevIN gets the *shape* of the
    daily pattern better (CORR), but rescaling each window loses information about the absolute
    level, which RSE in kWh punishes, especially for the largest customers. Section 6 shows that
    this loss comes from one customer.
  - **Improvement 2 (per-customer AR over 168 h): the hypothesis is confirmed, at the low end of
    the expected size.** RSE drops by 2.0 % (0.0988 vs 0.1008; expected 2–4 %), better in the
    seed comparison (paired t-test p = 0.026, Wilcoxon p = 0.020) and over the test hours
    (DM −2.78, p = 0.022). It also gives the best CORR of all models (0.9246 vs 0.9070) and more
    stable runs (std 0.0012 vs 0.0018). Giving every customer its own linear weights over the whole
    week lets the model follow each customer's level and weekly pattern; this is the part that
    matters most on Electricity. It is about as good as the stand-alone per-customer AR (0.0981),
    so the deep part still adds little on top of a good linear model here.
  - **The AR component matters most**, as the paper claims (Sec. 4.6, Fig. 5): without it RSE
    rises by 46 % (0.1476), significant in every test.
  - **The deep part does not clearly earn its complexity on Electricity:** the linear AR model
    (0.0981) is slightly better than LSTNet (0.1008), and DLinear slightly worse (0.1035), but
    neither difference is significant over the test hours (DM p = 0.38 and 0.19).

**Improvement 2 against the baseline, seed by seed** (the same seeds in the same order; raw p-values, before the Holm correction used in the table above):

| Model | Test RSE, 10 seeds |
|---|---|
| LSTNet-Skip | 0.1008 ± 0.0018 |
| LSTNet + per-customer AR over 168 h | 0.0988 ± 0.0012 |

LSTNet + per-customer AR is 0.0020 ± 0.0021 better in RSE than LSTNet-Skip, on 8 of 10 seeds (paired t-test p = 0.0131, significant; Wilcoxon p = 0.0195). The seeds alone move the baseline by 0.0050, more than most gaps between the two models, which is why we compare seed by seed.

![Test RSE per seed: baseline, improved model, and the gap between them](results/figures/paired_runs_electricity.png)

**Solar-Energy, horizon 24, test set** (the same design; skip-GRU size 10 as in `solar.sh`):

| Setting | Test RSE mean | ± spread (std) | Test CORR mean ± std | seeds |
|---|---|---|---|---|
| baseline: LSTNet-Skip (reproduction) | 0.4753 | 0.0249 | 0.8835 ± 0.0098 | 10 |
| + RevIN (improvement) | 0.5055 | 0.0137 | 0.8578 ± 0.0060 | 10 |
| − AR component (paper's ablation LSTw/oAR) | 0.4529 | 0.0104 | 0.8922 ± 0.0051 | 10 |
| one-day skip, p = 144, q = 438 (improvement 2) | 0.4551 | 0.0107 | 0.8881 ± 0.0057 | 10 |
| DLinear (comparison) | 0.5740 | 0.0011 | 0.8167 ± 0.0009 | 10 |
| AR, one model per plant (comparison, least squares, L = 512) | 0.5247 | – (deterministic) | 0.8462 | – |
| *Paper: LSTNet-Skip* | *0.4643* | – | *0.8870* | – |

  | Model vs LSTNet (Solar) | Mean RSE difference | Paired t-test p (Holm) | Wilcoxon p (raw) | DM statistic | DM p (Holm) |
  |---|---|---|---|---|---|
  | + RevIN | +0.0302 (worse) | 0.062 | 0.027 | +2.12 | 0.067 |
  | − AR component | −0.0224 (better) | 0.030 | 0.006 | −2.49 | 0.039 |
  | one-day skip (improvement 2) | −0.0202 (better) | 0.147 | 0.232 | −0.57 | 0.571 |
  | DLinear | +0.0987 (worse) | < 0.0001 | 0.002 | +4.70 | < 0.0001 |
  | AR (one model per plant) | +0.0494 (worse) | – (no seeds) | – | +5.81 | < 0.0001 |

  **What this shows for Solar:**
  - **The hypothesis is not confirmed here either:** RevIN makes RSE 6 % worse (0.5055 vs 0.4753)
    and also lowers CORR (0.8578 vs 0.8835); the RSE difference is significant before correction (t-test raw p = 0.021) but not after Holm over the five Solar comparisons (p = 0.062, DM p = 0.067), although Solar has the
    strong level shift RevIN is designed for. A possible reason, still to be checked: many input
    windows are mostly night zeros, and normalising such near-constant windows is unstable.
  - **Unlike the paper's ablation, LSTNet without its AR part is slightly better on Solar**
    (−0.0224, Holm p = 0.030, DM p = 0.039). Not yet analysed; possible reasons are the night zeros and
    the change of season, which make a fixed linear rule on the last 4 hours misleading.
  - **Here the deep part does earn its complexity:** LSTNet clearly beats both linear models
    (AR 0.5247, DLinear 0.5740; DM p < 0.0001).
  - **The one-day skip (improvement 2) helps on average, but we cannot call it a real
    improvement yet.** The mean test RSE drops from 0.4753 to 0.4551 (4 % better), which is close
    to what we predicted (about 0.455). But it beats LSTNet on only 5 of the 10 seeds. Most of the
    gain comes from three seeds where LSTNet did badly (seeds 4, 8 and 9, RSE 0.49 to 0.52); on
    the other seeds the two models are about the same. So none of the tests is significant
    (paired t-test Holm p = 0.147, Wilcoxon p = 0.23, DM p = 0.57). What the one-day skip clearly
    does is make training more stable: the spread over seeds is less than half (std 0.0107 vs
    0.0249). DM uses the forecasts averaged over the seeds, and averaging already hides most of
    LSTNet's bad seeds, so DM sees almost no difference (seed-averaged RSE 0.4405 vs 0.4444).
    Also, the model has 1.85 times more parameters (357,402 vs 193,002) and looks back 3 days
    instead of 28 hours, so even this small gain may come from the bigger model and the longer
    window, not from the skip itself.

**Both datasets together:** the answer to our question depends on the data. On Electricity
(regular daily and weekly cycles) linear models are as good as LSTNet and LSTNet's own AR part
does most of the work; on Solar (night zeros, weather, season change) the neural network clearly
wins and the AR part is not needed. RevIN did not improve RSE on either dataset.
- **Test set:** each model's test result is computed once, with the epoch chosen on validation. Improvement 2 was designed after we had seen test results of other models (see Limitations, section 6).

## 6 · Evaluation and error analysis (15 %)

- **Metrics that fit the task:** this is regression, so there is no confusion matrix. We use the paper's two metrics because they answer different questions: RSE pools squared errors in original units, so large series and level errors dominate it; CORR is computed per series and ignores level and scale, so every series counts equally. RevIN shows why both are needed: on Electricity it improves CORR and worsens RSE.
- **Where the models fail, Electricity** (`src/error_analysis.py`, `results/error_analysis_electricity_h24.md`; forecasts averaged over the seeds):
  - **One customer dominates RSE.** Customer #313 holds 26.5 % of the test load but makes 85.5 % of LSTNet's squared error (RevIN 87.9 %, improvement 2 86.7 %), and the 10 largest customers make 98.2 %. So RSE on Electricity mostly measures how well the models predict one customer.
  - **This changes how we read RevIN.** RevIN has a smaller squared error than LSTNet on 299 of 321 customers. Without #313, its mean RSE over seeds is better than LSTNet's (0.0985 ± 0.0011 vs 0.1042 ± 0.0040, paired t-test raw p = 0.003). Its overall loss comes from the largest customers, which fits its better CORR. This comparison was made after seeing the results, so it is a finding to test again, not a confirmed claim.
  - **Improvement 2 holds up without that customer:** better on 305 of 321 customers, and without #313 RSE 0.0980 ± 0.0015 vs 0.1042 ± 0.0040 (raw p = 0.0002). It also gains most in absolute terms on the largest customers: the 10 largest make up about 96 % of the drop in squared error.
  - **Level shift is not where the error is.** The 24 customers whose level changes by more than 20 % make only 0.2 % of LSTNet's squared error. This is a second reason why RevIN, chosen for those customers, could not lower RSE much.
  - **Time of day.** Almost all the error falls in the daytime: the 8 hours from 12 to 20 make about 56 % of it, the 4 hours from 04 to 08 only 1.5 %. The worst 1 % of test hours hold 17.4 % of LSTNet's squared error, and three of its five worst hours are in a row (steps 22072–22074). The file has no timestamps, so these hours assume that it starts at midnight.
  - **Worst customers.** The median correlation per customer is 0.943, but a few customers are barely predicted, such as #2 (−0.014) and #317 (0.213).

  ![Electricity: squared error by hour of the day](results/figures/error_by_time_electricity.png)

- **Where the models fail, Solar-Energy** (same script, `results/error_analysis_solar_h24.md`; forecasts averaged over the seeds):
  - **The error is in the daytime, mostly before noon.** The hours 08–12 make 58 % of LSTNet's squared error and 12–16 another 37 %. Nights make almost nothing: 56 % of the test steps are night with all plants at 0, but they give only 0.6 % of the error. So the night zeros we were worried about are easy for the models. The hard part is guessing how much sun there will be 4 hours later.
  - **The worst moments are short weather events.** The worst 1 % of test steps hold 13.1 % of the error, and the 5 worst steps come one after another (steps 47878–47882, around noon). This looks like one sudden change in the weather, such as clouds, that no model could see 4 hours before.
  - **No single plant dominates, unlike Electricity.** The largest plant (#5) makes 4.9 % of the error, and the 10 largest plants make 33 % (they produce 16.4 % of the output). The median correlation per plant is 0.897 and even the worst plant reaches 0.864. So on Solar the RSE is a fair summary of all plants.
  - **The level shift does not explain the errors.** The 123 plants whose level changes by more than 20 % make 83.5 % of the error, a bit less than their share of the plants (90 %). This fits the result that RevIN, which targets the level shift, did not help.
  - **How the models differ.** RevIN has the largest error in the morning (figure below) and is worse than LSTNet on 127 of 137 plants. LSTNet without AR is better on 136 of 137 plants, so its gain is spread over almost all plants. The one-day skip is better on only 62 of 137 plants: a bit lower around 11–13 h, a bit higher around 9–11 h, so overall it is close to LSTNet.
  - As for Electricity, the hours assume that the file starts at midnight.

  ![Solar-Energy: squared error by hour of the day](results/figures/error_by_time_solar.png)

- **Limitations, what our results do not show:**
  - *One horizon and one split.* We test horizon 24 only, on one chronological test period. The seeds measure training randomness, not how the models behave in other periods. On Solar-Energy the test period is also a single season.
  - *Improvement 2 was chosen after seeing test results.* Its idea came from test results of the AR baseline and the ablation, and its hypothesis was written before its own runs. The test set therefore influenced the choice of the idea, so its 2 % gain may be slightly optimistic and should be confirmed on another period.
  - *No grid search.* We use the authors' final settings and do not repeat their search, so we cannot say that these settings are the best for our improvements.
  - *The one-day skip on Solar changes three things at once:* the skip (24 → 144), the window (168 → 438) and the number of parameters. A run with window 438 and skip 24 would show which change matters; we did not have time for it.
  - *The per-customer AR (Electricity improvement 2) was tested on Electricity only,* and it was not compared statistically with the stand-alone AR model (0.0981).
  - *The authors' scaling uses the whole series,* test period included. Its effect is small (section 2), but it is a leak.
  - *RSE depends heavily on one customer.* Our Electricity ranking is largely a ranking on #313, so differences of a few percent say little about the typical customer.
  - *The explanations are readings, not proofs.* Why RevIN hurts on the largest customers and why Solar-Energy is better without AR were not tested with separate experiments.

## 7 · Conclusion

We reproduced LSTNet-Skip at horizon 24 on both datasets with the authors' settings and 10 seeds:
RSE 0.1008 ± 0.0018 on Electricity (paper 0.1007) and 0.4753 ± 0.0249 on Solar-Energy (paper
0.4643). Our question was whether LSTNet's deep part earns its complexity over simple linear
models, and the answer depends on the data. On Electricity it does not: a linear model with one
set of weights per customer and a long window (RSE 0.0981) is as good as LSTNet, and LSTNet loses
46 % of its accuracy without its own linear AR part. Building that insight into LSTNet, with one
AR model per customer over the whole week, was our successful improvement: RSE 0.0988 ± 0.0012,
2 % better than the reproduction and significant over seeds and over test hours (p ≈ 0.02). On
Solar the deep part does earn its complexity: LSTNet clearly beats both linear models (AR 0.5247,
DLinear 0.5740), and the AR part is not even needed. Our first improvement, RevIN, did not lower RSE on either dataset (Electricity +5 %, Solar +6 %), although it
improved CORR and stability on Electricity. The error analysis shows why: one customer makes 85 %
of LSTNet's squared error on Electricity, RevIN is better on 299 of 321 customers, and its overall
loss comes from the largest ones. On Solar we also tried a one-day skip (144 steps with a 3-day
window): it lowered the mean RSE by 4 % and made training more stable, but the gain was not
significant. Next we would repeat the comparisons on another period, test the per-series AR
highway on Solar, and check whether the skip or simply the longer window causes that small gain.

## 8 · Use of AI tools (5 %)

We used **Claude Code** (Anthropic) for programming, debugging, explanations and drafting report text. Chandana and Dinesh each used it for their own dataset. A short log of the main uses is in `AI_USAGE.md`.

**What it did:** it drafted most of the code in `src/` and `tests/`, the notebooks and a first version of most report sections. It also researched and suggested options for the improvements. We chose among them.

**What we did:** we chose the paper and the datasets, decided which improvements to test, ran the training on Colab, and went through the code line by line so that we can explain it.

**How we checked its output:**

- *Model:* a test shows that our forward pass equals the authors' code with the same weights (`tests/test_model.py`). The result (0.1008) also matches the paper (0.1007).
- *Data and metrics:* small tests with values checked by hand, for example that the newest input is exactly h steps before the target.
- *Unexpected results:* our AR beating the paper's AR led to the four checks in section 3. These checks follow the professor's feedback.
- *Numbers:* we recomputed the Electricity table and tests from the 50 raw result files, and they match exactly.

**Where it was wrong:** it first labelled the Solar copy-the-past baselines as if the data were hourly, which was fixed by using the real day length. It also misread a figure (9 instead of 8 better seeds) and gave a wrong best epoch; both were corrected against the result files. So we did not take its numbers on trust.

## 9 · Reproducibility (part of 25 %)

- **Run it:** `pip install -r requirements.txt`, then `bash scripts/download_data.sh` (the data is not in the repository) and `pytest tests/`. Training: open `notebooks/03_train_on_colab.ipynb` (Electricity) or `notebooks/03_train_on_colab_solar.ipynb` (Solar-Energy) on Google Colab and run the cells from top to bottom. A single run: `python src/train.py --data electricity --horizon 24 --model lstnet --seed 0`.
- **Seeds and hardware:** seeds 0–9 for every neural model; the same seed gives the same initial weights and batch order for every model, so runs are paired. Colab with an NVIDIA T4 GPU, about 4 s per epoch and 7 minutes per run on Electricity. Small numeric differences between machines are possible. Runs continue from their last checkpoint if Colab stops. Checkpoints and test predictions are too big for GitHub: [Electricity, Google Drive](https://drive.google.com/drive/folders/1qYeRl8aC5b4PAMzLaBK3ZAr2Sv9-4tPO?usp=sharing).
- **Where the results come from:**

  | Result | Made by |
  |---|---|
  | Figures and numbers of section 2 | `notebooks/01_explore_electricity.ipynb`, `notebooks/01_explore_solar.ipynb` |
  | Copy-the-past rules and AR | `src/baselines.py` → `results/baselines_<data>_h24.csv` |
  | One-neuron baseline | `notebooks/02_first_network.ipynb`, `src/neuron_baseline.py` → `results/neuron_<data>_h24.csv` |
  | Test result of each neural run | `src/train.py` → `results/runs/<run>.json` |
  | Tables and tests of section 5 | `src/stats.py` → `results/stats_<data>_h24.md` |
  | The four AR checks of section 3 | `src/check_ar.py` → `results/ar_checks_electricity.md` |
  | Error analysis of section 6 | `src/error_analysis.py` → `results/error_analysis_<data>_h24.md`, `results/figures/error_by_time_<data>.png` |
  | The fixed split | `src/save_split.py` → `results/splits.json` |

## References

**Papers**

- Lai, G., Chang, W.-C., Yang, Y., Liu, H. (2018). Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. *SIGIR 2018.* https://doi.org/10.1145/3209978.3210006
- Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., Choo, J. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. *ICLR 2022.*
- Zeng, A., Chen, M., Zhang, L., Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023.*

**Statistical methods**

- Diebold, F. X., Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics* 13(3), 253–263.
- Harvey, D., Leybourne, S., Newbold, P. (1997). Testing the Equality of Prediction Mean Squared Errors. *International Journal of Forecasting* 13(2), 281–291.
- Newey, W. K., West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica* 55(3), 703–708.
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics* 6(2), 65–70.
- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. *Biometrics Bulletin* 1(6), 80–83.

**Data**

- Authors' prepared files: https://github.com/laiguokun/multivariate-time-series-data
- Electricity: UCI Machine Learning Repository, *ElectricityLoadDiagrams20112014.* https://doi.org/10.24432/C58C86
- Solar-Energy: NREL (now National Laboratory of the Rockies, NLR), U.S. Department of Energy, *Solar Power Data for Integration Studies.* https://www.nlr.gov/grid/solar-power-data (accessed 07.11.2026)

**Code we read or compared with**

- LSTNet, the authors' code: https://github.com/laiguokun/LSTNet
- RevIN, the official code: https://github.com/ts-kim/RevIN
- DLinear, the official code: https://github.com/cure-lab/LTSF-Linear
