# Reproducing LSTNet on Electricity and Solar-Energy, and testing whether the deep part earns its complexity

**Deep Learning · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**
**Author(s):** Chandana Gurusiddappa (Electricity), Dineshkumar Swaminathan (Solar-Energy) · **Repository:** https://github.com/Chandana18G/lstnet-reproduction · **Date:** [hand-in date]

---

## Abstract

[Four or five sentences: the task, the paper, the data, your reproduced result next to the published one, your improvement and its measured effect (mean ± spread).]

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
| Licence | UCI: CC BY 4.0. Allows use, sharing and adaptation, also commercially, with attribution. The authors' copy has no licence file of its own; we cite both. |
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
| Source and version | Authors' repository `laiguokun/multivariate-time-series-data`, file `solar-energy/solar_AL.txt.gz` (last commit 7f402f1, 09.04.2017), downloaded 22.10.2026 with `scripts/download_data.sh`. Raw data: NREL *Solar Power Data for Integration Studies*, PV plants in Alabama, 2006 (paper Sec. 4.3). |
| Licence | NREL publishes the data openly for research. The authors' copy has no licence file of its own; we cite NREL and the authors. [Dinesh: confirm the exact terms on the NREL page.] |
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

**The recipe:** read the paper (or its code) like a recipe. One row per ingredient; leave nothing empty: if the paper does not say it, write "not stated" and what you chose.

| | The paper says | Where (page, table, code) | What we use |
|---|---|---|---|
| Data: version, split | | | |
| Model: layers, sizes, pretrained? | | | |
| Training: optimiser, learning rate, schedule | | | |
| Training: epochs, batch size | | | |
| Augmentation and preprocessing | | | |
| Metric, and how it was averaged | | | |

- **Setup:** [framework and versions; anything not in the table]
- **What the paper did not say,** and what you chose instead: [list, from the "not stated" rows]

| | Paper | Ours (mean ± spread, n seeds) |
|---|---|---|
| [metric] | [value] | [value] |

[Why the numbers differ, if they do: data version, preprocessing, compute, missing details. A difference that you explain is a result, too.]

## 4 · Your improvement (20 %)

- **Idea:** [what you change]
- **Why it should help:** [the reasoning: a weakness of the baseline you observed, a method from the course or from another paper]
- **Expected effect:** [what you expect to see, and on which metric]

## 5 · Experiments (part of 20 % and 15 %)

- **Design:** one change at a time; the same data splits, the same budget; at least three seeds per setting.

| Setting | [metric] mean | ± spread | seeds |
|---|---|---|---|
| baseline | | | |
| + [change 1] | | | |
| + [change 2] | | | |

- **Statistical comparison:** [test and result, for example a paired test over seeds or folds (Session 31)]
- **Test set:** used **once**, for the final models only.

## 6 · Evaluation and error analysis (15 %)

- [Metrics that fit the task (not only accuracy): confusion matrix, precision, recall, F1, IoU, …]
- [Where does the model fail? Examples of its most confident mistakes; which classes are confused, and why]
- [Limitations: what your results do **not** show]

## 7 · Conclusion

[What you reproduced, what you improved, what you would try next.]

## 8 · Use of AI tools (5 %)

[Which tools you used, for what (code, text, debugging), and how you **checked** their output. Be specific.]

## 9 · Reproducibility (part of 25 %)

- **Run it:** [the commands or notebook order; the environment, for example `requirements.txt`]
- **Seeds and hardware:** [seeds used; GPU or CPU; training time]
- **Where the results come from:** [which notebook/script makes which table and figure]

## References

[All papers, data sets, code and websites you used.]
