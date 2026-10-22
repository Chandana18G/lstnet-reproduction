# Reproducing LSTNet on Electricity and Solar-Energy, and testing whether the deep part earns its complexity

**Deep Learning · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**
**Author(s):** Chandana Gurusiddappa (Electricity), Dinesh [last name] (Solar-Energy) · **Repository:** https://github.com/Chandana18G/lstnet-reproduction · **Date:** [hand-in date]

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

| | |
|---|---|
| Source and version | [where from, which version, download date] |
| Licence | [licence, and what it allows] |
| Size | [number of samples, size of each sample] |
| Classes or targets | [list; class balance as counts or a bar chart] |
| Splits | [training / validation / test sizes; **how** you split: random, grouped (by patient, place, …), by time; why] |
| Preprocessing | [resizing, standardisation (statistics from the training set only), cleaning] |
| Leakage checks | [duplicates between splits? the same patient/place/time in two splits? preprocessing fitted on all data? what you did about it] |
| Shared with Biomedical Image Analysis? | [yes/no; if yes: which preparation was done for BIA, and say that it is not graded again here] |

[Two or three example samples per class, as a figure.]

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
