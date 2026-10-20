# LSTNet Reproduction: Deep Learning Project (SRH University Munich)

## Context
- Course: Deep Learning, M.Sc. Applied Data Science and AI, WS 2026/27, Prof. Dr. Judith Cerit.
- Task: reproduce a published result, improve the model with a clear justification, compare
  baseline vs improved with proper metrics, a held-out test set and a statistical test.
- Paper: Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks"
  (LSTNet), SIGIR 2018, arXiv 1703.07015. Paper no. 13 on the course list; the list suggests
  "remove one component and measure what it contributes; compare with a simple linear model".
- Team of two. Each person owns one dataset; the code is shared.
  - Chandana: **Electricity** (321 customers, hourly, 26,304 steps)
  - Dinesh: **Solar-Energy** (137 PV plants, 10-minute steps, 52,560 steps)
- The repo will be public on GitHub and linked on LinkedIn, so keep it clean and readable.

## Deadlines (the exam is 100% of the grade)
| Date | What |
|---|---|
| End of week 1 | Repo, data explored, first neural-network notebook |
| End of week 2 (Sun 25.10) | Data preprocessed, baseline started, first experiment table |
| **Thu 29.10** | **Mid-course presentation, 4 min, graded:** our baseline vs the paper + planned improvement |
| End of week 3 (Sun 01.11) | Baseline reproduced and compared; improvement defined and justified |
| End of week 4 | Improvement implemented, controlled experiments, statistics, draft report |
| **Sun 08.11** | Hand in report + code + README |
| **Wed 11.11** | 8-min class presentation + 15-min solo technical defence on our own code |

"A working model alone is not enough": each of us must explain every line, including the
teammate's shared code, without AI help.

## Rules for working on this repo
- AI use is allowed but must be documented: log meaningful help in `AI_USAGE.md` (tool, what for).
- Prefer simple, readable PyTorch over clever abstractions. Comment the *why*, not the obvious.
- Explain new code to Chandana in plain language; she is new to RNNs/GRUs. After each new file,
  ask her defence-style questions about it.
- Never commit the datasets. `data/` is in `.gitignore`; use `scripts/download_data.sh`.
- Training runs on free Google Colab GPU: save checkpoints regularly, keep runs resumable.
- No `Co-Authored-By: Claude` lines in commits.
- **Keep the code simple**: plain loops and functions she can explain alone; no clever tricks.
- **Don't edit `AI_USAGE.md` until the end of the project**; it is written then.

## The story
The paper argues the linear AR component is needed because neural networks cannot follow
changes in scale (Sec. 4.6, Fig. 6). So: can a simple linear model match LSTNet, and does fixing
scale shift directly (RevIN) make the deep part earn its complexity?

## Data
- Source: https://github.com/laiguokun/multivariate-time-series-data (`electricity/`, `solar-energy/`).
- Format: gzipped text, one line per time step, comma-separated, one column per series.
- Shapes: electricity (26304, 321), solar (52560, 137).
- Split: 60% train / 20% validation / 20% test, **chronological**, never shuffled across splits.
  Splits are defined by the target index; input windows may reach back into the previous split.
- Normalisation: match the authors' code (`normalize=2`: each series divided by its max |value|).
  That max is taken over the **whole** series, test included: slight leakage. Match it for
  comparability and say so in the report.

## Task definition
- Input: last q steps of all n series (n × q).
- Target: all n values at **one** time point, h steps after the window ends.
  Horizon 24 = a single target 24 steps ahead, not a 24-step sequence.

## Model: LSTNet-Skip (not LSTNet-Attn)
1. CNN: filters of width ω and height n (spans all series), ReLU, no pooling.
2. GRU over CNN features; keep the last hidden state h_R.
3. Skip-GRU over the CNN output (length q−ω+1): keep the last floor((q−ω)/p)·p steps, reshape
   into p sequences, shared GRU; keep the p final states h_S.
4. Dense: h_D = W_R·h_R + Σ W_S,i·h_S,i + b → n outputs.
5. AR highway: linear on last q_ar raw values per series, weights shared across series → h_L.
6. Output: ŷ = h_D + h_L.

### Settings: verified in the authors' code (2026-10-20)
`ele.sh`: `python main.py --horizon 24 --data electricity.txt --output_fun Linear`, all else
default: window 168, hidCNN = hidRNN = 100, CNN_kernel 6, skip 24, hidSkip 5, highway_window 24,
dropout 0.2, Adam lr 1e-3, batch 128, clip 10 (total grad norm), 100 epochs, L1 loss, seed 54321.
Model selection: keep the weights with the best validation RSE; report test at those weights.
Details that matter (all reproduced in `src/train.py` / `src/model.py`):
- The training loss is L1 **summed** and computed on **de-normalised** values
  (output * scale vs Y * scale), so large customers dominate the loss.
- Skip: pt = (window − kernel) / skip; the code takes the last pt·skip CNN steps. With
  `--skip` parsed as float the original code gives pt = 6.75; we use floor = 6 (144 steps).
- `solar.sh` sets `--hidSkip 10` and does **not** set `--horizon`, so it runs at horizon 12.
  For horizon 24 on Solar, pass `--horizon 24 --hid-skip 10`.

### Paper vs code differences (document each in the report)
- GRU activation: paper says ReLU in the candidate state; the code uses standard `nn.GRU` (tanh).
- Skip hidden size: paper grid {20, 50, 100}; code default 5.
- Loss: L1 or L2, chosen on validation (Sec. 3.7).

## Metrics
- RSE = sqrt(Σ(y−ŷ)²) / sqrt(Σ(y−mean(y))²), pooled over all series and test times (lower is better).
- CORR = Pearson correlation per series, averaged over series (higher is better). Like the
  authors' code, drop series with zero variance in the test targets; report how many.
- Compute metrics in the original scale (de-normalised).

## Reproduction targets (Table 2, horizon 24)
| Dataset | LSTNet-Skip RSE | CORR | AR RSE |
|---|---|---|---|
| Electricity | 0.1007 | 0.9119 | 0.1054 |
| Solar-Energy | 0.4643 | 0.8870 | 0.8699 |

Note: Electricity LSTNet-Skip RSE is 0.1007 at both h = 12 and h = 24; use the h = 24 column.

## Experiments (Electricity, h = 24)
| # | Model | Role | Priority |
|---|---|---|---|
| 1 | AR: one linear model per series (paper trains n separate models, Sec. 4.1) | Paper baseline | Must |
| 2 | LSTNet-Skip | Reproduction | Must |
| 3 | LSTNet + RevIN (Kim et al. 2022) | **The improvement** | Must |
| 4 | DLinear (Zeng et al. 2023), same task/scaling/metrics/q | Simple linear comparison | Should |
| 5 | LSTNet without AR | Ablation of the paper's claim | Should (cut first) |

Solar (teammate): paper tuned p only in 2–64, but one day = 144 steps. With q = 168, p = 144
gives 1-step skip sequences, so q ≥ 144·k + ω is needed (memory-heavy). Watch night zeros and
constant plants.

### Solar handoff (Dinesh, from 2026-10-19)
Solar runs on Dinesh's machine, not Chandana's laptop (7.7 GB RAM). The shared code already
supports Solar (`STEPS_PER_DAY["solar"] = 144` in `src/data.py`). His tasks, in order:
1. `bash scripts/download_data.sh`; check shape (52560, 137).
2. `notebooks/01_explore_solar.ipynb`, same structure as `01_explore_electricity.ipynb`.
3. `python src/baselines.py --data solar --horizon 24` (slower than Electricity), then add a
   Solar section to `results/experiments.md`.
4. Log AI use in `AI_USAGE.md`. Dinesh must be able to explain every line himself.

Early findings from a first look on 2026-10-19 (re-check them in the notebook):
- The split is also a season split: train = days 0–219 (Jan–early Aug), test = days 292–365
  (late Oct–Dec).
- 55% of values are 0; at 49.5% of steps all plants are 0 (night). No constant plants.
- Median autocorrelation 0.85 at 144 steps (1 day), only 0.32 at 24 steps (the horizon).
- Every plant produces less in test: median test/train mean ratio 0.71 (range 0.62–0.83).
  Strong distribution shift, so RevIN has more to gain on Solar than on Electricity.
- No plant has its maximum in the test period, so the authors' scaling leaks nothing here.
- AR per plant: validation RSE kept falling up to L = 512 (0.4064, below LSTNet-Skip's
  published test RSE 0.4643). The run was stopped before the test score: re-run to confirm.

## Statistics
- **10 seeds per model**, paired by seed (same seed → same init and batch order).
  With 5 seeds Wilcoxon can never reach p < 0.05. Report mean ± std + paired test.
- **Diebold–Mariano** on per-window test loss (squared error summed over series), seed-averaged
  forecasts, Newey–West variance with lag h−1 = 23, Harvey–Leybourne–Newbold correction.
- **Holm correction** over planned comparisons (each model vs LSTNet-Skip).

## Structure
```
lstnet-reproduction/
├── README.md, CLAUDE.md, AI_USAGE.md, requirements.txt
├── scripts/download_data.sh
├── notebooks/01_explore_electricity.ipynb, 01_explore_solar.ipynb
├── src/data.py, model.py, metrics.py, train.py, baselines.py
├── tests/test_data_and_metrics.py, test_model.py   (pytest tests/)
├── notebooks/03_train_on_colab.ipynb   (full training on a Colab GPU)
└── results/
```
CLI goal: `python src/train.py --data electricity --horizon 24 --skip 24 --seed 0`

## Plan
- **Week 1 catch-up (to Tue 20.10):** repo, data, exploration notebook (daily/weekly
  autocorrelation as in Fig. 3c, zero/constant series, scale changes), first tiny PyTorch network.
- **Week 2 (to 25.10):** `data.py`, `metrics.py` (tested on a toy array), AR baseline → first
  experiment table; start `model.py` part by part, checking tensor shapes. Consultation
  questions: is "close to 0.1007" enough? Is RevIN accepted as the improvement?
- **Week 3:** one LSTNet run close to target by Tue 27.10 (if off: check horizon, output_fun,
  loss, normalisation, checkpoint choice). Thu 29.10 slides. Then RevIN, DLinear, ablation;
  start multi-seed Colab runs (~10 GPU hours for Electricity, spread over sessions).
- **Week 4:** finish seeds, statistics, plots, report draft by Thu 05.11, hand-in Sun 08.11.
- **09–11.11:** presentation and defence practice; each explains the other's code.

## Status
- 2026-10-18: repo created, data downloaded, Electricity exploration notebook done.
- 2026-10-18: `02_first_network.ipynb`: one shared neuron (q = 24, h = 24), test RSE 0.1032 /
  CORR 0.8698 (one seed). "Same hour yesterday" alone gives RSE 0.1048, better than the paper's
  AR (0.1054); LSTNet-Skip (0.1007) is only ~4% better than that rule. Week-1 milestone done.
- 2026-10-18: `src/data.py`, `src/metrics.py` + 9 passing toy tests; `src/baselines.py`.
  AR per customer, L = 256 (chosen on valid from 2^0..2^9): test RSE 0.0981 / CORR 0.9231,
  better than the paper's AR (0.1054) AND LSTNet-Skip (0.1007). Cross-checked with an
  independent implementation. Same hour yesterday: 0.1048. Table in `results/experiments.md`.
  LSTNet now has to beat 0.0981 for the deep part to earn its complexity.
- 2026-10-20: `src/model.py` (LSTNet-Skip) + `tests/test_model.py` (15 tests pass; our
  forward pass matches the authors' code translated line by line). `src/train.py`: authors'
  training loop, resumable checkpoint `<checkpoint-dir>/<run>.pt`, `--no-ar` ablation flag,
  result in `results/runs/<run>.json`. Smoke test on CPU OK (incl. resume); ~2 min/epoch on CPU.
  `notebooks/03_train_on_colab.ipynb`: full training on a Colab GPU, checkpoints on Drive.
- 2026-10-20: `src/neuron_baseline.py`: first baseline with 3 seeds (course week-2 milestone),
  test RSE 0.1043 ± 0.0003. Results table now has a Seeds column.
- 2026-10-20: all code simplified for the defence (plain loops, fewer options, no
  sliding_window_view / RNG-state resume); results unchanged, tests pass.
  Hyperparameters other than window/skip/hid-skip are fixed in `model.py` defaults.
- Next: first full LSTNet run on Colab (seed 0) and compare with 0.1007 and our AR 0.0981.
- 2026-10-18: public repo github.com/Chandana18G/lstnet-reproduction created. Chandana chose
  **not** to add it to the portfolio or profile README yet: do that once LSTNet is reproduced.
