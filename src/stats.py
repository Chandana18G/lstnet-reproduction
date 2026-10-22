"""Results table over seeds and statistical tests: is a model really better than LSTNet?

Two tests for each comparison "model vs LSTNet":
  1. Over seeds: test RSE of seed k of the model vs seed k of LSTNet (same seed = same start and
     batch order), paired t-test and Wilcoxon signed-rank test. Needs >= 6 seeds for Wilcoxon
     to be able to reach p < 0.05; we use 10.
  2. Diebold-Mariano: over the test hours. For each hour the loss = squared error summed over
     series, using the forecast averaged over the seeds. Neighbouring hours are correlated, so
     the variance uses Newey-West weights up to lag h - 1, with the Harvey-Leybourne-Newbold
     small-sample correction.
The p-values of all comparisons are corrected with Holm's method (we test several at once).

Inputs:
  results/runs/<run>.json                     written by train.py (one per model and seed)
  <pred-dir>/<run>_test_predictions.npy        written by train.py next to the checkpoints
  results/predictions/<data>_h<h>_ar.npy       written by baselines.py

Usage:  python src/stats.py --data electricity --horizon 24 --pred-dir checkpoints
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

from data import load_raw, max_scale, split_targets
from metrics import window_losses

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["lstnet", "lstnet_revin", "lstnet_noar", "dlinear"]


def diebold_mariano(loss_a, loss_b, h):
    """DM test that two forecasts are equally good. Negative statistic = A has smaller loss."""
    d = loss_a - loss_b                              # loss difference for every test hour
    T = len(d)
    d_centred = d - d.mean()
    variance = (d_centred ** 2).sum() / T            # autocovariance at lag 0
    for k in range(1, h):                            # lags 1 .. h-1, Newey-West weights
        weight = 1 - k / h
        variance += 2 * weight * (d_centred[k:] * d_centred[:-k]).sum() / T
    dm = d.mean() / np.sqrt(variance / T)
    dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)   # Harvey-Leybourne-Newbold
    p = 2 * stats.t.sf(abs(dm), df=T - 1)            # two-sided p-value, t distribution
    return dm, p


def holm(p_values):
    """Holm correction: sort p-values, multiply the i-th smallest by (m - i), keep them rising."""
    m = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(m)
    running_max = 0.0
    for i, idx in enumerate(order):
        running_max = max(running_max, (m - i) * p_values[idx])
        corrected[idx] = min(1.0, running_max)
    return corrected


def load_runs(data_name, h, model):
    """{seed: result dict} for one model, from results/runs/*.json."""
    runs = {}
    for f in glob.glob(str(ROOT / "results" / "runs" / f"{data_name}_h{h}_{model}_skip*_seed*.json")):
        r = json.loads(Path(f).read_text())
        seed = int(r["run"].split("_seed")[-1])
        runs[seed] = r
    return runs


def mean_forecast(pred_dir, runs):
    """Average of the test predictions over all seeds of one model."""
    preds = [np.load(Path(pred_dir) / f"{r['run']}_test_predictions.npy") for r in runs.values()]
    return np.mean(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="electricity", choices=["electricity", "solar"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--pred-dir", default=str(ROOT / "checkpoints"))
    args = parser.parse_args()
    h = args.horizon

    runs = {m: load_runs(args.data, h, m) for m in MODELS}
    runs = {m: r for m, r in runs.items() if r}      # only models that were trained
    if "lstnet" not in runs:
        print("no LSTNet runs found in results/runs/ yet")
        return

    lines = [f"# Results over seeds: {args.data}, horizon {h}", "",
             "| Model | Seeds | Test RSE (mean ± std) | Test CORR (mean ± std) |", "|---|---|---|---|"]
    for m, r in runs.items():
        rse_values = np.array([x["test_rse"] for x in r.values()])
        corr_values = np.array([x["test_corr"] for x in r.values()])
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
        lines.append(f"| {m} | {len(r)} | {rse_values.mean():.4f} ± {sd(rse_values):.4f} | "
                     f"{corr_values.mean():.4f} ± {sd(corr_values):.4f} |")

    # True test values, to compute the losses for Diebold-Mariano
    data, scale = max_scale(load_raw(args.data))
    _, _, test = split_targets(len(data), 1, h)
    true = data[test] * scale
    loss_lstnet = window_losses(mean_forecast(args.pred_dir, runs["lstnet"]), true)

    others = [m for m in runs if m != "lstnet"]
    rows, p_seed, p_dm = [], [], []
    for m in others:
        seeds = sorted(set(runs[m]) & set(runs["lstnet"]))          # seeds both models have
        a = np.array([runs[m][s]["test_rse"] for s in seeds])
        b = np.array([runs["lstnet"][s]["test_rse"] for s in seeds])
        p_t = stats.ttest_rel(a, b).pvalue if len(seeds) > 1 else np.nan
        p_w = stats.wilcoxon(a, b).pvalue if len(seeds) > 1 else np.nan
        dm, p = diebold_mariano(window_losses(mean_forecast(args.pred_dir, runs[m]), true),
                                loss_lstnet, h)
        rows.append((m, len(seeds), (a - b).mean(), p_t, p_w, dm))
        p_seed.append(p_t)
        p_dm.append(p)

    # The paper's AR baseline (no seeds): Diebold-Mariano only
    ar_file = ROOT / "results" / "predictions" / f"{args.data}_h{h}_ar.npy"
    if ar_file.exists():
        dm, p = diebold_mariano(window_losses(np.load(ar_file), true), loss_lstnet, h)
        rows.append(("AR (baselines.py)", 0, np.nan, np.nan, np.nan, dm))
        p_seed.append(np.nan)
        p_dm.append(p)

    p_seed_holm = holm(np.nan_to_num(np.array(p_seed), nan=1.0))
    p_dm_holm = holm(np.array(p_dm))
    lines += ["", "## Each model vs LSTNet", "",
              "Negative difference / DM statistic = the model is better (smaller error) than LSTNet.",
              "p-values after Holm correction; the raw t-test and Wilcoxon p-values are in brackets.", "",
              "| Model | Paired seeds | Mean RSE difference | Paired t-test p | Wilcoxon p | DM statistic | DM p |",
              "|---|---|---|---|---|---|---|"]
    for i, (m, k, diff, p_t, p_w, dm) in enumerate(rows):
        seed_part = "–" if k == 0 else f"{p_seed_holm[i]:.4f} ({p_t:.4f})"
        w_part = "–" if k == 0 else f"({p_w:.4f})"
        diff_part = "–" if k == 0 else f"{diff:+.4f}"
        lines.append(f"| {m} | {k} | {diff_part} | {seed_part} | {w_part} | {dm:+.2f} | {p_dm_holm[i]:.4f} |")

    out = ROOT / "results" / f"stats_{args.data}_h{h}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
