"""Error analysis for report section 6: where do the models make their test errors?

Uses the forecast averaged over the seeds of each model (as Diebold-Mariano in stats.py), so the
RSE here is a little lower than the mean RSE over seeds. The data files have no timestamps, so
"time of day" assumes that the file starts at midnight.

Usage:  python src/error_analysis.py --data electricity --horizon 24 --pred-dir checkpoints
        -> results/error_analysis_<data>_h<h>.md, results/figures/error_by_time_<data>.png
"""
import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from data import STEPS_PER_DAY, load_raw, max_scale, split_targets
from metrics import corr, rse

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"electricity": ["lstnet", "lstnet_revin", "lstnet_ar168"],
          "solar": ["lstnet", "lstnet_revin", "lstnet_noar", "lstnet_skip144"]}
BLOCKS = 6                                     # the day is cut into 6 blocks of 4 hours


def run_name_start(data_name, h, model):
    """Start of the run names of one model. All models use skip 24, except the one-day skip
    (Solar improvement: "lstnet_skip144" = LSTNet trained with --skip 144 --window 438)."""
    if model == "lstnet_skip144":
        return f"{data_name}_h{h}_lstnet_skip144"
    return f"{data_name}_h{h}_{model}_skip24"


def mean_forecast(pred_dir, data_name, h, model):
    """Average test forecast over all seeds of one model. Returns (forecast, number of seeds)."""
    files = glob.glob(f"{pred_dir}/{run_name_start(data_name, h, model)}_seed*_test_predictions.npy")
    if not files:
        return None, 0
    return np.mean([np.load(f) for f in files], axis=0), len(files)


def rse_per_seed(pred_dir, data_name, h, model, true, keep):
    """Test RSE of every seed of one model, using only the series in `keep`."""
    files = sorted(glob.glob(f"{pred_dir}/{run_name_start(data_name, h, model)}_seed*_test_predictions.npy"))
    return np.array([rse(np.load(f)[:, keep], true[:, keep]) for f in files])


def share(part, total):
    return f"{100 * part / total:.1f} %"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="electricity", choices=["electricity", "solar"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--pred-dir", default=str(ROOT / "checkpoints"))
    args = parser.parse_args()
    h = args.horizon

    raw = load_raw(args.data)
    data, scale = max_scale(raw)
    train, _, test = split_targets(len(data), 1, h)
    true = data[test] * scale                  # (test steps, series), original units
    models, preds = [], {}
    for m in MODELS[args.data]:
        forecast, seeds = mean_forecast(args.pred_dir, args.data, h, m)
        if seeds:
            models.append(m)
            preds[m] = forecast
    errors = {m: (true - preds[m]) ** 2 for m in models}       # squared error per step and series
    base = models[0]
    header = "| " + " | ".join(models) + " |"
    line = "|---" * (len(models) + 1) + "|"
    lines = [f"# Error analysis: {args.data}, horizon {h}", "",
             "Forecast averaged over the seeds of each model. Shares are shares of the model's own "
             "total squared error, which is what RSE is built from.", ""]

    # 1. Overall
    lines += ["## 1. Overall", "", "| Metric " + header, line]
    lines.append("| RSE | " + " | ".join(f"{rse(preds[m], true):.4f}" for m in models) + " |")
    lines.append("| CORR | " + " | ".join(f"{corr(preds[m], true)[0]:.4f}" for m in models) + " |")

    # 2. Large against small series: RSE pools squared errors, so large series dominate it
    size = true.mean(axis=0)
    largest = np.argsort(size)[::-1][:10]
    rest = np.setdiff1d(np.arange(true.shape[1]), largest)
    lines += ["", "## 2. The 10 largest series against the rest", "", "| Squared error " + header, line]
    for name, group in [("10 largest series", largest), ("all other series", rest)]:
        cells = [f"{errors[m][:, group].sum():.3e} ({share(errors[m][:, group].sum(), errors[m].sum())})"
                 for m in models]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", f"The 10 largest series hold {share(true[:, largest].sum(), true.sum())} of the total test "
              "load or output."]

    # The single largest series, and every result again without it
    top = largest[0]
    others = np.arange(true.shape[1]) != top
    lines += ["", f"## 2b. The single largest series (#{top})", "",
              f"It holds {share(true[:, top].sum(), true.sum())} of the total test load or output.", "",
              "| " + header, line]
    lines.append(f"| Share of squared error from #{top} | "
                 + " | ".join(share(errors[m][:, top].sum(), errors[m].sum()) for m in models) + " |")
    lines.append(f"| Test RSE without #{top} (seed-averaged forecast) | "
                 + " | ".join(f"{rse(preds[m][:, others], true[:, others]):.4f}" for m in models) + " |")
    without = {m: rse_per_seed(args.pred_dir, args.data, h, m, true, others) for m in models}
    lines.append(f"| Test RSE without #{top} (mean ± std over seeds) | "
                 + " | ".join(f"{without[m].mean():.4f} ± {without[m].std(ddof=1):.4f}" for m in models) + " |")
    cells = ["–"] + [f"{stats.ttest_rel(without[m], without[base]).pvalue:.4f}" for m in models[1:]]
    lines.append(f"| Paired t-test against {base} without #{top} (raw p) | " + " | ".join(cells) + " |")
    cells = ["–"] + [f"{(errors[m].sum(axis=0) < errors[base].sum(axis=0)).sum()} of {true.shape[1]}" for m in models[1:]]
    lines.append(f"| Series with smaller squared error than {base} | " + " | ".join(cells) + " |")

    # 3. Series whose level shifts between training and test against the others
    train_end = train[-1] + 1
    train_mean = raw[:train_end].mean(axis=0)
    test_mean = raw[test].mean(axis=0)
    ok = train_mean > 0
    ratio = np.ones_like(train_mean)
    ratio[ok] = test_mean[ok] / train_mean[ok]
    shifted = (ratio < 0.8) | (ratio > 1.2)
    lines += ["", "## 3. Series whose level changes by more than 20 % from training to test", "",
              "| Squared error " + header, line]
    for name, group in [(f"changed ({shifted.sum()} series)", shifted), (f"unchanged ({(~shifted).sum()} series)", ~shifted)]:
        cells = [f"{errors[m][:, group].sum():.3e} ({share(errors[m][:, group].sum(), errors[m].sum())})"
                 for m in models]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    # 4. Time of day
    per_day = STEPS_PER_DAY[args.data]
    position = test % per_day                  # position of each target within its day
    block = position * BLOCKS // per_day
    lines += ["", "## 4. Time of day (4-hour blocks)", "", "| Hours " + header, line]
    for b in range(BLOCKS):
        chosen = block == b
        cells = [share(errors[m][chosen].sum(), errors[m].sum()) for m in models]
        lines.append(f"| {4 * b:02d}–{4 * (b + 1):02d} | " + " | ".join(cells) + " |")

    # 5. Steps where everything is zero (Solar: night)
    night = (true == 0).all(axis=1)
    if night.any():
        lines += ["", f"## 5. Steps where all series are 0 ({share(night.sum(), len(night))} of test steps)", "",
                  "| " + header, line]
        cells = [share(errors[m][night].sum(), errors[m].sum()) for m in models]
        lines.append("| Share of squared error | " + " | ".join(cells) + " |")

    # 6. The worst moments and the worst series of the baseline
    per_step = errors[base].sum(axis=1)
    worst_steps = np.argsort(per_step)[::-1]
    top = worst_steps[:max(1, len(per_step) // 100)]
    keep = true.std(axis=0) != 0
    centred_p = preds[base] - preds[base].mean(axis=0)
    centred_t = true - true.mean(axis=0)
    p_std = preds[base].std(axis=0)
    series_corr = np.full(true.shape[1], np.nan)
    valid = keep & (p_std != 0)
    series_corr[valid] = (centred_p[:, valid] * centred_t[:, valid]).mean(axis=0) / (p_std[valid] * true[:, valid].std(axis=0))
    worst_series = np.argsort(np.nan_to_num(series_corr, nan=2.0))[:5]
    lines += ["", f"## 6. Worst cases of {base}", "",
              f"* The worst 1 % of test steps hold {share(per_step[top].sum(), per_step.sum())} of the squared error.",
              "* The 5 worst test steps (file step, position in the day): "
              + ", ".join(f"{test[i]} ({test[i] % per_day})" for i in worst_steps[:5]) + ".",
              f"* Correlation per series: median {np.nanmedian(series_corr):.3f}; the 5 worst series: "
              + ", ".join(f"#{j} ({series_corr[j]:.3f})" for j in worst_series) + "."]

    # Figure: squared error by position in the day
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for m in models:
        by_position = [errors[m][position == k].sum() for k in range(per_day)]
        ax.plot(np.arange(per_day) * 24 / per_day, by_position, label=m)
    ax.set_xlabel("hour of the day (assuming the file starts at midnight)")
    ax.set_ylabel("total squared error")
    ax.set_title(f"{args.data}: where in the day the test error falls")
    ax.legend(frameon=False)
    plt.tight_layout()
    (ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figures" / f"error_by_time_{args.data}.png", dpi=150)

    out = ROOT / "results" / f"error_analysis_{args.data}_h{h}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
