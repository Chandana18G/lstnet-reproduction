"""Checks of our AR baseline, as asked in the professor's feedback (27.10.2026).

Our AR (one linear model per customer) reaches test RSE 0.0981 on Electricity at horizon 24;
the paper's AR reaches 0.1054. This script checks the four possible reasons:

  1. RSE exactly as in Eq. 10: our formula vs the authors' code (main.py / utils.py).
  2. The horizon: AR at horizons 3, 6, 12, 24, compared with the paper's AR row of Table 2.
     An off-by-one in the horizon would shift all of them.
  3. The scaling: AR with the scale computed on the training part only gives the same result.
  4. The window: test RSE for every window length L of the paper's grid 1, 2, 4, ..., 512.

Usage:  python src/check_ar.py      -> results/ar_checks_electricity.md   (about 1.5 hours on a laptop)
"""
from pathlib import Path

import numpy as np

from baselines import make_inputs
from data import load_raw, max_scale, split_targets
from metrics import rse

ROOT = Path(__file__).resolve().parent.parent
PAPER_AR = {3: 0.0995, 6: 0.1035, 12: 0.1050, 24: 0.1054}     # Table 2, Electricity, AR row
GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def fit_and_predict(data, train, valid, test, L, h):
    """One least-squares AR(L) per series, fitted once; predictions for validation and test."""
    train = train[train >= L + h - 1]
    pred_valid = np.zeros((len(valid), data.shape[1]))
    pred_test = np.zeros((len(test), data.shape[1]))
    for j in range(data.shape[1]):
        series = data[:, j].astype(np.float64)
        coef = np.linalg.lstsq(make_inputs(series, train, L, h), series[train], rcond=None)[0]
        pred_valid[:, j] = make_inputs(series, valid, L, h) @ coef
        pred_test[:, j] = make_inputs(series, test, L, h) @ coef
    return pred_valid, pred_test


def authors_rse(pred, true):
    """RSE as computed in the authors' code: sqrt(mean squared error) / normal_std(test values)."""
    n = true.size
    mse = ((pred - true) ** 2).sum() / n                     # their total_loss / n_samples
    normal_std = true.std(ddof=1) * np.sqrt((n - 1) / n)     # their normal_std(): torch std * sqrt((n-1)/n)
    return np.sqrt(mse) / normal_std


def main():
    raw = load_raw("electricity")
    data, scale = max_scale(raw)
    lines = ["# Checks of our AR baseline (Electricity)", ""]

    # --- Check 3 first (cheap to explain): scaling on the training part only -----------------
    train, valid, test = split_targets(len(data), 1, 24)
    train_end = valid[0]
    scale_train = np.abs(raw[:train_end]).max(axis=0)
    scale_train[scale_train == 0] = 1.0
    data_train_scaled = (raw / scale_train).astype(np.float32)

    # --- Check 2 and 4: horizons 3, 6, 12, 24, every window L ----------------------------------
    lines += ["## Check 2 and 4: horizon and window", "",
              "Test RSE of our AR for every window length L (one model per customer, least squares).",
              "**Bold** = the L chosen on validation (as in the paper, Sec. 4.4).", "",
              "| Horizon | " + " | ".join(f"L={L}" for L in GRID) + " | chosen L | ours (chosen L) | paper AR |",
              "|---|" + "---|" * (len(GRID) + 3)]
    for h in [3, 6, 12, 24]:
        train, valid, test = split_targets(len(data), 1, h)
        true_valid, true_test = data[valid] * scale, data[test] * scale
        valid_rse, test_rse = {}, {}
        for L in GRID:
            pv, pt = fit_and_predict(data, train, valid, test, L, h)
            valid_rse[L] = rse(pv * scale, true_valid)
            test_rse[L] = rse(pt * scale, true_test)
            print(f"h = {h:2d}  L = {L:3d}: valid RSE {valid_rse[L]:.4f}  test RSE {test_rse[L]:.4f}", flush=True)
            if h == 24 and L == 256:
                # Check 1: the two RSE formulas on the same predictions
                check1 = (rse(pt * scale, true_test), authors_rse(pt * scale, true_test))
                # Check 3: same model with the scale from the training part only
                _, pt2 = fit_and_predict(data_train_scaled, train, valid, test, L, h)
                check3 = rse(pt2 * scale_train, true_test)
        best = min(valid_rse, key=valid_rse.get)
        cells = [f"**{test_rse[L]:.4f}**" if L == best else f"{test_rse[L]:.4f}" for L in GRID]
        lines.append(f"| {h} | " + " | ".join(cells) + f" | {best} | {test_rse[best]:.4f} | {PAPER_AR[h]:.4f} |")

    # --- Check 1 and 3 results --------------------------------------------------------------
    lines += ["", "## Check 1: RSE exactly as in Eq. 10", "",
              "Same AR predictions (h = 24, L = 256), in original units (kWh), pooled over all customers",
              "and test hours, divided by the spread around the mean of the whole test set:", "",
              f"- our `metrics.rse`: {check1[0]:.6f}",
              f"- the authors' formula (`sqrt(MSE) / normal_std(test)`): {check1[1]:.6f}", "",
              "## Check 3: scaling on the training part only", "",
              f"- scale from the whole series (authors' code): test RSE {check1[0]:.6f}",
              f"- scale from the training part only: test RSE {check3:.6f}",
              "", "A least-squares fit per customer is unchanged when the customer is multiplied by a constant,",
              "so the scaling cannot change the AR result.", "",
              "## Off-by-one in the horizon", "",
              "`tests/test_data_and_metrics.py::test_window_and_target_are_h_steps_apart` checks that the newest",
              "input step is exactly h steps before the target. `make_inputs` uses the same rule",
              "(`series[t - h - L + 1 : t - h + 1]`)."]

    out = ROOT / "results" / "ar_checks_electricity.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
