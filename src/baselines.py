"""Baselines without a neural network: two "copy the past" rules and the paper's AR model.

AR as in the paper (Sec. 4.1): one separate linear model per series. For series j:

    prediction(t) = b + w_1 * y(t-h-L+1) + ... + w_L * y(t-h)

fitted with least squares on the training targets. The number of past steps L is chosen on
validation from the paper's grid 1, 2, 4, ..., 512.

Usage:  python src/baselines.py --data electricity --horizon 24
"""
import argparse
from pathlib import Path

import numpy as np

from data import STEPS_PER_DAY, load_raw, max_scale, split_targets
from metrics import corr, rse

RESULTS = Path(__file__).resolve().parent.parent / "results"


def make_inputs(series, targets, L, h):
    """One row per target t: the L values that end h steps before t, plus a 1 for the intercept."""
    rows = []
    for t in targets:
        past = series[t - h - L + 1 : t - h + 1]
        rows.append(np.append(past, 1.0))
    return np.array(rows)


def ar_predict(data, train, targets, L, h):
    """Fit one AR model per series on `train`, predict `targets`. Returns (len(targets), n)."""
    train = train[train >= L + h - 1]                  # targets with L full past steps
    pred = np.zeros((len(targets), data.shape[1]))
    for j in range(data.shape[1]):
        series = data[:, j].astype(np.float64)
        X = make_inputs(series, train, L, h)
        coef = np.linalg.lstsq(X, series[train], rcond=None)[0]   # least squares
        pred[:, j] = make_inputs(series, targets, L, h) @ coef
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="electricity", choices=["electricity", "solar"])
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    h = args.horizon

    data, scale = max_scale(load_raw(args.data))
    train, valid, test = split_targets(len(data), 1, h)   # q = 1: AR drops what it can't use
    true_valid = data[valid] * scale                       # back to original units
    true_test = data[test] * scale
    results = []

    # 1. Copy the value from one day / one week earlier
    day = STEPS_PER_DAY[args.data]
    for name, lag in [("same time yesterday", day), ("same time last week", 7 * day)]:
        pred_valid = data[valid - lag] * scale
        pred_test = data[test - lag] * scale
        results.append((name, rse(pred_valid, true_valid), corr(pred_valid, true_valid)[0],
                        rse(pred_test, true_test), corr(pred_test, true_test)[0]))

    # 2. AR: try every L on validation, keep the best
    best_L, best_rse = None, float("inf")
    for L in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        pred_valid = ar_predict(data, train, valid, L, h) * scale
        valid_rse = rse(pred_valid, true_valid)
        print(f"AR L = {L:3d}: valid RSE {valid_rse:.4f}", flush=True)
        if valid_rse < best_rse:
            best_L, best_rse = L, valid_rse

    pred_valid = ar_predict(data, train, valid, best_L, h) * scale
    pred_test = ar_predict(data, train, test, best_L, h) * scale   # test: only once, at the end
    results.append((f"AR, L = {best_L}", rse(pred_valid, true_valid), corr(pred_valid, true_valid)[0],
                    rse(pred_test, true_test), corr(pred_test, true_test)[0]))

    # Save predictions (for the statistical tests later) and the table
    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    np.save(RESULTS / "predictions" / f"{args.data}_h{h}_ar.npy", pred_test)

    lines = ["model,valid_rse,valid_corr,test_rse,test_corr"]
    print(f"\n{'model':<22}{'valid RSE':>10}{'valid CORR':>11}{'test RSE':>10}{'test CORR':>10}")
    for name, v_rse, v_corr, t_rse, t_corr in results:
        print(f"{name:<22}{v_rse:>10.4f}{v_corr:>11.4f}{t_rse:>10.4f}{t_corr:>10.4f}")
        lines.append(f"{name},{v_rse:.4f},{v_corr:.4f},{t_rse:.4f},{t_corr:.4f}")
    (RESULTS / f"baselines_{args.data}_h{h}.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
