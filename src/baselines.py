"""Non-deep baselines: seasonal naive forecasts and the paper's AR model.

AR as in the paper (Sec. 4.1): one separate linear model per series ("we just trained n models
independently"). For horizon h and lag order L, series j is predicted as

    y_j(t) = b_j + w_j,1 * y_j(t-h-L+1) + ... + w_j,L * y_j(t-h)

fitted by ordinary least squares on the training targets. L is chosen on validation RSE from
the paper's window grid 2^0 .. 2^9 (Sec. 4.4); one L for all series.

Usage:  python src/baselines.py --data electricity --horizon 24
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from data import load_raw, max_scale, split_targets
from metrics import corr, rse

RESULTS = Path(__file__).resolve().parent.parent / "results"


def seasonal_naive(data, targets, lag):
    """Predict each value by the value `lag` steps earlier (e.g. 24 = same hour yesterday)."""
    return data[targets - lag]


def fit_ar(series, train_targets, L, h):
    """Least-squares AR(L) for one series. Returns weights (L,) and intercept."""
    windows = sliding_window_view(series, L)                 # row k = series[k : k+L]
    X = windows[train_targets - h - L + 1]                   # inputs end exactly h before t
    X = np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)])  # column of ones -> intercept
    coef, *_ = np.linalg.lstsq(X, series[train_targets], rcond=None)
    return coef[:-1], coef[-1]


def predict_ar(series, targets, w, b, h):
    windows = sliding_window_view(series, len(w))
    return windows[targets - h - len(w) + 1] @ w + b


def ar_forecasts(data, train_t, eval_sets, L, h):
    """Fit one AR(L) per series on train_t; return predictions for each target set."""
    n = data.shape[1]
    train_t = train_t[train_t >= L + h - 1]                  # need L full input steps
    preds = [np.empty((len(t), n), dtype=np.float64) for t in eval_sets]
    for j in range(n):
        series = data[:, j].astype(np.float64)
        w, b = fit_ar(series, train_t, L, h)
        for p, t in zip(preds, eval_sets):
            p[:, j] = predict_ar(series, t, w, b, h)
    return preds


def scores(pred_scaled, targets, data, scale):
    """RSE and CORR in original units (kWh for electricity)."""
    pred, true = pred_scaled * scale, data[targets] * scale
    c, dropped = corr(pred, true)
    return rse(pred, true), c, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="electricity", choices=["electricity", "solar"])
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()
    h = args.horizon

    data, scale = max_scale(load_raw(args.data))
    # q only sets where the first training target lies; use the largest L of the grid so
    # every model is evaluated on exactly the same validation and test targets.
    lag_grid = [2 ** k for k in range(10)]
    tr, va, te = split_targets(len(data), q=max(lag_grid), h=h)
    tr_all, _, _ = split_targets(len(data), q=1, h=h)        # AR(L) may use earlier targets

    rows = []
    pred_dir = RESULTS / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Seasonal naive: only lags we would actually have observed (lag >= h).
    for name, lag in [("naive: same hour yesterday", 24), ("naive: same hour last week", 168)]:
        if lag < h:
            continue
        v = scores(seasonal_naive(data, va, lag), va, data, scale)
        pred_te = seasonal_naive(data, te, lag)
        t = scores(pred_te, te, data, scale)
        rows.append({"model": name, "setting": f"lag {lag}", "params": 0,
                     "valid_rse": v[0], "valid_corr": v[1], "test_rse": t[0], "test_corr": t[1],
                     "corr_dropped": t[2]})
        np.save(pred_dir / f"{args.data}_h{h}_naive_lag{lag}.npy", pred_te * scale)

    # AR: choose the lag order L on validation RSE.
    print(f"AR per series, {args.data}, horizon {h}")
    best = None
    for L in lag_grid:
        (pv,) = ar_forecasts(data, tr_all, [va], L, h)
        v_rse = scores(pv, va, data, scale)[0]
        print(f"  L = {L:3d}   valid RSE {v_rse:.4f}")
        if best is None or v_rse < best[1]:
            best = (L, v_rse)
    L = best[0]
    pv, pt = ar_forecasts(data, tr_all, [va, te], L, h)
    v, t = scores(pv, va, data, scale), scores(pt, te, data, scale)
    rows.append({"model": "AR, one model per series", "setting": f"L = {L} (best on valid)",
                 "params": (L + 1) * data.shape[1],
                 "valid_rse": v[0], "valid_corr": v[1], "test_rse": t[0], "test_corr": t[1],
                 "corr_dropped": t[2]})
    np.save(pred_dir / f"{args.data}_h{h}_ar_L{L}.npy", pt * scale)

    out = RESULTS / f"baselines_{args.data}_h{h}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})

    print(f"\n{'model':<30}{'valid RSE':>10}{'valid CORR':>11}{'test RSE':>10}{'test CORR':>10}")
    for r in rows:
        print(f"{r['model']:<30}{r['valid_rse']:>10.4f}{r['valid_corr']:>11.4f}"
              f"{r['test_rse']:>10.4f}{r['test_corr']:>10.4f}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
