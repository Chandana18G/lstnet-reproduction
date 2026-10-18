"""Evaluation metrics of the LSTNet paper (Sec. 4.2), computed on de-normalised values.

Both functions take arrays of shape (number of target steps, number of series).
"""
import numpy as np


def rse(pred, true):
    """Root relative squared error (lower is better), pooled over all series and steps.

    = total squared error / total squared distance of the truth from its overall mean.
    1.0 means "no better than always predicting the overall mean".
    """
    return np.sqrt(((true - pred) ** 2).sum()) / np.sqrt(((true - true.mean()) ** 2).sum())


def corr(pred, true):
    """Pearson correlation per series, averaged over series (higher is better).

    A series that is constant in `true` has no defined correlation; like the authors' code we
    leave it out of the average. Returns (mean correlation, number of series left out).
    A constant *prediction* gives a correlation of 0 for that series instead of a crash.
    """
    sp, st = pred.std(axis=0), true.std(axis=0)
    keep = st != 0
    cov = ((pred - pred.mean(axis=0)) * (true - true.mean(axis=0))).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(sp > 0, cov / (sp * st), 0.0)
    return float(c[keep].mean()), int((~keep).sum())


def window_losses(pred, true):
    """Squared error summed over series, one number per target step.

    This is the loss series for the Diebold-Mariano test later on.
    """
    return ((true - pred) ** 2).sum(axis=1)
