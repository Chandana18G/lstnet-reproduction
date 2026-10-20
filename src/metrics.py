"""The paper's metrics (Sec. 4.2). Inputs: arrays of shape (time steps, series), in original units."""
import numpy as np


def rse(pred, true):
    """Root relative squared error (lower is better).

    Our squared error compared with the squared error of always predicting the overall mean.
    """
    error = np.sqrt(((true - pred) ** 2).sum())
    spread = np.sqrt(((true - true.mean()) ** 2).sum())
    return error / spread


def corr(pred, true):
    """Correlation between prediction and truth for each series, averaged (higher is better).

    Series that never change in `true` are left out, as in the authors' code: their
    correlation is not defined. Returns (average correlation, number of series left out).
    """
    keep = true.std(axis=0) != 0
    p = pred[:, keep]
    t = true[:, keep]
    cov = ((p - p.mean(axis=0)) * (t - t.mean(axis=0))).mean(axis=0)
    c = cov / (p.std(axis=0) * t.std(axis=0))
    return float(c.mean()), int((~keep).sum())


def window_losses(pred, true):
    """Squared error summed over series, one number per time step (for the Diebold-Mariano test)."""
    return ((true - pred) ** 2).sum(axis=1)
