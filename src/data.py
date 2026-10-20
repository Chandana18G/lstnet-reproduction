"""Loading, scaling, splitting and windowing the LSTNet datasets.

Follows the authors' code (github.com/laiguokun/LSTNet) so that our numbers are comparable:
  * each series is divided by its maximum absolute value (their `normalize=2`);
  * the split is 60% / 20% / 20% in time order, decided by the hour of the *target*;
  * one example = q past steps of all series -> all series h steps after the window.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILES = {
    "electricity": "electricity.txt.gz",
    "solar": "solar_AL.txt.gz",
}
STEPS_PER_DAY = {
    "electricity": 24,    # hourly
    "solar": 144,         # every 10 minutes
}


def load_raw(name):
    """Return the raw (time steps, series) array, e.g. (26304, 321) for electricity."""
    path = DATA_DIR / FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `bash scripts/download_data.sh` first")
    return np.loadtxt(path, delimiter=",")


def max_scale(raw):
    """Divide every series by its max |value| over the whole period, as the authors do.

    The max includes the test period (slight leakage, kept for comparability with the paper).
    Returns the scaled float32 array and the per-series scale to undo it: raw = scaled * scale.
    """
    scale = np.abs(raw).max(axis=0)
    scale[scale == 0] = 1.0          # an all-zero series would otherwise divide by zero
    return (raw / scale).astype(np.float32), scale


def split_targets(T, q, h, train=0.6, valid=0.2):
    """Target time indices for train, validation and test.

    A target at index t needs the input rows t-h-q+1 .. t-h, so the first usable target is
    q+h-1. Inputs of validation/test windows may reach back into the previous part: those rows
    are in the past, so no information from the future is used.
    """
    train_end = int(train * T)
    valid_end = int((train + valid) * T)
    return (
        np.arange(q + h - 1, train_end),
        np.arange(train_end, valid_end),
        np.arange(valid_end, T),
    )


def window_slice(t, q, h):
    """Rows that form the input window for the target at index t."""
    return slice(t - h - q + 1, t - h + 1)


class Windows(Dataset):
    """PyTorch dataset of (x, y) pairs, cut out on demand to save memory.

    x: (q, n) = q past steps of all n series;  y: (n,) = all series at the target step.
    """

    def __init__(self, data, targets, q, h):
        self.data = torch.from_numpy(data)
        self.targets = targets
        self.q, self.h = q, h

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        t = self.targets[i]
        return self.data[window_slice(t, self.q, self.h)], self.data[t]


def make_loaders(data, q, h, batch_size=128, seed=0, generator=None):
    """Train/valid/test DataLoaders. Only training batches are shuffled (within training).

    Pass your own `generator` to save and restore the shuffle order when resuming a run.
    """
    tr, va, te = split_targets(len(data), q, h)
    gen = generator or torch.Generator().manual_seed(seed)   # same seed -> same batch order
    return (
        DataLoader(Windows(data, tr, q, h), batch_size=batch_size, shuffle=True, generator=gen),
        DataLoader(Windows(data, va, q, h), batch_size=512),
        DataLoader(Windows(data, te, q, h), batch_size=512),
    )
