"""Load, scale, split and cut the data into windows, as in the authors' code.

One example = the last q steps of all series (input) and all series h steps later (target).
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILES = {"electricity": "electricity.txt.gz", "solar": "solar_AL.txt.gz"}
STEPS_PER_DAY = {"electricity": 24, "solar": 144}      # hourly / every 10 minutes


def load_raw(name):
    """The raw data: one row per time step, one column per series."""
    return np.loadtxt(DATA_DIR / FILES[name], delimiter=",")


def max_scale(raw):
    """Divide each series by its largest absolute value (the authors' normalize=2).

    Returns the scaled data and the scale, so that raw = scaled * scale.
    """
    scale = np.abs(raw).max(axis=0)
    scale[scale == 0] = 1.0                            # avoid dividing by zero
    return (raw / scale).astype(np.float32), scale


def split_targets(T, q, h):
    """Target time steps for training (first 60%), validation (next 20%) and test (last 20%).

    The first training target is q + h - 1: the first one with q full input steps before it.
    """
    train_end = int(0.6 * T)
    valid_end = int(0.8 * T)
    train = np.arange(q + h - 1, train_end)
    valid = np.arange(train_end, valid_end)
    test = np.arange(valid_end, T)
    return train, valid, test


class Windows(Dataset):
    """Gives one (input, target) pair at a time, cut out of the data only when needed."""

    def __init__(self, data, targets, q, h):
        self.data = torch.from_numpy(data)
        self.targets = targets
        self.q = q
        self.h = h

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        t = self.targets[i]
        start = t - self.h - self.q + 1
        end = t - self.h + 1                           # the newest input step is t - h
        x = self.data[start:end]                       # shape (q, n)
        y = self.data[t]                               # shape (n,)
        return x, y


def make_loaders(data, q, h, batch_size=128, seed=0):
    """DataLoaders for training (shuffled), validation and test (in time order).

    The seed fixes the shuffle order, so two models with the same seed see the same batches.
    """
    train, valid, test = split_targets(len(data), q, h)
    shuffle_order = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(Windows(data, train, q, h), batch_size=batch_size, shuffle=True,
                          generator=shuffle_order)
    valid_dl = DataLoader(Windows(data, valid, q, h), batch_size=512)
    test_dl = DataLoader(Windows(data, test, q, h), batch_size=512)
    return train_dl, valid_dl, test_dl
