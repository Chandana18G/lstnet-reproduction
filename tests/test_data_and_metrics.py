"""Small hand-checkable tests for src/data.py and src/metrics.py.  Run: pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import Windows, max_scale, split_targets, window_slice  # noqa: E402
from metrics import corr, rse, window_losses  # noqa: E402


# ---------- data ----------

def test_window_and_target_are_h_steps_apart():
    # Toy data where each value is its own time index: easy to see which rows were taken.
    T, n, q, h = 50, 2, 4, 3
    data = np.repeat(np.arange(T, dtype=np.float32)[:, None], n, axis=1)
    t = 20
    x, y = Windows(data, np.array([t]), q, h)[0]
    assert x[:, 0].tolist() == [14, 15, 16, 17]      # t-h-q+1 .. t-h
    assert y.tolist() == [20, 20]
    assert y[0] - x[-1, 0] == h                       # newest input is exactly h before target


def test_split_is_chronological_and_complete():
    T, q, h = 100, 5, 3
    tr, va, te = split_targets(T, q, h)
    assert tr[0] == q + h - 1                         # first target with a full window
    assert tr[-1] < va[0] and va[-1] < te[0]          # strictly in time order
    assert te[-1] == T - 1
    assert len(va) == len(te) == 20


def test_first_window_starts_at_row_zero():
    q, h = 5, 3
    tr, _, _ = split_targets(100, q, h)
    assert window_slice(tr[0], q, h).start == 0


def test_max_scale_round_trip_and_zero_series():
    raw = np.array([[1.0, 0.0, -4.0], [2.0, 0.0, 2.0]])
    scaled, scale = max_scale(raw)
    assert scale.tolist() == [2.0, 1.0, 4.0]          # all-zero series gets scale 1, not 0
    assert np.abs(scaled).max() <= 1.0
    assert np.allclose(scaled * scale, raw)


# ---------- metrics ----------

def test_rse_perfect_and_mean_prediction():
    true = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]])
    assert rse(true, true) == 0.0
    assert rse(np.full_like(true, true.mean()), true) == pytest.approx(1.0)


def test_rse_by_hand():
    true = np.array([[0.0], [2.0]])                   # mean 1, squared distances 1 + 1 = 2
    pred = np.array([[1.0], [1.0]])                   # squared errors 1 + 1 = 2
    assert rse(pred, true) == pytest.approx(1.0)
    pred = np.array([[0.0], [1.0]])                   # squared error 1
    assert rse(pred, true) == pytest.approx(np.sqrt(1 / 2))


def test_corr_matches_numpy_and_drops_constant_series():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(30, 3))
    true[:, 2] = 7.0                                  # constant series -> must be dropped
    pred = true + rng.normal(scale=0.5, size=true.shape)
    value, dropped = corr(pred, true)
    expected = np.mean([np.corrcoef(pred[:, j], true[:, j])[0, 1] for j in range(2)])
    assert dropped == 1
    assert value == pytest.approx(expected)


def test_corr_perfect_and_opposite():
    true = np.array([[1.0], [2.0], [4.0]])
    assert corr(true, true)[0] == pytest.approx(1.0)
    assert corr(-true, true)[0] == pytest.approx(-1.0)
    assert corr(2 * true + 5, true)[0] == pytest.approx(1.0)   # scale and shift do not matter


def test_window_losses():
    true = np.array([[1.0, 1.0], [0.0, 0.0]])
    pred = np.array([[0.0, 3.0], [0.0, 0.0]])
    assert window_losses(pred, true).tolist() == [5.0, 0.0]
