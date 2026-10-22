"""Tests for the statistical tests in src/stats.py.  Run: pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stats import diebold_mariano, holm  # noqa: E402


def test_holm_by_hand():
    # Sorted: 0.01 * 3 = 0.03, 0.02 * 2 = 0.04, 0.04 * 1 = 0.04
    assert holm(np.array([0.04, 0.01, 0.02])).tolist() == pytest.approx([0.04, 0.03, 0.04])
    assert holm(np.array([0.9, 0.8])).tolist() == pytest.approx([1.0, 1.0])   # capped at 1


def test_dm_detects_a_clearly_better_forecast():
    rng = np.random.default_rng(0)
    loss_b = rng.normal(10, 1, size=2000) ** 2
    loss_a = loss_b * 0.5                               # A always has half the loss of B
    dm, p = diebold_mariano(loss_a, loss_b, h=24)
    assert dm < 0 and p < 0.001


def test_dm_no_difference_for_noise():
    rng = np.random.default_rng(1)
    loss_a = rng.normal(5, 1, size=2000)
    loss_b = rng.normal(5, 1, size=2000)
    _, p = diebold_mariano(loss_a, loss_b, h=24)
    assert p > 0.05
