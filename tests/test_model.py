"""Shape and correctness tests for src/model.py.  Run: pytest tests/"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import DLinear, LSTNet, RevIN, skip_sequences  # noqa: E402


def test_output_shape():
    model = LSTNet(n=7, window=48, hid_cnn=8, hid_rnn=6, hid_skip=3, kernel=6, skip=12, ar_window=10)
    x = torch.rand(5, 48, 7)
    assert model(x).shape == (5, 7)


def test_skip_sequences_hold_one_phase_each():
    # Time index as value: sequence j must contain the steps j, j+p, j+2p, ... of the tail.
    B, C, L, p = 2, 3, 13, 4                       # pt = (13 - 1) // 4 = 3 -> last 12 steps
    c = torch.arange(L, dtype=torch.float32).repeat(B, C, 1)
    s = skip_sequences(c, p)
    assert s.shape == (B * p, 3, C)
    assert s[0, :, 0].tolist() == [1, 5, 9]        # phase 0 of the tail (steps 1..12)
    assert s[3, :, 0].tolist() == [4, 8, 12]       # phase 3: ends with the newest step
    assert s[4, :, 0].tolist() == [1, 5, 9]        # second example starts again at phase 0


def test_window_too_short_for_skip():
    with pytest.raises(ValueError):
        LSTNet(n=3, window=20, kernel=6, skip=24)


def test_no_ar_and_no_skip_variants_run():
    x = torch.rand(4, 48, 5)
    assert LSTNet(n=5, window=48, hid_cnn=4, hid_rnn=4, skip=12, use_ar=False)(x).shape == (4, 5)
    assert LSTNet(n=5, window=48, hid_cnn=4, hid_rnn=4, skip=0)(x).shape == (4, 5)


def authors_forward(m, x, P, Ck, skip, hw):
    """The authors' forward pass (models/LSTNet.py), translated line by line to current
    PyTorch with sequence-first GRUs, using the weights of our model `m`."""
    batch_size, n = x.size(0), x.size(2)
    pt = (P - Ck) // skip
    c = x.view(-1, 1, P, n)
    c = F.relu(m.conv(c))
    c = torch.squeeze(c, 3)
    r = c.permute(2, 0, 1).contiguous()
    _, r = m.gru(r.permute(1, 0, 2))               # our GRU is batch_first
    r = torch.squeeze(r, 0)
    s = c[:, :, int(-pt * skip):].contiguous()
    s = s.view(batch_size, c.size(1), pt, skip)
    s = s.permute(2, 0, 3, 1).contiguous()
    s = s.view(pt, batch_size * skip, c.size(1))
    _, s = m.gru_skip(s.permute(1, 0, 2))
    s = s.view(batch_size, skip * m.hid_skip)
    r = torch.cat((r, s), 1)
    res = m.dense(r)
    z = x[:, -hw:, :]
    z = z.permute(0, 2, 1).contiguous().view(-1, hw)
    z = m.ar(z).view(-1, n)
    return res + z


def test_matches_authors_forward_pass():
    torch.manual_seed(0)
    P, Ck, skip, hw, n = 50, 6, 12, 10, 9
    m = LSTNet(n=n, window=P, hid_cnn=8, hid_rnn=7, hid_skip=3, kernel=Ck, skip=skip,
               ar_window=hw, dropout=0.0).eval()
    x = torch.rand(4, P, n)
    with torch.no_grad():
        assert torch.allclose(m(x), authors_forward(m, x, P, Ck, skip, hw), atol=1e-6)


def test_parameter_count_for_electricity():
    # Conv 100*6*321+100, GRU 3*(100*100+100*100+2*100), skip-GRU 3*(5*100+5*5+2*5),
    # dense (100+24*5)*321+321, AR 24+1
    m = LSTNet(n=321)
    expected = (100 * 6 * 321 + 100) + 3 * (2 * 100 * 100 + 200) + 3 * (5 * 100 + 25 + 10) \
        + (100 + 24 * 5) * 321 + 321 + 25
    assert sum(p.numel() for p in m.parameters()) == expected


def test_revin_round_trip():
    # Normalising and then denormalising the last value of a window gives it back unchanged.
    revin = RevIN(n=3)
    x = torch.rand(4, 10, 3) * 50 + 7
    z = revin.normalise(x)
    assert torch.allclose(z.mean(dim=1), torch.zeros(4, 3), atol=1e-4)    # each window: mean 0
    assert torch.allclose(revin.denormalise(z[:, -1, :]), x[:, -1, :], atol=1e-3)


def test_revin_lstnet_runs_and_ignores_level():
    # With RevIN, adding a constant to the input adds the same constant to the output.
    torch.manual_seed(0)
    m = LSTNet(n=5, window=48, hid_cnn=4, hid_rnn=4, skip=12, use_revin=True).eval()
    x = torch.rand(3, 48, 5)
    with torch.no_grad():
        assert torch.allclose(m(x + 10.0), m(x) + 10.0, atol=1e-3)


def test_dlinear_shape_and_trend():
    m = DLinear(window=48)
    x = torch.rand(6, 48, 4)
    assert m(x).shape == (6, 4)
    # The moving average of a constant series is the same constant
    const = torch.full((1, 2, 48), 3.0)
    assert torch.allclose(m.moving_average(const), const)
