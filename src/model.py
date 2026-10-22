"""LSTNet-Skip (Lai et al., SIGIR 2018), following the authors' PyTorch code.

Reference: github.com/laiguokun/LSTNet, models/LSTNet.py. Same layers, same order, same
dropout positions; written with batch_first GRUs and named steps so each part can be checked.

Shapes use: B = batch size, q = window, n = number of series, C = CNN filters.

    x (B, q, n)
    1. CNN       filters of height omega spanning all n series   -> c (B, C, q - omega + 1)
    2. GRU       over the CNN time steps, keep the last state     -> h_R (B, hid_rnn)
    3. Skip-GRU  p interleaved sequences, one per phase of the period (e.g. hour of day)
                                                                   -> h_S (B, p * hid_skip)
    4. Dense     [h_R, h_S] -> n outputs                           -> h_D (B, n)
    5. AR        one linear model on the last `ar_window` values, shared by all series
                                                                   -> h_L (B, n)
    output = h_D + h_L
"""
import torch
from torch import nn


def skip_sequences(c, p):
    """Cut the CNN output into p interleaved sequences for the skip-GRU.

    c: (B, C, L). Only the last pt * p steps are used, pt = (L - 1) // p, as in the authors'
    code (pt = (q - omega) / p). Sequence j holds steps j, j+p, j+2p, ... of that tail, so with
    p = 24 on hourly data each sequence sees one hour of the day across consecutive days.

    Returns (B * p, pt, C): one row per (example, phase), ready for a batch_first GRU.
    """
    B, C, L = c.shape
    pt = (L - 1) // p
    s = c[:, :, L - pt * p:]                  # (B, C, pt * p)   the last pt*p steps
    s = s.reshape(B, C, pt, p)                # time index = k * p + j  ->  [k, j]
    s = s.permute(0, 3, 2, 1)                 # (B, p, pt, C)    phase j first, then time k
    return s.reshape(B * p, pt, C)


class RevIN(nn.Module):
    """Reversible instance normalisation (Kim et al., ICLR 2022): our improvement.

    Before the model: each series in each input window is shifted to mean 0 and scaled to
    standard deviation 1, using only that window. After the model: the prediction is moved
    back to the window's own level and scale. So the network never has to learn the level of
    a series, only its shape; a change of level between training and test hurts less.
    """

    def __init__(self, n, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n))     # learnable scale per series
        self.beta = nn.Parameter(torch.zeros(n))     # learnable shift per series

    def normalise(self, x):                          # x: (B, q, n)
        self.mean = x.mean(dim=1, keepdim=True)      # (B, 1, n): mean of each window
        self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - self.mean) / self.std
        return x * self.gamma + self.beta

    def denormalise(self, y):                        # y: (B, n)
        y = (y - self.beta) / (self.gamma + self.eps)
        return y * self.std.squeeze(1) + self.mean.squeeze(1)


class LSTNet(nn.Module):
    def __init__(self, n, window=168, hid_cnn=100, hid_rnn=100, hid_skip=5, kernel=6,
                 skip=24, ar_window=24, dropout=0.2, use_ar=True, use_revin=False):
        super().__init__()
        self.n, self.window, self.skip, self.hid_skip = n, window, skip, hid_skip
        self.ar_window = ar_window if use_ar else 0
        self.revin = RevIN(n) if use_revin else None
        if skip > 0 and (window - kernel) // skip < 1:
            raise ValueError(f"window {window} too short for skip {skip}: no full skip sequence")

        # 1. CNN: a 2-D convolution over (time, series) whose height covers ALL n series,
        #    so each filter mixes every series at once and slides only along time.
        self.conv = nn.Conv2d(1, hid_cnn, kernel_size=(kernel, n))
        # 2. Recurrent part over the CNN features
        self.gru = nn.GRU(hid_cnn, hid_rnn, batch_first=True)
        # 3. Recurrent-skip part: one GRU shared by all p phase sequences
        if skip > 0:
            self.gru_skip = nn.GRU(hid_cnn, hid_skip, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # 4. Dense layer combining h_R and the p skip states into n outputs
        self.dense = nn.Linear(hid_rnn + (skip * hid_skip if skip > 0 else 0), n)
        # 5. Autoregressive "highway": the same weights for every series
        if self.ar_window > 0:
            self.ar = nn.Linear(self.ar_window, 1)

    def forward(self, x):                            # x: (B, q, n), already scaled
        B = x.shape[0]
        if self.revin is not None:                   # improvement: normalise each window
            x = self.revin.normalise(x)

        # 1. CNN -> (B, C, q - kernel + 1)
        c = torch.relu(self.conv(x.unsqueeze(1)))    # (B, C, q - kernel + 1, 1)
        c = self.dropout(c).squeeze(3)

        # 2. GRU over time; h has shape (1, B, hid_rnn): the last hidden state
        _, h = self.gru(c.permute(0, 2, 1))          # GRU wants (B, time, features)
        r = self.dropout(h.squeeze(0))

        # 3. Skip-GRU: last hidden state of each of the p phase sequences
        if self.skip > 0:
            _, hs = self.gru_skip(skip_sequences(c, self.skip))   # (1, B*p, hid_skip)
            s = self.dropout(hs.squeeze(0).reshape(B, self.skip * self.hid_skip))
            r = torch.cat([r, s], dim=1)

        # 4. Dense -> h_D (B, n)
        out = self.dense(r)

        # 5. AR on the last ar_window raw (scaled) values of each series -> h_L (B, n)
        if self.ar_window > 0:
            z = x[:, -self.ar_window:, :].permute(0, 2, 1)       # (B, n, ar_window)
            out = out + self.ar(z).squeeze(2)

        if self.revin is not None:                   # back to the window's level and scale
            out = self.revin.denormalise(out)
        return out                                   # no output activation ("Linear")


class DLinear(nn.Module):
    """DLinear (Zeng et al., AAAI 2023): a simple linear comparison model.

    Each input series is split into a smooth trend (moving average over 25 steps) and the rest
    (seasonal part). One linear layer per part, shared by all series, turns the q values into
    one prediction; the two predictions are added.
    """

    def __init__(self, window=168, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.average = nn.AvgPool1d(kernel, stride=1)   # mean of every 25-step stretch
        self.trend_linear = nn.Linear(window, 1)
        self.seasonal_linear = nn.Linear(window, 1)

    def moving_average(self, x):                     # x: (B, n, q)
        # Repeat the first and last value so the average has the same length as x
        pad = (self.kernel - 1) // 2
        front = x[:, :, :1].repeat(1, 1, pad)
        back = x[:, :, -1:].repeat(1, 1, pad)
        padded = torch.cat([front, x, back], dim=2)
        return self.average(padded)

    def forward(self, x):                            # x: (B, q, n)
        x = x.permute(0, 2, 1)                       # (B, n, q): one row per series
        trend = self.moving_average(x)
        seasonal = x - trend
        out = self.trend_linear(trend) + self.seasonal_linear(seasonal)   # (B, n, 1)
        return out.squeeze(2)
