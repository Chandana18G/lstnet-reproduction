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


class LSTNet(nn.Module):
    def __init__(self, n, window=168, hid_cnn=100, hid_rnn=100, hid_skip=5, kernel=6,
                 skip=24, ar_window=24, dropout=0.2, use_ar=True):
        super().__init__()
        self.n, self.window, self.skip, self.hid_skip = n, window, skip, hid_skip
        self.ar_window = ar_window if use_ar else 0
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

        return out                                   # no output activation ("Linear")
