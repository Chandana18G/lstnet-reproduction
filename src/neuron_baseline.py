"""First neural baseline: one neuron shared by all series (notebook 02 as a script).

The neuron sees the last q steps of a series and predicts the value h steps later. It is the
AR part of LSTNet on its own. The best epoch is chosen on validation; the test set is used
only once, at the end. Run with several seeds to see how much the result varies.

Usage:  python src/neuron_baseline.py --data electricity --horizon 24 --seeds 0 1 2
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import load_raw, make_loaders, max_scale
from metrics import corr, rse

RESULTS = Path(__file__).resolve().parent.parent / "results"


class OneNeuron(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.linear = nn.Linear(q, 1)                  # q weights + 1 bias

    def forward(self, x):                              # x: (batch, q, n)
        x = x.transpose(1, 2)                          # -> (batch, n, q)
        return self.linear(x).squeeze(-1)              # -> (batch, n)


def predict(model, loader, scale):
    """All predictions and targets of a loader, in original units."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x))
            trues.append(y)
    return torch.cat(preds).numpy() * scale, torch.cat(trues).numpy() * scale


def train_one_seed(data, scale, q, h, seed, epochs):
    torch.manual_seed(seed)                            # same seed -> same starting weights
    train_dl, valid_dl, test_dl = make_loaders(data, q, h, seed=seed)
    model = OneNeuron(q)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    best_rse, best_weights = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
        valid_rse = rse(*predict(model, valid_dl, scale))
        if valid_rse < best_rse:
            best_rse = valid_rse
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    pred, true = predict(model, test_dl, scale)
    return best_rse, rse(pred, true), corr(pred, true)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="electricity", choices=["electricity", "solar"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    data, scale = max_scale(load_raw(args.data))
    lines = ["seed,valid_rse,test_rse,test_corr"]
    results = []
    for seed in args.seeds:
        v_rse, t_rse, t_corr = train_one_seed(data, scale, args.window, args.horizon, seed, args.epochs)
        results.append((v_rse, t_rse, t_corr))
        lines.append(f"{seed},{v_rse:.4f},{t_rse:.4f},{t_corr:.4f}")
        print(f"seed {seed}: valid RSE {v_rse:.4f} | test RSE {t_rse:.4f} | test CORR {t_corr:.4f}", flush=True)

    results = np.array(results)
    print(f"mean ± std over seeds: test RSE {results[:, 1].mean():.4f} ± {results[:, 1].std(ddof=1):.4f}, "
          f"test CORR {results[:, 2].mean():.4f} ± {results[:, 2].std(ddof=1):.4f}")
    (RESULTS / f"neuron_{args.data}_h{args.horizon}.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
