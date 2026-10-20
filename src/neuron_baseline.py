"""First neural baseline: one neuron shared by all series (notebook 02 as a script), several seeds.

The neuron sees the last q steps of a series and predicts the value h steps after them; it is
LSTNet's AR component on its own. Weights are chosen on validation; the test set is used once,
at the end.

Usage:  python src/neuron_baseline.py --data electricity --horizon 24 --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import load_raw, make_loaders, max_scale
from metrics import corr, rse

ROOT = Path(__file__).resolve().parent.parent


class OneNeuron(nn.Module):
    def __init__(self, q):
        super().__init__()
        self.linear = nn.Linear(q, 1)

    def forward(self, x):                  # (B, q, n) -> (B, n)
        return self.linear(x.transpose(1, 2)).squeeze(-1)


def predict(model, loader, scale):
    model.eval()
    with torch.no_grad():
        p, t = zip(*[(model(x), y) for x, y in loader])
    return torch.cat(p).numpy() * scale, torch.cat(t).numpy() * scale


def run(data, scale, q, h, seed, epochs):
    torch.manual_seed(seed)                # initial weights
    train_dl, valid_dl, test_dl = make_loaders(data, q, h, seed=seed)   # batch order
    model = OneNeuron(q)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        v = rse(*predict(model, valid_dl, scale))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred, true = predict(model, test_dl, scale)          # test: only once, at the end
    return {"seed": seed, "valid_rse": float(best), "test_rse": float(rse(pred, true)),
            "test_corr": corr(pred, true)[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="electricity", choices=["electricity", "solar"])
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    data, scale = max_scale(load_raw(args.data))
    rows = []
    for s in args.seeds:
        r = run(data, scale, args.window, args.horizon, s, args.epochs)
        rows.append(r)
        print(f"seed {s}: valid RSE {r['valid_rse']:.4f} | test RSE {r['test_rse']:.4f} | test CORR {r['test_corr']:.4f}", flush=True)

    for k in ("valid_rse", "test_rse", "test_corr"):
        v = np.array([r[k] for r in rows])
        print(f"{k:10s} mean {v.mean():.4f} ± {v.std(ddof=1):.4f} (std over {len(v)} seeds)")
    out = ROOT / "results" / f"neuron_{args.data}_h{args.horizon}.json"
    out.write_text(json.dumps({"args": vars(args), "runs": rows}, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
