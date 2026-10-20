"""Train LSTNet-Skip and test it with the paper's metrics, as in the authors' main.py.

  * loss: L1, summed, on values in original units (prediction * scale vs target * scale)
  * Adam with learning rate 0.001; gradients clipped to a total norm of 10
  * after every epoch: validation RSE; keep the weights with the best validation RSE
  * at the end: test RSE and CORR with those weights (the test set is used only here)

After every epoch the run is saved to <checkpoint-dir>/<run name>.pt. If training stops
(e.g. Colab disconnects), start the same command again and it continues from there.

Usage:  python src/train.py --data electricity --horizon 24 --skip 24 --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import load_raw, make_loaders, max_scale
from metrics import corr, rse
from model import LSTNet

ROOT = Path(__file__).resolve().parent.parent


def predict(model, loader, scale, device):
    """All predictions and targets of a loader, in original units."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            trues.append(y)
    return torch.cat(preds).numpy() * scale, torch.cat(trues).numpy() * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="electricity", choices=["electricity", "solar"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--window", type=int, default=168)
    parser.add_argument("--skip", type=int, default=24)
    parser.add_argument("--hid-skip", type=int, default=5)       # authors: 5 electricity, 10 solar
    parser.add_argument("--no-ar", action="store_true")          # ablation: LSTNet without AR
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    args = parser.parse_args()

    name = f"{args.data}_h{args.horizon}_lstnet_skip{args.skip}_seed{args.seed}"
    if args.no_ar:
        name = name.replace("lstnet", "lstnet_noar")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"run {name} on {device}")

    # Data and model
    data, scale = max_scale(load_raw(args.data))
    scale_t = torch.tensor(scale, dtype=torch.float32, device=device)
    torch.manual_seed(args.seed)
    train_dl, valid_dl, test_dl = make_loaders(data, args.window, args.horizon, seed=args.seed)
    model = LSTNet(n=data.shape[1], window=args.window, skip=args.skip,
                   hid_skip=args.hid_skip, use_ar=not args.no_ar).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    # Continue from the checkpoint if there is one
    checkpoint = Path(args.checkpoint_dir) / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    first_epoch, best_rse, best_epoch, best_weights = 1, float("inf"), 0, None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        first_epoch = saved["epoch"] + 1
        best_rse, best_epoch, best_weights = saved["best_rse"], saved["best_epoch"], saved["best_weights"]
        print(f"continuing after epoch {saved['epoch']}")

    for epoch in range(first_epoch, args.epochs + 1):
        start = time.time()
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = (model(x) * scale_t - y * scale_t).abs().sum()   # L1 in original units
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimiser.step()

        pred, true = predict(model, valid_dl, scale, device)
        valid_rse = rse(pred, true)
        if valid_rse < best_rse:
            best_rse, best_epoch = valid_rse, epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d} | {time.time() - start:5.1f}s | valid RSE {valid_rse:.4f} | "
              f"valid CORR {corr(pred, true)[0]:.4f} | best so far {best_rse:.4f} (epoch {best_epoch})",
              flush=True)

        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "epoch": epoch, "best_rse": best_rse, "best_epoch": best_epoch,
                    "best_weights": best_weights}, checkpoint)

    # Test once, with the weights of the best validation epoch
    model.load_state_dict(best_weights)
    pred, true = predict(model, test_dl, scale, device)
    test_rse, (test_corr, dropped) = rse(pred, true), corr(pred, true)
    print(f"\nTEST  RSE {test_rse:.4f} | CORR {test_corr:.4f} (best epoch {best_epoch})")

    result = {"run": name, "best_epoch": best_epoch, "valid_rse": float(best_rse),
              "test_rse": float(test_rse), "test_corr": test_corr, "series_left_out_of_corr": dropped}
    (ROOT / "results" / "runs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "runs" / f"{name}.json").write_text(json.dumps(result, indent=2))
    np.save(Path(args.checkpoint_dir) / f"{name}_test_predictions.npy", pred.astype(np.float32))


if __name__ == "__main__":
    main()
