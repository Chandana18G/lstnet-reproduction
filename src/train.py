"""Train LSTNet-Skip and evaluate it with the paper's metrics.

Follows the authors' main.py (github.com/laiguokun/LSTNet):
  * loss = L1, summed, computed on de-normalised values (prediction * scale vs target * scale);
  * Adam, learning rate 1e-3, gradients clipped to a total norm of 10;
  * after every epoch: validation RSE; keep the weights with the best validation RSE;
  * at the end: test RSE and CORR of those best weights.

Runs are resumable: after every epoch everything needed to continue is saved to
<checkpoint dir>/<run name>/last.pt. Starting the same command again continues from there.

Example:
  python src/train.py --data electricity --horizon 24 --skip 24 --seed 0
  python src/train.py --data electricity --horizon 24 --seed 0 --epochs 1 --limit-batches 5   # smoke test
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


def parse_args():
    ap = argparse.ArgumentParser(description="Train LSTNet-Skip")
    ap.add_argument("--data", default="electricity", choices=["electricity", "solar"])
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--window", type=int, default=168)
    ap.add_argument("--skip", type=int, default=24)
    ap.add_argument("--hid-cnn", type=int, default=100)
    ap.add_argument("--hid-rnn", type=int, default=100)
    ap.add_argument("--hid-skip", type=int, default=5)
    ap.add_argument("--kernel", type=int, default=6)
    ap.add_argument("--ar-window", type=int, default=24)
    ap.add_argument("--no-ar", action="store_true", help="ablation: remove the AR component")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--loss", default="l1", choices=["l1", "l2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="run name (default: built from the settings)")
    ap.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"),
                    help="where last.pt is saved; on Colab point this to Google Drive")
    ap.add_argument("--limit-batches", type=int, default=0, help="smoke test: batches per epoch")
    return ap.parse_args()


def predict(model, loader, scale, device):
    """All predictions and targets of a loader, in original units (e.g. kWh)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            trues.append(y)
    return torch.cat(preds).numpy() * scale, torch.cat(trues).numpy() * scale


def main():
    args = parse_args()
    name = args.name or (f"{args.data}_h{args.horizon}_lstnet{'_noar' if args.no_ar else ''}"
                         f"_skip{args.skip}_seed{args.seed}")
    run_dir = Path(args.checkpoint_dir) / name
    out_dir = ROOT / "results" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"run {name} on {device}")

    # Data: same scaling, split and windows as every other model in the project
    data, scale = max_scale(load_raw(args.data))
    scale_t = torch.tensor(scale, dtype=torch.float32, device=device)
    torch.manual_seed(args.seed)                     # -> same initial weights for this seed
    gen = torch.Generator().manual_seed(args.seed)   # -> same batch order for this seed
    train_dl, valid_dl, test_dl = make_loaders(data, args.window, args.horizon,
                                               args.batch_size, generator=gen)

    model = LSTNet(n=data.shape[1], window=args.window, hid_cnn=args.hid_cnn,
                   hid_rnn=args.hid_rnn, hid_skip=args.hid_skip, kernel=args.kernel,
                   skip=args.skip, ar_window=args.ar_window, dropout=args.dropout,
                   use_ar=not args.no_ar).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Resume if a checkpoint of this run exists
    start_epoch, best_valid, best_state, history = 1, float("inf"), None, []
    ckpt_path = run_dir / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimiser.load_state_dict(ck["optimiser"])
        gen.set_state(ck["generator"])
        torch.set_rng_state(ck["torch_rng"])
        start_epoch, best_valid = ck["epoch"] + 1, ck["best_valid"]
        best_state, history = ck["best_state"], ck["history"]
        print(f"resumed after epoch {ck['epoch']} (best valid RSE {best_valid:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            if args.limit_batches and i >= args.limit_batches:
                break
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            err = model(x) * scale_t - y * scale_t           # error in original units
            loss = err.abs().sum() if args.loss == "l1" else (err ** 2).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimiser.step()
            total += loss.item()
            count += y.numel()

        pred, true = predict(model, valid_dl, scale, device)
        v_rse, (v_corr, _) = rse(pred, true), corr(pred, true)
        history.append({"epoch": epoch, "train_loss": total / count,
                        "valid_rse": float(v_rse), "valid_corr": v_corr})
        if v_rse < best_valid:
            best_valid = float(v_rse)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d} | {time.time() - t0:5.1f}s | train loss {total / count:9.3f} | "
              f"valid RSE {v_rse:.4f} | valid CORR {v_corr:.4f}"
              f"{'  *' if best_valid == v_rse else ''}", flush=True)

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(), "generator": gen.get_state(),
                    "torch_rng": torch.get_rng_state(), "best_valid": best_valid,
                    "best_state": best_state, "history": history, "args": vars(args)},
                   ckpt_path)

    # Test the weights with the best validation RSE
    model.load_state_dict(best_state)
    pred, true = predict(model, test_dl, scale, device)
    t_corr, dropped = corr(pred, true)
    result = {"run": name, "best_valid_rse": best_valid, "test_rse": float(rse(pred, true)),
              "test_corr": t_corr, "corr_dropped": dropped,
              "best_epoch": min(history, key=lambda r: r["valid_rse"])["epoch"],
              "args": vars(args), "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    np.save(out_dir / "test_predictions.npy", pred.astype(np.float32))
    print(f"\nTEST  RSE {result['test_rse']:.4f} | CORR {t_corr:.4f} "
          f"(best epoch {result['best_epoch']}, {dropped} series dropped from CORR)")
    print(f"saved {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
