"""Save the train / validation / test split to a file, so it is fixed and can be checked.

The split is by time (60 / 20 / 20, as in the paper). The file lists, for each part, the first
and last target time step and the number of examples, for window q = 168 and horizon h = 24.

Usage:  python src/save_split.py      -> results/splits.json
"""
import json
from pathlib import Path

from data import split_targets

ROOT = Path(__file__).resolve().parent.parent
LENGTHS = {"electricity": 26304, "solar": 52560}      # number of time steps in each file
Q, H = 168, 24

splits = {}
for name, T in LENGTHS.items():
    train, valid, test = split_targets(T, Q, H)
    splits[name] = {"time_steps": T, "window": Q, "horizon": H}
    for part, targets in [("train", train), ("valid", valid), ("test", test)]:
        splits[name][part] = {"first_target": int(targets[0]), "last_target": int(targets[-1]),
                              "examples": len(targets)}

out = ROOT / "results" / "splits.json"
out.write_text(json.dumps(splits, indent=2))
print(json.dumps(splits, indent=2))
