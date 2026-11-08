#!/bin/bash
# Solar improvement 2: LSTNet with a one-day skip (144 steps) and a 3-day window (438 steps).
# Everything else as in the authors' solar.sh (skip-GRU size 10). Run from the repo root.
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python src/train.py --data solar --horizon 24 --model lstnet \
        --window 438 --skip 144 --hid-skip 10 --seed $seed
done
