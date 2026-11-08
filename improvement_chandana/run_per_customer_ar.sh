#!/bin/bash
# Electricity improvement 2: LSTNet with one AR model per customer over the whole 168-hour window.
# Everything else as in the authors' ele.sh. Run from the repo root.
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python src/train.py --data electricity --horizon 24 --model lstnet_ar168 --seed $seed
done
