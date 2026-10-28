# Checks of our AR baseline (Electricity)

## Check 2 and 4: horizon and window

Test RSE of our AR for every window length L (one model per customer, least squares).
**Bold** = the L chosen on validation (as in the paper, Sec. 4.4).

| Horizon | L=1 | L=2 | L=4 | L=8 | L=16 | L=32 | L=64 | L=128 | L=256 | L=512 | chosen L | ours (chosen L) | paper AR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.4115 | 0.3531 | 0.3355 | 0.3142 | 0.2474 | 0.0893 | 0.0863 | 0.0835 | 0.0819 | **0.0818** | 512 | 0.0818 | 0.0995 |
| 6 | 0.5584 | 0.4773 | 0.4340 | 0.4248 | 0.2525 | 0.0983 | 0.0956 | 0.0931 | **0.0913** | 0.0912 | 256 | 0.0913 | 0.1035 |
| 12 | 0.4858 | 0.4856 | 0.4769 | 0.3990 | 0.1027 | 0.1018 | 0.0987 | 0.0975 | **0.0957** | 0.0954 | 256 | 0.0957 | 0.1050 |
| 24 | 0.1042 | 0.1041 | 0.1040 | 0.1037 | 0.1035 | 0.1019 | 0.1011 | 0.0995 | **0.0981** | 0.0979 | 256 | 0.0981 | 0.1054 |

## Check 1: RSE exactly as in Eq. 10

Same AR predictions (h = 24, L = 256), in original units (kWh), pooled over all customers
and test hours, divided by the spread around the mean of the whole test set:

- our `metrics.rse`: 0.098090
- the authors' formula (`sqrt(MSE) / normal_std(test)`): 0.098090

## Check 3: scaling on the training part only

- scale from the whole series (authors' code): test RSE 0.098090
- scale from the training part only: test RSE 0.098090

A least-squares fit per customer is unchanged when the customer is multiplied by a constant,
so the scaling cannot change the AR result.

## Off-by-one in the horizon

`tests/test_data_and_metrics.py::test_window_and_target_are_h_steps_apart` checks that the newest
input step is exactly h steps before the target. `make_inputs` uses the same rule
(`series[t - h - L + 1 : t - h + 1]`).
