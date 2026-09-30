# eames_real.py: does a scaling law fitted on small models predict a larger one?

Trains eight tiny character-level transformers at increasing width, fits
`L(N) = E + A / N^alpha` on the seven smaller ones, and compares the prediction with the
loss of the largest one. The "Powers of Ten" idea applied to model size: each step is a
multiple on a log axis, and the test is whether the straight line on log-log axes holds up
when extrapolated.

## Status

Smoke-tested only. The script runs end to end and resume has been tested, but a full run
with a meaningful step count has not been completed yet, so there is no result to report.
Replace this section with your held-out error once you have one.

## Setup

```
pip install torch numpy matplotlib
```

Matplotlib is only needed for the plot. Without it the script still prints the fit and the
held-out error. CPU only, no downloads: the corpus is your own Python standard library
source.

## Running

```
nohup python eames_real.py --steps 500 > run.log 2>&1 &
tail -f run.log
```

`nohup` keeps the run alive if the terminal disconnects. `tail -f` shows one line per
finished model with its loss and training time. Ctrl+C stops the tail, not the run.

| Option | Effect |
|---|---|
| `--steps N` | Training steps per model (default 1500) |
| `--skip-train` | Refit and replot from the cached `results.json` without training |

Progress is saved to `results.json` after every model. If the run is interrupted, rerun
the same command and it resumes at the next model. Cached rows are only reused when they
were trained with the same `--steps`.

Rough timing, scaled from a short test on a fast machine: about 10 minutes at 500 steps and
30 to 35 minutes at 1500 steps for all eight models. A 2-core machine may be slower.

## Outputs

- `run.log`: per-model losses and times, then the fit and held-out comparison
- `results.json`: `d`, `N`, `val_loss` and `steps` for each model
- `scaling_fit.png`: log-log plot of the data, both fits, and the held-out point in red

## What it does

1. Trains a 2-layer transformer at widths 16, 24, 32, 48, 64, 96, 128 and 192, all with the same data budget.
2. Records validation loss against non-embedding parameter count `N` (about 6.5 thousand to 890 thousand, roughly 2 decades).
3. Fits `E + A / N^alpha` to the seven smallest by gradient descent in log space, with `E` kept below the smallest observed loss.
4. Fits a pure power law (no floor) as a baseline.
5. Predicts the largest model and prints the percentage error for both fits.

## Reading the result

- The held-out error is the main number. Small and stable across reruns is modest evidence that the law holds in this setting.
- With only 2 decades of range, `E` and `alpha` trade off against each other. Expect the fitted values to move between runs even when the prediction is decent.
- If the largest model sits above the trend, the usual causes are undertraining at larger widths (the fixed step budget favors small models) and a learning rate not tuned per size. Try a larger `--steps` first.

## What it does not show

- Anything about capability. These models predict characters of Python source and have under a million parameters.
- Anything about cost, data supply or usefulness at scale.
- Extrapolation over a meaningful range. Published scaling-law work spans many more decades.
- A result, until a full run is completed and recorded above.

## GitHub Codespaces

The free plan includes 120 core-hours per month, which is 60 hours on a 2-core machine, with
a $0 default spending limit. Going over stops the codespace and does not bill. Stop the
codespace when a run finishes so it does not keep spending core hours.
