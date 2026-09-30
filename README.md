# eames_real.py: does a scaling law fitted on small models predict a larger one?

Trains eight tiny character-level transformers at increasing width, fits
`L(N) = E + A / N^alpha` on the seven smaller ones, and compares the prediction with the
loss of the largest one.

## Status

Two full runs completed (one seed each). In both, the fit missed the held-out model, and
the misses had opposite signs.

| Steps | Predicted | Actual | Error |
|---|---|---|---|
| 500 | 1.7873 | 1.7253 | +3.6% |
| 1500 | 1.3824 | 1.4578 | -5.2% |

Validation loss by model (N = non-embedding parameters):

| d | N | 500 steps | 1500 steps |
|---|---|---|---|
| 16 | 6,560 | 2.5688 | 2.3132 |
| 24 | 14,448 | 2.4795 | 2.1860 |
| 32 | 25,408 | 2.4102 | 2.0931 |
| 48 | 56,544 | 2.2805 | 1.8521 |
| 64 | 99,968 | 2.1939 | 1.7311 |
| 96 | 223,680 | 1.9933 | 1.5860 |
| 128 | 396,544 | 1.8393 | 1.5263 |
| 192 | 889,728 | 1.7253 | 1.4578 |

## Findings

- The log-log curve is not a straight line. At 1500 steps the local slope goes -0.072, -0.077, -0.153, -0.119, -0.109, -0.067, -0.057: steepest in the middle, then roughly halving at the large end. At 500 steps it steepened and then eased only at the last step.
- Predicting each model from the ones before it (pure power law), the error goes +6.6%, +2.8%, +1.1%, -2.0%, -5.2% at 1500 steps. Early fits were too pessimistic, late fits too optimistic, because the curve is bending.
- The fitted floor hit its lower bound (E = 0) in both runs, so the floor fit and the pure power law were the same line.
- Refitting with a floor on later models gave -0.5% on one window (models 2 to 6) and -4.3% on another (models 3 to 6). Those windows were chosen after seeing the answer, so this shows the prediction is unstable, not that it works.

## Untested explanations

- At 1500 steps each model sees the training text about 1.5 times (0.5 times at 500 steps), so the larger models may be starting to run out of fresh data.
- The learning rate was not tuned per width.
- One seed per run, so run-to-run noise is unknown.

## Setup

    pip install torch numpy matplotlib

## Running

Run one copy, in the foreground:

    python eames_real.py --steps 1500

Options: `--steps N` sets training steps per model (default 1500); `--skip-train` refits
and replots from the cached `results.json`. Progress is saved after every model, and
rerunning the same command resumes. Cached rows are reused only when `--steps` matches.
Check `pgrep -af eames_real` before starting a second copy. Commit `results.json` after a
run so a lost instance does not cost you the data.

Timing from the 1500-step run: about 25 minutes for all eight models, with d=192 taking
about 10 of them.

## Outputs

`results.json`, `scaling_fit.png`, and the fit printed at the end of the run.

## Limits

Two decades of model size, one seed, tiny models on Python source. This says nothing
about capability, cost, or larger scales.
