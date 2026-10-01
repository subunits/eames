# eames_real.py: does a scaling law fitted on small models predict a larger one?

Trains eight tiny character-level transformers at increasing width, fits
`L(N) = E + A / N^alpha` on the seven smaller ones, and compares the prediction with the
loss of the largest one. A seed sets both the weight initialisation and the
training-data order. Validation batches are fixed, so differences between seeds come only
from training.

## Status

Two sets of runs. They used different corpora (see Setup), so their numbers should not be
mixed.

**Run A: three seeds, 1500 steps** (Google Colab, Python 3.13.15, torch 2.11.0+cpu,
corpus 4,575,051 chars, 1.49 passes over the training split).

| Seed | Predicted | Actual | Error (E + A/N^a) | Fitted E | Error (pure power law) |
|---|---|---|---|---|---|
| 0 | 1.3591 | 1.4464 | -6.04% | 0.133 | -6.35% |
| 1 | 1.3583 | 1.4317 | -5.13% | 0.001 | -5.13% |
| 2 | 1.3600 | 1.4354 | -5.25% | 0.000 | -5.25% |

Across the three seeds the error is -5.47% (sd 0.49%) for the floor fit and -5.58% (sd
0.67%) for the pure power law. The held-out loss itself has mean 1.4378 and sd 0.0077
(0.53% of the mean). Three seeds is a thin basis for a standard deviation, so treat these
as rough.

**Run B: one seed, 500 and 1500 steps** (earlier run, corpus 4,688,935 chars, 1.46 and
0.49 passes). The fit missed the held-out model in both, with opposite signs.

| Steps | Predicted | Actual | Error |
|---|---|---|---|
| 500 | 1.7873 | 1.7253 | +3.6% |
| 1500 | 1.3824 | 1.4578 | -5.2% |

The 500-step result has not been repeated with more seeds, and not on the Run A corpus.

## Validation loss by model

Run A, 1500 steps (N = non-embedding parameters). Mean and sd are over seeds 0, 1, 2.

| d | N | Mean loss | sd | sd as % of mean |
|---|---|---|---|---|
| 16 | 6,560 | 2.3282 | 0.0286 | 1.23% |
| 24 | 14,448 | 2.1812 | 0.0200 | 0.91% |
| 32 | 25,408 | 2.0319 | 0.0054 | 0.26% |
| 48 | 56,544 | 1.8880 | 0.0078 | 0.41% |
| 64 | 99,968 | 1.7038 | 0.0078 | 0.46% |
| 96 | 223,680 | 1.5638 | 0.0030 | 0.19% |
| 128 | 396,544 | 1.5022 | 0.0027 | 0.18% |
| 192 | 889,728 | 1.4378 | 0.0077 | 0.53% |

Run B (one seed, old corpus):

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

- **At 1500 steps the miss is larger than seed noise.** All three seeds under-predict the
  held-out loss by 5 to 6%, while the held-out loss varies by about 0.5% between seeds.
  This is an informal comparison, not a significance test.
- **The size of the miss is similar across the two corpora.** Run B gave -5.2% at 1500
  steps and Run A gave -5.5% on average. They are not an exact replication, because the
  data differ.
- **The log-log curve is not a straight line.** On the Run A seed-mean losses the local
  slope goes -0.083, -0.126, -0.092, -0.180, -0.106, -0.070, -0.054: steepest in the
  middle, then about half the fitted slope (0.112) at the large end. Each individual seed
  shows the same shape. A fit that extrapolates the earlier slope therefore expects more
  improvement than the largest model delivers. This is consistent with the
  under-prediction, but the data do not show that the bend is its cause.
- **The fitted floor is unstable.** E was 0.133 for seed 0 and about 0 for seeds 1 and 2,
  so the floor fit and the pure power law agree for two seeds and differ for one. E is
  only approximately 0 in those cases (it is bounded below by 0 and does not reach it).
  With 2.1 decades of range, E and alpha trade off against each other.
- **Small models vary most between seeds.** The spread is 1.2% and 0.9% at d=16 and
  d=24, and 0.2 to 0.5% at the larger widths.

Earlier observations from Run B (one seed, old corpus), with the caveat that the code for
the next two items is not in this repository:

- Predicting each model from the ones before it with a pure power law, the error went
  +6.6%, +2.8%, +1.1%, -2.0%, -5.2% at 1500 steps: pessimistic early, optimistic late.
- Refitting with a floor on later models gave -0.5% on one window (models 2 to 6) and
  -4.3% on another (models 3 to 6). Those windows were chosen after seeing the answer, so
  this shows the prediction is unstable, not that it works.

## Untested explanations

- At 1500 steps each model sees the training text about 1.5 times, so larger models may be
  starting to run out of fresh data. This cannot explain the 500-step miss in Run B,
  where models saw about half the text.
- The learning rate (3e-3) was the same for every width and not tuned.
- The 500-step result has one seed only.

## Setup

    pip install torch numpy matplotlib

The corpus is the top-level `.py` files of the Python standard library on the machine
running the script, capped at 8 MB. Its size therefore depends on the Python version:
4,688,935 chars in Run B and 4,575,051 chars on Python 3.13.15 in Run A. The script
prints Python and torch versions, corpus size and passes over the data at the start of
every run. Only compare results produced on the same corpus.

## Running

    python eames_real.py --steps 1500 --seeds 0 1 2

Options:

- `--steps N` sets training steps per model (default 1500).
- `--seeds a b c` sets the seeds to run (default `0`). Decide the seeds before running.
- `--skip-train` skips training and refits and replots from the cached `results.json`,
  for the given `--steps` and `--seeds`.

Progress is saved to `results.json` after every model, keyed by width, steps and seed, and
rows for other settings are kept. Rerunning the same command resumes. Rows from the older
single-seed script have no `seed` field and are treated as seed 0. Check
`pgrep -af eames_real` before starting a second copy on a machine you control.

Timing, Run A on Colab CPU: about 33 minutes per seed at 1500 steps, with d=192 taking
about 13 of them.

## Outputs

`results.json`, `scaling_fit.png`, and a printed per-seed table with the mean and sd of
the prediction error across seeds.

## Limits

Two decades of model size, one held-out model, three seeds at one step count, tiny models
on Python source, and an untuned learning rate. This says nothing about capability, cost,
or larger scales, and it does not show that scaling laws fail in general.
