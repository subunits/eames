"""Powers of Ten on real models: does a scaling law fitted on small models
predict the loss of a larger one?

Steps
  1. Train tiny char-level transformers at increasing width (fixed data budget)
  2. Record validation loss vs non-embedding parameter count N
  3. Fit L(N) = E + A / N^alpha on all but the largest model
  4. Compare the prediction with the held-out model; plot on log-log axes
  5. Repeat for several seeds and report the spread of the prediction error

Corpus: your own Python standard library source (offline, no downloads).

Usage
  python eames_real.py --steps 1500                    # seed 0 only (as before)
  python eames_real.py --steps 1500 --seeds 0 1 2      # three seeds
  (progress is saved after every model; rerun the same command to resume)
  python eames_real.py --steps 1500 --seeds 0 1 2 --skip-train   # refit/replot only

A seed sets BOTH the weight initialisation and the training-data order.
Validation batches are fixed (seed 123), so differences between seeds come
only from training. Seed 0 reproduces the original single-seed runs exactly.
"""
import argparse
import glob
import json
import math
import os
import sys
import sysconfig
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

LN10 = math.log(10.0)
WIDTHS = [16, 24, 32, 48, 64, 96, 128, 192]   # d_model; last one is held out
N_LAYER = 2
BLOCK, BATCH = 64, 64
RESULTS = "results.json"


# ------------------------------------------------------------------ data
def load_corpus(max_bytes=8_000_000):
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "*.py")))
    chunks, total = [], 0
    for f in files:
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        chunks.append(s)
        total += len(s)
        if total >= max_bytes:
            break
    text = "".join(chunks)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(0.9 * len(data))
    return data[:cut], data[cut:], len(chars)


def get_batch(data, g):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,), generator=g)
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


# ----------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.heads = max(1, d // 16)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(BLOCK, d)
        self.blocks = nn.Sequential(*[Block(d) for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        return self.head(self.ln(self.blocks(h)))

    def n_params(self):
        """Non-embedding parameters (Kaplan et al. convention)."""
        return sum(p.numel() for p in self.blocks.parameters())


@torch.no_grad()
def evaluate(model, data, batches=40):
    g = torch.Generator().manual_seed(123)      # fixed: same val batches for every run
    model.eval()
    tot = 0.0
    for _ in range(batches):
        x, y = get_batch(data, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / batches


def train_one(d, train, val, vocab, steps, seed=0):
    torch.manual_seed(seed)                      # weight init
    g = torch.Generator().manual_seed(seed)      # training-data order
    model = TinyGPT(vocab, d)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps,
                                                pct_start=0.1)
    for _ in range(steps):
        x, y = get_batch(train, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model.n_params(), evaluate(model, val)


# ------------------------------------------------------------------ fits
def fit_shifted(N, loss, iters=300):
    """L = E + exp(a - alpha*ln10*x), x = log10 N - mean. E kept below min loss."""
    N, loss = N.double(), loss.double()
    x0 = torch.log10(N).mean()
    x = torch.log10(N) - x0
    raw_E = torch.tensor(0.0, dtype=torch.double, requires_grad=True)
    a = torch.tensor(math.log(float(loss.max())), dtype=torch.double, requires_grad=True)
    raw_al = torch.tensor(-1.0, dtype=torch.double, requires_grad=True)

    def predict(xx):
        E = loss.min() * torch.sigmoid(raw_E)
        al = F.softplus(raw_al)
        return E + torch.exp(a - al * LN10 * xx), E, al

    opt = torch.optim.LBFGS([raw_E, a, raw_al], lr=0.5, max_iter=iters,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = ((torch.log(predict(x)[0]) - torch.log(loss)) ** 2).mean()
        l.backward()
        return l

    opt.step(closure)
    with torch.no_grad():
        _, E, al = predict(x)
    return (lambda Nn: predict(torch.log10(torch.as_tensor(Nn, dtype=torch.double)) - x0)[0].detach(),
            E.item(), al.item())


def fit_powerlaw(N, loss):
    """L = C * N^-beta via least squares in log-log space (E = 0)."""
    X = torch.stack([torch.ones(len(N), dtype=torch.double),
                     torch.log10(N.double())], dim=1)
    y = torch.log10(loss.double())
    coef = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze()
    c, slope = coef[0].item(), coef[1].item()
    return (lambda Nn: 10 ** (c + slope * torch.log10(torch.as_tensor(Nn, dtype=torch.double))),
            -slope)


def fit_and_predict(N, L):
    """Fit on all but the largest model; predict the largest."""
    N_fit, L_fit = N[:-1], L[:-1]
    N_hold, L_hold = N[-1].item(), L[-1].item()
    pred_fn, E, alpha = fit_shifted(N_fit, L_fit)
    pl_fn, beta = fit_powerlaw(N_fit, L_fit)
    p1, p2 = float(pred_fn(N_hold)), float(pl_fn(N_hold))
    return {"pred_fn": pred_fn, "pl_fn": pl_fn, "E": E, "alpha": alpha, "beta": beta,
            "p_shift": p1, "p_pl": p2, "actual": L_hold,
            "err_shift": 100 * (p1 - L_hold) / L_hold,
            "err_pl": 100 * (p2 - L_hold) / L_hold}


def mean_std(vals):
    t = torch.tensor(vals, dtype=torch.double)
    sd = t.std().item() if len(vals) > 1 else float("nan")
    return t.mean().item(), sd


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="seeds to run (sets init AND data order); decide them before running")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    seeds = list(dict.fromkeys(args.seeds))

    # All rows are kept in the file, keyed by (width, steps, seed). Rows written by the
    # older single-seed script have no "seed" field and are treated as seed 0.
    store = {}
    if os.path.exists(RESULTS):
        for r in json.load(open(RESULTS)):
            r.setdefault("seed", 0)
            store[(r["d"], r["steps"], r["seed"])] = r

    def save():
        rows = sorted(store.values(), key=lambda r: (r["steps"], r["seed"], r["d"]))
        json.dump(rows, open(RESULTS, "w"), indent=2)    # save after every model

    if not args.skip_train:
        train, val, vocab = load_corpus()
        print(f"Python {sys.version.split()[0]}, torch {torch.__version__}")
        print(f"Corpus: {len(train) + len(val):,} chars "
              f"(train {len(train):,}, val {len(val):,}), vocab {vocab}")
        print(f"{args.steps} steps x {BATCH * BLOCK} tokens per model = "
              f"{args.steps * BATCH * BLOCK / len(train):.2f} passes over train; seeds {seeds}")
        for seed in seeds:
            print(f"\nSeed {seed}")
            for d in WIDTHS:
                key = (d, args.steps, seed)
                if key in store:
                    r = store[key]
                    print(f"  d={d:<4} N={r['N']:>9,}  val loss = {r['val_loss']:.4f}  (cached)")
                    continue
                t0 = time.time()
                N, vl = train_one(d, train, val, vocab, args.steps, seed)
                store[key] = {"d": d, "N": N, "val_loss": vl, "steps": args.steps, "seed": seed}
                print(f"  d={d:<4} N={N:>9,}  (10^{math.log10(N):.2f})  val loss = {vl:.4f}"
                      f"  [{time.time() - t0:.0f}s]", flush=True)
                save()
        save()

    # ---- gather complete seeds for this --steps
    by_seed = {}
    for s in seeds:
        if all((d, args.steps, s) in store for d in WIDTHS):
            by_seed[s] = [store[(d, args.steps, s)] for d in WIDTHS]
        else:
            print(f"\n(seed {s} skipped: not all {len(WIDTHS)} widths present for "
                  f"{args.steps} steps)")
    if not by_seed:
        print(f"No complete seeds in {RESULTS} for --steps {args.steps}.")
        return

    N = torch.tensor([r["N"] for r in next(iter(by_seed.values()))], dtype=torch.double)
    span = math.log10(N.max() / N.min())
    print(f"\nRange: {span:.2f} decades of parameters; held-out N = {N[-1].item():,.0f}")

    # ---- per-seed fit and held-out prediction
    results = {}
    print(f"\nPer seed: fit on {len(WIDTHS) - 1} smaller models, predict the largest "
          f"({args.steps} steps)")
    print(f"  {'seed':>4} {'actual':>8} {'E+A/N^a':>9} {'err%':>7} {'E':>6} {'alpha':>6}"
          f" {'pure PL':>9} {'err%':>7}")
    for s, rows in by_seed.items():
        L = torch.tensor([r["val_loss"] for r in rows], dtype=torch.double)
        res = fit_and_predict(N, L)
        results[s] = (L, res)
        print(f"  {s:>4} {res['actual']:>8.4f} {res['p_shift']:>9.4f} {res['err_shift']:>+7.2f}"
              f" {res['E']:>6.3f} {res['alpha']:>6.3f} {res['p_pl']:>9.4f} {res['err_pl']:>+7.2f}")

    # ---- spread across seeds
    n = len(results)
    errs_shift = [r["err_shift"] for _, r in results.values()]
    errs_pl = [r["err_pl"] for _, r in results.values()]
    held = [r["actual"] for _, r in results.values()]
    m_e, s_e = mean_std(errs_shift)
    m_p, s_p = mean_std(errs_pl)
    m_h, s_h = mean_std(held)
    print(f"\nAcross {n} seed(s):")
    print(f"  prediction error, E + A/N^a : mean {m_e:+.2f}%, sd {s_e:.2f}%")
    print(f"  prediction error, pure PL   : mean {m_p:+.2f}%, sd {s_p:.2f}%")
    print(f"  held-out loss               : mean {m_h:.4f}, sd {s_h:.4f} "
          f"({100 * s_h / m_h:.2f}% of mean)")
    if n < 3:
        print("  (fewer than 3 seeds: the sd is too rough to interpret)")
    else:
        verdict = ("larger than" if abs(m_e) > 2 * s_e else "not clearly larger than")
        print(f"  mean |error| is {verdict} twice the seed sd (informal check, not a test)")

    # ---- fit on the seed-mean curve
    L_mean = torch.stack([L for L, _ in results.values()]).mean(dim=0)
    mean_res = fit_and_predict(N, L_mean)
    print(f"\nFit on seed-mean curve: E + A/N^a -> {mean_res['p_shift']:.4f} "
          f"(error {mean_res['err_shift']:+.2f}%, E = {mean_res['E']:.3f}, "
          f"alpha = {mean_res['alpha']:.3f}); actual {mean_res['actual']:.4f}")
    print("With this few decades of range, E and alpha trade off against each other,")
    print("so treat the fitted E as weakly determined.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping plot. pip install matplotlib)")
        return

    grid = 10 ** torch.linspace(math.log10(N.min()) - 0.1, math.log10(N.max()) + 1.0, 200,
                                dtype=torch.double)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (s, (L, _)) in enumerate(results.items()):
        ax.loglog(N, L, ".", color="gray", alpha=0.5,
                  label="individual seeds" if i == 0 else None)
    ax.loglog(N[:-1], L_mean[:-1], "o", label="seed mean, training sizes (used in fit)")
    ax.loglog([N[-1].item()], [L_mean[-1].item()], "s", ms=9, color="crimson",
              label="seed mean, held-out model")
    ax.loglog(grid, mean_res["pred_fn"](grid), "-",
              label=f"E + A/N^a  (a={mean_res['alpha']:.2f})")
    ax.loglog(grid, mean_res["pl_fn"](grid), "--",
              label=f"pure power law (b={mean_res['beta']:.2f})")
    ax.axvline(N[:-1].max().item(), color="gray", ls=":", lw=1)
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("validation loss (nats/char)")
    ax.set_title(f"Scaling law fitted on small models, tested on a larger one "
                 f"({n} seed{'s' if n != 1 else ''})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("scaling_fit.png", dpi=150)
    print(f"\nSaved scaling_fit.png and {RESULTS}")


if __name__ == "__main__":
    main()
