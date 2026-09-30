"""Powers of Ten on real models: does a scaling law fitted on small models
predict the loss of a larger one?

Steps
  1. Train tiny char-level transformers at increasing width (fixed data budget)
  2. Record validation loss vs non-embedding parameter count N
  3. Fit L(N) = E + A / N^alpha on all but the largest model
  4. Compare the prediction with the held-out model; plot on log-log axes

Corpus: your own Python standard library source (offline, no downloads).

Usage
  python eames_real.py                 # train (cached) + fit + plot
  python eames_real.py --steps 3000    # longer training, cleaner curve
  (progress is saved after every model; rerun the same command to resume)
  python eames_real.py --skip-train    # refit/replot from results.json
"""
import argparse
import glob
import json
import math
import os
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
    g = torch.Generator().manual_seed(123)
    model.eval()
    tot = 0.0
    for _ in range(batches):
        x, y = get_batch(data, g)
        tot += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    model.train()
    return tot / batches


def train_one(d, train, val, vocab, steps):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
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


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if args.skip_train and os.path.exists(RESULTS):
        rows = json.load(open(RESULTS))
    else:
        train, val, vocab = load_corpus()
        print(f"Corpus: {len(train) + len(val):,} chars, vocab {vocab}, "
              f"{args.steps} steps x {BATCH * BLOCK} tokens per model")
        done = {}
        if os.path.exists(RESULTS):
            done = {r["d"]: r for r in json.load(open(RESULTS))
                    if r.get("steps") == args.steps}
        rows = []
        for d in WIDTHS:
            if d in done:
                r = done[d]
                rows.append(r)
                print(f"  d={d:<4} N={r['N']:>9,}  val loss = {r['val_loss']:.4f}  (cached)")
                continue
            t0 = time.time()
            N, vl = train_one(d, train, val, vocab, args.steps)
            rows.append({"d": d, "N": N, "val_loss": vl, "steps": args.steps})
            print(f"  d={d:<4} N={N:>9,}  (10^{math.log10(N):.2f})  val loss = {vl:.4f}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
            json.dump(rows, open(RESULTS, "w"), indent=2)   # save after every model

    N = torch.tensor([r["N"] for r in rows], dtype=torch.double)
    L = torch.tensor([r["val_loss"] for r in rows], dtype=torch.double)
    span = math.log10(N.max() / N.min())
    print(f"\nRange: {span:.2f} decades of parameters")

    N_fit, L_fit = N[:-1], L[:-1]
    N_hold, L_hold = N[-1].item(), L[-1].item()

    pred_fn, E, alpha = fit_shifted(N_fit, L_fit)
    pl_fn, beta = fit_powerlaw(N_fit, L_fit)
    p1, p2 = float(pred_fn(N_hold)), float(pl_fn(N_hold))

    print(f"\nFit on {len(N_fit)} smaller models, predict N = {N_hold:,.0f}:")
    print(f"  E + A/N^alpha : E = {E:.3f}, alpha = {alpha:.3f} -> "
          f"{p1:.4f}  (error {100 * (p1 - L_hold) / L_hold:+.2f}%)")
    print(f"  pure power law: beta = {beta:.3f}           -> "
          f"{p2:.4f}  (error {100 * (p2 - L_hold) / L_hold:+.2f}%)")
    print(f"  actual        : {L_hold:.4f}")
    print("\nWith this few decades of range, E and alpha trade off against each other,")
    print("so treat the fitted E as weakly determined. Compare the two errors above.")

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
    ax.loglog(N_fit, L_fit, "o", label="training sizes (used in fit)")
    ax.loglog([N_hold], [L_hold], "s", ms=9, color="crimson", label="held-out model")
    ax.loglog(grid, pred_fn(grid), "-", label=f"E + A/N^a  (a={alpha:.2f})")
    ax.loglog(grid, pl_fn(grid), "--", label=f"pure power law (b={beta:.2f})")
    ax.axvline(N_fit.max(), color="gray", ls=":", lw=1)
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("validation loss (nats/char)")
    ax.set_title("Scaling law fitted on small models, tested on a larger one")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("scaling_fit.png", dpi=150)
    print("\nSaved scaling_fit.png and results.json")


if __name__ == "__main__":
    main()
