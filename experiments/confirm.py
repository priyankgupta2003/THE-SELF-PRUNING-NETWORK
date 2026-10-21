"""Confirmatory run on FRESH seeds for the headline claim.

Why this exists: the 98% advantage of Fisher over magnitude was first seen in a
4-seed pilot (seeds 0-3) and then in the main sweep (seeds 0-9, overlapping
the pilot). To avoid claiming something we went looking for, the claim is
re-tested here on seeds 10-19, which were never run during development, with
every setting frozen. Writes results/confirmation.csv and results/CLAIM.md.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import csv
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyze import boot_ci, sign_flip_p  # noqa: E402
from tinyprune.data import load  # noqa: E402
from tinyprune.train import Config, train  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = list(range(10, 20))
LEVELS = [0.9, 0.95, 0.98]
DATA = load("digits", 0)


def run(job):
    crit, sp, seed = job
    r = train(Config(sparsity=sp, seed=seed, criterion=crit) if sp > 0 else Config(seed=seed), DATA)
    return dict(criterion=crit, target=sp, seed=seed, achieved=round(r["sparsity"], 6), test_acc=r["test_acc"])


def main():
    jobs = [("dense", 0.0, s) for s in SEEDS] + \
           [(c, sp, s) for c in ("fisher", "magnitude") for sp in LEVELS for s in SEEDS]
    with Pool(os.cpu_count()) as p:
        rows = p.map(run, jobs, chunksize=1)
    with open(os.path.join(OUT, "confirmation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    acc = {(r["criterion"], r["target"], r["seed"]): r["test_acc"] for r in rows}
    dense = np.mean([acc[("dense", 0.0, s)] for s in SEEDS])
    table = []
    for sp in LEVELS:
        f = np.array([acc[("fisher", sp, s)] for s in SEEDS])
        m = np.array([acc[("magnitude", sp, s)] for s in SEEDS])
        d = (f - m) * 100
        lo, hi = boot_ci(d)
        table.append((sp, f.mean(), f.std(ddof=1), m.mean(), m.std(ddof=1), d.mean(), lo, hi,
                      int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()), sign_flip_p(d)))
    h = next(t for t in table if t[0] == 0.98)
    lines = [
        "# Falsifiable claim",
        "",
        "**Claim.** On sklearn digits with a 64-128-64-10 ReLU MLP trained for 60 epochs with the cubic",
        "schedule in this repo, at **98% weight sparsity** (340 of 17,024 weights kept), Fisher/OBD",
        f"self-pruning reaches **{h[1]:.1%}** mean test accuracy versus **{h[3]:.1%}** for magnitude pruning",
        f"under the identical layerwise budget, schedule, init and batch order: a paired gain of",
        f"**{h[5]:+.1f} pp** (95% bootstrap CI [{h[6]:+.1f}, {h[7]:+.1f}]; wins/ties/losses {h[8]}/{h[9]}/{h[10]};",
        f"exact sign-flip p = {h[11]:.4f}, the smallest possible with 10 seeds), on fresh seeds 10-19.",
        f"Dense reference on the same seeds: {dense:.1%}.",
        "",
        "**What would falsify it.** Running `python experiments/confirm.py` with any other 10 seeds and",
        "getting a 95% CI for the paired difference that includes 0.",
        "",
        "**What it does NOT say.** At 90% sparsity Fisher is NOT better: magnitude is, if anything,",
        "slightly ahead (see table; on seeds 0-9 the gap was -0.08 pp, p = 0.82). Fisher's advantage appears only once the budget is tight enough that *which*",
        "weights survive matters more than the network's slack. Also, global magnitude pruning",
        "(results/paired_tests.csv) is within noise of layerwise Fisher up to 98%; Fisher's clear win",
        "over every baseline is at 99%. Part of what Fisher buys is a better per-layer allocation.",
        "",
        "## Confirmation set (seeds 10-19, never used during development)",
        "",
        "| sparsity | Fisher mean ± sd | magnitude mean ± sd | Fisher − mag (pp) | 95% CI | W/T/L | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in table:
        lines.append(f"| {t[0]:.0%} | {t[1]:.2%} ± {t[2]:.2%} | {t[3]:.2%} ± {t[4]:.2%} | {t[5]:+.2f} | "
                     f"[{t[6]:+.2f}, {t[7]:+.2f}] | {t[8]}/{t[9]}/{t[10]} | {t[11]:.4f} |")
    lines += ["", "Test set: 359 images, 1 image = 0.28 pp. Final-epoch accuracy only; nothing selected on test.",
              "", "Main-sweep paired tests (seeds 0-9) for every method pair: `results/paired_tests.csv`."]
    with open(os.path.join(OUT, "CLAIM.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
