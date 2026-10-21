"""Summaries, paired significance tests, Pareto plot and the falsifiable claim."""
import csv
import itertools
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
HEADLINE_SPARSITY = 0.9   # the level named in the brief; fixed in advance
PAIRS = [("fisher", "magnitude"), ("fisher-global", "magnitude-global"),
         ("fisher+regrow", "fisher"), ("fisher", "magnitude-global")]


def load():
    with open(os.path.join(OUT, "sweep_raw.csv")) as f:
        return list(csv.DictReader(f))


def sign_flip_p(d):
    """Exact two-sided paired permutation test (all 2^n sign flips)."""
    d = np.asarray(d)
    obs = abs(d.mean())
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    return float((np.abs((flips * d).mean(1)) >= obs - 1e-12).mean())


def boot_ci(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rows = load()
    acc = defaultdict(dict)    # (method, target) -> seed -> acc
    ach = defaultdict(list)
    for r in rows:
        acc[(r["method"], float(r["target"]))][int(r["seed"])] = float(r["test_acc"])
        ach[(r["method"], float(r["target"]))].append(float(r["achieved"]))
    dense = np.array(list(acc[("dense", 0.0)].values()))

    # ---- summary
    summ = []
    for (m, t), d in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        a = np.array(list(d.values()))
        summ.append(dict(method=m, target_sparsity=t, achieved_sparsity=round(float(np.mean(ach[(m, t)])), 5),
                         n_seeds=len(a), mean_acc=round(a.mean(), 4), std_acc=round(a.std(ddof=1), 4),
                         min_acc=round(a.min(), 4), max_acc=round(a.max(), 4),
                         pct_of_dense=round(100 * a.mean() / dense.mean(), 2)))
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    # ---- paired tests
    tests = []
    targets = sorted({t for (_, t) in acc if t > 0})
    for a, b in PAIRS:
        for t in targets:
            if (a, t) not in acc or (b, t) not in acc:
                continue
            seeds = sorted(set(acc[(a, t)]) & set(acc[(b, t)]))
            d = np.array([acc[(a, t)][s] - acc[(b, t)][s] for s in seeds]) * 100
            lo, hi = boot_ci(d)
            tests.append(dict(a=a, b=b, sparsity=t, n=len(d), mean_diff_pp=round(d.mean(), 3),
                              sd_diff_pp=round(d.std(ddof=1), 3), ci95_lo=round(lo, 3), ci95_hi=round(hi, 3),
                              wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()),
                              p_signflip=round(sign_flip_p(d), 4)))
    with open(os.path.join(OUT, "paired_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0])); w.writeheader(); w.writerows(tests)

    # ---- Pareto plot
    style = {"magnitude": ("#8a8a8a", "o", "-"), "magnitude-global": ("#8a8a8a", "s", ":"),
             "fisher": ("#1f6fb4", "o", "-"), "fisher-global": ("#1f6fb4", "s", ":"),
             "fisher+regrow": ("#d9822b", "^", "-")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, zoom in zip(axes, (False, True)):
        ax.axhspan(dense.mean() - dense.std(ddof=1), dense.mean() + dense.std(ddof=1), color="#000", alpha=0.06)
        ax.axhline(dense.mean(), color="#000", lw=0.8, label=f"dense ({dense.mean():.3f})")
        for m, (c, mk, ls) in style.items():
            ts = [t for t in targets if (m, t) in acc]
            mean = [np.mean(list(acc[(m, t)].values())) for t in ts]
            sd = [np.std(list(acc[(m, t)].values()), ddof=1) for t in ts]
            x = [100 * (1 - np.mean(ach[(m, t)])) for t in ts]
            ax.errorbar(x, mean, yerr=sd, color=c, marker=mk, ls=ls, capsize=3, ms=5, label=m)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xticks([50, 25, 10, 5, 2, 1]); ax.set_xticklabels(["50%", "25%", "10%", "5%", "2%", "1%"])
        ax.set_xlabel("weights remaining (log scale; sparsity 50% → 99%)")
        ax.set_ylabel("test accuracy (mean ± 1 sd over seeds)")
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
        if zoom:
            ax.set_ylim(0.85, 0.985); ax.set_title("zoom")
        else:
            ax.set_title("Sparsity–accuracy trade-off, sklearn digits, MLP 64-128-64-10")
            ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pareto.png"), dpi=140)

    # ---- claim
    h = next(x for x in tests if x["a"] == "fisher" and x["b"] == "magnitude" and x["sparsity"] == HEADLINE_SPARSITY)
    fa = next(x for x in summ if x["method"] == "fisher" and x["target_sparsity"] == HEADLINE_SPARSITY)
    ma = next(x for x in summ if x["method"] == "magnitude" and x["target_sparsity"] == HEADLINE_SPARSITY)
    lines = ["# Sweep report, seeds 0-9 (auto-generated by experiments/analyze.py; the headline claim lives in CLAIM.md)", "",
             f"At {HEADLINE_SPARSITY:.0%} weight sparsity (achieved {fa['achieved_sparsity']:.4f}) on sklearn digits "
             f"with a 64-128-64-10 MLP, Fisher/OBD self-pruning reaches **{fa['mean_acc']:.2%}** mean test accuracy "
             f"({fa['pct_of_dense']:.1f}% of dense {dense.mean():.2%}) versus **{ma['mean_acc']:.2%}** for magnitude "
             f"pruning ({ma['pct_of_dense']:.1f}% of dense), averaged over {h['n']} seeds with identical schedule, "
             f"budget, init and batch order (paired).", "",
             f"Paired difference: {h['mean_diff_pp']:+.2f} pp, 95% bootstrap CI [{h['ci95_lo']:+.2f}, {h['ci95_hi']:+.2f}] pp, "
             f"wins/ties/losses {h['wins']}/{h['ties']}/{h['losses']}, exact sign-flip p = {h['p_signflip']}.", "",
             "How to falsify: rerun `python experiments/run_sweep.py && python experiments/analyze.py` with "
             "different seeds (N_SEEDS, or shift SEEDS); the claim fails if the 95% CI of the paired difference "
             "includes 0 or flips sign.", "",
             "Test-set note: 359 test images, so one image = 0.28 pp. Accuracy is measured once at the end of "
             "training; nothing (no early stopping, no hyper-parameter choice) was selected on the test set "
             "during the sweep.", "", "## All paired comparisons", "",
             "| A | B | sparsity | mean A−B (pp) | 95% CI | W/T/L | p |", "|---|---|---|---|---|---|---|"]
    for x in tests:
        lines.append(f"| {x['a']} | {x['b']} | {x['sparsity']} | {x['mean_diff_pp']:+.2f} | "
                     f"[{x['ci95_lo']:+.2f}, {x['ci95_hi']:+.2f}] | {x['wins']}/{x['ties']}/{x['losses']} | {x['p_signflip']} |")
    with open(os.path.join(OUT, "sweep_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
