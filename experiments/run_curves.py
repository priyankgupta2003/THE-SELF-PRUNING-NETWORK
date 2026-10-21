"""Part 2/3 learning curves: dense training, and self-pruning to 90% sparsity."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tinyprune.data import load
from tinyprune.train import Config, train

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    data = load("digits", 0)
    runs = {"dense": train(Config(sparsity=0.0, record_curve=True), data),
            "fisher 90%": train(Config(sparsity=0.9, criterion="fisher", record_curve=True), data),
            "fisher+regrow 90%": train(Config(sparsity=0.9, criterion="fisher", regrow_frac=0.3,
                                              record_curve=True), data)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "learning_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "sparsity"])
        for name, r in runs.items():
            for c in r["curve"]:
                assert all(np.isfinite(v) for v in c.values()), "NaN in training curve"
                w.writerow([name, c["epoch"], c["train_loss"], c["train_acc"], c["test_loss"],
                            c["test_acc"], c["sparsity"]])

    colors = {"dense": "#4a4a4a", "fisher 90%": "#1f6fb4", "fisher+regrow 90%": "#d9822b"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, r in runs.items():
        ep = [c["epoch"] for c in r["curve"]]
        axes[0].plot(ep, [c["train_loss"] for c in r["curve"]], color=colors[name], label=f"{name} train")
        axes[0].plot(ep, [c["test_loss"] for c in r["curve"]], color=colors[name], ls="--", label=f"{name} test")
        axes[1].plot(ep, [c["test_acc"] for c in r["curve"]], color=colors[name], label=name)
        axes[2].plot(ep, [c["sparsity"] for c in r["curve"]], color=colors[name], label=name)
    axes[0].set_yscale("log"); axes[0].set_title("Cross-entropy loss"); axes[0].set_xlabel("epoch")
    axes[1].set_title("Test accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0.8, 1.0)
    axes[2].set_title("Weight sparsity (cubic schedule)"); axes[2].set_xlabel("epoch")
    for a in axes:
        a.grid(alpha=0.3); a.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False); axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curves.png"), dpi=140)
    for name, r in runs.items():
        print(f"{name:20s} final test acc {r['test_acc']:.4f}  sparsity {r['sparsity']:.4f}")


if __name__ == "__main__":
    main()
