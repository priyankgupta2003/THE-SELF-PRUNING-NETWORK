"""Does each importance score actually predict "loss increase if removed"?

At two moments (when pruning starts = 10% of training, and at the end of dense
training) we take a snapshot and compare, for each criterion:
  (1) single-weight test: Spearman rank correlation between predicted dL and
      the TRUE dL measured by zeroing each of 600 sampled weights and
      recomputing the full training loss;
  (2) group test: remove the lowest-scored 50/75/90% of each layer at once
      (one-shot, no retraining) and measure the true training loss.
Scores use the EMA statistics the pruner really has at that time, plus an
"oracle" version computed from the full training set, to separate
"bad criterion" from "noisy estimate".
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tinyprune.data import load
from tinyprune.tensor import Tensor, softmax_cross_entropy
from tinyprune.train import Config, accuracy, eval_loss, train

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def full_batch_stats(model, X, y):
    model.zero_grad()
    softmax_cross_entropy(model(Tensor(X, requires_grad=False)), y).backward()
    out = []
    for l in model.layers:
        B = X.shape[0]
        D = l.z.grad
        out.append(dict(g=l.w_eff.grad.copy(), F=B * (l.x.data ** 2).T @ (D ** 2)))
    return out


def criteria(w, g, F):
    return {
        "magnitude |w|": np.abs(w),
        "taylor1 |g w|": np.abs(g * w),
        "OBD 1/2 F w^2": 0.5 * F * w * w,
        "taylor2 1/2Fw^2 - g w": 0.5 * F * w * w - g * w,
    }


def analyse(model, pruner, X, y, Xte, yte, rng):
    base = eval_loss(model, X, y)
    ema = [dict(g=pruner._debiased(st, "g"), F=pruner._debiased(st, "F")) for st in pruner.stats]
    oracle = full_batch_stats(model, X, y)
    res = {"base_train_loss": base, "single": {}, "group": {}}
    # (1) single-weight correlation
    true_d, pred = [], {}
    samples = [(li, rng.integers(0, l.mask.shape[0]), rng.integers(0, l.mask.shape[1]))
               for li, l in enumerate(model.layers) for _ in range(200)]
    for li, i, j in samples:
        W = model.layers[li].W.data
        old = W[i, j]; W[i, j] = 0.0
        true_d.append(eval_loss(model, X, y) - base)
        W[i, j] = old
        w = old
        for src, st in [("ema", ema[li]), ("oracle", oracle[li])]:
            for k, v in criteria(np.array(w), st["g"][i, j], st["F"][i, j]).items():
                pred.setdefault(f"{k} [{src}]", []).append(float(v))
    true_d = np.array(true_d)
    for k, v in pred.items():
        res["single"][k] = spearman(v, true_d)
    # (2) group removal, one-shot, per layer
    for src, stats in [("ema", ema), ("oracle", oracle)]:
        for name in criteria(np.zeros(1), np.zeros(1), np.zeros(1)):
            for s in (0.5, 0.75, 0.9):
                saved = [l.W.data.copy() for l in model.layers]
                for l, st in zip(model.layers, stats):
                    sc = criteria(l.W.data, st["g"], st["F"])[name]
                    k = int(round(s * sc.size))
                    idx = np.argsort(sc.ravel())[:k]
                    l.W.data.ravel()[idx] = 0.0
                res["group"][f"{name} [{src}] @ {s}"] = dict(
                    train_loss=eval_loss(model, X, y), test_acc=accuracy(model, Xte, yte))
                for l, w in zip(model.layers, saved):
                    l.W.data[...] = w
    return res


def main():
    data = load("digits", 0)
    Xtr, ytr, Xte, yte = data
    rng = np.random.default_rng(0)
    snaps = {}

    def cb(step, total, model, pruner):
        if step in (int(0.10 * total), total):
            snaps[step / total] = analyse(model, pruner, Xtr, ytr, Xte, yte, rng)

    # sparsity 1e-9 -> pruner collects statistics but never removes anything
    train(Config(sparsity=1e-9, prune_start=2.0, criterion="fisher"), data, callback=cb)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "criterion_validation.json"), "w") as f:
        json.dump(snaps, f, indent=1)
    for t, r in snaps.items():
        print(f"\n=== snapshot at {t:.0%} of training (train loss {r['base_train_loss']:.4f}) ===")
        print("Spearman(predicted dL, true dL), 600 single weights:")
        for k, v in sorted(r["single"].items(), key=lambda kv: -kv[1]):
            print(f"  {v:+.3f}  {k}")
        print("One-shot group removal (train loss / test acc):")
        for k, v in r["group"].items():
            print(f"  {k:45s} {v['train_loss']:.4f} / {v['test_acc']:.4f}")


if __name__ == "__main__":
    main()
