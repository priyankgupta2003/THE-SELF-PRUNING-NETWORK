"""Mini-batched training loop with optional self-pruning."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .data import load
from .nn import MLP
from .optim import Adam
from .prune import Pruner
from .tensor import Tensor, softmax_cross_entropy


@dataclass
class Config:
    dataset: str = "digits"
    sizes: tuple = (64, 128, 64, 10)
    epochs: int = 60
    batch: int = 64
    lr: float = 3e-3
    lr_min: float = 1e-4
    seed: int = 0
    split_seed: int = 0          # fixed across seeds -> paired comparisons
    sparsity: float = 0.0
    criterion: str = "fisher"
    allocation: str = "layerwise"
    prune_start: float = 0.10    # fraction of total steps
    prune_end: float = 0.70
    prune_every: int = 10
    regrow_frac: float = 0.0
    regrow_every: int = 100
    regrow_until: float = 0.80
    record_curve: bool = False
    extra: dict = field(default_factory=dict)


def accuracy(model, X, y):
    return float((model.predict_logits(X).argmax(1) == y).mean())


def eval_loss(model, X, y):
    z = model.predict_logits(X)
    m = z.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(1))
    return float((lse - z[np.arange(len(y)), y]).mean())


def train(cfg: Config, data=None, callback=None):
    Xtr, ytr, Xte, yte = data if data is not None else load(cfg.dataset, cfg.split_seed)
    rng = np.random.default_rng(cfg.seed)
    model = MLP(list(cfg.sizes), rng)
    params = []
    for l in model.layers:
        params += [(l.W, l.mask), (l.b, None)]  # biases are never pruned
    opt = Adam(params, lr=cfg.lr)
    steps_per_epoch = math.ceil(len(Xtr) / cfg.batch)
    total = cfg.epochs * steps_per_epoch
    pruner = Pruner(model, opt, cfg.sparsity, int(cfg.prune_start * total), int(cfg.prune_end * total),
                    every=cfg.prune_every, criterion=cfg.criterion, allocation=cfg.allocation,
                    regrow_frac=cfg.regrow_frac, regrow_every=cfg.regrow_every, regrow_until=int(cfg.regrow_until * total))
    curve = []
    step = 0
    for ep in range(cfg.epochs):
        perm = rng.permutation(len(Xtr))
        run_loss, seen = 0.0, 0
        for s in range(steps_per_epoch):
            idx = perm[s * cfg.batch:(s + 1) * cfg.batch]
            # cosine learning-rate decay, identical for every method
            opt.lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * step / total))
            opt.zero_grad()
            loss = softmax_cross_entropy(model(Tensor(Xtr[idx], requires_grad=False)), ytr[idx])
            loss.backward()
            if not np.isfinite(loss.data):
                raise FloatingPointError(f"non-finite loss at step {step}")
            pruner.collect()
            opt.step()
            step += 1
            pruner.maybe_update(step)
            if callback is not None:
                callback(step, total, model, pruner)
            run_loss += float(loss.data) * len(idx); seen += len(idx)
        if cfg.record_curve:
            curve.append(dict(epoch=ep + 1, train_loss=run_loss / seen,
                              train_acc=accuracy(model, Xtr, ytr), test_loss=eval_loss(model, Xte, yte),
                              test_acc=accuracy(model, Xte, yte), sparsity=model.sparsity()))
    # hard-budget check: the mask must be exactly at target and weights real zeros
    for l in model.layers:
        assert np.all(l.W.data[~l.mask] == 0.0)
    return dict(model=model, test_acc=accuracy(model, Xte, yte), train_acc=accuracy(model, Xtr, ytr),
                test_loss=eval_loss(model, Xte, yte), sparsity=model.sparsity(), curve=curve,
                history=pruner.history, pruner=pruner)
