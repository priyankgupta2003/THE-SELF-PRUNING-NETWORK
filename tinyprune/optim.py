"""Optimizers, written to be correct under a changing sparsity mask.

What goes wrong in a naive Adam when weights are masked
-------------------------------------------------------
1. *Ghost momentum.* The masked gradient of a pruned weight is 0, but Adam's
   update is lr * m_hat / (sqrt(v_hat) + eps). If m is non-zero from before the
   weight was pruned, m decays geometrically yet stays non-zero, so the weight
   keeps drifting away from 0 for dozens of steps. Re-applying the mask in the
   forward pass hides this from the loss but the stored value is not a real
   zero, the "pruned" weight is secretly being trained by stale momentum, and
   on regrowth it re-enters with a value nobody chose.
2. *Stale moments on revival.* If m, v are kept while a weight is pruned and
   reused when it is regrown, the first updates of the revived weight are
   driven by the gradient history of a different network.
3. *Global bias correction.* Adam divides by (1 - beta^t) with a single global
   step counter t. Reset m = v = 0 for a revived weight at global step t=5000
   and the bias correction is ~1, so the first step is
       lr * (1-b1) g / sqrt((1-b2) g^2) = lr * 0.1/0.0316 ~= 3.16 * lr
   -- a 3x step-size spike exactly at the moment the weight is most fragile.

Fix implemented here
--------------------
* Updates, moments and step counts are only touched where the mask is 1.
* On prune AND on regrow, that entry's m, v and per-entry step counter are
  reset to 0 and the weight is set to exactly 0.0.
* Bias correction uses a *per-entry* step counter, so a revived connection
  behaves exactly like a freshly-initialised Adam parameter (first step has
  magnitude lr, as Adam intends). tests/test_masking.py checks this exactly.
"""
from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """params: list of (Tensor, mask_or_None). The mask array is held by
        reference, so the pruner can mutate it in place."""
        self.params = params
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.state = {}
        for p, _ in params:
            self.state[id(p)] = dict(m=np.zeros_like(p.data), v=np.zeros_like(p.data),
                                     t=np.zeros(p.data.shape, dtype=np.int64))

    def zero_grad(self):
        for p, _ in self.params:
            p.zero_grad()

    def reset_entries(self, p, where: np.ndarray):
        """Forget optimizer history for entries of p (called on prune and regrow)."""
        st = self.state[id(p)]
        st["m"][where] = 0.0
        st["v"][where] = 0.0
        st["t"][where] = 0

    def step(self):
        for p, mask in self.params:
            if p.grad is None:
                continue
            st = self.state[id(p)]
            live = np.ones(p.data.shape, dtype=bool) if mask is None else mask
            g = np.where(live, p.grad, 0.0)  # defensive: grad is already 0 there
            st["t"][live] += 1
            st["m"] = np.where(live, self.b1 * st["m"] + (1 - self.b1) * g, 0.0)
            st["v"] = np.where(live, self.b2 * st["v"] + (1 - self.b2) * g * g, 0.0)
            t = np.maximum(st["t"], 1)
            m_hat = st["m"] / (1 - self.b1 ** t)
            v_hat = st["v"] / (1 - self.b2 ** t)
            upd = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.wd:
                upd = upd + self.lr * self.wd * p.data  # decoupled (AdamW)
            p.data -= np.where(live, upd, 0.0)
            if mask is not None:
                p.data[~mask] = 0.0  # pruned weights are real zeros, always


class NaiveAdam:
    """Textbook Adam that ignores masks entirely (global t, no resets).

    Intentionally WRONG for sparse training. Kept only so the tests can
    demonstrate the failure modes the real ``Adam`` avoids.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for p, _ in params}
        self.v = {id(p): np.zeros_like(p.data) for p, _ in params}

    def step(self):
        self.t += 1
        for p, _ in self.params:
            g = p.grad
            m, v = self.m[id(p)], self.v[id(p)]
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            p.data -= self.lr * (m / (1 - self.b1 ** self.t)) / (
                np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)
