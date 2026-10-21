"""Self-pruning: importance criteria, cubic schedule, honest masking, regrowth.

Importance criterion ("fisher") -- what it estimates
---------------------------------------------------
Removing weight w_i means moving it from w_i to 0, i.e. dw_i = -w_i. A second
order Taylor expansion of the loss around the current weights gives

    dL_i  ~=  g_i * dw_i + 1/2 * H_ii * dw_i^2  =  -g_i w_i + 1/2 H_ii w_i^2

(cross terms H_ij dropped: we score one removal at a time -- the OBD
assumption of LeCun et al. 1990). We need g_i and H_ii:

* g_i  : an exponential moving average of the mini-batch gradient. A single
         mini-batch gradient is very noisy; its average estimates the
         full-training-set gradient that actually appears in the expansion.
* H_ii : for cross-entropy, the Hessian is approximated by the empirical
         Fisher F_ii = E_n[(d l_n / d w_i)^2] (Gauss-Newton argument; exact
         at the optimum for a well-specified model). The PER-SAMPLE gradient is
         needed -- squaring a mini-batch average would shrink F by ~1/B. For a
         linear layer z = x W the per-sample gradient is the outer product
         x_n (outer) delta_n, so
             F = (1/B) sum_n (x_n^2)(outer)(delta_n^2) = (1/B) (X^2)^T (Delta^2)
         -- one extra matmul per layer per step, no per-sample loop.

Full second-order score ("taylor2"):  1/2 F_ii w_i^2 - g_i w_i
Score actually used  ("fisher"):     1/2 F_ii w_i^2           (OBD)

Why drop the first-order term? Not by assumption -- by measurement
(experiments/validate_criterion.py). Scored ONE weight at a time, taylor2 is
the best predictor of the true loss increase early in training. But pruning
removes MANY weights at once, and the -g_i w_i terms are not independent:
the weights whose removal "looks helpful" to first order are exactly those
where -w_i points down the gradient, so removing hundreds of them together is
one big uncontrolled step along the gradient, and the neglected cross terms
H_ij w_i w_j add up coherently. Measured one-shot at 75% removal, taylor2 is
far worse than either magnitude or OBD. The quadratic term is >= 0 for every
weight, so OBD's errors do not line up in one direction. Near a stationary
point (where gradual pruning keeps the network, since Adam re-converges
between pruning events) E[g] ~ 0 anyway, which is OBD's own justification.

Why this beats |w|: magnitude assumes every weight's loss curvature is the
same. A large weight on a dead/uninformative input (x_i ~ 0 for most samples)
has F_ii ~ 0 and is free to remove; a small weight on a high-activity input
feeding a sensitive unit can be expensive. Saliency also lives in *loss
units*, so scores are comparable across layers of different fan-in, which
magnitude is not (He-init scale differs by layer).

Regrowth (RigL-style)
---------------------
For a pruned connection the parameter gradient is 0 by construction, but the
gradient w.r.t. the *effective* weight, dL/dW_eff = x^T delta, is well defined
and equals the first-order loss decrease from re-enabling it:
dL ~= g_i * dw_i, maximised in magnitude by large |g_i|. We regrow the
pruned entries with the largest EMA |dL/dW_eff|, initialise them at exactly 0
(so the network function does not jump at the moment of regrowth) with
freshly reset optimizer state, protect them from being dropped for a grace
period, and decay the swap fraction to 0 so the mask freezes before training
ends (stability: the final fine-tuning phase sees a fixed architecture).
"""
from __future__ import annotations

import math

import numpy as np


def cubic_sparsity(step: int, final: float, start: int, end: int, initial: float = 0.0) -> float:
    """Zhu & Gupta (2017) schedule: s_t = s_f + (s_i - s_f)(1 - (t-t0)/(t1-t0))^3.

    Prunes fast early (many redundant weights, network can recover) and slowly
    near the end (every remaining weight matters; give the optimizer time).
    """
    if step < start:
        return initial
    if step >= end:
        return final
    frac = (step - start) / max(1, end - start)
    return final + (initial - final) * (1.0 - frac) ** 3


class Pruner:
    def __init__(self, model, optimizer, final_sparsity: float, start_step: int, end_step: int,
                 every: int = 10, criterion: str = "fisher", allocation: str = "layerwise",
                 regrow_frac: float = 0.0, regrow_until: int | None = None,
                 regrow_every: int = 100, ema: float = 0.95, min_keep_frac: float = 0.01):
        assert criterion in ("magnitude", "fisher", "taylor1", "taylor2")
        assert allocation in ("layerwise", "global")
        if regrow_frac > 0 and allocation != "layerwise":
            raise ValueError("regrowth is implemented for layerwise allocation")
        self.model, self.opt = model, optimizer
        self.final, self.start, self.end, self.every = final_sparsity, start_step, end_step, every
        self.criterion, self.allocation = criterion, allocation
        self.regrow_frac = regrow_frac
        self.regrow_until = regrow_until if regrow_until is not None else end_step
        # swaps happen on a slower clock than pruning; a regrown weight is
        # protected for one full swap interval so it is judged only after it
        # has had regrow_every optimizer steps to become useful
        self.regrow_every = regrow_every
        self.grace = regrow_every
        self.ema, self.min_keep_frac = ema, min_keep_frac
        self.stats = [dict(g=np.zeros(l.mask.shape), F=np.zeros(l.mask.shape), n=0,
                           grown_at=np.full(l.mask.shape, -10**9))
                      for l in model.layers]
        self.history = []  # (step, target, achieved, n_regrown)
        self._reached = False

    # ---------------------------------------------------------------- stats
    def collect(self):
        """Call after loss.backward(), before optimizer.step()."""
        if self.final <= 0:
            return
        need_F = self.criterion in ("fisher", "taylor2")
        for l, st in zip(self.model.layers, self.stats):
            gd = l.w_eff.grad  # dense dL/dW_eff (nonzero also where pruned)
            st["g"] = self.ema * st["g"] + (1 - self.ema) * gd
            if need_F:
                X, D = l.x.data, l.z.grad  # D = dL/dz of the *mean* loss = delta_n / B
                B = X.shape[0]
                # per-sample grads x_n delta_n with delta_n = B * D_n:
                # F = (1/B) sum_n x_n^2 (B D_n)^2 = B * (X^2)^T (D^2)
                st["F"] = self.ema * st["F"] + (1 - self.ema) * (B * (X * X).T @ (D * D))
            st["n"] += 1

    def _debiased(self, st, key):
        c = 1.0 - self.ema ** max(1, st["n"])
        return st[key] / c

    def scores(self, i: int) -> np.ndarray:
        """Importance of every entry of layer i (higher = keep)."""
        l, st = self.model.layers[i], self.stats[i]
        w = l.W.data
        if self.criterion == "magnitude":
            return np.abs(w)
        g = self._debiased(st, "g")
        if self.criterion == "taylor1":
            return np.abs(g * w)
        F = self._debiased(st, "F")
        if self.criterion == "fisher":
            return 0.5 * F * w * w            # OBD with empirical-Fisher curvature
        return 0.5 * F * w * w - g * w        # full 2nd-order Taylor (ablation only)

    # -------------------------------------------------------------- masking
    def _apply(self, layer, new_mask: np.ndarray):
        old = layer.mask.copy()
        dropped = old & ~new_mask
        grown = ~old & new_mask
        layer.mask[...] = new_mask          # in place: optimizer holds a reference
        layer.W.data[~new_mask] = 0.0       # real zeros
        self.opt.reset_entries(layer.W, dropped | grown)
        return dropped, grown

    def _regrow_fraction(self, step: int) -> float:
        if self.regrow_frac <= 0 or step >= self.regrow_until or step % self.regrow_every:
            return 0.0
        return self.regrow_frac * 0.5 * (1 + math.cos(math.pi * step / self.regrow_until))

    def maybe_update(self, step: int):
        if self.final <= 0 or step < self.start:
            return
        swapping = self._regrow_fraction(step) > 0
        if step % self.every and not swapping:
            return
        if self._reached and not swapping:
            return
        target = cubic_sparsity(step, self.final, self.start, self.end)
        self._reached = target >= self.final
        if self.allocation == "global":
            self._update_global(target)
            regrown = 0
        else:
            regrown = sum(self._update_layer(i, target, step) for i in range(len(self.model.layers)))
        self.history.append((step, target, self.model.sparsity(), regrown))

    def _update_layer(self, i: int, target: float, step: int) -> int:
        l, st = self.model.layers[i], self.stats[i]
        n = l.mask.size
        k_keep = max(int(math.ceil(self.min_keep_frac * n)), int(round((1 - target) * n)))
        k_keep = min(k_keep, l.n_active)  # pruning is monotone except via regrowth
        s = self.scores(i)
        active = l.mask.copy()
        r = int(round(self._regrow_fraction(step) * k_keep))
        if r == 0 and k_keep == l.n_active:
            return 0
        protected = active & (step - st["grown_at"] < self.grace)
        n_prot = int(protected.sum())
        if n_prot > k_keep:  # budget wins over grace: keep the best protected ones
            ps = np.where(protected, s, -np.inf).ravel()
            protected = np.zeros(n, dtype=bool)
            protected[np.argpartition(-ps, k_keep - 1)[:k_keep]] = True
            protected = protected.reshape(l.mask.shape)
            n_prot = k_keep
        r = max(0, min(r, k_keep - n_prot, int((~active).sum())))
        # 1) keep the best (k_keep - r) active weights (protected ones first)
        cand = np.where(active & ~protected, s, -np.inf).ravel()
        n_free = k_keep - r - n_prot
        new = protected.copy().ravel()
        if n_free > 0:
            new[np.argpartition(-cand, n_free - 1)[:n_free]] = True
        new = new.reshape(l.mask.shape)
        # 2) regrow r previously-pruned entries with largest |dL/dW_eff|
        if r > 0:
            gabs = np.abs(self._debiased(st, "g"))
            pool = np.where(~active, gabs, -np.inf).ravel()  # excludes just-dropped
            idx = np.argpartition(-pool, r - 1)[:r]
            new.ravel()[idx] = True
            st["grown_at"].ravel()[idx] = step
        assert new.sum() == k_keep
        self._apply(l, new)
        return r

    def _update_global(self, target: float):
        layers = self.model.layers
        total = sum(l.mask.size for l in layers)
        k_total = int(round((1 - target) * total))
        k_total = min(k_total, sum(l.n_active for l in layers))
        if k_total == sum(l.n_active for l in layers):
            return
        scores = [np.where(l.mask, self.scores(i), -np.inf) for i, l in enumerate(layers)]
        new_masks = [np.zeros_like(l.mask) for l in layers]
        # each layer keeps a floor of its own best entries (no layer collapse)
        for l, s, nm in zip(layers, scores, new_masks):
            floor = min(l.n_active, int(math.ceil(self.min_keep_frac * l.mask.size)))
            nm.ravel()[np.argpartition(-s.ravel(), floor - 1)[:floor]] = True
        rest = k_total - sum(int(nm.sum()) for nm in new_masks)
        if rest > 0:
            flat = np.concatenate([np.where(nm, -np.inf, s).ravel() for s, nm in zip(scores, new_masks)])
            pick = np.argpartition(-flat, rest - 1)[:rest]
            offs = np.cumsum([0] + [l.mask.size for l in layers])
            for j, (a, b) in enumerate(zip(offs[:-1], offs[1:])):
                sel = pick[(pick >= a) & (pick < b)] - a
                new_masks[j].ravel()[sel] = True
        for l, nm in zip(layers, new_masks):
            self._apply(l, nm)
