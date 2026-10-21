"""A sparse-aware inference path -- the pruned weights are never touched.

Two independent reductions of work, both written from scratch:

1. Structural compaction (exact, no approximation):
   * a hidden unit with NO outgoing connections cannot affect the output ->
     delete its row of incoming weights and its bias;
   * a hidden unit with NO incoming connections outputs the constant
     act(b_j); its contribution act(b_j) * W_next[j, :] is folded into the next
     layer's bias and the unit is deleted.
   Repeated to a fixed point (deleting one unit can orphan another).

2. CSR storage of what remains: for each output unit j we store only the
   indices i and values of its live incoming weights. The forward pass
       y[:, j] = sum_{i in live(j)} x[:, i] * W[i, j]
   is computed as one gather + one segmented sum (np.add.reduceat), so the
   multiply-add count is exactly B * nnz, not B * fan_in * fan_out.

FLOPs are counted analytically (2 per multiply-add) for dense, masked-dense
and sparse paths; wall-clock is measured separately and reported honestly.
"""
from __future__ import annotations

import numpy as np


def _act(h, activation):
    return np.maximum(h, 0) if activation == "relu" else np.tanh(h)


class CSRLinear:
    def __init__(self, W: np.ndarray, mask: np.ndarray, b: np.ndarray):
        Wt = (W * mask).T  # (out, in): row j = incoming weights of unit j
        mt = mask.T
        self.out_dim, self.in_dim = Wt.shape
        rows, cols = np.nonzero(mt)  # row-major order -> grouped by output unit
        self.indices = cols.astype(np.int64)
        self.values = Wt[rows, cols].copy()
        counts = np.bincount(rows, minlength=self.out_dim)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.nonempty = np.nonzero(counts)[0]
        self.starts = self.indptr[self.nonempty]
        self.b = b.copy()

    @property
    def nnz(self):
        return int(self.values.size)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], self.out_dim))
        out[:] = self.b
        if self.nnz:
            prod = X[:, self.indices] * self.values           # B * nnz multiplies
            out[:, self.nonempty] += np.add.reduceat(prod, self.starts, axis=1)  # B * nnz adds
        return out


class SparseMLP:
    """Compacted + CSR version of a (pruned) MLP. Exact same function."""

    def __init__(self, model):
        self.activation = model.activation
        Ws = [l.W.data * l.mask for l in model.layers]
        Ms = [l.mask.copy() for l in model.layers]
        bs = [l.b.data.copy() for l in model.layers]
        self.removed_units = [0] * (len(Ws) - 1)
        changed = True
        while changed:
            changed = False
            for k in range(len(Ws) - 1):  # hidden layer k = output of Ws[k]
                out_deg = Ms[k + 1].sum(axis=1)
                in_deg = Ms[k].sum(axis=0)
                dead_out = out_deg == 0
                dead_in = (in_deg == 0) & ~dead_out
                if dead_in.any():
                    const = _act(bs[k][dead_in], self.activation)
                    bs[k + 1] = bs[k + 1] + const @ Ws[k + 1][dead_in]
                drop = dead_out | dead_in
                if drop.any():
                    keep = ~drop
                    Ws[k], Ms[k], bs[k] = Ws[k][:, keep], Ms[k][:, keep], bs[k][keep]
                    Ws[k + 1], Ms[k + 1] = Ws[k + 1][keep], Ms[k + 1][keep]
                    self.removed_units[k] += int(drop.sum())
                    changed = True
            # inputs feeding nothing can also be skipped (gather never reads them)
        self.layers = [CSRLinear(W, M, b) for W, M, b in zip(Ws, Ms, bs)]
        # compacted dense weights: dead units removed, remaining zeros kept
        self.compact = [(W * M, b) for W, M, b in zip(Ws, Ms, bs)]
        self.shapes = [(L.in_dim, L.out_dim) for L in self.layers]

    def predict_logits(self, X):
        h = X
        for i, L in enumerate(self.layers):
            h = L.forward(h)
            if i < len(self.layers) - 1:
                h = _act(h, self.activation)
        return h

    def predict_logits_compact_dense(self, X):
        """BLAS on the structurally smaller matrices (no CSR)."""
        h = X
        for i, (W, b) in enumerate(self.compact):
            h = h @ W + b
            if i < len(self.compact) - 1:
                h = _act(h, self.activation)
        return h

    @property
    def nnz(self):
        return sum(L.nnz for L in self.layers)


def flops_dense(model, batch: int) -> int:
    """Multiply-add FLOPs of the original dense architecture (what an unpruned
    model, or a masked model run densely, actually executes)."""
    return sum(2 * batch * l.mask.size + batch * l.mask.shape[1] for l in model.layers)


def flops_sparse(smodel: SparseMLP, batch: int) -> int:
    return sum(2 * batch * L.nnz + batch * L.out_dim for L in smodel.layers)
