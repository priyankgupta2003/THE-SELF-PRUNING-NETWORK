"""Part 4.2: does pruning actually reduce work?

For the seed-0 'fisher' models at each sparsity we report
  * active weights and CSR non-zeros after structural compaction
  * analytic FLOPs: dense architecture vs our sparse path
  * bytes to store the weights (dense float64 vs CSR values+indices+indptr)
  * wall-clock of three forward paths, median of many repeats:
      dense-masked : X @ (W*M) with NumPy BLAS, mask pre-applied -- the
                     "multiply by zeros" shortcut; same work as unpruned
      CSR (ours)   : compaction + CSR gather/segment-sum, written by us
      compact      : BLAS on the structurally smaller matrices left after
                     deleting dead hidden units (zeros inside remain)
All three paths are checked to give identical logits first.
"""
import csv
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tinyprune.data import load
from tinyprune.nn import MLP
from tinyprune.sparse import SparseMLP, flops_dense, flops_sparse

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def load_model(path):
    z = np.load(path)
    m = MLP([64, 128, 64, 10], np.random.default_rng(0))
    for i, l in enumerate(m.layers):
        l.W.data[...] = z[f"W{i}"]; l.mask[...] = z[f"M{i}"]; l.b.data[...] = z[f"b{i}"]
    return m


def bench(fn, X, min_time=0.3):
    fn(X)
    n, t0 = 0, time.perf_counter()
    times = []
    while time.perf_counter() - t0 < min_time or n < 20:
        a = time.perf_counter(); fn(X); times.append(time.perf_counter() - a); n += 1
    return float(np.median(times))


def main(method="fisher"):
    _, _, Xte, _ = load("digits", 0)
    Xbig = np.tile(Xte, (12, 1))  # 4308 rows, a throughput-style batch
    rows = []
    paths = sorted(glob.glob(os.path.join(OUT, "models", f"{method}_*.npz")),
                   key=lambda p: float(p.rsplit("_", 1)[1][:-4]))
    for path in [os.path.join(OUT, "models", "dense_0.0.npz")] + paths:
        m = load_model(path)
        s = SparseMLP(m)
        Wd = [(l.W.data * l.mask, l.b.data) for l in m.layers]  # mask applied ONCE, not per call

        def dense_fwd(X, Wd=Wd):
            h = X
            for i, (W, b) in enumerate(Wd):
                h = h @ W + b
                if i < len(Wd) - 1:
                    h = np.maximum(h, 0)
            return h
        for X in (Xte[:1], Xte, Xbig):
            ref = dense_fwd(X)
            assert np.allclose(s.predict_logits(X), ref, atol=1e-10)
            assert np.allclose(s.predict_logits_compact_dense(X), ref, atol=1e-10)
        dense_bytes = sum(l.W.data.nbytes for l in m.layers)
        csr_bytes = sum(L.values.nbytes + L.indices.nbytes + L.indptr.nbytes for L in s.layers)
        row = dict(model=os.path.basename(path)[:-4], sparsity=round(m.sparsity(), 4),
                   active_weights=sum(l.n_active for l in m.layers), csr_nnz=s.nnz,
                   compacted_shapes=" ".join(f"{a}x{b}" for a, b in s.shapes),
                   flops_dense_per_sample=flops_dense(m, 1), flops_sparse_per_sample=flops_sparse(s, 1),
                   flop_reduction=round(flops_dense(m, 1) / flops_sparse(s, 1), 2),
                   weight_bytes_dense=dense_bytes, weight_bytes_csr=csr_bytes)
        for name, X in [("b1", Xte[:1]), ("b359", Xte), ("b4308", Xbig)]:
            td = bench(dense_fwd, X)
            ts = bench(s.predict_logits, X)
            tc = bench(s.predict_logits_compact_dense, X)
            row[f"us_dense_{name}"] = round(td * 1e6, 1)
            row[f"us_csr_{name}"] = round(ts * 1e6, 1)
            row[f"us_compact_{name}"] = round(tc * 1e6, 1)
            row[f"speedup_csr_{name}"] = round(td / ts, 2)
            row[f"speedup_compact_{name}"] = round(td / tc, 2)
        rows.append(row)
        print({k: row[k] for k in ("model", "sparsity", "csr_nnz", "compacted_shapes", "flop_reduction",
                                   "speedup_csr_b1", "speedup_csr_b359", "speedup_compact_b1",
                                   "speedup_compact_b359", "speedup_compact_b4308")})
    with open(os.path.join(OUT, "cost.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
