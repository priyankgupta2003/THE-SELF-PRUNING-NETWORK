"""Sparsity sweep: every (method, target sparsity, seed). Writes results/sweep_raw.csv.

Seeds change initialisation and mini-batch order; the train/test split is
fixed (split_seed=0) so that method A and method B with the same seed see the
same init, the same batches and the same test set -> paired comparison.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process
import csv
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tinyprune.data import load
from tinyprune.sparse import SparseMLP, flops_dense, flops_sparse
from tinyprune.train import Config, train

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
METHODS = {
    "magnitude": dict(criterion="magnitude", allocation="layerwise"),
    "fisher": dict(criterion="fisher", allocation="layerwise"),
    "fisher+regrow": dict(criterion="fisher", allocation="layerwise", regrow_frac=0.3),
    "magnitude-global": dict(criterion="magnitude", allocation="global"),
    "fisher-global": dict(criterion="fisher", allocation="global"),
}
SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
SEEDS = list(range(int(os.environ.get("N_SEEDS", 10))))
DATA = load("digits", 0)


def run(job):
    method, sp, seed = job
    t0 = time.time()
    kw = METHODS.get(method, {})
    r = train(Config(sparsity=sp, seed=seed, **kw), DATA)
    m = r["model"]
    sm = SparseMLP(m)
    Xte = DATA[2]
    assert np.allclose(sm.predict_logits(Xte), m.predict_logits(Xte), atol=1e-10)
    if seed == 0:
        os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
        np.savez(os.path.join(OUT, "models", f"{method}_{sp}.npz"),
                 **{f"W{i}": l.W.data for i, l in enumerate(m.layers)},
                 **{f"M{i}": l.mask for i, l in enumerate(m.layers)},
                 **{f"b{i}": l.b.data for i, l in enumerate(m.layers)})
    return dict(method=method, target=sp, seed=seed, achieved=round(r["sparsity"], 6),
                test_acc=r["test_acc"], test_loss=r["test_loss"], train_acc=r["train_acc"],
                active_weights=sum(l.n_active for l in m.layers),
                csr_nnz_after_compaction=sm.nnz,
                hidden_units_removed="/".join(map(str, sm.removed_units)),
                flops_dense_b1=flops_dense(m, 1), flops_sparse_b1=flops_sparse(sm, 1),
                layer_active="/".join(str(l.n_active) for l in m.layers),
                regrown_total=sum(h[3] for h in r["history"]), seconds=round(time.time() - t0, 2))


def main():
    jobs = [("dense", 0.0, s) for s in SEEDS]
    jobs += [(m, sp, s) for m in METHODS for sp in SPARSITIES for s in SEEDS]
    with Pool(int(os.environ.get("WORKERS", os.cpu_count()))) as p:
        rows = p.map(run, jobs, chunksize=1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sweep_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} runs")


if __name__ == "__main__":
    main()
