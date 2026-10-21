import numpy as np

from tinyprune.nn import MLP
from tinyprune.prune import cubic_sparsity
from tinyprune.sparse import SparseMLP, flops_dense, flops_sparse


def _pruned_model(seed=0, keep=0.1):
    rng = np.random.default_rng(seed)
    m = MLP([20, 30, 15, 4], rng)
    for l in m.layers:
        l.b.data[:] = rng.normal(size=l.b.data.shape)
        l.mask[...] = rng.random(l.mask.shape) < keep
        l.W.data[~l.mask] = 0.0
    # force structural cases: unit 0 of hidden 1 has no outgoing, unit 1 no incoming
    m.layers[1].mask[0, :] = False; m.layers[1].W.data[0, :] = 0
    m.layers[0].mask[:, 1] = False; m.layers[0].W.data[:, 1] = 0
    m.layers[1].mask[1, 3] = True; m.layers[1].W.data[1, 3] = 0.5
    return m


def test_sparse_path_matches_dense_masked_forward():
    for seed, keep in [(0, 0.1), (1, 0.3), (2, 0.02), (3, 1.0)]:
        m = _pruned_model(seed, keep)
        X = np.random.default_rng(9).normal(size=(17, 20))
        s = SparseMLP(m)
        np.testing.assert_allclose(s.predict_logits(X), m.predict_logits(X), rtol=1e-12, atol=1e-12)


def test_compaction_removes_units_and_flops_shrink():
    m = _pruned_model(0, 0.1)
    s = SparseMLP(m)
    assert s.removed_units[0] >= 2
    assert s.nnz <= sum(l.n_active for l in m.layers)
    assert flops_sparse(s, 10) < 0.25 * flops_dense(m, 10)


def test_cubic_schedule_endpoints_and_monotone():
    vals = [cubic_sparsity(t, 0.9, 100, 700) for t in range(0, 900)]
    assert vals[0] == 0 and vals[99] == 0 and vals[-1] == 0.9
    assert all(b >= a for a, b in zip(vals, vals[1:]))
