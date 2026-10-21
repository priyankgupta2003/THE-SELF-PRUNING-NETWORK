"""The Part 1 correctness requirement: masked weights, gradients and optimizer state.

What "the gradient of a masked weight" means here
-------------------------------------------------
There are two legitimate quantities and the tests pin down both:
  * dL/dW at a pruned entry  == 0 exactly   (the parameter has no effect on L,
    because the forward pass uses W * M). This is what the optimizer consumes.
  * dL/dW_eff at that entry  == x^T delta    (the sensitivity of L to a
    connection that is currently absent). Used only for regrowth scoring.
"""
import numpy as np

from tinyprune.gradcheck import check_gradients, numerical_grad
from tinyprune.nn import MLP, Linear
from tinyprune.optim import Adam, NaiveAdam
from tinyprune.prune import Pruner
from tinyprune.tensor import Tensor, softmax_cross_entropy
from tinyprune.train import Config, train


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    layer = Linear(6, 4, rng)
    layer.b.data[:] = rng.normal(size=4)
    mask = rng.random((6, 4)) > 0.5
    layer.mask[...] = mask
    layer.W.data[~mask] = 0.0
    X = rng.normal(size=(5, 6))
    y = np.array([0, 1, 2, 3, 1])
    return layer, X, y


def test_masked_weight_contributes_exactly_zero_to_forward():
    layer, X, y = _setup()
    out1 = layer(Tensor(X, requires_grad=False)).data.copy()
    # even if something corrupts the stored value, the mask removes it
    layer.W.data[~layer.mask] = 123.0
    out2 = layer(Tensor(X, requires_grad=False)).data
    np.testing.assert_array_equal(out1, out2)
    # and it equals a matmul that never had those connections
    W_removed = np.where(layer.mask, layer.W.data, 0.0)
    np.testing.assert_allclose(out1, X @ W_removed + layer.b.data, rtol=0, atol=1e-12)


def test_param_grad_is_zero_and_effective_grad_is_correct_for_masked_entries():
    layer, X, y = _setup()
    loss = softmax_cross_entropy(layer(Tensor(X, requires_grad=False)), y)
    loss.backward()
    # (a) the parameter gradient at pruned positions is exactly zero
    assert np.all(layer.W.grad[~layer.mask] == 0.0)
    # (b) dL/dW_eff is dense and matches finite differences w.r.t. the
    #     effective weight -- including at positions that are pruned
    We = layer.w_eff.data.copy()
    f = lambda: float(softmax_cross_entropy(Tensor(X @ We + layer.b.data), y).data)
    num = numerical_grad(f, We)
    np.testing.assert_allclose(layer.w_eff.grad, num, rtol=1e-6, atol=1e-9)
    assert np.abs(layer.w_eff.grad[~layer.mask]).max() > 1e-4  # genuinely informative


def test_gradcheck_through_masked_network():
    rng = np.random.default_rng(3)
    model = MLP([5, 7, 3], rng)
    for l in model.layers:
        l.mask[...] = rng.random(l.mask.shape) > 0.4
        l.W.data[~l.mask] = 0.0
    X = rng.normal(size=(4, 5)); y = np.array([0, 2, 1, 2])
    # nudge away from ReLU kinks
    check_gradients(lambda: softmax_cross_entropy(model(Tensor(X, requires_grad=False)), y),
                    model.params, tol=1e-5)


def _train_steps(opt, model, X, y, n):
    for _ in range(n):
        model.zero_grad()
        softmax_cross_entropy(model(Tensor(X, requires_grad=False)), y).backward()
        opt.step()


def test_no_ghost_momentum_after_pruning_and_naive_adam_does_drift():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 8)); y = rng.integers(0, 3, 32)
    results = {}
    for name, cls in [("correct", Adam), ("naive", NaiveAdam)]:
        model = MLP([8, 16, 3], np.random.default_rng(1))
        params = [(l.W, l.mask) for l in model.layers] + [(l.b, None) for l in model.layers]
        opt = cls(params, lr=1e-2)
        _train_steps(opt, model, X, y, 30)                     # build up momentum
        l0 = model.layers[0]
        prune = np.abs(l0.W.data) < np.quantile(np.abs(l0.W.data), 0.5)
        l0.mask[prune] = False
        l0.W.data[prune] = 0.0
        if name == "correct":
            opt.reset_entries(l0.W, prune)
        _train_steps(opt, model, X, y, 20)
        results[name] = np.abs(l0.W.data[prune]).max()
        if name == "correct":
            st = opt.state[id(l0.W)]
            assert np.all(st["m"][prune] == 0) and np.all(st["v"][prune] == 0)
    assert results["correct"] == 0.0                     # real zeros
    assert results["naive"] > 1e-3                       # the bug, demonstrated


def test_revived_weight_follows_fresh_adam_trajectory_exactly():
    """A pruned-then-regrown entry must behave like a brand-new Adam parameter."""
    rng = np.random.default_rng(0)
    p = Tensor(rng.normal(size=5))
    mask = np.ones(5, dtype=bool)
    opt = Adam([(p, mask)], lr=0.01)
    for _ in range(200):                                   # long history -> t = 200
        p.grad = rng.normal(size=5); opt.step()
    mask[0] = False; p.data[0] = 0.0; opt.reset_entries(p, ~mask)   # prune
    for _ in range(30):
        p.grad = rng.normal(size=5) * mask; opt.step()
        assert p.data[0] == 0.0
    mask[0] = True; opt.reset_entries(p, np.eye(5, dtype=bool)[0])  # regrow at w=0
    # reference: a fresh single-parameter Adam starting at 0, same gradients
    q = Tensor(np.zeros(1)); ref = Adam([(q, None)], lr=0.01)
    traj, ref_traj = [], []
    for k in range(25):
        g = rng.normal(size=5)
        p.grad = g; opt.step(); traj.append(p.data[0])
        q.grad = g[:1]; ref.step(); ref_traj.append(q.data[0])
    np.testing.assert_allclose(traj, ref_traj, rtol=0, atol=1e-15)
    # first revived step has magnitude lr (not ~3.16*lr as with a global step counter)
    assert abs(abs(traj[0]) - 0.01) < 1e-8


def test_global_step_bias_correction_would_spike():
    """Documents failure mode 3: reset moments + global t>>1 gives a ~3.16x step."""
    b1, b2, lr, g = 0.9, 0.999, 0.01, 0.7
    t = 5000
    m, v = (1 - b1) * g, (1 - b2) * g * g
    step = lr * (m / (1 - b1 ** t)) / np.sqrt(v / (1 - b2 ** t))
    assert step / lr > 3.0


def test_per_sample_fisher_formula_matches_explicit_loop():
    layer, X, y = _setup(1)
    loss = softmax_cross_entropy(layer(Tensor(X, requires_grad=False)), y)
    loss.backward()
    B = X.shape[0]
    D = layer.z.grad
    fast = B * (X * X).T @ (D * D)
    slow = np.zeros_like(fast)
    for n in range(B):  # per-sample loss gradient wrt W_eff
        ln = softmax_cross_entropy(Tensor(X[n:n + 1] @ layer.w_eff.data + layer.b.data), y[n:n + 1])
        zn = Tensor(X[n:n + 1] @ layer.w_eff.data + layer.b.data)
        ln = softmax_cross_entropy(zn, y[n:n + 1]); ln.backward()
        gn = X[n:n + 1].T @ zn.grad
        slow += gn ** 2 / B
    np.testing.assert_allclose(fast, slow, rtol=1e-10)


def test_training_hits_exact_budget_with_real_zeros_and_regrowth():
    for kw in [dict(criterion="fisher"), dict(criterion="magnitude"),
               dict(criterion="fisher", regrow_frac=0.3), dict(criterion="fisher", allocation="global")]:
        r = train(Config(epochs=8, sparsity=0.9, **kw))
        assert abs(r["sparsity"] - 0.9) < 0.002, (kw, r["sparsity"])
        for l in r["model"].layers:
            assert np.all(l.W.data[~l.mask] == 0.0)
            st = r["pruner"].opt.state[id(l.W)]
            assert np.all(st["m"][~l.mask] == 0) and np.all(st["v"][~l.mask] == 0)
        assert r["test_acc"] > 0.8
    regrown = sum(h[3] for h in r["history"]) if False else None


def test_regrowth_actually_revives_connections():
    r = train(Config(epochs=8, sparsity=0.9, regrow_frac=0.3))
    assert sum(h[3] for h in r["history"]) > 0
