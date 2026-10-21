"""Finite-difference checks for every differentiable op in the engine."""
import numpy as np
import pytest

from tinyprune.gradcheck import as_param, check_gradients
from tinyprune.tensor import Tensor, softmax_cross_entropy

rng = np.random.default_rng(0)


def R(*shape, lo=-1.0, hi=1.0):
    return as_param(rng.uniform(lo, hi, size=shape))


def away_from_zero(*shape):
    """Values with |x| in [0.2, 1] -- keeps ReLU's kink out of the epsilon window."""
    s = rng.choice([-1.0, 1.0], size=shape)
    return as_param(s * rng.uniform(0.2, 1.0, size=shape))


# A random fixed "upstream" weighting turns any tensor into a scalar loss whose
# gradient exercises every entry differently (sum() alone would hide bugs that
# swap entries).
def scal(t):
    w = Tensor(np.random.default_rng(42).normal(size=t.shape), requires_grad=False)
    return (t * w).sum()


CASES = {
    "add": lambda: (lambda a, b: (lambda: scal(a + b), [a, b]))(R(3, 4), R(3, 4)),
    "sub": lambda: (lambda a, b: (lambda: scal(a - b), [a, b]))(R(3, 4), R(3, 4)),
    "mul": lambda: (lambda a, b: (lambda: scal(a * b), [a, b]))(R(3, 4), R(3, 4)),
    "div": lambda: (lambda a, b: (lambda: scal(a / b), [a, b]))(R(3, 4), R(3, 4, lo=0.5, hi=2.0)),
    "neg_rsub_rdiv": lambda: (lambda a: (lambda: scal(2.0 - a + 3.0 / a - (-a)), [a]))(R(3, 4, lo=0.5, hi=2.0)),
    "matmul": lambda: (lambda a, b: (lambda: scal(a @ b), [a, b]))(R(5, 3), R(3, 4)),
    "sum_all": lambda: (lambda a: (lambda: a.sum() * 1.7, [a]))(R(3, 4)),
    "sum_axis0": lambda: (lambda a: (lambda: scal(a.sum(axis=0)), [a]))(R(3, 4)),
    "sum_axis1_keepdims": lambda: (lambda a: (lambda: scal(a.sum(axis=1, keepdims=True)), [a]))(R(3, 4)),
    "mean_all": lambda: (lambda a: (lambda: a.mean() * 2.3, [a]))(R(3, 4)),
    "mean_axis0": lambda: (lambda a: (lambda: scal(a.mean(axis=0)), [a]))(R(3, 4)),
    "relu": lambda: (lambda a: (lambda: scal(a.relu()), [a]))(away_from_zero(3, 4)),
    "tanh": lambda: (lambda a: (lambda: scal(a.tanh()), [a]))(R(3, 4, lo=-3, hi=3)),
    "sigmoid": lambda: (lambda a: (lambda: scal(a.sigmoid()), [a]))(R(3, 4, lo=-6, hi=6)),
    # the required broadcasting case: (N, D) + (D,)
    "broadcast_ND_plus_D": lambda: (lambda a, b: (lambda: scal(a + b), [a, b]))(R(5, 4), R(4)),
    "broadcast_mul_N1": lambda: (lambda a, b: (lambda: scal(a * b), [a, b]))(R(5, 4), R(5, 1)),
    "broadcast_div_D": lambda: (lambda a, b: (lambda: scal(a / b), [a, b]))(R(5, 4), R(4, lo=0.5, hi=2)),
    # a node reused several times -> gradient accumulation must sum paths
    "reuse_accumulation": lambda: (lambda a: (lambda: scal(a * a + a.tanh() * a - a), [a]))(R(3, 4)),
    "softmax_ce": lambda: (lambda z: (lambda: softmax_cross_entropy(z, np.array([0, 2, 1, 2, 0])), [z]))(R(5, 3, lo=-3, hi=3)),
    "mlp_end_to_end": lambda: (lambda x, W1, b1, W2, b2: (
        lambda: softmax_cross_entropy(((x @ W1 + b1).tanh() @ W2 + b2), np.array([1, 0, 2, 1])),
        [x, W1, b1, W2, b2]))(R(4, 3), R(3, 5), R(5), R(5, 3), R(3)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_op_gradient(name):
    build, tensors = CASES[name]()
    err = check_gradients(build, tensors, tol=1e-6)
    assert err < 1e-6


def test_softmax_ce_is_stable_for_huge_logits():
    z = as_param([[1000.0, -1000.0, 0.0], [-5e4, 5e4, 1.0]])
    loss = softmax_cross_entropy(z, np.array([0, 1]))
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(z.grad))
    assert loss.data < 1e-6  # both rows confidently correct


def test_broadcast_grad_shape_and_value():
    # (N, D) + (D,): the bias gradient must be the column sum of upstream grad
    a, b = R(6, 4), R(4)
    (a + b).sum().backward()
    assert b.grad.shape == (4,)
    np.testing.assert_allclose(b.grad, np.full(4, 6.0))


def test_leaf_grads_accumulate_across_backward_calls():
    a = R(3)
    (a * 2.0).sum().backward()
    (a * 3.0).sum().backward()
    np.testing.assert_allclose(a.grad, np.full(3, 5.0))
