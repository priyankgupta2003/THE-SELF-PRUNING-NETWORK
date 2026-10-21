"""Finite-difference gradient checking."""
import numpy as np

from .tensor import Tensor


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences of scalar f() w.r.t. every entry of x (mutated in place)."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f()
        x[i] = old - eps
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def check_gradients(build_loss, tensors, eps=1e-6, tol=1e-6):
    """build_loss() -> scalar Tensor using ``tensors``. Returns max relative error.

    Relative error = |a - n| / max(1e-8, |a| + |n|), the standard metric that
    is meaningful both for large and tiny gradients.
    """
    for t in tensors:
        t.zero_grad()
    loss = build_loss()
    loss.backward()
    worst = 0.0
    for t in tensors:
        analytic = t.grad if t.grad is not None else np.zeros_like(t.data)
        numeric = numerical_grad(lambda: float(build_loss().data), t.data, eps)
        denom = np.maximum(1e-8, np.abs(analytic) + np.abs(numeric))
        worst = max(worst, float(np.max(np.abs(analytic - numeric) / denom)))
    if worst > tol:
        raise AssertionError(f"gradient check failed: max rel err {worst:.3e} > {tol}")
    return worst


def as_param(a) -> Tensor:
    return Tensor(np.array(a, dtype=np.float64), requires_grad=True)
