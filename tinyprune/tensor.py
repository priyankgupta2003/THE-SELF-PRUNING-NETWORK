"""A minimal reverse-mode automatic differentiation engine on top of NumPy.

Design notes
------------
* Every Tensor wraps a float64 NumPy array, remembers the Tensors it was
  computed from (``_parents``) and a closure (``_backward``) that pushes its
  own gradient into those parents.
* ``backward()`` builds a topological order of the graph *iteratively* (no
  recursion limit on deep graphs), seeds d(out)/d(out) = 1 and walks the order
  in reverse. Gradients are *accumulated* (+=), so a tensor used twice receives
  the sum of both contributions.
* Broadcasting: NumPy silently broadcasts in the forward pass, so the backward
  pass must "un-broadcast" -- sum the incoming gradient over every axis that
  was expanded -- before accumulating into a parent. ``_unbroadcast`` does this
  and is the reason (N, D) + (D,) backpropagates correctly.
* The graph is rebuilt on every forward pass (define-by-run). That is what
  makes masking safe: a mask is just another constant input to a ``*`` node,
  so when the mask changes between steps the next forward pass simply builds
  a graph with the new constant. Nothing in the engine caches structure.
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` down to ``shape`` (inverse of NumPy broadcasting)."""
    if grad.shape == shape:
        return grad
    # 1) leading axes that were prepended by broadcasting
    n_extra = grad.ndim - len(shape)
    if n_extra > 0:
        grad = grad.sum(axis=tuple(range(n_extra)))
    # 2) axes that had size 1 in the original and were stretched
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __array_priority__ = 1000  # make ndarray (op) Tensor dispatch to Tensor

    def __init__(self, data, parents=(), op: str = "", requires_grad: bool = True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._parents = tuple(parents)
        self._backward = lambda: None
        self.op = op

    # ------------------------------------------------------------------ utils
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op='{self.op}')"

    def zero_grad(self):
        self.grad = None

    def _accum(self, g: np.ndarray):
        """Accumulate an upstream gradient into this tensor (handles broadcasting)."""
        if not self.requires_grad:
            return
        g = _unbroadcast(g, self.data.shape)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g

    @staticmethod
    def _wrap(x) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    def _make(self, data, parents, op):
        rg = any(p.requires_grad for p in parents)
        return Tensor(data, parents=parents, op=op, requires_grad=rg)

    # ------------------------------------------------------ elementwise binary
    def __add__(self, other):
        other = self._wrap(other)
        out = self._make(self.data + other.data, (self, other), "add")

        def _backward():
            self._accum(out.grad)
            other._accum(out.grad)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = self._wrap(other)
        out = self._make(self.data - other.data, (self, other), "sub")

        def _backward():
            self._accum(out.grad)
            other._accum(-out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = self._make(self.data * other.data, (self, other), "mul")

        def _backward():
            self._accum(out.grad * other.data)
            other._accum(out.grad * self.data)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = self._wrap(other)
        out = self._make(self.data / other.data, (self, other), "div")

        def _backward():
            self._accum(out.grad / other.data)
            other._accum(-out.grad * self.data / (other.data ** 2))
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self._wrap(other) + self

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __rmul__(self, other):
        return self._wrap(other) * self

    def __rtruediv__(self, other):
        return self._wrap(other) / self

    # ---------------------------------------------------------------- matmul
    def __matmul__(self, other):
        other = self._wrap(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("matmul is implemented for 2-D operands only")
        out = self._make(self.data @ other.data, (self, other), "matmul")

        def _backward():
            # C = A B  ->  dA = dC B^T,  dB = A^T dC
            self._accum(out.grad @ other.data.T)
            other._accum(self.data.T @ out.grad)
        out._backward = _backward
        return out

    # ------------------------------------------------------------ reductions
    def sum(self, axis=None, keepdims=False):
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.data.shape))
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod(
            [self.data.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # -------------------------------------------------------- non-linearities
    def relu(self):
        out = self._make(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            # subgradient 0 at exactly 0 (the convention every framework uses)
            self._accum(out.grad * (self.data > 0))
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = self._make(t, (self,), "tanh")

        def _backward():
            self._accum(out.grad * (1.0 - t ** 2))
        out._backward = _backward
        return out

    def sigmoid(self):
        # numerically stable in both tails
        x = self.data
        s = np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                     np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))
        out = self._make(s, (self,), "sigmoid")

        def _backward():
            self._accum(out.grad * s * (1.0 - s))
        out._backward = _backward
        return out

    # -------------------------------------------------------------- backward
    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() without an argument needs a scalar output")
            grad = np.ones_like(self.data)

        # iterative post-order DFS -> topological order
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))

        # intermediate grads are fresh for this pass; leaves accumulate across
        # calls until the optimizer zeroes them (standard semantics)
        for node in topo:
            if node._parents:
                node.grad = None
        self.grad = np.asarray(grad, dtype=np.float64).copy() if self._parents else (
            (self.grad if self.grad is not None else 0) + grad)
        for node in reversed(topo):
            if node._parents and node.grad is not None:
                node._backward()


def softmax_cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean cross-entropy of softmax(logits) against integer labels.

    Fused for numerical stability: we never form softmax then log it.
        log p_y = z_y - logsumexp(z),  logsumexp(z) = m + log sum exp(z - m)
    with m = max(z) so every exp() argument is <= 0 (no overflow), and the
    sum is >= 1 (no log(0)). The gradient of the mean loss is the classic
        dL/dz = (softmax(z) - onehot(y)) / N.
    """
    z = logits.data
    n = z.shape[0]
    labels = np.asarray(labels, dtype=np.int64)
    m = z.max(axis=1, keepdims=True)
    shifted = z - m
    lse = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - lse
    loss = -log_probs[np.arange(n), labels].mean()
    out = Tensor(loss, parents=(logits,), op="softmax_ce",
                 requires_grad=logits.requires_grad)

    def _backward():
        p = np.exp(log_probs)
        p[np.arange(n), labels] -= 1.0
        logits._accum(out.grad * p / n)
    out._backward = _backward
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    """Plain NumPy stable softmax (inference only)."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
