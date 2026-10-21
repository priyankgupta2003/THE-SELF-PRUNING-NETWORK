"""Composable layers built only on the autodiff engine.

The masked Linear layer is the heart of "honest masking":

    W_eff = W (*) M        (M is a constant 0/1 tensor, no gradient)
    z     = x @ W_eff
    y     = z + b

Because the mask enters the graph as an ordinary multiplication, the engine
gives us two *different* gradients for free, and we deliberately keep both:

  * dL/dW     = (x^T dL/dz) (*) M   -> exactly 0 for pruned entries.
                                       This is what the optimizer sees.
  * dL/dW_eff =  x^T dL/dz          -> dense, nonzero for pruned entries.
                                       "How much would the loss move if this
                                       connection existed?" -- used for
                                       regrowth decisions, never for updates.

In addition the stored parameter value is kept at exactly 0.0 wherever M = 0
(the pruner writes zeros, the optimizer re-asserts them), so the forward
pass contributes a true zero even if someone bypasses the mask.
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Linear:
    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator, init: str = "he"):
        if init == "he":
            # Kaiming/He: Var(w) = 2/fan_in keeps the variance of ReLU
            # pre-activations constant across depth (ReLU zeroes half the mass).
            std = np.sqrt(2.0 / fan_in)
        elif init == "xavier":
            # Glorot: Var(w) = 2/(fan_in+fan_out); used on the logit layer so
            # initial logits are small and the initial loss is ~ln(C).
            std = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            raise ValueError(init)
        self.W = Tensor(rng.normal(0.0, std, size=(fan_in, fan_out)))
        self.b = Tensor(np.zeros(fan_out))
        self.mask = np.ones((fan_in, fan_out), dtype=bool)  # mutated in place by the pruner
        # filled on every forward pass (needed by the pruner's statistics)
        self.x: Tensor | None = None
        self.z: Tensor | None = None
        self.w_eff: Tensor | None = None

    def __call__(self, x: Tensor) -> Tensor:
        m = Tensor(self.mask.astype(np.float64), requires_grad=False)
        self.w_eff = self.W * m
        self.x = x
        self.z = x @ self.w_eff
        return self.z + self.b

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def n_weights(self):
        return self.mask.size

    @property
    def n_active(self):
        return int(self.mask.sum())


class MLP:
    def __init__(self, sizes, rng, activation: str = "relu"):
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            self.layers.append(Linear(a, b, rng, init="xavier" if last else
                                      ("he" if activation == "relu" else "xavier")))
        self.activation = activation

    def __call__(self, x) -> Tensor:
        h = x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = h.relu() if self.activation == "relu" else h.tanh()
        return h

    @property
    def params(self):
        return [p for l in self.layers for p in l.params]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def sparsity(self) -> float:
        tot = sum(l.n_weights for l in self.layers)
        act = sum(l.n_active for l in self.layers)
        return 1.0 - act / tot

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Inference with the training-path (dense masked) forward. NumPy only."""
        h = X
        for i, l in enumerate(self.layers):
            h = h @ (l.W.data * l.mask) + l.b.data
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0) if self.activation == "relu" else np.tanh(h)
        return h
