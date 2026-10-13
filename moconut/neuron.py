"""A single neuron as a patch object."""
from __future__ import annotations

import torch
from torch import nn

from .core import GATHER

__all__ = ["Neuron", "ACTIVATIONS", "AGGREGATIONS"]


ACTIVATIONS = {
    "identity": lambda z: z,
    "sigmoid": torch.sigmoid,
    "steep_sigmoid": lambda z: torch.sigmoid(4.9 * z),  # the sigmoid of the original NEAT paper
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sin": torch.sin,
    "gauss": lambda z: torch.exp(-z * z),
    "abs": torch.abs,
    "square": torch.square,
}

AGGREGATIONS = {
    "sum": lambda z: z.sum(-1),
    "product": lambda z: z.prod(-1),
    "max": lambda z: z.amax(-1),
    "min": lambda z: z.amin(-1),
    "mean": lambda z: z.mean(-1),
    "maxabs": lambda z: z.gather(-1, z.abs().argmax(-1, keepdim=True)).squeeze(-1),
}


def _key(source: str) -> str:
    # ParameterDict keys cannot contain '.', which sub-patch outlets use
    return source.replace(".", ":")


class Neuron(nn.Module):
    """activation(bias + response * aggregation(w_i * x_i)).

    Every cable plugged into the neuron gets its own weight, keyed by the
    cable's source, so cables stay plain wires:

        net.connect('x0', 'h', weight=0.5)
    """
    moconut_ports = GATHER

    def __init__(self, activation: str = "sigmoid", aggregation: str = "sum", bias: float = 0.0, response: float = 1.0):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'; choose from {sorted(ACTIVATIONS)}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation '{aggregation}'; choose from {sorted(AGGREGATIONS)}")
        self.activation = activation
        self.aggregation = aggregation
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.response = nn.Parameter(torch.tensor(float(response)))
        self.weights = nn.ParameterDict()

    def moconut_attach(self, source: str, weight: float | None = None) -> None:
        """Called by the patch when a cable is plugged in."""
        key = _key(source)
        if weight is None:
            if key in self.weights:
                return
            weight = torch.randn(()).item()
        self.weights[key] = nn.Parameter(torch.tensor(float(weight), dtype=self.bias.dtype, device=self.bias.device))

    def moconut_detach(self, source: str) -> None:
        """Called by the patch when a cable is unplugged."""
        if _key(source) in self.weights:
            del self.weights[_key(source)]

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if inputs:
            try:
                w = torch.stack([self.weights[_key(s)] for s in inputs])
            except KeyError as e:
                raise KeyError(f"neuron has no weight for the cable from {e.args[0]}") from None
            x = torch.stack(torch.broadcast_tensors(*inputs.values()), dim=-1)
            z = AGGREGATIONS[self.aggregation](x * w)
        else:
            z = torch.zeros_like(self.bias)
        return ACTIVATIONS[self.activation](self.bias + self.response * z)

    def __repr__(self) -> str:
        weights = ", ".join(f"{k}: {w.item():.3g}" for k, w in self.weights.items())
        return f"Neuron({self.activation}, {self.aggregation}, bias={self.bias.item():.3g}, weights={{{weights}}})"
