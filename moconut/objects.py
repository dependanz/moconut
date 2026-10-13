"""Small built-in objects, the patch equivalents of Pd's [+~], [*~], [expr~] and friends."""
from __future__ import annotations

import inspect
import math
from typing import Callable

import torch
from torch import nn

from .core import GATHER, Ports

__all__ = ["Add", "Mul", "Concat", "Op"]


class Add(nn.Module):
    """Sums every cable plugged into its inlet (fan-in sums, as in Pd's signal inlets)."""
    moconut_ports = GATHER

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(inputs.values(), torch.zeros(()))


class Mul(nn.Module):
    """Multiplies every cable plugged into its inlet."""
    moconut_ports = GATHER

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return math.prod(inputs.values(), start=torch.ones(()))


class Concat(nn.Module):
    """Concatenates its numbered inlets ('cat.0', 'cat.1', ...) along ``dim``."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, *xs: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, *xs), dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Op(nn.Module):
    """Wraps any function as an object: Op(torch.tanh), Op(lambda a, b: a - b).

    Inlets come from the function's positional parameters (minus any given as
    keyword arguments); functions without a readable signature take any number.
    """

    def __init__(self, fn: Callable, **kwargs):
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs
        self.moconut_ports = _ports_from_signature(fn, kwargs)

    def forward(self, *xs):
        return self.fn(*xs, **self.kwargs)

    def extra_repr(self) -> str:
        name = getattr(self.fn, "__name__", repr(self.fn))
        return name + "".join(f", {k}={v!r}" for k, v in self.kwargs.items())


def _ports_from_signature(fn: Callable, bound: dict) -> Ports:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return Ports(None, 1)
    names, required = [], 0
    for p in params:
        if p.name in bound:
            continue
        if p.kind is p.VAR_POSITIONAL:
            return Ports(None, required)
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            names.append(p.name)
            if p.default is p.empty:
                required = len(names)
    return Ports(tuple(names), required)
