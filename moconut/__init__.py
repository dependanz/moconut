"""moconut: build PyTorch models by patching objects together, Pure Data style."""
from .core import Cable, Patch, PatchError, Ports, detach_state
from .neuron import ACTIVATIONS, AGGREGATIONS, Neuron
from .objects import Add, Concat, Mul, Op
from . import neat

__version__ = "0.1.0.dev0"
