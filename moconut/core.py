"""The Patch: an nn.Module whose forward pass is wired with cables, Pure Data style."""
from __future__ import annotations

import collections
import difflib
import functools
import inspect
import keyword
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

__all__ = ["Patch", "Cable", "Ports", "PatchError", "detach_state"]


class PatchError(Exception):
    """Raised when a patch is wired incorrectly."""


################################################################
# Ports: what an object exposes to cables
################################################################
@dataclass(frozen=True)
class Ports:
    """The inlets and outlets of an object.

    inlets   -- inlet names, or None for any number of numbered inlets (``*args``)
    required -- how many leading inlets must be connected
    outlets  -- outlet names, or None if unknown (outlets are then addressed by index)
    gather   -- a single inlet that accepts any number of cables; the object is
                called with a dict {source spec: value} instead of positional args
    """
    inlets: tuple[str, ...] | None
    required: int
    outlets: tuple[str, ...] | None = None
    gather: bool = False


GATHER = Ports(("in",), 0, gather=True)


def ports_of(obj: nn.Module) -> Ports:
    """Ports of any object: a Patch, a moconut object, or a plain nn.Module."""
    if isinstance(obj, Patch):
        return Ports(obj.inlets, len(obj.inlets), obj.outlets)
    declared = getattr(obj, "moconut_ports", None)
    if declared is not None:
        return declared
    # Plain nn.Module: one inlet per positional parameter of forward()
    try:
        params = inspect.signature(obj.forward).parameters.values()
    except (TypeError, ValueError):
        return Ports(None, 0)
    names, required, variadic = [], 0, False
    for p in params:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            names.append(p.name)
            if p.default is p.empty:
                required = len(names)
        elif p.kind is p.VAR_POSITIONAL:
            variadic = True
    return Ports(None if variadic else tuple(names), required)


################################################################
# Cables
################################################################
@dataclass(eq=False)
class Cable:
    """A patch cable from an outlet to an inlet.

    Endpoints are (object name, port index); the patch's own inlets and outlets
    use the object name "" and the port name. A cable with delay=n carries the
    value its source produced n steps ago, starting from ``init``.
    """
    src: tuple[str, Any]
    dst: tuple[str, Any]
    src_spec: str
    dst_spec: str
    delay: int = 0
    init: float | torch.Tensor = 0.0

    @property
    def key(self) -> str:
        return f"{self.src_spec}->{self.dst_spec}"

    def __repr__(self) -> str:
        return self.key.replace("->", " -> ") + (f" (delay {self.delay})" if self.delay else "")


@dataclass
class _Plan:
    order: list[str]                         # objects to run each step, in order
    objects: dict[str, nn.Module]
    ports: dict[str, Ports]
    into: dict[str, dict[int, list[Cable]]]  # object -> inlet -> cables
    outlets: dict[str, Cable]                # patch outlet -> its cable
    delayed: list[Cable]                     # delayed cables whose state must be kept


def _check_identifier(name: Any, what: str) -> None:
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
        raise PatchError(f"{what} name {name!r} must be a Python identifier not starting with '_'")


def _did_you_mean(name: str, options) -> str:
    close = difflib.get_close_matches(name, list(options), n=1)
    return f"; did you mean '{close[0]}'?" if close else ""


@functools.lru_cache(maxsize=None)
def _outputs_type(names: tuple[str, ...]):
    return collections.namedtuple("Outputs", names)


def detach_state(state):
    """Detach every tensor in a patch state, for truncated backprop through time."""
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    if isinstance(state, tuple):
        return tuple(detach_state(v) for v in state)
    return state.detach() if torch.is_tensor(state) else state


################################################################
# Patch
################################################################
class Patch(nn.Module):
    """An nn.Module whose forward pass is derived from cables between its objects.

    Objects are ordinary submodules (any nn.Module, including other Patches).
    Cables connect the patch's inlets and the objects' outlets to the objects'
    inlets and the patch's outlets. A cable may carry a delay, which is the only
    way a cycle is allowed: every cycle must pass through a delayed cable.

    One call is one step. Patches with delayed cables are stateful and return
    ``(outputs, state)``; ``step`` always does, ``run`` steps over a sequence.
    """

    def __init__(self, inlets=(), outlets=()):
        super().__init__()
        inlets, outlets = tuple(inlets), tuple(outlets)
        for name in inlets + outlets:
            _check_identifier(name, "port")
            if name == "state":
                raise PatchError("'state' is reserved for the state argument; pick another port name")
        for kind, names in (("inlet", inlets), ("outlet", outlets)):
            if len(set(names)) != len(names):
                raise PatchError(f"duplicate {kind} names in {names}")
        self._inlets = inlets
        self._outlets = outlets
        self._cables: list[Cable] = []
        self._plan: _Plan | None = None

    ################################################################
    # Objects
    ################################################################
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, nn.Module) and "_cables" in self.__dict__:
            _check_identifier(name, "object")
            if name in self._inlets or name in self._outlets:
                raise PatchError(f"object name '{name}' is already a port of this patch")
            if hasattr(type(self), name):
                raise PatchError(f"object name '{name}' clashes with a Patch attribute")
            self._plan = None
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._modules:
            self._plan = None
        super().__delattr__(name)

    def put(self, name: str, obj: nn.Module) -> nn.Module:
        """Place an object in the patch (like Pd's Put > Object). Returns it."""
        if not isinstance(obj, nn.Module):
            raise PatchError(f"objects must be nn.Modules, got {type(obj).__name__}")
        if name in self._modules:
            raise PatchError(f"an object named '{name}' already exists")
        setattr(self, name, obj)
        return obj

    def remove(self, name: str) -> None:
        """Remove an object and every cable attached to it."""
        if name not in self._modules:
            raise PatchError(f"no object named '{name}'" + _did_you_mean(name, self._modules))
        for cable in [c for c in self._cables if name in (c.src[0], c.dst[0])]:
            self._unplug(cable)
        delattr(self, name)

    @property
    def inlets(self) -> tuple[str, ...]:
        return self._inlets

    @property
    def outlets(self) -> tuple[str, ...]:
        return self._outlets

    @property
    def objects(self) -> dict[str, nn.Module]:
        return dict(self._modules)

    @property
    def cables(self) -> tuple[Cable, ...]:
        return tuple(self._cables)

    ################################################################
    # Cables
    ################################################################
    def connect(self, src: str, dst: str, delay: int = 0, init: float | torch.Tensor = 0.0, **params) -> Cable:
        """Connect an outlet to an inlet.

        Addresses are 'name' (port 0, or a patch port) or 'name.port', where port
        is an index or a port name. ``delay=n`` makes the cable carry the value
        from n steps ago, starting at ``init`` (broadcast against the batch).
        Extra keyword arguments are handed to objects that take per-cable
        parameters, e.g. ``weight=`` for a Neuron.
        """
        if not isinstance(delay, int) or delay < 0:
            raise PatchError(f"delay must be a non-negative int, got {delay!r}")
        s, src_spec = self._resolve(src, "src")
        d, dst_spec = self._resolve(dst, "dst")
        cable = Cable(s, d, src_spec, dst_spec, delay, init)

        gather = bool(d[0]) and ports_of(self._modules[d[0]]).gather
        for c in self._cables:
            if c.src == s and c.dst == d:
                raise PatchError(f"cable {src_spec} -> {dst_spec} already exists")
            if c.dst == d and not gather:
                raise PatchError(f"{dst_spec} already has a cable from {c.src_spec}; only gather inlets take several")

        attach = getattr(self._modules[d[0]], "moconut_attach", None) if gather else None
        if attach is not None:
            attach(src_spec, **params)
        elif params:
            raise PatchError(f"{dst_spec} takes no cable parameters, got {sorted(params)}")
        self._cables.append(cable)
        self._plan = None
        return cable

    def chain(self, *specs: str) -> list[Cable]:
        """Connect each address to the next: chain('x', 'a', 'b', 'y')."""
        return [self.connect(a, b) for a, b in zip(specs, specs[1:])]

    def disconnect(self, src: str, dst: str) -> None:
        """Remove the cable from src to dst."""
        s, src_spec = self._resolve(src, "src")
        d, dst_spec = self._resolve(dst, "dst")
        for cable in self._cables:
            if cable.src == s and cable.dst == d:
                self._unplug(cable)
                return
        raise PatchError(f"no cable {src_spec} -> {dst_spec}")

    def _unplug(self, cable: Cable) -> None:
        self._cables.remove(cable)
        node = cable.dst[0]
        if node and node in self._modules:
            detach = getattr(self._modules[node], "moconut_detach", None)
            if detach is not None and ports_of(self._modules[node]).gather:
                detach(cable.src_spec)
        self._plan = None

    def _resolve(self, spec: str, side: str) -> tuple[tuple[str, Any], str]:
        """Turn an address into an endpoint and its canonical spelling."""
        if not isinstance(spec, str):
            raise PatchError(f"cable addresses are strings, got {spec!r}")
        is_src = side == "src"
        name, _, port = spec.partition(".")
        own = self._inlets if is_src else self._outlets
        if not port and name in own:
            return ("", name), name
        if name not in self._modules:
            what = "source" if is_src else "destination"
            raise PatchError(f"unknown {what} '{name}'" + _did_you_mean(name, [*self._modules, *own]))

        obj = self._modules[name]
        ports = ports_of(obj)
        names = ports.outlets if is_src else ports.inlets
        kind = "outlet" if is_src else "inlet"
        if not port:
            index = 0
        elif port.isdigit():
            index = int(port)
        elif names is not None and port in names:
            index = names.index(port)
        else:
            raise PatchError(f"'{name}' has no {kind} '{port}'" + (f"; it has {list(names)}" if names else ""))
        if names is not None and index >= len(names):
            raise PatchError(f"'{name}' has {len(names)} {kind}(s), so {kind} {index} does not exist")

        if isinstance(obj, Patch):
            return (name, index), f"{name}.{names[index]}"
        return (name, index), name if index == 0 else f"{name}.{index}"

    ################################################################
    # Checking and scheduling
    ################################################################
    def check(self) -> "Patch":
        """Validate the wiring now instead of on the first call."""
        self._get_plan()
        return self

    @property
    def stateful(self) -> bool:
        """True if any cable (here or in a sub-patch) carries a delay."""
        plan = self._get_plan()
        return bool(plan.delayed) or any(isinstance(plan.objects[n], Patch) and plan.objects[n].stateful for n in plan.order)

    def _get_plan(self) -> _Plan:
        if self._plan is None:
            self._plan = self._build_plan()
        return self._plan

    def _build_plan(self) -> _Plan:
        objects = dict(self._modules)
        ports = {name: ports_of(obj) for name, obj in objects.items()}
        into: dict[str, dict[int, list[Cable]]] = {name: {} for name in objects}
        outlets: dict[str, Cable] = {}
        for c in self._cables:
            for node in (c.src[0], c.dst[0]):
                if node and node not in objects:
                    raise PatchError(f"cable {c} refers to missing object '{node}'")
            node, port = c.dst
            if node:
                into[node].setdefault(port, []).append(c)
            else:
                outlets[port] = c
        for name in self._outlets:
            if name not in outlets:
                raise PatchError(f"outlet '{name}' is not connected")

        # Only objects that feed an outlet (directly, or through delays) are run
        feeding = {name: [c for cs in into[name].values() for c in cs] for name in objects}
        required: set[str] = set()
        stack = [c.src[0] for c in outlets.values() if c.src[0]]
        while stack:
            name = stack.pop()
            if name not in required:
                required.add(name)
                stack.extend(c.src[0] for c in feeding[name] if c.src[0])

        for name in required:
            p, connected = ports[name], sorted(into[name])
            if p.gather:
                continue
            if connected != list(range(len(connected))):
                gap = min(set(range(connected[-1])) - set(connected))
                raise PatchError(f"'{name}' inlet {gap} is not connected (inlets must be connected in order)")
            if len(connected) < p.required:
                raise PatchError(f"'{name}' needs {p.required} inlet(s) connected, has {len(connected)}")

        # Topological order over delay-free cables
        indegree = {name: 0 for name in objects if name in required}
        successors: dict[str, list[str]] = {name: [] for name in indegree}
        for name in indegree:
            for c in feeding[name]:
                if c.src[0] and not c.delay:
                    indegree[name] += 1
                    successors[c.src[0]].append(name)
        ready = collections.deque(name for name, n in indegree.items() if n == 0)
        order = []
        while ready:
            name = ready.popleft()
            order.append(name)
            for nxt in successors[name]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
        if len(order) < len(indegree):
            loop = self._find_loop({n for n, k in indegree.items() if k > 0}, feeding)
            raise PatchError(f"delay-free cycle: {' -> '.join(loop)}; give one of its cables delay=1")

        delayed = [c for name in order for c in feeding[name] if c.delay]
        delayed += [c for c in outlets.values() if c.delay]
        for name in order:
            if isinstance(objects[name], Patch):
                objects[name].check()
        return _Plan(order, objects, ports, into, outlets, delayed)

    @staticmethod
    def _find_loop(stuck: set[str], feeding: dict[str, list[Cable]]) -> list[str]:
        # Every stuck object has a stuck, delay-free predecessor; walk back until we repeat.
        name = sorted(stuck)[0]
        path: list[str] = []
        while name not in path:
            path.append(name)
            name = next(c.src[0] for c in feeding[name] if c.src[0] in stuck and not c.delay)
        loop = path[path.index(name):][::-1]
        # Start the report at the object placed first, so messages are stable
        position = {n: i for i, n in enumerate(feeding)}
        start = min(range(len(loop)), key=lambda i: position[loop[i]])
        loop = loop[start:] + loop[:start]
        return loop + [loop[0]]

    ################################################################
    # Running
    ################################################################
    def forward(self, *args, state=None, **kwargs):
        """One step. Returns outputs, or (outputs, state) if the patch is stateful."""
        outputs, new_state = self.step(*args, state=state, **kwargs)
        return (outputs, new_state) if self.stateful else outputs

    def step(self, *args, state=None, **kwargs):
        """One step. Always returns (outputs, state)."""
        values, new_state = self._step(self._bind(args, kwargs), state)
        return self._pack(values), new_state

    def run(self, *args, state=None, **kwargs):
        """Step over sequences (time is dim 0). Returns (stacked outputs, state)."""
        inputs = self._bind(args, kwargs)
        lengths = {v.shape[0] for v in inputs.values()}
        if len(lengths) != 1:
            raise PatchError(f"run() needs inputs with one common length along dim 0, got {sorted(lengths)}")
        collected: dict[str, list] = {name: [] for name in self._outlets}
        for t in range(lengths.pop()):
            values, state = self._step({k: v[t] for k, v in inputs.items()}, state)
            for name in self._outlets:
                collected[name].append(values[name])
        stacked = {k: torch.stack(torch.broadcast_tensors(*vs)) for k, vs in collected.items()}
        return self._pack(stacked), state

    def _bind(self, args, kwargs) -> dict[str, Any]:
        if len(args) > len(self._inlets):
            raise PatchError(f"patch has {len(self._inlets)} inlet(s) {list(self._inlets)}, got {len(args)} arguments")
        inputs = dict(zip(self._inlets, args))
        for name, value in kwargs.items():
            if name not in self._inlets:
                raise PatchError(f"unknown inlet '{name}'" + _did_you_mean(name, self._inlets))
            if name in inputs:
                raise PatchError(f"inlet '{name}' given twice")
            inputs[name] = value
        missing = [n for n in self._inlets if n not in inputs]
        if missing:
            raise PatchError(f"missing input for inlet(s) {missing}")
        return inputs

    def _pack(self, values: dict[str, Any]):
        if len(self._outlets) == 1:
            return values[self._outlets[0]]
        if not self._outlets:
            return None
        return _outputs_type(self._outlets)(**values)

    def _step(self, inputs: dict[str, Any], state: dict | None) -> tuple[dict[str, Any], dict]:
        plan = self._get_plan()
        state = state or {}
        new_state: dict = {}
        ref = next((v for v in inputs.values() if torch.is_tensor(v)), None)
        produced: dict[str, tuple] = {}

        def value(src):
            node, port = src
            if not node:
                return inputs[port]
            outs = produced[node]
            if port >= len(outs):
                raise PatchError(f"'{node}' produced {len(outs)} output(s), but a cable reads outlet {port}")
            return outs[port]

        def initial(c: Cable):
            if torch.is_tensor(c.init):
                return c.init if ref is None else c.init.to(device=ref.device, dtype=ref.dtype)
            return torch.tensor(float(c.init)) if ref is None else torch.full((), float(c.init), dtype=ref.dtype, device=ref.device)

        def read(c: Cable):
            if not c.delay:
                return value(c.src)
            held = state.get(c.key)
            return held[0] if held is not None else initial(c)

        for name in plan.order:
            obj, cables = plan.objects[name], plan.into[name]
            if plan.ports[name].gather:
                out = obj({c.src_spec: read(c) for c in cables.get(0, [])})
            else:
                args = [read(cables[i][0]) for i in range(len(cables))]
                if isinstance(obj, Patch):
                    sub_values, sub_state = obj._step(dict(zip(obj.inlets, args)), state.get(name))
                    if sub_state:
                        new_state[name] = sub_state
                    out = tuple(sub_values[k] for k in obj.outlets)
                else:
                    out = obj(*args)
            produced[name] = out if isinstance(out, tuple) else (out,)

        result = {name: read(c) for name, c in plan.outlets.items()}
        for c in plan.delayed:
            held = state.get(c.key) or (initial(c),) * c.delay
            new_state[c.key] = (*held[1:], value(c.src))
        return result, new_state

    def extra_repr(self) -> str:
        lines = [f"inlets={list(self._inlets)}, outlets={list(self._outlets)}"]
        lines += [f"cable: {c!r}" for c in self._cables]
        return "\n".join(lines)
