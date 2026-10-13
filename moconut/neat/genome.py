"""NEAT genomes, and how they become patches."""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from ..core import Patch
from ..neuron import Neuron
from .config import NEATConfig

__all__ = ["NodeGene", "ConnGene", "Genome", "InnovationTracker", "creates_cycle"]


@dataclass
class NodeGene:
    id: int
    kind: str                 # "input", "output" or "hidden"
    bias: float = 0.0
    response: float = 1.0
    activation: str = "steep_sigmoid"
    aggregation: str = "sum"


@dataclass
class ConnGene:
    innovation: int
    src: int
    dst: int
    weight: float
    enabled: bool = True


class InnovationTracker:
    """Hands out node ids and innovation numbers, so equal structure gets equal numbers.

    A connection between the same two nodes always gets the same innovation
    number. Splitting the same connection in the same generation gives the
    same new node, as in the original NEAT.
    """

    def __init__(self, next_node_id: int):
        self.next_node_id = next_node_id
        self._conns: dict[tuple[int, int], int] = {}
        self._splits: dict[int, int] = {}

    def conn(self, src: int, dst: int) -> int:
        return self._conns.setdefault((src, dst), len(self._conns))

    def split(self, innovation: int) -> int:
        if innovation not in self._splits:
            self._splits[innovation] = self.next_node_id
            self.next_node_id += 1
        return self._splits[innovation]

    def new_generation(self) -> None:
        self._splits.clear()


def creates_cycle(edges, src: int, dst: int) -> bool:
    """Would adding src -> dst to these (src, dst) edges close a cycle?"""
    if src == dst:
        return True
    successors: dict[int, list[int]] = {}
    for a, b in edges:
        successors.setdefault(a, []).append(b)
    seen, stack = {dst}, [dst]
    while stack:
        for nxt in successors.get(stack.pop(), ()):
            if nxt == src:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


class Genome:
    """Node genes and connection genes (keyed by innovation number)."""

    def __init__(self, nodes: dict[int, NodeGene], conns: dict[int, ConnGene]):
        self.nodes = nodes
        self.conns = conns
        self.fitness: float | None = None

    @classmethod
    def minimal(cls, config: NEATConfig, tracker: InnovationTracker, rng: random.Random) -> "Genome":
        """Every input wired straight to every output, with random weights."""
        nodes = {i: NodeGene(i, "input") for i in range(config.num_inputs)}
        for j in range(config.num_outputs):
            nid = config.num_inputs + j
            nodes[nid] = NodeGene(nid, "output", rng.gauss(0, config.weight_init_std), 1.0, config.activation, config.aggregation)
        conns = {}
        for i in range(config.num_inputs):
            for j in range(config.num_outputs):
                innovation = tracker.conn(i, config.num_inputs + j)
                conns[innovation] = ConnGene(innovation, i, config.num_inputs + j, rng.gauss(0, config.weight_init_std))
        return cls(nodes, conns)

    def copy(self) -> "Genome":
        return Genome({k: replace(n) for k, n in self.nodes.items()}, {k: replace(c) for k, c in self.conns.items()})

    def ids(self, kind: str) -> list[int]:
        return sorted(n for n, g in self.nodes.items() if g.kind == kind)

    def size(self) -> tuple[int, int]:
        """(hidden nodes, enabled connections)."""
        return len(self.ids("hidden")), sum(c.enabled for c in self.conns.values())

    ################################################################
    # Mutation
    ################################################################
    def mutate(self, config: NEATConfig, tracker: InnovationTracker, rng: random.Random) -> None:
        if rng.random() < config.add_node_rate:
            self.add_node(config, tracker, rng)
        if rng.random() < config.add_conn_rate:
            self.add_connection(config, tracker, rng)
        if rng.random() < config.weight_mutate_rate:
            self.mutate_weights(config, rng)

    def add_node(self, config: NEATConfig, tracker: InnovationTracker, rng: random.Random) -> int | None:
        """Split an enabled connection a -> b into a -> new (weight 1) and new -> b (old weight)."""
        enabled = [c for _, c in sorted(self.conns.items()) if c.enabled]
        if not enabled:
            return None
        old = rng.choice(enabled)
        nid = tracker.split(old.innovation)
        if nid in self.nodes:
            return None
        old.enabled = False
        self.nodes[nid] = NodeGene(nid, "hidden", 0.0, 1.0, config.activation, config.aggregation)
        for src, dst, weight in ((old.src, nid, 1.0), (nid, old.dst, old.weight)):
            innovation = tracker.conn(src, dst)
            self.conns[innovation] = ConnGene(innovation, src, dst, weight)
        return nid

    def add_connection(self, config: NEATConfig, tracker: InnovationTracker, rng: random.Random) -> ConnGene | None:
        """Connect two unconnected nodes. In feed-forward mode, never close a cycle."""
        sources = sorted(self.nodes)
        targets = [n for n in sources if self.nodes[n].kind != "input"]
        existing = {(c.src, c.dst) for c in self.conns.values()}
        enabled = [(c.src, c.dst) for c in self.conns.values() if c.enabled]
        for _ in range(config.add_conn_attempts):
            src, dst = rng.choice(sources), rng.choice(targets)
            if (src, dst) in existing:
                continue
            if config.feed_forward and creates_cycle(enabled, src, dst):
                continue
            innovation = tracker.conn(src, dst)
            gene = ConnGene(innovation, src, dst, rng.gauss(0, config.weight_init_std))
            self.conns[innovation] = gene
            return gene
        return None

    def mutate_weights(self, config: NEATConfig, rng: random.Random) -> None:
        """Perturb (or occasionally replace) every weight and bias."""
        def mutated(value: float) -> float:
            if rng.random() < config.weight_replace_rate:
                value = rng.gauss(0, config.weight_init_std)
            else:
                value += rng.gauss(0, config.weight_perturb_power)
            return max(-config.weight_limit, min(config.weight_limit, value))

        for c in self.conns.values():
            c.weight = mutated(c.weight)
        for n in self.nodes.values():
            if n.kind != "input":
                n.bias = mutated(n.bias)

    ################################################################
    # Crossover and speciation
    ################################################################
    @staticmethod
    def crossover(fitter: "Genome", other: "Genome", config: NEATConfig, rng: random.Random) -> "Genome":
        """Matching genes come from either parent; disjoint and excess genes from the fitter one."""
        conns = {}
        for innovation, mine in fitter.conns.items():
            theirs = other.conns.get(innovation)
            gene = replace(mine if theirs is None or rng.random() < 0.5 else theirs)
            if theirs is not None and not (mine.enabled and theirs.enabled):
                gene.enabled = rng.random() >= config.disable_inherit_rate
            conns[innovation] = gene
        nodes = {}
        for nid, mine in fitter.nodes.items():
            theirs = other.nodes.get(nid)
            nodes[nid] = replace(mine if theirs is None or rng.random() < 0.5 else theirs)
        child = Genome(nodes, conns)
        child.repair(config)
        return child

    def repair(self, config: NEATConfig) -> None:
        """In feed-forward mode, disable connections that close a cycle (crossover can combine them)."""
        if not config.feed_forward:
            return
        kept: list[tuple[int, int]] = []
        for _, c in sorted(self.conns.items()):
            if not c.enabled:
                continue
            if creates_cycle(kept, c.src, c.dst):
                c.enabled = False
            else:
                kept.append((c.src, c.dst))

    def distance(self, other: "Genome", config: NEATConfig) -> float:
        """NEAT's compatibility distance."""
        mine, theirs = set(self.conns), set(other.conns)
        cutoff = min(max(mine, default=-1), max(theirs, default=-1))
        unmatched = mine ^ theirs
        excess = sum(1 for i in unmatched if i > cutoff)
        disjoint = len(unmatched) - excess
        matching = mine & theirs
        weight = sum(abs(self.conns[i].weight - other.conns[i].weight) for i in matching) / len(matching) if matching else 0.0
        n = max(len(mine), len(theirs))
        n = 1 if n < 20 else n
        return (config.excess_coefficient * excess + config.disjoint_coefficient * disjoint) / n + config.weight_coefficient * weight

    ################################################################
    # Phenotype
    ################################################################
    def to_patch(self) -> Patch:
        """Build the network as a Patch.

        Inputs become inlets x0, x1, ...; outputs become outlets y0, y1, ...;
        every other node is a Neuron object named n<id>. Connections are added
        in innovation order, and one that would close a delay-free cycle gets
        delay=1, so cycles read the previous step's value.
        """
        inputs, outputs = self.ids("input"), self.ids("output")
        patch = Patch(inlets=[f"x{i}" for i in range(len(inputs))], outlets=[f"y{j}" for j in range(len(outputs))])
        names = {nid: f"x{i}" for i, nid in enumerate(inputs)}

        enabled = [c for _, c in sorted(self.conns.items()) if c.enabled]
        used = {c.src for c in enabled} | {c.dst for c in enabled} | set(outputs)
        for nid in sorted(used - set(inputs)):
            g = self.nodes[nid]
            names[nid] = f"n{nid}"
            patch.put(names[nid], Neuron(g.activation, g.aggregation, g.bias, g.response))

        instant: list[tuple[int, int]] = []
        for c in enabled:
            delay = 1 if creates_cycle(instant, c.src, c.dst) else 0
            if not delay:
                instant.append((c.src, c.dst))
            patch.connect(names[c.src], names[c.dst], delay=delay, weight=c.weight)
        for j, nid in enumerate(outputs):
            patch.connect(names[nid], f"y{j}")
        return patch

    def __repr__(self) -> str:
        hidden, enabled = self.size()
        return f"Genome(hidden={hidden}, connections={enabled}, fitness={self.fitness})"
