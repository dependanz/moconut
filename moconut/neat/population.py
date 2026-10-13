"""The NEAT generational loop: speciation, fitness sharing, reproduction."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from .config import NEATConfig
from .genome import Genome, InnovationTracker

__all__ = ["Population", "Species"]


@dataclass(eq=False)
class Species:
    id: int
    representative: Genome
    members: list[Genome] = field(default_factory=list)
    best_fitness: float = -math.inf
    last_improved: int = 0


class Population:
    """A NEAT population.

        pop = Population(NEATConfig(num_inputs=2, num_outputs=1), seed=0)
        best = pop.run(fitness, generations=300, threshold=3.9)

    ``fitness`` takes a Genome and returns a float (higher is better).
    """

    def __init__(self, config: NEATConfig, seed: int | None = None):
        self.config = config
        self.rng = random.Random(seed)
        self.tracker = InnovationTracker(next_node_id=config.num_inputs + config.num_outputs)
        self.genomes = [Genome.minimal(config, self.tracker, self.rng) for _ in range(config.pop_size)]
        self.species: list[Species] = []
        self.generation = 0
        self.best: Genome | None = None
        self.history: list[dict] = []
        self._next_species_id = 0
        self._speciate()

    def run(self, fitness: Callable[[Genome], float], generations: int, threshold: float | None = None,
            verbose: bool = False) -> Genome:
        """Evolve until a genome reaches ``threshold`` or ``generations`` run out. Returns the best genome."""
        for _ in range(generations):
            self.evaluate(fitness)
            if verbose:
                h = self.history[-1]
                print(f"gen {h['generation']:4d}  best {h['best']:.4f}  mean {h['mean']:.4f}  "
                      f"species {h['species']:3d}  champion {h['hidden']} hidden / {h['connections']} connections")
            if threshold is not None and self.best.fitness >= threshold:
                break
            self.reproduce()
        return self.best

    def evaluate(self, fitness: Callable[[Genome], float]) -> None:
        for g in self.genomes:
            g.fitness = float(fitness(g))
        champion = max(self.genomes, key=lambda g: g.fitness)
        if self.best is None or champion.fitness > self.best.fitness:
            self.best = champion.copy()
            self.best.fitness = champion.fitness
        hidden, connections = champion.size()
        self.history.append({
            "generation": self.generation,
            "best": champion.fitness,
            "mean": sum(g.fitness for g in self.genomes) / len(self.genomes),
            "species": len(self.species),
            "hidden": hidden,
            "connections": connections,
        })

    def reproduce(self) -> None:
        """Make the next generation from the evaluated one."""
        cfg, rng = self.config, self.rng
        champion = max(self.genomes, key=lambda g: g.fitness)

        # Stagnant species die out, unless they hold the current champion
        alive = []
        for s in self.species:
            top = max(m.fitness for m in s.members)
            if top > s.best_fitness:
                s.best_fitness, s.last_improved = top, self.generation
            if self.generation - s.last_improved < cfg.stagnation_limit or champion in s.members:
                alive.append(s)

        # Explicit fitness sharing: a species' offspring are proportional to its mean fitness
        floor = min(g.fitness for g in self.genomes)
        shares = [sum(m.fitness - floor for m in s.members) / len(s.members) for s in alive]
        spawn = _allocate(shares, cfg.pop_size)

        self.tracker.new_generation()
        children: list[Genome] = []
        for s, n in zip(alive, spawn):
            if n == 0:
                continue
            ranked = sorted(s.members, key=lambda g: g.fitness, reverse=True)
            if len(ranked) >= cfg.elitism_min_size:
                children.append(ranked[0].copy())
                n -= 1
            parents = ranked[:max(1, math.ceil(cfg.survival_threshold * len(ranked)))]
            for _ in range(n):
                children.append(self._offspring(parents, alive, s))

        for s in alive:
            s.representative = rng.choice(s.members)
            s.members = []
        self.species = alive
        self.genomes = children
        self.generation += 1
        self._speciate()

    def _offspring(self, parents: list[Genome], alive: list[Species], own: Species) -> Genome:
        cfg, rng = self.config, self.rng
        if len(parents) > 1 and rng.random() < cfg.crossover_rate:
            a, b = rng.sample(parents, 2)
            others = [s for s in alive if s is not own]
            if others and rng.random() < cfg.interspecies_rate:
                b = rng.choice(rng.choice(others).members)
            # The fitter parent goes first; on a tie, the smaller one
            a, b = sorted((a, b), key=lambda g: (-g.fitness, len(g.conns)))
            child = Genome.crossover(a, b, cfg, rng)
        else:
            child = rng.choice(parents).copy()
        child.mutate(cfg, self.tracker, rng)
        return child

    def _speciate(self) -> None:
        cfg = self.config
        for g in self.genomes:
            for s in self.species:
                if g.distance(s.representative, cfg) < cfg.compatibility_threshold:
                    s.members.append(g)
                    break
            else:
                self.species.append(Species(self._next_species_id, g, [g], last_improved=self.generation))
                self._next_species_id += 1
        self.species = [s for s in self.species if s.members]


def _allocate(shares: list[float], total: int) -> list[int]:
    """Split ``total`` proportionally to ``shares`` (largest remainder method)."""
    if not shares:
        return []
    weight = sum(shares)
    exact = [total * s / weight for s in shares] if weight > 0 else [total / len(shares)] * len(shares)
    counts = [int(x) for x in exact]
    by_remainder = sorted(range(len(shares)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in by_remainder[:total - sum(counts)]:
        counts[i] += 1
    return counts
