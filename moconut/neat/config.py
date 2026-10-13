from __future__ import annotations

from dataclasses import dataclass

from ..neuron import ACTIVATIONS, AGGREGATIONS


@dataclass
class NEATConfig:
    """NEAT hyperparameters. Defaults follow Stanley & Miikkulainen (2002) where the paper gives a value."""
    num_inputs: int
    num_outputs: int
    pop_size: int = 150
    feed_forward: bool = True        # False allows cycles; cables that close one get delay=1

    # Node genes
    activation: str = "steep_sigmoid"
    aggregation: str = "sum"

    # Compatibility distance: c1 * excess / N + c2 * disjoint / N + c3 * mean weight difference
    excess_coefficient: float = 1.0
    disjoint_coefficient: float = 1.0
    weight_coefficient: float = 0.4
    compatibility_threshold: float = 3.0

    # Mutation
    weight_mutate_rate: float = 0.8   # chance a genome's weights and biases are mutated
    weight_replace_rate: float = 0.1  # per weight: chance it is replaced instead of perturbed
    weight_perturb_power: float = 0.5
    weight_init_std: float = 1.0
    weight_limit: float = 30.0
    add_node_rate: float = 0.03
    add_conn_rate: float = 0.05
    add_conn_attempts: int = 20

    # Reproduction
    crossover_rate: float = 0.75       # the rest of the offspring come from mutation alone
    interspecies_rate: float = 0.001
    disable_inherit_rate: float = 0.75  # a gene disabled in either parent stays disabled
    survival_threshold: float = 0.2     # fraction of each species allowed to reproduce
    elitism_min_size: int = 5           # species this big keep their champion unchanged
    stagnation_limit: int = 15          # generations without improvement before a species dies

    def __post_init__(self):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation '{self.activation}'")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation '{self.aggregation}'")
        if self.num_inputs < 1 or self.num_outputs < 1:
            raise ValueError("NEAT needs at least one input and one output")
