import random

import pytest
import torch

from moconut.neat import ConnGene, Genome, InnovationTracker, NEATConfig, NodeGene, Population, creates_cycle


def setup(**overrides):
    config = NEATConfig(num_inputs=2, num_outputs=1, **overrides)
    tracker = InnovationTracker(next_node_id=3)
    rng = random.Random(0)
    return config, tracker, rng, Genome.minimal(config, tracker, rng)


def outputs(genome, x0, x1):
    return genome.to_patch()(torch.tensor(x0), torch.tensor(x1))


def test_minimal_genome_wires_inputs_to_outputs():
    _, _, _, g = setup()
    assert g.ids("input") == [0, 1] and g.ids("output") == [2]
    assert sorted((c.src, c.dst) for c in g.conns.values()) == [(0, 2), (1, 2)]
    net = g.to_patch()
    assert net.inlets == ("x0", "x1") and net.outlets == ("y0",)
    assert not net.stateful


def test_patch_computes_the_genome():
    _, _, _, g = setup()
    out = g.nodes[2]
    w = {c.src: c.weight for c in g.conns.values()}
    expected = torch.sigmoid(torch.tensor(4.9 * (out.bias + w[0] * 0.3 + w[1] * -0.7)))
    assert torch.allclose(outputs(g, [0.3], [-0.7]), expected)


def test_add_node_preserves_function_with_identity_neurons():
    config, tracker, rng, g = setup(activation="identity")
    before = outputs(g, [0.3, 1.0], [-0.7, 2.0])
    nid = g.add_node(config, tracker, rng)
    assert g.nodes[nid].kind == "hidden"
    assert sum(not c.enabled for c in g.conns.values()) == 1
    assert torch.allclose(outputs(g, [0.3, 1.0], [-0.7, 2.0]), before)


def test_same_split_in_same_generation_gets_same_ids():
    config, tracker, _, a = setup()
    b = a.copy()
    rng_a, rng_b = random.Random(1), random.Random(1)
    assert a.add_node(config, tracker, rng_a) == b.add_node(config, tracker, rng_b)
    assert set(a.conns) == set(b.conns)
    tracker.new_generation()
    c = Genome.minimal(config, tracker, random.Random(5))
    assert c.add_node(config, tracker, random.Random(1)) != a.ids("hidden")[0]


def test_feed_forward_mutation_never_makes_cycles():
    config, tracker, rng, g = setup(add_node_rate=0.5, add_conn_rate=0.9)
    for _ in range(300):
        g.mutate(config, tracker, rng)
    net = g.to_patch()
    assert all(c.delay == 0 for c in net.cables)
    net.check()


def test_recurrent_mode_delays_cycle_closing_connections():
    config, tracker, rng, g = setup(feed_forward=False, add_node_rate=0.5, add_conn_rate=0.9)
    for _ in range(300):
        g.mutate(config, tracker, rng)
    net = g.to_patch()
    assert any(c.delay for c in net.cables)
    ys, _ = net.run(torch.rand(5, 3), torch.rand(5, 3))
    assert ys.shape == (5, 3)


def test_self_loop_reads_previous_step():
    config, tracker, _, g = setup(feed_forward=False, activation="identity")
    for c in g.conns.values():
        c.weight = 1.0
    g.nodes[2].bias = 0.0
    innovation = tracker.conn(2, 2)
    g.conns[innovation] = ConnGene(innovation, 2, 2, 0.5)
    ys, _ = g.to_patch().run(torch.tensor([1.0, 0.0, 0.0]), torch.zeros(3))
    assert ys.tolist() == [1.0, 0.5, 0.25]


def test_crossover_takes_structure_from_the_fitter_parent():
    config, tracker, rng, fitter = setup()
    other = fitter.copy()
    fitter.add_node(config, tracker, rng)
    other.add_connection(NEATConfig(2, 1, feed_forward=False), tracker, rng)  # a gene only `other` has
    child = Genome.crossover(fitter, other, config, rng)
    assert set(child.conns) == set(fitter.conns)
    assert set(child.nodes) == set(fitter.nodes)


def test_distance():
    config, tracker, rng, g = setup()
    assert g.distance(g.copy(), config) == 0.0
    h = g.copy()
    h.add_node(config, tracker, rng)      # one gene disabled, two excess genes added
    assert h.distance(g, config) == pytest.approx(2 * config.excess_coefficient)


def test_creates_cycle():
    edges = [(0, 2), (2, 3)]
    assert creates_cycle(edges, 3, 2)
    assert creates_cycle(edges, 3, 3)
    assert not creates_cycle(edges, 0, 3)


def test_neat_solves_xor():
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])

    @torch.no_grad()
    def fitness(genome):
        return 4.0 - ((genome.to_patch()(*x.T) - y) ** 2).sum().item()

    pop = Population(NEATConfig(num_inputs=2, num_outputs=1), seed=0)
    best = pop.run(fitness, generations=150, threshold=3.9)
    assert best.fitness >= 3.9
    assert ((best.to_patch()(*x.T) > 0.5).float() == y).all()
