import pytest
import torch

from moconut import Neuron, Patch, PatchError


def two_input_net(activation="identity", aggregation="sum"):
    net = Patch(inlets=["x0", "x1"], outlets=["y"])
    net.put("h", Neuron(activation, aggregation, bias=0.5))
    net.connect("x0", "h", weight=2.0)
    net.connect("x1", "h", weight=-1.0)
    net.connect("h", "y")
    return net


def test_weighted_sum_plus_bias():
    net = two_input_net()
    x0, x1 = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 0.0])
    assert net(x0, x1).tolist() == [0.5 + 2 - 3, 0.5 + 4 - 0]


def test_aggregation_and_activation():
    net = two_input_net("tanh", "product")
    x0, x1 = torch.tensor([1.0]), torch.tensor([3.0])
    expected = torch.tanh(torch.tensor(0.5 + (2.0 * 1.0) * (-1.0 * 3.0)))
    assert torch.allclose(net(x0, x1), expected)


def test_weights_follow_cables():
    net = two_input_net()
    assert set(net.h.weights) == {"x0", "x1"}
    net.disconnect("x1", "h")
    assert set(net.h.weights) == {"x0"}
    net.connect("x1", "h")                       # no weight given: a random one is created
    assert set(net.h.weights) == {"x0", "x1"}


def test_recurrent_self_connection():
    net = Patch(inlets=["x"], outlets=["y"])
    net.put("h", Neuron("identity"))
    net.connect("x", "h", weight=1.0)
    net.connect("h", "h", delay=1, weight=0.5)   # h_t = x_t + 0.5 h_{t-1}
    net.connect("h", "y")
    ys, _ = net.run(torch.tensor([1.0, 0.0, 0.0]))
    assert ys.tolist() == [1.0, 0.5, 0.25]


def test_gradients_reach_weights_bias_and_response():
    net = two_input_net("sigmoid")
    net(torch.rand(4), torch.rand(4)).sum().backward()
    assert all(p.grad is not None for p in net.parameters())


def test_neuron_without_inputs_outputs_its_bias():
    net = Patch(outlets=["y"])
    net.put("h", Neuron("identity", bias=0.25))
    net.connect("h", "y")
    assert net().item() == 0.25


def test_cable_parameters_only_go_to_objects_that_take_them():
    net = Patch(inlets=["x"], outlets=["y"])
    with pytest.raises(PatchError, match="takes no cable parameters"):
        net.connect("x", "y", weight=1.0)


def test_unknown_activation():
    with pytest.raises(ValueError, match="unknown activation"):
        Neuron("swishy")
