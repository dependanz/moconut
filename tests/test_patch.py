import pytest
import torch
from torch import nn

import moconut
from moconut import Patch, PatchError


class AutoEncoder(Patch):
    def __init__(self, in_dim=784, latent_dim=256):
        super().__init__(inlets=["x"], outlets=["x_hat", "z"])
        self.encode = nn.Linear(in_dim, latent_dim)
        self.enc_act = nn.ReLU()
        self.decode = nn.Linear(latent_dim, in_dim)
        self.dec_act = nn.Sigmoid()
        self.chain("x", "encode", "enc_act", "decode", "dec_act", "x_hat")
        self.connect("enc_act", "z")


class Elman(Patch):
    """h_t = tanh(W_in x_t + W_rec h_{t-1}), with the loop closed by a delayed cable."""
    def __init__(self, in_dim, hid_dim):
        super().__init__(inlets=["x"], outlets=["h"])
        self.inp = nn.Linear(in_dim, hid_dim)
        self.rec = nn.Linear(hid_dim, hid_dim, bias=False)
        self.sum = moconut.Add()
        self.act = nn.Tanh()
        self.chain("x", "inp", "sum")
        self.chain("sum", "act", "h")
        self.chain("act", "rec")
        self.connect("rec", "sum", delay=1)


def elman_reference(cell, xs):
    h, hs = torch.zeros(xs.shape[1], cell.rec.in_features), []
    for x in xs:
        h = torch.tanh(cell.inp(x) + cell.rec(h))
        hs.append(h)
    return torch.stack(hs)


class Boom(nn.Module):
    def forward(self, x):
        raise AssertionError("pruned objects must not run")


################################################################
# Stateless patches
################################################################
def test_autoencoder_matches_plain_pytorch():
    torch.manual_seed(0)
    ae = AutoEncoder()
    x = torch.rand(32, 784)
    x_hat, z = ae(x)
    z_ref = torch.relu(ae.encode(x))
    assert torch.allclose(z, z_ref)
    assert torch.allclose(x_hat, torch.sigmoid(ae.decode(z_ref)))
    assert ae(x).z.shape == (32, 256)
    assert not ae.stateful


def test_gradients_reach_every_object():
    ae = AutoEncoder(8, 4)
    x = torch.rand(5, 8)
    nn.functional.mse_loss(ae(x).x_hat, x).backward()
    assert all(p.grad is not None for p in ae.parameters())


def test_single_outlet_returns_tensor_and_inputs_bind_by_name():
    p = Patch(inlets=["a", "b"], outlets=["y"])
    p.put("diff", moconut.Op(lambda a, b: a - b))
    p.connect("a", "diff.a")
    p.connect("b", "diff.b")
    p.connect("diff", "y")
    a, b = torch.tensor([3.0]), torch.tensor([1.0])
    assert p(a, b).item() == 2.0
    assert p(b=b, a=a).item() == 2.0
    with pytest.raises(PatchError, match="missing input"):
        p(a)


def test_gather_concat_and_tuple_outlets():
    class PlusMinus(nn.Module):
        def forward(self, x):
            return x, -x

    p = Patch(inlets=["x"], outlets=["total", "joined"])
    p.put("pm", PlusMinus())
    p.put("add", moconut.Add())
    p.put("cat", moconut.Concat(dim=-1))
    p.connect("x", "pm")
    p.connect("pm.0", "add")
    p.connect("pm.1", "add")
    p.connect("x", "add")             # fan-in: three cables into one gather inlet
    p.connect("pm.0", "cat.0")
    p.connect("pm.1", "cat.1")
    p.connect("add", "total")
    p.connect("cat", "joined")
    x = torch.tensor([[1.0, 2.0]])
    out = p(x)
    assert torch.equal(out.total, x)
    assert torch.equal(out.joined, torch.cat([x, -x], dim=-1))


def test_unused_objects_are_pruned():
    p = Patch(inlets=["x"], outlets=["y"])
    p.put("boom", Boom())
    p.put("id", nn.Identity())
    p.connect("x", "boom")
    p.chain("x", "id", "y")
    assert p(torch.ones(1)).item() == 1.0


def test_nested_patch_by_port_name():
    outer = Patch(inlets=["x"], outlets=["code"])
    outer.put("ae", AutoEncoder(8, 3))
    outer.put("twice", moconut.Op(lambda t: 2 * t))
    outer.chain("x", "ae")
    outer.chain("ae.z", "twice", "code")
    x = torch.rand(4, 8)
    assert torch.allclose(outer(x), 2 * outer.ae(x).z)


################################################################
# Delays and state
################################################################
def test_elman_cycle_matches_reference_loop():
    torch.manual_seed(0)
    cell = Elman(3, 5)
    xs = torch.randn(7, 4, 3)
    hs, state = cell.run(xs)
    assert cell.stateful
    assert torch.allclose(hs, elman_reference(cell, xs), atol=1e-6)

    # Stepping by hand gives the same thing
    state, stepped = None, []
    for x in xs:
        h, state = cell(x, state=state)
        stepped.append(h)
    assert torch.allclose(torch.stack(stepped), hs)


def test_backprop_through_time():
    cell = Elman(3, 5)
    hs, _ = cell.run(torch.randn(6, 2, 3))
    hs[-1].sum().backward()
    assert cell.rec.weight.grad is not None and cell.rec.weight.grad.abs().sum() > 0


def test_delay_two_is_a_shift_register():
    p = Patch(inlets=["x"], outlets=["y"])
    p.connect("x", "y", delay=2)
    ys, state = p.run(torch.arange(1.0, 6.0))
    assert ys.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]
    ys, _ = p.run(torch.tensor([9.0]), state=state)
    assert ys.tolist() == [4.0]


def test_delay_init_value():
    p = Patch(inlets=["x"], outlets=["y"])
    p.connect("x", "y", delay=1, init=-1.0)
    ys, _ = p.run(torch.tensor([5.0, 6.0]))
    assert ys.tolist() == [-1.0, 5.0]


def test_nested_stateful_patch_threads_its_state():
    torch.manual_seed(0)
    outer = Patch(inlets=["x"], outlets=["h"])
    outer.put("cell", Elman(3, 4))
    outer.chain("x", "cell", "h")
    assert outer.stateful
    xs = torch.randn(5, 2, 3)
    hs, state = outer.run(xs)
    assert "cell" in state
    assert torch.allclose(hs, elman_reference(outer.cell, xs), atol=1e-6)


def test_parent_notices_when_a_sub_patch_becomes_stateful():
    inner = Patch(inlets=["x"], outlets=["y"])
    inner.connect("x", "y")
    outer = Patch(inlets=["x"], outlets=["y"])
    outer.put("inner", inner)
    outer.chain("x", "inner", "y")
    assert not outer.stateful
    inner.disconnect("x", "y")
    inner.connect("x", "y", delay=1)
    assert outer.stateful
    y, state = outer(torch.ones(1))
    assert y.item() == 0.0 and "inner" in state


def test_detach_state():
    cell = Elman(2, 3)
    _, state = cell.run(torch.randn(3, 1, 2))
    detached = moconut.detach_state(state)
    assert not detached["rec->sum"][0].requires_grad


################################################################
# Wiring errors
################################################################
def test_unknown_names_suggest_a_fix():
    p = Patch(inlets=["x"], outlets=["y"])
    p.put("encode", nn.Identity())
    with pytest.raises(PatchError, match="did you mean 'encode'"):
        p.connect("x", "encdoe")


def test_delay_free_cycle_is_named():
    p = Patch(inlets=["x"], outlets=["y"])
    p.put("a", moconut.Add())
    p.put("b", nn.Identity())
    p.chain("x", "a", "b", "y")
    p.connect("b", "a")
    with pytest.raises(PatchError, match="delay-free cycle: a -> b -> a"):
        p.check()
    p.disconnect("b", "a")
    p.connect("b", "a", delay=1)
    p.check()


def test_self_loop_needs_a_delay():
    p = Patch(inlets=["x"], outlets=["y"])
    p.put("a", moconut.Add())
    p.chain("x", "a", "y")
    p.connect("a", "a")
    with pytest.raises(PatchError, match="a -> a"):
        p.check()


def test_one_cable_per_plain_inlet():
    p = Patch(inlets=["x", "z"], outlets=["y"])
    p.put("lin", nn.Linear(2, 2))
    p.connect("x", "lin")
    with pytest.raises(PatchError, match="already has a cable"):
        p.connect("z", "lin")


def test_unconnected_outlet_and_inlet_order():
    p = Patch(inlets=["x"], outlets=["y", "w"])
    p.put("cat", moconut.Concat())
    p.connect("x", "cat.1")
    p.connect("cat", "y")
    with pytest.raises(PatchError, match="outlet 'w' is not connected"):
        p.check()
    p.connect("x", "w")
    with pytest.raises(PatchError, match="inlet 0 is not connected"):
        p.check()


def test_names_must_not_clash():
    p = Patch(inlets=["x"], outlets=["y"])
    with pytest.raises(PatchError, match="already a port"):
        p.put("x", nn.Identity())
    with pytest.raises(PatchError, match="clashes"):
        p.put("forward", nn.Identity())
    with pytest.raises(PatchError, match="clashes"):
        p.put("double", nn.Identity())    # nn.Module.double() would be shadowed


def test_remove_drops_cables_and_replans():
    p = Patch(inlets=["x"], outlets=["y"])
    p.put("neg", moconut.Op(torch.neg))
    p.chain("x", "neg", "y")
    assert p(torch.ones(1)).item() == -1.0
    p.remove("neg")
    assert p.cables == ()
    p.connect("x", "y")
    assert p(torch.ones(1)).item() == 1.0
