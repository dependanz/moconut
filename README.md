# moconut

<!-- `m.o.c.o.n.u.t. oversees construction of new universal templates` ... nah. -->

**Pure Data for neural networks.** A `moconut.Patch` is a PyTorch `nn.Module` whose
forward pass you don't write: you place objects and connect them with cables, and
moconut works out the rest. Because the network is a graph you can inspect and edit,
it can be rewired by hand or by evolution, and trained with backprop.

> Status: early rewrite (0.1.0.dev0). The API may change.

## Patches

```python
import torch
from torch import nn
import moconut


class AutoEncoder(moconut.Patch):
    def __init__(self, in_dim=784, latent_dim=256):
        super().__init__(inlets=["x"], outlets=["x_hat", "z"])

        # Objects: any nn.Module, including other Patches
        self.encode = nn.Linear(in_dim, latent_dim)
        self.enc_act = nn.ReLU()
        self.decode = nn.Linear(latent_dim, in_dim)
        self.dec_act = nn.Sigmoid()

        # Cables: no forward() needed
        self.chain("x", "encode", "enc_act", "decode", "dec_act", "x_hat")
        self.connect("enc_act", "z")


ae = AutoEncoder()
x_hat, z = ae(torch.rand(32, 784))
```

- Addresses are `"name"` (port 0, or one of the patch's own ports) or `"name.port"`,
  where the port is an index or a name, e.g. `"ae.z"` for a sub-patch's outlet.
- Wiring is checked before the first call: unknown names (with a "did you mean"),
  unconnected outlets, two cables into one inlet, and cycles without a delay.
- Built-in objects: `Add` and `Mul` (any number of cables into one inlet),
  `Concat`, `Op(fn)` to wrap any function, and `Neuron`.

## Cycles and state

Like Pd's signal domain, every cycle must pass through a delay. A cable with
`delay=n` carries the value its source produced `n` steps ago (starting from 0).
One call is one step, and a patch with delays returns `(outputs, state)`:

```python
class Elman(moconut.Patch):
    def __init__(self, in_dim, hid_dim):
        super().__init__(inlets=["x"], outlets=["h"])
        self.inp = nn.Linear(in_dim, hid_dim)
        self.rec = nn.Linear(hid_dim, hid_dim, bias=False)
        self.sum = moconut.Add()
        self.act = nn.Tanh()
        self.chain("x", "inp", "sum")
        self.chain("sum", "act", "h")
        self.chain("act", "rec")
        self.connect("rec", "sum", delay=1)   # closes the loop


cell = Elman(28, 128)
h, state = cell(x_t, state=None)              # one step
hs, state = cell.run(xs)                      # step over dim 0; backprop through time just works
```

## Neurons

A `Neuron` computes `activation(bias + response * aggregation(w_i * x_i))`, with one
weight per incoming cable:

```python
net = moconut.Patch(inlets=["x0", "x1"], outlets=["y"])
net.put("h", moconut.Neuron("tanh"))
net.connect("x0", "h", weight=0.5)
net.connect("x1", "h", weight=-1.0)
net.connect("h", "h", delay=1, weight=0.9)   # recurrent self-connection
net.connect("h", "y")
```

## NEAT

`moconut.neat` implements NEAT (Stanley & Miikkulainen, 2002): innovation numbers,
add-node and add-connection mutations, crossover, speciation with fitness sharing,
and stagnation. Every genome becomes a `Patch` through `genome.to_patch()`.

```python
import torch
from moconut.neat import NEATConfig, Population

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([0.0, 1.0, 1.0, 0.0])

@torch.no_grad()
def fitness(genome):
    return 4.0 - ((genome.to_patch()(*X.T) - Y) ** 2).sum().item()

best = Population(NEATConfig(num_inputs=2, num_outputs=1), seed=0).run(fitness, 300, threshold=3.9)
```

With the paper's settings this solves XOR in all of 30 seeds, in 22.5 generations on
average. With `NEATConfig(feed_forward=False)`, connections may form cycles; a
connection that would close a delay-free cycle gets `delay=1`, so feed-forward paths
still take a single step. See `examples/neat_xor.py`.

## Roadmap

- [x] Patch core: objects, cables, delays, checks, stepping
- [x] Neurons and NEAT
- [ ] More objects: `Switch`, `Split`, `Constant`, `Param`, `Probe`
- [ ] Compile neuron patches into batched tensor ops
- [ ] Backprop on evolved topologies (Backprop NEAT), mixed with ordinary layers
