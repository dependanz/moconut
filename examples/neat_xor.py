"""Evolve XOR with NEAT. Every genome is evaluated as a moconut.Patch."""
import torch

from moconut.neat import NEATConfig, Population

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([0.0, 1.0, 1.0, 0.0])


@torch.no_grad()
def fitness(genome):
    net = genome.to_patch()
    return 4.0 - ((net(*X.T) - Y) ** 2).sum().item()


if __name__ == "__main__":
    pop = Population(NEATConfig(num_inputs=2, num_outputs=1), seed=0)
    best = pop.run(fitness, generations=300, threshold=3.9, verbose=True)

    net = best.to_patch()
    print(f"\n{best}\n")
    print(net)
    with torch.no_grad():
        for x, y, out in zip(X.tolist(), Y.tolist(), net(*X.T).tolist()):
            print(f"{x} -> {out:.3f} (target {y})")
