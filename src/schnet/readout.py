import torch
import torch.nn as nn

class Readout(nn.Module):
    def __init__(self, f_dim):
        super().__init__()
        self.atom_to_energy = nn.Linear(f_dim, 1)

    def forward(self, x):
        e_atoms = self.atom_to_energy(x)  # (n_atoms, 1)
        E = e_atoms.sum(dim=0)            # (1,)
        return E
