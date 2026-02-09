import torch
import torch.nn as nn
from .atom_embedding import AtomEmbedding
from .interaction_block import InteractionBlock
from .readout import Readout

class SchNet(nn.Module):
    def __init__(self, n_atom_types, f_dim, rbf_dim, n_interactions):
        super().__init__()
        self.embedding = AtomEmbedding(n_atom_types, f_dim)
        self.interactions = nn.ModuleList([
            InteractionBlock(f_dim, rbf_dim) for _ in range(n_interactions)
        ])
        self.readout = Readout(f_dim)

    def forward(self, Z, rbf, neighbors):
        x = self.embedding(Z)
        for block in self.interactions:
            x = block(x, rbf, neighbors)
        E = self.readout(x)
        return E
