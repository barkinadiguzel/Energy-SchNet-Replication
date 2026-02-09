import torch
import torch.nn as nn

class AtomEmbedding(nn.Module):
    def __init__(self, n_atom_types, f_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_atom_types, f_dim)

    def forward(self, Z):
        return self.embedding(Z)
