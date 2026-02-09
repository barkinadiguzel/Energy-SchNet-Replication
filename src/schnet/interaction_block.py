import torch
import torch.nn as nn
from .cfconv import CFConv

class InteractionBlock(nn.Module):
    def __init__(self, f_dim, rbf_dim):
        super().__init__()
        self.atom_layer1 = nn.Linear(f_dim, f_dim)
        self.cfconv = CFConv(f_dim, rbf_dim)
        self.atom_layer2 = nn.Linear(f_dim, f_dim)
        self.activation = nn.Softplus()  

    def forward(self, x, rbf, neighbors):
        v = self.atom_layer1(x)
        v = self.activation(v)
        v = self.cfconv(v, rbf, neighbors)
        v = self.atom_layer2(v)
        v = self.activation(v)
        x = x + v
        return x
