import torch
import torch.nn as nn
import torch.nn.functional as F

class CFConv(nn.Module):
    def __init__(self, f_dim, rbf_dim):
        super().__init__()
        self.filter_network = nn.Sequential(
            nn.Linear(rbf_dim, f_dim),
            nn.Softplus(),   
            nn.Linear(f_dim, f_dim)
        )

    def forward(self, x, rbf, neighbors):
        W = self.filter_network(rbf)  
        x_j = x[neighbors]             
        out = x_j * W
        out = out.sum(dim=1)         
        return out
