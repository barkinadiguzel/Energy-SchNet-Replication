import torch
from .atom_features import atom_type_to_index
from .distance_matrix import compute_distance_matrix

def rbf_expand(dist_matrix, centers, gamma):
    d = dist_matrix.unsqueeze(-1)  
    c = torch.tensor(centers).view(1, 1, -1)  
    rbf = torch.exp(-gamma * (d - c)**2)
    return rbf

def build_molecule(atom_list, positions, centers, gamma):
    Z = atom_type_to_index(atom_list)
    dist_matrix = compute_distance_matrix(positions)
    rbf = rbf_expand(dist_matrix, centers, gamma)
    n_atoms = len(atom_list)
    neighbors = torch.arange(n_atoms).unsqueeze(0).repeat(n_atoms, 1)
    
    return Z, rbf, neighbors
