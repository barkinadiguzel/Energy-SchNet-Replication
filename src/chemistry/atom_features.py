import torch

ATOM_TYPES = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4
}

def atom_type_to_index(atom_list):
    indices = [ATOM_TYPES[a] for a in atom_list]
    return torch.tensor(indices, dtype=torch.long)
