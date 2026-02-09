import torch

def compute_distance_matrix(positions):
    diff = positions.unsqueeze(1) - positions.unsqueeze(0) 
    dist_matrix = torch.norm(diff, dim=-1)                
    return dist_matrix
