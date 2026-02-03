import torch
import numpy as np
from numba import jit

def find_cosine_distance(source_representation, test_representation):
    """
    Computes Cosine Distance using PyTorch for batch processing.
    Args:
        source_representation: Tensor or list
        test_representation: Tensor or list
    Returns:
        float or Tensor: Cosine distance (1 - cosine_similarity)
    """
    if not isinstance(source_representation, torch.Tensor):
        source_representation = torch.tensor(source_representation)
    if not isinstance(test_representation, torch.Tensor):
        test_representation = torch.tensor(test_representation)

    # Ensure tensors are on the same device and float type
    if source_representation.device != test_representation.device:
       test_representation = test_representation.to(source_representation.device)

    a = source_representation
    b = test_representation
    
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    # L2 Normalize
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    
    # Cosine Similarity: dot product of normalized vectors
    # If comparing one to many, use matmul
    # a: (N, D), b: (M, D) -> (N, M)
    sim = torch.mm(a_norm, b_norm.t())
    dist = 1 - sim
    
    return dist

def find_euclidean_distance(source_representation, test_representation):
    if not isinstance(source_representation, torch.Tensor):
        source_representation = torch.tensor(source_representation)
    if not isinstance(test_representation, torch.Tensor):
        test_representation = torch.tensor(test_representation)

    if source_representation.device != test_representation.device:
       test_representation = test_representation.to(source_representation.device)
       
    return torch.pairwise_distance(source_representation, test_representation)

def l2_normalize(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return torch.nn.functional.normalize(x, p=2, dim=-1)

# Numba optimized versions for CPU-heavy tasks or loop-based fallbacks
@jit(nopython=True)
def find_cosine_distance_cpu(a, b):
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1 - (dot / (norm_a * norm_b))

@jit(nopython=True)
def find_euclidean_distance_cpu(a, b):
    return np.linalg.norm(a - b)

