from typing import Union
import numpy as np
import torch


def l2_normalize(x: Union[np.ndarray, torch.Tensor], axis: int = -1) -> Union[np.ndarray, torch.Tensor]:
    """Normalizes vector or batch of vectors to unit length."""
    if isinstance(x, torch.Tensor):
        return torch.nn.functional.normalize(x, p=2, dim=axis)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    return x / norm


def find_cosine_distance(
    source_representation: Union[np.ndarray, torch.Tensor, list],
    test_representation: Union[np.ndarray, torch.Tensor, list],
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes Cosine Distance (1 - cosine_similarity).
    Supports NumPy arrays, PyTorch tensors, and 1D/2D batches.
    """
    if isinstance(source_representation, torch.Tensor) and isinstance(test_representation, torch.Tensor):
        if source_representation.device != test_representation.device:
            test_representation = test_representation.to(source_representation.device)

        a = source_representation.float()
        b = test_representation.float()
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if b.ndim == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        sim = torch.mm(a_norm, b_norm.t())
        dist = 1.0 - sim

        if dist.numel() == 1:
            return float(dist.item())
        return dist

    # Fallback to NumPy
    a = np.asarray(source_representation, dtype=np.float32)
    b = np.asarray(test_representation, dtype=np.float32)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)

    sim = np.dot(a_norm, b_norm.T)
    dist = 1.0 - sim

    if dist.size == 1:
        return float(dist.item())
    return dist


def find_euclidean_distance(
    source_representation: Union[np.ndarray, torch.Tensor, list],
    test_representation: Union[np.ndarray, torch.Tensor, list],
) -> Union[float, np.ndarray, torch.Tensor]:
    """Computes Euclidean distance between representations."""
    if isinstance(source_representation, torch.Tensor) and isinstance(test_representation, torch.Tensor):
        if source_representation.device != test_representation.device:
            test_representation = test_representation.to(source_representation.device)
        a = source_representation.float()
        b = test_representation.float()
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if b.ndim == 1:
            b = b.unsqueeze(0)
        dist = torch.cdist(a, b)
        if dist.numel() == 1:
            return float(dist.item())
        return dist

    a = np.asarray(source_representation, dtype=np.float32)
    b = np.asarray(test_representation, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Broadcast difference
    dist = np.linalg.norm(a[:, np.newaxis, :] - b[np.newaxis, :, :], axis=2)
    if dist.size == 1:
        return float(dist.item())
    return dist


def find_cosine_distance_cpu(a: np.ndarray, b: np.ndarray) -> float:
    """Convenience CPU cosine distance between two 1D vectors."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def find_euclidean_distance_cpu(a: np.ndarray, b: np.ndarray) -> float:
    """Convenience CPU euclidean distance between two 1D vectors."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.linalg.norm(a - b))


