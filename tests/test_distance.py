import numpy as np
import torch
import pytest
from facenn.utils.distance import (
    find_cosine_distance,
    find_euclidean_distance,
    find_cosine_distance_cpu,
    find_euclidean_distance_cpu,
    l2_normalize,
)


def test_l2_normalize_numpy():
    v = np.array([3.0, 4.0], dtype=np.float32)
    normed = l2_normalize(v)
    assert np.allclose(np.linalg.norm(normed), 1.0)
    assert np.allclose(normed, [0.6, 0.8])


def test_l2_normalize_torch():
    t = torch.tensor([[3.0, 4.0], [1.0, 0.0]], dtype=torch.float32)
    normed = l2_normalize(t, axis=-1)
    assert torch.allclose(torch.norm(normed, p=2, dim=-1), torch.tensor([1.0, 1.0]))


def test_cosine_distance_identical():
    a = np.array([1.0, 2.0, 3.0])
    dist = find_cosine_distance(a, a)
    assert pytest.approx(dist, abs=1e-5) == 0.0


def test_cosine_distance_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    dist = find_cosine_distance(a, b)
    assert pytest.approx(dist, abs=1e-5) == 1.0


def test_cosine_distance_torch():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    dist = find_cosine_distance(a, b)
    assert pytest.approx(dist, abs=1e-5) == 1.0


def test_euclidean_distance():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert pytest.approx(find_euclidean_distance(a, b), abs=1e-5) == 5.0
    assert pytest.approx(find_euclidean_distance_cpu(a, b), abs=1e-5) == 5.0
