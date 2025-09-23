import numpy as np
import scipy.linalg
from typing import Tuple


def compute_angles(V1: np.ndarray, V2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute angles between vectors in V1 and V2 (column-wise)."""
    cos_thetas = np.einsum('ij,ij->j', V1, V2)
    thetas = np.arccos(cos_thetas)
    return cos_thetas, thetas


def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between subspaces spanned by columns of V1 and V2.

    Returns array of shape (min(m1, m2), nt) where V1 has shape (n, m1, nt)
    and V2 has shape (n, m2, nt).
    """
    n, m1, nt = V1.shape
    _, m2, _ = V2.shape
    theta = np.empty((min(m1, m2), nt), dtype=float)
    for i in range(nt):
        theta[:, i] = scipy.linalg.subspace_angles(V1[:, :, i], V2[:, :, i])
    return theta


__all__ = [
    "compute_angles",
    "principal_angles",
]

