import numpy as np
import scipy.linalg
from typing import Tuple


def compute_angles(V1: np.ndarray, V2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute angles between vectors in V1 and V2 (column-wise)."""
    cos_thetas = np.einsum("ij,ij->j", V1, V2)
    # Clamp for numerical safety to avoid NaNs from tiny overshoots
    cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
    thetas = np.arccos(cos_thetas)
    return cos_thetas, thetas


def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between subspaces spanned by columns of V1 and V2.

    Time-first convention: V1 has shape (nt, n, m1), V2 has shape (nt, n, m2).
    Returns array of shape (nt, min(m1, m2)).
    """
    nt, _, m1 = V1.shape
    _, _, m2 = V2.shape
    theta = np.empty((nt, min(m1, m2)), dtype=float)
    for i in range(nt):
        theta[i] = scipy.linalg.subspace_angles(V1[i], V2[i])
    return theta


__all__ = [
    "compute_angles",
    "principal_angles",
]
