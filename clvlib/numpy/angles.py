import numpy as np
import scipy.linalg
from typing import Tuple
from tqdm.auto import tqdm


def compute_angles(v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute angles between vectors in v1 and v2 (row-wise).

    Assumes unit-length vectors along rows; clamps cosine to [-1, 1] for
    numerical stability.
    """
    cos_thetas = np.einsum("ij,ij->i", v1, v2)
    cos_thetas = np.clip(cos_thetas, -1.0, 1.0)
    thetas = np.arccos(cos_thetas)
    return cos_thetas, thetas


def principal_angles(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between subspaces spanned by columns of V1 and V2. Only works for unit length vectors

    Time-first convention: V1 has shape (nt, n, m1), V2 has shape (nt, n, m2).
    Returns array of shape (nt, min(m1, m2)).
    """
    nt, _, m1 = v1.shape
    _, _, m2 = v2.shape
    theta = np.empty((nt, min(m1, m2)), dtype=float)
    for i in tqdm(range(nt), leave=False):
        theta[i] = np.squeeze(scipy.linalg.subspace_angles(v1[i], v2[i]))
    return theta


__all__ = [
    "compute_angles",
    "principal_angles",
]
