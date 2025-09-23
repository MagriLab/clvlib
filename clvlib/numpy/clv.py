import numpy as np
import scipy.linalg
from typing import Tuple


def _normalize_columns(A: np.ndarray) -> np.ndarray:
    return A / np.linalg.norm(A, axis=0, keepdims=True)


def _clvs(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute the Covariant Lyapunov Vectors (CLVs) using Ginelli et al. (2007).

    Parameters
    ----------
    Q : ndarray, shape (n_dim, n_lyap, n_time)
        Timeseries of Gram–Schmidt vectors.
    R : ndarray, shape (n_lyap, n_lyap, n_time)
        Timeseries of upper-triangular R matrices from QR decomposition.
    """
    n_dim, n_lyap, n_time = Q.shape

    C = np.empty((n_lyap, n_lyap, n_time), dtype=Q.dtype)
    V = np.empty((n_dim, n_lyap, n_time), dtype=Q.dtype)

    C[:, :, -1] = np.eye(n_lyap)
    V[:, :, -1] = Q[:, :, -1] @ C[:, :, -1]

    for i in reversed(range(n_time - 1)):
        C_next = scipy.linalg.solve_triangular(
            R[:, :, i], C[:, :, i + 1], lower=False, overwrite_b=True, check_finite=False
        )
        C[:, :, i] = _normalize_columns(C_next)
        V[:, :, i] = Q[:, :, i] @ C[:, :, i]

    V = _normalize_columns(V)
    return V


__all__ = [
    "_clvs",
    "_normalize_columns",
]

