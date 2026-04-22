import numpy as np
import scipy.linalg
from tqdm.auto import tqdm


def _normalize(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = np.linalg.norm(C, axis=0)
    safe_D = D.copy()
    safe_D[safe_D == 0] = 1.0
    return C / safe_D[np.newaxis, :], D


def _ginelli(Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backward (standard) Ginelli algorithm."""

    n_time, n_dim, n_lyap = Q.shape
    V = np.empty((n_time, n_dim, n_lyap), dtype=Q.dtype)
    D_history = np.empty((n_time, n_lyap), dtype=Q.dtype)

    C = np.eye(n_lyap, dtype=Q.dtype)
    V[-1] = Q[-1] @ C
    D_history[-1] = 1.0

    for i in tqdm(range(n_time - 2, -1, -1), leave=False):
        C = scipy.linalg.solve_triangular(
            R[i], C, lower=False, overwrite_b=True, check_finite=False
        )
        C, D = _normalize(C)
        V[i] = Q[i] @ C
        D_history[i] = D
    return V, D_history


def _clvs(Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _ginelli(Q, R)


__all__ = ["_clvs"]
