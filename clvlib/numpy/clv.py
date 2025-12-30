import numpy as np
import scipy.linalg
from tqdm.auto import tqdm


def _ginelli(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Backward (standard) Ginelli algorithm."""

    n_time, n_dim, n_lyap = Q.shape
    V = np.empty((n_time, n_dim, n_lyap), dtype=Q.dtype)

    C = np.eye(n_lyap, dtype=Q.dtype)
    V[-1] = Q[-1] @ C

    for i in tqdm(range(n_time - 2, -1, -1), leave=False):
        C = scipy.linalg.solve_triangular(
            R[i], C, lower=False, overwrite_b=True, check_finite=False
        )
        C /= np.linalg.norm(C, axis=0, keepdims=True)
        V[i] = Q[i] @ C
    return V


def _upwind_ginelli(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Upwind (forward-shifted) Ginelli algorithm variant."""

    n_time, n_dim, n_lyap = Q.shape
    V = np.empty((n_time, n_dim, n_lyap), dtype=Q.dtype)

    C = np.eye(n_lyap, dtype=Q.dtype)
    V[-1] = Q[-1] @ C

    for i in tqdm(range(n_time - 2, -1, -1), leave=False):
        C = scipy.linalg.solve_triangular(
            R[i + 1], C, lower=False, overwrite_b=True, check_finite=False
        )
        C /= np.linalg.norm(C, axis=0, keepdims=True)
        V[i] = Q[i] @ C
    return V


_GINELLI_METHODS = {
    "ginelli": _ginelli,
    "upwind": _upwind_ginelli,
    "upwind_ginelli": _upwind_ginelli,
}


def _clvs(
    Q: np.ndarray, R: np.ndarray, *, ginelli_method: str = "ginelli"
) -> np.ndarray:
    """Dispatch CLV reconstruction to the selected Ginelli variant."""

    try:
        solver = _GINELLI_METHODS[ginelli_method.lower()]
    except KeyError as exc:  
        available = ", ".join(sorted(_GINELLI_METHODS))
        raise ValueError(
            f"Unknown ginelli_method '{ginelli_method}'. Available: {available}."
        ) from exc

    V = solver(Q, R)
    return V


__all__ = ["_clvs"]
