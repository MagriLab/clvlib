import numpy as np
import scipy.linalg


def _normalize(A: np.ndarray) -> np.ndarray:
    return A / np.linalg.norm(A, axis=0, keepdims=True)


def _ginelli(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Backward (standard) Ginelli algorithm."""

    n_time, n_dim, n_lyap = Q.shape
    V = np.empty((n_time, n_dim, n_lyap), dtype=Q.dtype)

    C = np.eye(n_lyap, dtype=Q.dtype)
    V[-1] = Q[-1] @ C

    for i in reversed(range(n_time - 1)):
        C = scipy.linalg.solve_triangular(R[i], C, lower=False, overwrite_b=True, check_finite=False)
        V[i] = Q[i] @ _normalize(C)
    return V


def _upwind_ginelli(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Upwind (forward-shifted) Ginelli algorithm variant."""

    n_time, n_dim, n_lyap = Q.shape
    V = np.empty((n_time, n_dim, n_lyap), dtype=Q.dtype)

    C = np.eye(n_lyap, dtype=Q.dtype)
    V[-1] = Q[-1] @ C

    for i in reversed(range(n_time - 1)):
        C = scipy.linalg.solve_triangular(R[i + 1], C, lower=False, overwrite_b=True, check_finite=False)
        V[i] = Q[i] @ _normalize(C)
    return V


_GINELLI_METHODS = {
    "standard": _ginelli,
    "ginelli": _ginelli,
    "backward": _ginelli,
    "upwind": _upwind_ginelli,
    "upwind_ginelli": _upwind_ginelli,
}


def _clvs(Q: np.ndarray, R: np.ndarray, *, ginelli_method: str = "standard") -> np.ndarray:
    """Dispatch CLV reconstruction to the selected Ginelli variant."""

    try:
        solver = _GINELLI_METHODS[ginelli_method.lower()]
    except KeyError as exc:  # pragma: no cover - defensive path
        available = ", ".join(sorted(_GINELLI_METHODS))
        raise ValueError(
            f"Unknown ginelli_method '{ginelli_method}'. Available: {available}."
        ) from exc

    V = solver(Q, R)
    return V 


__all__ = [
    "_clvs",
    "_normalize_columns",
]
