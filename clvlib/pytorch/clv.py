import torch

Tensor = torch.Tensor


def _normalize(A: Tensor) -> Tensor:
    return A / torch.linalg.norm(A, dim=1, keepdim=True)


def _solve_upper_triangular(R: Tensor, C: Tensor) -> Tensor:
    if hasattr(torch.linalg, "solve_triangular"):
        return torch.linalg.solve_triangular(R, C, upper=True)
    return torch.triangular_solve(C, R, upper=True).solution


def _ginelli(Q: Tensor, R: Tensor) -> Tensor:
    """Backward (standard) Ginelli algorithm."""
    n_time, n_dim, n_lyap = Q.shape
    V = torch.empty((n_time, n_dim, n_lyap), dtype=Q.dtype, device=Q.device)

    C = torch.eye(n_lyap, dtype=Q.dtype, device=Q.device)
    V[-1] = Q[-1] @ C

    for i in reversed(range(n_time - 1)):
        C = _solve_upper_triangular(R[i], C)
        C = _normalize(C)
        V[i] = Q[i] @ C
    return V


def _upwind_ginelli(Q: Tensor, R: Tensor) -> Tensor:
    """Upwind (forward-shifted) Ginelli algorithm variant."""
    n_time, n_dim, n_lyap = Q.shape
    V = torch.empty((n_time, n_dim, n_lyap), dtype=Q.dtype, device=Q.device)

    C = torch.eye(n_lyap, dtype=Q.dtype, device=Q.device)
    V[-1] = Q[-1] @ C

    for i in reversed(range(n_time - 1)):
        C = _solve_upper_triangular(R[i + 1], C)
        C = _normalize(C)
        V[i] = Q[i] @ C
    return V


_GINELLI_METHODS = {
    "standard": _ginelli,
    "ginelli": _ginelli,
    "backward": _ginelli,
    "upwind": _upwind_ginelli,
    "upwind_ginelli": _upwind_ginelli,
}


def _clvs(Q: Tensor, R: Tensor, *, ginelli_method: str = "standard") -> Tensor:
    """Dispatch CLV reconstruction to the selected Ginelli variant."""
    try:
        solver = _GINELLI_METHODS[ginelli_method.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_GINELLI_METHODS))
        raise ValueError(
            f"Unknown ginelli_method '{ginelli_method}'. Available: {available}."
        ) from exc

    V = solver(Q, R)
    return V / torch.linalg.norm(V, dim=1, keepdim=True)


__all__ = [
    "_clvs",
    "_normalize_columns",
]
