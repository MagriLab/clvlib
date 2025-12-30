import torch
from tqdm.auto import tqdm

Tensor = torch.Tensor


def _ginelli(Q: Tensor, R: Tensor) -> Tensor:
    """Ginelli algorithm."""
    n_time, n_dim, n_lyap = Q.shape
    V = torch.empty((n_time, n_dim, n_lyap), dtype=Q.dtype, device=Q.device)

    C = torch.eye(n_lyap, dtype=Q.dtype, device=Q.device)
    V[-1] = Q[-1] @ C

    for i in tqdm(range(n_time - 2, -1, -1), leave=False):
        C = torch.linalg.solve_triangular(R[i], C, upper=True)
        C /= torch.norm(C, dim=0, keepdim=True)
        V[i] = Q[i] @ C
    return V


def _upwind_ginelli(Q: Tensor, R: Tensor) -> Tensor:
    """Upwind (forward-shifted) Ginelli algorithm variant."""
    n_time, n_dim, n_lyap = Q.shape
    V = torch.empty((n_time, n_dim, n_lyap), dtype=Q.dtype, device=Q.device)

    C = torch.eye(n_lyap, dtype=Q.dtype, device=Q.device)
    V[-1] = Q[-1] @ C

    for i in tqdm(range(n_time - 2, -1, -1), leave=False):
        C = torch.linalg.solve_triangular(R[i + 1], C, upper=True)
        C /= torch.norm(C, dim=0, keepdim=True)
        V[i] = Q[i] @ C
    return V


_GINELLI_METHODS = {
    "ginelli": _ginelli,
    "upwind": _upwind_ginelli,
    "upwind_ginelli": _upwind_ginelli,
}


def _clvs(Q: Tensor, R: Tensor, *, ginelli_method: str = "ginelli") -> Tensor:
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


__all__ = [
    "_clvs",
]
