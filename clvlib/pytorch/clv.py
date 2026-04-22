import torch
from tqdm.auto import tqdm

Tensor = torch.Tensor


def _normalize(C: Tensor) -> tuple[Tensor, Tensor]:
    D = torch.linalg.norm(C, dim=0)
    safe_D = torch.where(D == 0, torch.ones_like(D), D)
    return C / safe_D.unsqueeze(0), D


def _ginelli(Q: Tensor, R: Tensor) -> tuple[Tensor, Tensor]:
    """Ginelli algorithm."""
    n_time, n_dim, n_lyap = Q.shape
    V = torch.empty((n_time, n_dim, n_lyap), dtype=Q.dtype, device=Q.device)
    D_history = torch.empty((n_time, n_lyap), dtype=Q.dtype, device=Q.device)

    C = torch.eye(n_lyap, dtype=Q.dtype, device=Q.device)
    V[-1] = Q[-1] @ C
    D_history[-1] = 1

    for i in tqdm(range(n_time - 2, -1, -1), leave=False):
        C = torch.linalg.solve_triangular(R[i], C, upper=True)
        C, D = _normalize(C)
        V[i] = Q[i] @ C
        D_history[i] = D
    return V, D_history


def _clvs(Q: Tensor, R: Tensor) -> tuple[Tensor, Tensor]:
    return _ginelli(Q, R)


__all__ = [
    "_clvs",
]
