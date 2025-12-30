import torch
from typing import Tuple
from tqdm.auto import tqdm

Tensor = torch.Tensor


def compute_angles(V1: Tensor, V2: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute angles between corresponding rows of V1 and V2.

    Assumes unit-length vectors along rows; clamps cosine to [-1, 1]
    for numerical stability.
    """
    cos_thetas = torch.einsum("ij,ij->i", V1, V2)
    cos_thetas = torch.clamp(cos_thetas, -1.0, 1.0)
    thetas = torch.arccos(cos_thetas)
    return cos_thetas, thetas


def principal_angles(V1: Tensor, V2: Tensor) -> Tensor:
    """Principal angles (radians) between subspaces spanned by columns of V1 and V2. Only works for unit length vectors"""
    nt, _, m1 = V1.shape
    _, _, m2 = V2.shape
    dim = min(m1, m2)
    theta = torch.empty((nt, dim), dtype=V1.dtype, device=V1.device)
    for i in tqdm(range(nt), leave=False):
        Q1, _ = torch.linalg.qr(V1[i], mode="reduced")
        Q2, _ = torch.linalg.qr(V2[i], mode="reduced")
        singular_values = torch.linalg.svdvals(Q1.transpose(-2, -1) @ Q2)
        singular_values = torch.clamp(singular_values, -1.0, 1.0)
        theta[i] = torch.arccos(singular_values[:dim])
    return theta


__all__ = [
    "compute_angles",
    "principal_angles",
]
