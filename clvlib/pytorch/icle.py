import torch
from typing import Callable
from tqdm.auto import tqdm

Tensor = torch.Tensor


def compute_ICLE(
    jacobian_function: Callable,
    trajectory: Tensor,
    time: Tensor,
    CLV_history: Tensor,
    *args,
    k_step: int = 1,
) -> Tensor:
    """Compute instantaneous covariant Lyapunov exponents (ICLEs)."""
    if not isinstance(k_step, int):
        raise TypeError("k_step must be an integer.")
    if k_step < 1:
        raise ValueError("k_step must be at least 1.")
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    if trajectory.ndim != 2:
        raise ValueError("trajectory must be two-dimensional.")
    if CLV_history.ndim != 3:
        raise ValueError("CLV_history must be three-dimensional.")

    n_time, n_state = trajectory.shape
    n_samples, n_clv_state, m = CLV_history.shape
    if n_state != n_clv_state:
        raise ValueError(
            "trajectory and CLV_history must share the same state dimension."
        )
    if n_time != time.shape[0]:
        raise ValueError("trajectory  and time must share the same number of samples.")
    if n_samples == 0:
        raise ValueError("CLV_history must contain at least one time sample.")

    sample_indices = torch.arange(
        0, k_step * n_samples, k_step, dtype=torch.long, device=time.device
    )
    if sample_indices.shape[0] != n_samples:
        raise RuntimeError("Unexpected number of samples inferred from CLV_history.")
    if sample_indices[-1].item() >= n_time:
        raise ValueError(
            "CLV history length is incompatible with the provided trajectory/time for this k_step."
        )

    trajectory_indices = sample_indices.to(trajectory.device)
    states = trajectory.index_select(0, trajectory_indices)
    times = time.index_select(0, sample_indices)

    return _compute_icle_series(jacobian_function, states, times, CLV_history, *args)


def _compute_icle_series(
    jacobian_function: Callable,
    sampled_states: Tensor,
    sampled_times: Tensor,
    CLV_history: Tensor,
    *args,
) -> Tensor:
    J_history = _compute_jacobian_time_history(
        jacobian_function, sampled_states, sampled_times, *args
    )
    return torch.einsum("tik,tij,tjk->tk", CLV_history, J_history, CLV_history)


def _compute_jacobian_time_history(
    jacobian_function: Callable,
    sampled_states: Tensor,
    sampled_times: Tensor,
    *args,
) -> Tensor:
    n_samples, n_state = sampled_states.shape
    J_history = torch.empty(
        (n_samples, n_state, n_state),
        dtype=sampled_states.dtype,
        device=sampled_states.device,
    )
    for idx in tqdm(range(n_samples), leave=False):
        J_history[idx] = jacobian_function(
            sampled_times[idx], sampled_states[idx], *args
        )
    return J_history


__all__ = [
    "compute_ICLE",
]
