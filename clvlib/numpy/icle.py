import numpy as np
from typing import Callable


def compute_ICLE(
    jacobian_function: Callable,
    solution: np.ndarray,
    time: np.ndarray,
    CLV_history: np.ndarray,
    *args,
    k_step: int = 1,
) -> np.ndarray:
    """Compute instantaneous covariant Lyapunov exponents (ICLEs)."""
    if not isinstance(k_step, int):
        raise TypeError("k_step must be an integer.")
    if k_step < 1:
        raise ValueError("k_step must be at least 1.")
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    if solution.ndim != 2:
        raise ValueError("solution must be two-dimensional.")
    if CLV_history.ndim != 3:
        raise ValueError("CLV_history must be three-dimensional.")

    n_state, n_time = solution.shape
    n_clv_state, m, n_samples = CLV_history.shape
    if n_state != n_clv_state:
        raise ValueError("solution and CLV_history must share the same state dimension.")
    if n_time != time.size:
        raise ValueError("solution and time must share the same number of samples.")
    if n_samples == 0:
        raise ValueError("CLV_history must contain at least one time sample.")

    sample_indices = np.arange(0, k_step * n_samples, k_step, dtype=int)
    if sample_indices.size != n_samples:
        raise RuntimeError("Unexpected number of samples inferred from CLV_history.")
    if sample_indices[-1] >= n_time:
        raise ValueError(
            "CLV history length is incompatible with the provided solution/time for this k_step."
        )

    states = solution[:, sample_indices]
    times = time[sample_indices]

    return _compute_icle_series(jacobian_function, states, times, CLV_history, *args)


def _compute_icle_series(
    jacobian_function: Callable,
    sampled_states: np.ndarray,
    sampled_times: np.ndarray,
    CLV_history: np.ndarray,
    *args,
) -> np.ndarray:
    J_history = _compute_jacobian_time_history(jacobian_function, sampled_states, sampled_times, *args)
    ICLE = np.einsum('ikt,ijt,jkt->kt', CLV_history, J_history, CLV_history)
    return ICLE


def _compute_jacobian_time_history(
    jacobian_function: Callable,
    sampled_states: np.ndarray,
    sampled_times: np.ndarray,
    *args,
) -> np.ndarray:
    n_state, n_samples = sampled_states.shape
    J_history = np.empty((n_state, n_state, n_samples), dtype=float)
    for idx in range(n_samples):
        J_history[:, :, idx] = jacobian_function(sampled_times[idx], sampled_states[:, idx], *args)
    return J_history


__all__ = [
    "compute_ICLE",
]

