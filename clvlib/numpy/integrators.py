import numpy as np
from typing import Callable, Tuple
from .steppers import VariationalStepper


def _qr_mgs(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """QR decomposition using Modified Gram–Schmidt."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()

    for j in range(n):
        norm = np.sqrt(np.sum(V[:, j] * V[:, j]))
        Q[:, j] = V[:, j] / norm
        R[j, j] = norm
        for k in range(j + 1, n):
            R[j, k] = np.dot(Q[:, j], V[:, k])
            V[:, k] = V[:, k] - R[j, k] * Q[:, j]

    return Q, R


def _lyap_int(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    stepper: VariationalStepper,
    *args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    nt = len(t)
    n = trajectory.shape[0]

    Q_history = np.empty((n, n, nt))
    R_history = np.empty((n, n, nt))
    LE_history = np.empty((n, nt))

    Q = np.eye(n)
    Q_history[:, :, 0] = Q
    R_history[:, :, 0] = np.eye(n)
    LE_history[:, 0] = 0.0
    log_sums = 0.0

    for i in range(nt - 1):
        _, Q = stepper(f, Df, t[i], trajectory[:, i], Q, dt, *args)
        Q, R = np.linalg.qr(Q)
        Q_history[:, :, i + 1] = Q
        R_history[:, :, i + 1] = R
        log_sums += np.log(np.abs(np.diag(R)))
        LE_history[:, i + 1] = log_sums / ((i + 1) * dt)

    return LE_history[:, -1], LE_history, Q_history, R_history


def _lyap_int_k_step(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    k_step: int,
    stepper: VariationalStepper,
    *args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    nt = t.size
    n = trajectory.shape[0]
    n_step = ((nt - 1) // k_step) + 1

    Q_history = np.empty((n, n, n_step), dtype=float)
    R_history = np.empty((n, n, n_step), dtype=float)
    LE_history = np.empty((n, n_step), dtype=float)

    Q = np.eye(n, dtype=float)
    log_sums = np.zeros(n, dtype=float)

    Q_history[:, :, 0] = Q
    R_history[:, :, 0] = np.eye(n, dtype=float)
    LE_history[:, 0] = 0.0

    j = 0
    for i in range(nt - 1):
        _, Q = stepper(f, Df, t[i], trajectory[:, i], Q, dt, *args)
        if (i + 1) % k_step == 0:
            Q, R = np.linalg.qr(Q)
            Q_history[:, :, j + 1] = Q
            R_history[:, :, j + 1] = R
            log_sums += np.log(np.abs(np.diag(R)))
            LE_history[:, j + 1] = log_sums / ((j + 1) * k_step * dt)
            j += 1

    return LE_history[:, -1], LE_history, Q_history, R_history


def run_variational_integrator(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
    stepper: VariationalStepper = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if stepper is None:
        raise ValueError("stepper must be provided (use steppers.resolve_stepper)")
    if k_step > 1:
        return _lyap_int_k_step(f, Df, trajectory, t, k_step, stepper, *args)
    return _lyap_int(f, Df, trajectory, t, stepper, *args)


__all__ = [
    "run_variational_integrator",
    "_qr_mgs",
]

