import numpy as np
from typing import Callable, Tuple

from numba import njit
import scipy.linalg

from .steppers import VariationalStepper


@njit(fastmath=True)
def gram_schmidt_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        # v = A[:, j].copy()
        v = np.empty(m, dtype=np.float64)
        for r in range(m):
            v[r] = A[r, j]

        for i in range(j):
            # R[i, j] = dot(Q[:, i], A[:, j])
            s = 0.0
            for k in range(m):
                s += Q[k, i] * A[k, j]
            R[i, j] = s

            # v -= R[i, j] * Q[:, i]
            c = R[i, j]
            for k in range(m):
                v[k] -= c * Q[k, i]

        # R[j, j] = norm(v)
        s2 = 0.0
        for k in range(m):
            s2 += v[k] * v[k]
        Rjj = np.sqrt(s2)
        R[j, j] = Rjj

        # Q[:, j] = v / R[j, j]
        inv = 1.0 / Rjj
        for k in range(m):
            Q[k, j] = v[k] * inv

    return Q, R


def _compute_qr(Q: np.ndarray, qr_method: str) -> Tuple[np.ndarray, np.ndarray]:
    method = qr_method.lower()
    if method in {"householder", "scipy", "qr"}:
        return scipy.linalg.qr(Q, overwrite_a=True, mode="full", check_finite=False)
    if method in {"gs", "gram-schmidt", "gram_schmidt", "numba"}:
        return gram_schmidt_qr(np.ascontiguousarray(Q, dtype=np.float64))
    available = "householder, gs"
    raise ValueError(f"Unknown qr_method '{qr_method}'. Available: {available}.")


def _lyap_int(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    stepper: VariationalStepper,
    *args,
    qr_method: str = "householder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    nt = t.size
    n = trajectory.shape[1]

    # Time-first histories: (nt, n, n) and (nt, n)
    Q_history = np.empty((nt, n, n), dtype=float)
    R_history = np.empty((nt, n, n), dtype=float)
    LE_history = np.empty((nt, n), dtype=float)

    Q = np.eye(n, dtype=float)
    Q_history[0] = Q
    R_history[0] = np.eye(n, dtype=float)
    LE_history[0] = 0.0
    log_sums = np.zeros(n, dtype=float)

    for i in range(nt - 1):
        _, Q = stepper(f, Df, t[i], trajectory[i], Q, dt, *args)
        Q, R = _compute_qr(Q, qr_method)
        Q_history[i + 1] = Q
        R_history[i + 1] = R
        log_sums += np.log(np.abs(np.diag(R)))
        LE_history[i + 1] = log_sums / ((i + 1) * dt)

    return LE_history[-1], LE_history, Q_history, R_history


def _lyap_int_k_step(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    k_step: int,
    stepper: VariationalStepper,
    *args,
    qr_method: str = "householder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    nt = t.size
    n = trajectory.shape[1]
    n_step = ((nt - 1) // k_step) + 1

    # Time-first histories with k-step sampling: (n_step, n, n) and (n_step, n)
    Q_history = np.empty((n_step, n, n), dtype=float)
    R_history = np.empty((n_step, n, n), dtype=float)
    LE_history = np.empty((n_step, n), dtype=float)

    Q = np.eye(n, dtype=float)
    log_sums = np.zeros(n, dtype=float)

    Q_history[0] = Q
    R_history[0] = np.eye(n, dtype=float)
    LE_history[0] = 0.0

    j = 0
    for i in range(nt - 1):
        _, Q = stepper(f, Df, t[i], trajectory[i], Q, dt, *args)
        if (i + 1) % k_step == 0:
            Q, R = _compute_qr(Q, qr_method)
            Q_history[j + 1] = Q
            R_history[j + 1] = R
            log_sums += np.log(np.abs(np.diag(R)))
            LE_history[j + 1] = log_sums / ((j + 1) * k_step * dt)
            j += 1

    return LE_history[-1], LE_history, Q_history, R_history


def _lyap_int_from_x0(
    f: Callable,
    Df: Callable,
    x0: np.ndarray,
    t: np.ndarray,
    stepper: VariationalStepper,
    *args,
    qr_method: str = "householder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate state and variational system from an initial condition.

    Returns (LE_final, LE_history, Q_history, R_history, trajectory).
    """
    dt = t[1] - t[0]
    nt = t.size
    n = x0.size

    trajectory = np.empty((nt, n), dtype=float)
    trajectory[0] = x0

    Q_history = np.empty((nt, n, n), dtype=float)
    R_history = np.empty((nt, n, n), dtype=float)
    LE_history = np.empty((nt, n), dtype=float)

    Q = np.eye(n, dtype=float)
    x = x0.astype(float, copy=True)

    Q_history[0] = Q
    R_history[0] = np.eye(n, dtype=float)
    LE_history[0] = 0.0
    log_sums = np.zeros(n, dtype=float)

    for i in range(nt - 1):
        x, Q = stepper(f, Df, t[i], x, Q, dt, *args)
        trajectory[i + 1] = x
        Q, R = _compute_qr(Q, qr_method)
        Q_history[i + 1] = Q
        R_history[i + 1] = R
        log_sums += np.log(np.abs(np.diag(R)))
        LE_history[i + 1] = log_sums / ((i + 1) * dt)

    return LE_history[-1], LE_history, Q_history, R_history, trajectory


def _lyap_int_k_step_from_x0(
    f: Callable,
    Df: Callable,
    x0: np.ndarray,
    t: np.ndarray,
    k_step: int,
    stepper: VariationalStepper,
    *args,
    qr_method: str = "householder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """k-step integration from an initial condition.

    Returns (LE_final, LE_history, Q_history, R_history, trajectory).
    """
    dt = t[1] - t[0]
    nt = t.size
    n = x0.size
    n_step = ((nt - 1) // k_step) + 1

    trajectory = np.empty((nt, n), dtype=float)
    trajectory[0] = x0

    Q_history = np.empty((n_step, n, n), dtype=float)
    R_history = np.empty((n_step, n, n), dtype=float)
    LE_history = np.empty((n_step, n), dtype=float)

    Q = np.eye(n, dtype=float)
    x = x0.astype(float, copy=True)
    log_sums = np.zeros(n, dtype=float)

    Q_history[0] = Q
    R_history[0] = np.eye(n, dtype=float)
    LE_history[0] = 0.0

    j = 0
    for i in range(nt - 1):
        x, Q = stepper(f, Df, t[i], x, Q, dt, *args)
        trajectory[i + 1] = x
        if (i + 1) % k_step == 0:
            Q, R = _compute_qr(Q, qr_method)
            Q_history[j + 1] = Q
            R_history[j + 1] = R
            log_sums += np.log(np.abs(np.diag(R)))
            LE_history[j + 1] = log_sums / ((j + 1) * k_step * dt)
            j += 1

    return LE_history[-1], LE_history, Q_history, R_history, trajectory


def run_variational_integrator(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
    stepper: VariationalStepper = None,
    qr_method: str = "householder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if stepper is None:
        raise ValueError("stepper must be provided (use steppers.resolve_stepper)")
    if k_step > 1:
        return _lyap_int_k_step(
            f, Df, trajectory, t, k_step, stepper, *args, qr_method=qr_method
        )
    return _lyap_int(f, Df, trajectory, t, stepper, *args, qr_method=qr_method)


def run_state_variational_integrator(
    f: Callable,
    Df: Callable,
    x0: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
    stepper: VariationalStepper = None,
    qr_method: str = "householder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate state and variational equations starting from ``x0``.

    Returns (LE_final, LE_history, Q_history, R_history, trajectory).
    """
    if stepper is None:
        raise ValueError("stepper must be provided (use steppers.resolve_stepper)")
    if k_step > 1:
        return _lyap_int_k_step_from_x0(
            f, Df, x0, t, k_step, stepper, *args, qr_method=qr_method
        )
    return _lyap_int_from_x0(
        f, Df, x0, t, stepper, *args, qr_method=qr_method
    )


__all__ = [
    "run_variational_integrator",
    "run_state_variational_integrator",
]
