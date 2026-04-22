import numpy as np
from typing import Callable, Tuple, Union
import scipy.linalg
from tqdm.auto import tqdm
from .steppers import VariationalStepper


def _qr(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Economic mode keeps the number of columns equal to the input, which is
    # required when computing only a subset of Lyapunov vectors.
    Q, R = scipy.linalg.qr(
        Q, overwrite_a=True, mode="economic", check_finite=False
    )
    s = np.sign(np.diag(R))
    s[s == 0.0] = 1.0
    Q = Q * s[np.newaxis, :]
    R = s[:, np.newaxis] * R
    return Q, R


def _resolve_n_lyap(n_lyap: Union[int, None], n: int) -> int:
    if n_lyap is None:
        return n
    if not isinstance(n_lyap, int):
        raise TypeError("n_lyap must be an integer or None.")
    if n_lyap < 1:
        raise ValueError("n_lyap must be at least 1.")
    if n_lyap > n:
        raise ValueError(f"n_lyap ({n_lyap}) cannot exceed system dimension ({n}).")
    return n_lyap


def _lyap_int(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    stepper: VariationalStepper,
    *args,
    n_lyap: Union[int, None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    nt = t.size
    n = trajectory.shape[1]
    m = _resolve_n_lyap(n_lyap, n)

    # Time-first histories: (nt, n, n) and (nt, n)
    Q_history = np.empty((nt, n, m), dtype=float)
    R_history = np.empty((nt, m, m), dtype=float)
    LE_history = np.empty((nt, m), dtype=float)

    Q = np.eye(n, m, dtype=float)
    Q_history[0] = Q
    R_history[0] = np.eye(m, dtype=float)
    LE_history[0] = 0.0
    log_sums = np.zeros(m, dtype=float)

    for i in tqdm(range(nt - 1), leave=False):
        _, Q = stepper(f, Df, t[i], trajectory[i], Q, dt, *args)
        Q, R = _qr(Q)
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
    n_lyap: Union[int, None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    nt = t.size
    n = trajectory.shape[1]
    m = _resolve_n_lyap(n_lyap, n)
    n_step = ((nt - 1) // k_step) + 1

    # Time-first histories with k-step sampling: (n_step, n, n) and (n_step, n)
    Q_history = np.empty((n_step, n, m), dtype=float)
    R_history = np.empty((n_step, m, m), dtype=float)
    LE_history = np.empty((n_step, m), dtype=float)

    Q = np.eye(n, m, dtype=float)
    Q_history[0] = Q
    R_history[0] = np.eye(m, dtype=float)
    LE_history[0] = 0.0
    log_sums = np.zeros(m, dtype=float)

    j = 0
    for i in tqdm(range(nt - 1), leave=False):
        _, Q = stepper(f, Df, t[i], trajectory[i], Q, dt, *args)
        if (i + 1) % k_step == 0:
            Q, R = _qr(Q)
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
    n_lyap: Union[int, None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate state and variational system from an initial condition.

    Returns (LE_final, LE_history, Q_history, R_history, trajectory).
    """
    dt = t[1] - t[0]
    nt = t.size
    n = x0.size
    m = _resolve_n_lyap(n_lyap, n)

    trajectory = np.empty((nt, n), dtype=float)
    trajectory[0] = x0

    Q_history = np.empty((nt, n, m), dtype=float)
    R_history = np.empty((nt, m, m), dtype=float)
    LE_history = np.empty((nt, m), dtype=float)

    Q = np.eye(n, m, dtype=float)
    x = x0.astype(float, copy=True)

    Q_history[0] = Q
    R_history[0] = np.eye(m, dtype=float)
    LE_history[0] = 0.0
    log_sums = np.zeros(m, dtype=float)

    for i in tqdm(range(nt - 1), leave=False):
        x, Q = stepper(f, Df, t[i], x, Q_history[i], dt, *args)
        trajectory[i + 1] = x
        Q, R = _qr(Q)
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
    n_lyap: Union[int, None],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """k-step integration from an initial condition.

    Returns (LE_final, LE_history, Q_history, R_history, trajectory).
    """
    dt = t[1] - t[0]
    nt = t.size
    n = x0.size
    m = _resolve_n_lyap(n_lyap, n)
    n_step = ((nt - 1) // k_step) + 1

    trajectory = np.empty((nt, n), dtype=float)
    trajectory[0] = x0

    Q_history = np.empty((n_step, n, m), dtype=float)
    R_history = np.empty((n_step, m, m), dtype=float)
    LE_history = np.empty((n_step, m), dtype=float)

    Q = np.eye(n, m, dtype=float)
    x = x0.astype(float, copy=True)

    Q_history[0] = Q
    R_history[0] = np.eye(m, dtype=float)
    LE_history[0] = 0.0
    log_sums = np.zeros(m, dtype=float)

    j = 0
    for i in tqdm(range(nt - 1), leave=False):
        x, Q = stepper(f, Df, t[i], x, Q_history[i], dt, *args)
        trajectory[i + 1] = x
        if (i + 1) % k_step == 0:
            Q, R = _qr(Q)
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
    stepper: VariationalStepper,
    n_lyap: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate variational equations along a provided trajectory.

    Returns (LE_final, LE_history, Q_history, R_history).
    """
    if k_step > 1:
        return _lyap_int_k_step(
            f,
            Df,
            trajectory,
            t,
            k_step,
            stepper,
            *args,
            n_lyap=n_lyap,
        )
    return _lyap_int(f, Df, trajectory, t, stepper, *args, n_lyap=n_lyap)


def run_state_variational_integrator(
    f: Callable,
    Df: Callable,
    x0: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
    stepper: VariationalStepper,
    n_lyap: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate state and variational equations starting from ``x0``.

    Returns (LE_final, LE_history, Q_history, R_history, trajectory).
    """
    if k_step > 1:
        return _lyap_int_k_step_from_x0(
            f,
            Df,
            x0,
            t,
            k_step,
            stepper,
            *args,
            n_lyap=n_lyap,
        )
    return _lyap_int_from_x0(f, Df, x0, t, stepper, *args, n_lyap=n_lyap)


__all__ = [
    "run_variational_integrator",
    "run_state_variational_integrator",
]
