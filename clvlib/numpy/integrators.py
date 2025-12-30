import numpy as np
from typing import Callable, Tuple, Union
import scipy.linalg
from numba import njit
from tqdm.auto import tqdm
from .steppers import VariationalStepper


QRSolver = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@njit
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


def _qr_householder(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Economic mode keeps the number of columns equal to the input, which is
    # required when computing only a subset of Lyapunov vectors.
    return scipy.linalg.qr(
        Q, overwrite_a=True, mode="economic", check_finite=False
    )


def _qr_numba(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return gram_schmidt_qr(np.ascontiguousarray(Q, dtype=np.float64))


_QR_METHODS = {
    "householder": _qr_householder,
    "gs": _qr_numba,
    "gram-schmidt": _qr_numba,
    "gram_schmidt": _qr_numba,
}


def _resolve_qr_method(qr_method: Union[str, QRSolver]) -> QRSolver:
    if callable(qr_method):
        return qr_method
    method_key = qr_method.lower()
    try:
        return _QR_METHODS[method_key]
    except KeyError as exc:
        available = ", ".join(sorted(_QR_METHODS))
        raise ValueError(
            f"Unknown qr_method '{qr_method}'. Available: {available}."
        ) from exc


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
    qr_solver: QRSolver,
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
        Q, R = qr_solver(Q)
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
    qr_solver: QRSolver,
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
            Q, R = qr_solver(Q)
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
    qr_solver: QRSolver,
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
        Q, R = qr_solver(Q)
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
    qr_solver: QRSolver,
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
            Q, R = qr_solver(Q)
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
    qr_method: Union[str, QRSolver] = "householder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate variational equations along a provided trajectory.

    Returns (LE_final, LE_history, Q_history, R_history).
    """
    qr_solver = _resolve_qr_method(qr_method)
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
            qr_solver=qr_solver,
        )
    return _lyap_int(
        f, Df, trajectory, t, stepper, *args, n_lyap=n_lyap, qr_solver=qr_solver
    )


def run_state_variational_integrator(
    f: Callable,
    Df: Callable,
    x0: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
    stepper: VariationalStepper,
    n_lyap: Union[int, None] = None,
    qr_method: Union[str, QRSolver] = "householder",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate state and variational equations starting from ``x0``.

    Returns (LE_final, LE_history, Q_history, R_history, trajectory).
    """
    qr_solver = _resolve_qr_method(qr_method)
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
            qr_solver=qr_solver,
        )
    return _lyap_int_from_x0(
        f, Df, x0, t, stepper, *args, n_lyap=n_lyap, qr_solver=qr_solver
    )


__all__ = [
    "run_variational_integrator",
    "run_state_variational_integrator",
]
