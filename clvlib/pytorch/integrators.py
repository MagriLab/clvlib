import torch
from tqdm.auto import tqdm
from typing import Callable, Tuple, Union

from .steppers import VariationalStepper

Tensor = torch.Tensor


def gram_schmidt_qr(A: torch.Tensor):
    """
    Performs QR decomposition using Classical Gram-Schmidt orthogonalization.

    Args:
        A (torch.Tensor): Input matrix of shape (m, n)

    Returns:
        Q (torch.Tensor): Orthonormal matrix of shape (m, n)
        R (torch.Tensor): Upper triangular matrix of shape (n, n)
    """
    m, n = A.shape
    Q = torch.zeros((m, n), dtype=A.dtype, device=A.device)
    R = torch.zeros((n, n), dtype=A.dtype, device=A.device)

    for j in range(n):
        v = A[:, j].clone()
        for i in range(j):
            R[i, j] = torch.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = torch.norm(v, p=2)
        Q[:, j] = v / R[j, j]

    return Q, R


def _qr_householder(Q: Tensor) -> Tuple[Tensor, Tensor]:
    return torch.linalg.qr(Q, mode="reduced")


def _qr_gram_schmidt(Q: Tensor) -> Tuple[Tensor, Tensor]:
    return gram_schmidt_qr(Q)


QRSolver = Callable[[Tensor], Tuple[Tensor, Tensor]]


_QR_METHODS = {
    "householder": _qr_householder,
    "gs": _qr_gram_schmidt,
    "gram-schmidt": _qr_gram_schmidt,
    "gram_schmidt": _qr_gram_schmidt,
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
    trajectory: Tensor,
    t: Tensor,
    stepper: VariationalStepper,
    *args,
    n_lyap: Union[int, None],
    qr_solver: QRSolver,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    dt = float((t[1] - t[0]).item())
    nt = t.numel()
    n = trajectory.shape[1]
    m = _resolve_n_lyap(n_lyap, n)
    dtype = trajectory.dtype
    device = trajectory.device

    Q_history = torch.empty((nt, n, m), dtype=dtype, device=device)
    R_history = torch.empty((nt, m, m), dtype=dtype, device=device)
    LE_history = torch.empty((nt, m), dtype=dtype, device=device)

    Q = torch.eye(n, m, dtype=dtype, device=device)
    Q_history[0] = Q
    R_history[0] = torch.eye(m, dtype=dtype, device=device)
    LE_history[0] = torch.zeros(m, dtype=dtype, device=device)
    log_sums = torch.zeros(m, dtype=dtype, device=device)

    for i in tqdm(range(nt - 1), leave=False):
        _, Q = stepper(f, Df, float(t[i].item()), trajectory[i], Q, dt, *args)
        Q, R = qr_solver(Q)
        Q_history[i + 1] = Q
        R_history[i + 1] = R
        log_sums = log_sums + torch.log(torch.abs(torch.diagonal(R)))
        LE_history[i + 1] = log_sums / ((i + 1) * dt)

    return LE_history[-1], LE_history, Q_history, R_history


def _lyap_int_k_step(
    f: Callable,
    Df: Callable,
    trajectory: Tensor,
    t: Tensor,
    k_step: int,
    stepper: VariationalStepper,
    *args,
    n_lyap: Union[int, None],
    qr_solver: QRSolver,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    dt = float((t[1] - t[0]).item())
    nt = t.numel()
    n = trajectory.shape[1]
    m = _resolve_n_lyap(n_lyap, n)
    dtype = trajectory.dtype
    device = trajectory.device
    n_step = ((nt - 1) // k_step) + 1

    Q_history = torch.empty((n_step, n, m), dtype=dtype, device=device)
    R_history = torch.empty((n_step, m, m), dtype=dtype, device=device)
    LE_history = torch.empty((n_step, m), dtype=dtype, device=device)

    Q = torch.eye(n, m, dtype=dtype, device=device)
    log_sums = torch.zeros(m, dtype=dtype, device=device)

    Q_history[0] = Q
    R_history[0] = torch.eye(m, dtype=dtype, device=device)
    LE_history[0] = torch.zeros(m, dtype=dtype, device=device)

    j = 0
    for i in tqdm(range(nt - 1), leave=False):
        _, Q = stepper(f, Df, float(t[i].item()), trajectory[i], Q, dt, *args)
        if (i + 1) % k_step == 0:
            Q, R = qr_solver(Q)
            Q_history[j + 1] = Q
            R_history[j + 1] = R
            log_sums = log_sums + torch.log(torch.abs(torch.diagonal(R)))
            LE_history[j + 1] = log_sums / ((j + 1) * k_step * dt)
            j += 1

    return LE_history[-1], LE_history, Q_history, R_history


def _lyap_int_from_x0(
    f: Callable,
    Df: Callable,
    x0: Tensor,
    t: Tensor,
    stepper: VariationalStepper,
    *args,
    n_lyap: Union[int, None],
    qr_solver: QRSolver,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dt = float((t[1] - t[0]).item())
    nt = t.numel()
    n = x0.numel()
    m = _resolve_n_lyap(n_lyap, n)
    dtype = x0.dtype
    device = x0.device

    trajectory = torch.empty((nt, n), dtype=dtype, device=device)
    trajectory[0] = x0

    Q_history = torch.empty((nt, n, m), dtype=dtype, device=device)
    R_history = torch.empty((nt, m, m), dtype=dtype, device=device)
    LE_history = torch.empty((nt, m), dtype=dtype, device=device)

    Q = torch.eye(n, m, dtype=dtype, device=device)
    x = x0.clone()

    Q_history[0] = Q
    R_history[0] = torch.eye(m, dtype=dtype, device=device)
    LE_history[0] = torch.zeros(m, dtype=dtype, device=device)
    log_sums = torch.zeros(m, dtype=dtype, device=device)

    for i in tqdm(range(nt - 1), leave=False):
        x, Q = stepper(f, Df, float(t[i].item()), x, Q, dt, *args)
        trajectory[i + 1] = x
        Q, R = qr_solver(Q)
        Q_history[i + 1] = Q
        R_history[i + 1] = R
        log_sums = log_sums + torch.log(torch.abs(torch.diagonal(R)))
        LE_history[i + 1] = log_sums / ((i + 1) * dt)

    return LE_history[-1], LE_history, Q_history, R_history, trajectory


def _lyap_int_k_step_from_x0(
    f: Callable,
    Df: Callable,
    x0: Tensor,
    t: Tensor,
    k_step: int,
    stepper: VariationalStepper,
    *args,
    n_lyap: Union[int, None],
    qr_solver: QRSolver,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dt = float((t[1] - t[0]).item())
    nt = t.numel()
    n = x0.numel()
    m = _resolve_n_lyap(n_lyap, n)
    dtype = x0.dtype
    device = x0.device
    n_step = ((nt - 1) // k_step) + 1

    trajectory = torch.empty((nt, n), dtype=dtype, device=device)
    trajectory[0] = x0

    Q_history = torch.empty((n_step, n, m), dtype=dtype, device=device)
    R_history = torch.empty((n_step, m, m), dtype=dtype, device=device)
    LE_history = torch.empty((n_step, m), dtype=dtype, device=device)

    Q = torch.eye(n, m, dtype=dtype, device=device)
    x = x0.clone()
    log_sums = torch.zeros(m, dtype=dtype, device=device)

    Q_history[0] = Q
    R_history[0] = torch.eye(m, dtype=dtype, device=device)
    LE_history[0] = torch.zeros(m, dtype=dtype, device=device)

    j = 0
    for i in tqdm(range(nt - 1), leave=False):
        x, Q = stepper(f, Df, float(t[i].item()), x, Q, dt, *args)
        trajectory[i + 1] = x
        if (i + 1) % k_step == 0:
            Q, R = qr_solver(Q)
            Q_history[j + 1] = Q
            R_history[j + 1] = R
            log_sums = log_sums + torch.log(torch.abs(torch.diagonal(R)))
            LE_history[j + 1] = log_sums / ((j + 1) * k_step * dt)
            j += 1

    return LE_history[-1], LE_history, Q_history, R_history, trajectory


def run_variational_integrator(
    f: Callable,
    Df: Callable,
    trajectory: Tensor,
    t: Tensor,
    *args,
    k_step: int = 1,
    stepper: VariationalStepper,
    n_lyap: Union[int, None] = None,
    qr_method: Union[str, QRSolver] = "householder",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
    x0: Tensor,
    t: Tensor,
    *args,
    k_step: int = 1,
    stepper: VariationalStepper,
    n_lyap: Union[int, None] = None,
    qr_method: Union[str, QRSolver] = "householder",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    "gram_schmidt_qr",
]
