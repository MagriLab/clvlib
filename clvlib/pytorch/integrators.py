import torch
from tqdm.auto import tqdm
from typing import Callable, Tuple, Union

from .steppers import VariationalStepper

Tensor = torch.Tensor


def _qr(Q: Tensor) -> Tuple[Tensor, Tensor]:
    Q, R = torch.linalg.qr(Q, mode="reduced")
    s = torch.sign(torch.diagonal(R))
    s = torch.where(s == 0, torch.ones_like(s), s)
    Q = Q * s.unsqueeze(0)
    R = s.unsqueeze(1) * R
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
    trajectory: Tensor,
    t: Tensor,
    stepper: VariationalStepper,
    *args,
    n_lyap: Union[int, None],
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
        Q, R = _qr(Q)
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
            Q, R = _qr(Q)
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
        Q, R = _qr(Q)
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
            Q, R = _qr(Q)
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
    x0: Tensor,
    t: Tensor,
    *args,
    k_step: int = 1,
    stepper: VariationalStepper,
    n_lyap: Union[int, None] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
