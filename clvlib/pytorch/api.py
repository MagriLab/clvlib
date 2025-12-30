import torch
from typing import Callable, Tuple, Union

from .steppers import VariationalStepper, resolve_stepper
from .integrators import run_variational_integrator, run_state_variational_integrator
from .clv import _clvs

Tensor = torch.Tensor


def lyap_analysis(
    f: Callable,
    Df: Callable,
    trajectory: Tensor,
    t: Tensor,
    *args,
    k_step: int = 1,
    stepper: Union[str, VariationalStepper, None] = "rk4",
    n_lyap: Union[int, None] = None,
    qr_method: str = "householder",
    ginelli_method: str = "ginelli",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Run Lyapunov-exponent integration and compute the associated CLVs.
    Set `k_step` > 1 to use the k-step variational integrator.
    Use `n_lyap` to restrict the number of Lyapunov exponents/vectors.
    Returns (LE, LE_history, BLV_history, CLV_history).
    """
    n, _ = _validate_lyap_inputs(f, Df, trajectory, t, k_step)
    n_vec = _resolve_n_lyap(n_lyap, n)

    step = resolve_stepper(stepper)
    LE, LE_history, BLV_history, CLV_history = _compute_lyap_outputs(
        f,
        Df,
        trajectory,
        t,
        *args,
        k_step=k_step,
        stepper=step,
        n_lyap=n_vec,
        qr_method=qr_method,
        ginelli_method=ginelli_method,
    )

    expected_time_samples = BLV_history.shape[0]
    if CLV_history.shape != (expected_time_samples, n, n_vec):
        raise RuntimeError(
            "CLV history has inconsistent shape: "
            f"expected {(expected_time_samples, n, n_vec)}, got {CLV_history.shape}."
        )

    return LE, LE_history, BLV_history, CLV_history


def lyap_exp(
    f: Callable,
    Df: Callable,
    trajectory: Tensor,
    t: Tensor,
    *args,
    k_step: int = 1,
    stepper: Union[str, VariationalStepper, None] = "rk4",
    return_blv: bool = False,
    n_lyap: Union[int, None] = None,
    qr_method: str = "householder",
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """
    Run Lyapunov-exponent integration without computing CLVs.
    Returns (LE, LE_history[, BLV_history]).
    """
    n, _ = _validate_lyap_inputs(f, Df, trajectory, t, k_step)
    n_vec = _resolve_n_lyap(n_lyap, n)

    step = resolve_stepper(stepper)
    LE, LE_history, BLV_history, _ = run_variational_integrator(
        f,
        Df,
        trajectory,
        t,
        *args,
        k_step=k_step,
        stepper=step,
        n_lyap=n_vec,
        qr_method=qr_method,
    )

    if return_blv:
        return LE, LE_history, BLV_history

    return LE, LE_history


def lyap_analysis_from_ic(
    f: Callable,
    Df: Callable,
    x0: Tensor,
    t: Tensor,
    *args,
    k_step: int = 1,
    stepper: Union[str, VariationalStepper, None] = "rk4",
    n_lyap: Union[int, None] = None,
    qr_method: str = "householder",
    ginelli_method: str = "ginelli",
) -> Union[
    Tuple[Tensor, Tensor, Tensor, Tensor],
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
]:
    """
    Run Lyapunov-exponent integration and compute CLVs starting from an initial condition.
    Returns (LE, LE_history, BLV_history, CLV_history, trajectory).
    """
    _validate_lyap_ic_inputs(f, Df, x0, t, k_step)
    n_vec = _resolve_n_lyap(n_lyap, x0.numel())

    step = resolve_stepper(stepper)
    LE, LE_history, BLV_history, R_history, trajectory = (
        run_state_variational_integrator(
            f,
            Df,
            x0,
            t,
            *args,
            k_step=k_step,
            stepper=step,
            n_lyap=n_vec,
            qr_method=qr_method,
        )
    )
    CLV_history = _clvs(BLV_history, R_history, ginelli_method=ginelli_method)
    return LE, LE_history, BLV_history, CLV_history, trajectory


def lyap_exp_from_ic(
    f: Callable,
    Df: Callable,
    x0: Tensor,
    t: Tensor,
    *args,
    k_step: int = 1,
    stepper: Union[str, VariationalStepper, None] = "rk4",
    return_blv: bool = False,
    n_lyap: Union[int, None] = None,
    qr_method: str = "householder",
) -> Union[
    Tuple[Tensor, Tensor],
    Tuple[Tensor, Tensor, Tensor],
    Tuple[Tensor, Tensor, Tensor, Tensor],
]:
    """
    Run Lyapunov-exponent integration from an initial condition.
    Returns (LE, LE_history[, BLV_history], trajectory).
    """
    _validate_lyap_ic_inputs(f, Df, x0, t, k_step)
    n_vec = _resolve_n_lyap(n_lyap, x0.numel())

    step = resolve_stepper(stepper)
    LE, LE_history, BLV_history, _R_history, trajectory = (
        run_state_variational_integrator(
            f,
            Df,
            x0,
            t,
            *args,
            k_step=k_step,
            stepper=step,
            n_lyap=n_vec,
            qr_method=qr_method,
        )
    )
    if return_blv:
        return LE, LE_history, BLV_history, trajectory
    return LE, LE_history, trajectory


def _validate_lyap_inputs(
    f: Callable,
    Df: Callable,
    trajectory: Tensor,
    t: Tensor,
    k_step: int,
) -> Tuple[int, int]:
    if not callable(f):
        raise TypeError("f must be callable.")
    if not callable(Df):
        raise TypeError("Df must be callable.")

    if not isinstance(trajectory, torch.Tensor):
        raise TypeError("trajectory must be a torch.Tensor.")
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape (nt, n).")

    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor.")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional.")
    if t.numel() < 2:
        raise ValueError("t must contain at least two time points.")

    if not isinstance(k_step, int):
        raise TypeError("k_step must be an integer.")
    if k_step < 1:
        raise ValueError("k_step must be at least 1.")

    nt, n = trajectory.shape
    if nt != t.shape[0]:
        raise ValueError(
            f"trajectory has {nt} time samples but t has {t.shape[0]} entries."
        )

    return n, nt


def _validate_lyap_ic_inputs(
    f: Callable,
    Df: Callable,
    x0: Tensor,
    t: Tensor,
    k_step: int,
) -> None:
    if not callable(f):
        raise TypeError("f must be callable.")
    if not callable(Df):
        raise TypeError("Df must be callable.")

    if not isinstance(x0, torch.Tensor):
        raise TypeError("x0 must be a torch.Tensor.")
    if x0.ndim != 1:
        raise ValueError("x0 must be one-dimensional.")
    if x0.numel() < 1:
        raise ValueError("x0 must contain at least one state variable.")

    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor.")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional.")
    if t.numel() < 2:
        raise ValueError("t must contain at least two time points.")

    if not isinstance(k_step, int):
        raise TypeError("k_step must be an integer.")
    if k_step < 1:
        raise ValueError("k_step must be at least 1.")


def _compute_lyap_outputs(
    f: Callable,
    Df: Callable,
    trajectory: Tensor,
    t: Tensor,
    *args,
    k_step: int = 1,
    stepper: Union[str, VariationalStepper, None] = "rk4",
    n_lyap: Union[int, None] = None,
    qr_method: str = "householder",
    ginelli_method: str = "ginelli",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    step = resolve_stepper(stepper)
    LE, LE_history, Q_history, R_history = run_variational_integrator(
        f,
        Df,
        trajectory,
        t,
        *args,
        k_step=k_step,
        stepper=step,
        n_lyap=n_lyap,
        qr_method=qr_method,
    )
    CLV_history = _clvs(Q_history, R_history, ginelli_method=ginelli_method)
    return LE, LE_history, Q_history, CLV_history


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


__all__ = [
    "lyap_analysis",
    "lyap_exp",
    "lyap_analysis_from_ic",
    "lyap_exp_from_ic",
]
