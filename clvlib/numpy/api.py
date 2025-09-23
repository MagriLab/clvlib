import numpy as np
from typing import Callable, Tuple, Union

from .steppers import VariationalStepper, resolve_stepper
from .integrators import run_variational_integrator
from .clv import _clvs


def lyap_analysis(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
    stepper: Union[str, VariationalStepper, None] = "rk4",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Lyapunov-exponent integration and compute the associated CLVs.
    Set `k_step` > 1 to use the k-step variational integrator.
    Returns (BLV_history, LE, LE_history, CLV_history).
    """
    n, _ = _validate_lyap_inputs(f, Df, trajectory, t, k_step)

    LE, LE_history, BLV_history, CLV_history = _compute_lyap_outputs(
        f, Df, trajectory, t, *args, k_step=k_step, stepper=stepper
    )

    expected_time_samples = BLV_history.shape[-1]
    if CLV_history.shape != (n, n, expected_time_samples):
        raise RuntimeError(
            "CLV history has inconsistent shape: "
            f"expected {(n, n, expected_time_samples)}, got {CLV_history.shape}."
        )

    return LE, LE_history, BLV_history, CLV_history


def lyap_exp(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
    stepper: Union[str, VariationalStepper, None] = "rk4",
    return_blv: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run Lyapunov-exponent integration without computing CLVs.
    Returns (LE, LE_history[, BLV_history]).
    """
    _validate_lyap_inputs(f, Df, trajectory, t, k_step)

    BLV_history, _, LE, LE_history = run_variational_integrator(
        f, Df, trajectory, t, *args, k_step=k_step, stepper=resolve_stepper(stepper)
    )

    if return_blv:
        return LE, LE_history, BLV_history

    return LE, LE_history


def _validate_lyap_inputs(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    k_step: int,
) -> Tuple[int, int]:
    if not callable(f):
        raise TypeError("f must be callable.")
    if not callable(Df):
        raise TypeError("Df must be callable.")

    if not isinstance(trajectory, np.ndarray):
        raise TypeError("trajectory must be a numpy.ndarray.")
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape (n, nt).")

    if not isinstance(t, np.ndarray):
        raise TypeError("t must be a numpy.ndarray.")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional.")
    if t.size < 2:
        raise ValueError("t must contain at least two time points.")

    if not isinstance(k_step, int):
        raise TypeError("k_step must be an integer.")
    if k_step < 1:
        raise ValueError("k_step must be at least 1.")

    n, nt = trajectory.shape
    if nt != t.size:
        raise ValueError(
            f"trajectory has {nt} time samples but t has {t.size} entries."
        )

    return n, nt


def _compute_lyap_outputs(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
    stepper: Union[str, VariationalStepper, None] = "rk4",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step = resolve_stepper(stepper)
    LE, LE_history, Q_history, R_history = run_variational_integrator(
        f, Df, trajectory, t, *args, k_step=k_step, stepper=step
    )
    CLV_history = _clvs(Q_history, R_history)
    return  LE, LE_history, Q_history, CLV_history


__all__ = [
    "lyap_analysis",
    "lyap_exp",
]

