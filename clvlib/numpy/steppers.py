import numpy as np
from typing import Callable, Tuple, Protocol, Dict, Union


class VariationalStepper(Protocol):
    def __call__(
        self,
        f: Callable,
        Df: Callable,
        t: float,
        x: np.ndarray,
        V: np.ndarray,
        dt: float,
        *args,
    ) -> Tuple[np.ndarray, np.ndarray]: ...


def _var_euler_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: np.ndarray,
    V: np.ndarray,
    dt: float,
    *args,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward Euler step for state and variational system."""
    k1 = dt * f(t, x, *args)
    K1 = dt * (Df(t, x, *args) @ V)
    return x + k1, V + K1


def _var_rk2_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: np.ndarray,
    V: np.ndarray,
    dt: float,
    *args,
) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit midpoint (RK2) for state and variational system."""
    k1 = dt * f(t, x, *args)
    K1 = dt * (Df(t, x, *args) @ V)
    k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1, *args)
    K2 = dt * (Df(t + 0.5 * dt, x + 0.5 * k1, *args) @ (V + 0.5 * K1))
    return x + k2, V + K2


def _var_rk4_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: np.ndarray,
    V: np.ndarray,
    dt: float,
    *args,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fourth-order Runge–Kutta for state and variational system."""
    # State
    k1 = dt * f(t, x, *args)
    k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1, *args)
    k3 = dt * f(t + 0.5 * dt, x + 0.5 * k2, *args)
    k4 = dt * f(t + dt, x + k3, *args)

    # Variational
    K1 = dt * (Df(t, x, *args) @ V)
    K2 = dt * (Df(t + 0.5 * dt, x + 0.5 * k1, *args) @ (V + 0.5 * K1))
    K3 = dt * (Df(t + 0.5 * dt, x + 0.5 * k2, *args) @ (V + 0.5 * K2))
    K4 = dt * (Df(t + dt, x + k3, *args) @ (V + K3))

    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    V_next = V + (K1 + 2 * K2 + 2 * K3 + K4) / 6.0
    return x_next, V_next


def _discrete_var_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: np.ndarray,
    V: np.ndarray,
    dt: float = 0.0,
    *args,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete-time variational step: x_{k+1} = f(x_k), V_{k+1} = Df(x_k) V_k.
    Note that dt is ignored in this stepper.
    """
    x_next = f(t, x, *args)
    V_next = Df(t, x, *args) @ V
    return x_next, V_next


_STEPPERS: Dict[str, VariationalStepper] = {
    "euler": _var_euler_step,
    "rk2": _var_rk2_step,
    "rk4": _var_rk4_step,
    "discrete": _discrete_var_step,
}


def resolve_stepper(
    stepper: Union[str, VariationalStepper, None],
) -> VariationalStepper:
    """Resolve a stepper identifier or callable to a concrete stepper.

    - None or 'rk4' -> fourth-order Runge–Kutta variational stepper
    - 'rk2' -> explicit midpoint RK2
    - 'euler' -> forward Euler
    - 'discrete' -> discrete-time map variational step
    - callable -> used as-is (must match VariationalStepper signature)
    """
    if stepper is None:
        return _var_rk4_step
    if isinstance(stepper, str):
        key = stepper.lower()
        try:
            return _STEPPERS[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown stepper '{stepper}'. Valid: {list(_STEPPERS.keys())}"
            ) from exc
    if callable(stepper):
        return stepper  # duck-typed VariationalStepper
    raise TypeError("stepper must be a string, callable, or None")


def register_stepper(name: str, func: VariationalStepper) -> None:
    """Register a custom stepper globally for this module."""
    if not callable(func):
        raise TypeError("func must be callable")
    _STEPPERS[name.lower()] = func


__all__ = [
    "VariationalStepper",
    "resolve_stepper",
    "register_stepper",
    "_var_euler_step",
    "_var_rk2_step",
    "_var_rk4_step",
    "_discrete_var_step",
]
