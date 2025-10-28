import torch
from typing import Callable, Tuple, Protocol, Dict, Union

Tensor = torch.Tensor


class VariationalStepper(Protocol):
    def __call__(
        self,
        f: Callable,
        Df: Callable,
        t: float,
        x: Tensor,
        V: Tensor,
        dt: float,
        *args,
    ) -> Tuple[Tensor, Tensor]: ...


def _var_euler_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: Tensor,
    V: Tensor,
    dt: float,
    *args,
) -> Tuple[Tensor, Tensor]:
    """Forward Euler step for state and variational system."""
    k1 = f(t, x, *args) * dt
    K1 = Df(t, x, *args) @ V * dt
    return x + k1, V + K1


def _var_rk2_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: Tensor,
    V: Tensor,
    dt: float,
    *args,
) -> Tuple[Tensor, Tensor]:
    """Explicit midpoint (RK2) for state and variational system."""
    k1 = f(t, x, *args) * dt
    K1 = Df(t, x, *args) @ V * dt
    k2 = f(t + 0.5 * dt, x + 0.5 * k1, *args) * dt
    K2 = Df(t + 0.5 * dt, x + 0.5 * k1, *args) @ (V + 0.5 * K1) * dt
    return x + k2, V + K2


def _var_rk4_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: Tensor,
    V: Tensor,
    dt: float,
    *args,
) -> Tuple[Tensor, Tensor]:
    """Fourth-order Runge–Kutta for state and variational system."""
    k1 = f(t, x, *args) * dt
    k2 = f(t + 0.5 * dt, x + 0.5 * k1, *args) * dt
    k3 = f(t + 0.5 * dt, x + 0.5 * k2, *args) * dt
    k4 = f(t + dt, x + k3, *args) * dt

    K1 = Df(t, x, *args) @ V * dt
    K2 = Df(t + 0.5 * dt, x + 0.5 * k1, *args) @ (V + 0.5 * K1) * dt
    K3 = Df(t + 0.5 * dt, x + 0.5 * k2, *args) @ (V + 0.5 * K2) * dt
    K4 = Df(t + dt, x + k3, *args) @ (V + K3) * dt

    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    V_next = V + (K1 + 2 * K2 + 2 * K3 + K4) / 6.0
    return x_next, V_next


def _discrete_var_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: Tensor,
    V: Tensor,
    dt: float,
    *args,
) -> Tuple[Tensor, Tensor]:
    """Discrete-time variational step."""
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
    """Resolve a stepper identifier or callable to a concrete stepper."""
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
        return stepper
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
