"""PyTorch-backed implementations of clvlib routines."""

try:
    import torch as _torch
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise ModuleNotFoundError(
            "clvlib PyTorch support requires the optional 'torch' dependency. "
            "Install it with `pip install \"clvlib[pytorch]\"`."
        ) from exc
    raise

del _torch

from .api import (
    lyap_analysis,
    lyap_exp,
    lyap_analysis_from_ic,
    lyap_exp_from_ic,
)
from .angles import compute_angles, principal_angles
from .icle import compute_ICLE
from .steppers import resolve_stepper, register_stepper, VariationalStepper

__all__ = [
    "lyap_analysis",
    "lyap_exp",
    "lyap_analysis_from_ic",
    "lyap_exp_from_ic",
    "compute_angles",
    "principal_angles",
    "compute_ICLE",
    "resolve_stepper",
    "register_stepper",
    "VariationalStepper",
]
