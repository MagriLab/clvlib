"""clvlib: Lyapunov exponents and Covariant Lyapunov Vector utilities.

Public API mirrors the NumPy backend for convenience while keeping the
module split clean. The Numba backend (if available) is exposed as
``clvlib.numba`` without modifying its contents.
"""

from . import numpy as numpy_backend
from .numpy import (
    lyap_analysis,
    lyap_exp,
    compute_angles,
    principal_angles,
    compute_ICLE,
    resolve_stepper,
    register_stepper,
    VariationalStepper,
)

numpy = numpy_backend

__all__ = [
    "lyap_analysis",
    "lyap_exp",
    "compute_angles",
    "principal_angles",
    "compute_ICLE",
    "resolve_stepper",
    "register_stepper",
    "VariationalStepper",
    "numpy",
]

try:
    from . import numba as numba_backend
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name == "numba":
        numba = None
    else:  # importing clvlib.numba failed for another reason
        raise
else:
    numba = numba_backend
    __all__.append("numba")

__version__ = "0.1.0"
