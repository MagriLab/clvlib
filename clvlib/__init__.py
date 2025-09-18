"""clvlib: Lyapunov exponents and Covariant Lyapunov Vector utilities."""

from . import numpy as numpy_backend
from .numpy import (
    lyap_analysis,
    compute_angles,
    principal_angles,
    compute_ICLE,
)

numpy = numpy_backend

__all__ = [
    "lyap_analysis",
    "compute_angles",
    "principal_angles",
    "compute_ICLE",
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
