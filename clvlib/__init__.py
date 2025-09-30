"""clvlib: Lyapunov exponents and Covariant Lyapunov Vector utilities.

Public API mirrors the NumPy backend for convenience while keeping the
module split clean. The Numba backend (if available) is exposed as
``clvlib.numba`` without modifying its contents.
"""

from . import numpy as numpy_backend
from .numpy import (
    lyap_analysis,
    lyap_exp,
    lyap_analysis_from_ic,
    lyap_exp_from_ic,
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
    "lyap_analysis_from_ic",
    "lyap_exp_from_ic",
    "compute_angles",
    "principal_angles",
    "compute_ICLE",
    "resolve_stepper",
    "register_stepper",
    "VariationalStepper",
    "numpy",
]


from . import numba as numba_backend
from . import pytorch as pytorch_backend

# Expose backends at top level for convenience
numba = numba_backend
pytorch = pytorch_backend
__all__ += ["numba", "pytorch"]
