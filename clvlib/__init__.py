"""clvlib: Lyapunov exponents and CLV utilities.

This package provides routines to integrate variational equations,
compute Lyapunov exponents, and compute Covariant Lyapunov Vectors
via the Ginelli method.
"""

from .lyap_fun import (
    varRK4_step,
    LE_int,
    compute_CLV,
    qr_mgs,
)

__all__ = [
    "varRK4_step",
    "LE_int",
    "compute_CLV",
    "qr_mgs",
]

__version__ = "0.1.0"

