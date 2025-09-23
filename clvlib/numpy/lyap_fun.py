"""Backward-compatible shim for numpy backend.

Exports public APIs from modular submodules (api, angles, icle).
"""

from .api import lyap_analysis, lyap_exp
from .angles import compute_angles, principal_angles
from .icle import compute_ICLE

__all__ = [
    "lyap_analysis",
    "lyap_exp",
    "compute_angles",
    "principal_angles",
    "compute_ICLE",
]

