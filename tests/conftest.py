"""Pytest configuration to ensure local imports work without install.

Adds the repository root to `sys.path` so `import clvlib` succeeds when
running tests directly (e.g., from the `tests/` directory) without
`pip install -e .`.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

