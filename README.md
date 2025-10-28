# clvlib

Utilities to integrate Lyapunov exponents and compute Covariant Lyapunov Vectors (CLVs) with NumPy and PyTorch backends.

## Highlights
- Variational integrators with RK4/RK2/Euler steppers, optional discrete-time maps, and k-step sampling.
- Multiple QR decompositions (Householder or Gram–Schmidt) backed by SciPy/Numba (NumPy) or native Torch ops.
- Ginelli’s algorithm for CLVs plus helpers for principal angles and instantaneous CLVs (ICLEs).
- Identical APIs for both `clvlib.numpy` and `clvlib.pytorch` modules to ease CPU/GPU workflows.
- Fully type annotated codebase validated by `mypy`, `ruff`, and `pytest`.

## Installation
```bash
pip install clvlib
```

For development work (tests, linters, typing):
```bash
pip install -e .[dev]
```

## Quickstart (NumPy)
```python
import numpy as np
from clvlib.numpy import lyap_analysis_from_ic

# Lorenz '63 system ----------------------------------------------------------
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

def lorenz(t: float, x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            SIGMA * (x[1] - x[0]),
            x[0] * (RHO - x[2]) - x[1],
            x[0] * x[1] - BETA * x[2],
        ],
        dtype=float,
    )

def jacobian(t: float, x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [-SIGMA, SIGMA, 0.0],
            [RHO - x[2], -1.0, -x[0]],
            [x[1], x[0], -BETA],
        ],
        dtype=float,
    )

times = np.linspace(0.0, 40.0, 4001)
x0 = np.array([8.0, 0.0, 30.0], dtype=float)

LE, history, blv_history, clv_history, traj = lyap_analysis_from_ic(
    lorenz,
    jacobian,
    x0,
    times,
    stepper="rk4",
    qr_method="householder",
    ginelli_method="ginelli",
)

print("Asymptotic Lyapunov exponents:", LE)
```

## Quickstart (PyTorch)
```python
import torch
from clvlib.pytorch import lyap_exp_from_ic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA = torch.tensor(10.0, device=device)
RHO = torch.tensor(28.0, device=device)
BETA = torch.tensor(8.0 / 3.0, device=device)

def lorenz(t: float, x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            SIGMA * (x[1] - x[0]),
            x[0] * (RHO - x[2]) - x[1],
            x[0] * x[1] - BETA * x[2],
        )
    )

def jacobian(t: float, x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            [-SIGMA, SIGMA, 0.0],
            [RHO - x[2], -1.0, -x[0]],
            [x[1], x[0], -BETA],
        ],
        dtype=x.dtype,
        device=x.device,
    )

times = torch.linspace(0.0, 40.0, 4001, device=device)
x0 = torch.tensor([8.0, 0.0, 30.0], dtype=torch.float64, device=device)

LE, history, trajectory = lyap_exp_from_ic(
    lorenz, jacobian, x0, times, stepper="rk4", qr_method="householder"
)
```

## Angles and instantaneous CLVs
```python
from clvlib.numpy import compute_angles, principal_angles, compute_ICLE

# Pairwise vector angles
cosine, theta = compute_angles(blv_history[-1], clv_history[-1])

# Principal angles between subspaces (time-first arrays)
angles = principal_angles(blv_history[:, :, :2], clv_history[:, :, :2])

# Instantaneous covariant exponents sampled every k_step iterations
icle = compute_ICLE(jacobian, traj, times, clv_history, k_step=1)
```

## Tutorials and examples
- `tutorials/lorenz_numpy_quickstart.ipynb` – step-by-step walkthrough reproducing the NumPy quickstart, plotting LE convergence and CLV geometry.

Consider adding additional examples (e.g., discrete-time maps, higher-dimensional models, GPU benchmarks) under `tutorials/` or `examples/`.

## Testing & linting
```bash
ruff check
ruff format --check
mypy clvlib
pytest
```

## Contributing
- File issues or ideas in GitHub.
- Run the test suite (see above) before submitting a pull request.
- Keep documentation and examples in sync across the NumPy and PyTorch APIs.

## License
No explicit license is defined yet. Choose and add one (e.g., MIT, BSD-3-Clause) before distributing binaries.
