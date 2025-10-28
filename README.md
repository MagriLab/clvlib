# clvlib

`clvlib` provides utilities for computing Lyapunov exponents and Covariant Lyapunov Vectors (CLVs). This library has implementations in both `NumPy` and `PyTorch`. The Lyapunov exponents and backward Lyapunov vectors are computed with the Benettin algorithm, and the library gives the possibility to select the QR decomposition to be used. In particular, there are two possibilities, either using Householder reflection `householder` or Grham-Schmidt decomposition `grahm-schmidt`. The former is faster, as it is called using scipy, and more stable, however it interacts weirdly with Ginelli's algorithm for computation of the CLVs, for more information check the tutorials. The latter is slower and more unstable but ineracts better with Ginelli's algorithm.

To overcome this incompatibility between the Househodler reflections we introduce a novel modification to Ginelli's algorithm, that we call upwind Ginelli `upwind_ginelli`,

## Installation
```bash
pip install clvlib
```

For development work (tests, linters, typing):
```bash
pip install -e .[dev]
```

## Quickstart
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

LE, LE_history, blv_history, clv_history, traj = lyap_analysis_from_ic(
    lorenz,
    jacobian,
    x0,
    times,
    stepper="rk4",
    qr_method="householder",
    ginelli_method="upwind_ginelli",
)

print("Asymptotic Lyapunov exponents:", LE)
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
