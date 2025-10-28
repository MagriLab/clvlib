# clvlib

`clvlib` is a library for computing Lyapunov exponents and Covariant Lyapunov Vectors (CLVs) with NumPy and PyTorch backends. Under the hood it implements the Benettin algorithm[^1], giving you control over the re-orthonormalisation step through selectable QR routines: `householder` (SciPy-backed, fast, numerically robust) or `gram-schmidt` (Numba-powered and friendlier to some CLV post-processing).

Householder-based updates may clash with the classical Ginelli reconstruction of CLVs[^2], so this package introduces an alternative variant, `upwind_ginelli`, that remains stable with either QR option. Have a look at the tutorials for a deeper dive into the trade-offs.

The variational stepper is modular. Standard Euler, RK2, RK4, and discrete-time steppers are bundled, but you can register your own functions for the integrators, and, when working with NumPy, JIT-compile them with Numba for extra speed.

## Installation
```bash
pip install clvlib
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
cosine, theta = compute_angles(clv_history[:, :, 0], clv_history[:, :, 1])

# Principal angles between subspaces (time-first arrays)
angles = principal_angles(clv_history[:, :, -1:], clv_history[:, :, :-1])

# Instantaneous covariant exponents sampled every k_step iterations
icle = compute_ICLE(jacobian, traj, times, clv_history, k_step=1)
```

## License
No explicit license is defined yet. Choose and add one (e.g., MIT, BSD-3-Clause) before distributing binaries.

## References

[^1]: Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J.-M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 1: Theory. Meccanica, 15(1), 9–20.

[^2]: Ginelli, F., Poggi, P., Turchi, A., Chaté, H., Livi, R., & Politi, A. (2007). Characterizing dynamics with covariant Lyapunov vectors. Physical Review Letters, 99(13), 130601.
