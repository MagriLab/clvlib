# clvlib

`clvlib` is a library for computing Lyapunov exponents and Covariant Lyapunov Vectors (CLVs) with NumPy and PyTorch. Lyapunov exponents are computed using Benettin's algorithm [[1]](#R1). This library gives you control over the re-orthonormalisation step through selectable QR routines: `householder` (SciPy, fast and numerically robust) or `gram-schmidt` (accelerated using Numba). The CLVs are computed using Ginelli's algorithm [[2]](#R2).

Householder-based updates may clash with the classical Ginelli reconstruction of CLVs [[2]](#R2), so this package introduces an alternative variant, `upwind_ginelli`, that remains stable with either QR option. Have a look at the tutorials for a deeper dive.

The variational stepper used to integrate the variational system is modular. Standard Euler, RK2, RK4, and discrete-time steppers are bundled, but you can register your own functions for the integrators.

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

See `tutorials/lorenz_numpy_quickstart.ipynb` for the NumPy walkthrough, and `tutorials/lorenz_pytorch_quickstart.ipynb` for the PyTorch version.

Want only the most unstable directions? Pass `n_lyap=k` to any of the Lyapunov helpers (`lyap_exp`, `lyap_analysis`, and their `_from_ic` counterparts) to compute just the leading `k` exponents/BLVs/CLVs.


## Angles and instantaneous CLVs
```python
from clvlib.numpy import compute_angles, principal_angles, compute_ICLE

# Pairwise vector angles
cosine, theta = compute_angles(clv_history[:, :, 0], clv_history[:, :, 1])

# Principal angles between subspaces
angles = principal_angles(clv_history[:, :, -1:], clv_history[:, :, :-1])

# Instantaneous covariant exponents sampled every k_step iterations
icle = compute_ICLE(jacobian, traj, times, clv_history, k_step=1)
```

## Citation
If `clvlib` contributes to your published work, please cite it as:

```
@misc{consonni_clvlib_2025,
  author    = {Riccardo Consonni},
  title     = {clvlib},
  year      = {2025},
  url       = {https://github.com/riccardo-consonni/clvlib},
}
```

## License
Published under the MIT License. See `LICENSE` for the full text.

## References
<a id="R1">**[1]**</a> Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J.-M. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 1: Theory. Meccanica, 15(1), 9–20.
<a id="R2">**[2]**</a> Ginelli, F., Poggi, P., Turchi, A., Chaté, H., Livi, R., & Politi, A. (2007). Characterizing dynamics with covariant Lyapunov vectors. Physical Review Letters, 99(13), 130601.
