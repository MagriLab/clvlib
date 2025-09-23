import matplotlib.pyplot as plt
import numpy as np
from clvlib.numpy import lyap_analysis, compute_ICLE

def lorenz96(_: float, x: np.ndarray, forcing: float) -> np.ndarray:
    """Lorenz-96 vector field (pure NumPy)."""
    xp1 = np.roll(x, -1)
    xm2 = np.roll(x, 2)
    xm1 = np.roll(x, 1)
    return (xp1 - xm2) * xm1 - x + forcing


def lorenz96_jacobian(_: float, x: np.ndarray, forcing: float) -> np.ndarray:  # noqa: ARG001
    """Jacobian matrix of the Lorenz-96 system (pure NumPy)."""
    k = x.size
    jac = np.zeros((k, k), dtype=np.float64)

    idx = np.arange(k)
    im1 = (idx - 1) % k
    im2 = (idx - 2) % k
    ip1 = (idx + 1) % k

    jac[idx, im1] = x[ip1] - x[im2]
    jac[idx, ip1] = x[im1]
    jac[idx, im2] = -x[im1]
    jac[idx, idx] = -1.0
    return jac


def main():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    data = np.load("benchmarks/lorenz96_solution.npz", allow_pickle=True)
    t_loaded = data["t"]
    x_loaded = data["x"]
    x_loaded = x_loaded.T  # Transpose to shape (N, len(t))
    F = 8

    # Perform Lyapunov analysis
    LE, LE_history, Q_history, CLV_history= lyap_analysis(lorenz96, lorenz96_jacobian, x_loaded, t_loaded, F, k_step=1, )
    ICLEs = compute_ICLE(lorenz96_jacobian, x_loaded, t_loaded, CLV_history, F)

if __name__ == "__main__":
    main()
