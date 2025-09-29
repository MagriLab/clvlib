import numpy as np
import scipy.linalg
from typing import Callable, Tuple
from numba import njit

@njit(fastmath=True)
def gram_schmidt_qr(A):
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        # v = A[:, j].copy()
        v = np.empty(m, dtype=np.float64)
        for r in range(m):
            v[r] = A[r, j]

        for i in range(j):
            # R[i, j] = dot(Q[:, i], A[:, j])
            s = 0.0
            for k in range(m):
                s += Q[k, i] * A[k, j]
            R[i, j] = s

            # v -= R[i, j] * Q[:, i]
            c = R[i, j]
            for k in range(m):
                v[k] -= c * Q[k, i]

        # R[j, j] = norm(v)
        s2 = 0.0
        for k in range(m):
            s2 += v[k] * v[k]
        Rjj = np.sqrt(s2)
        R[j, j] = Rjj

        # Q[:, j] = v / R[j, j]
        inv = 1.0 / Rjj
        for k in range(m):
            Q[k, j] = v[k] * inv

    return Q, R

def lorenz96(t, X, F=10):
        Xdot = np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) - X + F
        return Xdot

def lorenz96_jacobian(t, X, F=10):
    K = X.shape[0]
    J_xx = np.full((K, K), 0.0)

    idx = np.arange(K)
    im1 = (idx - 1) % K
    ip1 = (idx + 1) % K
    im2 = (idx - 2) % K

    # Fill columns directly (vectorized)
    J_xx[idx, im1] = X[ip1] - X[im2]
    J_xx[idx, ip1] = X[im1]
    J_xx[idx, im2] = -X[im1]
    J_xx[idx, idx] = -1.0

    return J_xx


def lyap_int(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    *args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    nt = t.size
    n = trajectory.shape[1]

    # Time-first histories: (nt, n, n) and (nt, n)
    Q_history = np.empty((nt, n, n), dtype=float)
    R_history = np.empty((nt, n, n), dtype=float)
    LE_history = np.empty((nt, n), dtype=float)

    Q = np.eye(n, dtype=float)
    Q_history[0] = Q
    R_history[0] = np.eye(n, dtype=float)
    LE_history[0] = 0.0
    log_sums = np.zeros(n, dtype=float)

    for i in range(nt - 1):
        _, Q = _var_rk4_step(f, Df, t[i], trajectory[i], Q, dt, *args)
        Q, R = gram_schmidt_qr(Q)
        Q_history[i + 1] = Q
        R_history[i + 1] = R
        log_sums += np.log(np.abs(np.diag(R)))
        LE_history[i + 1] = log_sums / ((i + 1) * dt)

    return LE_history[-1], LE_history, Q_history, R_history


def normalize_columns(A: np.ndarray) -> np.ndarray:
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def clvs(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute the Covariant Lyapunov Vectors (CLVs) using Ginelli et al. (2007).

    Parameters
    ----------
    Q : ndarray, shape (n_time, n_dim, n_lyap)
        Time-first series of Gram–Schmidt vectors.
    R : ndarray, shape (n_time, n_lyap, n_lyap)
        Time-first series of upper-triangular R matrices from QR decomposition.
    """
    n_time, n_dim, n_lyap = Q.shape

    # Coordinates of CLVs in GS basis and the CLVs themselves (time-first)
    C = np.empty((n_time, n_lyap, n_lyap), dtype=Q.dtype)
    V = np.empty((n_time, n_dim, n_lyap), dtype=Q.dtype)

    C[-1] = np.eye(n_lyap)
    V[-1] = Q[-1] @ C[-1]

    for i in reversed(range(n_time - 1)):
        C_next = scipy.linalg.solve_triangular(
            R[i], C[i + 1], lower=False, overwrite_b=True, check_finite=False
        )
        C[i] = normalize_columns(C_next)
        V[i] = Q[i] @ C[i]

    # Normalize CLVs along the state axis for each (t, k)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V


def _var_rk4_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: np.ndarray,
    V: np.ndarray,
    dt: float,
    *args,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fourth-order Runge–Kutta for state and variational system."""
    # State
    k1 = dt * f(t, x, *args)
    k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1, *args)
    k3 = dt * f(t + 0.5 * dt, x + 0.5 * k2, *args)
    k4 = dt * f(t + dt, x + k3, *args)

    # Variational
    K1 = dt * (Df(t, x, *args) @ V)
    K2 = dt * (Df(t + 0.5 * dt, x + 0.5 * k1, *args) @ (V + 0.5 * K1))
    K3 = dt * (Df(t + 0.5 * dt, x + 0.5 * k2, *args) @ (V + 0.5 * K2))
    K4 = dt * (Df(t + dt, x + k3, *args) @ (V + K3))

    x_next = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    V_next = V + (K1 + 2 * K2 + 2 * K3 + K4) / 6.0
    return x_next, V_next

def main():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    data = np.load("benchmarks/lorenz96_solution.npz", allow_pickle=True)
    t_loaded = data["t"]
    x_loaded = data["x"]  # Expect shape (len(t), N) (time-first)
    F = 8

    # Perform Lyapunov analysis
    LE, LE_history, Q_history, R_history = lyap_int(
        lorenz96, lorenz96_jacobian, x_loaded, t_loaded, F,
    )
    CLV_history = clvs(Q_history, R_history)


if __name__ == "__main__":
    main()
