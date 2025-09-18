import numpy as np
import scipy.linalg
from typing import Callable, Tuple

def lyap_analysis(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Lyapunov-exponent integration and compute the associated CLVs.
    (Type and dimensionality checks included.)
    Set `k_step` > 1 to use the k-step variational integrator.
    """
    if not callable(f):
        raise TypeError("f must be callable.")
    if not callable(Df):
        raise TypeError("Df must be callable.")

    if not isinstance(trajectory, np.ndarray):
        raise TypeError("trajectory must be a numpy.ndarray.")
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape (n, nt).")

    if not isinstance(t, np.ndarray):
        raise TypeError("t must be a numpy.ndarray.")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional.")
    if t.size < 2:
        raise ValueError("t must contain at least two time points.")

    if not isinstance(k_step, int):
        raise TypeError("k_step must be an integer.")
    if k_step < 1:
        raise ValueError("k_step must be at least 1.")

    n, nt = trajectory.shape
    if nt != t.size:
        raise ValueError(
            f"trajectory has {nt} time samples but t has {t.size} entries."
        )

    Q_history, R_history, LE, LE_history, CLV_history = _compute_lyap_outputs(
        f, Df, trajectory, t, *args, k_step=k_step
    )

    expected_time_samples = Q_history.shape[-1]
    if CLV_history.shape != (n, n, expected_time_samples):
        raise RuntimeError(
            "CLV history has inconsistent shape: "
            f"expected {(n, n, expected_time_samples)}, got {CLV_history.shape}."
        )

    return Q_history, R_history, LE, LE_history, CLV_history

def _compute_lyap_outputs(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    *args,
    k_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper that runs the Lyapunov integration and CLV computation.
    Selects between standard and k-step integrators based on `k_step`.
    """
    if k_step > 1:
        Q_history, R_history, LE, LE_history = _lyap_int_k_step(
            f, Df, trajectory, t, k_step, *args
        )
    else:
        Q_history, R_history, LE, LE_history = _lyap_int(
            f, Df, trajectory, t, *args
        )
    CLV_history = _clvs(Q_history, R_history)
    return Q_history, R_history, LE, LE_history, CLV_history



def _lyap_int(f: Callable, Df: Callable, trajectory: np.ndarray, t: np.ndarray, *args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Lyapunov exponents of a dynamical system
    using variational equations and QR re-orthonormalization.

    Parameters
    ----------
    f : Callable
        System dynamics, f(t, x, *args) -> dx/dt.
    Df : Callable
        Jacobian of the system, Df(t, x, *args) -> ∂f/∂x.
    x0 : ndarray, shape (n,)
        Initial state vector.
    t : ndarray, shape (nt,)
        Time grid for integration.
    *args : tuple
        Extra parameters passed to f and Df.

    Returns
    -------
    x_traj : ndarray, shape (n, nt)
        State trajectory.
    Q_history : ndarray, shape (n, n, nt)
        History of orthonormal perturbation vectors.
    R_history : ndarray, shape (n, n, nt)
        History of upper triangular matrices from QR.
    LE : ndarray, shape (n,)
        Final Lyapunov exponents.
    LE_history : ndarray, shape (n, nt)
        Evolution of Lyapunov exponents over time.
    """
    dt = t[1] - t[0]
    nt = len(t)
    n = trajectory.shape[0]

    # Allocate arrays
    Q_history = np.empty((n, n, nt))
    R_history = np.empty((n, n, nt))
    LE_history = np.empty((n, nt))
    

    # Initial perturbations: identity
    Q = np.eye(n)
    Q_history[:, :, 0] = Q
    R_history[:, :, 0] = np.eye(n)
    LE_history[:, 0] = 0.0
    log_sums = 0.

    # Time integration loop
    for i in range(nt - 1):
        # Integrate state + variational system
        _, Q = _var_rk4_step(f, Df, t[i], trajectory[:, i], Q, dt, *args)

        # Re-orthonormalize perturbations (keep MGS as requested)
        Q, R = _qr_mgs(Q)
        Q_history[:, :, i + 1] = Q
        R_history[:, :, i + 1] = R

        # Update cumulative Lyapunov sums
        log_sums += np.log(np.abs(np.diag(R)))
        LE_history[:, i + 1] = log_sums / ((i + 1) * dt)

    return Q_history, R_history, LE_history[:,-1], LE_history

def _lyap_int_k_step(
    f: Callable,
    Df: Callable,
    trajectory: np.ndarray,
    t: np.ndarray,
    k_step: int,
    *args
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    nt = t.size
    n = trajectory.shape[0]
    n_step = ((nt - 1) // k_step) + 1

    Q_history = np.empty((n, n, n_step), dtype=float)
    R_history = np.empty((n, n, n_step), dtype=float)
    LE_history = np.empty((n, n_step), dtype=float)

    Q = np.eye(n, dtype=float)
    log_sums = np.zeros(n, dtype=float)

    Q_history[:, :, 0] = Q
    R_history[:, :, 0] = np.eye(n, dtype=float)
    LE_history[:, 0] = 0.0

    j = 0

    for i in range(nt - 1):
        _, Q = _var_rk4_step(f, Df, t[i], trajectory[:, i], Q, dt, *args)

        if ((i + 1) % k_step == 0):
            Q, R = _qr_mgs(Q)
            Q_history[:, :, j + 1] = Q
            R_history[:, :, j + 1] = R
            log_sums += np.log(np.abs(np.diag(R)))
            LE_history[:, j + 1] = log_sums / ((j + 1) * k_step * dt)
            j += 1

    return Q_history, R_history, LE_history[:,-1], LE_history


def _clvs(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute the Covariant Lyapunov Vectors (CLVs) using the method of
    Ginelli et al. (PRL, 2007).

    Parameters
    ----------
    Q : ndarray, shape (n_dim, n_lyap, n_time)
        Timeseries of Gram–Schmidt vectors.
    R : ndarray, shape (n_lyap, n_lyap, n_time)
        Timeseries of upper-triangular R matrices from QR decomposition.

    Returns
    -------
    V : ndarray, shape (n_dim, n_lyap, n_time)
        Covariant Lyapunov Vectors (CLVs). Each slice V[:, :, t] contains
        the CLVs at time index t.
    """
    n_dim, n_lyap, n_time = Q.shape

    # CLV coordinates in GS basis
    C = np.empty((n_lyap, n_lyap, n_time), dtype=Q.dtype)
    D = np.empty((n_lyap, n_time), dtype=Q.dtype)  # norms of CLVs in GS basis
    V = np.empty((n_dim, n_lyap, n_time), dtype=Q.dtype)

    # Initialize last step
    C[:, :, -1] = np.eye(n_lyap)
    D[:, -1] = np.ones(n_lyap)
    V[:, :, -1] = Q[:, :, -1] @ C[:, :, -1]

    # Backward iteration
    for i in reversed(range(n_time - 1)):
        # Solve R_i * C_i = C_{i+1}
        C_next = scipy.linalg.solve_triangular(
            R[:, :, i], C[:, :, i + 1], lower=False, overwrite_b=True, check_finite=False
        )
        C[:, :, i], D[:, i] = _normalize_columns(C_next)

        # Compute the CLVs 
        V[:, :, i] = Q[:, :, i] @ C[:, :, i]

    # Normalize CLVs across (lyap, time) without using 2D helper
    V, _ = _normalize_columns(V)

    return V


def _var_rk4_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: np.ndarray,
    V: np.ndarray,
    dt: float,
    *args
) -> np.ndarray:
    """
    Perform a single 4th-order Runge–Kutta step for both the state and the
    associated variational equation.

    Parameters
    ----------
    f : callable
        ODE function f(t, x, *args) returning dx/dt as ndarray.
    Df : callable
        Jacobian function Df(t, x, *args) returning the matrix ∂f/∂x.
    t : float
        Current time.
    x : ndarray, shape (n,)
        Current state vector.
    V : ndarray, shape (n, m)
        Current variational matrix (e.g., tangent dynamics).
    dt : float
        Step size.
    *args : tuple
        Extra parameters passed to both f and Df.

    Returns
    -------
    x_next : ndarray, shape (n,)
        State at t + dt.
    V_next : ndarray, shape (n, m)
        Variational matrix at t + dt.
    """
    # ---- State integration ----
    k1 = dt * f(t, x, *args)
    k2 = dt * f(t + 0.5*dt, x + 0.5*k1, *args)
    k3 = dt * f(t + 0.5*dt, x + 0.5*k2, *args)
    k4 = dt * f(t + dt,     x + k3,     *args)

    # ---- Variational integration ----
    K1 = dt * (Df(t, x, *args) @ V)
    K2 = dt * (Df(t + 0.5*dt, x + 0.5*k1, *args) @ (V + 0.5*K1))
    K3 = dt * (Df(t + 0.5*dt, x + 0.5*k2, *args) @ (V + 0.5*K2))
    K4 = dt * (Df(t + dt,     x + k3,     *args) @ (V + K3))

    # ---- Combine increments ----
    x_next = x + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    V_next = V + (K1 + 2*K2 + 2*K3 + K4) / 6.0

    return x_next, V_next

def _qr_mgs(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    QR decomposition using the Modified Gram–Schmidt algorithm.
    Numba-compatible (no Python exceptions).

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.

    Returns
    -------
    Q : ndarray, shape (m, n)
        Orthonormal basis vectors (Q.T @ Q ≈ I).
    R : ndarray, shape (n, n)
        Upper triangular matrix.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()

    for j in range(n):
        # Norm of column j
        norm = np.sqrt(np.sum(V[:, j] * V[:, j]))
        Q[:, j] = V[:, j] / norm
        R[j, j] = norm
        # Orthogonalize remaining columns
        for k in range(j + 1, n):
            R[j, k] = np.dot(Q[:, j], V[:, k])
            V[:, k] = V[:, k] - R[j, k] * Q[:, j]

    return Q, R

def _normalize_columns(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the columns of a 2D matrix.

    Parameters
    ----------
    A : ndarray, shape (n, m)
        Input matrix.

    Returns
    -------
    Q : ndarray, shape (n, m)
        Matrix with normalized columns.
    norms : ndarray, shape (m,)
        Norm of each column before normalization.
    """
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    norms_safe = np.where(norms == 0.0, 1.0, norms)
    return A / norms_safe, norms

def compute_angles(V1: np.ndarray, V2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the angles between two vectors in V1 and V2.

    Parameters
    ----------
    V1 : ndarray
        Array of shape (timesteps, subspace_dim, vectors) representing the first set of vectors.
    V2 : ndarray
        Array of shape (timesteps, subspace_dim, vectors) representing the second set of vectors.

    Returns
    -------
    thetas : ndarray
        Angles (in degrees) between the two vectors.
    """
    # Compute the principal angles between the two subspaces
    cos_thetas = np.einsum('ij,ij->j', V1, V2) 
    thetas     = np.arccos(cos_thetas)
    return cos_thetas, thetas

def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """
    Compute the principal angles between two subspaces spanned by the columns
    of V1 and V2.

    Parameters
    ----------
    V1 : ndarray, shape (n, m1, nt)
        First set of vectors spanning a subspace.
    V2 : ndarray, shape (n, m2, nt)
        Second set of vectors spanning a subspace.

    Returns
    -------
    thetas : ndarray, shape (min(m1, m2),nt)
        Principal angles in radians between the two subspaces.
    """
    # Compute the smallest principal angle at each time step
    n, m1, nt = V1.shape
    _, m2, _  = V2.shape
    theta = np.empty((min(m1, m2), nt), dtype=float)
    for i in range(nt):
        theta[:,i] = scipy.linalg.subspace_angles(V1[:,:,i], V2[:,:,i])

    return theta

def compute_ICLE(
    jacobian_function: Callable,
    solution: np.ndarray,
    time: np.ndarray,
    CLV_history: np.ndarray,
    *args,
    k_step: int = 1,
) -> np.ndarray:
    """
    Compute instantaneous covariant Lyapunov exponents (ICLEs) for one or more
    CLVs along a trajectory.
    """
    if not isinstance(k_step, int):
        raise TypeError("k_step must be an integer.")
    if k_step < 1:
        raise ValueError("k_step must be at least 1.")
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    if solution.ndim != 2:
        raise ValueError("solution must be two-dimensional.")
    if CLV_history.ndim != 3:
        raise ValueError("CLV_history must be three-dimensional.")

    n_state, n_time = solution.shape
    n_clv_state, m, n_samples = CLV_history.shape
    if n_state != n_clv_state:
        raise ValueError("solution and CLV_history must share the same state dimension.")
    if n_time != time.size:
        raise ValueError("solution and time must share the same number of samples.")
    if n_samples == 0:
        raise ValueError("CLV_history must contain at least one time sample.")

    sample_indices = np.arange(0, k_step * n_samples, k_step, dtype=int)
    if sample_indices.size != n_samples:
        raise RuntimeError("Unexpected number of samples inferred from CLV_history.")
    if sample_indices[-1] >= n_time:
        raise ValueError(
            "CLV history length is incompatible with the provided solution/time for this k_step."
        )

    states = solution[:, sample_indices]
    times = time[sample_indices]

    return _compute_icle_series(jacobian_function, states, times, CLV_history, *args)


def _compute_icle_series(
    jacobian_function: Callable,
    sampled_states: np.ndarray,
    sampled_times: np.ndarray,
    CLV_history: np.ndarray,
    *args,
) -> np.ndarray:
    """Helper that evaluates J(t)xv and assembles the ICLE time series."""
    _, m, n_samples = CLV_history.shape
    VTJV = np.empty((m, n_samples), dtype=float)
    for idx in range(n_samples):
        J = jacobian_function(sampled_times[idx], sampled_states[:, idx], *args)
        VTJV[:, idx] = np.einsum(
            "ik,ij,jk->k", CLV_history[:, :, idx], J, CLV_history[:, :, idx]
        )
    return VTJV
