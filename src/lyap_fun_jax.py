"""JAX-based utilities for Lyapunov exponent and CLV computations."""
from __future__ import annotations
from functools import partial
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp

Array = jnp.ndarray

@jax.jit
def varRK4_step(
    f: Callable,
    Df: Callable,
    t: float,
    x: jnp.ndarray,
    V: jnp.ndarray,
    dt: float,
    *args
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    k1 = dt * f(t, x, *args)
    k2 = dt * f(t + 0.5*dt, x + 0.5*k1, *args)
    k3 = dt * f(t + 0.5*dt, x + 0.5*k2, *args)
    k4 = dt * f(t + dt,     x + k3,     *args)

    K1 = dt * (Df(t, x, *args) @ V)
    K2 = dt * (Df(t + 0.5*dt, x + 0.5*k1, *args) @ (V + 0.5*K1))
    K3 = dt * (Df(t + 0.5*dt, x + 0.5*k2, *args) @ (V + 0.5*K2))
    K4 = dt * (Df(t + dt,     x + k3,     *args) @ (V + K3))

    x_next = x + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    V_next = V + (K1 + 2*K2 + 2*K3 + K4) / 6.0

    return x_next, V_next

@jax.jit
def _normalize_columns(A: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    norms = jnp.linalg.norm(A, axis=0, keepdims=True)
    norms_safe = jnp.where(norms == 0.0, 1.0, norms)
    return A / norms_safe, norms

@jax.jit
def qr_mgs(A: Array) -> Tuple[Array, Array]:
    m, n = A.shape

    Q0 = jnp.zeros((m, n), dtype=A.dtype)
    R0 = jnp.zeros((n, n), dtype=A.dtype)

    def outer_body(j, carry):
        Qc, Rc, Vc = carry

        v = Vc[:, j]
        norm = jnp.linalg.norm(v)
        norm = jnp.where(norm == 0.0, 1.0, norm)
        q = v / norm
        Qc = Qc.at[:, j].set(q)
        Rc = Rc.at[j, j].set(norm)

        def inner_body(k, inner_carry):
            Rin, Vin = inner_carry
            proj = jnp.dot(q, Vin[:, k])
            Rin = Rin.at[j, k].set(proj)
            Vin = Vin.at[:, k].add(-proj * q)
            return Rin, Vin
        
        Rc, Vc = jax.lax.fori_loop(j + 1, n, inner_body, (Rc, Vc))
        return Qc, Rc, Vc
    
    Q, R, _ = jax.lax.fori_loop(0, n, outer_body, (Q0, R0, A))
    return Q, R


def LE_int(
    f: Callable,
    Df: Callable,
    x0: jnp.ndarray,
    t: jnp.ndarray,
    *args
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    n = x0.size

    # Allocate arrays
    x_traj = jnp.empty((n, nt))
    x_traj[:, 0] = x0
    Q_history = jnp.empty((n, n, nt))
    R_history = jnp.empty((n, n, nt))
    LE_history = jnp.empty((n, nt))
    LE_history[:, 0] = 0.0

    # Initial perturbations: identity
    Q = jnp.eye(n)
    Q_history[:, :, 0] = Q
    R_history[:, :, 0] = jnp.eye(n)

    LE = jnp.zeros(n)

    # Time integration loop
    for i in range(nt - 1):
        # Integrate state + variational system
        x_next, Q_next = varRK4_step(f, Df, t[i], x_traj[:, i], Q, dt, *args)
        x_traj[:, i + 1] = x_next

        # Re-orthonormalize perturbations (keep MGS as requested)
        Q, R = qr_mgs(Q_next)
        Q_history[:, :, i + 1] = Q
        R_history[:, :, i + 1] = R

        # Update cumulative Lyapunov sums
        LE += jnp.log(jnp.abs(jnp.diag(R)))
        LE_history[:, i + 1] = LE / ((i + 1) * dt)

    # Normalize final exponents
    LE /= (nt - 1) * dt

    return x_traj, Q_history, R_history, LE, LE_history

def compute_CLV(Q: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
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
    C = jnp.empty((n_lyap, n_lyap, n_time), dtype=Q.dtype)
    D = jnp.empty((n_lyap, n_time), dtype=Q.dtype)  # norms of CLVs in GS basis
    V = jnp.empty((n_dim, n_lyap, n_time), dtype=Q.dtype)

    # Initialize last step
    C[:, :, -1] = jnp.eye(n_lyap)
    D[:, -1] = jnp.ones(n_lyap)
    V[:, :, -1] = Q[:, :, -1] @ C[:, :, -1]

    # Backward iteration
    for i in reversed(range(n_time - 1)):
        # Solve R_i * C_i = C_{i+1}
        C_next = jsp.linalg.solve_triangular(
            R[:, :, i], C[:, :, i + 1], lower=False, overwrite_b=True, check_finite=False
        )
        C[:, :, i], D[:, i] = _normalize_columns(C_next)

        # Compute the CLVs 
        V[:, :, i] = Q[:, :, i] @ C[:, :, i]

    # Normalize CLVs across (lyap, time) without using 2D helper
    V, _ = _normalize_columns(V)

    return V

@jax.jit
def compute_angles(V1: jnp.ndarray, V2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    cos_thetas = jnp.einsum('ij,ij->j', V1, V2) 
    thetas     = jnp.arccos(cos_thetas)
    return cos_thetas, thetas

@jax.jit
def compute_ICLE(jacobian_function, solution, time, CLV_history, *args):
    """
    Compute instantaneous covariant Lyapunov exponents (ICLEs) for one or more
    CLVs along a trajectory.
    """
    # one-step computation at a single time index
    def step(t, x, V):
        J = jacobian_function(t, x, *args)        # (n, n)
        return jnp.einsum("ik,ij,jk->k", V, J, V) # (m,)

    # vectorize across time
    VTJV = jax.vmap(step, in_axes=(0, 1, 2), out_axes=1)(time, solution, CLV_history)
    return VTJV


@jax.jit
def principal_angles(V1: jnp.ndarray, V2: jnp.ndarray) -> jnp.ndarray:
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
    thetas : ndarray, shape (min(m1, m2), nt)
        Principal angles in radians between the two subspaces.
    """
    compute = lambda v1, v2: jsp.linalg.subspace_angles(v1, v2)
    # vmap over the time axis
    return jax.vmap(compute, in_axes=(2, 2), out_axes=1)(V1, V2)