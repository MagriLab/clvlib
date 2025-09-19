"""Benchmark the speed of the Lyapunov/CLV routines.

The script integrates a Lorenz-63 trajectory once and then measures the
execution time of both the Numba-backed and NumPy-backed implementations.
Each backend receives a configurable number of warm-up runs (to trigger JIT
compilation where applicable) before the timed repetitions.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from numba import njit

from clvlib.numba.lyap_fun import lyap_analysis as lyap_analysis_numba
from clvlib.numpy.lyap_fun import lyap_analysis as lyap_analysis_numpy


@dataclass
class BenchmarkConfig:
    dt: float = 0.01
    steps: int = 10_000
    repeats: int = 10
    warmup: int = 1
    k_step: int = 1
    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    rho: float = 28.0
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class BackendSpec:
    name: str
    lyap_fn: Callable
    f: Callable
    jac: Callable


@dataclass
class BackendResult:
    name: str
    timings: np.ndarray


def lorenz96(t, X, F=10):
        Xdot = np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) - X + F
        return Xdot

def lorenz96_jacobian(t, X, F=10):
    K = len(X)
    # Initialize the Jacobian matrix
    J_xx = np.zeros((K, K))
    idx = np.arange(K)
    idx_im1 = (idx - 1) % K  # X[i-1]
    idx_ip1 = (idx + 1) % K  # X[i+1]
    idx_im2 = (idx - 2) % K  # X[i-2]
    J_xx[idx, idx_im1] = X[idx_ip1] - X[idx_im2]
    J_xx[idx, idx_ip1] = X[idx_im1]
    J_xx[idx, idx_im2] = -X[idx_im1]
    J_xx[idx, idx] = -1

    return J_xx


def lorenz63(_, x: np.ndarray, sigma: float, beta: float, rho: float) -> np.ndarray:
    """Lorenz-63 vector field (pure Python)."""
    dx = np.empty_like(x)
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = x[0] * (rho - x[2]) - x[1]
    dx[2] = x[0] * x[1] - beta * x[2]
    return dx


def jac_lorenz63(_, x: np.ndarray, sigma: float, beta: float, rho: float) -> np.ndarray:
    """Jacobian matrix of the Lorenz-63 system (pure Python)."""
    return np.array(
        [
            [-sigma, sigma, 0.0],
            [rho - x[2], -1.0, -x[0]],
            [x[1], x[0], -beta],
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def lorenz63_numba(t: float, x: np.ndarray, sigma: float, beta: float, rho: float) -> np.ndarray:
    """Lorenz-63 vector field compiled with Numba."""
    dx = np.empty_like(x)
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = x[0] * (rho - x[2]) - x[1]
    dx[2] = x[0] * x[1] - beta * x[2]
    return dx


@njit(cache=True)
def jac_lorenz63_numba(t: float, x: np.ndarray, sigma: float, beta: float, rho: float) -> np.ndarray:
    """Jacobian of Lorenz-63 compiled with Numba."""
    return np.array(
        [
            [-sigma, sigma, 0.0],
            [rho - x[2], -1.0, -x[0]],
            [x[1], x[0], -beta],
        ],
        dtype=np.float64,
    )


def integrate_lorenz(config: BenchmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the Lorenz system with RK4 to generate a reference trajectory."""
    steps = config.steps
    dt = config.dt
    t = np.linspace(0.0, steps * dt, steps + 1, dtype=np.float64)
    trajectory = np.empty((3, steps + 1), dtype=np.float64)

    x = np.asarray(config.x0, dtype=np.float64)
    trajectory[:, 0] = x

    for i in range(steps):
        ti = t[i]
        k1 = lorenz63(ti, x, config.sigma, config.beta, config.rho)
        k2 = lorenz63(ti + 0.5 * dt, x + 0.5 * dt * k1, config.sigma, config.beta, config.rho)
        k3 = lorenz63(ti + 0.5 * dt, x + 0.5 * dt * k2, config.sigma, config.beta, config.rho)
        k4 = lorenz63(ti + dt, x + dt * k3, config.sigma, config.beta, config.rho)

        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[:, i + 1] = x

    return trajectory, t


def run_benchmark(config: BenchmarkConfig) -> None:
    trajectory, t = integrate_lorenz(config)

    args = (config.sigma, config.beta, config.rho)

    backends: List[BackendSpec] = [
        BackendSpec("numba", lyap_analysis_numba, lorenz63_numba, jac_lorenz63_numba),
        BackendSpec("numpy", lyap_analysis_numpy, lorenz63, jac_lorenz63),
    ]

    results: List[BackendResult] = []
    for backend in backends:
        results.append(
            _benchmark_backend(backend, trajectory, t, args, config)
        )

    print(
        f"Benchmark settings: dt={config.dt}, steps={config.steps}, k_step={config.k_step}, "
        f"warmup={config.warmup}, repeats={config.repeats}"
    )

    for result in results:
        timings = result.timings
        print(
            f"[{result.name}] timings (s): "
            + ", ".join(f"{val:.4f}" for val in timings)
        )
        if timings.size > 1:
            std = timings.std(ddof=1) if timings.size > 1 else 0.0
        else:
            std = 0.0
        print(f"[{result.name}] mean ± std: {timings.mean():.4f} ± {std:.4f} s")

    if len(results) >= 2:
        baseline = results[0]
        for result in results[1:]:
            ratio = result.timings.mean() / baseline.timings.mean()
            print(f"Speed ratio {result.name}/{baseline.name}: {ratio:.2f}x")


def _benchmark_backend(
    backend: BackendSpec,
    trajectory: np.ndarray,
    t: np.ndarray,
    args: tuple,
    config: BenchmarkConfig,
) -> BackendResult:
    for _ in range(max(config.warmup, 0)):
        backend.lyap_fn(
            backend.f,
            backend.jac,
            trajectory,
            t,
            *args,
            k_step=config.k_step,
        )

    timings: List[float] = []
    for _ in range(config.repeats):
        start = time.perf_counter()
        backend.lyap_fn(
            backend.f,
            backend.jac,
            trajectory,
            t,
            *args,
            k_step=config.k_step,
        )
        timings.append(time.perf_counter() - start)

    return BackendResult(backend.name, np.array(timings, dtype=np.float64))


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark the Lyapunov analysis backends (Numba vs NumPy)"
    )
    parser.add_argument("--dt", type=float, default=BenchmarkConfig.dt, help="Time step for the base trajectory")
    parser.add_argument("--steps", type=int, default=BenchmarkConfig.steps, help="Number of integration steps")
    parser.add_argument("--repeats", type=int, default=BenchmarkConfig.repeats, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=BenchmarkConfig.warmup, help="Warm-up runs for JIT compilation")
    parser.add_argument("--k-step", type=int, default=BenchmarkConfig.k_step, help="Stride used in lyap_analysis")
    parser.add_argument("--sigma", type=float, default=BenchmarkConfig.sigma, help="Lorenz sigma parameter")
    parser.add_argument("--beta", type=float, default=BenchmarkConfig.beta, help="Lorenz beta parameter")
    parser.add_argument("--rho", type=float, default=BenchmarkConfig.rho, help="Lorenz rho parameter")
    parser.add_argument(
        "--x0",
        type=float,
        nargs=3,
        default=BenchmarkConfig.x0,
        metavar=("x", "y", "z"),
        help="Initial condition for the Lorenz system",
    )

    args = parser.parse_args()
    return BenchmarkConfig(
        dt=args.dt,
        steps=args.steps,
        repeats=args.repeats,
        warmup=args.warmup,
        k_step=args.k_step,
        sigma=args.sigma,
        beta=args.beta,
        rho=args.rho,
        x0=tuple(args.x0),
    )


def main() -> None:
    config = parse_args()
    run_benchmark(config)


if __name__ == "__main__":
    main()
