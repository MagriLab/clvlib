"""Benchmark the speed of the Lyapunov/CLV routines on Lorenz-96.

The script integrates a Lorenz-96 trajectory for a list of state dimensions and
then measures the execution time of both the Numba-backed and NumPy-backed
implementations. Each backend receives a configurable number of warm-up runs
(to trigger JIT compilation where applicable) before the timed repetitions.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np
from numba import njit

from clvlib.numba.lyap_fun import lyap_analysis as lyap_analysis_numba
from clvlib.numpy.lyap_fun import lyap_analysis as lyap_analysis_numpy


@dataclass
class BenchmarkConfig:
    dt: float = 0.01
    steps: int = 60
    repeats: int = 2
    warmup: int = 1
    k_step: int = 5
    forcing: float = 8.0
    dims: Sequence[int] = field(
        default_factory=lambda: (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    )
    perturbation: float = 0.01


@dataclass
class BackendSpec:
    name: str
    lyap_fn: Callable
    f: Callable
    jac: Callable


@dataclass
class BackendResult:
    name: str
    dim: int
    timings: np.ndarray


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


@njit(cache=True)
def lorenz96_numba(t: float, x: np.ndarray, forcing: float) -> np.ndarray:  # noqa: ARG001
    k = x.size
    dx = np.empty_like(x)
    for i in range(k):
        im2 = (i - 2) % k
        im1 = (i - 1) % k
        ip1 = (i + 1) % k
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + forcing
    return dx


@njit(cache=True)
def lorenz96_jacobian_numba(t: float, x: np.ndarray, forcing: float) -> np.ndarray:  # noqa: ARG001
    k = x.size
    jac = np.zeros((k, k))
    for i in range(k):
        im2 = (i - 2) % k
        im1 = (i - 1) % k
        ip1 = (i + 1) % k
        jac[i, im1] = x[ip1] - x[im2]
        jac[i, ip1] = x[im1]
        jac[i, im2] = -x[im1]
        jac[i, i] = -1.0
    return jac


def integrate_lorenz96(dim: int, config: BenchmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the Lorenz-96 system with RK4 to generate a trajectory."""
    steps = config.steps
    dt = config.dt
    t = np.linspace(0.0, steps * dt, steps + 1, dtype=np.float64)
    trajectory = np.empty((dim, steps + 1), dtype=np.float64)

    x = np.full(dim, config.forcing, dtype=np.float64)
    if dim > 0:
        x[0] += config.perturbation
    trajectory[:, 0] = x

    for i in range(steps):
        ti = t[i]
        k1 = lorenz96(ti, x, config.forcing)
        k2 = lorenz96(ti + 0.5 * dt, x + 0.5 * dt * k1, config.forcing)
        k3 = lorenz96(ti + 0.5 * dt, x + 0.5 * dt * k2, config.forcing)
        k4 = lorenz96(ti + dt, x + dt * k3, config.forcing)

        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[:, i + 1] = x

    return trajectory, t


def _benchmark_backend(
    backend: BackendSpec,
    trajectory: np.ndarray,
    t: np.ndarray,
    args: tuple,
    config: BenchmarkConfig,
) -> np.ndarray:
    state_input = trajectory[:, 0] if backend.lyap_fn is lyap_analysis_numpy else trajectory
    for _ in range(max(config.warmup, 0)):
        backend.lyap_fn(
            backend.f,
            backend.jac,
            state_input,
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
            state_input,
            t,
            *args,
            k_step=config.k_step,
        )
        timings.append(time.perf_counter() - start)

    return np.array(timings, dtype=np.float64)


def run_benchmark(config: BenchmarkConfig) -> None:
    dims = list(config.dims)
    if not dims:
        raise ValueError("No state dimensions provided for benchmarking.")

    backends: List[BackendSpec] = [
        BackendSpec("numba", lyap_analysis_numba, lorenz96_numba, lorenz96_jacobian_numba),
        BackendSpec("numpy", lyap_analysis_numpy, lorenz96, lorenz96_jacobian),
    ]

    print(
        f"Benchmark settings: dt={config.dt}, steps={config.steps}, "
        f"k_step={config.k_step}, warmup={config.warmup}, repeats={config.repeats}, "
        f"forcing={config.forcing}"
    )

    for dim in dims:
        print("\n" + "=" * 20)
        print(f"Dimension: {dim}")
        try:
            trajectory, t = integrate_lorenz96(dim, config)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to integrate Lorenz-96 for dim={dim}: {exc}")
            continue

        args = (config.forcing,)
        results: List[BackendResult] = []

        for backend in backends:
            try:
                timings = _benchmark_backend(backend, trajectory, t, args, config)
            except Exception as exc:  # noqa: BLE001
                print(f"[{backend.name}] failed for dim={dim}: {exc}")
                continue
            results.append(BackendResult(backend.name, dim, timings))

        for result in results:
            timings = result.timings
            print(
                f"[{result.name}] timings (s): "
                + ", ".join(f"{val:.4f}" for val in timings)
            )
            std = timings.std(ddof=1) if timings.size > 1 else 0.0
            print(
                f"[{result.name}] mean ± std: {timings.mean():.4f} ± {std:.4f} s"
            )

        if len(results) >= 2:
            baseline = results[0]
            for result in results[1:]:
                ratio = result.timings.mean() / baseline.timings.mean()
                print(f"Speed ratio {result.name}/{baseline.name}: {ratio:.2f}x")

        del trajectory, t


def parse_args() -> BenchmarkConfig:
    default_cfg = BenchmarkConfig()
    parser = argparse.ArgumentParser(
        description="Benchmark the Lyapunov analysis backends (Numba vs NumPy)"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=default_cfg.dt,
        help="Time step for the base trajectory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=default_cfg.steps,
        help="Number of integration steps",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=default_cfg.repeats,
        help="Number of timed runs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=default_cfg.warmup,
        help="Warm-up runs for JIT compilation",
    )
    parser.add_argument(
        "--k-step",
        type=int,
        default=default_cfg.k_step,
        help="Stride used in lyap_analysis",
    )
    parser.add_argument(
        "--forcing",
        type=float,
        default=default_cfg.forcing,
        help="Lorenz-96 forcing parameter F",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=default_cfg.perturbation,
        help="Initial perturbation added to the first component",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=None,
        help="State dimensions to benchmark",
    )

    args = parser.parse_args()
    dims = tuple(args.dims) if args.dims is not None else tuple(default_cfg.dims)

    return BenchmarkConfig(
        dt=args.dt,
        steps=args.steps,
        repeats=args.repeats,
        warmup=args.warmup,
        k_step=args.k_step,
        forcing=args.forcing,
        dims=dims,
        perturbation=args.perturbation,
    )


def main() -> None:
    config = parse_args()
    run_benchmark(config)


if __name__ == "__main__":
    main()
