import pytest
torch = pytest.importorskip("torch")

from clvlib.pytorch import lyap_exp, lyap_analysis


def _make_linear_system(eigs):
    dtype = torch.float64
    A = torch.diag(torch.tensor(eigs, dtype=dtype))

    def f(t, x, *args):  # noqa: ARG001
        return A @ x

    def Df(t, x, *args):  # noqa: ARG001
        return A

    return A, f, Df


def test_torch_lyap_exp_linear_system_householder():
    eigs = [0.3, -0.2, 0.0]
    A, f, Df = _make_linear_system(eigs)

    T = 5.0
    steps = 2000
    t = torch.linspace(0.0, T, steps + 1, dtype=torch.float64)
    n = len(eigs)
    trajectory = torch.zeros((t.numel(), n), dtype=torch.float64)  # Fixed point trajectory in 0

    LE, LE_history = lyap_exp(f, Df, trajectory, t, stepper="rk4")
    expected = torch.sort(torch.tensor(eigs, dtype=torch.float64), descending=True).values
    assert torch.allclose(torch.sort(LE, descending=True).values, expected, atol=1e-2)
    assert LE_history.shape == (t.numel(), n)


def test_torch_lyap_exp_linear_system_gs_qr():
    eigs = [0.25, -0.1]
    A, f, Df = _make_linear_system(eigs)

    T = 4.0
    steps = 1200
    t = torch.linspace(0.0, T, steps + 1, dtype=torch.float64)
    n = len(eigs)
    trajectory = torch.zeros((t.numel(), n), dtype=torch.float64)

    LE, _ = lyap_exp(f, Df, trajectory, t, stepper="rk4", qr_method="gs")
    expected = torch.sort(torch.tensor(eigs, dtype=torch.float64), descending=True).values
    assert torch.allclose(torch.sort(LE, descending=True).values, expected, atol=2e-2)


def test_torch_lyap_analysis_shapes_kstep():
    eigs = [0.2, -0.05]
    A, f, Df = _make_linear_system(eigs)

    T = 3.0
    steps = 999
    t = torch.linspace(0.0, T, steps + 1, dtype=torch.float64)
    n = len(eigs)
    trajectory = torch.zeros((t.numel(), n), dtype=torch.float64)

    k_step = 7
    LE, LE_hist, BLV_hist, CLV_hist = lyap_analysis(
        f, Df, trajectory, t, stepper="rk4", k_step=k_step
    )
    n_step = ((t.numel() - 1) // k_step) + 1
    assert LE.shape == (n,)
    assert LE_hist.shape == (n_step, n)
    assert BLV_hist.shape == (n_step, n, n)
    assert CLV_hist.shape == (n_step, n, n)


def test_torch_lyap_analysis_ginelli_upwind():
    eigs = [0.15, -0.05]
    A, f, Df = _make_linear_system(eigs)

    T = 2.5
    steps = 800
    t = torch.linspace(0.0, T, steps + 1, dtype=torch.float64)
    n = len(eigs)
    trajectory = torch.zeros((t.numel(), n), dtype=torch.float64)

    LE, LE_hist, BLV_hist, CLV_hist = lyap_analysis(
        f, Df, trajectory, t, stepper="rk4", k_step=1, ginelli_method="upwind"
    )
    assert LE.shape == (n,)
    assert LE_hist.shape == (t.numel(), n)
    assert BLV_hist.shape == (t.numel(), n, n)
    assert CLV_hist.shape == (t.numel(), n, n)
    expected = torch.sort(torch.tensor(eigs, dtype=torch.float64), descending=True).values
    assert torch.allclose(torch.sort(LE, descending=True).values, expected, atol=2e-2)


def test_torch_clvs_equal_eigenvectors_for_diagonal_system():
    # Use diagonal A with descending eigenvalues so column order matches
    eigs = [0.4, 0.1, -0.2]
    A, f, Df = _make_linear_system(eigs)

    T = 2.0
    steps = 600
    t = torch.linspace(0.0, T, steps + 1, dtype=torch.float64)
    n = len(eigs)
    trajectory = torch.zeros((t.numel(), n), dtype=torch.float64)

    LE, LE_hist, BLV_hist, CLV_hist = lyap_analysis(
        f, Df, trajectory, t, stepper="rk4", k_step=1, ginelli_method="ginelli"
    )

    # Canonical eigenvectors for diagonal A
    I = torch.eye(n, dtype=torch.float64)
    # Check each CLV column aligns with the corresponding eigenvector across time (up to sign)
    for k in range(n):
        ev = I[:, k]
        # Dot products over time
        dots = torch.einsum("ti, i -> t", CLV_hist[:, :, k], ev)
        assert torch.allclose(torch.abs(dots), torch.ones_like(dots), atol=1e-6, rtol=0.0)
