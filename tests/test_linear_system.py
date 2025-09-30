import numpy as np

from clvlib.numpy import lyap_exp, lyap_analysis


def _make_linear_system(eigs):
    A = np.diag(np.array(eigs, dtype=float))

    def f(t, x, *args):  # noqa: ARG001
        return A @ x

    def Df(t, x, *args):  # noqa: ARG001
        return A

    return A, f, Df


def test_lyap_exp_linear_system_householder():
    eigs = [0.3, -0.2, 0.0]
    A, f, Df = _make_linear_system(eigs)

    T = 5.0
    steps = 2000
    t = np.linspace(0.0, T, steps + 1)
    n = len(eigs)
    trajectory = np.zeros((t.size, n), dtype=float)

    LE, LE_history = lyap_exp(f, Df, trajectory, t, stepper="rk4")
    expected = np.sort(np.array(eigs))[::-1]
    assert np.allclose(np.sort(LE)[::-1], expected, atol=1e-2)
    assert LE_history.shape == (t.size, n)


def test_lyap_exp_linear_system_gs_qr():
    eigs = [0.25, -0.1]
    A, f, Df = _make_linear_system(eigs)

    T = 4.0
    steps = 1200
    t = np.linspace(0.0, T, steps + 1)
    n = len(eigs)
    trajectory = np.zeros((t.size, n), dtype=float)

    LE, _ = lyap_exp(f, Df, trajectory, t, stepper="rk4", qr_method="gs")
    expected = np.sort(np.array(eigs))[::-1]
    assert np.allclose(np.sort(LE)[::-1], expected, atol=2e-2)


def test_lyap_analysis_shapes_kstep():
    eigs = [0.2, -0.05]
    A, f, Df = _make_linear_system(eigs)

    T = 3.0
    steps = 999
    t = np.linspace(0.0, T, steps + 1)
    n = len(eigs)
    trajectory = np.zeros((t.size, n), dtype=float)

    k_step = 5
    LE, LE_hist, BLV_hist, CLV_hist = lyap_analysis(
        f, Df, trajectory, t, stepper="rk4", k_step=k_step
    )
    n_step = ((t.size - 1) // k_step) + 1
    assert LE.shape == (n,)
    assert LE_hist.shape == (n_step, n)
    assert BLV_hist.shape == (n_step, n, n)
    assert CLV_hist.shape == (n_step, n, n)

