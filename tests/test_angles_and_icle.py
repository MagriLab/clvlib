import numpy as np

from clvlib.numpy import compute_angles, principal_angles, compute_ICLE, lyap_analysis


def test_compute_angles_clamping_and_identity():
    V1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    V2 = (1.0 + 1e-12) * V1
    cos, theta = compute_angles(V1, V2)
    assert np.allclose(cos, np.array([1.0, 1.0]), atol=0, rtol=0)
    assert np.allclose(theta, np.array([0.0, 0.0]), atol=0, rtol=0)


def test_principal_angles_identical_subspaces():
    nt = 3
    n = 3
    m = 2
    Id = np.eye(n, dtype=float)
    V1 = np.repeat(Id[:, :m][None, :, :], nt, axis=0)
    V2 = np.repeat(Id[:, :m][None, :, :], nt, axis=0)
    angles = principal_angles(V1, V2)
    assert angles.shape == (nt, m)
    assert np.allclose(angles, 0.0, atol=1e-12)


def test_icle_with_known_clv_and_constant_jacobian():
    # Diagonal linear system: dx/dt = A x, with constant Jacobian A
    eigs = np.array([0.3, -0.2, 0.0], dtype=float)
    A = np.diag(eigs)

    def jac(t, x):  # noqa: ARG001
        return A

    nt = 10
    n = A.shape[0]
    t = np.linspace(0.0, 1.0, nt)
    sol = np.zeros((nt, n), dtype=float)

    # Provide exact CLVs as identity across time
    CLV_history = np.repeat(np.eye(n, dtype=float)[None, :, :], nt, axis=0)

    icle = compute_ICLE(jac, sol, t, CLV_history)
    assert icle.shape == (nt, n)
    # Each time slice should equal the diagonal of A
    for i in range(nt):
        assert np.allclose(icle[i], eigs, atol=1e-12)


def test_icle_kstep_sampling_matches_expected():
    # Same setup but using k_step sampling
    eigs = np.array([0.2, -0.1], dtype=float)
    A = np.diag(eigs)

    def jac(t, x):  # noqa: ARG001
        return A

    nt = 21
    n = A.shape[0]
    t = np.linspace(0.0, 1.0, nt)
    sol = np.zeros((nt, n), dtype=float)
    k_step = 5
    n_samples = ((nt - 1) // k_step) + 1
    CLV_history = np.repeat(np.eye(n, dtype=float)[None, :, :], n_samples, axis=0)
    icle = compute_ICLE(jac, sol, t, CLV_history, k_step=k_step)
    assert icle.shape == (n_samples, n)
    for i in range(n_samples):
        assert np.allclose(icle[i], eigs, atol=1e-12)


def test_clvs_match_eigenvectors_in_linear_diagonal_case():
    # Diagonal linear system where eigenvectors are canonical basis
    eigs = [0.5, 0.2, -0.1]
    A = np.diag(np.array(eigs, dtype=float))

    def f(t, x):  # noqa: ARG001
        return A @ x

    def Df(t, x):  # noqa: ARG001
        return A

    T = 2.0
    steps = 500
    t = np.linspace(0.0, T, steps + 1)
    n = len(eigs)
    traj = np.zeros((t.size, n), dtype=float)

    LE, LE_hist, BLV_hist, CLV_hist = lyap_analysis(
        f, Df, traj, t, stepper="rk4", k_step=1, ginelli_method="ginelli"
    )

    Id = np.eye(n, dtype=float)
    for k in range(n):
        e = Id[:, k]
        dots = np.einsum("ti,i->t", CLV_hist[:, :, k], e)
        assert np.allclose(np.abs(dots), 1.0, atol=1e-6)
