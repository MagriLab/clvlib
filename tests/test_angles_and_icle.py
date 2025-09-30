import numpy as np

from clvlib.numpy import compute_angles, principal_angles, compute_ICLE


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
    I = np.eye(n, dtype=float)
    V1 = np.repeat(I[:, :m][None, :, :], nt, axis=0)
    V2 = np.repeat(I[:, :m][None, :, :], nt, axis=0)
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

