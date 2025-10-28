import pytest
from clvlib.pytorch import compute_angles, principal_angles, compute_ICLE, lyap_analysis

torch = pytest.importorskip("torch")


def test_torch_compute_angles_clamping_and_identity():
    dtype = torch.float64
    V1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)
    V2 = (1.0 + 1e-12) * V1
    cos, theta = compute_angles(V1, V2)
    assert torch.allclose(
        cos, torch.tensor([1.0, 1.0], dtype=dtype), atol=0.0, rtol=0.0
    )
    assert torch.allclose(
        theta, torch.tensor([0.0, 0.0], dtype=dtype), atol=0.0, rtol=0.0
    )


def test_torch_principal_angles_identical_subspaces():
    dtype = torch.float64
    nt = 3
    n = 3
    m = 2
    Id = torch.eye(n, dtype=dtype)
    V1 = Id[:, :m].unsqueeze(0).repeat(nt, 1, 1)
    V2 = Id[:, :m].unsqueeze(0).repeat(nt, 1, 1)
    angles = principal_angles(V1, V2)
    assert angles.shape == (nt, m)
    assert torch.allclose(
        angles, torch.zeros((nt, m), dtype=dtype), atol=1e-12, rtol=0.0
    )


def test_torch_icle_with_known_clv_and_constant_jacobian():
    # Diagonal linear system: dx/dt = A x, with constant Jacobian A
    dtype = torch.float64
    eigs = torch.tensor([0.3, -0.2, 0.0], dtype=dtype)
    A = torch.diag(eigs)

    def jac(t, x):  # noqa: ARG001
        return A

    nt = 10
    n = A.shape[0]
    t = torch.linspace(0.0, 1.0, nt, dtype=dtype)
    sol = torch.zeros((nt, n), dtype=dtype)

    # Provide exact CLVs as identity across time
    CLV_history = torch.eye(n, dtype=dtype).unsqueeze(0).repeat(nt, 1, 1)

    icle = compute_ICLE(jac, sol, t, CLV_history)
    assert icle.shape == (nt, n)
    for i in range(nt):
        assert torch.allclose(icle[i], eigs, atol=1e-12, rtol=0.0)


def test_torch_icle_kstep_sampling_matches_expected():
    # Same setup but using k_step sampling
    dtype = torch.float64
    eigs = torch.tensor([0.2, -0.1], dtype=dtype)
    A = torch.diag(eigs)

    def jac(t, x):  # noqa: ARG001
        return A

    nt = 21
    n = A.shape[0]
    t = torch.linspace(0.0, 1.0, nt, dtype=dtype)
    sol = torch.zeros((nt, n), dtype=dtype)
    k_step = 5
    n_samples = ((nt - 1) // k_step) + 1
    CLV_history = torch.eye(n, dtype=dtype).unsqueeze(0).repeat(n_samples, 1, 1)
    icle = compute_ICLE(jac, sol, t, CLV_history, k_step=k_step)
    assert icle.shape == (n_samples, n)
    for i in range(n_samples):
        assert torch.allclose(icle[i], eigs, atol=1e-12, rtol=0.0)


def test_torch_clvs_match_eigenvectors_in_linear_diagonal_case():
    # Diagonal linear system where eigenvectors are canonical basis
    dtype = torch.float64
    eigs = torch.tensor([0.5, 0.2, -0.1], dtype=dtype)
    A = torch.diag(eigs)

    def f(t, x):  # noqa: ARG001
        return A @ x

    def Df(t, x):  # noqa: ARG001
        return A

    T = 2.0
    steps = 500
    t = torch.linspace(0.0, T, steps + 1, dtype=dtype)
    n = A.shape[0]
    traj = torch.zeros((t.numel(), n), dtype=dtype)

    LE, LE_hist, BLV_hist, CLV_hist = lyap_analysis(
        f, Df, traj, t, stepper="rk4", k_step=1, ginelli_method="ginelli"
    )

    Id = torch.eye(n, dtype=dtype)
    for k in range(n):
        e = Id[:, k]
        dots = torch.einsum("ti,i->t", CLV_hist[:, :, k], e)
        assert torch.allclose(
            torch.abs(dots), torch.ones_like(dots), atol=1e-6, rtol=0.0
        )
