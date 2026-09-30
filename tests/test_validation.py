"""Regression tests against values validated independently during development
(dense construction, time-domain evolution, exact-U resolvent).
Run:  python tests/test_validation.py      (or: pytest tests/)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lindblad_mbpt import sigma_coefficients, G_exact, G_perturbative

def test_L2_sigma():
    c = sigma_coefficients(2, gamma=0.05, omega=0.3)
    assert np.allclose(np.diag(c["Sigma1"]), 0.5, atol=1e-8)          # Hartree, n=1/2 per spin
    assert abs(c["Sigma2"][0, 0] - (-0.0983746 - 0.02955265j)) < 1e-5
    assert abs(c["Sigma2"][0, 1] - (0.15320501 + 0.01748043j)) < 1e-5

def test_L3_sigma():
    c = sigma_coefficients(3, gamma=0.05, omega=0.3)
    assert np.allclose(np.diag(c["Sigma1"]), 1 / 3, atol=1e-8)
    assert abs(c["Sigma2"][0, 0] - (0.11634976 - 0.03922257j)) < 1e-5
    assert abs(c["Sigma2"][0, 2] - (-0.18386554 + 0.01905415j)) < 1e-5

def test_preconditioner_agrees_with_unpreconditioned():
    a = sigma_coefficients(3, precondition=True)["Sigma2"]
    b = sigma_coefficients(3, precondition=False)["Sigma2"]
    assert np.max(np.abs(a - b)) < 1e-7

def test_exact_reduces_to_G0_at_U0():
    c = sigma_coefficients(3)
    assert np.max(np.abs(G_exact(3, 0.0) - c["G0"])) < 1e-8

def test_benchmark_row_L3():
    # Table 2 row U/t = 0.14 (U = 0.1 with t = 0.7), element G_00
    c = sigma_coefficients(3)
    Gp = G_perturbative(c, 0.1)[0, 0]; Ge = G_exact(3, 0.1)[0, 0]
    assert abs(Gp - (1.721 - 0.218j)) < 2e-3
    assert abs(Ge - (1.732 - 0.224j)) < 2e-3

def test_dephasing_selfenergy_closed_form():
    # Theorem 1: G0 = [w - h + i*gamma/2 + i*eta]^-1 for the free chain
    L, t, g, w, eta = 3, 0.7, 0.05, 0.3, 3e-3
    h = np.zeros((L, L))
    for i in range(L - 1): h[i, i + 1] = h[i + 1, i] = -t
    G0 = np.linalg.inv((w + 1j * (g / 2 + eta)) * np.eye(L) - h)
    assert np.max(np.abs(sigma_coefficients(3)["G0"] - G0)) < 1e-6

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
