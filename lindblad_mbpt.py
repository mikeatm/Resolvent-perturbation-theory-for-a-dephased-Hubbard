"""
lindblad_mbpt.py -- resolvent perturbation theory for the dephased Hubbard chain.

Companion code for: M. Atambo, "Resolvent perturbation theory for a dephased
Hubbard chain: exact benchmarks and the failure of naive single-species
diagrammatics".

Model (open chain, L sites, spin-1/2 fermions):
    H = -t sum_{i,s} (c+_{i s} c_{i+1 s} + h.c.) + U sum_i n_{i up} n_{i dn}
    dissipation: jump operators L_i = sqrt(gamma) * n_i^tot   (NO extra gamma
    in front of the dissipator).  Note L_i^dag L_i = gamma * (n_i^tot)^2, and
    (n_i^tot)^2 != n_i^tot because n_i^tot has eigenvalue 2.

Conventions: column-stacking vectorization, vec(A X B) = (B^T (x) A) vec(X);
Green's function G0^{-1}(w) = w - h; resolvent R0 = -(i*Omega + L0)^{-1},
Omega = w + i*eta.

Everything is sector-projected: (N_up, N_dn) is conserved, so the resolvent
acting on c^dag rho / rho c^dag lives on two coherence blocks.

Public functions
    sigma_coefficients(L, ...)   -> dict with G0,G1,G2,Sigma1,Sigma2
    G_exact(L, U, ...)           -> exact (non-perturbative) G(w) at finite U
    G_perturbative(...)          -> G0 + U G1 + U^2 G2
No code runs at import time.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["sigma_coefficients", "G_exact", "G_perturbative", "build_operators"]

_I2 = sp.eye(2, format="csr", dtype=complex)
_c = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))
_P = _I2 - 2 * (_c.conj().T @ _c)


def _skron(*Ms):
    out = Ms[0]
    for M in Ms[1:]:
        out = sp.kron(out, M, format="csr")
    return out


def _vec(M):
    return M.reshape(-1, order="F")


def _unvec(v, nr, nc):
    return v.reshape(nr, nc, order="F")


def build_operators(L):
    """Jordan-Wigner fermion operators; mode order (1up,1dn,2up,2dn,...)."""
    nmodes = 2 * L
    cm = [_skron(*([_P] * k + [_c] + [_I2] * (nmodes - k - 1))) for k in range(nmodes)]
    cu = [cm[2 * i] for i in range(L)]
    cd = [cm[2 * i + 1] for i in range(L)]
    cud = [op.conj().T.tocsr() for op in cu]
    cdd = [op.conj().T.tocsr() for op in cd]
    nu = [(cud[i] @ cu[i]).tocsr() for i in range(L)]
    nd = [(cdd[i] @ cd[i]).tocsr() for i in range(L)]
    ntot = [(nu[i] + nd[i]).tocsr() for i in range(L)]
    return dict(cu=cu, cd=cd, cud=cud, cdd=cdd, nu=nu, nd=nd, ntot=ntot, dimP=2 ** nmodes)


def _sector_basis(ops, Nup, Ndn):
    du = np.array(sum(ops["nu"]).diagonal()).real
    dd = np.array(sum(ops["nd"]).diagonal()).real
    return [i for i in range(ops["dimP"]) if abs(du[i] - Nup) < 1e-9 and abs(dd[i] - Ndn) < 1e-9]


def _restrict(X, R, C=None):
    C = R if C is None else C
    return X.tocsr()[R, :][:, C]


def _block_generator(H0, HU, ntot, R, C, gamma, U, L):
    """Liouvillian on the coherence block rho[R, C] (column-stacked)."""
    dR, dC = len(R), len(C)
    IR = sp.eye(dR, format="csr", dtype=complex)
    IC = sp.eye(dC, format="csr", dtype=complex)
    H0R, H0C = _restrict(H0, R), _restrict(H0, C)
    HUR, HUC = _restrict(HU, R), _restrict(HU, C)
    Lg = -1j * (sp.kron(IC, H0R, format="csr") - sp.kron(H0C.T, IR, format="csr"))
    Lg = Lg - 1j * U * (sp.kron(IC, HUR, format="csr") - sp.kron(HUC.T, IR, format="csr"))
    for i in range(L):
        nR, nC = _restrict(ntot[i], R), _restrict(ntot[i], C)
        n2R, n2C = _restrict(ntot[i] @ ntot[i], R), _restrict(ntot[i] @ ntot[i], C)
        Lg = Lg + gamma * (sp.kron(nC.conj(), nR, format="csr")
                           - 0.5 * sp.kron(IC, n2R, format="csr")
                           - 0.5 * sp.kron(n2C.T, IR, format="csr"))
    return Lg.tocsc()


def _gcrot(A, b, M, tol):
    try:
        x, info = spla.gcrotmk(A, b, rtol=tol, m=20, k=10, M=M)
    except TypeError:  # older SciPy
        x, info = spla.gcrotmk(A, b, tol=tol, m=20, k=10, M=M)
    return x


def _mf_preconditioner(H0, ntot, R, C, gamma, omega, eta, n_avg, L):
    """Kronecker-aware preconditioner: drop the density-density 'bubble' term
    (keeping its mean-field part) and solve the Kronecker-sum system exactly
    with a dense Sylvester equation of dimension O(d0)."""
    dR, dC = len(R), len(C)
    H0R = _restrict(H0, R).toarray(); H0C = _restrict(H0, C).toarray()
    N2R = sum(_restrict(ntot[i] @ ntot[i], R) for i in range(L)).toarray()
    N2C = sum(_restrict(ntot[i] @ ntot[i], C) for i in range(L)).toarray()
    NR = sum(_restrict(ntot[i], R) for i in range(L)).toarray()
    NC = sum(_restrict(ntot[i], C) for i in range(L)).toarray()
    Om = omega + 1j * eta
    A = -1j * H0R - 0.5 * gamma * N2R + gamma * n_avg * NR + 1j * Om * np.eye(dR, dtype=complex)
    B = 1j * H0C.T - 0.5 * gamma * N2C.T + gamma * n_avg * NC.T
    Bt = B.T.copy()
    return lambda v: _vec(la.solve_sylvester(A, Bt, _unvec(v, dR, dC)))


def _setup(L, t_hop, gamma, Nf):
    ops = build_operators(L)
    H0 = -t_hop * sum(ops["cud"][i] @ ops["cu"][i + 1] + ops["cud"][i + 1] @ ops["cu"][i]
                      + ops["cdd"][i] @ ops["cd"][i + 1] + ops["cdd"][i + 1] @ ops["cd"][i]
                      for i in range(L - 1)).tocsr()
    HU = sum(ops["nu"][i] @ ops["nd"][i] for i in range(L)).tocsr()
    Nf = (1 if L <= 3 else L // 2) if Nf is None else Nf
    S0 = _sector_basis(ops, Nf, Nf)
    SR = _sector_basis(ops, Nf + 1, Nf)
    SL = _sector_basis(ops, Nf - 1, Nf)
    return ops, H0, HU, Nf, S0, SR, SL


def _assembler(ops, S0, SR, SL, L):
    cud_a = [_restrict(ops["cud"][i], SR, S0) for i in range(L)]
    cu_a = [_restrict(ops["cu"][i], S0, SR) for i in range(L)]
    cud_b = [_restrict(ops["cud"][i], S0, SL) for i in range(L)]
    cu_b = [_restrict(ops["cu"][i], SL, S0) for i in range(L)]

    def assemble(applyA, applyB, rho):
        """G_ij = -i[Tr(c_i R(c_j^dag rho)) + Tr(R(rho c_j^dag) c_i)];
        each solve depends only on j, so it is cached per j."""
        pa, pb = {}, {}
        for j in range(L):
            pa[j] = _unvec(applyA(_vec(cud_a[j] @ rho)), len(SR), len(S0))
            pb[j] = _unvec(applyB(_vec(rho @ cud_b[j])), len(S0), len(SL))
        G = np.zeros((L, L), dtype=complex)
        for i in range(L):
            for j in range(L):
                G[i, j] = -1j * (np.trace(cu_a[i] @ pa[j]) + np.trace(pb[j] @ cu_b[i]))
        return G
    return assemble


def sigma_coefficients(L, t_hop=0.7, gamma=0.05, omega=0.3, eta=3e-3,
                       Nf=None, tol=1e-10, precondition=True, verbose=False):
    """Green's-function coefficients G0,G1,G2 and self-energies Sigma1,Sigma2
    (resolvent-defined, retarded) about the maximally mixed sector state."""
    ops, H0, HU, Nf, S0, SR, SL = _setup(L, t_hop, gamma, Nf)
    n_avg = Nf / L
    nt = ops["ntot"]
    if verbose:
        print(f"L={L}  sector dims: d0={len(S0)} raised={len(SR)} lowered={len(SL)}")
    L0_R = _block_generator(H0, HU, nt, SR, S0, gamma, 0.0, L)
    L0_L = _block_generator(H0, HU, nt, S0, SL, gamma, 0.0, L)
    LU_R = _block_generator(H0, HU, nt, SR, S0, gamma, 1.0, L) - L0_R
    LU_L = _block_generator(H0, HU, nt, S0, SL, gamma, 1.0, L) - L0_L
    L00 = _block_generator(H0, HU, nt, S0, S0, gamma, 0.0, L)
    LU00 = _block_generator(H0, HU, nt, S0, S0, gamma, 1.0, L) - L00

    rho0 = np.eye(len(S0), dtype=complex) / len(S0)
    # stationary-state corrections: rhs vanishes exactly since [H_U, rho0] = 0
    rhs1 = -(LU00 @ _vec(rho0))
    if np.linalg.norm(rhs1) < 1e-12:
        rho1 = np.zeros_like(rhs1); rho2 = np.zeros_like(rhs1)
    else:
        A00 = (L00 + 1e-7j * sp.eye(L00.shape[0], format="csc", dtype=complex)).tocsc()
        M00 = None
        if precondition:
            M00 = spla.LinearOperator(A00.shape, _mf_preconditioner(H0, nt, S0, S0, gamma, 0.0, 1e-7, n_avg, L))
        rho1 = _gcrot(A00, rhs1, M00, tol)
        rho2 = _gcrot(A00, -(LU00 @ rho1), M00, tol)
    rho1m = _unvec(rho1, len(S0), len(S0)); rho2m = _unvec(rho2, len(S0), len(S0))

    Om = omega + 1j * eta
    A = (1j * Om * sp.eye(L0_R.shape[0], format="csc", dtype=complex) + L0_R).tocsc()
    B = (1j * Om * sp.eye(L0_L.shape[0], format="csc", dtype=complex) + L0_L).tocsc()
    Ma = Mb = None
    if precondition:
        Ma = spla.LinearOperator(A.shape, _mf_preconditioner(H0, nt, SR, S0, gamma, omega, eta, n_avg, L))
        Mb = spla.LinearOperator(B.shape, _mf_preconditioner(H0, nt, S0, SL, gamma, omega, eta, n_avg, L))
    R0a = lambda v: -_gcrot(A, v, Ma, tol)
    R0b = lambda v: -_gcrot(B, v, Mb, tol)
    R1a = lambda v: R0a(LU_R @ R0a(v)); R1b = lambda v: R0b(LU_L @ R0b(v))
    R2a = lambda v: R0a(LU_R @ R1a(v)); R2b = lambda v: R0b(LU_L @ R1b(v))

    asm = _assembler(ops, S0, SR, SL, L)
    G0 = asm(R0a, R0b, rho0)
    G1 = asm(R1a, R1b, rho0) + asm(R0a, R0b, rho1m)
    G2 = asm(R2a, R2b, rho0) + asm(R1a, R1b, rho1m) + asm(R0a, R0b, rho2m)
    G0i = np.linalg.inv(G0)
    S1 = G0i @ G1 @ G0i
    S2 = -G0i @ G1 @ G0i @ G1 @ G0i + G0i @ G2 @ G0i
    return dict(G0=G0, G1=G1, G2=G2, Sigma1=S1, Sigma2=S2)


def G_perturbative(coef, U):
    return coef["G0"] + U * coef["G1"] + U ** 2 * coef["G2"]


def G_exact(L, U, t_hop=0.7, gamma=0.05, omega=0.3, eta=3e-3, Nf=None):
    """Non-perturbative G(w) at finite U about the same reference state rho0
    (exactly stationary at every U because [H_U, rho0] = 0). Sparse LU; small L."""
    ops, H0, HU, Nf, S0, SR, SL = _setup(L, t_hop, gamma, Nf)
    nt = ops["ntot"]
    Om = omega + 1j * eta
    LR = _block_generator(H0, HU, nt, SR, S0, gamma, U, L)
    LL = _block_generator(H0, HU, nt, S0, SL, gamma, U, L)
    A = (1j * Om * sp.eye(LR.shape[0], format="csc", dtype=complex) + LR).tocsc()
    B = (1j * Om * sp.eye(LL.shape[0], format="csc", dtype=complex) + LL).tocsc()
    luA, luB = spla.splu(A), spla.splu(B)
    rho0 = np.eye(len(S0), dtype=complex) / len(S0)
    asm = _assembler(ops, S0, SR, SL, L)
    return asm(lambda v: -luA.solve(v), lambda v: -luB.solve(v), rho0)
