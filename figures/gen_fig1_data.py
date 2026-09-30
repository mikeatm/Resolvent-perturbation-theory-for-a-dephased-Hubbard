"""
SELF-CONTAINED script to generate Figure 1 (Sigma^(2)_00(omega), L=3, gamma=0.05).
No external file dependencies. All names local to gen_fig1() to avoid any
global-namespace collision if run alongside other scripts in the same kernel.

One-time cost: diagonalizing the 4096x4096 Liouvillian block takes ~60-100s
(reported in the paper, Table 1). After that, each omega point is fast.
"""
import numpy as np
import time

def gen_fig1(omegas, t_hop=0.7, gamma=0.05, save_path='fig1_data.npz'):
    I2 = np.eye(2)
    c  = np.array([[0,1],[0,0]], dtype=complex)
    P  = I2 - 2*(c.conj().T @ c)
    def kron(*Ms):
        out = Ms[0]
        for M in Ms[1:]: out = np.kron(out, M)
        return out

    L = 3
    nmodes = 2*L
    def mode_op(k):
        ops = [P]*k + [c] + [I2]*(nmodes-k-1)
        return kron(*ops)
    cm  = [mode_op(k) for k in range(nmodes)]
    cu  = [cm[2*i] for i in range(L)]
    cd  = [cm[2*i+1] for i in range(L)]
    cud = [op.conj().T for op in cu]
    cdd = [op.conj().T for op in cd]
    nu  = [cud[i]@cu[i] for i in range(L)]
    nd  = [cdd[i]@cd[i] for i in range(L)]
    ntot = [nu[i]+nd[i] for i in range(L)]
    dimP = 2**nmodes; dimS = dimP*dimP; Iid = np.eye(dimP)
    print(f"L={L}, dimP={dimP}, dimS={dimS}")

    def vec(M):  return M.reshape(-1, order='F')
    def unvec(v,d): return v.reshape(d,d,order='F')
    def liouvillian_H(Hm):
        return -1j*(kron(Iid,Hm) - kron(Hm.T, Iid))
    def liouvillian_dissipator(Lop):
        Ld = Lop.conj().T
        return kron(Lop.conj(),Lop) - 0.5*kron(Iid, Ld@Lop) - 0.5*kron((Ld@Lop).T, Iid)

    H0 = -t_hop*sum(cud[i]@cu[i+1]+cud[i+1]@cu[i]+cdd[i]@cd[i+1]+cdd[i+1]@cd[i] for i in range(L-1))
    Lcal0 = liouvillian_H(H0)
    for i in range(L):
        Lcal0 = Lcal0 + liouvillian_dissipator(np.sqrt(gamma)*ntot[i])
    H_U = sum(nu[i]@nd[i] for i in range(L))
    Lcal1 = liouvillian_H(H_U)

    Ntot_up, Ntot_dn = sum(nu), sum(nd)
    sector = [idx for idx in range(dimP) if
              abs(np.trace(Ntot_up@np.diag((np.arange(dimP)==idx).astype(float))).real-1)<1e-9 and
              abs(np.trace(Ntot_dn@np.diag((np.arange(dimP)==idx).astype(float))).real-1)<1e-9]
    rho_ss0 = np.zeros((dimP,dimP), dtype=complex)
    for idx in sector: rho_ss0[idx,idx] = 1.0/len(sector)
    resid = np.max(np.abs(Lcal0@vec(rho_ss0)))
    print("steady-state residual:", resid); assert resid < 1e-8

    print("diagonalizing Lcal0 (one-time, ~60-100s)...")
    t0=time.time()
    Lcal0_pinv = np.linalg.pinv(Lcal0, rcond=1e-10)
    rho1_vec = Lcal0_pinv @ (-(Lcal1@vec(rho_ss0)))
    rho2_vec = Lcal0_pinv @ (-(Lcal1@rho1_vec))
    print("Tr(rho1):", np.trace(unvec(rho1_vec,dimP)))

    evals, R = np.linalg.eig(Lcal0)
    Rinv = np.linalg.inv(R)
    M = Rinv @ Lcal1 @ R
    print(f"diagonalization done in {time.time()-t0:.1f}s")

    ops  = {i: cu[i]  for i in range(L)}
    opsd = {i: cud[i] for i in range(L)}
    sites = list(range(L))
    eta = 3e-3

    def apply_R0(v,d): return -(R @ (d * (Rinv @ v)))
    def apply_R1(v,d): return R @ (d * (M @ (d * (Rinv @ v))))
    def apply_R2(v,d): return -(R @ (d * (M @ (d * (M @ (d * (Rinv @ v)))))))

    def assemble_G(apply_fn, d, rho_mat):
        Gm = np.zeros((L,L), dtype=complex)
        for ii,i in enumerate(sites):
            for jj,j in enumerate(sites):
                s1 = vec(opsd[j]@rho_mat); t1 = np.trace(ops[i]@unvec(apply_fn(s1,d),dimP))
                s2 = vec(rho_mat@opsd[j]); t2 = np.trace(unvec(apply_fn(s2,d),dimP)@ops[i])
                Gm[ii,jj] = -1j*(t1+t2)
        return Gm

    def Sigma_at(omega):
        Omega = omega+1j*eta
        d = 1.0/(1j*Omega+evals)
        rhos = {0: rho_ss0, 1: unvec(rho1_vec,dimP), 2: unvec(rho2_vec,dimP)}
        applies = {0: apply_R0, 1: apply_R1, 2: apply_R2}
        def G_order(order):
            Gm = np.zeros((L,L),dtype=complex)
            for k in range(order+1):
                Gm += assemble_G(applies[k], d, rhos[order-k])
            return Gm
        G0,G1,G2 = G_order(0),G_order(1),G_order(2)
        G0inv = np.linalg.inv(G0)
        Sigma2 = -G0inv@G1@G0inv@G1@G0inv + G0inv@G2@G0inv
        return Sigma2

    S00 = np.zeros(len(omegas), dtype=complex)
    S01 = np.zeros(len(omegas), dtype=complex)
    t0=time.time()
    for k,om in enumerate(omegas):
        S2 = Sigma_at(om)
        S00[k]=S2[0,0]; S01[k]=S2[0,1]
        if k%10==0: print(f"point {k}/{len(omegas)}, t={time.time()-t0:.1f}s", flush=True)
    print(f"done in {time.time()-t0:.1f}s")
    np.savez(save_path, omegas=omegas, S00=S00, S01=S01)
    print("sanity check @ omega=0.3:", S00[np.argmin(np.abs(omegas-0.3))], " expect 0.11635-0.03922j")
    return omegas, S00, S01

if __name__ == '__main__':
    omegas = np.linspace(-1.5, 1.5, 41)
    gen_fig1(omegas)
