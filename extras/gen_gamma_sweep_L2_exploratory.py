"""
SELF-CONTAINED script to generate Figure 2 (gamma-sweep, L=2). No external
file dependencies. All logic local to gen_fig2() to avoid namespace collision
with other scripts (e.g. the L=3 figure 1 script) run in the same kernel.
"""
import numpy as np
import time

def gen_fig2(gammas, t_hop=0.7, omega_test=0.3, save_path='fig2_data.npz'):
    I2 = np.eye(2)
    c  = np.array([[0,1],[0,0]], dtype=complex)
    P  = I2 - 2*(c.conj().T @ c)
    def kron(*Ms):
        out = Ms[0]
        for M in Ms[1:]: out = np.kron(out, M)
        return out
    nmodes = 4
    def mode_op(k):
        ops = [P]*k + [c] + [I2]*(nmodes-k-1)
        return kron(*ops)
    cm = [mode_op(k) for k in range(nmodes)]
    cmd = [op.conj().T for op in cm]
    c1u,c1d_,c2u,c2d_ = cm
    c1ud,c1dd,c2ud,c2dd = cmd
    n1u,n1d = c1ud@c1u, c1dd@c1d_
    n2u,n2d = c2ud@c2u, c2dd@c2d_
    n1tot,n2tot = n1u+n1d, n2u+n2d
    dimP = 16; dimS = 256; Iid = np.eye(dimP)

    def vec(M): return M.reshape(-1, order='F')
    def unvec(v,d): return v.reshape(d,d,order='F')
    def liouvillian_H(Hm):
        return -1j*(kron(Iid,Hm) - kron(Hm.T, Iid))
    def liouvillian_dissipator(L):
        Ld = L.conj().T
        return kron(L.conj(),L) - 0.5*kron(Iid, Ld@L) - 0.5*kron((Ld@L).T, Iid)

    H0 = -t_hop*(c1ud@c2u + c2ud@c1u + c1dd@c2d_ + c2dd@c1d_)
    H_U = n1u@n1d + n2u@n2d
    Lcal1 = liouvillian_H(H_U)

    Ntot_up, Ntot_dn = n1u+n2u, n1d+n2d
    sector = [idx for idx in range(dimP) if
              abs(np.trace(Ntot_up@np.diag((np.arange(dimP)==idx).astype(float))).real-1)<1e-9 and
              abs(np.trace(Ntot_dn@np.diag((np.arange(dimP)==idx).astype(float))).real-1)<1e-9]
    rho_ss0 = np.zeros((dimP,dimP), dtype=complex)
    for idx in sector: rho_ss0[idx,idx] = 1.0/len(sector)

    ops = {0:c1u,1:c2u}; opsd = {0:c1ud,1:c2ud}
    def assemble_G(R_used, rho_mat):
        Gm = np.zeros((2,2), dtype=complex)
        for i in range(2):
            for j in range(2):
                s1 = vec(opsd[j]@rho_mat); t1=np.trace(ops[i]@unvec(R_used@s1,dimP))
                s2 = vec(rho_mat@opsd[j]); t2=np.trace(unvec(R_used@s2,dimP)@ops[i])
                Gm[i,j] = -1j*(t1+t2)
        return Gm

    def Sigma2_for_gamma(gamma, eta_scale=0.06):
        Lcal0 = liouvillian_H(H0) + liouvillian_dissipator(np.sqrt(gamma)*n1tot) + liouvillian_dissipator(np.sqrt(gamma)*n2tot)
        resid = np.max(np.abs(Lcal0@vec(rho_ss0)))
        assert resid < 1e-8, f"steady-state residual too large: {resid}"
        Lcal0_pinv = np.linalg.pinv(Lcal0, rcond=1e-10)
        rho1_vec = Lcal0_pinv @ (-(Lcal1@vec(rho_ss0)))
        rho2_vec = Lcal0_pinv @ (-(Lcal1@rho1_vec))

        evals, R = np.linalg.eig(Lcal0)
        Rinv = np.linalg.inv(R)
        M = Rinv @ Lcal1 @ R
        eta = max(gamma*eta_scale, 1e-3)
        Omega = omega_test + 1j*eta
        d = 1.0/(1j*Omega + evals)
        R0 = -(R*d) @ Rinv
        R1 = (R*d) @ M @ (d[:,None]*Rinv)
        R2 = -(R*d) @ M @ (d[:,None]*M) @ (d[:,None]*Rinv)

        Rs={0:R0,1:R1,2:R2}; rhos={0:vec(rho_ss0),1:rho1_vec,2:rho2_vec}
        def G_order(order):
            Gm = np.zeros((2,2),dtype=complex)
            for k in range(order+1):
                Gm += assemble_G(Rs[k], unvec(rhos[order-k],dimP))
            return Gm
        G0,G1,G2 = G_order(0),G_order(1),G_order(2)
        G0inv = np.linalg.inv(G0)
        return -G0inv@G1@G0inv@G1@G0inv + G0inv@G2@G0inv

    S00 = np.zeros(len(gammas), dtype=complex)
    S01 = np.zeros(len(gammas), dtype=complex)
    t0=time.time()
    for k,g in enumerate(gammas):
        S2 = Sigma2_for_gamma(g)
        S00[k]=S2[0,0]; S01[k]=S2[0,1]
    print(f"{len(gammas)} points in {time.time()-t0:.2f}s")
    np.savez(save_path, gammas=gammas, S00=S00, S01=S01)
    print("sanity: gamma=0.005:", S00[0], " gamma=1.0:", S00[-1])
    return gammas, S00, S01

if __name__ == '__main__':
    gammas = np.concatenate([np.linspace(0.005,0.1,20), np.linspace(0.12,1.0,15)])
    gen_fig2(gammas)
