# lindblad_mbpt

Code accompanying *Resolvent perturbation theory for a dephased Hubbard chain:
exact benchmarks and the failure of naive single-species diagrammatics*
(M. Atambo). Zenodo DOI: `10.5281/zenodo.XXXXXXX`.

## What it computes
For an open Hubbard chain with jump operators `L_i = sqrt(gamma) n_i^tot`
(no extra factor of gamma in front of the dissipator), it returns the exact
Green's-function coefficients `G0, G1, G2` in powers of `U` about the maximally
mixed `(N_up,N_dn)`-sector state, the resolvent-defined retarded self-energies
`Sigma1, Sigma2`, and the exact finite-`U` Green's function used for the
benchmark (Table 2). `[H_U, rho0] = 0` makes `rho0` stationary at every `U`.

## Quick start
```python
from lindblad_mbpt import sigma_coefficients, G_exact, G_perturbative
c = sigma_coefficients(L=4, gamma=0.05, omega=0.3)   # t = 0.7 by default
print(c["Sigma2"])
print(G_perturbative(c, 0.1)[0,0], G_exact(3, 0.1)[0,0])
```
Filling: `N_up = N_dn = 1` for `L <= 3`, `L//2` for `L >= 4` (override with `Nf=`).

## Tests
Tested with Python 3.12, NumPy 2.4, SciPy 1.17.
`python tests/test_validation.py` (about 1 min). They check `L=2,3` values,
agreement with/without the preconditioner, `G_exact(U=0) = G0`, the `L=3`
Table 2 row, and the closed form `G0 = [w - h + i gamma/2]^-1` (Theorem 1).
Reference timing on one CPU core: `L=5` about 20 s (sandbox machine).

## Figures
`figures/gen_fig1_data.py` builds Fig. 1 data from scratch (one-time dense
diagonalization of the `L=3` block, minutes), then `figures/make_fig1.py`.
`extras/` holds an exploratory `gamma` sweep that is **not** used in the paper.

## Caveats (also stated in the paper)
* GCROT run times varied by up to ~40x across environments/SciPy versions;
  timings are indicative, not guaranteed.
* The Lehmann representation assumes a diagonalizable generator.
* The steady state is not unique in general (see paper, Sec. 5); results are
  defined relative to the specified reference state.
* `L=6` takes on the order of 20 minutes and several GB of RAM.
