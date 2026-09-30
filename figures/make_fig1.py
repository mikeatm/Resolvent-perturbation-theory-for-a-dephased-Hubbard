"""Make Fig. 1 from fig1_data.npz (produced by gen_fig1_data.py)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
d = np.load("fig1_data.npz")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(d["omegas"], d["S00"].real, color="#1f77b4", lw=1.3, label=r"$\mathrm{Re}\,\Sigma^{(2)}_{00}$")
ax.plot(d["omegas"], d["S00"].imag, color="#d62728", lw=1.3, label=r"$\mathrm{Im}\,\Sigma^{(2)}_{00}$")
ax.axhline(0, color="gray", lw=.5); ax.axvline(0, color="gray", lw=.5)
ax.set_xlabel(r"$\omega/t$"); ax.set_ylabel(r"$\Sigma^{(2)}_{00}(\omega)$")
ax.legend(frameon=False, loc="upper right"); fig.tight_layout(pad=.3)
fig.savefig("fig1_sigma_omega.pdf")
