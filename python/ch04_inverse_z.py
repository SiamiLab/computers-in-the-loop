"""Chapter 4 companion: difference equations and the inverse z-transform.

Reproduces:
  * one convolution two ways: step * 0.5^k = 2 - 0.5^k;
  * m(k) = e(k) - e(k-1) - m(k-1) with a step input: m(k) = (-1)^k;
  * x(k+2) - 3x(k+1) + 2x(k) = 0, x(0) = 0, x(1) = 1: X(z) = z/((z-1)(z-2)),
    inverted three ways (partial fractions, long division, residues): 2^k - 1;
  * a repeated pole: z/((z-1)^2 (z-2)) -> 2^k - k - 1;
  * no factor z in the numerator: 1/(z - 0.5);
  * Parseval: sum (0.5^k)^2 = 4/3, in time and in frequency;
  * the exit ticket: z^2/((z-1)(z-0.2)) -> 5/4 - (1/4) 0.2^k.

Run:  python3 ch04_inverse_z.py      (figures go to ./figures)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import quad

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
N = 10
k = np.arange(N)


def long_division(num, den, n):
    """First n samples of E(z) = num/den (descending powers of z, deg num <= deg den):
    the impulse response of the filter with the same coefficients in z^-1."""
    num = np.r_[np.zeros(len(den) - len(num)), num]
    imp = np.r_[1.0, np.zeros(n - 1)]
    return signal.lfilter(num, den, imp)


def residues_over_z(num, den):
    """Partial fractions of E(z)/z: returns residues r_i and poles p_i (simple poles)."""
    r, p, _ = signal.residue(num, np.polymul(den, [1, 0]))
    return r, p


# ---------------------------------------------------------------- convolution
f_time = np.convolve(np.ones(N), 0.5**k)[:N]
f_z = long_division([1, 0, 0], np.polymul([1, -1], [1, -0.5]), N)    # z^2/((z-1)(z-0.5))
print("Convolution of the unit step with 0.5^k")
print("  direct sum      :", np.round(f_time, 4))
print("  from E1(z)E2(z) :", np.round(f_z, 4))
print("  2 - 0.5^k       :", np.round(2 - 0.5**k, 4))

# ---------------------------------------------------------------- difference equation, step input
e = np.ones(N)
m = np.zeros(N)
for kk in range(N):
    m[kk] = e[kk] - (e[kk - 1] if kk else 0) - (m[kk - 1] if kk else 0)
print("\nm(k) = e(k) - e(k-1) - m(k-1), unit step:", m.astype(int), " (M(z) = z/(z+1): (-1)^k)")

# ---------------------------------------------------------------- initial conditions, three inversions
x = np.zeros(N)
x[1] = 1
for kk in range(N - 2):
    x[kk + 2] = 3 * x[kk + 1] - 2 * x[kk]
num, den = [1, 0], [1, -3, 2]                      # X(z) = z/((z-1)(z-2))
r, p = residues_over_z(num, den)
pf = sum(ri * pi**k for ri, pi in zip(r, p) if abs(pi) > 1e-12)
print("\nx(k+2) - 3x(k+1) + 2x(k) = 0, x(0) = 0, x(1) = 1")
print("  recursion       :", x.astype(int))
print("  partial fractions of X(z)/z: residues", np.round(r, 6), "at poles", np.round(p, 6))
print("  sum k_i p_i^k   :", np.round(pf.real).astype(int))
print("  long division   :", np.round(long_division(num, den, N)).astype(int))
res = [(2.0**kk) / (2 - 1) + 1.0 / (1 - 2) for kk in k]   # residues of z^k/((z-1)(z-2)) at 2 and 1
print("  residues        :", np.round(res).astype(int), " -> x(k) = 2^k - 1")

# ---------------------------------------------------------------- repeated pole
num, den = [1, 0], np.polymul(np.polymul([1, -1], [1, -1]), [1, -2])
r, p, _ = signal.residue(num, np.polymul(den, [1, 0]))
print("\nRepeated pole, E(z) = z/((z-1)^2 (z-2))")
print("  E(z)/z: residues", np.round(r, 6) + 0.0, "poles", np.round(p, 6) + 0.0)
print("  (the repeated pole is listed twice: k11 for 1/(z-1), then k12 for 1/(z-1)^2)")
print("  long division   :", np.round(long_division(num, den, N)).astype(int))
print("  2^k - k - 1     :", (2**k - k - 1))

# ---------------------------------------------------------------- no factor z
print("\nE(z) = 1/(z - 0.5): long division", np.round(long_division([1], [1, -0.5], 6), 4),
      " = -2 delta(k) + 2 (0.5)^k:", np.round(-2 * (k[:6] == 0) + 2 * 0.5 ** k[:6], 4))

# ---------------------------------------------------------------- Parseval
energy_time = np.sum(0.25 ** np.arange(200))
energy_freq = quad(lambda th: 1 / (1.25 - np.cos(th)), 0, 2 * np.pi)[0] / (2 * np.pi)
print(f"\nParseval for 0.5^k: time {energy_time:.6f}, frequency {energy_freq:.6f}, exact 4/3 = {4 / 3:.6f}")

# ---------------------------------------------------------------- exit ticket
num, den = [1, 0, 0], np.polymul([1, -1], [1, -0.2])
print("\nExit ticket, E(z) = z^2/((z-1)(z-0.2))")
print("  long division      :", np.round(long_division(num, den, 6), 4))
print("  5/4 - (1/4) 0.2^k  :", np.round(1.25 - 0.25 * 0.2 ** k[:6], 4))

# check yourself 1: (3z^2 + z)/((z-1)(z+0.5))
num, den = [3, 1, 0], np.polymul([1, -1], [1, 0.5])
print("Check yourself, (3z^2 + z)/((z-1)(z+0.5)):", np.round(long_division(num, den, 5), 4),
      " = 8/3 + (1/3)(-0.5)^k:", np.round(8 / 3 + (-0.5) ** k[:5] / 3, 4))

# ---------------------------------------------------------------- figure
th = np.linspace(0, 2 * np.pi, 400)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].stem(k, f_time)
axes[0].axhline(2, ls="--", color="gray", lw=0.8)
axes[0].set(xlabel="k", ylabel="f(k)", title="Step * 0.5^k = 2 - 0.5^k")
axes[1].plot(th, 1 / (1.25 - np.cos(th)))
axes[1].axhline(4 / 3, ls="--", color="gray", lw=0.8, label="mean = 4/3")
axes[1].set(xlabel="theta (rad)", ylabel="|F(e^{j theta})|^2", title="Parseval: same energy")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ch04_convolution_parseval.png", dpi=150)
print("\nSaved figures/ch04_convolution_parseval.png")
