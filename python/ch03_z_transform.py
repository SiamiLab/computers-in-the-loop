"""Chapter 3 companion: discrete-time systems and the z-transform.

Reproduces:
  * the satellite (unit inertia, torque through a sample-and-hold): the
    difference equation theta(k+2) - 2 theta(k+1) + theta(k) = T^2/2 [tau(k+1) + tau(k)],
    checked against the exact solution theta(t) = t^2/2 for a constant torque,
    and the pulse transfer function T^2 (z + 1) / (2 (z - 1)^2);
  * the definition E(z) = sum e_k z^-k: partial sums against the closed forms
    of a few pairs, inside and outside the region of convergence;
  * the final value theorem on the position loop with F(z) = (0.368z + 0.264)/((z-1)(z-0.368)):
    the error for K = 1 (stable, e -> 0) and K = 3 (unstable, the formula still says 0).

Run:  python3 ch03_z_transform.py      (figures go to ./figures)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- satellite
T = 1.0
n = 8
tau = np.ones(n + 2)                 # constant torque from k = 0
theta = np.zeros(n + 2)
omega = np.zeros(n + 2)
for k in range(n + 1):               # exact integration over each period (torque held)
    theta[k + 1] = theta[k] + T * omega[k] + T**2 / 2 * tau[k]
    omega[k + 1] = omega[k] + T * tau[k]
print("Satellite, T = 1 s, unit torque from rest")
print("  theta(k)          :", theta[:n])
print("  k^2/2             :", 0.5 * np.arange(n) ** 2)
lhs = theta[2:n] - 2 * theta[1:n - 1] + theta[:n - 2]
rhs = T**2 / 2 * (tau[1:n - 1] + tau[:n - 2])
print("  second-order difference equation holds:", np.allclose(lhs, rhs))
# the same samples from the pulse transfer function T^2 (z+1) / (2 (z-1)^2), driven by a step
num, den = [0, T**2 / 2, T**2 / 2], np.polymul([1, -1], [1, -1])
print("  Theta/Tau = (T^2/2)(z+1)/(z-1)^2, step response:", signal.lfilter(num, den, np.ones(n)))
print("  T = 0.5 s: T^2/2 =", 0.5**2 / 2)

# ---------------------------------------------------------------- definition and ROC
print("\nPartial sums of sum e_k z^-k (200 terms) against the closed form")
k = np.arange(200)
for name, ek, Ez, roc in [("a^k, a = 0.5", 0.5**k, lambda z: z / (z - 0.5), 0.5),
                          ("(-0.8)^k", (-0.8) ** k, lambda z: z / (z + 0.8), 0.8),
                          ("unit step", np.ones(200), lambda z: z / (z - 1), 1.0)]:
    for z in (1.5, 2.0 * np.exp(0.7j)):
        s = np.sum(ek * z ** (-k.astype(float)))
        print(f"  {name:12s} z = {complex(z):.3f}: series {complex(s):.6f}  closed form {complex(Ez(z)):.6f}"
              f"   (ROC |z| > {roc})")
z = 0.6
print(f"  0.5^k at z = 0.6 (inside the ROC): partial sums 50, 100, 200 terms:",
      [round(float(np.sum(0.5**k[:N] * z ** (-k[:N].astype(float)))), 4) for N in (50, 100, 200)],
      " closed form", round(z / (z - 0.5), 4))
z = 0.4
print(f"  0.5^k at z = 0.4 (outside the ROC): partial sums 10, 20, 40 terms:",
      [f"{np.sum(0.5**k[:N] * z ** (-k[:N].astype(float))):.3g}" for N in (10, 20, 40)], " (diverge)")

# ---------------------------------------------------------------- scaling and shifts
z = 1.7 + 0.4j
E = lambda z: z / (z - 0.5)
print("\nShift rules for e_k = 0.5^k at z =", z)
print("  Z[e(k-3)]  =", np.round(z**-3 * E(z), 6), " closed form 1/(z^2(z-0.5)) =", np.round(1 / (z**2 * (z - 0.5)), 6))
print("  Z[e(k+2)]  =", np.round(z**2 * (E(z) - 1 - 0.5 / z), 6), " = 0.25 z/(z-0.5) =", np.round(0.25 * z / (z - 0.5), 6))
print("  Z[(-0.5)^k] by scaling, E(-z) =", np.round(E(-z), 6), " = z/(z+0.5) =", np.round(z / (z + 0.5), 6))

# ---------------------------------------------------------------- position loop
e1 = np.exp(-1.0)                                     # exact: 0.368 = e^-1, 0.264 = 1 - 2e^-1
Fn = [e1, 1 - 2 * e1]
Fd = np.polymul([1, -1], [1, -e1])
print("\nPosition loop, F(z) = (0.368z + 0.264)/((z-1)(z-0.368)) (exact e^-1 used), unit step reference")
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for K, c in [(1, "C0"), (3, "C3")]:
    char = np.polyadd(Fd, K * np.array([0, *Fn]))       # (z-1)(z-0.368) + K(0.368z + 0.264)
    poles = np.roots(char)
    e = signal.lfilter(Fd, char, np.ones(16))            # E/R = 1/(1 + K F)
    print(f"  K = {K}: characteristic polynomial {np.round(char, 3)}, poles {np.round(poles, 3)},"
          f" |z| = {abs(poles[0]):.3f}")
    print(f"         e(k) = {np.round(e[:10], 3)}")
    ax.plot(np.arange(16), e, "o-", ms=3, color=c, label=f"K = {K}")
print("  final value formula: lim_{z->1} z/(1 + K F(z)) = 0 for every K (F has a pole at z = 1);")
print("  it is the limit of e(k) only when the loop is stable (0 < K < 2.39).")
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="k", ylabel="e(k)", title="Position loop: error samples")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ch03_position_error.png", dpi=150)
print("\nSaved figures/ch03_position_error.png")
