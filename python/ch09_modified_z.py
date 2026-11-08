"""Chapter 9 companion: the modified z-transform and closed-loop sampled systems.

Convention (as in the chapter): F(z,m) = sum_k f((k+m)T) z^-(k+1), 0 <= m < 1,
the transform of f delayed by (1-m)T.

Reproduces:
  * Example A1: f = e^-t, F(z,m) = e^{-mT}/(z - e^{-T}), and the two ends m = 0, m -> 1;
  * Example A2: ZOH + 1/(s+1), unit step: c((k+m)T) = 1 - e^{-(k+m)T};
  * Example A3: a delay of 0.4T (m = 0.6), T = 1 s:
    G(z,m) = (0.4512z + 0.1809)/(z(z - 0.3679)), step samples 0, 0.4512, 0.7981, 0.9257;
  * two samplers that are not synchronized: G1(z,0.2), G2(z,0.8), c(T) = 0.000398;
  * Example B1: ZOH + 1/(s+1) in a unity loop, T = ln 2: C/R = 0.5/z, checked by
    simulating the sampled loop with the plant integrated exactly;
  * Example B3 (two samplers): C/R = G1G2/(1 + G1G2 + HG2(z)), checked the same way
    with G1 = ZOH + 1/(s+1), G2 = ZOH + 1/(s+2), H = 1/(s+3).

Run:  python3 ch09_modified_z.py      (figures go to ./figures)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import expm

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)


def zoh(num, den, T):
    """ZOH equivalent (numerator, denominator in powers of z) from one matrix exponential."""
    A, B, C, D = signal.tf2ss(num, den)
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * T)
    numz, denz = signal.ss2tf(E[:n, :n], E[:n, n:], C, D)
    return numz[0], denz


def series(num, den, n):
    """First n coefficients of num(z)/den(z) in powers of z^-1 (same length vectors)."""
    num = np.r_[np.zeros(len(den) - len(num)), num]
    return signal.lfilter(num, den, np.r_[1.0, np.zeros(n - 1)])


# ---------------------------------------------------------------- Example A1
T = 0.7
z = 1.5 * np.exp(0.6j)
k = np.arange(400)
print("Example A1: f(t) = e^-t, T = 0.7 s, at z =", np.round(z, 4))
for m in (0.0, 0.3, 0.8):
    s = np.sum(np.exp(-(k + m) * T) * z ** (-(k + 1.0)))
    print(f"  m = {m}: series {s:.6f},  e^(-mT)/(z - e^-T) = {np.exp(-m * T) / (z - np.exp(-T)):.6f}")
F = z / (z - np.exp(-T))
print(f"  m = 0 gives F(z)/z = {F / z:.6f};  m -> 1 gives F(z) - f(0) = {F - 1:.6f}"
      f" = e^-T/(z - e^-T) = {np.exp(-T) / (z - np.exp(-T)):.6f}")

# ---------------------------------------------------------------- Example A2
T = 1.0
p = np.exp(-T)
print("\nExample A2: ZOH + 1/(s+1), unit step, T = 1 s: C(z,m) = 1/(z-1) - e^(-mT)/(z - e^-T)")
for m in (0.25, 0.5):
    num = np.r_[0, (1 - np.exp(-m * T)), np.exp(-m * T) - p]
    c = series(num, np.polymul([1, -1], [1, -p]), 6)[1:]                # coefficients of z^-(k+1)
    print(f"  m = {m}: c((k+m)T) = {np.round(c, 4)};  1 - e^-(k+m)T = "
          f"{np.round(1 - np.exp(-(np.arange(5) + m) * T), 4)}")

# ---------------------------------------------------------------- Example A3
m = 0.6
num = [1 - np.exp(-m * T), np.exp(-m * T) - p]
den = np.polymul([1, 0], [1, -p])
print(f"\nExample A3: delay 0.4T, m = {m}: G(z,m) = ({num[0]:.4f} z + {num[1]:.4f})/(z (z - {p:.4f}))")
print(f"  DC gain G(1,m) = {np.polyval(num, 1) / np.polyval(den, 1):.4f};  zero at z = {-num[1] / num[0]:.3f}")
cs = signal.lfilter(np.r_[0, num], den, np.ones(6))
print("  step samples:", np.round(cs[:4], 4), "  1 - e^-(k - 0.4):",
      np.round(np.r_[0, 1 - np.exp(-(np.arange(1, 4) - 0.4))], 4))

# ---------------------------------------------------------------- nonsynchronous samplers
T, h = 0.05, 0.2
p = np.exp(-T)
g1 = [1 - np.exp(-h * T), np.exp(-h * T) - p]                # G1(z, h) numerator over z(z - p)
mm = 1 - h
g2 = [mm * T, (1 - mm) * T]                                  # (z-1)/z * T(mz + 1 - m)/(z-1)^2 -> over z(z-1)
print(f"\nTwo samplers, T = {T} s, offset hT = {h * T:.2f} s")
print(f"  G1(z,0.2) = ({g1[0]:.5f} z + {g1[1]:.5f})/(z (z - {p:.5f}))")
print(f"  G2(z,0.8) = ({g2[0]:.2f} z + {g2[1]:.2f})/(z (z - 1))")
print(f"  c(T) = {g2[0] * g1[0]:.6f}  (a(0.01) = {1 - np.exp(-0.01):.5f} held for 0.04 s:"
      f" {0.04 * (1 - np.exp(-0.01)):.6f})")


# ---------------------------------------------------------------- closed loops, simulated
def step_matrices(A, B, T):
    n = A.shape[0]
    M = np.zeros((n + B.shape[1], n + B.shape[1]))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * T)
    return E[:n, :n], E[:n, n:]


# Example B1: e(k) = r - c(k), plant 1/(s+1) driven by the held error
T = np.log(2)
Phi, Gam = step_matrices(np.array([[-1.0]]), np.array([[1.0]]), T)
x = np.zeros(1)
cB1 = []
for kk in range(8):
    cB1.append(x[0])
    x = Phi @ x + Gam @ np.array([1.0 - x[0]])
print("\nExample B1, T = ln 2: simulated c(kT) =", np.round(cB1, 4), "  (C/R = 0.5/z: 0, 0.5, 0.5, ...)")

# Example B3: E1 = R - C, E2 = G1 E1* - H C, C = G2 E2*  (H acts on the continuous C)
T = 0.3
A = np.array([[-1.0, 0, 0], [0, -2.0, 0], [0, 1.0, -3.0]])   # x1: G1 plant, xc: G2 plant, xh: H
B = np.array([[1.0, 0], [0, 1.0], [0, 0]])
Phi, Gam = step_matrices(A, B, T)
x = np.zeros(3)
c_sim = []
for kk in range(25):
    c_sim.append(x[1])
    e1 = 1.0 - x[1]
    e2 = x[0] - x[2]
    x = Phi @ x + Gam @ np.array([e1, e2])
n1, d1 = zoh([1], [1, 1], T)
n2, d2 = zoh([1], [1, 2], T)
nh, dh = zoh([1], np.polymul([1, 2], [1, 3]), T)                 # HG2(z) = Z[ZOH H(s) /(s+2)]
# C/R = G1G2 / (1 + G1G2 + HG2) with every term over a common denominator
N12, D12 = np.polymul(n1, n2), np.polymul(d1, d2)
num = np.polymul(N12, dh)
den = np.polyadd(np.polyadd(np.polymul(D12, dh), np.polymul(N12, dh)), np.polymul(nh, D12))
num = np.r_[np.zeros(len(den) - len(num)), num]                  # align powers of z
c_z = signal.lfilter(num, den, np.ones(25))
print("Example B3 (T = 0.3 s), unit step:")
print("  simulated loop      :", np.round(c_sim[:8], 5))
print("  G1G2/(1+G1G2+HG2(z)):", np.round(c_z[:8], 5))
print("  largest difference over 25 samples:", f"{np.max(np.abs(np.array(c_sim) - c_z)):.1e}")

# ---------------------------------------------------------------- figure
# left: Example A2, sweeping m traces the output between the samples
T, p = 1.0, np.exp(-1.0)
tt, cc = [], []
for m in np.linspace(0, 0.99, 34):
    num = np.r_[0, 1 - np.exp(-m * T), np.exp(-m * T) - p]
    c = series(num, np.polymul([1, -1], [1, -p]), 8)        # c[k+1] = c((k+m)T)
    for kk in range(6):
        tt.append((kk + m) * T)
        cc.append(c[kk + 1])
tc = np.linspace(0, 6, 600)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(tc, 1 - np.exp(-tc), "k", lw=1, label="1 - e^-t")
axes[0].plot(tt, cc, ".", ms=3, color="C0", label="C(z,m), m swept")
axes[0].plot(np.arange(7), 1 - np.exp(-np.arange(7.0)), "o", color="C3", label="samples")
axes[0].set(xlabel="t (s)", title="Example A2: between the samples")
axes[0].legend(fontsize=8)
# right: Example B3, simulated loop against the sampled-graph formula
axes[1].plot(0.3 * np.arange(25), c_sim, "o", ms=4, label="simulated loop")
axes[1].plot(0.3 * np.arange(25), c_z, "x", ms=5, label="G1G2/(1 + G1G2 + HG2)")
axes[1].set(xlabel="t (s)", title="Example B3 with two samplers, T = 0.3 s")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ch09_modified_z.png", dpi=150)
print("\nSaved figures/ch09_modified_z.png")
