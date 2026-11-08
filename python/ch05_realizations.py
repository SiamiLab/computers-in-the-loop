"""Chapter 5 companion: realizations, state variables, and the sample-and-hold.

Reproduces:
  * G(z) = (z + 0.5)/(z^2 - 0.75z + 0.125) run three ways from rest with a unit
    step: the direct form (four delays), the two-delay form, and the state
    equations x(k+1) = A x(k) + B e(k), m(k) = C x(k); all give
    m_k = 4 (1 - 0.5^k)^2;
  * G(z) = C (zI - A)^-1 B evaluated at test points;
  * the sampled-signal table on E(s) = 1/((s+1)(s+2)) with T = ln 2:
    E(z) = 0.25 z/((z - 0.5)(z - 0.25));
  * a ramp through the sample-and-hold, and the hold-error bound T max|de/dt|.

Run:  python3 ch05_realizations.py      (figures go to ./figures)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

b0, b1 = 1.0, 0.5                 # numerator  beta0 z + beta1
a1, a2 = -0.75, 0.125             # denominator z^2 + alpha1 z + alpha2
N = 13
e = np.ones(N)

# direct form: m_k = b0 e_{k-1} + b1 e_{k-2} - a1 m_{k-1} - a2 m_{k-2}
m_dir = np.zeros(N)
for k in range(N):
    em1 = e[k - 1] if k >= 1 else 0.0
    em2 = e[k - 2] if k >= 2 else 0.0
    mm1 = m_dir[k - 1] if k >= 1 else 0.0
    mm2 = m_dir[k - 2] if k >= 2 else 0.0
    m_dir[k] = b0 * em1 + b1 * em2 - a1 * mm1 - a2 * mm2

# two-delay form: x2(k+1) = e_k - a1 x2 - a2 x1, x1(k+1) = x2, m = x2 + 0.5 x1
x1 = x2 = 0.0
m_two, X1, X2 = [], [], []
for k in range(N):
    m_two.append(b0 * x2 + b1 * x1)
    X1.append(x1)
    X2.append(x2)
    x1, x2 = x2, e[k] - a1 * x2 - a2 * x1

# state equations (the same two delays as matrices)
A = np.array([[0.0, 1.0], [-a2, -a1]])
B = np.array([[0.0], [1.0]])
C = np.array([[b1, b0]])
x = np.zeros((2, 1))
m_ss = []
for k in range(N):
    m_ss.append((C @ x).item())
    x = A @ x + B * e[k]

closed = 4 * (1 - 0.5 ** np.arange(N)) ** 2
print("G(z) = (z + 0.5)/(z^2 - 0.75z + 0.125), unit step from rest")
print("  x1(k)            :", np.round(X1[:6], 4))
print("  x2(k)            :", np.round(X2[:6], 4))
print("  direct form      :", np.round(m_dir[:6], 4))
print("  two-delay form   :", np.round(m_two[:6], 4))
print("  state equations  :", np.round(m_ss[:6], 4))
print("  4 (1 - 0.5^k)^2  :", np.round(closed[:6], 4))
print("  final value G(1) =", (1 + 0.5) / (1 - 0.75 + 0.125))
r, p, _ = signal.residue([1, 0.5], np.polymul([1, a1, a2], [1, -1]))
print("  residues of M(z)/z at", np.round(p, 4), ":", np.round(r, 4))
for z in (2.0, 1.1 + 0.9j):
    G_ss = (C @ np.linalg.solve(z * np.eye(2) - A, B)).item()
    print(f"  z = {z}: C (zI - A)^-1 B = {complex(G_ss):.6f},  G(z) = {complex((z + 0.5) / (z * z - 0.75 * z + 0.125)):.6f}")
print("  det(zI - A) coefficients:", np.round(np.poly(A), 4))

# ---------------------------------------------------------------- sampled-signal table
T = np.log(2)
k = np.arange(8)
samples = np.exp(-k * T) - np.exp(-2 * k * T)
from_table = signal.lfilter([0, 0.25, 0], np.polymul([1, -0.5], [1, -0.25]), np.r_[1.0, np.zeros(7)])
print("\nE(s) = 1/((s+1)(s+2)), T = ln 2")
print("  samples e(kT)              :", np.round(samples, 5))
print("  series of 0.25z/((z-0.5)(z-0.25)):", np.round(from_table, 5))

# ---------------------------------------------------------------- ramp through the hold
T = 0.5
t = np.linspace(0, 3, 3001)
held = T * np.floor(t / T + 1e-12)
gap = t - held
print(f"\nRamp through the ZOH, T = {T}: gap e(t) - ebar(t) between {gap.min():.3f} and {gap.max():.3f},"
      f" mean {gap.mean():.3f} (about T/2 = {T / 2})")
print(f"Hold error for sin(2 pi t) below 0.05 needs T <= 0.05/(2 pi) = {0.05 / (2 * np.pi):.5f} s"
      f" ({1 / (0.05 / (2 * np.pi)):.0f} samples per cycle)")
Tq = 0.05 / (2 * np.pi)
tt = np.linspace(0, 1, 200001)
err = np.abs(np.sin(2 * np.pi * tt) - np.sin(2 * np.pi * Tq * np.floor(tt / Tq)))
print(f"  largest error with that T: {err.max():.4f}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].stem(np.arange(N), m_dir)
axes[0].axhline(4, ls="--", color="gray", lw=0.8)
axes[0].set(xlabel="k", ylabel="m_k", title="Step response of G(z): every realization")
tt = np.linspace(0, 4, 800)
Ts = 0.5
axes[1].plot(tt, 1 - np.exp(-tt), label="e(t) = 1 - e^-t")
axes[1].step(tt, 1 - np.exp(-Ts * np.floor(tt / Ts)), where="post", label="held, T = 0.5 s")
axes[1].plot(Ts * np.arange(9), 1 - np.exp(-Ts * np.arange(9)), "o", ms=4, label="same samples")
axes[1].set(xlabel="t (s)", title="Same samples, different signals")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ch05_realizations.png", dpi=150)
print("\nSaved figures/ch05_realizations.png")
