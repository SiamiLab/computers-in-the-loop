"""Chapter 6 companion: the starred transform, aliasing, and the zero-order hold.

Reproduces:
  * E*(s) = sum e(kT) e^{-kTs} for the unit step and for 1 - e^{-t}, summed
    directly and from the residue formula (Examples 1 and 2);
  * Example 3, the delayed signal [1 - e^{-(t-1)}] u(t-1) with T = 0.5 s: the
    residue formula used on E(s) as it stands gives the wrong samples at k = 0, 1;
    taking the whole-period delay out first gives the right ones;
  * aliasing: cos(2 pi 7 t) and cos(2 pi 3 t) have the same samples at 10 Hz;
    a hidden oscillation sin(pi t/T) vanishes at every sample;
  * the frequency response of the ZOH, T = 0.1 s: 0.984 T and -18 deg at 1 Hz,
    0.637 T and -90 deg at 5 Hz.

Run:  python3 ch06_starred_aliasing.py      (figures go to ./figures)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
K = np.arange(3000)


def starred(samples, s, T):
    """E*(s) = sum_k e(kT) e^{-kTs}."""
    return np.sum(samples * np.exp(-K[:len(samples)] * T * s))


T = 0.5
print("Starred transforms, T = 0.5 s, at two values of s")
for s in (0.8 + 2.0j, 2.0 + 0.0j):
    step = starred(np.ones(3000), s, T)
    print(f"  s = {s}: unit step  series {step:.6f}   1/(1 - e^-Ts) = {1 / (1 - np.exp(-T * s)):.6f}")
    f = starred(1 - np.exp(-K * T), s, T)
    res = 1 / (1 - np.exp(-T * s)) - 1 / (1 - np.exp(-T * (s + 1)))     # residues at 0 and -1
    print(f"            1 - e^-t   series {f:.6f}   residue formula {res:.6f}")

# ---------------------------------------------------------------- Example 3: a delay of 2T
kT = K * T
e_true = np.where(kT >= 1, 1 - np.exp(-(kT - 1)), 0.0)
e_wrong = 1 - np.exp(-(kT - 1))                     # what the residues of e^{-s}/(s(s+1)) describe
print("\nExample 3, e(t) = [1 - e^{-(t-1)}] u(t-1), T = 0.5 s")
print("  true samples          :", np.round(e_true[:6], 4))
print("  residues on E(s) as is:", np.round(e_wrong[:6], 4), "  (wrong at k = 0, 1)")
s = 1.3 + 0.7j
closed = (1 - np.exp(-0.5)) * np.exp(-1.5 * s) / ((1 - np.exp(-0.5 * s)) * (1 - np.exp(-0.5 * (s + 1))))
fix = np.exp(-2 * T * s) * (1 / (1 - np.exp(-T * s)) - 1 / (1 - np.exp(-T * (s + 1))))
print(f"  at s = {s}: series {starred(e_true, s, T):.6f}, closed form {closed:.6f},"
      f" delay taken out first {fix:.6f}")
print(f"  first nonzero sample e(1.5) = 1 - e^-0.5 = {1 - np.exp(-0.5):.4f}")

# ---------------------------------------------------------------- aliasing and hidden oscillations
T = 0.1
k = np.arange(11)
s7 = np.cos(2 * np.pi * 7 * k * T)
s3 = np.cos(2 * np.pi * 3 * k * T)
print("\nAliasing at fs = 10 Hz: max |cos(2pi 7 kT) - cos(2pi 3 kT)| =", f"{np.max(np.abs(s7 - s3)):.2e}")
print("  folding rule: 7 Hz shows up at |7 - 10| =", abs(7 - 10), "Hz")
print("  cos(2 pi t) at fs = 1 Hz gives the samples", np.round(np.cos(2 * np.pi * np.arange(5)), 6),
      "like the constant 1")
print("  hidden oscillation sin(pi t/T): samples", np.round(np.sin(np.pi * np.arange(6)), 12) + 0.0)

# ---------------------------------------------------------------- ZOH frequency response
T = 0.1
ws = 2 * np.pi / T


def Gho(w):
    return (1 - np.exp(-1j * w * T)) / (1j * w)


print(f"\nZOH, T = {T} s (ws = {ws:.1f} rad/s)")
for f_hz in (1.0, 5.0):
    G = Gho(2 * np.pi * f_hz)
    print(f"  {f_hz:3.0f} Hz: |G|/T = {abs(G) / T:.3f}, phase = {np.degrees(np.angle(G)):.1f} deg")
print(f"  gain at ws/2 = 2T/pi = {2 / np.pi:.3f} T")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
tt = np.linspace(0, 0.5, 1000)
axes[0].plot(tt, np.cos(2 * np.pi * 7 * tt), label="7 Hz")
axes[0].plot(tt, np.cos(2 * np.pi * 3 * tt), "--", label="3 Hz")
axes[0].plot(k[:6] * T, s3[:6], "ko", label="samples, fs = 10 Hz")
axes[0].set(xlabel="t (s)", title="Aliasing: same samples")
axes[0].legend(fontsize=8, loc="lower left")
w = np.linspace(1e-6, 3 * ws, 2000)
axes[1].plot(w / ws, np.abs(Gho(w)) / T)
axes[1].set(xlabel="omega / omega_s", ylabel="|G_ho(j omega)| / T", title="ZOH magnitude")
fig.tight_layout()
fig.savefig(FIG / "ch06_aliasing_zoh.png", dpi=150)
print("\nSaved figures/ch06_aliasing_zoh.png")
