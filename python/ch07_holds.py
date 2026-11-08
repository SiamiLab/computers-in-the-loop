"""Chapter 7 companion: data reconstruction and data holds.

Reproduces:
  * the zero-order, first-order and fractional-order holds as transfer functions
      G_ho(s) = (1 - e^{-Ts})/s
      G_h1(s) = (1 + Ts)/T [(1 - e^{-Ts})/s]^2
      G_hb(s) = (1 - b e^{-Ts})(1 - e^{-Ts})/s + b/(T s^2) (1 - e^{-Ts})^2
    and their gain and phase at wT = 0.5 and at ws/2 (the comparison table);
  * where the ZOH and FOH lags are equal (wT = 2.33, 0.37 ws) and the peak gain
    of the FOH (about 1.59 T near 0.34 ws);
  * the images of a held sinusoid at ws/10: amplitudes 0.984, 0.109, 0.089;
  * the three holds reconstructing one sampled signal (figure).

Run:  python3 ch07_holds.py      (figures go to ./figures)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
T = 1.0                         # results are quoted in units of T
ws = 2 * np.pi / T


def G_zoh(w):
    s = 1j * w
    return (1 - np.exp(-T * s)) / s


def G_frac(w, beta):
    s = 1j * w
    x = np.exp(-T * s)
    return (1 - beta * x) * (1 - x) / s + beta / (T * s**2) * (1 - x) ** 2


def G_foh(w):
    s = 1j * w
    return (1 + T * s) / T * ((1 - np.exp(-T * s)) / s) ** 2


def gain_phase(G):
    return abs(G) / T, np.degrees(np.angle(G))


print("Gain (in units of T) and phase (deg)")
print("                     wT = 0.5              w = ws/2")
for name, f in [("ZOH          ", G_zoh), ("beta = 0.5   ", lambda w: G_frac(w, 0.5)),
                ("FOH (beta=1) ", G_foh)]:
    g1, p1 = gain_phase(f(0.5 / T))
    g2, p2 = gain_phase(f(np.pi / T))
    p2 = p2 - 360 if p2 > 0 else p2           # report the lag as a negative angle
    print(f"  {name}   {g1:.3f} T, {p1:6.1f}       {g2:.3f} T, {p2:7.1f}")
print("  the fractional hold with beta = 1 equals the FOH:",
      np.allclose(G_frac(np.array([0.3, 1.7]), 1.0), G_foh(np.array([0.3, 1.7]))))

# equal lags: the ZOH lags by wT/2, the FOH by wT - atan(wT)
x = brentq(lambda v: v / 2 - (v - np.arctan(v)), 0.5, 3.0)
print(f"\nEqual phase lag where wT/2 = wT - atan(wT): wT = {x:.2f}, w = {x / (2 * np.pi):.2f} ws")
w = np.linspace(0.01, ws, 200001)
g = np.abs(G_foh(w)) / T
print(f"Peak gain of the FOH: {g.max():.2f} T at w = {w[g.argmax()] / ws:.2f} ws")

# a held sinusoid at w0 = ws/10: components at |w0 + n ws| with amplitude |sinc(pi (w0 + n ws)/ws)|
w0 = ws / 10
print("\nZOH on sin(w0 t), w0 = ws/10: amplitude of the component at |w0 + n ws|")
for n in (0, -1, 1):
    v = np.pi * (w0 + n * ws) / ws
    print(f"  n = {n:2d}, |w0 + n ws| = {abs(w0 + n * ws) / ws:.1f} ws: {abs(np.sin(v) / v):.3f}")

# low-frequency delay of the fractional hold: (1 - beta) T/2
for beta in (0.0, 0.5, 1.0):
    wl = 1e-3
    print(f"  beta = {beta}: low-frequency delay = {-np.angle(G_frac(wl, beta)) / wl:.4f} T"
          f" (formula (1 - beta)/2 = {(1 - beta) / 2:.4f})")

# ---------------------------------------------------------------- reconstruction figure
Ts = 0.5
f = lambda t: 1.1 + 0.8 * np.sin(0.9 * t) + 0.12 * t
tk = Ts * np.arange(0, 21)
ek = f(tk)
t = np.linspace(0, 10, 2001)
k = np.minimum((t / Ts).astype(int), len(tk) - 1)
zoh = ek[k]
prev = ek[np.maximum(k - 1, 0)]
slope = (ek[k] - prev) / Ts
foh = ek[k] + slope * (t - tk[k])
half = ek[k] + 0.5 * slope * (t - tk[k])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(t, f(t), "k", lw=1, label="e(t)")
axes[0].plot(t, zoh, label="ZOH")
axes[0].plot(t, half, label="beta = 0.5")
axes[0].plot(t, foh, label="FOH")
axes[0].plot(tk, ek, "ko", ms=3)
axes[0].set(xlabel="t (s)", title="Reconstruction from samples, T = 0.5 s")
axes[0].legend(fontsize=8)
wn = np.linspace(0.001, 2 * ws, 2000)
axes[1].plot(wn / ws, np.abs(G_zoh(wn)) / T, label="ZOH")
axes[1].plot(wn / ws, np.abs(G_foh(wn)) / T, label="FOH")
axes[1].set(xlabel="omega / omega_s", ylabel="|G_h| / T", title="Hold magnitudes")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ch07_holds.png", dpi=150)
print("\nSaved figures/ch07_holds.png")
