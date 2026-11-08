"""EECE 5610, Chapter 12: root locus and frequency response of sampled-data loops.

The running example of the chapter: a zero-order hold in front of Gp(s) = 1/(s(s+1)),
sampled with T = 1 s, a gain K and unity feedback.

This script reproduces the main numbers and figures of the chapter:
  1. G(z) by zero-order-hold discretization,
  2. the root locus in the z-plane: breakaway points, the circle, the critical gain,
  3. the gain that gives zeta = 0.5,
  4. the critical gain for T = 0.1, 0.5, 1 s,
  5. the w-plane transfer function and its Bode plot, with gain and phase margins,
  6. the Nyquist plot and the encirclement count Z = N + P,
  7. the root locus with one sample of computation delay, D(z) = K/z.

Run:   python3 ch12_root_locus_frequency.py
Plots are written as PNG files to the folder  ch12_figures/  next to this script.
Needs numpy, scipy and matplotlib.
"""
import os

import matplotlib

matplotlib.use("Agg")                      # write files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import brentq

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch12_figures")
os.makedirs(OUT, exist_ok=True)


# -----------------------------------------------------------------------------------------
# 1. The pulse transfer function G(z) = (1 - z^-1) Z[Gp(s)/s]
# -----------------------------------------------------------------------------------------
def zoh_model(T):
    """Return (num, den) of G(z) for a ZOH in front of 1/(s(s+1)), coefficients in z."""
    num, den, _ = signal.cont2discrete(([1.0], [1.0, 1.0, 0.0]), T, method="zoh")
    num = np.trim_zeros(np.ravel(num), "f")
    return num / den[0], den / den[0]


T = 1.0
num, den = zoh_model(T)
a, b = num                                 # G(z) = (a z + b) / (z^2 - (1+a) z + a)
print("1. G(z) for T = 1 s")
print(f"   numerator   {a:.4f} z + {b:.4f}      (e^-1 = {np.exp(-1):.4f}, 1 - 2e^-1 = {1 - 2*np.exp(-1):.4f})")
print(f"   denominator z^2 {den[1]:+.4f} z {den[2]:+.4f}  = (z - 1)(z - {np.exp(-1):.4f})")


def G(z):
    return np.polyval(num, z) / np.polyval(den, z)


def closed_loop_poles(K, num=num, den=den):
    """Roots of the characteristic polynomial den(z) + K num(z)."""
    return np.roots(np.polyadd(den, K * num))


# -----------------------------------------------------------------------------------------
# 2. Root locus: zero, circle, breakaway, critical gain
# -----------------------------------------------------------------------------------------
z0 = -b / a                                # the zero of G(z), equal to 2 - e
R = np.sqrt((1 - z0) * (a - z0))           # radius of the circle centered at the zero
# breakaway/break-in: A'B - AB' = 0 with A = den, B = num
cond = np.polysub(np.polymul(np.polyder(den), num), np.polymul(den, np.polyder(num)))
zb, zi = sorted(np.roots(cond).real, reverse=True)
K_of = lambda s: -np.polyval(den, s) / np.polyval(num, s)
Kcr = brentq(lambda K: max(abs(closed_loop_poles(K))) - 1, 1.0, 5.0)
theta = abs(np.angle(closed_loop_poles(Kcr)[0]))
print("\n2. Root locus")
print(f"   zero z0 = {z0:.4f} (2 - e = {2 - np.e:.4f}); circle radius R = {R:.4f}")
print(f"   breakaway {zb:.4f} at K = {K_of(zb):.3f};  break-in {zi:.4f} at K = {K_of(zi):.2f}")
print(f"   critical gain {Kcr:.3f}  ((e-1)/(e-2) = {(np.e - 1)/(np.e - 2):.3f}), "
      f"crossing angle {theta:.3f} rad, {2*np.pi/theta:.2f} samples per cycle")


def zeta_wn(z, T=1.0):
    """Equivalent damping ratio and natural frequency of a z-plane pole (Chapter 10)."""
    r, th = abs(z), abs(np.angle(z))
    L = np.hypot(np.log(r), th)
    return -np.log(r) / L, L / T


# -----------------------------------------------------------------------------------------
# 3. The gain for zeta = 0.5
# -----------------------------------------------------------------------------------------
K05 = brentq(lambda K: zeta_wn(closed_loop_poles(K)[0])[0] - 0.5, 0.2, 1.0)
p05 = closed_loop_poles(K05)[0]
print("\n3. Target damping")
print(f"   zeta = 0.5 at K = {K05:.3f}: poles {p05.real:.3f} +- j{abs(p05.imag):.3f}, "
      f"wn = {zeta_wn(p05)[1]:.3f} rad/s, ramp error 1/K = {1/K05:.2f}")

# Figure: the root locus with the unit circle and the zeta = 0.5 spiral
fig, ax = plt.subplots(figsize=(7, 5.5))
Ks = np.concatenate([np.linspace(0, 0.2, 400), np.linspace(0.2, 40, 4000)])
locus = np.array([np.sort_complex(closed_loop_poles(K)) for K in Ks])
ax.plot(locus.real, locus.imag, ".", ms=1.5, color="crimson")
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(t), np.sin(t), color="gray", lw=0.8)
ph = np.linspace(0, np.pi, 300)
c = 0.5 / np.sqrt(1 - 0.25)
for s in (1, -1):
    ax.plot(np.exp(-c * ph) * np.cos(ph), s * np.exp(-c * ph) * np.sin(ph), "--", color="tab:blue", lw=0.8)
ax.plot([1, a], [0, 0], "kx", ms=8, mew=2)
ax.plot([z0], [0], "ko", mfc="white", ms=7)
pc = closed_loop_poles(Kcr)
ax.plot(pc.real, pc.imag, "o", color="crimson")
ax.plot([p05.real] * 2, [p05.imag, -p05.imag], "o", color="green")
ax.set_xlim(-2.6, 1.4)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect("equal")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_title("Root locus, ZOH + 1/(s(s+1)), T = 1 s")
ax.set_xlabel("Re z")
ax.set_ylabel("Im z")
fig.savefig(os.path.join(OUT, "ch12_root_locus.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------------------
# 4. Effect of the sampling period on the critical gain
# -----------------------------------------------------------------------------------------
print("\n4. Critical gain against T")
print("   T      zero     Kcr     2/T+1/3   theta    theta/T")
for Ti in (0.1, 0.5, 1.0):
    n_i, d_i = zoh_model(Ti)
    Kc = brentq(lambda K: max(abs(closed_loop_poles(K, n_i, d_i))) - 1, 0.5, 100.0)
    th_i = abs(np.angle(closed_loop_poles(Kc, n_i, d_i)[0]))
    print(f"   {Ti:<5}  {-n_i[1]/n_i[0]:.3f}  {Kc:7.2f}  {2/Ti + 1/3:7.2f}   {th_i:.3f}    {th_i/Ti:.2f}")

# -----------------------------------------------------------------------------------------
# 5. The w-plane: Gw(w) = G(z) with z = (1 + wT/2)/(1 - wT/2); Bode plot and margins
# -----------------------------------------------------------------------------------------
def Gw(w, T=1.0):
    return G((1 + w * T / 2) / (1 - w * T / 2))


ww = np.logspace(-1.3, 2, 600)
resp = Gw(1j * ww)
mag_db = 20 * np.log10(abs(resp))
phase = np.degrees(np.unwrap(np.angle(resp)))
wwc = brentq(lambda x: abs(Gw(1j * x)) - 1, 0.1, 1.5)          # gain crossover
wwp = brentq(lambda x: Gw(1j * x).imag, 1.0, 3.0)              # phase crossover
PM = 180 + np.degrees(np.angle(Gw(1j * wwc)))
GM = 1 / abs(Gw(1j * wwp))
to_omega = lambda x: (2 / T) * np.arctan(x * T / 2)            # undo the warping
print("\n5. Margins at K = 1 (w-plane)")
print(f"   pole at w = -{2*(1-a)/(1+a):.4f} (2 tanh(1/2) = {2*np.tanh(0.5):.4f}), "
      f"zeros at w = {2/T:.0f} and w = -{2*(a+b)/(a-b):.2f}")
print(f"   PM = {PM:.1f} deg at ww = {wwc:.3f} (omega = {to_omega(wwc):.3f} rad/s)")
print(f"   GM = {GM:.3f} = {20*np.log10(GM):.2f} dB at ww = {wwp:.3f} (omega = {to_omega(wwp):.3f} rad/s)")
print(f"   GM equals the critical gain of the root locus: {np.isclose(GM, Kcr)}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
ax1.semilogx(ww, mag_db, color="crimson", label="sampled, Gw(j ww)")
ax1.semilogx(ww, -20 * np.log10(ww * np.sqrt(1 + ww**2)), "--", color="gray", label="analog Gp(j w)")
ax1.axhline(0, color="k", lw=0.6)
ax1.axvline(wwc, color="k", ls=":", lw=0.8)
ax1.axvline(wwp, color="k", ls=":", lw=0.8)
ax1.set_ylabel("magnitude (dB)")
ax1.set_ylim(-45, 30)
ax1.legend()
ax2.semilogx(ww, phase, color="crimson")
ax2.semilogx(ww, -90 - np.degrees(np.arctan(ww)), "--", color="gray")
ax2.axhline(-180, color="k", lw=0.6)
ax2.axvline(wwc, color="k", ls=":", lw=0.8)
ax2.axvline(wwp, color="k", ls=":", lw=0.8)
ax2.set_ylabel("phase (deg)")
ax2.set_xlabel("ww (rad/s)")
ax1.set_title(f"w-plane Bode plot, K = 1: PM = {PM:.1f} deg, GM = {20*np.log10(GM):.2f} dB")
fig.savefig(os.path.join(OUT, "ch12_bode_w.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------------------
# 6. Nyquist plot and Z = N + P
# -----------------------------------------------------------------------------------------
def contour(eps=1e-3, m=200000):
    """Unit circle counterclockwise, with a small outward half-circle around z = 1."""
    t0 = 2 * np.arcsin(eps / 2)
    th = np.linspace(t0, 2 * np.pi - t0, m)
    phi = np.linspace(np.angle(np.exp(-1j * t0) - 1), np.angle(np.exp(1j * t0) - 1), 20000)
    return np.concatenate([np.exp(1j * th), 1 + eps * np.exp(1j * phi)])


def clockwise_encirclements(curve, q):
    d = np.diff(np.angle(np.append(curve, curve[0]) - q))
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return -int(round(d.sum() / (2 * np.pi)))


Lgam = G(contour())
print("\n6. Nyquist: P = 0 (the pole at z = 1 is indented and counts as inside)")
print(f"   crossings of the negative real axis: {-1/Kcr:.3f} and G(-1) = {G(-1.0):.4f}")
for K in (1.0, 10.0, 30.0):
    N = clockwise_encirclements(Lgam, -1 / K)
    Z = int(sum(abs(closed_loop_poles(K)) > 1))
    print(f"   K = {K:5.1f}: N = {N}, Z = N + P = {N}, poles outside the circle: {Z}")

th = np.linspace(0.3, np.pi, 800)
Lh = G(np.exp(1j * th))
fig, ax = plt.subplots(figsize=(5.5, 6.5))
ax.plot(Lh.real, Lh.imag, color="crimson", label="0 < w < ws/2")
ax.plot(Lh.real, -Lh.imag, ":", color="crimson", label="mirror image")
ax.plot(np.cos(t), np.sin(t), color="gray", lw=0.8)
ax.plot([-1], [0], "k+", ms=10, mew=2)
ax.axvline(-1.5, color="gray", ls="--", lw=0.8)
ax.set_xlim(-2.0, 1.1)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect("equal")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.legend(loc="upper right")
ax.set_title("Nyquist plot of G(e^{jwT}), T = 1 s")
fig.savefig(os.path.join(OUT, "ch12_nyquist.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------------------
# 7. One sample of computation delay: D(z) = K/z
# -----------------------------------------------------------------------------------------
den_d = np.polymul(den, [1.0, 0.0])        # z (z - 1)(z - a)
Kd = brentq(lambda K: max(abs(closed_loop_poles(K, num, den_d))) - 1, 0.3, 2.0)
pd = closed_loop_poles(Kd, num, den_d)
print("\n7. With one sample of delay")
print(f"   asymptote center {(1 + a - z0)/2:.3f}; critical gain {Kd:.3f}; "
      f"crossing angle {max(np.angle(pd[abs(pd.imag) > 1e-9])):.3f} rad; third pole {min(pd.real):.3f}")

fig, ax = plt.subplots(figsize=(5.5, 5.5))
Ks = np.linspace(0, 8, 6000)
loc = np.array([np.sort_complex(closed_loop_poles(K, num, den_d)) for K in Ks])
ax.plot(loc.real, loc.imag, ".", ms=1.5, color="crimson")
ax.plot(np.cos(t), np.sin(t), color="gray", lw=0.8)
ax.plot([1, a, 0], [0, 0, 0], "kx", ms=8, mew=2)
ax.plot([z0], [0], "ko", mfc="white", ms=7)
ax.axvline((1 + a - z0) / 2, color="gray", ls="--", lw=0.8)
ax.set_xlim(-1.2, 1.5)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal")
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_title("Root locus with D(z) = K/z, T = 1 s")
fig.savefig(os.path.join(OUT, "ch12_root_locus_delay.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nFigures written to {OUT}")
