"""Chapter 10 companion: closed-loop time response and steady-state accuracy.

Loop: R -> (+/-) -> sampler -> D(z) -> ZOH -> G_p(s) -> C, unity feedback.

Reproduces:
  * Example 1, G_p = 4/(s+2), T = 0.1 s: G(z) = 0.3625/(z - 0.8187), closed-loop pole
    0.4562, tau = 0.127 s, c(kT) -> 2/3; the pole 3e^{-2T} - 2 turns negative for
    T > 0.203 s and leaves the unit circle for T > 0.549 s;
  * Example 2, G_p = 1/(s(s+1)), T = 1 s: poles 0.5 +- j0.618, zeta = 0.25,
    wn = 0.92 rad/s, tau = 4.36 s; peak 1.40 at the samples and 1.45 between them;
  * the effect of T on the same loop (table), and the limit T = 3.92 s;
  * Example B: an accumulator D = z/(z-1) removes the 50% step error;
  * the laser loop G_p = 2/((s+1)(s+4)) with D = k: stable for k < 54.7 (T = 0.1 s)
    and k < 6.78 (T = 1 s);
  * the exit ticket: Example 2 with K = 2.

Run:  python3 ch10_time_response.py      (figures go to ./figures)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import expm
from scipy.optimize import brentq

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)


def zoh(num, den, T):
    """ZOH equivalent, numerator padded to the length of the denominator."""
    numz, denz, _ = signal.cont2discrete((num, den), T, method="zoh")
    numz = np.atleast_2d(numz)[0]
    return np.r_[np.zeros(len(denz) - len(numz)), numz], denz


def closed_loop(numz, denz, K=1.0):
    """C/R = K G/(1 + K G) for G = numz/denz (same lengths)."""
    return K * numz, denz + K * numz


def specs(zp, T):
    """zeta, wn, tau from a z-plane pole r e^{j theta}."""
    r, th = abs(zp), abs(np.angle(zp))
    L = np.hypot(np.log(r), th)
    return -np.log(r) / L, L / T, -T / np.log(r)


def hybrid_step(num, den, T, D=1.0, n=20, sub=50):
    """Unit-step response of the sampled loop with D = constant gain, between the
    samples too: the plant is integrated exactly on a grid of T/sub."""
    A, B, C, Dd = signal.tf2ss(num, den)
    nx = A.shape[0]
    M = np.zeros((nx + 1, nx + 1))
    M[:nx, :nx], M[:nx, nx:] = A, B
    E = expm(M * T / sub)
    Ph, Ga = E[:nx, :nx], E[:nx, nx:]
    x = np.zeros((nx, 1))
    t, y = [], []
    for k in range(n):
        u = D * (1.0 - (C @ x).item())             # sample the error, hold the control
        for j in range(sub):
            t.append((k + j / sub) * T)
            y.append((C @ x).item())
            x = Ph @ x + Ga * u
    return np.array(t), np.array(y)


# ---------------------------------------------------------------- Example 1
T = 0.1
nz, dz = zoh([4], [1, 2], T)
nc, dc = closed_loop(nz, dz)
pole = -dc[1]
zeta, wn, tau = specs(pole, T)
c = signal.lfilter(nc, dc, np.ones(6))
print("Example 1: G_p = 4/(s+2), T = 0.1 s")
print(f"  G(z) = {nz[-1]:.4f}/(z - {-dz[1]:.4f});  C/R = {nc[-1]:.4f}/(z - {pole:.4f})")
print(f"  tau = -T/ln r = {tau:.3f} s (analog loop 4/(s+6): {1 / 6:.3f} s);  equivalent s-pole {np.log(pole) / T:.3f}")
print("  c(kT) =", np.round(c, 4), " -> 2/3")
print(f"  closed-loop pole 3e^(-2T) - 2 < 0 for T > ln(1.5)/2 = {np.log(1.5) / 2:.3f} s;"
      f" < -1 for T > ln(3)/2 = {np.log(3) / 2:.3f} s")

# ---------------------------------------------------------------- Example 2
T = 1.0
nz, dz = zoh([1], [1, 1, 0], T)
nc, dc = closed_loop(nz, dz)
zp = np.roots(dc)
zp = zp[np.argmax(zp.imag)]
zeta, wn, tau = specs(zp, T)
c = signal.lfilter(nc, dc, np.ones(21))
t, y = hybrid_step([1], [1, 1, 0], T, n=21)
print("\nExample 2: G_p = 1/(s(s+1)), T = 1 s")
print(f"  G(z) = ({nz[1]:.4f} z + {nz[2]:.4f})/(z^2 - {-dz[1]:.4f} z + {dz[2]:.4f})")
print(f"  characteristic polynomial z^2 - {-dc[1]:.4f} z + {dc[2]:.4f}: poles {zp.real:.3f} +- j{zp.imag:.3f}"
      f" = {abs(zp):.3f} at +-{np.angle(zp):.3f} rad")
print(f"  zeta = {zeta:.3f}, wn = {wn:.3f} rad/s, tau = {tau:.2f} s, Mp = {100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)):.1f}%,"
      f" {2 * np.pi / np.angle(zp):.1f} samples per cycle")
print("  c(k), k = 0..10:", np.round(c[:11], 3))
print(f"  peak at the samples {c.max():.3f}; true peak between samples {y.max():.3f} at t = {t[y.argmax()]:.2f} s")
k_in = np.nonzero(np.abs(c - 1) > 0.02)[0][-1] + 1
print(f"  samples within 2% from k = {k_in} on")
_, ya = signal.step(([1], [1, 1, 1]), T=np.linspace(0, 20, 20001))
print(f"  analog loop 1/(s^2+s+1): overshoot {100 * (ya.max() - 1):.0f}%")

# ---------------------------------------------------------------- effect of T
print("\nEffect of T on Example 2 (z^2 + (T-2) z + (1 - T e^-T) = 0)")
print("   T     poles                    |z|     zeta   peak of samples")
for T in (0.1, 0.5, 1.0, 2.0, 4.0):
    nz, dz = zoh([1], [1, 1, 0], T)
    nc, dc = closed_loop(nz, dz)
    rts = np.roots(dc)
    rmax = np.max(np.abs(rts))
    cs = signal.lfilter(nc, dc, np.ones(int(60 / T) + 1))
    if rmax < 1 and np.all(np.abs(rts.imag) > 1e-9):
        zp = rts[np.argmax(rts.imag)]
        zstr = f"{specs(zp, T)[0]:.2f}"
        pstr = f"{np.round(zp.real, 3) + 0.0:.3f} +- j{zp.imag:.3f}"
    else:
        zstr, pstr = " --", ", ".join(f"{r.real:.3f}" for r in rts)
    print(f"  {T:3.1f}   {pstr:22s}  {rmax:.3f}   {zstr:5s}  {cs.max() if rmax < 1 else float('inf'):.2f}")
Tlim = brentq(lambda T: T * (1 + np.exp(-T)) - 4, 1, 6)
print(f"  a real pole reaches z = -1 when T (1 + e^-T) = 4: T = {Tlim:.2f} s")

# ---------------------------------------------------------------- Example B: digital integrator
T = np.log(2)
G = (np.array([0, 0.5]), np.array([1, -0.5]))
print("\nExample B: ZOH + 1/(s+1), T = ln 2, G(z) = 0.5/(z - 0.5)")
print("  without D: c(k) =", np.round(signal.lfilter([0, 0.5], [1, 0], np.ones(6)), 4), " (C/R = 0.5/z)")
nc, dc = [0, 0.5, 0], [1, -1, 0.5]                 # D G/(1 + D G) with D = z/(z-1)
c = signal.lfilter(nc, dc, np.ones(12))
zp = np.roots(dc)[0]
print("  with D = z/(z-1): c(k) =", np.round(c[:9], 4))
print(f"  poles 0.5 +- j0.5: r = {abs(zp):.3f}, theta = pi/4, zeta = {specs(zp, T)[0]:.2f}")
ramp = np.arange(400) * T
e = ramp - signal.lfilter(nc, dc, ramp)
print(f"  ramp error with D: {e[-1]:.4f} = T = ln 2")

# ---------------------------------------------------------------- the laser loop with D = k
print("\nLaser loop, G_p = 2/((s+1)(s+4)), D = k")
for T in (0.1, 1.0):
    nz, dz = zoh([2], [1, 5, 4], T)
    f = lambda kk: np.max(np.abs(np.roots(dz + kk * nz))) - 1
    kmax = brentq(f, 0.5, 5000 if T < 0.5 else 50)
    print(f"  T = {T}: G(z) = ({nz[1]:.4f} z + {nz[2]:.4f})/(z^2 - {-dz[1]:.4f} z + {dz[2]:.4f}),"
          f" stable for -2 < k < {kmax:.2f};  K_p = k G(1) = k x {np.sum(nz) / np.sum(dz):.3f}")
print(f"  5% step error needs k >= {2 / 0.05 - 2:.0f}; with T = 1 s the best is 2/(2 + 6.78) = {2 / 8.78:.2f}")

# ---------------------------------------------------------------- exit ticket
T = 1.0
nz, dz = zoh([1], [1, 1, 0], T)
nc, dc = closed_loop(nz, dz, K=2.0)
zp = np.roots(dc)
zp = zp[np.argmax(zp.imag)]
c = signal.lfilter(nc, dc, np.ones(60))
print(f"\nExit ticket, K = 2: z^2 - {-dc[1]:.3f} z + {dc[2]:.3f}, poles {zp.real:.3f} +- j{zp.imag:.3f},"
      f" |z| = {abs(zp):.3f}, zeta = {specs(zp, T)[0]:.2f}, peak of the samples {c.max():.2f}, ramp error 1/K_v = 0.5")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
T = 1.0
t, y = hybrid_step([1], [1, 1, 0], T, n=21)
nz, dz = zoh([1], [1, 1, 0], T)
c = signal.lfilter(*closed_loop(nz, dz), np.ones(21))
ta = np.linspace(0, 20, 400)
_, ya = signal.step(([1], [1, 1, 1]), T=ta)
axes[0].plot(ta, ya, "k--", lw=1, label="analog loop")
axes[0].plot(t, y, color="gray", label="sampled loop c(t)")
axes[0].plot(np.arange(21), c, "o", color="C3", ms=4, label="samples c(kT)")
axes[0].set(xlabel="t (s)", title="Example 2, T = 1 s")
axes[0].legend(fontsize=8)
for T in (0.1, 0.5, 1.0, 2.0):
    nz, dz = zoh([1], [1, 1, 0], T)
    n = int(25 / T) + 1
    axes[1].plot(np.arange(n) * T, signal.lfilter(*closed_loop(nz, dz), np.ones(n)), ".-", ms=3, label=f"T = {T} s")
axes[1].set(xlabel="t (s)", title="Example 2: the samples for several T")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ch10_time_response.png", dpi=150)
print("\nSaved figures/ch10_time_response.png")
