"""Chapter 1 companion: the laser example.

For a > 0, an analog integral controller K/s around 1/(s+a) is stable for every K > 0.
Put the same controller on a computer (sample, hold, integrate with the
rectangle rule) and the loop degrades as the sampling period T grows, and is
unstable for T > 4/a.  This script reproduces the numbers of the chapter:

  * the analog design (a = 2, K = 1): y(t) = 1 - (1 + t) e^{-t}
  * the digital loop  u_k = u_{k-1} + K T e_{k-1},  y_{k+1} = p y_k + b u_k
    with p = e^{-aT}, b = (1 - p)/a  (exact at the samples)
  * step and disturbance peaks for several T, the size of the closed-loop
    roots, the phase margin, and the quantization step of a 12-bit converter.

Run:  python3 ch01_laser_loop.py      (figures go to ./figures)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

a, K = 2.0, 1.0          # plant 1/(s+a), integral gain K (K = a^2/4: critical damping)


# ---------------------------------------------------------------- analog design
def y_analog(t):
    """Unit-step response of K/(s^2 + a s + K) = 1/(s+1)^2."""
    return 1 - (1 + t) * np.exp(-t)


t = np.linspace(0, 20, 200001)
y = y_analog(t)
outside = np.nonzero(np.abs(y - 1) > 0.02)[0]
print("Analog design, a = 2, K = 1:  Y/R = 1/(s+1)^2")
print(f"  2% settling time                  = {t[outside[-1]]:.2f} s")
print(f"  disturbance peak t e^-t at t = 1  = {np.exp(-1):.3f}")


# ---------------------------------------------------------------- digital loop
def digital_loop(T, r=1.0, d=0.0, n=None):
    """Simulate the sampled loop; returns k, y_k.  r: step reference, d: step
    disturbance at the plant input (both start at k = 0, loop at rest)."""
    p = np.exp(-a * T)
    b = (1 - p) / a
    n = n or int(np.ceil(40 / T)) + 2
    y = np.zeros(n)
    u = np.zeros(n)
    e = np.zeros(n)
    for k in range(n - 1):
        e[k] = r - y[k]
        if k >= 1:
            u[k] = u[k - 1] + K * T * e[k - 1]       # rectangle rule, one sample old
        y[k + 1] = p * y[k] + b * (u[k] + d)         # ZOH + plant, exact
    return np.arange(n), y


print("\nDigital loop, same K, growing T")
print("   T (s)   step peak   disturbance peak   |lambda|   decay per second")
for T in (0.1, 0.5, 1.0, 1.5, 2.0, 2.5):
    p = np.exp(-a * T)
    b = (1 - p) / a
    lam2 = p + b * K * T                           # |lambda|^2 = product of the complex roots
    _, ys = digital_loop(T)
    _, yd = digital_loop(T, r=0.0, d=1.0)
    if lam2 < 1:
        print(f"   {T:4.1f}    {ys.max():7.3f}     {yd.max():10.3f}         "
              f"{np.sqrt(lam2):6.3f}      {-np.log(np.sqrt(lam2)) / T:6.3f}")
    else:
        print(f"   {T:4.1f}    {'oscillates' if np.isclose(lam2, 1) else 'grows':>10}"
              f"                         {np.sqrt(lam2):6.3f}      {-np.log(np.sqrt(lam2)) / T + 0.0:6.3f}")

# hand table for T = 1 s
T = 1.0
k, ys = digital_loop(T, n=8)
print("\nT = 1 s, first samples of y_k:", np.round(ys[:7], 4))
print("Stability limit of the digital loop: T < 4/a =", 4 / a, "s")


# ---------------------------------------------------------------- phase margin
# delay estimate: L = 1/(s(s+2)) with a delay of about T
wc = np.sqrt(np.sqrt(5) - 2)
pm0 = 90 - np.degrees(np.arctan(wc / 2))
print(f"\nAnalog loop: crossover {wc:.3f} rad/s, phase margin {pm0:.1f} deg")
print("   T (s)   PM, delay estimate   PM of the sampled loop")
for T in (0.1, 0.5, 1.0, 1.5, 2.0):
    p = np.exp(-a * T)
    b = (1 - p) / a
    w = np.linspace(1e-4, np.pi / T, 400001)
    z = np.exp(1j * w * T)
    L = K * T * b / ((z - 1) * (z - p))            # D(z) = KT/(z-1), G(z) = b/(z-p)
    i = np.argmin(np.abs(np.abs(L) - 1))
    pm = (np.degrees(np.angle(L[i])) + 360) % 360 - 180   # 180 + angle, wrapped to (-180, 180]
    pm = 0.0 if abs(pm) < 0.05 else pm
    print(f"   {T:4.1f}        {pm0 - np.degrees(wc * T):6.1f}             {pm:6.1f}")

# ---------------------------------------------------------------- quantization
q = 20 / 2**12
print(f"\n12-bit converter on +-10 V: q = {1000 * q:.2f} mV, rounding error <= {500 * q:.2f} mV")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
tt = np.linspace(0, 12, 600)
axes[0].plot(tt, y_analog(tt), "k--", lw=1, label="analog design")
for T, c in [(0.5, "C0"), (1.0, "C1"), (1.5, "C3")]:
    k, ys = digital_loop(T, n=int(12 / T) + 1)
    axes[0].plot(k * T, ys, "o-", ms=3, color=c, label=f"samples, T = {T} s")
axes[0].set(xlabel="t (s)", ylabel="y", title="Step response")
axes[0].legend(fontsize=8)
for T, c in [(2.0, "C0"), (2.5, "C3")]:
    k, ys = digital_loop(T, n=int(40 / T) + 1)
    axes[1].plot(k * T, ys, "o-", ms=3, color=c, label=f"T = {T} s")
axes[1].set(xlabel="t (s)", title="At and past the limit T = 4/a")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ch01_laser_step.png", dpi=150)
print("\nSaved figures/ch01_laser_step.png")
