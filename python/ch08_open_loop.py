"""Chapter 8 companion: open-loop sampled-data systems.

Reproduces:
  * E(s) = 1/((s+1)(s+2)), T = 0.5 s: E(z) = 0.239 z/((z - 0.607)(z - 0.368)),
    from the samples and from the residues;
  * ZOH + 1/(s+1): the pulse transfer function (1 - e^-T)/(z - e^-T), by brute
    force (integrating the plant over one period) and by an exact state-space
    discretization; step samples 0, 0.393, 0.632, 0.777 (T = 0.5 s); DC gain G(1);
  * where the samplers sit: G1 = 1/s, G2 = 1/(s+1), impulse input,
    (a) a sampler between the blocks, (b) no sampler between them;
  * the digital filter m(k) = 2e(k) - e(k-1) in front of ZOH + 1/(s+1);
  * how large T may be: ZOH equivalents of 25/(s^2 + 2s + 25) for T = 1 s and
    T = 0.1 s, and the true peak 1.527 against the largest sample.

Run:  python3 ch08_open_loop.py      (figures go to ./figures)
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
    """ZOH equivalent of num/den: Phi, Gamma from one matrix exponential, then
    G(z) = C (zI - Phi)^-1 Gamma + D as numerator/denominator in powers of z."""
    A, B, C, D = signal.tf2ss(num, den)
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * T)
    Phi, Gam = E[:n, :n], E[:n, n:]
    numz, denz = signal.ss2tf(Phi, Gam, C, D)
    return np.trim_zeros(np.round(numz[0], 14), "f"), denz


def sg(v, d=4):
    """' + 0.1234' or ' - 0.1234' for printing polynomials."""
    return f" {'-' if v < 0 else '+'} {abs(v):.{d}f}"


# ---------------------------------------------------------------- E(z) of a sampled signal
T = 0.5
a, b = np.exp(-T), np.exp(-2 * T)
k = np.arange(8)
print("E(s) = 1/((s+1)(s+2)), T = 0.5 s")
print(f"  E(z) = {a - b:.3f} z/((z - {a:.3f})(z - {b:.3f}))")
series = signal.lfilter([0, a - b], np.polymul([1, -a], [1, -b]), np.r_[1.0, np.zeros(7)])
print("  samples e^-kT - e^-2kT :", np.round(np.exp(-k * T) - np.exp(-2 * k * T), 4))
print("  series of E(z)         :", np.round(series, 4))

# ---------------------------------------------------------------- ZOH + 1/(s+1)
numz, denz = zoh([1], [1, 1], T)
print("\nZOH + 1/(s+1), T = 0.5 s")
print(f"  exact discretization: G(z) = {numz[-1]:.4f}/(z{sg(denz[1])});  (1 - e^-T)/(z - e^-T) = "
      f"{1 - a:.4f}/(z - {a:.4f})")
c = np.zeros(6)
for kk in range(5):                      # brute force: integrate the plant over one period
    c[kk + 1] = a * c[kk] + (1 - a) * 1.0
print("  step samples (brute force):", np.round(c, 3), "  1 - e^-kT:", np.round(1 - np.exp(-np.arange(6) * T), 3))
print(f"  DC gain G(1) = {np.polyval(numz, 1) / np.polyval(denz, 1):.4f} = G_p(0)")

# ---------------------------------------------------------------- configurations (a) and (b)
imp = np.r_[1.0, np.zeros(5)]
ca = signal.lfilter([1, 0, 0], np.polymul([1, -1], [1, -a]), imp)          # G1(z) G2(z)
cb = signal.lfilter([0, 1 - a, 0], np.polymul([1, -1], [1, -a]), imp)      # G1G2(z)
print("\nG1 = 1/s, G2 = 1/(s+1), T = 0.5 s, unit impulse sequence in")
print("  (a) sampler between the blocks, G1(z)G2(z):", np.round(ca[:4], 3))
print("  (b) no sampler between them,    G1G2(z)   :", np.round(cb[:4], 3))

# ---------------------------------------------------------------- digital filter
print("\nDigital filter D(z) = (2z - 1)/z before ZOH + 1/(s+1), unit step, T = 0.5 s")
m = signal.lfilter([2, -1], [1], np.ones(8))                # m(k) = 2e(k) - e(k-1)
cf = signal.lfilter([0, 1 - a], [1, -a], m)
print("  with the filter   :", np.round(cf[:5], 3), "  formula 1 - (2a-1) a^(k-1):",
      np.round(np.r_[0, 1 - (2 * a - 1) * a ** (np.arange(1, 5) - 1)], 3))
print("  without it (D = 1):", np.round(signal.lfilter([0, 1 - a], [1, -a], np.ones(5)), 3))

# ---------------------------------------------------------------- how large can T be?
num, den = [25], [1, 2, 25]
tt = np.linspace(0, 9, 90001)
_, yc = signal.step((num, den), T=tt)
print("\nG_p = 25/(s^2 + 2s + 25): zeta = 0.2, wn = 5 rad/s")
print(f"  continuous step: peak {yc.max():.3f} at t = {tt[yc.argmax()]:.2f} s,"
      f" period 2 pi/wd = {2 * np.pi / (5 * np.sqrt(1 - 0.04)):.2f} s")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, T in zip(axes, (1.0, 0.1)):
    numz, denz = zoh(num, den, T)
    n = int(round((6 if T < 0.5 else 9) / T)) + 1
    cs = signal.lfilter(np.r_[0, numz], denz, np.ones(n))
    print(f"  T = {T}: G(z) = ({numz[0]:.4f} z{sg(numz[1])})/(z^2{sg(denz[1])} z{sg(denz[2])}),"
          f" G(1) = {np.polyval(numz, 1) / np.polyval(denz, 1):.4f}")
    print(f"          samples {np.round(cs[:4], 3)} ...; largest sample {cs.max():.3f}"
          f" at k = {cs.argmax()} (t = {cs.argmax() * T:.1f} s)")
    m_ = tt <= (n - 1) * T
    ax.plot(tt[m_], yc[m_], color="gray", label="c(t)")
    ax.plot(np.arange(n) * T, cs, "o", color="C3", ms=4, label="c(kT)")
    ax.set(xlabel="t (s)", title=f"T = {T} s")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ch08_how_large_T.png", dpi=150)
print("\nSaved figures/ch08_how_large_T.png")
