"""Chapter 11 companion: stability of sampled-data systems.

Reproduces:
  * the sampled integrator under feedback, H(z) = T/(z - (1 - KT)), T = 0.1 s:
    sum |h| for K = 5, 10, 15, 25; stable for 0 < K < 2/T;
  * Routh's array (with the epsilon rule and the auxiliary polynomial), on the
    chapter's examples: s^4 + s^3 + 2s^2 + 3s + 2, the gain range 0 < K < 204/25,
    s^3 - 7s + 6, s^3 + 3s^2 + 4s + 12, and s^5 + s^4 + 3s^3 + 3s^2 - 4s - 4;
  * the bilinear substitution Q_w(w) = (1-w)^n Q((1+w)/(1-w)) followed by Routh,
    and Jury's table, both on
      ZOH + 1/(s(s+1)), T = 1 s, gain K:            0 < K < 2.39
      the same loop with one sample of delay:       0 < K < 0.805
      the laser loop with D = k:                    -2 < k < 54.7 (T = 0.1 s)
      the exit ticket, accumulator on 0.5/(z-0.5):  0 < K < 6
    every range is cross-checked with the roots of the characteristic polynomial.

Run:  python3 ch11_stability.py      (figures go to ./figures)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P
from scipy import signal
from scipy.optimize import brentq

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)


# ---------------------------------------------------------------- tools
def routh(c, eps=1e-6, verbose=False, var="s"):
    """Routh array of the polynomial with descending coefficients c.
    Returns the first column and the number of sign changes (roots in Re s > 0)."""
    c = np.array(c, float)
    n = len(c) - 1
    width = n // 2 + 1
    pad = lambda r: np.r_[r, np.zeros(width - len(r))]
    rows = [pad(c[0::2]), pad(c[1::2])]
    notes = []
    for r in range(1, n + 1):
        cur = rows[r]
        if np.all(np.abs(cur) < 1e-12):                     # a row of zeros
            up, d = rows[r - 1], n - r + 1
            aux = [float(up[j]) for j in range(width) if d - 2 * j >= 0]
            notes.append(f"row {var}^{n - r} is zero: auxiliary P(s) of degree {d}, coefficients {aux}")
            rows[r] = cur = pad(np.array([up[j] * (d - 2 * j) for j in range(width) if d - 2 * j > 0]))
        elif abs(cur[0]) < 1e-12:                           # a zero first entry
            notes.append(f"zero first entry in row {var}^{n - r}: replaced by eps = {eps}")
            cur = cur.copy()
            cur[0] = eps
            rows[r] = cur
        if r == n:
            break
        up = rows[r - 1]
        rows.append(np.array([(cur[0] * up[j + 1] - up[0] * cur[j + 1]) / cur[0]
                              for j in range(width - 1)] + [0.0]))
    first = np.array([row[0] for row in rows[:n + 1]])
    changes = int(np.sum(np.sign(first[:-1]) != np.sign(first[1:])))
    if verbose:
        for r, row in enumerate(rows[:n + 1]):
            print(f"    {var}^{n - r}: " + "  ".join(f"{v:9.4g}" for v in row))
        for t in notes:
            print("    " + t)
    return first, changes


def bilinear(c):
    """Q_w(w) = (1-w)^n Q((1+w)/(1-w)) for Q with descending coefficients c."""
    n = len(c) - 1
    out = np.zeros(n + 1)
    for i, ci in enumerate(c):
        k = n - i
        out += ci * P.polymul(P.polypow([1, 1], k), P.polypow([1, -1], n - k))[:n + 1]
    return out[::-1]


def jury(c):
    """Jury's test on descending coefficients (leading coefficient > 0)."""
    a = np.array(c[::-1], float)                 # a0, a1, ..., an
    n = len(a) - 1
    if not (np.sum(a) > 0 and (-1) ** n * np.polyval(c, -1) > 0 and abs(a[0]) < a[-1]):
        return False
    row = a
    while len(row) > 3:
        m = len(row) - 1
        row = np.array([row[0] * row[j] - row[m] * row[m - j] for j in range(m)])
        if not abs(row[0]) > abs(row[-1]):
            return False
    return True


def stable_by_roots(c):
    return bool(np.all(np.abs(np.roots(c)) < 1))


def gain_limit(qfun, lo, hi, test):
    """Largest gain in (lo, hi) at which test(qfun(K)) still holds (bisection on the edge)."""
    return brentq(lambda K: (1.0 if test(qfun(K)) else -1.0), lo, hi, xtol=1e-10)


# ---------------------------------------------------------------- the sampled integrator
T = 0.1
print("Sampled integrator T/(z-1) with feedback K, T = 0.1 s: H(z) = T/(z - (1 - KT))")
for K in (5, 10, 15, 25):
    pole = 1 - K * T
    h = T * pole ** np.arange(200.0)
    s = np.sum(np.abs(h))
    print(f"  K = {K:2d}: pole {pole:5.2f},  sum |h| = {s:.3g}" + ("  (grows)" if abs(pole) >= 1 else ""))
print(f"  stable for 0 < K < 2/T = {2 / T:g}")

# ---------------------------------------------------------------- Routh examples
print("\nRouth: s^4 + s^3 + 2s^2 + 3s + 2")
_, ch = routh([1, 1, 2, 3, 2], verbose=True)
print(f"  sign changes: {ch}; roots {np.round(np.roots([1, 1, 2, 3, 2]), 3)}")
Kmax = gain_limit(lambda K: [1, 5, 8, 6, K], 0.1, 12, lambda c: routh(c)[1] == 0)
print(f"Gain range of s^4 + 5s^3 + 8s^2 + 6s + K: 0 < K < {Kmax:.4f} (204/25 = {204 / 25:.4f})")
print(f"  at K = 204/25 the roots on the axis: {np.round(np.roots([1, 5, 8, 6, 204 / 25]), 3)}")
for c in ([1, 0, -7, 6], [1, 3, 4, 12], [1, 1, 3, 3, -4, -4]):
    print("Routh:", c)
    _, ch = routh(c, verbose=True)
    print(f"  right half-plane roots: {ch};  roots {np.round(np.roots(c), 3)}")

# ---------------------------------------------------------------- sampled loops
e1 = np.exp(-1.0)
# laser loop: exact ZOH equivalent of 2/((s+1)(s+4)) with T = 0.1 s
nl, dl, _ = signal.cont2discrete(([2], [1, 5, 4]), 0.1, method="zoh")
nl = np.atleast_2d(nl)[0][-2:]
print(f"\nLaser loop, T = 0.1 s: G(z) = ({nl[0]:.4f} z + {nl[1]:.4f})/(z^2 - {-dl[1]:.4f} z + {dl[2]:.4f})")
loops = {
    "ZOH + 1/(s(s+1)), T = 1 s, gain K":
        (lambda K: [1, e1 * K - (1 + e1), e1 + (1 - 2 * e1) * K], 0.01, 10),
    "same loop, one sample of delay (D = K/z)":
        (lambda K: [1, -(1 + e1), e1 * (1 + K), (1 - 2 * e1) * K], 0.01, 3),
    "laser loop, T = 0.1 s, D = k":
        (lambda K: [1, dl[1] + nl[0] * K, dl[2] + nl[1] * K], 0.01, 200),
    "exit ticket: D = Kz/(z-1), G = 0.5/(z-0.5)":
        (lambda K: [1, 0.5 * K - 1.5, 0.5], 0.01, 20),
}
print("\nUpper gain limits, three ways")
for name, (q, lo, hi) in loops.items():
    k_roots = gain_limit(q, lo, hi, stable_by_roots)
    k_bil = gain_limit(q, lo, hi, lambda c: routh(bilinear(c))[1] == 0 and np.all(bilinear(c) > 0))
    k_jury = gain_limit(q, lo, hi, jury)
    print(f"  {name:45s}: roots {k_roots:.4f}, bilinear + Routh {k_bil:.4f}, Jury {k_jury:.4f}")

print("\nThe delayed loop, K = 0.5: Q_w(w) =", np.round(bilinear(loops["same loop, one sample of delay (D = K/z)"][0](0.5)), 4))
routh(bilinear(loops["same loop, one sample of delay (D = K/z)"][0](0.5)), verbose=True, var="w")
# laser loop: the exact ZOH coefficients used above come from G(z) of 2/((s+1)(s+4)), T = 0.1 s
print("Laser loop lower limit: Q(1) = 0 at k =", round(brentq(lambda k: np.polyval(
    loops["laser loop, T = 0.1 s, D = k"][0](k), 1), -10, 0), 3))

# ---------------------------------------------------------------- figure
Ks = np.linspace(0.01, 3.0, 600)
r0 = [np.max(np.abs(np.roots(loops["ZOH + 1/(s(s+1)), T = 1 s, gain K"][0](K)))) for K in Ks]
r1 = [np.max(np.abs(np.roots(loops["same loop, one sample of delay (D = K/z)"][0](K)))) for K in Ks]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(Ks, r0, label="D = K")
axes[0].plot(Ks, r1, label="D = K/z (one sample of delay)")
axes[0].axhline(1, color="k", lw=0.6)
axes[0].set(xlabel="K", ylabel="largest |pole|", title="ZOH + 1/(s(s+1)), T = 1 s")
axes[0].legend(fontsize=8)
tri = np.array([[-2, 1], [2, 1], [0, -1], [-2, 1]])
axes[1].fill(tri[:, 0], tri[:, 1], alpha=0.2, label="stable: |a0| < 1, Q(1) > 0, Q(-1) > 0")
Kp = np.linspace(0, 3, 50)
axes[1].plot(e1 * Kp - (1 + e1), e1 + (1 - 2 * e1) * Kp, "r", label="(a1, a0) as K goes 0 to 3")
axes[1].set(xlabel="a1", ylabel="a0", title="Second-order test for z^2 + a1 z + a0")
axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "ch11_stability.png", dpi=150)
print("\nSaved figures/ch11_stability.png")
