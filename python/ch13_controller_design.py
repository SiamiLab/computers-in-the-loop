"""Chapter 13 companion: designing digital controllers.

Run:  python3 ch13_controller_design.py
Needs numpy, scipy and matplotlib. Prints the numbers of the chapter and saves four plots
in the folder figures/ next to this file:

  ch13_emulation.png     analog lead vs. its Tustin emulation at T = 0.2 s and T = 1 s
  ch13_direct_lead.png   lead D(z) designed on the z-plane (T = 1 s), samples and c(t)
  ch13_deadbeat.png      minimum-time deadbeat (ripple) and ripple-free deadbeat
  ch13_pid.png           digital PID (positional form) on 4/((s+1)(s+4)), T = 0.05 s

The plant of Part A and Part B is G_p(s) = 1/(s(s+1)) driven by a zero-order hold.
Polynomials are numpy-style lists of coefficients in descending powers of s or z.
"""
import os

import matplotlib

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Turning G_c(s) into D(z)
# ---------------------------------------------------------------------------
def substitute(num_s, den_s, top, bot):
    """Replace s by top(z)/bot(z) in num_s/den_s and clear the fractions.

    top and bot are polynomials in z of degree <= 1. Forward Euler: s = (z-1)/T, so
    top = [1, -1], bot = [T]. Backward Euler: top = [1, -1], bot = [T, 0].
    Tustin: top = [2/T, -2/T], bot = [1, 1].
    """
    n = max(len(num_s), len(den_s)) - 1

    def convert(p):
        p = np.concatenate([np.zeros(n + 1 - len(p)), p])
        out = np.zeros(1)
        for i, coef in enumerate(p):
            k = n - i  # this coefficient multiplies s^k
            term = np.array([coef])
            for _ in range(k):
                term = np.polymul(term, top)
            for _ in range(n - k):
                term = np.polymul(term, bot)
            out = np.polyadd(out, term)
        return out

    num_z, den_z = convert(np.asarray(num_s, float)), convert(np.asarray(den_s, float))
    return num_z / den_z[0], den_z / den_z[0]


def forward_euler(num_s, den_s, T):
    return substitute(num_s, den_s, [1.0, -1.0], [T])


def backward_euler(num_s, den_s, T):
    return substitute(num_s, den_s, [1.0, -1.0], [T, 0.0])


def tustin(num_s, den_s, T, prewarp=None):
    """Tustin rule; with prewarp = w1 the factor 2/T becomes w1/tan(w1 T/2)."""
    c = 2.0 / T if prewarp is None else prewarp / np.tan(prewarp * T / 2)
    return substitute(num_s, den_s, [c, -c], [1.0, 1.0])


def matched_pole_zero(num_s, den_s, T):
    """Map poles and zeros by z = e^{sT}, zeros at infinity to z = -1, match the DC gain."""
    zeros, poles = np.roots(num_s), np.roots(den_s)
    zd = list(np.exp(zeros * T)) + [-1.0] * (len(poles) - len(zeros))
    num_z, den_z = np.real(np.poly(zd)), np.real(np.poly(np.exp(poles * T)))
    gain = (np.polyval(num_s, 0) / np.polyval(den_s, 0)) / (np.polyval(num_z, 1) / np.polyval(den_z, 1))
    return gain * num_z, den_z


def zoh_matrices(A, B, T):
    """Phi = e^{AT} and Gamma = int_0^T e^{A tau} d tau B, from one matrix exponential."""
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = B
    P = expm(M * T)
    return P[:n, :n], P[:n, n]


def zoh_equivalent_first_order(K, a, b, T):
    """ZOH (step-invariant) equivalent of K(s+a)/(s+b) = K + K(a-b)/(s+b).

    State space: x' = -b x + e, u = K(a-b) x + K e.
    """
    Phi, Gam = zoh_matrices(np.array([[-b]]), np.array([1.0]), T)
    phi, gam = Phi[0, 0], Gam[0]
    # D(z) = K + K(a-b) gam/(z - phi)
    return np.array([K, -K * phi + K * (a - b) * gam]), np.array([1.0, -phi])


# ---------------------------------------------------------------------------
# 2. The plant seen by the computer, and loop simulation
# ---------------------------------------------------------------------------
A_RUN = np.array([[0.0, 1.0], [0.0, -1.0]])  # 1/(s(s+1)): states c and dc/dt
B_RUN = np.array([0.0, 1.0])
C_RUN = np.array([1.0, 0.0])


def pulse_tf_running(T):
    """G(z) = (b1 z + b0)/((z-1)(z-e^{-T})) for ZOH + 1/(s(s+1))."""
    p = np.exp(-T)
    return np.array([T - 1 + p, 1 - p - T * p]), np.polymul([1, -1], [1, -p])


def simulate_loop(Dn, Dd, A, B, C, T, n_samples, fine=20):
    """Unit-step response of the sampled loop r -> e -> D(z) -> ZOH -> plant -> c.

    The controller runs as a difference equation; the plant is propagated exactly between
    samples with matrix exponentials, so c(t) between the samples is exact too.
    Returns: t_k, c(kT), u(kT), t_fine, c(t_fine).
    """
    Dn = np.concatenate([np.zeros(len(Dd) - len(Dn)), Dn]) / Dd[0]
    Dd = np.asarray(Dd, float) / Dd[0]
    order = len(Dd) - 1
    steps = [zoh_matrices(A, B, T * j / fine) for j in range(fine)]
    Phi, Gam = zoh_matrices(A, B, T)
    x = np.zeros(A.shape[0])
    e_hist = np.zeros(order + 1)  # e(k), e(k-1), ...
    u_hist = np.zeros(order + 1)  # u(k), u(k-1), ...
    ck, uk, tf, cf = [], [], [], []
    for k in range(n_samples):
        e_hist = np.roll(e_hist, 1)
        e_hist[0] = 1.0 - C @ x
        u = Dn @ e_hist - Dd[1:] @ u_hist[:-1]
        u_hist = np.roll(u_hist, 1)
        u_hist[0] = u
        ck.append(C @ x)
        uk.append(u)
        for j, (Pj, Gj) in enumerate(steps):
            tf.append((k + j / fine) * T)
            cf.append(C @ (Pj @ x + Gj * u))
        x = Phi @ x + Gam * u
    return np.arange(n_samples) * T, np.array(ck), np.array(uk), np.array(tf), np.array(cf)


def zeta_of(z):
    """Damping ratio of a z-plane pole (through s = ln(z)/T)."""
    s = np.log(complex(z))
    return -s.real / abs(s)


# ---------------------------------------------------------------------------
# 3. Part A: one lead compensator, six conversions
# ---------------------------------------------------------------------------
print("=== Part A: emulation of Gc(s) = 4(s+1)/(s+2), T = 0.2 s ===")
num_c, den_c = [4.0, 4.0], [1.0, 2.0]
T = 0.2
designs = {
    "forward Euler": forward_euler(num_c, den_c, T),
    "backward Euler": backward_euler(num_c, den_c, T),
    "Tustin": tustin(num_c, den_c, T),
    "Tustin, prewarped": tustin(num_c, den_c, T, prewarp=np.sqrt(2)),
    "matched pole-zero": matched_pole_zero(num_c, den_c, T),
    "ZOH equivalent": zoh_equivalent_first_order(4.0, 1.0, 2.0, T),
}
gn, gd = pulse_tf_running(T)
for name, (Dn, Dd) in designs.items():
    poles = np.roots(np.polyadd(np.polymul(Dd, gd), np.polymul(Dn, gn)))
    pair = poles[np.argmax(poles.imag)]
    _, ck, _, _, _ = simulate_loop(Dn, Dd, A_RUN, B_RUN, C_RUN, T, 200, fine=1)
    print(f"{name:18s} D(z) = {Dn[0]:.4f}(z - {-Dn[1] / Dn[0]:.4f})/(z - {-Dd[1]:.4f})"
          f"   poles {pair.real:.4f} +/- j{pair.imag:.4f}, zeta {zeta_of(pair):.2f},"
          f" overshoot {100 * (ck.max() - 1):.1f}%")

# the same Tustin recipe at T = 0.2 s and at T = 1 s
fig, ax = plt.subplots(figsize=(7, 3.6))
t = np.linspace(0, 14, 1000)
ax.plot(t, 1 - np.exp(-t) * (np.cos(np.sqrt(3) * t) + np.sin(np.sqrt(3) * t) / np.sqrt(3)),
        "k--", label="analog loop")
for Ts, style in [(0.2, "o"), (1.0, "s")]:
    Dn, Dd = tustin(num_c, den_c, Ts)
    tk, ck, _, tf, cf = simulate_loop(Dn, Dd, A_RUN, B_RUN, C_RUN, Ts, int(14 / Ts), fine=10)
    line, = ax.plot(tf, cf, lw=0.8)
    ax.plot(tk, ck, style, ms=3, color=line.get_color(), label=f"Tustin, T = {Ts} s")
    print(f"Tustin at T = {Ts}: peak of the samples {ck.max():.3f}")
ax.axhline(1, color="gray", lw=0.5)
ax.set_xlabel("t (s)")
ax.set_ylabel("c")
ax.legend()
ax.set_title("Emulation: the same analog lead at two sampling periods")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "ch13_emulation.png"), dpi=150)
plt.close(fig)

# the hold's phase lag at crossover
wc = np.sqrt(2 * np.sqrt(5) - 2)
pm = 90 - np.degrees(np.arctan(wc / 2))
print(f"crossover {wc:.3f} rad/s, analog PM {pm:.1f} deg")
for Ts in [0.05, 0.1, 0.2, 0.5, 1.0]:
    lag = np.degrees(wc * Ts / 2)
    print(f"  T = {Ts:4}: hold lag {lag:4.1f} deg, PM estimate {pm - lag:4.1f} deg")

# ---------------------------------------------------------------------------
# 4. Part B: a lead controller designed on the z-plane (T = 1 s)
# ---------------------------------------------------------------------------
print("\n=== Part B: direct design, T = 1 s ===")
T = 1.0
gn, gd = pulse_tf_running(T)
c = np.e - 2  # plant zero at -c
# D(z) = Kc (z - e^-1)/z  ->  1 + DG = 0 is z^2 + (K'-1) z + K' c = 0, K' = Kc e^-1.
# Complex roots lie on |z + c| = sqrt(c(1+c)); intersect with the zeta = 0.5 curve.
theta = brentq(lambda th: np.exp(-th / np.sqrt(3))
               - (-c * np.cos(th) + np.sqrt(c**2 * np.cos(th) ** 2 + c)), 0.1, 2.5)
r = np.exp(-theta / np.sqrt(3))
Kprime = r**2 / c  # product of the roots is K'c
Kc = Kprime * np.e
print(f"design pole {r * np.cos(theta):.4f} +/- j{r * np.sin(theta):.4f}, "
      f"Kc = {Kc:.3f}, Kv = {Kprime * (np.e - 1):.3f}")
Dn, Dd = np.array([Kc, -Kc * np.exp(-1)]), np.array([1.0, 0.0])
tk, ck, uk, tf, cf = simulate_loop(Dn, Dd, A_RUN, B_RUN, C_RUN, T, 15, fine=20)
print("c(k):", np.round(ck[:11], 4))
print("u(k):", np.round(uk[:8], 4))
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(tf, cf, "r-", lw=1, label="c(t)")
ax.plot(tk, ck, "ro", ms=4, label="c(kT)")
ax.step(tk, uk, where="post", color="tab:blue", lw=1, label="u (held)")
ax.axhline(1, color="gray", lw=0.5)
ax.set_xlabel("t (s)")
ax.legend()
ax.set_title("Lead controller 1.158(z - 0.3679)/z, T = 1 s")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "ch13_direct_lead.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Deadbeat: minimum time (ripple) and ripple-free
# ---------------------------------------------------------------------------
def ragazzini(F_num, F_den, gn, gd):
    """D = F/(G(1-F)) for F = F_num/F_den, as polynomials (no cancellation is done)."""
    one_minus_F = np.polysub(F_den, F_num)
    return np.polymul(F_num, gd), np.polymul(gn, one_minus_F)


fig, axes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
for name, (Dn, Dd) in {
    "minimum time, F = 1/z": (np.array([np.e, -1.0]), np.array([1.0, c])),
    "ripple-free, F = (z + c)/((e-1) z^2)": (np.e / (np.e - 1) * np.array([1.0, -np.exp(-1)]),
                                             np.array([1.0, (np.e - 2) / (np.e - 1)])),
}.items():
    tk, ck, uk, tf, cf = simulate_loop(Dn, Dd, A_RUN, B_RUN, C_RUN, T, 12, fine=40)
    print(f"{name}: c(k) = {np.round(ck[:4], 4)}, u(k) = {np.round(uk[:4], 4)}, "
          f"peak of c(t) = {cf.max():.3f}")
    line, = axes[0].plot(tf, cf, lw=1, label=name)
    axes[0].plot(tk, ck, "o", ms=3, color=line.get_color())
    axes[1].step(tk, uk, where="post", color=line.get_color(), lw=1)
# the same controllers from Ragazzini's formula, before cancelling common factors
Dn_rf, Dd_rf = ragazzini(np.array([1.0, c]) / (np.e - 1), np.array([1.0, 0.0, 0.0]), gn, gd)
print("ripple-free D(z), uncancelled form: zeros", np.round(np.roots(Dn_rf), 4),
      "poles", np.round(np.roots(Dd_rf), 4))
axes[0].axhline(1, color="gray", lw=0.5)
axes[0].set_ylabel("c")
axes[0].legend(fontsize=8)
axes[1].set_ylabel("u")
axes[1].set_xlabel("t (s)")
axes[0].set_title("Deadbeat for ZOH + 1/(s(s+1)), T = 1 s")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "ch13_deadbeat.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Digital PID, positional form, on G_p = 4/((s+1)(s+4))
# ---------------------------------------------------------------------------
print("\n=== Digital PID ===")
Kp, Ki, Kd, T, tau_f = 4.0, 4.0, 0.75, 0.05, 0.05
alpha, gD = tau_f / (tau_f + T), Kd / (tau_f + T)
A_p = np.array([[-1.0, 1.0], [0.0, -4.0]])  # x1 = c, x2 = output of 4/(s+4)
B_p = np.array([0.0, 4.0])
C_p = np.array([1.0, 0.0])
Phi, Gam = zoh_matrices(A_p, B_p, T)
x = np.zeros(2)
uI = uD = e_old = 0.0
ck, uk = [], []
for k in range(int(4 / T) + 1):
    c_now = C_p @ x
    e = 1.0 - c_now
    uI = uI + Ki * T * e                       # integral, backward Euler
    uD = alpha * uD + gD * (e - e_old)         # filtered derivative, backward Euler
    u = Kp * e + uI + uD                       # positional form
    e_old = e
    ck.append(c_now)
    uk.append(u)
    x = Phi @ x + Gam * u
ck = np.array(ck)
print(f"u(0) = {uk[0]:.2f}; peak {ck.max():.4f} at t = {ck.argmax() * T:.2f} s")
t = np.linspace(0, 4, 400)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(t, 1 - np.exp(-2 * t) + t * np.exp(-2 * t), "k--", label="analog ideal PID")
ax.plot(np.arange(len(ck)) * T, ck, "r.", ms=4, label="digital PID samples")
ax.axhline(1, color="gray", lw=0.5)
ax.set_xlabel("t (s)")
ax.set_ylabel("c")
ax.legend()
ax.set_title("PID: Kp = 4, Ki = 4, Kd = 0.75, filter 0.05 s, T = 0.05 s")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "ch13_pid.png"), dpi=150)
plt.close(fig)

print(f"\nplots saved in {FIGDIR}")
