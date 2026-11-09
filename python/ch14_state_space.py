#!/usr/bin/env python3
"""Chapter 14 companion: state-space design for a sampled plant.

The running example is a ZOH followed by G_p(s) = 1/(s(s+1)), sampled at T = 1 s,
with states x1 = c (position) and x2 = dc/dt (velocity).

What this script does, in the order of the chapter:
  1. samples the plant exactly: Phi = e^{AT}, Gamma = (int_0^T e^{A s} ds) B;
  2. checks controllability and observability with the rank tests;
  3. places the poles at 0.5 +- j0.5 (Ackermann), and designs a deadbeat gain;
  4. adds the reference gain Nbar, and integral action by state augmentation;
  5. designs prediction and current observers with poles at 0.2, 0.2;
  6. forms the controller-observer D(z) and checks the closed-loop poles;
  7. computes the discrete LQR gain by iterating the Riccati equation.

Each step prints its numbers; the figures go to ./ch14_figures/ next to this file.
Needs numpy, scipy and matplotlib.

Run:  python3 ch14_state_space.py
"""
import os

import numpy as np
import scipy.linalg as sla
import matplotlib

matplotlib.use("Agg")  # write image files; no window needed
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "ch14_figures")
os.makedirs(FIGDIR, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sample_plant(A, B, T):
    """Exact ZOH sampling. The matrix exponential of [[A, B], [0, 0]] * T
    contains Phi in its top-left block and Gamma in its top-right block."""
    n, nu = B.shape
    M = np.zeros((n + nu, n + nu))
    M[:n, :n] = A
    M[:n, n:] = B
    E = sla.expm(M * T)
    return E[:n, :n], E[:n, n:]


def ctrb(Phi, Gam):
    """Controllability matrix [Gam, Phi Gam, ..., Phi^(n-1) Gam]."""
    n = Phi.shape[0]
    return np.hstack([np.linalg.matrix_power(Phi, i) @ Gam for i in range(n)])


def obsv(Phi, C):
    """Observability matrix [C; C Phi; ...; C Phi^(n-1)]."""
    n = Phi.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(Phi, i) for i in range(n)])


def poly_of_matrix(coef, M):
    """alpha(M) = M^n + a1 M^(n-1) + ... + an I, with coef = [1, a1, ..., an]."""
    n = len(coef) - 1
    return sum(c * np.linalg.matrix_power(M, n - i) for i, c in enumerate(coef))


def ackermann(Phi, Gam, coef):
    """State-feedback gain K (a row) with det(zI - Phi + Gam K) = alpha(z)."""
    n = Phi.shape[0]
    last_row = np.zeros((1, n))
    last_row[0, -1] = 1.0
    return last_row @ np.linalg.solve(ctrb(Phi, Gam), poly_of_matrix(coef, Phi))


def observer_gain(Phi, C, coef):
    """Prediction-observer gain L (a column) with det(zI - Phi + L C) = alpha(z).
    Duality: it is the transpose of Ackermann's gain for the pair (Phi^T, C^T)."""
    return ackermann(Phi.T, C.T, coef).T


def show(name, value):
    print(f"{name:>28s} = {value}")


# ---------------------------------------------------------------------------
# 1. the plant and its exact sampled model
# ---------------------------------------------------------------------------
A = np.array([[0.0, 1.0], [0.0, -1.0]])
B = np.array([[0.0], [1.0]])
C = np.array([[1.0, 0.0]])
T = 1.0
Phi, Gam = sample_plant(A, B, T)
a = np.exp(-T)

print("1. Sampled model")
show("Phi", Phi.round(4).tolist())
show("Gamma", Gam.ravel().round(4).tolist())
# compare with the closed forms found by hand
assert np.allclose(Phi, [[1, 1 - a], [0, a]]) and np.allclose(Gam, [[T - 1 + a], [1 - a]])
# pulse transfer function G(z) = C (zI - Phi)^{-1} Gamma at one test point
z0 = 0.3 + 0.7j
G_state = (C @ np.linalg.solve(z0 * np.eye(2) - Phi, Gam))[0, 0]
G_table = (a * z0 + 1 - 2 * a) / ((z0 - 1) * (z0 - a))
show("G(z0) two ways", f"{G_state:.6f}  {G_table:.6f}")

# ---------------------------------------------------------------------------
# 2. controllability and observability
# ---------------------------------------------------------------------------
print("\n2. Rank tests")
show("det Ctrb", round(np.linalg.det(ctrb(Phi, Gam)), 4))
show("det Obsv", round(np.linalg.det(obsv(Phi, C)), 4))
# the oscillator 1/(s^2 + 1) sampled at T = pi loses both
Ao = np.array([[0.0, 1.0], [-1.0, 0.0]])
for To in [np.pi / 2, np.pi]:
    Po, Go = sample_plant(Ao, B, To)
    show(f"oscillator rank Ctrb, T={To:.3f}", np.linalg.matrix_rank(ctrb(Po, Go), tol=1e-9))

# ---------------------------------------------------------------------------
# 3. pole placement and deadbeat
# ---------------------------------------------------------------------------
print("\n3. State feedback")
alpha_c = [1.0, -1.0, 0.5]  # (z - 0.5 - 0.5j)(z - 0.5 + 0.5j)
K = ackermann(Phi, Gam, alpha_c)
show("K (poles 0.5 +- j0.5)", K.ravel().round(4).tolist())
show("eig(Phi - Gamma K)", np.linalg.eigvals(Phi - Gam @ K).round(4).tolist())
Kdb = ackermann(Phi, Gam, [1.0, 0.0, 0.0])
show("deadbeat K", Kdb.ravel().round(4).tolist())
show("(Phi - Gamma Kdb)^2", np.abs((Phi - Gam @ Kdb) @ (Phi - Gam @ Kdb)).round(12).tolist())


def regulate(Kgain, x0, steps):
    """Run x(k+1) = (Phi - Gamma K) x(k); also return c(t) between samples.
    Inside one period the input is constant, so the exact solution is
    c(kT + tau) = x1 + x2 (1 - e^{-tau}) + m (tau - 1 + e^{-tau})."""
    x = np.array(x0, dtype=float)
    t_fine, c_fine, m_seq = [], [], []
    for k in range(steps):
        m = -(Kgain @ x)[0]
        m_seq.append(m)
        for tau in np.linspace(0, T, 21)[:-1]:
            t_fine.append(k * T + tau)
            c_fine.append(x[0] + x[1] * (1 - np.exp(-tau)) + m * (tau - 1 + np.exp(-tau)))
        x = Phi @ x + Gam[:, 0] * m
    t_fine.append(steps * T)
    c_fine.append(x[0])
    return np.array(t_fine), np.array(c_fine), np.array(m_seq)


fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for Kgain, lab in [(K, "poles 0.5 +- j0.5"), (Kdb, "deadbeat")]:
    t, c, m = regulate(Kgain, [1.0, 0.0], 8)
    ax[0].plot(t, c, label=lab)
    ax[0].plot(t[::20], c[::20], "o", ms=3, color=ax[0].lines[-1].get_color())
    ax[1].step(np.arange(9), np.append(m, m[-1]), where="post", label=lab)
ax[0].set(xlabel="t (s)", ylabel="c(t)", title="regulation from x(0) = [1, 0]")
ax[1].set(xlabel="t (s)", ylabel="m(t)", title="control (held by the ZOH)")
for a_ in ax:
    a_.grid(alpha=0.3)
    a_.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "ch14_pole_placement.png"), dpi=150)
plt.close(fig)

# deadbeat effort grows like 1/T^2
print("   deadbeat effort for several T (x(0) = [1, 0]):")
for Ts in [2.0, 1.0, 0.5, 0.2, 0.1]:
    P_, G_ = sample_plant(A, B, Ts)
    Kt = ackermann(P_, G_, [1.0, 0.0, 0.0])
    print(f"      T = {Ts:4.1f}   K = {Kt.ravel().round(3)}   m(0) = {-Kt[0, 0]:9.3f}")

# ---------------------------------------------------------------------------
# 4. reference gain and integral action
# ---------------------------------------------------------------------------
print("\n4. Tracking")
Nbar = 1.0 / (C @ np.linalg.solve(np.eye(2) - Phi + Gam @ K, Gam))[0, 0]
show("Nbar", round(Nbar, 4))
# augmented model: x_I(k+1) = x_I(k) + r(k) - c(k)
Phi_a = np.block([[Phi, np.zeros((2, 1))], [-C, np.ones((1, 1))]])
Gam_a = np.vstack([Gam, [[0.0]]])
Ka = ackermann(Phi_a, Gam_a, np.polymul(alpha_c, [1.0, -0.5]))
k1, k2, kI = Ka[0, 0], Ka[0, 1], -Ka[0, 2]
show("[k1, k2, kI]", [round(float(k1), 4), round(float(k2), 4), round(float(kI), 4)])

N = 31
w = np.where(np.arange(N) >= 15, 0.2, 0.0)  # constant input disturbance from k = 15
c_nbar, c_int = np.zeros(N), np.zeros(N)
x = np.zeros(2)
for k in range(N):
    c_nbar[k] = x[0]
    m = -(K @ x)[0] + Nbar * 1.0
    x = Phi @ x + Gam[:, 0] * (m + w[k])
xa = np.zeros(3)
for k in range(N):
    c_int[k] = xa[0]
    m = -(Ka @ xa)[0]
    xa = Phi_a @ xa + Gam_a[:, 0] * (m + w[k]) + np.array([0.0, 0.0, 1.0])
show("final c, Nbar only", round(c_nbar[-1], 3))
show("final c, integral action", round(c_int[-1], 3))

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(c_nbar, "o-", ms=3, label="Nbar only")
ax.plot(c_int, "s-", ms=3, label="integral action")
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.axvline(15, color="gray", lw=0.8, ls="--")
ax.set(xlabel="k", ylabel="c(k)", title="unit step; +0.2 added at the plant input from k = 15")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "ch14_integral_action.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. observers
# ---------------------------------------------------------------------------
print("\n5. Observers, poles at 0.2, 0.2")
alpha_o = [1.0, -0.4, 0.04]
L = observer_gain(Phi, C, alpha_o)
Lc = observer_gain(Phi, C @ Phi, alpha_o)  # current observer: C Phi in place of C
show("prediction L", L.ravel().round(4).tolist())
show("current Lc", Lc.ravel().round(4).tolist())
show("eig(Phi - L C)", np.linalg.eigvals(Phi - L @ C).real.round(4).tolist())
show("eig(Phi - Lc C Phi)", np.linalg.eigvals(Phi - Lc @ C @ Phi).real.round(4).tolist())

# separation: feed back the estimate, compare with true-state feedback
steps = 11
x, xh, xs = np.array([1.0, 0.0]), np.zeros(2), np.array([1.0, 0.0])
c_obs, c_sf, xh1 = [], [], []
for k in range(steps):
    c_obs.append(x[0])
    c_sf.append(xs[0])
    xh1.append(xh[0])
    m = -(K @ xh)[0]
    m_sf = -(K @ xs)[0]
    c_meas = (C @ x)[0]
    xh = Phi @ xh + Gam[:, 0] * m + L[:, 0] * (c_meas - (C @ xh)[0])
    x = Phi @ x + Gam[:, 0] * m
    xs = Phi @ xs + Gam[:, 0] * m_sf
big = np.block([[Phi, -Gam @ K], [L @ C, Phi - Gam @ K - L @ C]])
show("closed-loop poles (x, xhat)", np.sort_complex(np.linalg.eigvals(big)).round(4).tolist())

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(c_sf, "o-", ms=3, label="c(k), m = -K x")
ax.plot(c_obs, "s-", ms=3, label="c(k), m = -K xhat")
ax.plot(xh1, "^--", ms=3, label="estimate xhat1(k)")
ax.set(xlabel="k", title="observer-based regulator, xhat(0) = 0")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "ch14_observer.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. the controller-observer as a transfer function D(z)
# ---------------------------------------------------------------------------
print("\n6. D(z) = K (zI - Phi + Gamma K + L C)^{-1} L")
F = Phi - Gam @ K - L @ C
den_D = np.poly(F)  # det(zI - F)
# numerator: D(z) * det(zI - F) is a first-degree polynomial; fit it from two points
zz = np.array([0.0, 1.0])
vals = [(K @ np.linalg.solve(z * np.eye(2) - F, L))[0, 0] * np.polyval(den_D, z) for z in zz]
num_D = np.array([vals[1] - vals[0], vals[0]])
show("numerator of D", num_D.round(4).tolist())
show("denominator of D", den_D.round(4).tolist())
num_G, den_G = np.array([a, 1 - 2 * a]), np.poly([1.0, a])
char = np.polyadd(np.polymul(den_G, den_D), np.polymul(num_G, num_D))
show("1 + D G = 0, polynomial", char.round(4).tolist())
show("roots", np.sort_complex(np.roots(char)).round(4).tolist())

# ---------------------------------------------------------------------------
# 7. discrete LQR
# ---------------------------------------------------------------------------
print("\n7. LQR, Q = C^T C")
Q = C.T @ C


def lqr_by_iteration(rho, iters=200):
    """Iterate P <- Q + Phi'P Phi - Phi'P Gam (rho + Gam'P Gam)^{-1} Gam'P Phi from P = 0."""
    P = np.zeros((2, 2))
    for _ in range(iters):
        s = rho + (Gam.T @ P @ Gam)[0, 0]
        P = Q + Phi.T @ P @ Phi - (Phi.T @ P @ Gam) @ (Gam.T @ P @ Phi) / s
    Kl = (Gam.T @ P @ Phi) / (rho + (Gam.T @ P @ Gam)[0, 0])
    return P, Kl


fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for rho in [10.0, 1.0, 0.1]:
    P, Kl = lqr_by_iteration(rho)
    P_are = sla.solve_discrete_are(Phi, Gam, Q, np.array([[rho]]))
    assert np.allclose(P, P_are)  # iteration and the direct solver agree
    poles = np.linalg.eigvals(Phi - Gam @ Kl)
    print(f"   rho = {rho:4.1f}:  K = {Kl.ravel().round(4)}   poles = {poles.round(4)}   |z| = {abs(poles).round(4)}")
    x = np.array([1.0, 0.0])
    cs, ms = [], []
    for k in range(11):
        cs.append(x[0])
        ms.append(-(Kl @ x)[0])
        x = (Phi - Gam @ Kl) @ x
    ax[0].plot(cs, "o-", ms=3, label=f"rho = {rho:g}")
    ax[1].step(range(11), ms, where="post", label=f"rho = {rho:g}")
ax[0].set(xlabel="k", ylabel="c(k)", title="LQR regulation from x(0) = [1, 0]")
ax[1].set(xlabel="t (s)", ylabel="m(t)", title="LQR control")
for a_ in ax:
    a_.grid(alpha=0.3)
    a_.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "ch14_lqr.png"), dpi=150)
plt.close(fig)

print(f"\nFigures written to {FIGDIR}")
