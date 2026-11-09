#!/usr/bin/env python3
"""Chapter 16 companion: how fast must networked agents sample?

Run it with
    python3 ch16_networks_sampling.py
It prints the main numbers of the chapter and saves six figures (PNG) in a folder
ch16_figures/ next to this file:

  0. ch16_agreement.png     three agents in a line agree by hand (T = 0.5) or not (T = 0.8)
  1. ch16_roots.png         Euler roots of one agent as T grows (alpha = beta = 1 and 5)
  2. ch16_tmax_vs_lmax.png  largest stable T against the largest Laplacian eigenvalue
  3. ch16_fast_slow.png     positions of five agents: fast and slow agents, sparse and dense graphs
  4. ch16_density.png       largest stable T as links are added, path to complete, 8 agents
  5. ch16_crossing.png      steady-state disagreement under noise: ring against complete graph

Model (forward Euler, as in the notes): x(k+1) = x + T v, v(k+1) = v + T m, with
m = -Lx x - Lv v. Requires numpy, scipy and matplotlib. Change the CAPITAL constants to experiment.
"""
import itertools
import math
import os

import numpy as np
from scipy import linalg, optimize

import matplotlib
matplotlib.use("Agg")            # draw to files, no window needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ch16_figures")
os.makedirs(OUT, exist_ok=True)
SPARSE, DENSE = "#1D5FA8", "#C8102E"          # the notes' blue and red
AGENTS = ["#1D5FA8", "#C8102E", "#2E7D32", "#E67E22", "#1F2A33"]   # agents 1-5, as in the notes


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------------------------
# Graphs and Laplacians
# ---------------------------------------------------------------------------------------------
def laplacian(edges, n):
    L = np.zeros((n, n))
    for i, j in edges:
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
    return L


def four_graphs(n):
    return {"path": [(i, i + 1) for i in range(n - 1)],
            "ring": [(i, (i + 1) % n) for i in range(n)],
            "star": [(0, i) for i in range(1, n)],
            "complete": list(itertools.combinations(range(n), 2))}


# ---------------------------------------------------------------------------------------------
# One agent (one mode): the bounds and the variances
# ---------------------------------------------------------------------------------------------
def tmax_euler(alpha, beta):
    """Largest stable T of x(k+1) = x + T v, v(k+1) = v - T(alpha x + beta v)."""
    if beta ** 2 < 4 * alpha:
        return beta / alpha                      # complex pair leaves through |z| = 1
    return 4 / (beta + math.sqrt(beta ** 2 - 4 * alpha))   # real root leaves through z = -1


def tmax_exact(alpha, beta):
    """Same feedback, exact zero-order-hold model (Gamma = [T^2/2, T])."""
    return min(2 / beta, 2 * beta / alpha)


def mode_variances(alpha, beta, T):
    """Steady-state variances (p11, p22) of one mode under unit-intensity noise."""
    den = (beta - T * alpha) * (4 - 2 * T * beta + T * T * alpha)
    return (2 - T * beta + T * T * alpha) / (alpha * den), 2 / den


def network(Lx, Lv, T):
    n = Lx.shape[0]
    return np.block([[np.eye(n), T * np.eye(n)], [-T * Lx, np.eye(n) - T * Lv]])


def disagreement_radius(Lx, Lv, T):
    """Largest root size of the network once the common motion (mode 1) is removed."""
    n = Lx.shape[0]
    lam, U = np.linalg.eigh(Lx)
    W = U[:, 1:]                                  # orthonormal basis orthogonal to the ones vector
    P = np.kron(np.eye(2), W)
    return max(abs(np.linalg.eigvals(P.T @ network(Lx, Lv, T) @ P)))


def largest_stable_T(Lx, Lv, hi=3.0):
    """Bisection on the root size: an independent check of the closed-form bound."""
    Ts = np.linspace(1e-4, hi, 3000)
    bad = next(T for T in Ts if disagreement_radius(Lx, Lv, T) >= 1)
    lo, up = 1e-4, bad
    for _ in range(60):
        mid = (lo + up) / 2
        lo, up = (mid, up) if disagreement_radius(Lx, Lv, mid) < 1 else (lo, mid)
    return lo


# ---------------------------------------------------------------------------------------------
# 0. Agreement by hand: agents that set their own velocity, x(k+1) = x(k) - T L x(k)
# ---------------------------------------------------------------------------------------------
print("Two agents, one link, from x = (4, 0):")
for T in [0.25, 1.0, 1.25]:
    x = np.array([4.0, 0.0])
    rows = [x.copy()]
    for _ in range(3):
        x = x - T * laplacian([(0, 1)], 2) @ x
        rows.append(x.copy())
    print(f"  T = {T}: " + "  ".join(f"({a:g}, {b:g})" for a, b in rows) + f"   gap factor 1 - 2T = {1 - 2 * T:g}")
L3 = laplacian([(0, 1), (1, 2)], 3)
print(f"Three agents in a line: eigenvalues of L = {np.round(np.linalg.eigvalsh(L3), 6)}; agreement for T < 2/3")
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, T in zip(axs, [0.5, 0.8]):
    x = np.array([3.0, 0.0, 0.0])
    X = [x.copy()]
    for _ in range(10):
        x = x - T * L3 @ x
        X.append(x.copy())
    X = np.array(X)
    for i in range(3):
        ax.plot(X[:, i], "-o", ms=3, color=AGENTS[i], label=f"agent {i + 1}")
    ax.set_title(f"three agents in a line, T = {T}", fontsize=10)
    ax.set_xlabel("k")
    ax.set_ylim((-12, 12) if T > 0.6 else (-1, 3.4))
axs[0].legend(fontsize=8)
save(fig, "ch16_agreement.png")
print("Two masses with a spring and a damper (alpha = beta = 2): Tmax =", tmax_euler(2, 2))

# ---------------------------------------------------------------------------------------------
# 1. One agent: the bound and where the roots go
# ---------------------------------------------------------------------------------------------
print("One agent, m = -alpha x - beta v")
for a, b in [(1, 1), (5, 5), (2, 1), (2, 4)]:
    print(f"  alpha = {a}, beta = {b}:  Tmax Euler = {tmax_euler(a, b):.4f},"
          f"  exact model = {tmax_exact(a, b):.4f}")
fig, ax = plt.subplots(figsize=(5.2, 4.6))
th = np.linspace(0, 2 * math.pi, 300)
ax.plot(np.cos(th), np.sin(th), color="0.2", lw=1)
for (a, b), col, Tend in [((1, 1), SPARSE, 1.0), ((5, 5), DENSE, tmax_euler(5, 5))]:
    for s in np.roots([1, b, a]):                  # Euler roots move as z = 1 + T s
        Tline = np.linspace(0, Tend, 50)
        z = 1 + Tline * s
        ax.plot(z.real, z.imag, color=col, lw=2.5)
        ax.plot(z[-1].real, z[-1].imag, "o", color=col)
ax.set_aspect("equal")
ax.axhline(0, color="0.6", lw=0.6)
ax.axvline(0, color="0.6", lw=0.6)
ax.set_title("Euler roots z = 1 + Ts: blue alpha = beta = 1 (to T = 1),\n"
             "red alpha = beta = 5 (to T = 0.553)", fontsize=9)
save(fig, "ch16_roots.png")

# ---------------------------------------------------------------------------------------------
# 2. Five agents, four graphs, L_x = L_v = L
# ---------------------------------------------------------------------------------------------
N_AGENTS = 5
G5 = four_graphs(N_AGENTS)
print(f"\n{N_AGENTS} agents, Lx = Lv = L")
print("  graph      links  dmax  lambda_max  Tmax(formula)  Tmax(bisection)  exact model  platoon")
EIG = {}
for name, edges in G5.items():
    L = laplacian(edges, N_AGENTS)
    lam = np.sort(np.linalg.eigvalsh(L))
    EIG[name] = lam
    t_form = min(tmax_euler(l, l) for l in lam[1:])
    t_bis = largest_stable_T(L, L)
    t_exact = min(tmax_exact(l, l) for l in lam[1:])
    t_plat = min(tmax_euler(l + 1, l + 1) for l in lam)
    print(f"  {name:9s}  {len(edges):5d}  {int(L.diagonal().max()):4d}  {lam[-1]:10.4f}"
          f"  {t_form:13.4f}  {t_bis:15.4f}  {t_exact:11.4f}  {t_plat:7.4f}")
    print(f"             eigenvalues {np.round(lam, 4)}")

lm = np.linspace(0.05, 10, 400)
MARK = {"path": "o", "ring": "^", "star": "s", "complete": "D"}
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 2.3], hspace=0.45, wspace=0.15)
# top row: the four graphs, drawn with the same five agents
pos = {i: (np.cos(np.pi / 2 + 2 * np.pi * i / N_AGENTS), np.sin(np.pi / 2 + 2 * np.pi * i / N_AGENTS))
       for i in range(N_AGENTS)}
for k, (name, edges) in enumerate(G5.items()):
    g = fig.add_subplot(gs[0, k])
    for i, j in edges:
        g.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="0.35", lw=1.2, zorder=1)
    deg = laplacian(edges, N_AGENTS).diagonal()
    for i in range(N_AGENTS):
        busiest = deg[i] == deg.max()
        g.scatter(*pos[i], s=70, zorder=2, color=DENSE if busiest else "white", edgecolor="k", lw=1)
    l = EIG[name][-1]
    g.set_title(f"{name}  {MARK[name] and ''}\n{len(edges)} links, " + r"$\lambda_{\max}$" + f" = {l:.2f}\n"
                + r"$T_{\max}$" + f" = {tmax_euler(l, l):.3f} s", fontsize=8)
    g.plot(0, -1.55, MARK[name], color="k", ms=6, mfc="none")
    g.set_xlim(-1.35, 1.35); g.set_ylim(-1.75, 1.3); g.set_aspect("equal"); g.axis("off")
# bottom: the bound against the largest eigenvalue, with each graph marked
ax = fig.add_subplot(gs[1, :])
ax.plot(lm, [tmax_euler(l, l) for l in lm], color=DENSE, lw=2.5,
        label="Euler model (each agent computes x + Tv)")
ax.plot(lm, [tmax_exact(l, l) for l in lm], "--", color=SPARSE, lw=1.6,
        label="exact model (a mass driven through a hold)")
ax.axvline(4, color="0.6", lw=0.8, ls=":")
ax.text(4.08, 0.05, r"$\lambda_{\max} = 4$", fontsize=8, color="0.4")
for name, off in [("path", -0.12), ("ring", 0.12), ("star", -0.12), ("complete", 0.12)]:
    l = EIG[name][-1]
    ax.plot(l + off, tmax_euler(l, l), MARK[name], color=DENSE, ms=7, label=None)
    ax.plot(l + off, tmax_exact(l, l), MARK[name], mfc="none", color=SPARSE, ms=7)
ax.annotate("path and ring\n(same busiest agent: 2 links)", (3.62, 1.0), xytext=(0.6, 0.62),
            fontsize=8, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
ax.annotate("star and complete\n(busiest agent: 4 links)", (5.0, 0.553), xytext=(6.0, 0.75),
            fontsize=8, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
ax.set_ylim(0, 1.25)
ax.set_xlabel(r"largest Laplacian eigenvalue $\lambda_{\max}$")
ax.set_ylabel(r"largest stable period $T_{\max}$ (s)")
ax.legend(fontsize=8, loc="upper right")
ax.set_title(r"$L_x = L_v = L$; filled markers: Euler model, open markers: exact model;"
             "\nred agents are the busiest (largest degree); markers at the same point are nudged apart", fontsize=8.5)
save(fig, "ch16_tmax_vs_lmax.png")

# ---------------------------------------------------------------------------------------------
# 3. Simulations: fast and slow agents, sparse and dense graphs
# ---------------------------------------------------------------------------------------------
X0 = np.array([2, -1, 1.5, -2.5, 0.0])
V0 = np.array([0, 0.5, -0.5, 0.3, -0.3])
T_END = 12.0


def simulate(L, T):
    F = network(L, L, T)
    s = np.r_[X0, V0]
    xs = [s[:N_AGENTS].copy()]
    for _ in range(int(T_END / T + 1e-9)):
        s = F @ s
        xs.append(s[:N_AGENTS].copy())
    return np.arange(len(xs)) * T, np.array(xs)


fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
cases = [("ring", 0.25, "ring, T = 0.25 s (fast agents)", 6), ("ring", 1.05, "ring, T = 1.05 s (slow agents)", 6),
         ("path", 0.6, "path (sparse), T = 0.6 s", 8), ("complete", 0.6, "complete (dense), T = 0.6 s", 8)]
for ax, (name, T, title, ylim) in zip(axs.flat, cases):
    t, X = simulate(laplacian(G5[name], N_AGENTS), T)
    for i in range(N_AGENTS):
        ax.plot(t, X[:, i], "-o", ms=2.5, lw=1, color=AGENTS[i])
    ax.set_ylim(-ylim, ylim)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
for ax in axs[1]:
    ax.set_xlabel("t = kT (s)")
for ax in axs[:, 0]:
    ax.set_ylabel("x_i(kT)")
save(fig, "ch16_fast_slow.png")

# ---------------------------------------------------------------------------------------------
# 4. Adding links one at a time, 8 agents
# ---------------------------------------------------------------------------------------------
N8 = 8
edges = [(i, i + 1) for i in range(N8 - 1)]
order = [(0, N8 - 1)] + [tuple(sorted((i, (i + k) % N8))) for k in (2, 3) for i in range(N8)] \
    + [(i, i + 4) for i in range(N8 // 2)]
seq = [list(edges)]
for e in order:
    if e not in edges:
        edges.append(e)
        seq.append(list(edges))
g = lambda l: tmax_euler(l, l)     # L_x = L_v = L
links, tf, tb, lo_b, up_b = [], [], [], [], []
for E in seq:
    L = laplacian(E, N8)
    lmax, d = np.linalg.eigvalsh(L)[-1], L.diagonal().max()
    links.append(len(E))
    tf.append(g(lmax))
    tb.append(largest_stable_T(L, L))
    lo_b.append(g(2 * d))
    up_b.append(g(d + 1))
print(f"\n{N8} agents, links added one at a time: largest stable T (formula, bisection)")
for m_, a_, b_ in zip(links, tf, tb):
    print(f"  {m_:2d} links: {a_:.4f}  {b_:.4f}")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.fill_between(links, lo_b, up_b, step="mid", color=SPARSE, alpha=0.15,
                label="degree band g(2 dmax) .. g(dmax + 1)")
ax.step(links, tf, where="mid", color=DENSE, lw=1.5)
ax.plot(links, tb, "o", color=DENSE, ms=4, label="largest stable T")
ax.set_xlabel("number of links (path = 7, ring = 8, complete = 28)")
ax.set_ylabel("largest stable T")
ax.set_ylim(0, 1.12)
ax.legend(fontsize=8)
save(fig, "ch16_density.png")

# ---------------------------------------------------------------------------------------------
# 5. Noise: steady-state disagreement, ring against complete graph
# ---------------------------------------------------------------------------------------------
def V(name, T, which):
    """Sum over modes 2..n of the per-mode variance (which = 0 positions, 1 velocities)."""
    return sum(mode_variances(l, l, T)[which] for l in EIG[name][1:])


def V_lyapunov(name, T, which):
    """The same number from the discrete Lyapunov equation of the whole disagreement network."""
    L = laplacian(G5[name], N_AGENTS)
    lam, U = np.linalg.eigh(L)
    P2 = np.kron(np.eye(2), U[:, 1:])
    Fd = P2.T @ network(L, L, T) @ P2
    Gd = P2.T @ np.vstack([np.zeros((N_AGENTS, N_AGENTS)), math.sqrt(T) * np.eye(N_AGENTS)])
    P = P2 @ linalg.solve_discrete_lyapunov(Fd, Gd @ Gd.T) @ P2.T
    blk = slice(0, N_AGENTS) if which == 0 else slice(N_AGENTS, 2 * N_AGENTS)
    return np.trace(P[blk, blk])


print("\nSteady-state disagreement (closed form; Lyapunov equation of the network)")
print("   T     ring Vv   complete Vv   ring Vx   complete Vx")
for T in [0.1, 0.2, 0.4, 0.5]:
    print(f"  {T:.1f}  {V('ring', T, 1):8.3f}  {V('complete', T, 1):11.3f}  {V('ring', T, 0):8.3f}"
          f"  {V('complete', T, 0):11.3f}   (check: {V_lyapunov('ring', T, 1):.3f},"
          f" {V_lyapunov('complete', T, 1):.3f})")
cross_v = optimize.brentq(lambda T: V("ring", T, 1) - V("complete", T, 1), 0.2, 0.55)
cross_x = optimize.brentq(lambda T: V("ring", T, 0) - V("complete", T, 0), 0.3, 0.5527)
print(f"  crossing: T = {cross_v:.4f} (velocities), T = {cross_x:.4f} (positions);"
      f" complete graph unstable above {1 - 1 / math.sqrt(5):.4f}")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, which, lab in [(axs[0], 1, "velocity disagreement V_v"), (axs[1], 0, "position disagreement V_x")]:
    Tr = np.linspace(0.001, 0.75, 400)
    Tc = np.linspace(0.001, 0.548, 400)
    ax.semilogy(Tr, [V("ring", T, which) for T in Tr], color=SPARSE, lw=2.2, label="ring (sparse)")
    ax.semilogy(Tc, [V("complete", T, which) for T in Tc], color=DENSE, lw=2.2, label="complete (dense)")
    ax.axvline(1 - 1 / math.sqrt(5), color=DENSE, ls="--", lw=1)
    c = cross_v if which == 1 else cross_x
    ax.plot(c, V("ring", c, which), "ko", ms=5)
    ax.set_xlabel("T (s)")
    ax.set_title(lab, fontsize=10)
    ax.set_ylim(0.05, 60)
    ax.grid(alpha=0.3, which="both")
axs[0].legend(fontsize=8)
save(fig, "ch16_crossing.png")
