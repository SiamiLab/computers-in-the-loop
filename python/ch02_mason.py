"""Chapter 2 companion: block diagrams and Mason's gain formula.

Reproduces:
  * the robot-arm position loop: J_eq, B_eq, K, a, the closed loop
    25/(s^2 + 6s + 25) and the effect of keeping the inductance L = 0.01 H;
  * Mason's gain formula, implemented for any signal flow graph whose branch
    gains are numbers (or functions of s evaluated at one point).  It lists the
    forward paths and loops by depth-first search, builds Delta and the
    cofactors, and is checked against a direct solution of the node equations;
  * Example 2 (two forward paths, two loops that do not touch): 5/8;
  * Example 3 (three interlaced loops) against the block-diagram reduction;
  * the rate-feedback loop: K/(s^2 + 2s + K).

Run:  python3 ch02_mason.py      (figures go to ./figures)
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- robot arm
Jm = Bm = 0.01
Jarm = Barm = 4.0
n, R, Kt, Km = 0.1, 1.0, 0.5, 0.5
J = Jm + n**2 * Jarm
B = Bm + n**2 * Barm
K = n * Kt / (R * J)
a = (R * B + Kt * Km) / (R * J)
print("Robot arm, reflected to the motor shaft")
print(f"  J_eq = {J:.3f}, B_eq = {B:.3f}, K = {K:.3f}, a = {a:.3f}  ->  G(s) = {K:g}/(s(s+{a:g}))")
Gc = 25.0
den = [1, a, K * Gc]
poles = np.roots(den)
wn = np.sqrt(den[2])
print(f"  G_c = 25: closed loop 25/(s^2 + {a:g}s + 25), poles {np.round(poles, 4)},"
      f" wn = {wn:g} rad/s, zeta = {a / (2 * wn):g}")
L = 0.01
motor = np.polyadd(np.polymul([L, R], [J, B]), [Kt * Km])      # (R + Ls)(Js + B) + Kt Km
print(f"  with L = 0.01 H: (R+Ls)(Js+B) + KtKm = {motor[0]:g} (s^2 + {motor[1] / motor[0]:g}s"
      f" + {motor[2] / motor[0]:g}), poles of G: 0, {np.round(np.roots(motor), 2)}")
print(f"  equal real closed-loop poles: G_c = a^2/4 = {a**2 / 4:g}")


# ---------------------------------------------------------------- Mason's formula
def paths_and_loops(branches, src, snk):
    """branches: dict {(from, to): gain}.  Returns forward paths and loops as
    lists of (node list, gain)."""
    out = {}
    for (u, v), g in branches.items():
        out.setdefault(u, []).append((v, g))
    nodes = sorted({u for u, _ in branches} | {v for _, v in branches})
    paths = []

    def dfs(node, visited, gain):
        if node == snk:
            paths.append((visited[:], gain))
            return
        for v, g in out.get(node, []):
            if v not in visited:
                dfs(v, visited + [v], gain * g)
    dfs(src, [src], 1.0)

    loops = []
    order = {v: i for i, v in enumerate(nodes)}

    def dfs_loop(start, node, visited, gain):
        for v, g in out.get(node, []):
            if v == start:
                loops.append((visited[:], gain * g))
            elif v not in visited and order[v] > order[start]:
                dfs_loop(start, v, visited + [v], gain * g)
    for s0 in nodes:                         # each loop is found once, from its lowest node
        dfs_loop(s0, s0, [s0], 1.0)
    return paths, loops


def delta(loops):
    """1 - sum L_i + sum of non-touching pairs - ... ."""
    total = 1.0
    for r in range(1, len(loops) + 1):
        for combo in combinations(loops, r):
            sets = [set(nl) for nl, _ in combo]
            if all(not (sets[i] & sets[j]) for i in range(r) for j in range(i + 1, r)):
                total += (-1) ** r * np.prod([g for _, g in combo])
    return total


def mason(branches, src, snk):
    paths, loops = paths_and_loops(branches, src, snk)
    D = delta(loops)
    num = sum(g * delta([lp for lp in loops if not set(lp[0]) & set(nodes)]) for nodes, g in paths)
    return num / D, paths, loops


def node_equations(branches, src, snk):
    """Solve x = A x + b x_src directly (each node is the sum of what enters it)."""
    nodes = sorted({u for u, _ in branches} | {v for _, v in branches})
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), complex)
    for (u, v), g in branches.items():
        A[idx[v], idx[u]] += g
    M = np.eye(len(nodes)) - A
    M[idx[src], :] = 0
    M[idx[src], idx[src]] = 1
    rhs = np.zeros(len(nodes), complex)
    rhs[idx[src]] = 1
    return np.linalg.solve(M, rhs)[idx[snk]]


# Example 2: two forward paths, two loops that do not touch
def example2(G1, G2, G3, G4, G5, G6, H1, H2):
    return {("R", "x1"): G1, ("x1", "x2"): G2, ("x2", "x1"): -H1, ("x2", "x3"): G3,
            ("R", "x3"): G6, ("x3", "x4"): G4, ("x4", "x3"): -H2, ("x4", "C"): G5}


br = example2(1, 1, 1, 1, 1, 2, 1, 3)
T, paths, loops = mason(br, "R", "C")
print("\nExample 2 with G1..G5 = 1, G6 = 2, H1 = 1, H2 = 3")
print("  forward paths:", [("-".join(p), g) for p, g in paths])
print("  loops:        ", [("-".join(p), g) for p, g in loops])
print(f"  Mason: T_CR = {T:.4f} (5/8 = {5 / 8}),  node equations: {node_equations(br, 'R', 'C').real:.4f}")


# Example 3: three interlaced loops; compare with the reduced formula at s = 0.7 + 1.3j
def blocks(s):
    G1, G2, G3, G4 = 2 / (s + 1), 3 / (s + 2), 1 / s, 4 / (s + 3)
    H1, H2, H3 = 0.5, s / (s + 5), 1.0
    return G1, G2, G3, G4, H1, H2, H3


for s in (0.7 + 1.3j, 2.0 + 0j):
    G1, G2, G3, G4, H1, H2, H3 = blocks(s)
    br3 = {("R", "E1"): 1, ("E1", "E2"): G1, ("E2", "E3"): G2, ("E3", "y"): G3, ("y", "C"): G4,
           ("C", "E3"): H1, ("y", "E2"): -H2, ("C", "E1"): -H3}
    Tm, _, loops3 = mason(br3, "R", "C")
    Tred = G1 * G2 * G3 * G4 / (1 - G3 * G4 * H1 + G2 * G3 * H2 + G1 * G2 * G3 * G4 * H3)
    print(f"\nExample 3 at s = {s}: {len(loops3)} loops; Mason {Tm:.6f}, reduction {Tred:.6f},"
          f" node equations {node_equations(br3, 'R', 'C'):.6f}")

# exit ticket: tachometer (rate) feedback inside a unity position loop, plant 1/(s(s+1))
#   R -> e -> [K] -> x -> [1/(s+1)] -> v (speed) -> [1/s] -> C,  v -> x with -1,  C -> e with -1
Kg = 4.0
for s in (0.5 + 0.5j, 3.0 + 0j):
    brt = {("R", "e"): 1, ("e", "x"): Kg, ("x", "v"): 1 / (s + 1), ("v", "C"): 1 / s,
           ("v", "x"): -1, ("C", "e"): -1}
    Tm, _, _ = mason(brt, "R", "C")
    print(f"Rate feedback, K = {Kg:g}, s = {s}: Mason {Tm:.6f},  K/(s^2+2s+K) = {Kg / (s * s + 2 * s + Kg):.6f}")

# ---------------------------------------------------------------- figure: arm step responses
t = np.linspace(0, 2.5, 1000)
_, y2 = signal.step(([25], den), T=t)
G3 = (np.array([n * Kt]), np.polymul(motor, [1, 0]))
cl3 = (G3[0] * Gc, np.polyadd(G3[1], G3[0] * Gc))
_, y3 = signal.step(cl3, T=t)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(t, y2, label="L neglected: 25/(s^2+6s+25)")
ax.plot(t, y3, "--", label="L = 0.01 H (third order)")
ax.set(xlabel="t (s)", ylabel="arm angle / command", title="Robot-arm position loop, G_c = 25")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "ch02_arm_step.png", dpi=150)
print(f"\nPeak overshoot: {100 * (y2.max() - 1):.1f}% (L neglected), {100 * (y3.max() - 1):.1f}% (L kept)")
print("Saved figures/ch02_arm_step.png")
