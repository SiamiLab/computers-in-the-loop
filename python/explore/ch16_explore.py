r"""**How fast must the agents sample?** Each agent $i$ updates its position and velocity every
$T$ seconds, $x_i(k+1) = x_i + T v_i$, $v_i(k+1) = v_i + T m_i$, with $m = -L x - L v$ for the graph
Laplacian $L$. A mode with Laplacian eigenvalue $\lambda$ is stable for $T < 1$ when
$\lambda < 4$, and for $T < 4/(\lambda + \sqrt{\lambda^2 - 4\lambda})$ when $\lambda \ge 4$.
Start with the ring of five agents at $T = 0.25$ s, then push $T$ past 1 s. Switch to the complete
graph: more links, a larger $\lambda_{\max}$, and the limit drops to $0.553$ s. On the right, each
mode's roots $z = 1 + Ts$ walk out from $z = 1$ as $T$ grows.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
AGENT_COLORS = [ORANGE, BLUE, TEAL, GRAY, "#8E44AD", "#C0392B", "#27AE60", "#2C3E50", "#D4AC0D",
                "#7F8C8D"]
X0 = np.array([2.0, -1.0, 1.5, -2.5, 0.0, 1.0, -1.8, 2.2, -0.5, 0.8])
V0 = np.array([0.0, 0.5, -0.5, 0.3, -0.3, 0.2, -0.4, 0.1, 0.4, -0.2])
T_END = 12.0


def _laplacian(graph, n):
    if graph == "path":
        edges = [(i, i + 1) for i in range(n - 1)]
    elif graph == "ring":
        edges = [(i, (i + 1) % n) for i in range(n)]
    elif graph == "star":
        edges = [(0, i) for i in range(1, n)]
    else:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    L = np.zeros((n, n))
    for i, j in edges:
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
    return L, len(edges)


def _t_limit(lam_x, lam_v):
    """Largest stable T of one mode of the Euler model (second-order test)."""
    if lam_v ** 2 < 4 * lam_x:
        return lam_v / lam_x
    return 4 / (lam_v + np.sqrt(lam_v ** 2 - 4 * lam_x))


def explore(graph="ring", agents=5, T=0.25):
    n = int(agents)
    L, links = _laplacian(graph, n)
    lam = np.sort(np.linalg.eigvalsh(L))
    lam[np.abs(lam) < 1e-9] = 0.0
    limits = [_t_limit(l, l) for l in lam[1:]]         # mode 1 (lambda = 0) is the common motion
    t_max = min(limits)

    F = np.block([[np.eye(n), T * np.eye(n)], [-T * L, np.eye(n) - T * L]])
    s = np.r_[X0[:n], V0[:n] - V0[:n].mean()]
    steps = int(np.floor(T_END / T + 1e-9))
    X = [s[:n].copy()]
    for _ in range(steps):
        s = F @ s
        X.append(s[:n].copy())
    X = np.array(X)
    t = T * np.arange(steps + 1)

    roots = [1 + T * np.roots([1.0, l, l]).astype(complex) for l in lam[1:]]
    r_dis = max(np.max(np.abs(r)) for r in roots)
    stable = r_dis < 1 - 1e-12

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.9, 1]})
    for i in range(n):
        ax1.plot(t, X[:, i], "-o", color=AGENT_COLORS[i], lw=1.2, ms=3 if T >= 0.2 else 1.5)
    ax1.set_xlim(0, T_END)
    lim1 = 4.0 if stable else 8.0
    ax1.set_ylim(-lim1, lim1)
    ax1.set_xlabel("t = kT (s)")
    ax1.set_ylabel("position $x_i(kT)$")
    verdict = "consensus" if stable else "unstable"
    ax1.set_title(f"{graph}, {n} agents, T = {T:g} s: {verdict} (limit {t_max:.3f} s)", fontsize=11)

    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color="k", lw=1)
    Tline = np.linspace(0, T, 40)
    for l, r in zip(lam[1:], roots):
        for sp in np.roots([1.0, l, l]).astype(complex):
            z = 1 + Tline * sp
            ax2.plot(z.real, z.imag, color=GRAY, lw=1)
        ax2.plot(r.real, r.imag, "x", color=ORANGE, ms=10, mew=2.2)
    ax2.plot([1], [0], "o", color=BLUE, ms=6)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    lim = max(1.3, 1.1 * r_dis)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Re z")
    ax2.set_ylabel("Im z")
    ax2.set_title("roots of the modes $\\lambda_2, \\dots, \\lambda_n$", fontsize=11)
    fig.tight_layout()

    print(f"{graph}, {n} agents, {links} links:  Laplacian eigenvalues {np.round(lam, 4)}")
    print(f"largest stable T = {t_max:.4f} s (lambda_max = {lam[-1]:.4f})")
    print(f"T = {T:g} s: largest disagreement root |z| = {r_dis:.4f}  ->  "
          + ("stable: the agents reach consensus" if stable else "NOT stable: the agents fly apart"))
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
    interact(explore,
             graph=Dropdown(options=["path", "ring", "star", "complete"], value="ring",
                            description="graph"),
             agents=IntSlider(value=5, min=3, max=10, step=1, description="agents",
                              continuous_update=False),
             T=FloatSlider(value=0.25, min=0.05, max=1.5, step=0.05, description="T (s)",
                           continuous_update=False))
except ImportError:
    explore()
