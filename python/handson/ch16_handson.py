r"""**Hands-on, Chapter 16.** These cells go with the Hands-on exercises at the end of
Chapter 16. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
AGENT_COLORS = [BLUE, ORANGE, TEAL, GRAY, "#B4500A", "#1F2A33", "#7A9CC6", "#C9A227"]


def laplacian(n, edges):
    L = np.zeros((n, n))
    for i, j in edges:
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
    return L


def graph_edges(name, n):
    if name == "path":
        return [(i, i + 1) for i in range(n - 1)]
    if name == "ring":
        return [(i, (i + 1) % n) for i in range(n)]
    if name == "star":
        return [(0, i) for i in range(1, n)]
    return [(i, j) for i in range(n) for j in range(i + 1, n)]       # complete


def disagreement_radius(Lx, Lv, T):
    """Spectral radius of the Euler network F restricted to the disagreement modes."""
    n = Lx.shape[0]
    P = np.linalg.qr(np.column_stack([np.ones(n), np.eye(n)[:, : n - 1]]))[0][:, 1:]
    I = np.eye(n - 1)
    F = np.block([[I, T * I], [-T * P.T @ Lx @ P, I - T * P.T @ Lv @ P]])
    return max(abs(np.linalg.eigvals(F)))


def largest_T(Lx, Lv, Tcap=3.0):
    """First T at which some disagreement mode reaches the unit circle (scan, then bisect)."""
    Ts = np.linspace(1e-3, Tcap, 1500)
    bad = [T for T in Ts if disagreement_radius(Lx, Lv, T) >= 1 - 1e-12]
    if not bad:
        return np.inf
    hi = bad[0]
    lo = hi - (Ts[1] - Ts[0])
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if disagreement_radius(Lx, Lv, mid) < 1 - 1e-12 else (lo, mid)
    return lo


def simulate(Lx, Lv, T, x0, v0, tend=12.0):
    x, v = np.array(x0, float), np.array(v0, float)
    k = int(round(tend / T))
    X = [x.copy()]
    for _ in range(k):
        x, v = x + T * v, v + T * (-Lx @ x - Lv @ v)
        X.append(x.copy())
    return np.arange(k + 1) * T, np.array(X)


# ---------------------------------------------------------------- H1
def h1_leader(m=6, T=0.3):
    """H1: six drones in a ring (nodes 1..6) and a leader (node 0) linked to drones 1..m; Lx = Lv = L."""
    edges = [(i, i % 6 + 1) for i in range(1, 7)] + [(0, i) for i in range(1, m + 1)]
    L = laplacian(7, edges)
    lam = np.sort(np.linalg.eigvalsh(L))
    Tm = largest_T(L, L)
    x0 = [0.0, 2, -1, 1.5, -2.5, 0.5, -0.5]
    v0 = [0.0, 0.5, -0.5, 0.3, -0.3, 0.2, -0.2]
    t, X = simulate(L, L, T, x0, v0)
    X = X - X.mean(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i in range(7):
        ax.plot(t, X[:, i], "o-", ms=2.5, lw=1, color=AGENT_COLORS[i % 8],
                label="leader" if i == 0 else None)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("t = kT (s)")
    ax.set_ylabel("x_i - average")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"leader linked to {m}: eigenvalues {np.round(lam, 3)}, Tmax = {Tm:.3f} s; "
                 f"T = {T:g} s", fontsize=10)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_two_rows(T=1.0, zeta=0.5):
    """H2: complete bipartite graph, two rows of three robots; Lv = L, Lx = zeta L."""
    L = laplacian(6, [(i, j) for i in range(3) for j in range(3, 6)])
    lam = np.sort(np.linalg.eigvalsh(L))
    Tm = largest_T(zeta * L, L)
    x0 = [1.0, 0.5, 0.8, -1.0, -0.4, -0.9]
    t, X = simulate(zeta * L, L, T, x0, np.zeros(6), tend=8.0)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i in range(6):
        ax.plot(t, X[:, i], "o-", ms=3, lw=1, color=BLUE if i < 3 else ORANGE,
                label=("row 1" if i == 0 else "row 2" if i == 3 else None))
    ax.set_ylim(-5, 5)
    ax.set_xlabel("t = kT (s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"eigenvalues {np.round(lam, 3)}, zeta = {zeta:g}: Tmax = {Tm:.3f} s; "
                 f"T = {T:g} s", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_two_graphs(position_graph="complete", velocity_graph="path", n=4, T=0.1):
    """H3: springs on one graph, dampers on another; Euler network, largest stable T."""
    Lx = laplacian(n, graph_edges(position_graph, n))
    Lv = laplacian(n, graph_edges(velocity_graph, n))
    shared = np.allclose(Lx @ Lv, Lv @ Lx)
    Tm = largest_T(Lx, Lv)
    Ts = np.linspace(0.01, 1.4, 300)
    rad = [disagreement_radius(Lx, Lv, T_) for T_ in Ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax1.plot(Ts, rad, color=BLUE, lw=1.6)
    ax1.axhline(1, color="k", lw=0.8, ls="--")
    ax1.plot(T, disagreement_radius(Lx, Lv, T), "o", color=ORANGE, ms=8)
    ax1.set_ylim(0, 2)
    ax1.set_xlabel("T (s)")
    ax1.set_ylabel("largest |z| of the disagreement")
    rng = np.random.default_rng(3)
    t, X = simulate(Lx, Lv, T, rng.uniform(-2, 2, n), rng.uniform(-0.5, 0.5, n))
    X = X - X.mean(axis=1, keepdims=True)
    for i in range(n):
        ax2.plot(t, X[:, i], lw=1, color=AGENT_COLORS[i % 8])
    ax2.set_ylim(-4, 4)
    ax2.set_xlabel("t = kT (s)")
    ax2.set_ylabel("x_i - average")
    fig.suptitle(f"springs: {position_graph}, dampers: {velocity_graph}, n = {n}: Tmax = {Tm:.3f} s"
                 f" ({'shared' if shared else 'no shared'} eigenvectors)", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_radio_budget(graph="(R) ring", T=0.25):
    """H4: eight robots; (R) ring, (D) ring + diagonals, (S) ring + skip-one links; three tests."""
    n = 8
    edges = [(i, (i + 1) % n) for i in range(n)]
    if graph.startswith("(D)"):
        edges += [(i, i + 4) for i in range(4)]
    if graph.startswith("(S)"):
        edges += [(i, (i + 2) % n) for i in range(n)]
    L = laplacian(n, edges)
    dmax = int(max(np.diag(L)))
    Tm = largest_T(L, L)
    r = disagreement_radius(L, L, T)
    env = r ** (15 / T)
    tests = [T <= 0.8 * Tm + 1e-12, env <= 0.05, dmax / T <= 10 + 1e-12]
    x0 = [2, -1, 1.5, -2.5, 0, 1, -0.5, -0.5]
    v0 = [0, 0.5, -0.5, 0.3, -0.3, 0.2, -0.1, -0.1]
    t, X = simulate(L, L, T, x0, v0, tend=15.0)
    X = X - X.mean(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i in range(n):
        ax.plot(t, X[:, i], lw=1, color=AGENT_COLORS[i])
    ax.set_ylim(-4, 4)
    ax.set_xlabel("t = kT (s)")
    ax.set_ylabel("x_i - average")
    yn = ["yes" if ok else "no" for ok in tests]
    ax.set_title(f"{graph}, T = {T:.2f} s: margin {yn[0]} (Tmax {Tm:.3f}), agreement {yn[1]} "
                 f"(|z|^(15/T) = {env:.3g}), radio {yn[2]} ({dmax / T:.1f}/s)", fontsize=10)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
    interact(h1_leader, m=IntSlider(value=6, min=1, max=6, description="links m",
                                    continuous_update=False),
             T=FloatSlider(value=0.3, min=0.05, max=1.0, step=0.01, description="T (s)",
                           continuous_update=False))
    interact(h2_two_rows, T=FloatSlider(value=1.0, min=0.05, max=1.2, step=0.01, description="T (s)",
                                        continuous_update=False),
             zeta=FloatSlider(value=0.5, min=0.1, max=2.0, step=0.05, description="zeta",
                              continuous_update=False))
    graphs = ["path", "ring", "star", "complete"]
    interact(h3_two_graphs,
             position_graph=Dropdown(options=graphs, value="complete", description="springs"),
             velocity_graph=Dropdown(options=graphs, value="path", description="dampers"),
             n=IntSlider(value=4, min=3, max=8, description="n", continuous_update=False),
             T=FloatSlider(value=0.1, min=0.02, max=1.2, step=0.01, description="T (s)",
                           continuous_update=False))
    interact(h4_radio_budget,
             graph=Dropdown(options=["(R) ring", "(D) ring + diagonals", "(S) ring + skip-one"],
                            value="(R) ring", description="graph"),
             T=FloatSlider(value=0.25, min=0.05, max=0.8, step=0.05, description="T (s)",
                           continuous_update=False))
except ImportError:
    h1_leader()
    h2_two_rows()
    h3_two_graphs()
    h4_radio_budget()
