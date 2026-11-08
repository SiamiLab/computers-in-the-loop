r"""**Hands-on, Chapter 14.** These cells go with the Hands-on exercises at the end of
Chapter 14. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


# ---------------------------------------------------------------- H1
def h1_two_zones(b=0.4, start="imbalance [1, -1]"):
    """H1: one heater, two zones x(k+1) = diag(0.5, b) x + [1, 1]' m; reach x(2) = 0."""
    Phi = np.diag([0.5, b])
    Gam = np.array([1.0, 1.0])
    x0 = np.array([1.0, -1.0]) if start.startswith("imbalance") else np.array([1.0, 1.0])
    # x(2) = Phi^2 x0 + Phi Gam m0 + Gam m1 = 0
    M = np.column_stack([Phi @ Gam, Gam])
    if abs(np.linalg.det(M)) < 1e-9:
        m = np.array([np.nan, np.nan])
    else:
        m = np.linalg.solve(M, -Phi @ Phi @ x0)
    n = 8
    x = np.zeros((n + 1, 2))
    x[0] = x0
    u = np.zeros(n + 1)
    u[:2] = np.nan_to_num(m)
    for k in range(n):
        x[k + 1] = Phi @ x[k] + Gam * u[k]
    k = np.arange(n + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6))
    ax1.plot(k, x[:, 0], "o-", color=ORANGE, ms=5, label="zone 1")
    ax1.plot(k, x[:, 1], "s-", color=BLUE, ms=5, label="zone 2")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("k")
    ax1.set_ylabel("deviation (deg C)")
    ax1.legend(loc="lower right", fontsize=9)
    ax2.step(k, u, where="post", color=TEAL, lw=1.8)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("k")
    ax2.set_ylabel("heater m(k) (W)")
    if np.isnan(m[0]):
        fig.suptitle(f"b = {b:.2f}: the two zones cannot be separated (rank 1)", fontsize=11)
    else:
        fig.suptitle(f"b = {b:.2f}, start {x0}: m(0) = {m[0]:.2f} W, m(1) = {m[1]:.2f} W",
                     fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_tank_design(design="classmate"):
    """H2: two tanks, poles 0.2 and 0.4 wanted; step response with the classmate's or the fixed K, N."""
    Phi = np.array([[0.5, 0.0], [0.25, 0.75]])
    Gam = np.array([1.0, 0.0])
    C = np.array([0.0, 1.0])
    K, N = (np.array([0.97, 0.77]), 0.5) if design == "classmate" else (np.array([0.65, 0.77]), 1.92)
    n = 20
    x = np.zeros(2)
    c = np.zeros(n + 1)
    for k in range(n):
        m = -K @ x + N * 1.0
        x = Phi @ x + Gam * m
        c[k + 1] = C @ x
    poles = np.linalg.eigvals(Phi - np.outer(Gam, K))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(np.arange(n + 1), c, "o-", color=ORANGE, ms=5, label="lower level c(k)")
    ax.axhline(1, color=GRAY, ls="--", lw=1, label="r = 1")
    ax.set_ylim(-0.1, 1.3)
    ax.set_xlabel("k")
    ax.set_title(f"{design}: K = {K}, N = {N}, poles {np.round(np.sort(poles.real), 3)}, "
                 f"final c = {c[-1]:.3f}", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_expensive_control(log10_rho=0.0, plant="reactor: x+ = 2x + m"):
    """H3: scalar LQR, J = sum x^2 + rho m^2; closed-loop pole against rho."""
    a = 2.0 if plant.startswith("reactor") else 0.5

    def pole(rho):
        p = solve_discrete_are(np.array([[a]]), np.array([[1.0]]), np.array([[1.0]]),
                               np.array([[rho]]))[0, 0]
        K = a * p / (rho + p)
        return a - K, K, p

    rhos = np.logspace(-2, 4, 200)
    q_curve = [pole(r)[0] for r in rhos]
    rho = 10 ** log10_rho
    q, K, p = pole(rho)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax1.semilogx(rhos, q_curve, color=BLUE, lw=1.6)
    ax1.plot(rho, q, "o", color=ORANGE, ms=9)
    ax1.axhline(1 / a if a > 1 else a, color=GRAY, ls="--", lw=1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("rho")
    ax1.set_ylabel("closed-loop pole")
    k = np.arange(13)
    x = q ** k
    ax2.plot(k, x, "o-", color=ORANGE, ms=5, label="x(k), x(0) = 1")
    ax2.step(k, -K * x, where="post", color=TEAL, lw=1.6, label="m(k)")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("k")
    ax2.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{plant}, rho = {rho:.3g}: p = {p:.4g}, K = {K:.4f}, pole = {q:.4f}",
                 fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_velocity_observer(p1=0.3, p2=0.3):
    """H4: conveyor, Phi = [[1, 0.05], [0, 0.5]], c = x1; prediction observer with poles p1, p2."""
    Phi = np.array([[1.0, 0.05], [0.0, 0.5]])
    C = np.array([[1.0, 0.0]])
    Ob = np.vstack([C, C @ Phi])
    alpha = Phi @ Phi - (p1 + p2) * Phi + p1 * p2 * np.eye(2)
    L = alpha @ np.linalg.solve(Ob, np.array([0.0, 1.0]))
    F = Phi - np.outer(L, C)
    n = 25
    e = np.zeros((n + 1, 2))
    e[0] = [0.0, 20.0]                 # velocity guess off by 20 mm/s
    for k in range(n):
        glitch = 1.0 if k == 10 else 0.0   # one encoder reading off by 1 mm
        e[k + 1] = F @ e[k] - L * glitch
    fast = max(abs(p1), abs(p2)) <= 0.36 + 1e-12
    quiet = abs(L[1]) <= 1.25 + 1e-12
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(np.arange(n + 1), e[:, 1], "o-", color=ORANGE, ms=4, label="velocity error (mm/s)")
    ax.axvline(10, color=GRAY, ls=":", lw=1)
    ax.axhspan(-1.25, 1.25, color=BLUE, alpha=0.12, label="glitch budget 1.25 mm/s")
    ax.set_ylim(-6, 21)
    ax.set_xlabel("k")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"L = [{L[0]:.3f}, {L[1]:.3f}]: fast enough {'yes' if fast else 'no'}, "
                 f"glitch test {'yes' if quiet else 'no'}", fontsize=11)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, Dropdown
    interact(h1_two_zones, b=FloatSlider(value=0.4, min=0.0, max=0.5, step=0.01, description="b",
                                         continuous_update=False),
             start=Dropdown(options=["imbalance [1, -1]", "uniform [1, 1]"],
                            value="imbalance [1, -1]", description="start"))
    interact(h2_tank_design, design=Dropdown(options=["classmate", "corrected"],
                                             value="classmate", description="design"))
    interact(h3_expensive_control,
             log10_rho=FloatSlider(value=0.0, min=-2, max=4, step=0.1, description="log10 rho",
                                   continuous_update=False),
             plant=Dropdown(options=["reactor: x+ = 2x + m", "stable: x+ = 0.5x + m"],
                            value="reactor: x+ = 2x + m", description="plant"))
    interact(h4_velocity_observer,
             p1=FloatSlider(value=0.3, min=-0.5, max=0.6, step=0.01, description="p1",
                            continuous_update=False),
             p2=FloatSlider(value=0.3, min=-0.5, max=0.6, step=0.01, description="p2",
                            continuous_update=False))
except ImportError:
    h1_two_zones()
    h2_tank_design()
    h3_expensive_control()
    h4_velocity_observer()
