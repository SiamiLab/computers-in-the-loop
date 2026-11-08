r"""**Hands-on, Chapter 11.** These cells go with the Hands-on exercises at the end of
Chapter 11. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def unit_circle(ax, r=None):
    c = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(c), np.sin(c), color=GRAY, lw=1)
    if r is not None:
        ax.plot(r * np.cos(c), r * np.sin(c), ":", color=TEAL, lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.25, 1.25)


# ---------------------------------------------------------------- H1
def h1_worst_input(K=3.0):
    """H1: y(k+1) = 0.6 y(k) + 0.4 u(k), u = K (r - y); a step and the worst-case reference."""
    n = 30
    pole = 0.6 - 0.4 * K
    h = np.r_[0.0, 0.4 * K * pole ** np.arange(n - 1)]       # impulse response
    bound = np.sum(np.abs(h)) if abs(pole) < 1 else np.inf

    def run(r):
        y = np.zeros(n + 1)
        for k in range(n):
            y[k + 1] = 0.6 * y[k] + 0.4 * K * (r[k] - y[k])
        return y

    step = np.ones(n)
    alt = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)        # alternating reference
    ys, ya = run(step), run(alt)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(np.arange(n + 1), ys, "o-", color=BLUE, ms=4, lw=1, label=f"step: peak {np.max(np.abs(ys)):.2f} A")
    ax.plot(np.arange(n + 1), ya, "s-", color=ORANGE, ms=4, lw=1,
            label=f"alternating reference: peak {np.max(np.abs(ya)):.2f} A")
    if np.isfinite(bound):
        ax.axhline(bound, color=TEAL, ls="--", lw=1, label=f"sum |h| = {bound:.2f} A")
        ax.axhline(-bound, color=TEAL, ls="--", lw=1)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("k")
    ax.set_ylabel("current y (A)")
    ax.set_title(f"K = {K:.2f}: closed-loop pole {pole:.2f}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_heater(K=0.2):
    """H2: Q(z) = z^3 - 2.4 z^2 + (1.85 + 0.1K) z - 0.45; poles and the step response."""
    Q = [1, -2.4, 1.85 + 0.1 * K, -0.45]
    p = np.roots(Q)
    n = 150
    # closed loop Y/R = 0.1 K z / Q(z): y(k) = 2.4 y(k-1) - (1.85+0.1K) y(k-2) + 0.45 y(k-3) + 0.1 K r(k-2)
    y = np.zeros(n)
    for k in range(2, n):
        y[k] = 2.4 * y[k - 1] - (1.85 + 0.1 * K) * y[k - 2] \
            + 0.45 * (y[k - 3] if k >= 3 else 0.0) + 0.1 * K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(np.arange(n), y, color=ORANGE, lw=1.4)
    ax1.axhline(1, color=GRAY, lw=0.6)
    ax1.set_ylim(-1, 3)
    ax1.set_xlabel("t (s)")
    ax1.set_title(f"K = {K:.3f}: largest |pole| {max(abs(p)):.4f}", fontsize=11)
    unit_circle(ax2)
    ax2.plot(p.real, p.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax2.set_title("closed-loop poles", fontsize=10)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_pi_triangle(Kp=5.0, Ki=100.0):
    """H3: drone altitude, G(z) = T/(z-1), PI D(z) = Kp + Ki T z/(z-1), T = 0.1 s."""
    T = 0.1
    x, y = T * Kp, T ** 2 * Ki
    p = np.roots([1, x + y - 2, 1 - x])
    n = 60
    h = np.zeros(n + 1)
    acc, href = 0.0, 1.0
    for k in range(n):
        e = href - h[k]
        acc += e
        u = Kp * e + Ki * T * acc
        h[k + 1] = h[k] + T * u
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2, 1.2, 1]})
    ax1.plot(T * np.arange(n + 1), h, "o-", color=ORANGE, ms=3, lw=1)
    ax1.axhline(1, color=GRAY, lw=0.6)
    ax1.set_ylim(-1, 3)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("altitude (m)")
    ax1.set_title(f"Kp = {Kp:g}, Ki = {Ki:g}: largest |pole| {max(abs(p)):.3f}", fontsize=11)
    ax2.fill([0, 20, 0], [0, 0, 400], color=BLUE, alpha=0.15)
    ax2.plot([0, 20, 0, 0], [0, 0, 400, 0], color=BLUE, lw=1)
    ax2.plot(Kp, Ki, "o", color=ORANGE, ms=8)
    ax2.set_xlim(-3, 25)
    ax2.set_ylim(-30, 450)
    ax2.set_xlabel("Kp (1/s)")
    ax2.set_ylabel("Ki (1/s$^2$)")
    ax2.set_title("stable region", fontsize=10)
    unit_circle(ax3)
    ax3.plot(p.real, p.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax3.set_title("poles", fontsize=10)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------- H4
def h4_gimbal(K=1.1):
    """H4: G(z) = 0.1(z+1)/((z-1)(z-0.6)), T = 0.05 s, D = K; target pans at 0.5 rad/s."""
    T, rate = 0.05, 0.5
    p = np.roots([1, 0.1 * K - 1.6, 0.6 + 0.1 * K])
    p2 = np.roots([1, 0.2 * K - 1.6, 0.6 + 0.2 * K])          # the loop at 2K
    n = 120
    y = np.zeros(n + 1)
    e = np.zeros(n + 1)
    for k in range(n + 1):
        e[k] = rate * T * k - y[k]
        if k + 1 <= n:
            # y(k+1) = 1.6 y(k) - 0.6 y(k-1) + 0.1 K (e(k) + e(k-1))
            y[k + 1] = 1.6 * y[k] - 0.6 * (y[k - 1] if k >= 1 else 0.0) \
                + 0.1 * K * (e[k] + (e[k - 1] if k >= 1 else 0.0))
    ok = K >= 1 and max(abs(p2)) < 1 and max(abs(p)) <= 0.85 + 1e-9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(T * np.arange(n + 1), e, color=ORANGE, lw=1.4, label="tracking error (rad)")
    ax1.axhline(0.05, color=TEAL, ls="--", lw=1, label="spec 0.05 rad")
    ax1.set_ylim(-0.05, 0.2)
    ax1.set_xlabel("t (s)")
    ax1.set_title(f"K = {K:.3f}: |z| max {max(abs(p)):.3f}, at 2K {max(abs(p2)):.3f}: "
                  f"{'PASS' if ok else 'FAIL'}", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    unit_circle(ax2, r=0.85)
    ax2.plot(p.real, p.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax2.plot(p2.real, p2.imag, "+", color=BLUE, ms=10, mew=1.5)
    ax2.set_title("poles at K (x) and 2K (+)", fontsize=10)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(h1_worst_input, K=FloatSlider(value=3.0, min=0.1, max=3.9, step=0.05, description="K",
                                           continuous_update=False))
    interact(h2_heater, K=FloatSlider(value=0.2, min=0.01, max=2.0, step=0.005, readout_format=".3f",
                                      description="K", continuous_update=False))
    interact(h3_pi_triangle,
             Kp=FloatSlider(value=5.0, min=-2.0, max=24.0, step=0.5, description="Kp (1/s)",
                            continuous_update=False),
             Ki=FloatSlider(value=100.0, min=-20.0, max=440.0, step=5.0, description="Ki (1/s^2)",
                            continuous_update=False))
    interact(h4_gimbal, K=FloatSlider(value=1.1, min=0.05, max=4.2, step=0.01, readout_format=".2f",
                                      description="K", continuous_update=False))
except ImportError:
    h1_worst_input()
    h2_heater()
    h3_pi_triangle()
    h4_gimbal()
