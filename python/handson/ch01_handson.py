r"""**Hands-on, Chapter 1.** These cells go with the Hands-on exercises at the end of
Chapter 1. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


# ---------------------------------------------------------------- H1
def h1_staircase(T=0.05):
    """H1: a 1 Hz sine through a sample-and-hold; measure the lag of the fundamental."""
    f = 1.0
    t = np.linspace(0, 2, 4000, endpoint=False)
    held = np.sin(2 * np.pi * f * np.floor(t / T) * T)
    # fundamental of the staircase: project on sin and cos over two whole periods
    a = 2 * np.mean(held * np.sin(2 * np.pi * f * t))
    b = 2 * np.mean(held * np.cos(2 * np.pi * f * t))
    amp, phase = np.hypot(a, b), np.degrees(np.arctan2(b, a))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, np.sin(2 * np.pi * f * t), color=GRAY, lw=1.4, label="e(t)")
    ax.step(t, held, where="post", color=ORANGE, lw=1.6, label="sample-and-hold output")
    ax.plot(t, amp * np.sin(2 * np.pi * f * t + np.radians(phase)), "--", color=BLUE, lw=1.4,
            label="its fundamental")
    ax.set_xlabel("t (s)")
    ax.set_title(f"T = {T:g} s: measured lag {-phase:.2f} deg = {-phase / 360 * 1000:.1f} ms, "
                 f"amplitude x {amp:.4f}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_euler_vs_exact(T=0.25):
    """H2: the plant y' = -4y + 4u, unit step held from y0 = 0: Euler recursion vs exact samples."""
    n = int(np.ceil(3 / T)) + 1
    k = np.arange(n)
    p = np.exp(-4 * T)
    y_exact = 1 - p ** k
    y_euler = np.zeros(n)
    for i in range(n - 1):
        y_euler[i + 1] = (1 - 4 * T) * y_euler[i] + 4 * T
    t = np.linspace(0, k[-1] * T, 600)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, 1 - np.exp(-4 * t), color=GRAY, lw=1.4, label="true y(t)")
    ax.plot(k * T, y_exact, "o", color=ORANGE, ms=7, label=f"exact samples, pole {p:.3f}")
    ax.plot(k * T, y_euler, "s--", color=TEAL, ms=5, lw=1,
            label=f"Euler recursion, pole {1 - 4 * T:.3f}")
    ax.set_ylim(-2.5, 3.5)
    ax.set_xlabel("t (s)")
    ax.set_title(f"T = {T:g} s", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_delay(c=0.5, delay="B: reading one period late"):
    """H3: tank level y' = u, held input, c = KT; program A (no delay) or B (one period late)."""
    n, r = 30, 1.0
    y = np.zeros(n + 1)
    for k in range(n):
        if delay.startswith("A"):
            y[k + 1] = y[k] + c * (r - y[k])
        else:
            y[k + 1] = y[k] + c * (r - (y[k - 1] if k >= 1 else 0.0))
    roots = np.roots([1, -(1 - c)]) if delay.startswith("A") else np.roots([1, -1, c])
    rho = max(abs(roots))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(np.arange(n + 1), y, "o-", color=ORANGE, ms=4, lw=1)
    ax1.axhline(r, color="k", lw=0.6)
    ax1.set_ylim(-1, 3)
    ax1.set_xlabel("k")
    ax1.set_ylabel("$y_k$")
    ax1.set_title(f"{delay[:1]}, c = {c:.2f}: largest |root| = {rho:.3f} "
                  f"({'decays' if rho < 1 else 'does not decay'})", fontsize=11)
    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.plot(roots.real, roots.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.6, 1.6)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_title("roots $\\lambda$", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_rounded_integrator(T=0.2, bits=12):
    """H4: u_k = u_{k-1} + K T e_{k-1} stored on a grid q = 8/2^bits; plant 1/(s+2), K = 1, r = 1."""
    a, K, r = 2.0, 1.0, 1.0
    q = 8 / 2 ** bits
    p = np.exp(-a * T)
    b = (1 - p) / a
    n = int(np.ceil(40 / T))
    y = np.zeros(n + 1)
    u = np.zeros(n + 1)
    for k in range(n):
        if k >= 1:                                  # u_k = u_{k-1} + K T e_{k-1}, rounded
            u[k] = np.round((u[k - 1] + K * T * (r - y[k - 1])) / q) * q
        y[k + 1] = p * y[k] + b * u[k]
    band = q / (2 * K * T)
    t = np.arange(n + 1) * T
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, r - y, color=ORANGE, lw=1.4, label="error e = r - y")
    ax.axhspan(-band, band, color=BLUE, alpha=0.15, label=f"stall band |e| < {band:.2e}")
    ax.axhline(0.002, color=TEAL, ls="--", lw=1, label="spec |e| = 0.002")
    ax.axhline(-0.002, color=TEAL, ls="--", lw=1)
    ax.set_ylim(-0.01, 0.03)
    ax.set_xlabel("t (s)")
    ax.set_title(f"T = {T:g} s, {bits} bits: final error {r - y[-1]:.2e}, "
                 f"gain margin 2/T = {2 / T:.1f}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
    interact(h1_staircase, T=FloatSlider(value=0.05, min=0.01, max=0.3, step=0.01,
                                         description="T (s)", continuous_update=False))
    interact(h2_euler_vs_exact, T=FloatSlider(value=0.25, min=0.05, max=0.8, step=0.05,
                                              description="T (s)", continuous_update=False))
    interact(h3_delay, c=FloatSlider(value=0.5, min=0.05, max=2.2, step=0.05, description="c = KT",
                                     continuous_update=False),
             delay=Dropdown(options=["A: no delay", "B: reading one period late"],
                            value="B: reading one period late", description="program"))
    interact(h4_rounded_integrator,
             T=FloatSlider(value=0.2, min=0.05, max=1.0, step=0.01, description="T (s)",
                           continuous_update=False),
             bits=IntSlider(value=12, min=8, max=18, description="bits", continuous_update=False))
except ImportError:
    h1_staircase()
    h2_euler_vs_exact()
    h3_delay()
    h4_rounded_integrator()
