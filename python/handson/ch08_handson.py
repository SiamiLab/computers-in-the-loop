r"""**Hands-on, Chapter 8.** These cells go with the Hands-on exercises at the end of
Chapter 8. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def held_response(num, den, u, T, m=200):
    """Continuous response of num/den to the held sequence u (period T), m points per period."""
    t = np.arange(len(u) * m) * (T / m)
    _, y, _ = signal.lsim(signal.lti(num, den), np.repeat(u, m), t, interp=False)
    return t, y


# ---------------------------------------------------------------- H1
def h1_double_integrator(T=0.1, command="alternating (-1)^k"):
    """H1: held acceleration command into 1/s^2; altitude y(t) and its samples."""
    n = 20
    k = np.arange(n)
    u = (-1.0) ** k if command.startswith("alt") else np.ones(n)
    t, y = held_response([1.0], [1.0, 0, 0], u, T)
    num, den, _ = signal.cont2discrete(([1.0], [1.0, 0, 0]), T, method="zoh")
    zeros = np.roots(np.trim_zeros(np.ravel(num), "f"))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, y, color=GRAY, lw=1.4, label="y(t)")
    ax.plot(k * T, y[::200], "o", color=ORANGE, ms=6, label="samples y(kT)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"T = {T:g} s, {command}: zero of G(z) at {zeros[0].real:+.3f}", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_hold_or_impulses(T=0.3466):
    """H2: belt 3/(s+2): ZOH equivalent vs the impulse-driven model Z[Gp], unit step."""
    p = np.exp(-2 * T)
    k = np.arange(12)
    right = 1.5 * (1 - p ** k)                      # (z-1)/z Z[3/(s(s+2))]
    wrong = 3 * (1 - p ** (k + 1)) / (1 - p)        # Z[3/(s+2)] times z/(z-1)
    t, y = held_response([3.0], [1.0, 2.0], np.ones(len(k)), T)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, y, color=GRAY, lw=1.4, label="belt speed c(t), held 1 V step")
    ax.plot(k * T, right, "o", color=ORANGE, ms=7, label="(z-1)/z Z[Gp/s]")
    ax.plot(k * T, wrong, "s--", color=TEAL, ms=5, lw=1, label="Gho(z) Gp(z)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m/s")
    ax.legend(loc="center right", fontsize=9)
    ax.set_title(f"T = {T:.3f} s: DC gain 1.5 (right) vs {3 / (1 - p):.2f} (slip)", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_crane(T=1.5):
    """H3: lightly damped crane payload (2 s swing) and the samples of its step response."""
    sig = np.log(2) / 2
    wn2 = sig ** 2 + np.pi ** 2
    tt = np.linspace(0, 12, 3000)
    c = lambda t: 1 - np.exp(-sig * t) * (np.cos(np.pi * t) + sig / np.pi * np.sin(np.pi * t))
    k = np.arange(int(40 / T) + 1)                # 40 s of samples for the test below
    ck = c(k * T)
    mono = bool(np.all(np.diff(ck) >= -1e-12) and np.all(ck <= 1 + 1e-12))
    poles = np.exp(np.array([-sig + 1j * np.pi, -sig - 1j * np.pi]) * T)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(tt, c(tt), color=GRAY, lw=1.2, label="payload c(t)")
    ax1.plot(k * T, ck, "o-", color=ORANGE, ms=6, lw=1, label="samples")
    ax1.axhline(1, color="k", lw=0.6)
    ax1.set_ylim(-0.1, 1.9)
    ax1.set_xlabel("t (s)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(0, 12)
    ax1.set_title(f"T = {T:g} s: largest sample {ck.max():.3f} (true peak {c(1.0):.3f}), "
                  f"monotone: {'yes' if mono else 'no'}", fontsize=10)
    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.plot(poles.real, poles.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_title(f"poles of G(z), wn^2 = {wn2:.2f}", fontsize=10)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_delay_line(n=6):
    """H4: dryer 2 e^{-0.6 s}/(s+1) with T = 0.6/n; step response of the sampled model."""
    T = 0.6 / n
    p = np.exp(-T)
    kmax = int(np.ceil(5 / T))
    k = np.arange(kmax + 1)
    c = np.where(k >= n + 1, 2 * (1 - p ** (k - n)), 0.0)
    tt = np.linspace(0, kmax * T, 2000)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(tt, np.where(tt > 0.6, 2 * (1 - np.exp(-(tt - 0.6))), 0), color=GRAY, lw=1.4,
            label="c(t)")
    ax.plot(k * T, c, "o", color=ORANGE, ms=4, label="samples of the model")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("deg C per W")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"n = {n}, T = {T:.4f} s: G(z) = {2 * (1 - p):.3f} / (z^{n} (z - {p:.3f})), "
                 f"first nonzero c({n + 1}) = {c[n + 1]:.3f}, G(1) = 2", fontsize=10)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
    interact(h1_double_integrator,
             T=FloatSlider(value=0.1, min=0.05, max=0.5, step=0.05, description="T (s)",
                           continuous_update=False),
             command=Dropdown(options=["alternating (-1)^k", "step"], value="alternating (-1)^k",
                              description="command"))
    interact(h2_hold_or_impulses, T=FloatSlider(value=0.3466, min=0.05, max=1.0, step=0.01,
                                                description="T (s)", continuous_update=False))
    interact(h3_crane, T=FloatSlider(value=1.5, min=0.5, max=2.5, step=0.05, description="T (s)",
                                     continuous_update=False))
    interact(h4_delay_line, n=IntSlider(value=6, min=1, max=12, description="n = 0.6/T",
                                        continuous_update=False))
except ImportError:
    h1_double_integrator()
    h2_hold_or_impulses()
    h3_crane()
    h4_delay_line()
