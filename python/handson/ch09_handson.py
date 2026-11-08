r"""**Hands-on, Chapter 9.** These cells go with the Hands-on exercises at the end of
Chapter 9. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def integrator_loop(T, t0, K, n=40, m=200, r=1.0):
    """Conveyor y' = u; u_k = K (r - y(kT)) reaches the hold at kT + t0 (0 <= t0 <= T).
    y(t) is piecewise linear, so it is computed exactly at its corners and interpolated."""
    corners, values = [0.0], [0.0]
    y, u_old = 0.0, 0.0                            # y(kT); command on the hold before kT + t0
    for k in range(n):
        u_new = K * (r - y)
        y_mid = y + u_old * t0                     # at kT + t0, the new command takes over
        y = y_mid + u_new * (T - t0)               # at (k+1)T
        corners += [k * T + t0, (k + 1) * T]
        values += [y_mid, y]
        u_old = u_new
    t = np.linspace(0, n * T, n * m + 1)
    return t, np.interp(t, corners, values)


# ---------------------------------------------------------------- H1
def h1_gateway(h=0.25, T=0.04):
    """H1: the gateway re-reads the held command at (k+h)T and holds it again."""
    n = 8
    t = np.linspace(0, n * T, 2000)
    u_ctrl = np.ones_like(t)                         # unit step command from the controller
    u_gate = np.where(t >= h * T, 1.0, 0.0)          # what the motor receives
    y = np.maximum(t - h * T, 0.0)                   # position, m (u = 1 m/s)
    k = np.arange(1, n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6))
    ax1.step(t, u_ctrl, where="post", color=GRAY, lw=1.4, label="controller D/A")
    ax1.step(t, u_gate, where="post", color=ORANGE, lw=1.4, label="gateway D/A")
    ax1.set_ylim(-0.1, 1.3)
    ax1.set_xlabel("t (s)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title(f"h = {h:.2f}: delay {1000 * h * T:.0f} ms", fontsize=11)
    ax2.plot(t, y, color=BLUE, lw=1.4, label="position y(t)")
    ax2.plot(k * T, np.maximum(k * T - h * T, 0), "o", color=ORANGE, ms=6, label="samples y(kT)")
    ax2.plot(k * T, k * T, "x", color=TEAL, ms=7, label="h = 0")
    ax2.set_xlabel("t (s)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title(f"y(T) = {k[0] * T - h * T:.3f} m", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_which_m(t0_frac=0.25):
    """H2: pump 1/(s+1), T = ln 2, computation delay t0 = t0_frac T; samples of the step response."""
    T = np.log(2)
    t0 = t0_frac * T
    tt = np.linspace(0, 6 * T, 1200)
    c = np.where(tt > t0, 1 - np.exp(-(tt - t0)), 0.0)
    k = np.arange(7)
    ck = np.where(k * T > t0, 1 - np.exp(-(k * T - t0)), 0.0)
    m_right, m_wrong = 1 - t0_frac, t0_frac
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(tt, c, color=GRAY, lw=1.4, label="c(t)")
    ax.plot(k * T, ck, "o", color=ORANGE, ms=7, label=f"samples: model with m = {m_right:.2f}")
    wrong = np.where(k * T > (1 - t0_frac) * T, 1 - np.exp(-(k * T - (1 - t0_frac) * T)), 0.0)
    ax.plot(k * T, wrong, "s--", color=TEAL, ms=5, lw=1, label=f"model with m = {m_wrong:.2f}")
    ax.axvline(t0, color=BLUE, ls=":", lw=1)
    ax.set_xlabel("t (s)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"t0 = {t0_frac:.2f} T: c(1) = {ck[1]:.3f}", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_deadbeat_delay(t0_ms=10.0):
    """H3: deadbeat conveyor loop (K = 1/T, T = 40 ms) with computation time t0."""
    T = 0.04
    t0 = t0_ms / 1000
    t, y = integrator_loop(T, t0, 1 / T, n=15)
    D = t0 / T
    poles = np.roots([1, -D, D])
    k = np.arange(16)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(t, y, color=GRAY, lw=1.4, label="y(t)")
    ax1.plot(k * T, y[k * 200], "o", color=ORANGE, ms=5, label="samples")
    ax1.axhline(1, color="k", lw=0.6)
    ax1.set_ylim(-0.1, 2.0)
    ax1.set_xlabel("t (s)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"t0 = {t0_ms:g} ms: sample peak {y[k * 200].max():.3f}, true peak {y.max():.3f}",
                  fontsize=11)
    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.plot(poles.real, poles.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_title(f"|z| = {max(abs(poles)):.3f}", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_choose_T(T_ms=40.0, t0_ms=5.0):
    """H4: fixed computation time t0, K = 1/T; true peak and the error envelope at 0.2 s."""
    T, t0 = T_ms / 1000, t0_ms / 1000
    n = int(np.ceil(0.4 / T)) + 2
    t, y = integrator_loop(T, t0, 1 / T, n=n)
    D = t0 / T
    rz = max(abs(np.roots([1, -D, D])))
    env = rz ** (0.2 / T)
    ok = (y.max() - 1 <= 0.10 + 1e-9) and env <= 0.02
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, y, color=ORANGE, lw=1.4, label="y(t)")
    ax.axhline(1.1, color=TEAL, ls="--", lw=1, label="10% overshoot")
    ax.axvline(0.2, color=GRAY, ls=":", lw=1)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(-0.1, 1.5)
    ax.set_xlabel("t (s)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"T = {T_ms:g} ms: true overshoot {100 * (y.max() - 1):.1f}%, envelope at 0.2 s "
                 f"{env:.4f}: {'PASS' if ok else 'FAIL'}", fontsize=11)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(h1_gateway, h=FloatSlider(value=0.25, min=0.0, max=0.95, step=0.05, description="h",
                                       continuous_update=False),
             T=FloatSlider(value=0.04, min=0.01, max=0.1, step=0.01, description="T (s)",
                           continuous_update=False))
    interact(h2_which_m, t0_frac=FloatSlider(value=0.25, min=0.0, max=0.95, step=0.05,
                                             description="t0 / T", continuous_update=False))
    interact(h3_deadbeat_delay, t0_ms=FloatSlider(value=10, min=0, max=40, step=1,
                                                  description="t0 (ms)", continuous_update=False))
    interact(h4_choose_T, T_ms=FloatSlider(value=40, min=10, max=100, step=1, description="T (ms)",
                                           continuous_update=False),
             t0_ms=FloatSlider(value=5, min=0, max=10, step=0.5, description="t0 (ms)",
                               continuous_update=False))
except ImportError:
    h1_gateway()
    h2_which_m()
    h3_deadbeat_delay()
    h4_choose_T()
