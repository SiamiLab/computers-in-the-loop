r"""**Hands-on, Chapter 2.** These cells go with the Hands-on exercises at the end of
Chapter 2. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


# ---------------------------------------------------------------- H1
def h1_sensor_offset(K=3.8):
    """H1: nozzle heater G = 0.5/(s+0.1), gain K; fan d = -4 W at t = 1 s, offset v = 3 C at t = 4 s."""
    t = np.linspace(0, 8, 4001)
    d = np.where(t >= 1, -4.0, 0.0)
    n = np.where(t >= 4, 3.0, 0.0)
    # y' = -0.1 y + 0.5 (K (r - y - n) + d), with r = 0 (deviation from the setpoint)
    sys = signal.StateSpace([[-0.1 - 0.5 * K]], [[0.5, -0.5 * K]], [[1.0]], [[0.0, 0.0]])
    _, y, _ = signal.lsim(sys, np.column_stack([d, n]), t)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, y, color=ORANGE, lw=1.8, label="true nozzle temperature y")
    ax.plot(t, y + n, "--", color=BLUE, lw=1.4, label="measured y + v")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(1, color=GRAY, lw=0.8, ls=":")
    ax.axvline(4, color=GRAY, lw=0.8, ls=":")
    ax.text(1.05, 2.8, "fan on", color=GRAY, fontsize=9)
    ax.text(4.05, 2.8, "offset appears", color=GRAY, fontsize=9)
    ax.set_ylim(-4.5, 3.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("deviation (C)")
    pole = -(0.1 + 0.5 * K)
    ax.set_title(f"K = {K:g} W/C: pole {pole:.2f}, fan step {y[t < 4][-1]:.3f} C, "
                 f"offset step {y[-1] - y[t < 4][-1]:.3f} C", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_node_check(H1=1.0, F=1.0):
    """H2: the graph with two loops meeting at x2 (G1 = G2 = 1); node equations vs both formulas."""
    G1 = G2 = 1.0
    H2s = np.linspace(0, 4, 41)
    direct = []
    for H2 in H2s:
        # unknowns x1, x2, x3 with R = 1
        A = np.array([[1, H1, 0], [-G1, 1, H2], [0, -G2, 1]], dtype=float)
        x = np.linalg.solve(A, [1, 0, F])
        direct.append(x[2])
    good = (G1 * G2 + F * (1 + G1 * H1)) / (1 + G1 * H1 + G2 * H2s)
    bad = (G1 * G2 + F) / ((1 + G1 * H1) * (1 + G2 * H2s))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(H2s, direct, "o", color=ORANGE, ms=6, label="node equations solved directly")
    ax.plot(H2s, good, color=BLUE, lw=1.6, label="corrected Mason formula")
    ax.plot(H2s, bad, "--", color=TEAL, lw=1.4, label="classmate's formula")
    ax.set_xlabel("$H_2$")
    ax.set_ylabel("$C/R$")
    ax.set_title(f"$G_1 = G_2 = 1$, $H_1$ = {H1:g}, F = {F:g}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_drone(Kp=2.0, Kv=8.0):
    """H3: altitude loop h'' = a + D, v_c = Kp (h_c - h), a = Kv (v_c - h'); 1 m step, then D = -0.6."""
    den = [1, Kv, Kp * Kv]
    t = np.linspace(0, 8, 2001)
    _, h_ref = signal.step(signal.TransferFunction([Kp * Kv], den), T=t)
    t2 = t[t <= 4]
    _, h_d = signal.step(signal.TransferFunction([-0.6], den), T=t2)
    h = h_ref.copy()
    h[t >= 4] += h_d[: np.count_nonzero(t >= 4)]
    poles = np.roots(den)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(t, h, color=ORANGE, lw=1.6)
    ax1.axhline(1, color="k", lw=0.6)
    ax1.axvline(4, color=GRAY, lw=0.8, ls=":")
    ax1.text(4.05, 0.2, "downdraft -0.6 m/s$^2$", color=GRAY, fontsize=9)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("altitude h (m)")
    over = max(0.0, (h_ref.max() - 1) * 100)
    ax1.set_title(f"Kp = {Kp:g}, Kv = {Kv:g}: overshoot {over:.2f}%, "
                  f"offset {100 * (1 - h[-1]):.2f} cm", fontsize=11)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.axvline(0, color=GRAY, lw=0.8)
    ax2.plot(poles.real, poles.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax2.set_xlim(-25, 2)
    ax2.set_ylim(-14, 14)
    kind = "real" if Kv * Kv - 4 * Kp * Kv >= -1e-9 else "complex"   # discriminant
    ax2.set_title(f"poles ({kind})", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_gearbox(ratio=50, k=6.0):
    """H4: pan head through a reducer n = 1/ratio; P control k; checks slew, overshoot, settling."""
    Jm, JL, R, Kt, Km, wmax = 0.001, 0.5, 0.6, 0.06, 0.06, 250.0
    n = 1.0 / ratio
    J = Jm + n ** 2 * JL
    a = Kt * Km / (R * J)
    K = n * Kt / (R * J)
    den = [1, a, k * K]
    poles = np.roots(den)
    wn = np.sqrt(k * K)
    zeta = a / (2 * wn)
    t = np.linspace(0, 6, 3001)
    _, th = signal.step(signal.TransferFunction([k * K], den), T=t)
    outside = np.nonzero(np.abs(th - 1) > 0.02)[0]
    ts = t[outside[-1]] if len(outside) else 0.0
    slew = wmax * n
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, th, color=ORANGE, lw=1.8, label="head angle, 1 rad step")
    ax.axhspan(0.98, 1.02, color=BLUE, alpha=0.15, label="2% band")
    ax.axhline(1.05, color=TEAL, ls="--", lw=1, label="5% overshoot")
    ax.axvline(2, color=GRAY, ls=":", lw=1, label="t = 2 s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("$\\theta_L$ (rad)")
    ok = lambda c: "pass" if c else "FAIL"
    ax.set_title(f"1:{ratio}: a = {a:.2f}, poles {poles[0]:.2f}, {poles[1]:.2f}; "
                 f"slew {slew:.2f} rad/s ({ok(slew >= 4)}), zeta {zeta:.2f} ({ok(zeta >= 0.7)}), "
                 f"slowest pole {max(poles.real):.2f} ({ok(max(poles.real) <= -2 + 1e-9)}), "
                 f"settles in {ts:.2f} s", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, IntSlider
    interact(h1_sensor_offset, K=FloatSlider(value=3.8, min=0.2, max=60, step=0.2,
                                             description="K (W/C)", continuous_update=False))
    interact(h2_node_check,
             H1=FloatSlider(value=1.0, min=0, max=3, step=0.1, description="H1",
                            continuous_update=False),
             F=FloatSlider(value=1.0, min=-2, max=3, step=0.1, description="F",
                           continuous_update=False))
    interact(h3_drone,
             Kp=FloatSlider(value=1.0, min=0.1, max=8, step=0.1, description="Kp (1/s)",
                            continuous_update=False),
             Kv=FloatSlider(value=8.0, min=1, max=24, step=0.5, description="Kv (1/s)",
                            continuous_update=False))
    interact(h4_gearbox,
             ratio=IntSlider(value=20, min=5, max=100, description="N_L/N_m",
                             continuous_update=False),
             k=FloatSlider(value=6.0, min=0.5, max=20, step=0.1, description="k (V/rad)",
                           continuous_update=False))
except ImportError:
    h1_sensor_offset()
    h2_node_check()
    h3_drone()
    h4_gearbox()
