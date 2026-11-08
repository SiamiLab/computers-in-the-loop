r"""**Hands-on, Chapter 13.** These cells go with the Hands-on exercises at the end of
Chapter 13. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


# ---------------------------------------------------------------- H1
def h1_deadbeat_motor(T=0.1):
    """H1: Gp = a/(s+a), a = 10; deadbeat F = 1/z; u(0) = 1/(1-p), then u = 1; c(t) between samples."""
    a = 10.0
    p = np.exp(-a * T)
    n = int(np.ceil(0.5 / T)) + 1
    u = np.r_[1 / (1 - p), np.ones(n - 1)]
    t, c, ck = [], [], [0.0]
    x = 0.0
    for k in range(n):
        tau = np.linspace(0, T, 40)
        t.append(k * T + tau)
        c.append(u[k] + (x - u[k]) * np.exp(-a * tau))   # exact solution with u held
        x = u[k] + (x - u[k]) * p
        ck.append(x)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    ax1.plot(np.concatenate(t), np.concatenate(c), color=BLUE, lw=1.4, label="speed c(t)")
    ax1.plot(T * np.arange(n + 1), ck, "o", color=ORANGE, ms=4, label="samples")
    ax1.axhline(1, color=GRAY, lw=0.6)
    ax1.set_ylim(0, 1.3)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title(f"T = {1000 * T:.1f} ms: u(0) = {u[0]:.2f}, settled at t = T", fontsize=11)
    ax2.step(T * np.arange(n + 1), np.r_[u, u[-1]], where="post", color=ORANGE, lw=1.4)
    ax2.axhline(2, color=TEAL, ls="--", lw=1, label="drive limit |u| = 2")
    ax2.set_ylim(0, max(3, 1.1 * u[0]))
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("u")
    ax2.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_hold_lag(T=0.05):
    """H2: Gc = 566(s+5)/(s+20), Tustin prewarped at 20 rad/s, on 1/(s(s+5)) with a ZOH: exact sampled PM."""
    K, a, wc = 566.0, 5.0, 20.0
    p = np.exp(-a * T)
    b1, b0 = (a * T - 1 + p) / a ** 2, (1 - p - a * T * p) / a ** 2   # ZOH of 1/(s(s+a))
    c1 = wc / np.tan(wc * T / 2)
    w = np.linspace(0.5, np.pi / T * 0.999, 4000)
    z = np.exp(1j * w * T)
    D = K * ((c1 + 5) * z - (c1 - 5)) / ((c1 + 20) * z - (c1 - 20))
    G = (b1 * z + b0) / ((z - 1) * (z - p))
    L = D * G
    La = K / (1j * w * (1j * w + 20))
    i = np.argmin(abs(abs(L) - 1))
    ia = np.argmin(abs(abs(La) - 1))
    pm = 180 + np.degrees(np.angle(L[i]))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.semilogx(w, np.degrees(np.unwrap(np.angle(La))), color=GRAY, lw=1.4, label="analog loop")
    ax.semilogx(w, np.degrees(np.unwrap(np.angle(L))), color=ORANGE, lw=1.4, label="sampled loop")
    ax.axhline(-180, color=GRAY, lw=0.6)
    ax.axvline(w[i], color=ORANGE, ls=":", lw=1)
    ax.set_ylim(-270, -80)
    ax.set_xlabel("frequency (rad/s)")
    ax.set_ylabel("phase (deg)")
    ax.set_title(f"T = {1000 * T:.1f} ms: PM analog {180 + np.degrees(np.angle(La[ia])):.1f}, sampled {pm:.1f} deg, "
                 f"estimate {45 - np.degrees(wc * T / 2):.1f} deg", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_windup(umax=0.25, form="positional"):
    """H3: tank level, Gp = 1/s, T = 0.1 s, PI Kp = 1, Ki = 0.25, |u| <= umax, step 0 -> 1 m."""
    T, Kp, Ki, n = 0.1, 1.0, 0.25, 400
    h = np.zeros(n + 1)
    u = np.zeros(n)
    e = np.zeros(n)
    uI = uprev = eprev = 0.0
    for k in range(n):
        e[k] = 1.0 - h[k]
        if form == "positional":
            uI += Ki * T * e[k]
            u[k] = np.clip(Kp * e[k] + uI, -umax, umax)
        else:
            u[k] = np.clip(uprev + Kp * (e[k] - eprev) + Ki * T * e[k], -umax, umax)
        uprev, eprev = u[k], e[k]
        h[k + 1] = h[k] + T * u[k]
    pos, neg = T * e[e > 0].sum(), T * e[e < 0].sum()
    t = T * np.arange(n + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    ax1.plot(t, h, color=BLUE, lw=1.4)
    ax1.axhline(1, color=GRAY, lw=0.6)
    ax1.fill_between(t, h, 1, where=h < 1, color=ORANGE, alpha=0.15)
    ax1.fill_between(t, h, 1, where=h > 1, color=TEAL, alpha=0.25)
    ax1.set_ylim(0, 1.8)
    ax1.set_ylabel("level (m)")
    ax1.set_title(f"{form}, umax = {umax:.2f} m/s: overshoot {100 * max(h.max() - 1, 0):.1f}%, "
                  f"error areas +{pos:.2f} / {neg:.2f} m s", fontsize=11)
    ax2.step(t[:-1], u, where="post", color=ORANGE, lw=1.4)
    ax2.set_ylabel("u (m/s)")
    ax2.set_xlabel("t (s)")
    plt.show()


# ---------------------------------------------------------------- H4
def h4_derivative_filter(tau_f=0.03):
    """H4: D term Kd/(tau_f+T) (z-1)/(z-alpha), T = 0.01 s, Kd = 0.5; gain up to half the sampling frequency."""
    T, Kd, wc = 0.01, 0.5, 4.0
    alpha = tau_f / (tau_f + T)
    w = np.linspace(0.1, np.pi / T, 3000)
    z = np.exp(1j * w * T)
    DD = Kd / (tau_f + T) * (z - 1) / (z - alpha)
    D0 = Kd / T * (z - 1) / z                         # no filter
    nyq = 2 * Kd / (2 * tau_f + T)
    lag = np.degrees(np.arctan(wc * tau_f))
    ok = nyq <= 20 and lag <= 10
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.semilogx(w, abs(D0), color=GRAY, lw=1.2, label="no filter")
    ax.semilogx(w, abs(DD), color=ORANGE, lw=1.6, label=f"tau_f = {1000 * tau_f:.0f} ms")
    ax.axhline(20, color=TEAL, ls="--", lw=1, label="noise limit 20")
    ax.set_xlabel("frequency (rad/s)")
    ax.set_ylabel("gain of the D term")
    ax.set_title(f"gain at half the sampling frequency {nyq:.1f}, filter lag at 4 rad/s {lag:.1f} deg: "
                 f"{'PASS' if ok else 'FAIL'}", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, Dropdown
    interact(h1_deadbeat_motor, T=FloatSlider(value=0.1, min=0.01, max=0.2, step=0.001, readout_format=".3f",
                                              description="T (s)", continuous_update=False))
    interact(h2_hold_lag, T=FloatSlider(value=0.05, min=0.002, max=0.08, step=0.001, readout_format=".3f",
                                        description="T (s)", continuous_update=False))
    interact(h3_windup, umax=FloatSlider(value=0.25, min=0.1, max=2.0, step=0.01, description="umax (m/s)",
                                         continuous_update=False),
             form=Dropdown(options=["positional", "velocity"], value="positional", description="form"))
    interact(h4_derivative_filter, tau_f=FloatSlider(value=0.03, min=0.0, max=0.08, step=0.001,
                                                     readout_format=".3f", description="tau_f (s)",
                                                     continuous_update=False))
except ImportError:
    h1_deadbeat_motor()
    h2_hold_lag()
    h3_windup()
    h4_derivative_filter()
