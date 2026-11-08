r"""**Hands-on, Chapter 3.** These cells go with the Hands-on exercises at the end of
Chapter 3. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def first_order_step(p, n):
    """x(k+1) = p x(k) + (1 - p) u(k), u = 1, from rest: samples x(0..n-1)."""
    x = np.zeros(n)
    for k in range(n - 1):
        x[k + 1] = p * x[k] + (1 - p)
    return x


def settle_index(x, tol=0.02):
    """first k after which |x - 1| <= tol for good."""
    bad = np.nonzero(np.abs(x - 1) > tol)[0]
    return 0 if len(bad) == 0 else bad[-1] + 1


# ---------------------------------------------------------------- H1
def h1_pole_and_T(T_motor_ms=1.0, T_green_s=60.0):
    """H1: motor (time constant 0.0995 s) and greenhouse (86.6 s); same law x+ = p x + (1-p) u."""
    tau_m, tau_g = -1e-3 / np.log(0.99), -60 / np.log(0.5)
    pm, pg = np.exp(-T_motor_ms * 1e-3 / tau_m), np.exp(-T_green_s / tau_g)
    nm = int(np.ceil(0.8 / (T_motor_ms * 1e-3))) + 1
    ng = int(np.ceil(600 / T_green_s)) + 1
    xm, xg = first_order_step(pm, nm), first_order_step(pg, ng)
    km, kg = settle_index(xm), settle_index(xg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax1.plot(np.arange(nm), xm, ".", color=ORANGE, ms=3, label=f"motor, p = {pm:.5f}")
    ax1.plot(np.arange(ng), xg, "o", color=BLUE, ms=5, label=f"greenhouse, p = {pg:.3f}")
    ax1.set_xscale("symlog", linthresh=10)
    ax1.set_xlabel("sample k")
    ax1.set_title(f"in samples: motor {km}, greenhouse {kg}", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9)
    # log time axis: start at k = 1 (t = 0 cannot be drawn)
    ax2.plot(np.arange(1, nm) * T_motor_ms * 1e-3, xm[1:], color=ORANGE, lw=1.6, label="motor")
    ax2.plot(np.arange(1, ng) * T_green_s, xg[1:], "o-", color=BLUE, ms=4, lw=1,
             label="greenhouse")
    ax2.set_xscale("log")
    ax2.set_xlim(1e-3, 700)
    ax2.set_xlabel("t (s)")
    ax2.set_title(f"in seconds: motor {km * T_motor_ms * 1e-3:.3f} s, "
                  f"greenhouse {kg * T_green_s:.0f} s", fontsize=11)
    ax2.legend(loc="lower right", fontsize=9)
    for ax in (ax1, ax2):
        ax.axhspan(0.98, 1.02, color=TEAL, alpha=0.15)
        ax.set_ylim(0, 1.1)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_insulin(q0=2.0, basal=0.5):
    """H2: q(k+1) = 0.8 q(k) + u(k); the recursion against the flawed formula 2.5 (1 - 0.8^k)."""
    n = 25
    q = np.zeros(n)
    q[0] = q0
    for k in range(n - 1):
        q[k + 1] = 0.8 * q[k] + basal
    k = np.arange(n)
    flawed = basal / 0.2 * (1 - 0.8 ** k)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(k, q, "o-", color=ORANGE, ms=5, lw=1, label="recursion q(k+1) = 0.8 q(k) + u(k)")
    ax.plot(k, flawed, "s--", color=TEAL, ms=4, lw=1, label="classmate's q(k)")
    ax.axhline(basal / 0.2, color=GRAY, lw=0.8)
    ax.set_xlabel("k (periods of 5 min)")
    ax.set_ylabel("insulin on board (U)")
    ax.set_title(f"q(0) = {q0:g} U, basal {basal:g} U: q(1) = {q[1]:.3f}, q(2) = {q[2]:.3f}, "
                 f"final {basal / 0.2:.2f} U", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_hum(N=5):
    """H3: weigh scale at 600 Hz, 1 kg step at k = 0 plus 0.2 kg of 60 Hz hum; average of N samples."""
    T = 1 / 600
    k = np.arange(-20, 80)
    e = np.where(k >= 0, 1.0, 0.0) + 0.2 * np.cos(2 * np.pi * 60 * k * T)
    c = np.array([e[i - N + 1:i + 1].mean() for i in range(N - 1, len(k))])
    kc = k[N - 1:]
    late = c[kc >= max(N - 1, 0) + 20]
    ripple = (late.max() - late.min()) / 2
    zeros = np.exp(2j * np.pi * np.arange(1, N) / N)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(k * T * 1000, e, color=GRAY, lw=1, label="raw reading")
    ax1.plot(kc * T * 1000, c, ".-", color=ORANGE, ms=4, lw=1.2, label=f"average of {N}")
    ax1.axvline((N - 1) * T * 1000, color=BLUE, ls=":", lw=1)
    ax1.set_xlabel("t (ms)")
    ax1.set_ylabel("kg")
    ax1.set_title(f"N = {N}: remaining hum amplitude {ripple:.4f} kg", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9)
    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.plot(zeros.real, zeros.imag, "o", mfc="none", color=BLUE, ms=8, label="zeros of H")
    ax2.plot(0, 0, "x", color=ORANGE, ms=10, mew=2, label=f"{N - 1} poles")
    hum = np.exp(1j * np.pi / 5 * np.array([1, -1]))
    ax2.plot(hum.real, hum.imag, "*", color=TEAL, ms=12, label="hum")
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.4, 1.4)
    ax2.set_ylim(-1.3, 1.3)
    ax2.legend(loc="upper left", fontsize=7)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_gimbal(T=0.25, A=0.032, pattern="A, -A"):
    """H4: J theta'' = tau, torque held over each period; pulse pattern, then zero torque."""
    J = 0.02
    signs = {"A, -A": [1, -1], "A, 0, -A": [1, 0, -1], "A, A, -A, -A": [1, 1, -1, -1]}[pattern]
    taus = A * np.array(signs + [0, 0, 0])
    t_all, th_all = [], []
    th, w = 0.0, 0.0
    for k, tau in enumerate(taus):
        s = np.linspace(0, T, 40)
        t_all.append(k * T + s)
        th_all.append(th + w * s + tau / J * s ** 2 / 2)
        th, w = th + w * T + tau / J * T ** 2 / 2, w + tau / J * T
    t_all, th_all = np.concatenate(t_all), np.concatenate(th_all)
    n_move = len(signs)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t_all, th_all, color=ORANGE, lw=1.6, label="angle between samples")
    ks = np.arange(len(taus) + 1)
    ax.plot(ks * T, np.interp(ks * T, t_all, th_all), "o", color=BLUE, ms=6, label="samples")
    ax.axhline(0.1, color=TEAL, ls="--", lw=1, label="target 0.1 rad")
    ax.axvline(0.5, color=GRAY, ls=":", lw=1, label="t = 0.5 s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("$\\theta$ (rad)")
    ok = lambda c: "pass" if c else "FAIL"
    ax.set_title(f"T = {T:g} s, A = {A:.4f} N m: final {th:.4f} rad, velocity {w:.2e}; "
                 f"torque {ok(abs(A) <= 0.04 + 1e-12)}, time {n_move * T:.3f} s "
                 f"({ok(n_move * T <= 0.5 + 1e-12)})", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
    interact(h1_pole_and_T,
             T_motor_ms=FloatSlider(value=1.0, min=0.1, max=20, step=0.1, description="T motor (ms)",
                                    continuous_update=False),
             T_green_s=FloatSlider(value=60, min=10, max=300, step=10, description="T green (s)",
                                   continuous_update=False))
    interact(h2_insulin,
             q0=FloatSlider(value=2.0, min=0, max=4, step=0.1, description="q(0) (U)",
                            continuous_update=False),
             basal=FloatSlider(value=0.5, min=0, max=1, step=0.05, description="basal (U)",
                               continuous_update=False))
    interact(h3_hum, N=IntSlider(value=5, min=1, max=30, description="N", continuous_update=False))
    interact(h4_gimbal,
             T=FloatSlider(value=0.25, min=0.05, max=0.4, step=0.005, description="T (s)",
                           continuous_update=False),
             A=FloatSlider(value=0.032, min=0.0, max=0.06, step=0.001, description="A (N m)",
                           readout_format=".3f", continuous_update=False),
             pattern=Dropdown(options=["A, -A", "A, 0, -A", "A, A, -A, -A"], value="A, -A",
                              description="pulses"))
except ImportError:
    h1_pole_and_T()
    h2_insulin()
    h3_hum()
    h4_gimbal()
