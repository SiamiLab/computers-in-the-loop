r"""**Hands-on, Chapter 15.** These cells go with the Hands-on exercises at the end of
Chapter 15. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


# ---------------------------------------------------------------- H1
def h1_averaging(N=16, signal="moving"):
    """H1: 12-bit A/D over 0-20 A; average N readings per period T = 1 ms."""
    q, T, periods = 20 / 4096, 1e-3, 400
    Tf = T / N
    t = (np.arange(periods * N) + 1) * Tf
    if signal == "moving":
        x = 10 + 2 * np.sin(2 * np.pi * 50 * t) + 0.7 * np.sin(2 * np.pi * 173 * t)
    else:
        x = np.full_like(t, 7.502)
    read = np.round(x / q) * q
    err_one = read - x
    err_avg = (read.reshape(periods, N).mean(axis=1) - x.reshape(periods, N).mean(axis=1))
    r1, rN = np.sqrt(np.mean(err_one ** 2)), np.sqrt(np.mean(err_avg ** 2))
    delay = (N - 1) / 2 * Tf
    fig, ax = plt.subplots(figsize=(10, 3.6))
    k = np.arange(60)
    ax.plot(k, 1e3 * err_one[(k + 1) * N - 1], "o", color=GRAY, ms=4, label="one reading")
    ax.plot(k, 1e3 * err_avg[k], "s-", color=ORANGE, ms=4, lw=1, label=f"average of {N}")
    ax.axhspan(-1e3 * q / 2, 1e3 * q / 2, color=BLUE, alpha=0.08)
    ax.set_xlabel("k")
    ax.set_ylabel("quantization error (mA)")
    ax.legend(loc="upper right", fontsize=9)
    bits = np.log2(r1 / rN) if rN > 0 else np.inf
    ax.set_title(f"{signal}: rms {1e3 * r1:.3f} -> {1e3 * rN:.3f} mA ({bits:.2f} bits), "
                 f"delay {1e3 * delay:.3f} ms = {np.degrees(200 * delay):.1f} deg at 200 rad/s",
                 fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_roundoff(extra_bits=0, a=0.96):
    """H2: m(k) = Q[a m(k-1) + (1-a) e(k)], A/D 12 bits on 0-10 V, internal step q / 2^extra_bits."""
    q = 10 / 4096
    qi = q / 2 ** extra_bits
    n = 40000
    rng = np.random.default_rng(1)          # load cell with vibration: the signal crosses many levels
    e_true = 5 + 2 * np.sin(2 * np.pi * np.arange(n) / 997.0) + 0.2 * rng.standard_normal(n)
    e_ad = np.round(e_true / q) * q
    m_ideal = np.zeros(n)        # no A/D, no rounding
    m_ad = np.zeros(n)           # A/D only
    m_both = np.zeros(n)         # A/D and internal rounding
    for k in range(1, n):
        m_ideal[k] = a * m_ideal[k - 1] + (1 - a) * e_true[k]
        m_ad[k] = a * m_ad[k - 1] + (1 - a) * e_ad[k]
        m_both[k] = np.round((a * m_both[k - 1] + (1 - a) * e_ad[k]) / qi) * qi
    s = slice(2000, n)
    r_ad = np.sqrt(np.mean((m_ad[s] - m_ideal[s]) ** 2))
    r_ro = np.sqrt(np.mean((m_both[s] - m_ad[s]) ** 2))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    kk = np.arange(3000, 3300)
    ax.plot(kk, 1e3 * (m_ad[kk] - m_ideal[kk]), color=BLUE, lw=1.2, label="A/D noise at m")
    ax.plot(kk, 1e3 * (m_both[kk] - m_ad[kk]), color=ORANGE, lw=1.2, label="round-off noise at m")
    ax.set_xlabel("k")
    ax.set_ylabel("mV")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"a = {a}, {extra_bits} extra bits: rms A/D {1e3 * r_ad:.3f} mV, "
                 f"round-off {1e3 * r_ro:.3f} mV", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_fractional_delay(delta=0.25, c=1.0):
    """H3: theta' = u, u(k) = K (r - theta(kT)) written delta*T after the sample; c = K T."""
    n, r = 40, 1.0
    th = np.zeros(n + 1)
    u_prev = 0.0
    for k in range(n):
        u = c * (r - th[k])                     # u in units of 1/T
        th[k + 1] = th[k] + (1 - delta) * u + delta * u_prev
        u_prev = u
    roots = np.roots([1, c * (1 - delta) - 1, c * delta])
    rho = max(abs(roots))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(np.arange(n + 1), th, "o-", color=ORANGE, ms=4, lw=1)
    ax1.axhline(r, color="k", lw=0.6)
    ax1.set_ylim(-1, 3)
    ax1.set_xlabel("k")
    ax1.set_ylabel("theta(k)")
    ax1.set_title(f"delta = {delta:.2f}, c = {c:.2f}: largest |root| = {rho:.3f} "
                  f"({'decays' if rho < 1 else 'does not decay'})", fontsize=11)
    w = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(w), np.sin(w), color=GRAY, lw=1)
    ax2.plot(roots.real, roots.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.6, 1.6)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_title("roots", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_phase_budget(T_ms=1.2, schedule="(c) compute early"):
    """H4: wc = 60 rad/s, PM 55 deg; hold + Butterworth prefilter (0.1 at wN) + computation delay."""
    wc, pm0, tau_c = 60.0, 55.0, 0.4e-3

    def lags(T, sched):
        wf = (np.pi / T) / 99 ** 0.25
        x = wc / wf
        hold = wc * T / 2
        pre = np.arctan2(np.sqrt(2) * x, 1 - x ** 2)
        dly = wc * (T if sched.startswith("(b)") else 0.05e-3)
        return np.degrees([hold, pre, dly])

    T = T_ms * 1e-3
    h, p, d = lags(T, schedule)
    pm = pm0 - h - p - d
    load = tau_c / T
    Ts = np.linspace(0.3e-3, 2.0e-3, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1, 2]})
    ax1.bar(["hold", "prefilter", "delay"], [h, p, d], color=[BLUE, TEAL, ORANGE])
    ax1.set_ylabel("lag at wc (deg)")
    for sched, col in [("(b) next tick", GRAY), ("(c) compute early", BLUE)]:
        ax2.plot(1e3 * Ts, [pm0 - sum(lags(Tk, sched)) for Tk in Ts], color=col, lw=1.6, label=sched)
    ax2.plot(T_ms, pm, "o", color=ORANGE, ms=9)
    ax2.axhline(45, color="k", ls="--", lw=0.8)
    ax2.axvline(2 * tau_c * 1e3, color="k", ls=":", lw=0.8)
    ax2.set_xlabel("T (ms)")
    ax2.set_ylabel("PM estimate (deg)")
    ax2.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"T = {T_ms:.1f} ms, {schedule}: PM {pm:.1f} deg "
                 f"({'yes' if pm >= 45 else 'no'}), load {100 * load:.0f}% "
                 f"({'yes' if load <= 0.5 + 1e-12 else 'no'})", fontsize=11)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
    interact(h1_averaging, N=IntSlider(value=16, min=1, max=256, description="N",
                                       continuous_update=False),
             signal=Dropdown(options=["moving", "steady"], value="moving", description="signal"))
    interact(h2_roundoff, extra_bits=IntSlider(value=0, min=0, max=8, description="extra bits",
                                               continuous_update=False),
             a=Dropdown(options=[0.8, 0.92, 0.96, 0.99], value=0.96, description="pole a"))
    interact(h3_fractional_delay,
             delta=FloatSlider(value=0.25, min=0.0, max=1.0, step=0.05, description="delta",
                               continuous_update=False),
             c=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.05, description="c = KT",
                           continuous_update=False))
    interact(h4_phase_budget,
             T_ms=FloatSlider(value=1.2, min=0.3, max=2.0, step=0.1, description="T (ms)",
                              continuous_update=False),
             schedule=Dropdown(options=["(b) next tick", "(c) compute early"],
                               value="(c) compute early", description="schedule"))
except ImportError:
    h1_averaging()
    h2_roundoff()
    h3_fractional_delay()
    h4_phase_budget()
